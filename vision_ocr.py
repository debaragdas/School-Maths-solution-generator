"""
vision_ocr.py — fallback structure detection for books whose PDF text
layer is unusable. Some SEBA textbooks embed Assamese glyphs with a
custom/legacy font encoding: the *visual* rendering looks correct, but
fitz.get_text() returns mojibake (wrong Unicode code points), because
the font's internal glyph-index-to-Unicode map is non-standard. No
amount of regex-tuning fixes that — the text layer is simply lying.

When that's detected, this module renders the affected pages as images
and asks Gemini to *look* at them (vision, not text extraction) to find
chapter/exercise heading pages. This keeps the pipeline 100% automatic
— no manual page-number input ever required — at the cost of a small
number of extra Gemini calls, which are cached to disk so a book is
only ever vision-scanned once.
"""
import os
import re
import json
import hashlib
import fitz  # PyMuPDF
from google.genai import types

import config
from solver import get_client, _strip_json_fences
from utils import logger, retry_with_backoff

BENGALI_BLOCK = range(0x0980, 0x0A00)  # covers Assamese (shares the Bengali Unicode block)


def _extract_list(parsed, key: str) -> list:
    """Safely pulls a list of dict entries out of a Gemini JSON response
    for the shape {key: [...]}, without ever raising.

    CONFIRMED BUG THIS FIXES: every call site used to do
    `parsed.get(key, [])` directly. The prompt asks the model to
    "Return ONLY this JSON" as {key: [...]}, but an LLM will sometimes
    simplify and return the bare array `[...]` instead of the wrapper
    object — this is a real, observed class of JSON-mode deviation, not
    a hypothetical. When that happens, `parsed` is a `list`, and
    `list.get` doesn't exist -> `AttributeError: 'list' object has no
    attribute 'get'`. That line sat OUTSIDE the try/except each caller
    used for the API call + json.loads, so instead of degrading to
    "skip this one bad batch/page" (what both docstrings claim happens),
    it aborted the ENTIRE scan for every remaining batch/page, including
    ones from earlier iterations already appended to `results` in the
    same call -- because the exception propagates out of the function
    before `return results` is reached, and the caller's own except
    block discards whatever was collected so far.

    Handles: dict with the key (normal case), bare list (self-heals the
    common LLM deviation above), and anything else (logged, treated as
    no entries this round -- never a crash)."""
    if isinstance(parsed, dict):
        items = parsed.get(key, [])
    elif isinstance(parsed, list):
        items = parsed
    else:
        logger.warning(f"⚠️ Vision response for '{key}' was neither a dict nor a list "
                        f"(got {type(parsed).__name__}) — treating as empty this round.")
        return []
    if not isinstance(items, list):
        logger.warning(f"⚠️ Vision response's '{key}' field was not a list "
                        f"(got {type(items).__name__}) — treating as empty this round.")
        return []
    return [item for item in items if isinstance(item, dict)]


# ------------------------------------------------------------------
# RELIABILITY CHECK — decide, once per book, whether get_text() can be
# trusted at all.
# ------------------------------------------------------------------
def is_text_layer_reliable(pdf_path: str, sample_pages: int = 15, threshold: float = 0.15) -> bool:
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    n = min(sample_pages, total_pages) or 1
    sample_indices = sorted({int(i * total_pages / n) for i in range(n)})

    total_chars, bengali_chars = 0, 0
    for idx in sample_indices:
        text = doc[idx].get_text()
        for ch in text:
            if ch.isspace():
                continue
            total_chars += 1
            if ord(ch) in BENGALI_BLOCK:
                bengali_chars += 1
    doc.close()

    ratio = (bengali_chars / total_chars) if total_chars else 0.0
    reliable = ratio >= threshold
    logger.info(
        f"{'✅' if reliable else '⚠️'} Text-layer Assamese-character ratio: {ratio:.1%} "
        f"({'reliable — using fast text-based detection' if reliable else 'UNRELIABLE — falling back to Gemini Vision'})"
    )
    return reliable


# ------------------------------------------------------------------
# CACHING — a full-book vision scan is the most expensive call in the
# whole pipeline, so it's paid for once per book, ever.
# ------------------------------------------------------------------
def _book_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(pdf_path: str, kind: str, scope: str = "") -> str:
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    suffix = f"_{scope}" if scope else ""
    return os.path.join(config.TEMP_DIR, f"vision_index_{kind}_{_book_hash(pdf_path)}{suffix}.json")


def _load_cache(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# VISION SCAN
# ------------------------------------------------------------------
def _render_page_png(doc, page_index_0based: int, dpi: int = 150) -> bytes:
    pix = doc[page_index_0based].get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def _build_vision_prompt(kind: str, class_name: int, batch_pages: list[int]) -> str:
    page_list = ", ".join(f"image {i} = book page {p}" for i, p in enumerate(batch_pages))

    if kind == "chapter":
        what = (
            "the VISUAL START of a new numbered chapter — a large heading such as "
            "'অধ্যায় ৭' / 'Chapter 7', or an ordinal-word chapter title like "
            "'প্ৰথম অধ্যায়' (= Chapter 1), 'দ্বিতীয় অধ্যায়' (= Chapter 2), "
            "'তৃতীয় অধ্যায়' (= Chapter 3), etc. Such a heading is large, bold, "
            "and sits near the top of the page."
        )
        field_line = (
            '"chapter_number": <integer, the chapter this heading starts>, '
            '"chapter_title": "<the chapter\'s own Assamese title text as printed '
            'right after/below the number, e.g. \\"ত্ৰিভুজ\\" — the subject topic '
            'name, NOT the word \\"chapter\\" or the number itself. null if no '
            'title text is visible on the page.>"'
        )
    else:
        what = (
            "the VISUAL START of a new 'Exercise X.Y' / 'অনুশীলনী X.Y' heading — "
            "bold text near the top of the page, e.g. 'Exercise 7.3' or 'অনুশীলনী: ৭.৩'."
        )
        field_line = '"exercise_label": "<X.Y as a string, e.g. \\"7.3\\">"'

    return f"""
You are looking at {len(batch_pages)} consecutive pages from a Class
{class_name} SEBA/NCERT Mathematics textbook, given to you as images IN
ORDER ({page_list}).

For EACH image, decide whether that page contains {what}

Return ONLY this JSON object, nothing else, no commentary, no code fences:
{{"headings": [{{"page_index_in_batch": <0-based index of the image within
THIS batch>, {field_line}}}]}}

If no page in this batch contains such a heading, return {{"headings": []}}.
Only report a heading on the page where it visually STARTS — never on a
page that merely continues content from an earlier heading.
"""


def _is_retryable_api_error(e: Exception) -> bool:
    msg = str(e)
    return any(marker in msg for marker in (
        "429", "RESOURCE_EXHAUSTED", "rate limit", "Rate limit",
        "quota", "Quota", "UNAVAILABLE", "DEADLINE_EXCEEDED", "503",
    ))


@retry_with_backoff(times=5, base_delay=5.0, max_delay=90.0, retryable_check=_is_retryable_api_error)
def _call_vision_batch(client, parts: list, prompt: str, model: str = None):
    """model routing (see config.MODEL ROUTING): heading/title scans are
    cheap, high-volume visual search -> SCAN tier; figure bounding-box
    detection is accuracy-critical crop sourcing -> VISION tier. Callers
    pass the tier explicitly; None keeps the VISION default so any
    future caller that forgets to choose errs towards accuracy, not
    cheapness."""
    response = client.models.generate_content(
        model=model or config.GEMINI_VISION_MODEL_ID,
        contents=[*parts, prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1000),
            response_mime_type="application/json",
        ),
    )
    return response


def vision_scan_for_headings(pdf_path: str, start_page: int, end_page: int, kind: str,
                              class_name: int, batch_size: int = 6) -> list[dict]:
    """kind: 'chapter' or 'exercise'. Returns [{"page": int, "label": str}, ...]
    in ascending page order, de-duplicated. Never raises on a single bad
    batch — a batch that fails to parse is logged and skipped, so one
    hiccup can't halt full automation; it just risks missing a heading in
    that specific 6-page window, which the completeness checks downstream
    will catch."""
    doc = fitz.open(pdf_path)
    client = get_client()
    results = []

    pages = list(range(start_page, end_page + 1))
    total_batches = (len(pages) + batch_size - 1) // batch_size
    for b, batch_start in enumerate(range(0, len(pages), batch_size), start=1):
        batch_pages = pages[batch_start: batch_start + batch_size]
        logger.info(f"👁️ Vision scan ({kind}) batch {b}/{total_batches}: pages {batch_pages[0]}-{batch_pages[-1]}")

        parts = [
            types.Part.from_bytes(data=_render_page_png(doc, p - 1), mime_type="image/png")
            for p in batch_pages
        ]
        prompt = _build_vision_prompt(kind, class_name, batch_pages)

        try:
            response = _call_vision_batch(client, parts, prompt,
                                          model=config.GEMINI_SCAN_MODEL_ID)
            parsed = json.loads(_strip_json_fences(response.text or "{}"))
            headings = _extract_list(parsed, "headings")
        except Exception as e:
            logger.warning(f"⚠️ Vision batch {b} ({kind}) failed to parse, skipping this window: {e}")
            continue

        for h in headings:
            idx = h.get("page_index_in_batch")
            label = h.get("chapter_number") if kind == "chapter" else h.get("exercise_label")
            if idx is None or label is None or not (0 <= idx < len(batch_pages)):
                continue
            entry = {"page": batch_pages[idx], "label": str(label)}
            if kind == "chapter":
                title = h.get("chapter_title")
                if title and isinstance(title, str) and title.strip():
                    entry["title"] = title.strip()
            results.append(entry)

    doc.close()

    results.sort(key=lambda r: r["page"])
    deduped = []
    for r in results:
        if not deduped or deduped[-1]["label"] != r["label"]:
            deduped.append(r)
    return deduped


def get_chapter_index_cached(pdf_path: str, class_name: int) -> list[dict]:
    """Whole-book chapter heading index, computed via vision once and
    cached forever after (keyed by the PDF's own content hash, so a
    replaced/updated book file automatically invalidates the cache)."""
    cache_file = _cache_path(pdf_path, "chapters")
    cached = _load_cache(cache_file)
    if cached is not None:
        logger.info(f"📦 Using cached vision chapter index ({len(cached)} headings).")
        return cached

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    logger.warning(
        f"🔎 No cached chapter index — running a full-book Gemini Vision scan "
        f"({total_pages} pages, one-time cost, cached at {cache_file})."
    )
    index = vision_scan_for_headings(pdf_path, 1, total_pages, kind="chapter", class_name=class_name)
    _save_cache(cache_file, index)
    return index


def get_exercise_index_for_range(pdf_path: str, start_page: int, end_page: int,
                                  class_name: int, chapter: int) -> list[dict]:
    """Exercise heading index for one chapter's page range. Cached per
    chapter (small, cheap to redo, but caching still saves cost on
    repeated runs of the same chapter)."""
    cache_file = _cache_path(pdf_path, "exercises", scope=f"ch{chapter}")
    cached = _load_cache(cache_file)
    if cached is not None:
        logger.info(f"📦 Using cached vision exercise index for Chapter {chapter} ({len(cached)} headings).")
        return cached

    index = vision_scan_for_headings(pdf_path, start_page, end_page, kind="exercise", class_name=class_name)
    _save_cache(cache_file, index)
    return index


# ------------------------------------------------------------------
# FIGURE LOCATION — vision fallback for book_diagram_extractor.py.
#
# ROOT CAUSE this exists to fix (found during the Exercise 3.2 / 7.3
# audit): book_diagram_extractor's vector-cluster path gates every
# candidate on a printed caption found via page.get_textbox() —
# fitz's TEXT-LAYER lookup. That works fine for books with a normal,
# extractable text layer. But several SEBA books (confirmed on the
# Class 9 book used for Chapter 3) render body text itself as vector
# path objects (outlined glyphs), not real embedded font text — the
# exact same condition is_text_layer_reliable() already detects for
# chapter/exercise-heading purposes (doc.get_text() returns nothing
# usable). On those pages, page.get_textbox() ALSO returns nothing
# for a figure caption band, so _find_caption_figure_ref() always
# fails, every vector cluster gets silently dropped (the caption
# gate, correctly, refuses to guess), and every question referencing
# that figure falls through to a generic AI-redrawn diagram — the
# exact Figure 3.14 B/C/D/E/G/H/L/M mismatch found in the audit.
#
# Fix: when the text layer is unreliable for this book, don't even
# attempt PDF-text caption reads. Render the page as an image and ask
# Gemini Vision to do what a human would do — look at the page and
# report each figure's bounding region (normalized 0-1 coordinates)
# and its printed caption number, if any. The actual pixels used for
# the final crop still come from the deterministic
# page.get_pixmap(clip=rect) path in book_diagram_extractor.py, never
# from anything the model draws or generates — vision only answers
# "where is it and what number is printed under it", the same category
# of task is_text_layer_reliable's sibling functions already do for
# chapter/exercise headings. Cached to disk per exercise PDF (via
# book_figure_index.py), so this is paid once, ever, per file.
# ------------------------------------------------------------------
def locate_figures_via_vision(pdf_path: str, class_name: int = 0) -> list[dict]:
    """Returns [{"page_index": int (0-based), "bbox_norm": [x0,y0,x1,y1]
    in 0..1 page-fraction coords, "figure_ref": str|None}, ...] for every
    distinct printed diagram/figure found by looking at each page image.
    Never raises — a page that fails to parse is just skipped, same
    policy as vision_scan_for_headings, so one bad batch degrades to
    "fewer figures found," never a crashed run."""
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    client = get_client()
    results = []

    for page_index in range(total_pages):
        png = _render_page_png(doc, page_index, dpi=150)
        prompt = f"""
You are looking at one page from a Class {class_name or ''} SEBA/NCERT
Mathematics textbook (image supplied). Find every distinct printed
DIAGRAM or FIGURE on this page — a geometric drawing, graph, coordinate
plane, number line, chart, or construction. Do NOT count blocks of body
text, headings, or decorative rule lines as figures.

For each figure found, report its bounding box as a fraction of the
page's full width/height (0.0 = left/top edge, 1.0 = right/bottom edge),
and the figure number printed as its caption (e.g. "চিত্ৰ 3.14" /
"Figure 3.14"), if one is visible near it.

Return ONLY this JSON, no commentary, no code fences:
{{"figures": [{{"bbox": [x0, y0, x1, y1], "figure_ref": "<e.g. \\"3.14\\", or null if no caption is visible>"}}]}}

If this page has no diagrams at all, return {{"figures": []}}.
"""
        try:
            response = _call_vision_batch(
                client, [types.Part.from_bytes(data=png, mime_type="image/png")], prompt
            )
            parsed = json.loads(_strip_json_fences(response.text or "{}"))
            figures = _extract_list(parsed, "figures")
        except Exception as e:
            logger.warning(f"⚠️ Vision figure-location failed on page {page_index + 1}, skipping: {e}")
            continue

        for fig in figures:
            bbox = fig.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x0, y0, x1, y1 = (max(0.0, min(1.0, float(v))) for v in bbox)
            if x1 <= x0 or y1 <= y0:
                continue
            ref = fig.get("figure_ref")
            if ref is not None:
                ref = re.sub(r"[^\d.]", "", str(ref)) or None
            results.append({"page_index": page_index, "bbox_norm": [x0, y0, x1, y1], "figure_ref": ref})

    doc.close()
    logger.info(f"👁️ Vision figure-location: found {len(results)} figure(s) across {total_pages} page(s) "
                f"(text-layer-unreliable fallback path).")
    return results


def get_chapter_title_via_vision(pdf_path: str, chapter_number: int, start_page: int,
                                  class_name: int) -> str:
    """Targeted, cheap chapter-title lookup: renders just the chapter's
    OWN opening page (plus one lookahead page, in case the title sits
    just below a full-page illustration) and asks Gemini to read the
    title off it directly. This is deliberately separate from the
    whole-book chapter-boundary scan (get_chapter_index_cached) — that
    scan optimizes for finding page RANGES cheaply across an entire
    book, but a book whose text layer is otherwise reliable would never
    trigger it at all, silently leaving the title blank. A 1-2 page
    vision call is cheap enough to always be worth trying regardless of
    whether the whole-book scan ever ran, and is more reliable since
    it's looking at the exact page where the title is printed."""
    cache_file = _cache_path(pdf_path, "chapter_title", scope=f"ch{chapter_number}")
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached.get("title", "") or ""

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    end_page = min(start_page + 1, total_pages)
    parts = [
        types.Part.from_bytes(data=_render_page_png(doc, p - 1), mime_type="image/png")
        for p in range(start_page, end_page + 1)
    ]
    doc.close()

    prompt = f"""
You are looking at the opening page(s) of Chapter {chapter_number} from a Class
{class_name} SEBA/NCERT textbook. Find the chapter's own topic/title text —
the subject name printed after the chapter number (e.g. "ত্ৰিভুজ" for a
Triangles chapter, "ৰেখা আৰু কোণ" for a Lines and Angles chapter). It is
usually large, bold, and near the top of the first page.

Return ONLY this JSON object, no commentary: {{"title": "<the title text
exactly as printed, in its original script>"}}. If genuinely no such
title is visible on these pages, return {{"title": null}}.
"""
    title = ""
    try:
        client = get_client()
        response = _call_vision_batch(client, parts, prompt,
                                      model=config.GEMINI_SCAN_MODEL_ID)
        parsed = json.loads(_strip_json_fences(response.text or "{}"))
        candidate = parsed.get("title")
        if candidate and isinstance(candidate, str):
            title = candidate.strip()
    except Exception as e:
        logger.warning(f"⚠️ Chapter title vision lookup failed for Chapter {chapter_number}: {e}")

    _save_cache(cache_file, {"title": title})
    return title
