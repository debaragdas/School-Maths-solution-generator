"""
chapter_detector.py — find the page range for one chapter. Pure Python,
no AI. Tries the PDF's own bookmarks/TOC first (fast, exact when
present); falls back to scanning page text for chapter headings.

Returns 1-indexed, inclusive (start_page, end_page).
"""
import re
import fitz  # PyMuPDF
from utils import logger

_ASSAMESE_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_ORDINAL_WORDS = {
    "প্ৰথম": 1, "প্রথম": 1, "দ্বিতীয়": 2, "তৃতীয়": 3, "চতুৰ্থ": 4, "চতুর্থ": 4,
    "পঞ্চম": 5, "ষষ্ঠ": 6, "সপ্তম": 7, "অষ্টম": 8, "নৱম": 9, "নবম": 9, "দশম": 10,
    "একাদশ": 11, "দ্বাদশ": 12, "ত্রয়োদশ": 13, "চতুর্দশ": 14, "চতুৰ্দশ": 14, "পঞ্চদশ": 15,
}

_CHAPTER_PATTERNS = [
    re.compile(r"chapter\s*[:\-]?\s*(\d+)", re.IGNORECASE),
    re.compile(r"অধ্যায়\s*[:\-]?\s*([০-৯\d]+)"),
]


def _normalize(text: str) -> str:
    return text.translate(_ASSAMESE_DIGITS)


def _find_chapter_number_in_text(text: str) -> int | None:
    normalized = _normalize(text)
    for pattern in _CHAPTER_PATTERNS:
        m = pattern.search(normalized)
        if m:
            return int(m.group(1))
    for word, num in _ORDINAL_WORDS.items():
        if word in text and ("অধ্যায়" in text or "chapter" in text.lower()):
            return num
    return None


def _headings_from_toc(doc) -> list[tuple[int, int]]:
    """Returns [(page_number, chapter_number), ...] from PDF bookmarks, if usable."""
    toc = doc.get_toc(simple=True)  # [[level, title, page], ...]
    headings = []
    for level, title, page in toc:
        if level != 1:
            continue
        num = _find_chapter_number_in_text(title)
        if num is not None:
            headings.append((page, num))
    return headings


def _headings_from_page_scan(doc) -> list[tuple[int, int]]:
    """Fallback: scan every page's text layer for a chapter heading.
    Only trusts a match found in the first third of the page (headings
    sit at the top; a stray in-body mention of 'Chapter 3' later on the
    page shouldn't be treated as a new heading)."""
    headings = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue
        top_slice = text[: max(200, len(text) // 3)]
        num = _find_chapter_number_in_text(top_slice)
        if num is not None:
            headings.append((i, num))
    return headings


_BENGALI_BLOCK = range(0x0980, 0x0A00)


def _looks_like_real_title(text: str) -> bool:
    """Guards against exactly the 'অধ্যায় ৬P' bug: a vision-OCR misread
    (or a stray bookmark fragment) that's technically non-empty but is
    garbage — e.g. a single stray Latin letter. Requires at least 2
    non-space characters, and that at least half of them are actually
    Assamese/Bengali-script, before a candidate title is trusted enough
    to show in the PDF header."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 2:
        return False
    bengali_count = sum(1 for c in chars if ord(c) in _BENGALI_BLOCK)
    return (bengali_count / len(chars)) >= 0.5


def get_chapter_title(pdf_path: str, chapter_number: int, start_page: int,
                       class_name: int = 0, text_layer_reliable: bool = True) -> str:
    """Best-effort title lookup (e.g. 'ত্রিভুজ' for Chapter 7) — cosmetic
    only, shows in the PDF header, never load-bearing for pipeline logic.

    Order of attempts:
      1. PDF bookmarks (only trusted when the text layer is reliable —
         a garbled-encoding book's bookmarks are just as untrustworthy
         as its body text).
      2. A targeted, cheap Gemini Vision call scoped to JUST the
         chapter's own opening page(s) (see
         vision_ocr.get_chapter_title_via_vision) — this runs
         regardless of whether the whole-book boundary scan ever ran,
         which is what previously left the title blank for any book
         whose text layer was reliable enough to skip vision entirely
         for page-range detection.
      3. Empty string — the template simply omits the ": <name>" suffix
         rather than falling back to an awkward English placeholder
         like "Chapter 7" glued onto an Assamese heading.
    """
    if text_layer_reliable:
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc(simple=True)
            doc.close()
            for level, title, _page in toc:
                if level == 1 and _find_chapter_number_in_text(title) == chapter_number:
                    cleaned = re.sub(r"(chapter|অধ্যায়)\s*[:\-]?\s*[০-৯\d]+\s*[:\-]?\s*", "",
                                      title, flags=re.IGNORECASE).strip()
                    if cleaned and _looks_like_real_title(cleaned):
                        return cleaned
        except Exception:
            pass

    try:
        from vision_ocr import get_chapter_title_via_vision
        title = get_chapter_title_via_vision(pdf_path, chapter_number, start_page, class_name)
        if title and _looks_like_real_title(title):
            return title
    except Exception as e:
        logger.warning(f"⚠️ Could not look up a vision-detected chapter title: {e}")

    return ""


def get_chapter_page_range(pdf_path: str, chapter_number: int, class_name: int = 0,
                            text_layer_reliable: bool = True) -> tuple[int, int]:
    """
    Fully automatic — never prompts for manual page numbers. Tries the
    cheap deterministic path first (bookmarks, then a text-layer page
    scan); if the text layer is known-unreliable (custom/legacy Assamese
    font encodings scramble get_text() output — see vision_ocr.py) or the
    chapter simply isn't found that way, it falls back to a one-time,
    cached Gemini Vision scan of the actual page images instead of
    stopping the pipeline.
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    headings: list[tuple[int, int]] = []
    source = "bookmarks"

    if text_layer_reliable:
        headings = _headings_from_toc(doc)
        if not headings or chapter_number not in [n for _, n in headings]:
            headings = _headings_from_page_scan(doc)
            source = "page-scan"
    doc.close()

    headings = sorted(set(headings))
    matches = [p for p, n in headings if n == chapter_number]

    if not matches:
        if not text_layer_reliable:
            logger.warning(
                f"⚠️ Text layer for this PDF is unreliable — skipping text-based chapter "
                f"detection entirely and using Gemini Vision."
            )
        else:
            logger.warning(
                f"⚠️ Chapter {chapter_number} not found via {source} — falling back to "
                f"Gemini Vision rather than asking for manual page numbers."
            )
        from vision_ocr import get_chapter_index_cached
        vision_index = get_chapter_index_cached(pdf_path, class_name)
        vision_matches = [h["page"] for h in vision_index if h["label"] == str(chapter_number)]
        if not vision_matches:
            raise ValueError(
                f"Chapter {chapter_number} could not be located even via Gemini Vision. "
                f"Headings detected across the whole book: {vision_index}"
            )
        start_page = vision_matches[0]
        later = sorted(h["page"] for h in vision_index if h["page"] > start_page and h["label"] != str(chapter_number))
        end_page = (later[0] - 1) if later else total_pages
        logger.info(f"📖 Chapter {chapter_number} located via Gemini Vision: pages {start_page}-{end_page}")
        return start_page, end_page

    start_page = matches[0]
    later = sorted(p for p, n in headings if p > start_page and n != chapter_number)
    end_page = (later[0] - 1) if later else total_pages

    logger.info(f"📖 Chapter {chapter_number} located via {source}: pages {start_page}-{end_page}")
    return start_page, end_page
