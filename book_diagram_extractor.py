"""
book_diagram_extractor.py — extracts the original printed diagram
images straight from the source exercise PDF, so a question with a
genuine textbook figure can show THAT instead of an AI-approximated
SVG. This is what the Strict Hierarchy Rule in the render step depends
on: original book image > AI diagram_spec > nothing.

IMPORTANT: this does NOT use doc.extract_image(xref) to pull the raw
embedded image bytes. PyMuPDF's extract_image() returns the image data
exactly as embedded in the PDF's object stream — it does NOT apply the
page's transformation matrix (rotation/mirroring/skew) that the PDF's
content stream applies when actually PLACING that image on the page.
A source image stored mirrored-then-flipped-back-via-CTM at display
time extracts as mirrored raw bytes — visually backwards, even though
it displays correctly in any normal PDF viewer.

The fix: for each detected image's bounding box on the page, render
that exact region of the fully-composited page via
`page.get_pixmap(clip=rect)` — this produces pixels exactly as a human
viewing the PDF would see them, transformation matrix and all, because
it goes through the same rendering path as viewing the whole page.

TWO SOURCES OF CANDIDATE DIAGRAMS, because textbook figures are NOT
always embedded raster images:

  1. RASTER — a figure embedded as a literal image XObject
     (`page.get_image_info`). Handled by `_raster_candidates`.
  2. VECTOR — a figure drawn directly with PDF path operators (lines,
     rects, bezier curves) — axes, plotted points, dashed guide lines,
     triangle constructions, angle arcs, tick/equal marks, etc. This is
     how most coordinate-geometry and geometry-construction figures in
     this book are actually produced (confirmed by inspecting the
     source PDFs: pages containing Figure 3.13/3.14/7.39/7.40 etc. have
     ZERO or only incidental raster images, but 30-650 individual
     vector path objects). The old raster-only extractor silently
     found nothing on these pages, so every question referencing one
     of these figures fell through to the AI `diagram_spec` fallback —
     producing a generic, lower-fidelity, sometimes mislabeled
     redrawing instead of the real textbook figure. This is the exact
     root cause of the diagram mismatches found in the Exercise 3.2
     and Exercise 7.3 audit (Figure 3.14's B/C/D/E/G/H/L/M point
     layout, and Figures 7.39/7.40's triangle constructions).
     Handled by `_vector_candidates`.

Vector candidates are gated behind a stricter rule than raster ones:
a vector cluster is only ever treated as a real figure if a numbered
caption ("চিত্ৰ 3.14" / "Figure 7.39") is found immediately next to
it. Raw vector geometry is common on a textbook page for reasons that
have nothing to do with diagrams (rule lines under headings, boxed
"মন্তব্য"/"কাৰ্য" callouts, table borders), and none of those carry a
figure caption — so requiring the caption is what keeps this
deterministic and false-positive-free, in line with "never guess, never
approximate". A vector region without a readable caption is simply not
emitted; that question keeps falling back to the AI diagram_spec exactly
as before, which is a safe default (Case B in the architecture), not
a wrong figure (Case A done incorrectly).

MATCHING: each candidate image also gets its printed caption checked
(just below, then just above, its own bounding box) for a numbered
figure reference such as "চিত্ৰ 3.14" / "Figure 3.14"
(`_find_caption_figure_ref`, via `utils.extract_figure_reference` — a
plain regex, not AI). solver.py's `_attach_book_diagrams` uses that
number to match a question naming the same figure to this EXACT
image, wherever it sits on the page. Only when neither the image nor
the question carries a recognizable figure number does matching fall
back to the page-order heuristic (assuming the book's diagrams appear
in the same top-to-bottom order as their questions, and that Gemini
correctly flagged the question via "has_book_diagram"). That fallback
is a solid heuristic for standard textbook layouts, but is no longer
the only mechanism — books that print explicit figure numbers (the
common case for SEBA/NCERT) get an exact, verifiable match instead.
"""
import re
import fitz
from utils import logger, extract_figure_reference

try:
    import pytesseract
    from PIL import Image
    import io as _io
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

# ------------------------------------------------------------------
# STAGE 2 VALIDATION — deterministic OCR cross-check for figure_ref.
#
# A figure_ref can arrive from two very different kinds of evidence:
#   - vector path: read straight off the PDF's own text objects via
#     page.get_textbox() — exact, but only works when the PDF has a
#     real extractable text layer.
#   - vision path: an image model's best read of the caption in a
#     rendered page image — useful exactly when the text layer isn't
#     usable, but a single model call is still one source of evidence,
#     not proof.
# Per the "multi-stage validation / OCR verification" requirement,
# every figure_ref — regardless of which path produced it — gets a
# second, independent, fully local/deterministic opinion from
# Tesseract OCR run directly on a tight crop of the printed caption
# band beneath the figure. Tesseract is not an LLM: it cannot be
# swayed by context, only by pixels. If it confidently reads a
# DIFFERENT figure number than what's being claimed, that is treated
# as a real disagreement and the ref is dropped (fail-safe — the
# question then either finds no matching image, or falls back to
# page-order pooling only for raster/vector — never for vision — per
# solver.py's matching policy). If Tesseract can't read anything
# (common — this project's book fonts aren't guaranteed to be in
# Tesseract's default trained data, and Assamese-specific traineddata
# frequently isn't installed on a given machine), that is treated as
# "inconclusive", not "wrong" — the original ref is kept rather than
# discarded on the mere absence of a second opinion, since requiring
# a positive OCR match unconditionally would silently break every
# figure lookup on any machine without the right language pack
# installed, which is a worse outcome than the rare mismatch this
# check exists to catch.
# ------------------------------------------------------------------
_CAPTION_BAND_HEIGHT_PT = 30
# Separator between the two number groups (e.g. "7" and "18" in "7.18")
# is matched loosely — same OCR-robustness fix as utils.py's own
# _FIGURE_REF_PATTERN: a scanned caption band is exactly where a '.'
# is most likely to be misread as '-'/en dash, or to pick up stray
# whitespace, by Tesseract. This only ever INCREASES how often an
# independent OCR reading can positively CONFIRM the claimed ref (see
# _cross_verify_figure_ref below); a strict pattern here doesn't cause
# wrong drops, it just silently under-uses the cross-check by leaving
# more genuinely-matching cases as merely "inconclusive".
_FIGURE_NUM_PATTERN = re.compile(r"(\d{1,2})\s*[.\-–—]\s*(\d{1,2})")


def _ocr_caption_band(page, bbox) -> str | None:
    """Returns a figure number string (e.g. "3.14") read independently
    by Tesseract from the small band directly beneath `bbox`, or None
    if OCR isn't available / found nothing readable there. Never
    raises — every failure mode here degrades to 'inconclusive'."""
    if not _HAS_TESSERACT:
        return None
    try:
        band = fitz.Rect(bbox[0] - 15, bbox[3] - 2, bbox[2] + 15, bbox[3] + _CAPTION_BAND_HEIGHT_PT)
        band = band & page.rect
        if band.is_empty or band.width < 5 or band.height < 5:
            return None
        pix = page.get_pixmap(clip=band, dpi=300)
        img = Image.open(_io.BytesIO(pix.tobytes("png")))
        text = ""
        for lang in ("asm+eng", "ben+eng", "eng"):
            try:
                text = pytesseract.image_to_string(img, lang=lang, config="--psm 7")
                if text.strip():
                    break
            except Exception:
                continue
        if not text.strip():
            return None
        m = _FIGURE_NUM_PATTERN.search(text)
        return f"{m.group(1)}.{m.group(2)}" if m else None
    except Exception as e:
        logger.debug(f"OCR caption cross-check skipped ({e}).")
        return None


def _cross_verify_figure_ref(page, bbox, claimed_ref: str | None, source: str) -> str | None:
    """Applies the policy described above. `source` is only used for a
    more specific log message — the check itself is identical for
    every candidate origin."""
    if not claimed_ref:
        return claimed_ref
    ocr_ref = _ocr_caption_band(page, bbox)
    if ocr_ref is None:
        return claimed_ref  # inconclusive — keep the original claim
    if ocr_ref == claimed_ref:
        return claimed_ref  # independently confirmed
    logger.warning(f"⚠️ Figure-caption disagreement on page {page.number + 1}: {source} path claimed "
                    f"figure_ref='{claimed_ref}' but independent OCR of the caption band reads "
                    f"'{ocr_ref}' instead — dropping the ref rather than trusting either blindly "
                    f"(this candidate falls back to safer matching).")
    return None

# --- Vector-cluster specific tuning -----------------------------------
# Two vector path bboxes are considered part of the same figure if they
# are within this many PDF points of each other (touching or a small
# gap — e.g. an axis line and a point label sitting just past its tip).
VECTOR_MERGE_GAP_PT = 12
# A single path this thin (a horizontal/vertical rule line, e.g. the
# blue divider under a section heading) is never itself a diagram
# component and is dropped before clustering — but it can still end up
# inside a cluster's bbox if it happens to overlap real figure content
# (e.g. an axis line, which is exactly as thin — that's fine, axis
# lines are wanted). This threshold only filters ISOLATED thin rules
# that run most of the page width, which no diagram axis does.
_RULE_MIN_WIDTH_FRACTION = 0.6
_RULE_MAX_HEIGHT_PT = 1.5


# Below this fraction of sampled non-whitespace characters actually
# falling in the Assamese/Bengali Unicode block, we no longer trust
# fitz.get_textbox()/get_text() on this PDF AT ALL — not just for
# chapter/exercise headings (vision_ocr.is_text_layer_reliable already
# guards that), but for figure CAPTIONS too. Several SEBA books render
# body text as vector-outlined glyphs rather than real embedded font
# text, which means page.get_textbox() returns nothing for a caption
# band even though the caption is clearly visible on the rendered page.
# Root-caused during the Exercise 3.2 audit: Figure 3.14's caption
# search always came back empty on that book, so every vector cluster
# for that figure was (correctly, per the caption gate) dropped, and
# the question fell through to a generic AI-redrawn diagram instead of
# the real book figure. See vision_ocr.locate_figures_via_vision for
# the fallback this triggers.
_TEXT_LAYER_SAMPLE_PAGES = 8
_TEXT_LAYER_RATIO_THRESHOLD = 0.15
_BENGALI_BLOCK = range(0x0980, 0x0A00)

MIN_DIAGRAM_WIDTH_PT = 40
MIN_DIAGRAM_HEIGHT_PT = 40
MAX_AREA_FRACTION = 0.85  # skip near-full-page images — almost certainly a
                          # scanned background/watermark, not a figure
RENDER_DPI = 200          # resolution for the clipped-region render — print quality

# How far above/below a figure's own bbox to look for its printed caption
# (e.g. "চিত্ৰ 3.14" / "Figure 3.14"), in PDF points. Captions are almost
# always immediately below a figure, occasionally above — both are checked.
CAPTION_SEARCH_MARGIN_PT = 28




def _text_layer_reliable(doc) -> bool:
    """Cheap, local echo of vision_ocr.is_text_layer_reliable (duplicated
    rather than imported to avoid a hard dependency from this module on
    solver.py's Gemini client at import time — this check itself needs
    no network call). If most sampled non-whitespace characters aren't
    in the Assamese/Bengali Unicode block, get_text()/get_textbox() is
    not returning usable text on this PDF, so caption search must not
    be attempted at all — see the module-level comment above."""
    total_pages = doc.page_count
    n = min(_TEXT_LAYER_SAMPLE_PAGES, total_pages) or 1
    sample_indices = sorted({int(i * total_pages / n) for i in range(n)})
    total_chars, bengali_chars = 0, 0
    for idx in sample_indices:
        for ch in doc[idx].get_text():
            if ch.isspace():
                continue
            total_chars += 1
            if ord(ch) in _BENGALI_BLOCK:
                bengali_chars += 1
    ratio = (bengali_chars / total_chars) if total_chars else 0.0
    return ratio >= _TEXT_LAYER_RATIO_THRESHOLD


def _find_caption_figure_ref(page, bbox):
    """Looks for a printed figure caption (e.g. 'চিত্ৰ 3.14') in a band of
    text just below — then just above — the figure's own bounding box, and
    returns its normalized figure number via extract_figure_reference().
    This is what lets a question that says 'চিত্ৰ 3.14 চোৱা' be matched to
    the actual figure captioned '3.14' in the book, instead of relying only
    on the assumption that figures and questions appear in the same order.
    """
    x0, y0, x1, y1 = bbox
    page_rect = page.rect

    below = fitz.Rect(x0 - 10, y1, x1 + 10, min(y1 + CAPTION_SEARCH_MARGIN_PT, page_rect.y1))
    above = fitz.Rect(x0 - 10, max(y0 - CAPTION_SEARCH_MARGIN_PT, page_rect.y0), x1 + 10, y0)

    for band in (below, above):
        try:
            text = page.get_textbox(band)
        except Exception:
            text = ""
        ref = extract_figure_reference(text)
        if ref:
            return ref
    return None


def _raster_candidates(doc) -> list[tuple]:
    """Returns [(page_index, bbox, kind), ...] for embedded image XObjects,
    de-duplicated per page and with running headers/logos (an xref that
    repeats across most of the document) dropped. kind is always 'raster'.
    """
    total_pages = doc.page_count
    raw = []                 # [(page_index, xref, bbox), ...]
    xref_page_count = {}

    for page_index in range(total_pages):
        page = doc[page_index]
        page_area = page.rect.width * page.rect.height
        seen_on_this_page = set()
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref", 0)
            bbox = info.get("bbox")
            if not xref or not bbox or xref in seen_on_this_page:
                continue
            seen_on_this_page.add(xref)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w < MIN_DIAGRAM_WIDTH_PT or h < MIN_DIAGRAM_HEIGHT_PT:
                continue
            if page_area and (w * h) / page_area > MAX_AREA_FRACTION:
                continue
            xref_page_count[xref] = xref_page_count.get(xref, 0) + 1
            raw.append((page_index, xref, bbox))

    repeat_threshold = max(2, total_pages // 2)
    return [(pi, bbox, "raster") for pi, xref, bbox in raw
            if xref_page_count[xref] <= repeat_threshold]


def _is_isolated_rule_line(rect, page_rect) -> bool:
    """True for a thin horizontal/vertical rule spanning most of the page —
    e.g. the divider line under a section heading. These are page furniture,
    never diagram content, and must not seed or pollute a vector cluster."""
    w, h = rect.width, rect.height
    if h <= _RULE_MAX_HEIGHT_PT and w >= _RULE_MIN_WIDTH_FRACTION * page_rect.width:
        return True
    if w <= _RULE_MAX_HEIGHT_PT and h >= _RULE_MIN_WIDTH_FRACTION * page_rect.height:
        return True
    return False


def _cluster_rects(rects: list, gap: float) -> list:
    """Greedily merges a list of fitz.Rect into connected-component groups,
    where two rects belong to the same group if their gap-padded boxes
    overlap. Returns one merged bounding Rect per group. O(n^2) passes
    until stable — fine for the tens/low-hundreds of path objects on a
    single textbook page."""
    groups = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        merged = []
        used = [False] * len(groups)
        for i in range(len(groups)):
            if used[i]:
                continue
            current = fitz.Rect(groups[i])
            padded = fitz.Rect(current.x0 - gap, current.y0 - gap,
                                current.x1 + gap, current.y1 + gap)
            for j in range(i + 1, len(groups)):
                if used[j]:
                    continue
                if padded.intersects(groups[j]):
                    current |= groups[j]
                    padded = fitz.Rect(current.x0 - gap, current.y0 - gap,
                                        current.x1 + gap, current.y1 + gap)
                    used[j] = True
                    changed = True
            used[i] = True
            merged.append(current)
        groups = merged
    return groups


def _page_vector_clusters(page) -> list:
    """Returns the raw, merged vector-path bounding rects for one page —
    BEFORE any min-size/max-area filtering or caption gating. This is the
    shared clustering step used both by the normal single-page vector
    path (_vector_candidates) and by the cross-page figure-continuation
    detector (_stitch_cross_page_figures) below, so page.get_drawings()
    and _cluster_rects are only ever run once per page regardless of
    which caller(s) need the result."""
    page_rect = page.rect
    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if not r or r.is_empty:
            continue
        if _is_isolated_rule_line(r, page_rect):
            continue
        rects.append(r)
    if not rects:
        return []
    return _cluster_rects(rects, VECTOR_MERGE_GAP_PT)


def _cluster_key(page_index, rect) -> tuple:
    """A hashable, float-rounding-stable identity for one page's cluster
    rect, used to mark a cluster as 'already consumed by a cross-page
    stitch' so it isn't ALSO emitted as an incomplete single-page figure
    (see the `exclude` parameter of _vector_candidates)."""
    return (page_index, (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1)))


def _vector_candidates(doc, clusters_by_page: dict | None = None, exclude: set | None = None) -> list[tuple]:
    """Returns [(page_index, bbox, kind, ref), ...] for figures drawn with
    PDF vector path operators rather than embedded as an image (axes,
    plotted points, triangle/geometry constructions, etc — see module
    docstring). Only clusters that sit next to a numbered figure caption
    are returned, which is what keeps this precise instead of guessing at
    arbitrary page decoration. kind is always 'vector'.

    `clusters_by_page` lets a caller (extract_diagram_images) reuse
    clustering already computed once for the cross-page continuation
    check instead of recomputing page.get_drawings() a second time; if
    omitted, clusters are computed fresh (keeps this function usable
    standalone, e.g. from a REPL or a future caller).

    `exclude`: a set of _cluster_key(...) values for clusters that were
    already consumed by _stitch_cross_page_figures as one half of a
    two-page figure — these must be skipped here, or a stitched figure
    would ALSO ship a second, truncated, single-page duplicate of its
    own bottom half (the exact bug this feature fixes — see that
    function's docstring).
    """
    exclude = exclude or set()
    out = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        clusters = (clusters_by_page.get(page_index, []) if clusters_by_page is not None
                    else _page_vector_clusters(page))

        for cluster in clusters:
            if _cluster_key(page_index, cluster) in exclude:
                continue
            w, h = cluster.width, cluster.height
            if w < MIN_DIAGRAM_WIDTH_PT or h < MIN_DIAGRAM_HEIGHT_PT:
                continue
            if page_area and (w * h) / page_area > MAX_AREA_FRACTION:
                continue
            # Gate: require a numbered caption right next to this cluster.
            # Without one, this is left alone rather than risking a false
            # positive (e.g. a "কাৰ্য"/"মন্তব্য" callout box) — that question
            # simply keeps using the AI diagram_spec fallback, same as today.
            ref = _find_caption_figure_ref(page, (cluster.x0, cluster.y0, cluster.x1, cluster.y1))
            if not ref:
                continue
            out.append((page_index, (cluster.x0, cluster.y0, cluster.x1, cluster.y1), "vector", ref))
    return out


# ------------------------------------------------------------------
# MULTI-PAGE FIGURE HANDLING — a figure occasionally straddles a page
# break in the source PDF (a tall geometric construction or a wide
# statistics chart the book's own layout continues onto the next
# printed page). Before this fix, every page was scanned in complete
# isolation: the TOP half (no caption of its own — captions print once,
# under the bottom half) was correctly dropped by the caption gate, but
# the BOTTOM half — which DOES sit next to the printed caption — passed
# every check on its own and was emitted as if it were the WHOLE
# figure, silently cropping off everything above the page break. That
# is exactly the kind of wrong-not-missing output the "never guess,
# never approximate" policy exists to prevent: a truncated diagram is
# Case A (book figure) done INCORRECTLY, not the safe Case B fallback.
# ------------------------------------------------------------------
_PAGE_EDGE_TOLERANCE_PT = 6.0          # how close to the physical page edge counts as "touching" it
_CROSS_PAGE_MIN_X_OVERLAP_FRACTION = 0.5  # horizontal alignment required to treat two edge clusters as one figure


def _touches_bottom_edge(rect, page_rect) -> bool:
    return rect.y1 >= page_rect.y1 - _PAGE_EDGE_TOLERANCE_PT


def _touches_top_edge(rect, page_rect) -> bool:
    return rect.y0 <= page_rect.y0 + _PAGE_EDGE_TOLERANCE_PT


def _x_overlap_fraction(a, b) -> float:
    """Fraction of the NARROWER rect's width that horizontally overlaps
    the other — used to tell 'the same figure continuing onto the next
    page' (near-identical x-range) apart from 'two unrelated figures
    that each merely happen to touch a page edge' (little/no x overlap)."""
    ix0, ix1 = max(a.x0, b.x0), min(a.x1, b.x1)
    inter = max(0.0, ix1 - ix0)
    narrower = min(a.width, b.width)
    return inter / narrower if narrower > 0 else 0.0


def _stack_pixmaps_vertically(pix1, pix2) -> bytes:
    """Vertically concatenates two PyMuPDF pixmaps — the tail of one page
    and the head of the next — into a single PNG, left-aligned and
    padded to the wider of the two with white so neither half is
    stretched or distorted. Pillow is already a hard project dependency
    (see requirements.txt; used the same way by diagram_final_check.py,
    html_renderer.py, and others)."""
    from PIL import Image
    import io as _io
    img1 = Image.open(_io.BytesIO(pix1.tobytes("png"))).convert("RGB")
    img2 = Image.open(_io.BytesIO(pix2.tobytes("png"))).convert("RGB")
    width = max(img1.width, img2.width)
    stacked = Image.new("RGB", (width, img1.height + img2.height), "white")
    stacked.paste(img1, (0, 0))
    stacked.paste(img2, (0, img1.height))
    buf = _io.BytesIO()
    stacked.save(buf, format="PNG")
    return buf.getvalue()


def _stitch_cross_page_figures(doc, clusters_by_page: dict) -> tuple[list, set]:
    """Detects a vector-drawn figure split by a page break: a cluster on
    page N touching page N's BOTTOM edge, immediately followed by a
    cluster on page N+1 touching page N+1's TOP edge, aligned in x (the
    same figure continuing — not two unrelated figures that each
    happen to sit at a page edge). Gated exactly like every other
    vector figure — a numbered caption must be found (checked next to
    the continuation first, since that's where a book's layout engine
    actually prints it, then defensively above the first-page part) —
    so this can only ever ADD a correctly-stitched figure, never
    fabricate one where no printed caption confirms it belongs.

    Returns (stitched_images, consumed_cluster_keys). `stitched_images`
    is already in the same [{"bytes", "ext", "figure_ref", "source"}, ...]
    shape extract_diagram_images returns for every other candidate.
    `consumed_cluster_keys` lets the caller exclude these same clusters
    from _vector_candidates' normal single-page emission (its `exclude`
    parameter), so a stitched figure never ALSO ships a second,
    truncated duplicate of its own bottom half.
    """
    stitched: list = []
    consumed: set = set()

    for page_index in range(doc.page_count - 1):
        page1, page2 = doc[page_index], doc[page_index + 1]
        rect1, rect2 = page1.rect, page2.rect

        bottom_clusters = [c for c in clusters_by_page.get(page_index, [])
                            if _touches_bottom_edge(c, rect1)]
        top_clusters = [c for c in clusters_by_page.get(page_index + 1, [])
                         if _touches_top_edge(c, rect2)]
        if not bottom_clusters or not top_clusters:
            continue

        for c1 in bottom_clusters:
            best, best_overlap = None, 0.0
            for c2 in top_clusters:
                if _cluster_key(page_index + 1, c2) in consumed:
                    continue
                overlap = _x_overlap_fraction(c1, c2)
                if overlap > best_overlap:
                    best, best_overlap = c2, overlap
            if best is None or best_overlap < _CROSS_PAGE_MIN_X_OVERLAP_FRACTION:
                continue
            c2 = best

            ref = _find_caption_figure_ref(page2, (c2.x0, c2.y0, c2.x1, c2.y1))
            if not ref:
                ref = _find_caption_figure_ref(page1, (c1.x0, c1.y0, c1.x1, c1.y1))
            if not ref:
                continue  # no verifiable caption — leave both halves alone, same as any other ungated vector cluster

            try:
                r1 = (fitz.Rect(c1.x0, c1.y0, c1.x1, c1.y1) + (-3, -3, 3, 0)) & rect1
                r2 = (fitz.Rect(c2.x0, c2.y0, c2.x1, c2.y1) + (-3, 0, 3, 3)) & rect2
                pix1 = page1.get_pixmap(clip=r1, dpi=RENDER_DPI)
                pix2 = page2.get_pixmap(clip=r2, dpi=RENDER_DPI)
                png_bytes = _stack_pixmaps_vertically(pix1, pix2)
            except Exception as e:
                logger.warning(f"⚠️ Failed to render cross-page diagram spanning pages "
                                f"{page_index + 1}-{page_index + 2} (figure_ref={ref}): {e}")
                continue

            figure_ref = _cross_verify_figure_ref(page2, (c2.x0, c2.y0, c2.x1, c2.y1), ref, "vector_multipage")
            stitched.append({"bytes": png_bytes, "ext": "png", "figure_ref": figure_ref,
                              "source": "vector_multipage"})
            consumed.add(_cluster_key(page_index, c1))
            consumed.add(_cluster_key(page_index + 1, c2))
            logger.info(f"📎 Stitched a two-page figure '{figure_ref}' spanning pages "
                        f"{page_index + 1}-{page_index + 2} (x-overlap {best_overlap:.0%}).")
            break  # one continuation match per bottom-edge cluster is enough

    return stitched, consumed


def _vision_candidates(pdf_path: str, doc, class_name: int) -> list[tuple]:
    """Returns [(page_index, bbox_pdf_points, 'vision', ref), ...] — the
    fallback path for books where _text_layer_reliable(doc) is False, so
    the normal vector-cluster + PDF-text-caption path (_vector_candidates)
    cannot work at all (see module + _text_layer_reliable comments). Bbox
    coordinates come back from Gemini Vision as 0..1 page fractions and
    are converted to PDF points here so downstream code (the final crop
    loop in extract_diagram_images) never has to know which path a given
    candidate came from. Lazy-imports vision_ocr to avoid a module-load-
    time circular import (vision_ocr -> solver -> book_figure_index ->
    book_diagram_extractor); only needed, and only paid for, on the rare
    unreliable-text-layer path, and even then only once per PDF thanks to
    book_figure_index.py's on-disk caching of whatever this returns."""
    from vision_ocr import locate_figures_via_vision

    out = []
    try:
        figures = locate_figures_via_vision(pdf_path, class_name=class_name)
    except Exception as e:
        logger.warning(f"⚠️ Vision figure-location failed for {pdf_path} ({e}) — "
                        f"no book diagrams will be found on this unreliable-text-layer PDF; "
                        f"all questions fall back to AI-generated diagrams.")
        return out

    for fig in figures:
        page_index = fig["page_index"]
        if not (0 <= page_index < doc.page_count):
            continue
        page_rect = doc[page_index].rect
        x0n, y0n, x1n, y1n = fig["bbox_norm"]
        bbox = (
            page_rect.x0 + x0n * page_rect.width,
            page_rect.y0 + y0n * page_rect.height,
            page_rect.x0 + x1n * page_rect.width,
            page_rect.y0 + y1n * page_rect.height,
        )
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < MIN_DIAGRAM_WIDTH_PT or h < MIN_DIAGRAM_HEIGHT_PT:
            continue
        # Defensive dedup: the vision model can occasionally return two
        # overlapping boxes for what's physically one figure (e.g. one
        # tight, one slightly looser). Two "different figures" that
        # overlap this heavily on the same page would be visually
        # indistinguishable to a student anyway, so keep only the first
        # (and prefer whichever one actually carries a caption ref).
        dup_of = None
        for existing in out:
            if existing[0] != page_index:
                continue
            a, b = fitz.Rect(bbox), fitz.Rect(existing[1])
            inter = a & b
            if inter and not inter.is_empty and inter.get_area() > 0.55 * min(a.get_area(), b.get_area()):
                dup_of = existing
                break
        if dup_of is not None:
            if dup_of[3] is None and fig.get("figure_ref"):
                out[out.index(dup_of)] = (page_index, dup_of[1], "vision", fig.get("figure_ref"))
            continue
        out.append((page_index, bbox, "vision", fig.get("figure_ref")))
    return out


def merge_raster_and_vector_candidates(raster: list[tuple], vector: list[tuple]) -> list[list]:
    """Merges `raster` ([(page_index, bbox, "raster"), ...]) with `vector`
    ([(page_index, bbox, kind, figure_ref), ...], kind "vector" or "vision")
    into one list of [page_index, bbox, kind, figure_ref].

    When a vector/vision candidate's bbox is essentially the same region as
    an already-found raster candidate on the same page (>60% area overlap
    relative to the smaller box), the raster crop is kept — it's already
    known-correct pixels, rendered straight from the page — but if the
    vector/vision path independently read a figure number for that same
    physical figure, that number is ADOPTED onto the raster entry rather
    than discarded.

    THIS IS A DELIBERATE FIX, not the original behaviour. The previous
    version paired every raster bbox with a hardcoded figure_ref=None and,
    on finding an overlapping vector/vision candidate, dropped that
    candidate — ref included — entirely:

        combined = [(pi, bbox, "raster", None) for pi, bbox, _kind in raster]
        for pi, bbox, kind, ref in vector:
            if not _overlaps_existing(pi, bbox):
                combined.append((pi, bbox, kind, ref))

    For any book whose real figures are embedded raster images (the common
    case — confirmed for this project's own cached temp/figure_index/*/
    manifests, which show every single entry as source="raster",
    figure_ref=null), this guaranteed every raster figure stayed ref=None
    forever, even on a run where the vector-caption or Gemini Vision path
    correctly identified its number, because that number had nowhere to
    go. This was the actual root cause of the "figure_ref remains null"
    bug — not a failure of caption reading or vision itself.

    The fix mirrors the dedup pattern already used inside
    `_vision_candidates` for two overlapping *vision* boxes (see its
    "Defensive dedup" comment): keep the known-good crop, adopt whichever
    ref exists. See test_book_diagram_extractor.py::test_merge_candidates_* for the
    regression tests covering both the original bug and this fix.
    """
    combined: list[list] = [[pi, bbox, "raster", None] for pi, bbox, _kind in raster]
    for pi, bbox, kind, ref in vector:
        idx = _overlap_index(combined, pi, bbox)
        if idx is None:
            combined.append([pi, bbox, kind, ref])
        elif ref and combined[idx][3] is None:
            # Same physical figure: keep the raster crop, adopt the number.
            combined[idx][3] = ref
    return combined


def _overlap_index(combined_list, pi, bbox):
    """Pure-arithmetic bounding-box overlap check (no PyMuPDF dependency —
    intentional, so this and merge_raster_and_vector_candidates stay
    importable/testable in an environment without `fitz` installed).
    Returns the index of the first same-page entry in `combined_list`
    whose bbox overlaps `bbox` by more than 60% of the smaller box's
    area, or None if there's no such entry."""
    bx0, by0, bx1, by1 = bbox
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    for idx, (r_pi, r_bbox, _kind, _ref) in enumerate(combined_list):
        if r_pi != pi:
            continue
        rx0, ry0, rx1, ry1 = r_bbox
        r_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
        ix0, iy0 = max(bx0, rx0), max(by0, ry0)
        ix1, iy1 = min(bx1, rx1), min(by1, ry1)
        inter_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter_area > 0 and inter_area > 0.6 * min(b_area, r_area):
            return idx
    return None


# Exposed as an attribute for direct unit testing of the overlap check in
# isolation (see test_book_diagram_extractor.py) — bound at module-import
# time so it's always available regardless of test execution order.
merge_raster_and_vector_candidates._overlap_index = _overlap_index


def extract_diagram_images(pdf_path: str, class_name: int = 0) -> list[dict]:
    """Returns [{"bytes": <PNG bytes>, "ext": "png", "figure_ref": str|None}, ...]
    in page order, filtered to plausible per-question diagram figures. Each
    image is a freshly-rendered pixmap of that exact page region — never raw
    extracted bytes — so rotation/mirroring baked into the page's own
    transform is always correct, whether the underlying figure is an
    embedded raster image, drawn with vector path operators, or located via
    the vision fallback. "figure_ref" (e.g. "3.14"), when present, lets
    solver.py match this exact image to a question that names that figure
    number explicitly, rather than relying only on page order.

    class_name is only used if the vision fallback path is taken (unreliable
    text layer) — it's passed through to the vision prompt for context and
    has no effect on the text-layer-reliable path.
    """
    doc = fitz.open(pdf_path)

    text_layer_ok = _text_layer_reliable(doc)
    raster = _raster_candidates(doc)                                  # (pi, bbox, 'raster')

    cross_page_images: list = []

    if text_layer_ok:
        # Normal path: real embedded text lets us read printed captions
        # directly off the page, so vector clustering + caption-gating
        # (see module docstring) is precise and cheap (zero extra API calls).
        # Clustering is computed once per page and shared between the
        # cross-page continuation check and the normal single-page path
        # (see _page_vector_clusters).
        clusters_by_page = {p: _page_vector_clusters(doc[p]) for p in range(doc.page_count)}
        cross_page_images, consumed = _stitch_cross_page_figures(doc, clusters_by_page)
        vector = _vector_candidates(doc, clusters_by_page=clusters_by_page, exclude=consumed)  # (pi, bbox, 'vector', ref)
    else:
        # Fallback path: this book's body text is itself vector-drawn
        # (outlined glyphs), so page.get_textbox() cannot read a caption
        # for ANY vector cluster — the caption gate would silently reject
        # every real figure on the page. Ask Gemini Vision to locate
        # figures + captions directly from the rendered page image instead
        # (see vision_ocr.locate_figures_via_vision for why this is safe:
        # it only answers "where/what number", the actual crop pixels
        # still come from the deterministic page.get_pixmap call below).
        logger.info(f"📖 {pdf_path}: PDF text layer unreliable for caption reading — "
                    f"using Gemini Vision to locate figures instead of vector-cluster "
                    f"+ text-caption matching.")
        vector = _vision_candidates(pdf_path, doc, class_name)         # (pi, bbox, 'vision', ref)

    combined = merge_raster_and_vector_candidates(raster, vector)
    combined.sort(key=lambda t: (t[0], t[1][1]))  # page order, then top-to-bottom

    images = []
    non_raster_count = 0
    for page_index, bbox, kind, precomputed_ref in combined:
        try:
            page = doc[page_index]
            # Small padding around the detected bbox so thin border
            # strokes / labels right at the edge of the figure aren't
            # clipped off.
            rect = fitz.Rect(bbox) + (-3, -3, 3, 3)
            rect = rect & page.rect  # clamp to the actual page bounds
            pix = page.get_pixmap(clip=rect, dpi=RENDER_DPI)
            png_bytes = pix.tobytes("png")
            if precomputed_ref:
                figure_ref = precomputed_ref
            elif text_layer_ok:
                figure_ref = _find_caption_figure_ref(page, bbox)
            else:
                # No page-order fallback exists anywhere downstream —
                # solver.py::_attach_book_diagrams is strict exact-ref-only
                # (see its own docstring). An image with figure_ref=None
                # simply becomes unmatchable to any question and is
                # correctly never attached, per the STRICT HIERARCHY
                # policy — this is NOT a fallback matching path, just the
                # deliberate absence of one. (This comment previously
                # claimed a "page-order fallback handles matching" here,
                # which was stale/inaccurate — found during this audit.)
                figure_ref = None

            # Stage-2 validation: an independent, deterministic (non-AI)
            # OCR opinion on the same caption band. See module docstring
            # for the full policy — this can only ever REMOVE a ref it
            # actively disagrees with; it never invents one. # type: ignore
            figure_ref = _cross_verify_figure_ref(page, bbox, figure_ref, kind)
        except Exception as e:
            logger.warning(f"⚠️ Failed to render diagram region (kind={kind}, "
                            f"page {page_index + 1}): {e}")
            continue
        if kind != "raster":
            non_raster_count += 1
        images.append({"bytes": png_bytes, "ext": "png", "figure_ref": figure_ref, "source": kind})

    raster_count = len(images) - non_raster_count  # count before cross-page images are folded in
    images.extend(cross_page_images)

    doc.close()
    with_ref = sum(1 for img in images if img.get("figure_ref"))
    logger.info(f"🖼️ Extracted {len(images)} candidate book-diagram image(s) "
                f"({raster_count} raster, {non_raster_count} "
                f"{'vector-drawn' if text_layer_ok else 'vision-located'}, "
                f"{len(cross_page_images)} stitched across a page break, "
                f"{with_ref} with a readable figure-number caption) — rendered "
                f"via page.get_pixmap, transformation-matrix-correct, never mirrored.")
    return images
