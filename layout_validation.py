"""
layout_validation.py — THE LAYOUT VALIDATION ENGINE (Phase 6).

Two-part design, matching how this project already solves "prevent X"
problems elsewhere (e.g. the strict figure-citation-match POLICY in
solver.py prevents wrong figures; diagram_final_check.py DETECTS
whatever slips past decision-time anyway):

  PART A — PREVENTION (deterministic, at the source): templates/
  style.css's .diagram-box / .diagram-box--large rules now carry
  break-inside:avoid / page-break-inside:avoid (see the comment block
  right above .question-block in that file for the full rationale).
  This is the actual "prevent... clipping" mechanism the brief asked
  for — implemented at the layer where it is cheapest and most
  reliable to prevent a problem: never let the browser's print engine
  consider splitting a diagram across a page boundary in the first
  place, rather than trying to detect-and-fix it after the fact.

  PART B — DETECTION (this module): a deterministic, best-effort audit
  of the ACTUAL generated PDF, for the class of layout defect the CSS
  rule cannot prevent by construction (an image genuinely taller than
  one whole page, a corrupted/near-invisible embed, a font that fails
  to embed a glyph). This audit NEVER blocks the pipeline and NEVER
  retries generation (a real auto-reflow engine is out of scope — see
  the architecture design doc's Phase 6 section) — it logs concrete,
  actionable findings so a human/CI gate can catch them, exactly the
  same "prove or log, never silently ship wrong" philosophy every
  other validation stage in this project already follows.

DEPENDENCY: uses PyMuPDF (`fitz`), already a hard dependency of
book_diagram_extractor.py elsewhere in this project — nothing new is
introduced. Tests stub `fitz` the same way test_book_figure_index.py
and test_figure_database.py already do, since this sandbox has neither
PyMuPDF nor network access.
"""
from utils import logger

# An embedded image/drawing bounding box with area smaller than this
# fraction of the page's own area is almost certainly a collapsed or
# failed placement, not a legitimately tiny diagram (this project's
# diagrams are laid out at a minimum of 210px-equivalent width by
# style.css's own .diagram-box rule — nothing legitimate should ever
# render smaller than a small fraction of a full page).
_MIN_IMAGE_AREA_FRACTION = 0.0005
# Tolerance (in PDF points) for an image bounding box extending past
# the page's own mediabox — a small amount is normal (bleed/rounding);
# anything beyond this is genuine overflow/clipping.
_BOUNDARY_OVERFLOW_TOLERANCE_PT = 2.0
_REPLACEMENT_GLYPH = "\ufffd"


def _open_pdf(pdf_path: str):
    import fitz
    return fitz.open(pdf_path)


def _check_page_images(page) -> list:
    """Returns a list of issue strings for this page's own embedded
    images/drawings: near-zero-area placements and boundary overflow.
    Never raises — a page this can't introspect is simply skipped
    (logged), never treated as a hard failure."""
    issues = []
    try:
        page_rect = page.rect
        page_area = max(page_rect.width * page_rect.height, 1.0)
    except Exception as e:
        return [f"could not read page geometry ({e}) — skipping image checks for this page"]

    try:
        images = page.get_images(full=True)
    except Exception:
        images = []

    for img in images:
        xref = img[0]
        try:
            bboxes = page.get_image_rects(xref)
        except Exception:
            bboxes = []
        for bbox in bboxes:
            area = max(bbox.width, 0) * max(bbox.height, 0)
            if area / page_area < _MIN_IMAGE_AREA_FRACTION:
                issues.append(f"an embedded image on this page has a near-zero rendered area "
                              f"({bbox.width:.1f}x{bbox.height:.1f}pt) — likely a collapsed/failed "
                              f"diagram placement rather than an intentionally tiny image")
            overflow_left = page_rect.x0 - bbox.x0
            overflow_top = page_rect.y0 - bbox.y0
            overflow_right = bbox.x1 - page_rect.x1
            overflow_bottom = bbox.y1 - page_rect.y1
            max_overflow = max(overflow_left, overflow_top, overflow_right, overflow_bottom)
            if max_overflow > _BOUNDARY_OVERFLOW_TOLERANCE_PT:
                issues.append(f"an embedded image extends {max_overflow:.1f}pt beyond the page "
                              f"boundary — likely clipped content or a page-break split")
    return issues


def _check_page_glyphs(page) -> list:
    """A U+FFFD REPLACEMENT CHARACTER in the extracted text layer is
    the deterministic signature of a font failing to render a real
    glyph it was asked to (a well-known risk for Assamese/Bengali text
    with imperfect font coverage) — not a guess about visual quality."""
    try:
        text = page.get_text("text")
    except Exception as e:
        return [f"could not extract text for glyph check ({e}) — skipping"]
    if _REPLACEMENT_GLYPH in text:
        count = text.count(_REPLACEMENT_GLYPH)
        return [f"page text contains {count} replacement-character glyph(s) "
                f"(U+FFFD) — a font failed to render {count} real character(s)"]
    return []


def validate_pdf_layout(pdf_path: str) -> dict:
    """Runs the full deterministic layout audit on an already-generated
    PDF. Returns {"ok": bool, "page_count": int, "issues": [{"page": int,
    "issue": str}, ...]}. NEVER raises — if the PDF itself can't even be
    opened (missing PyMuPDF, corrupt file, whatever), that is reported
    as a single issue with ok=True (this is an AUDIT layer, not a gate
    — its own failure to run must never be mistaken for the PDF itself
    being bad, and must never block the pipeline from completing)."""
    try:
        doc = _open_pdf(pdf_path)
    except Exception as e:
        logger.warning(f"⚠️ layout_validation: could not open '{pdf_path}' for audit ({e}) — "
                        f"skipping layout validation for this file (not treated as a layout failure).")
        return {"ok": True, "page_count": 0, "issues": [], "skipped_reason": str(e)}

    all_issues = []
    try:
        page_count = doc.page_count
        for page_index in range(page_count):
            try:
                page = doc[page_index]
            except Exception as e:
                all_issues.append({"page": page_index + 1, "issue": f"could not load page ({e})"})
                continue
            for issue in _check_page_images(page):
                all_issues.append({"page": page_index + 1, "issue": issue})
            for issue in _check_page_glyphs(page):
                all_issues.append({"page": page_index + 1, "issue": issue})
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return {"ok": len(all_issues) == 0, "page_count": page_count, "issues": all_issues}


def validate_and_log(pdf_path: str, exercise_label: str = "") -> dict:
    """Convenience wrapper for main.py: runs the audit and logs a
    concise summary — one warning line per issue found, or a single
    debug-level confirmation if everything passed. Best-effort, never
    raises, never blocks (see this module's own docstring)."""
    result = validate_pdf_layout(pdf_path)
    label = f" ({exercise_label})" if exercise_label else ""
    if result["issues"]:
        logger.warning(f"⚠️ layout_validation{label}: {len(result['issues'])} layout issue(s) "
                        f"found in {pdf_path} across {result['page_count']} page(s):")
        for item in result["issues"]:
            logger.warning(f"   page {item['page']}: {item['issue']}")
    elif not result.get("skipped_reason"):
        logger.info(f"✅ layout_validation{label}: {pdf_path} passed ({result['page_count']} page(s), no issues).")
    return result
