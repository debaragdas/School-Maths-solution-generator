"""
diagram_final_check.py — THE FINAL VALIDATION STAGE, run once per
question after rendering has already happened and immediately before
that question's HTML (and eventually the whole exercise's PDF) is
assembled. This is deliberately a SEPARATE stage from
diagram_decision.py: the decision engine decides what SHOULD be drawn
before any drawing happens; this module checks what WAS ACTUALLY
produced, because a decision of GENERATED_DIAGRAM or BOOK_DIAGRAM is a
prediction, not a guarantee — diagram_renderer.py's own render_diagram
can still come back empty for reasons the decision engine cannot see in
advance (an unexpected exception, its own post-render marker-coverage
gate rejecting the result), and a book crop can still be a genuinely
bad image (corrupt, blank, absurdly small) even though its citation was
verified correctly.

Full pipeline, end to end:

    Question text
        │  (AI — prompts.py + solver.py's parsing)
        ▼
    Gemini's raw understanding (diagram_spec, book_diagram_base64, ...)
        │
        ▼  diagram_decision.py — DECISION (deterministic, no drawing)
    NO_DIAGRAM | BOOK_DIAGRAM | GENERATED_DIAGRAM
        │
        ▼  diagram_renderer.py — RENDERING (deterministic, no deciding)
    "" | <svg>...</svg>   (for GENERATED_DIAGRAM)
    book_diagram_base64   (for BOOK_DIAGRAM, untouched pass-through)
        │
        ▼  THIS MODULE — FINAL VALIDATION (deterministic, no drawing,
        │  no deciding what SHOULD happen — only confirms what DID)
    reconciled, provably-correct (svg, book_diagram_base64) pair
        │
        ▼  html_renderer.py — assembles the question's HTML
        ▼  pdf_generator.py — HTML -> PDF

Every check below either PROVES its outcome (states exactly what was
inspected and what threshold it passed/failed) or safely omits — this
module has no capacity to "retry" a render (that would mean re-running
diagram_renderer.py or re-asking Gemini, neither of which happens
here), so the only deterministic action ever taken on a failure is
"clear the diagram from this question and record why", never a silent
continue.
"""
import base64
import re
from io import BytesIO

from PIL import Image, ImageStat

from utils import logger
from diagram_decision import NO_DIAGRAM, BOOK_DIAGRAM, GENERATED_DIAGRAM
import vision_validation

# Minimum acceptable book-figure crop size, in actual rendered pixels
# (not the PDF-point thresholds book_diagram_extractor.py already
# applies at extraction time — this is a second, independent check on
# the FINAL bytes that made it all the way to this question, in case
# something between extraction and here ever corrupts/truncates them).
_MIN_BOOK_IMAGE_PX = 20
# A crop this visually flat (near-zero standard deviation across pixel
# values) is almost certainly a blank/whitespace region rather than an
# actual figure — the deterministic signature of a bounding box that
# missed the real diagram.
_BLANK_IMAGE_STDDEV_THRESHOLD = 2.0

_VIEWBOX_PATTERN = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
_TEXT_ELEMENT_PATTERN = re.compile(r"<text\s+([^>]*)>(.*?)</text>", re.DOTALL)
_X_ATTR_IN_TAG_PATTERN = re.compile(r'\bx="(-?[\d.]+)"')
_Y_ATTR_IN_TAG_PATTERN = re.compile(r'\by="(-?[\d.]+)"')
# Every coordinate-bearing attribute the renderer emits anywhere
# (lines, circles, text) — used for the clipping check below.
_ALL_X_ATTR_PATTERN = re.compile(r'\b(?:x|x1|x2|cx)="(-?[\d.]+)"')
_ALL_Y_ATTR_PATTERN = re.compile(r'\b(?:y|y1|y2|cy)="(-?[\d.]+)"')
# Small tolerance for stroke width / anti-aliasing right at the edge —
# not a fuzziness knob for "is this roughly inside", just enough room
# that a 2px-wide stroke centered exactly on the boundary isn't flagged.
_CLIP_MARGIN_PX = 8


def _qnum(question: dict) -> str:
    return f"{question.get('question_number')}{question.get('sub_part') or ''}"


# ---------------------------------------------------------------------
# V40 — HUMAN REVIEW ROUTING (relaxed validators).
#
# Only a genuinely TECHNICALLY INVALID output (corrupted/undecodable
# image bytes, no image at all, or a broken/missing render) is ever
# omitted outright below. Every other check in this module is a
# heuristic/confidence signal (label-overlap guess, clipping estimate,
# crop-size/blankness threshold, geometry-reproducibility or chart-
# count cross-check) — these no longer clear the diagram from the
# question. Instead they KEEP the diagram published and flag the
# question for Human Review (the same `needs_review`/`review_notes`
# mechanism math_verifier.py and solver.py's coordinate tie-break
# already use — see interactive_review.py / review_webapp.py, which
# both already surface `needs_review` to the reviewer). Human Review is
# the final decision maker on anything that isn't provably broken.
# ---------------------------------------------------------------------

def _flag_needs_review(question: dict, reason: str) -> None:
    question["needs_review"] = True
    question.setdefault("review_notes", []).append(reason)
    logger.info(f"ℹ️ Q{_qnum(question)}: diagram kept and flagged for Human Review ({reason}) — "
                f"not rejected, since this is a heuristic signal, not a technical defect.")


# ---------------------------------------------------------------------
# BOOK DIAGRAM: crop validation on the actual final bytes.
# ---------------------------------------------------------------------

def _check_book_image_decodes(base64_bytes: str) -> tuple[bool, str]:
    """TECHNICAL validity only: do the bytes decode to a real image at
    all? A decode failure (corrupt/truncated data) is the one book-
    diagram condition that is unambiguously unusable — there is
    nothing for a human reviewer to look at."""
    try:
        raw = base64.b64decode(base64_bytes)
        img = Image.open(BytesIO(raw))
        img.load()  # force full decode now, not lazily later — catches truncated data
    except Exception as e:
        return False, f"book diagram image failed to decode ({e})"
    return True, ""


def _check_book_image_quality(base64_bytes: str) -> tuple[bool, str]:
    """HEURISTIC quality signals only (never a hard rejection reason as
    of V40): a degenerately small crop or a visually flat/near-blank
    crop is often a mis-cropped bounding box, but a small or plain
    figure can legitimately look like this too — a human reviewer
    decides, this function only flags."""
    try:
        raw = base64.b64decode(base64_bytes)
        img = Image.open(BytesIO(raw))
    except Exception:
        return True, ""  # decode-validity is _check_book_image_decodes's job, not this one's

    w, h = img.size
    if w < _MIN_BOOK_IMAGE_PX or h < _MIN_BOOK_IMAGE_PX:
        return False, f"book diagram crop is only {w}x{h}px — unusually small for a real figure"

    try:
        stddev = ImageStat.Stat(img.convert("L")).stddev[0]
        if stddev < _BLANK_IMAGE_STDDEV_THRESHOLD:
            return False, f"book diagram crop appears visually flat/near-blank (pixel stddev {stddev:.2f})"
    except Exception:
        pass  # blank-detection is a defense-in-depth extra, never block on it failing to run

    return True, ""


# ---------------------------------------------------------------------
# GENERATED DIAGRAM: overlap + clipping check on the actual final SVG.
# ---------------------------------------------------------------------

def _extract_text_positions(svg: str) -> list:
    positions = []
    for attrs, _content in _TEXT_ELEMENT_PATTERN.findall(svg):
        xm = _X_ATTR_IN_TAG_PATTERN.search(attrs)
        ym = _Y_ATTR_IN_TAG_PATTERN.search(attrs)
        if xm and ym:
            positions.append((round(float(xm.group(1)), 1), round(float(ym.group(1)), 1)))
    return positions


def _check_no_overlapping_labels(svg: str) -> tuple[bool, str]:
    """Two separate <text> elements rendered at the EXACT same (x, y)
    is a strong overlap SIGNAL — not proof the published diagram is
    unusable (as of V40 this is a HEURISTIC check: it flags for Human
    Review rather than rejecting, since a human can often still read a
    mostly-overlapping label just fine). Deliberately conservative
    (exact-coordinate match only) so it can never falsely flag two
    labels that are merely close together but legitimately both
    readable; it only catches a genuine literal stack."""
    positions = _extract_text_positions(svg)
    counts: dict = {}
    for pos in positions:
        counts[pos] = counts.get(pos, 0) + 1
    duplicated = [pos for pos, count in counts.items() if count > 1]
    if duplicated:
        return False, (f"{len(duplicated)} label position(s) have more than one text element "
                        f"drawn at the exact same coordinates (e.g. {duplicated[0]})")
    return True, ""


def _check_no_clipped_content(svg: str) -> tuple[bool, str]:
    """Every coordinate-bearing attribute the renderer emits (line
    endpoints, circle centers, text positions) should fall within the
    SVG's own declared viewBox, with a small tolerance for stroke width.
    A coordinate meaningfully outside that box is a strong SIGNAL of
    content that would be clipped by the viewBox (as of V40 this is a
    HEURISTIC check: it flags for Human Review rather than rejecting,
    since some clipping is cosmetic/minor and a human can judge that
    faster and more accurately than a fixed pixel-margin guess)."""
    vb = _VIEWBOX_PATTERN.search(svg)
    if not vb:
        return True, ""  # no viewBox to check against — nothing to prove either way
    width, height = float(vb.group(1)), float(vb.group(2))

    bad_x = [float(m.group(1)) for m in _ALL_X_ATTR_PATTERN.finditer(svg)
             if float(m.group(1)) < -_CLIP_MARGIN_PX or float(m.group(1)) > width + _CLIP_MARGIN_PX]
    bad_y = [float(m.group(1)) for m in _ALL_Y_ATTR_PATTERN.finditer(svg)
             if float(m.group(1)) < -_CLIP_MARGIN_PX or float(m.group(1)) > height + _CLIP_MARGIN_PX]
    if bad_x or bad_y:
        offender = bad_x[0] if bad_x else bad_y[0]
        return False, f"content drawn outside the {width:.0f}x{height:.0f} viewBox (e.g. coordinate {offender:.1f})"
    return True, ""


# ---------------------------------------------------------------------
# THE FINAL CHECK ITSELF.
# ---------------------------------------------------------------------

def final_pre_pdf_check(question: dict, diagram_svg: str) -> dict:
    """Call this once per question, AFTER diagram_renderer.render_diagram
    has already produced `diagram_svg` for it (empty string if nothing
    rendered). Returns {"diagram_svg", "book_diagram_base64",
    "book_diagram_mime", "book_diagram_figure_ref", "final_decision",
    "final_reason"} — the reconciled, provably-checked values that
    should actually be used to build this question's HTML. Never
    raises; every failure mode degrades to omitting the diagram and
    recording why, never a silent continue."""
    decision = question.get("diagram_decision")
    book_b64 = question.get("book_diagram_base64")
    book_mime = question.get("book_diagram_mime")
    book_ref = question.get("book_diagram_figure_ref")

    if decision is None:
        # Backward compatibility: if the caller never ran
        # diagram_decision.decide_diagram() on this question (e.g. a
        # direct call to render_exercise_html that bypasses solver.py's
        # pipeline entirely), infer the decision from whichever fields
        # are actually present rather than defaulting to NO_DIAGRAM and
        # incorrectly nuking a legitimate diagram this function was
        # never told the provenance of.
        if book_b64:
            decision = BOOK_DIAGRAM
        elif diagram_svg:
            decision = GENERATED_DIAGRAM
        else:
            decision = NO_DIAGRAM

    def _omit(reason: str) -> dict:
        logger.warning(f"⚠️ Q{_qnum(question)}: final pre-PDF check omitting this diagram "
                        f"({reason}).")
        return {"diagram_svg": "", "book_diagram_base64": None, "book_diagram_mime": None,
                "book_diagram_figure_ref": None, "final_decision": NO_DIAGRAM, "final_reason": reason}

    def _ok(reason: str) -> dict:
        return {"diagram_svg": diagram_svg, "book_diagram_base64": book_b64,
                "book_diagram_mime": book_mime, "book_diagram_figure_ref": book_ref,
                "final_decision": decision, "final_reason": reason}

    if decision == GENERATED_DIAGRAM:
        # "no missing required diagram": the decision engine approved a
        # diagram_spec, but render_diagram can still legitimately come
        # back empty (an unexpected exception, or its own post-render
        # marker-coverage gate rejecting the output) — both already
        # logged at the point of failure inside diagram_renderer.py,
        # but this question's OWN diagram_decision field must not go on
        # claiming "GENERATED_DIAGRAM" when nothing was actually drawn.
        # This is a genuine TECHNICAL failure (broken/missing render —
        # there is nothing to show anyone), so it's still a hard reject.
        if not diagram_svg:
            return _omit("decision was GENERATED_DIAGRAM but rendering produced no SVG "
                          "(see diagram_renderer.py's own warning log for the specific cause)")

        # TECHNICAL validity only — a non-finite coordinate or a
        # malformed/degenerate viewBox is a genuinely corrupted/broken
        # SVG, not a judgement call, so this alone can still reject.
        ok, reason = vision_validation.check_no_invalid_numbers(diagram_svg)
        if not ok:
            return _omit(f"vision validation: {reason}")
        ok, reason = vision_validation.check_valid_viewbox(diagram_svg)
        if not ok:
            return _omit(f"vision validation: {reason}")

        # HEURISTIC signals from here on (label overlap, clipping
        # estimate, geometry-reproducibility self-check, chart-count
        # cross-check) — never reject; keep the diagram published and
        # flag the question for Human Review instead, which is the
        # final decision maker on anything that isn't provably broken.
        ok, reason = _check_no_overlapping_labels(diagram_svg)
        if not ok:
            _flag_needs_review(question, f"diagram check: {reason}")
        ok, reason = _check_no_clipped_content(diagram_svg)
        if not ok:
            _flag_needs_review(question, f"diagram check: {reason}")
        ok, reason = vision_validation.check_triangle_geometry_reproducible(question.get("diagram_spec"))
        if not ok:
            _flag_needs_review(question, f"vision validation: {reason}")
        ok, reason = vision_validation.check_chart_element_count_matches_data(
            question.get("diagram_spec"), diagram_svg)
        if not ok:
            _flag_needs_review(question, f"vision validation: {reason}")

        if question.get("needs_review"):
            return _ok("generated diagram passed technical validation but was flagged for "
                        "Human Review by one or more heuristic checks (see review_notes) — "
                        "diagram is still published; a human makes the final call")
        return _ok("generated diagram passed the final pre-PDF check "
                    "(no missing render, no technical defects, and every heuristic check clean)")

    if decision == BOOK_DIAGRAM:
        # TECHNICAL validity only: no bytes at all, or bytes that don't
        # decode to a real image, are genuinely unusable — there is
        # nothing for a human to review — so these alone can still
        # reject.
        if not book_b64:
            return _omit("decision was BOOK_DIAGRAM but no book_diagram_base64 is actually present")
        ok, reason = _check_book_image_decodes(book_b64)
        if not ok:
            return _omit(reason)

        # HEURISTIC signal from here on (small-crop / near-blank
        # threshold) — never reject; keep the crop published and flag
        # for Human Review, which can tell at a glance whether a small
        # or plain-looking crop is actually fine.
        ok, reason = _check_book_image_quality(book_b64)
        if not ok:
            _flag_needs_review(question, f"diagram check: {reason}")

        if question.get("needs_review"):
            return _ok(f"book diagram (চিত্ৰ {book_ref}) is decodable and published, but was "
                       f"flagged for Human Review (see review_notes) — a human makes the final call")
        return _ok(f"book diagram (চিত্ৰ {book_ref}) passed the final pre-PDF check "
                    f"(decodable, correctly sized, not blank)")

    # decision == NO_DIAGRAM (or missing/unrecognized): "correct diagram
    # selected" also means proving the NEGATIVE case — nothing should
    # exist here at all. If something does (a regression anywhere
    # upstream leaving stale fields set), clear it rather than let a
    # stray diagram slip through a path this check didn't anticipate.
    if diagram_svg or book_b64:
        return _omit("decision was NO_DIAGRAM but a diagram_svg/book_diagram_base64 was still "
                      "present on this question — clearing it rather than trusting a decision "
                      "the pipeline itself just contradicted")
    return _ok("no diagram was required and none is present")
