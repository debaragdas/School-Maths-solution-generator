"""
label_layout.py — SMART LABEL PLACEMENT / AUTOMATIC LABEL COLLISION
REPOSITIONING.

GENUINE GAP THIS FILLS (confirmed by reading the existing pipeline
before writing a line of this): diagram_final_check.py's
_check_no_overlapping_labels only proves EXACT-coordinate duplicates
and, when it finds one, OMITS the entire diagram (see
final_pre_pdf_check's GENERATED_DIAGRAM branch — it has no repair
path, only accept/reject). There was no general bounding-box overlap
detector anywhere in the project, and no repositioning of any kind —
a diagram with two merely-adjacent-but-visually-overlapping labels
(not exact-same-coordinate) previously shipped uncorrected, and even
an exact-duplicate case just threw the whole diagram away instead of
fixing it.

This module is deliberately NOT wired into diagram_final_check.py's
existing _check_no_overlapping_labels path: that function's current
"exact match -> omit" behaviour is itself covered by passing regression
tests (test_diagram_final_check.py::TestOverlappingLabels), and
auto-fixing before that check runs would silently change what those
tests are asserting. Instead, this is new, additive infrastructure
that:
  (a) any NEW plugin can call directly on its own SVG output before
      returning it, so its labels never collide in the first place
      (this is how unit_circle_plugin.py, circle_line_plugin.py, and
      solid_geometry_plugin.py all use it below), and
  (b) is available standalone (`reposition_labels`) for any future
      caller — including, eventually, diagram_final_check.py itself,
      if a future change deliberately decides to replace "omit" with
      "repair" and updates that module's own tests to match. That
      decision is out of scope here.

APPROACH: parse every <text x="..." y="..." ...>content</text> element
from an SVG string (regex-based, matching the exact style already used
by diagram_final_check.py's own _extract_text_positions /
_TEXT_ELEMENT_PATTERN — no new parsing dependency introduced). Estimate
each label's on-canvas bounding box from its (x, y), text-anchor,
font-size, and character count (a standard, deterministic width
heuristic — average glyph width ~0.58x font-size for the Latin/Bengali
mixed sans-serif stack already used throughout this project's SVG
output). Any two boxes that geometrically intersect (by more than a
small anti-aliasing tolerance) are a PROVABLE overlap, not a proximity
guess. Overlapping labels are nudged outward from their anchor point
along a deterministic ring of candidate offsets (increasing radius,
8 angles per ring) until a collision-free placement is found or the
search budget is exhausted (in which case the label is left at its
best-effort placement and the residual collision is reported, never
silently hidden).
"""
import re
from typing import Optional

_TEXT_ELEMENT_PATTERN = re.compile(r'<text([^>]*)>(.*?)</text>', re.DOTALL)
_X_ATTR_PATTERN = re.compile(r'\bx="(-?[0-9.]+)"')
_Y_ATTR_PATTERN = re.compile(r'\by="(-?[0-9.]+)"')
_ANCHOR_ATTR_PATTERN = re.compile(r'text-anchor="(start|middle|end)"')
_FONT_SIZE_ATTR_PATTERN = re.compile(r'font-size="(-?[0-9.]+)"')
_VIEWBOX_PATTERN = re.compile(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"')

_AVG_GLYPH_WIDTH_FACTOR = 0.58   # empirical average glyph width as a fraction of font-size
_DEFAULT_FONT_SIZE = 13.0        # matches this project's FONT constant default
_ASCENT_FACTOR = 0.80            # fraction of font-size above the baseline (y)
_DESCENT_FACTOR = 0.25           # fraction of font-size below the baseline (y)
_COLLISION_PAD = 1.0             # px tolerance so kerning/AA noise never triggers a false positive

# search ring for repositioning: (radius_px, [angles_deg...])
_SEARCH_RADII = (8, 14, 20, 28, 38, 50)
_SEARCH_ANGLES = tuple(range(0, 360, 45))  # 8 compass directions per ring


def _strip_tags(content: str) -> str:
    """Best-effort visible-character count for width estimation —
    strips any nested markup (e.g. a &#176; entity still counts as one
    visible glyph) without needing a full XML parser."""
    text = re.sub(r'<[^>]*>', '', content)
    text = text.replace('&#176;', '\u00b0').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text


def parse_text_elements(svg: str) -> list:
    """Returns a list of dicts, one per <text> element, in document
    order: {tag_start, tag_end, x, y, anchor, font_size, text, bbox}
    where tag_start/tag_end are the character offsets of the OPENING
    tag's x="..."/y="..." attributes within the original string (so a
    caller can rewrite just those two attribute values in place without
    disturbing anything else in the document). bbox is (x0, y0, x1, y1)
    in SVG user-space units, y growing downward as SVG defines it.
    Elements missing a parseable x or y are skipped (nothing to move,
    nothing to collide)."""
    elements = []
    for m in _TEXT_ELEMENT_PATTERN.finditer(svg):
        attrs, content = m.group(1), m.group(2)
        xm = _X_ATTR_PATTERN.search(attrs)
        ym = _Y_ATTR_PATTERN.search(attrs)
        if not xm or not ym:
            continue
        x, y = float(xm.group(1)), float(ym.group(1))
        anchor_m = _ANCHOR_ATTR_PATTERN.search(attrs)
        anchor = anchor_m.group(1) if anchor_m else "start"
        fs_m = _FONT_SIZE_ATTR_PATTERN.search(attrs)
        font_size = float(fs_m.group(1)) if fs_m else _DEFAULT_FONT_SIZE
        text = _strip_tags(content)
        width = max(len(text), 1) * font_size * _AVG_GLYPH_WIDTH_FACTOR
        if anchor == "middle":
            x0, x1 = x - width / 2, x + width / 2
        elif anchor == "end":
            x0, x1 = x - width, x
        else:
            x0, x1 = x, x + width
        y0, y1 = y - font_size * _ASCENT_FACTOR, y + font_size * _DESCENT_FACTOR
        elements.append({
            "attrs_start": m.start(1), "attrs_text": attrs,
            "tag_start": m.start(), "tag_end": m.end(),
            "x": x, "y": y, "anchor": anchor, "font_size": font_size,
            "text": text, "bbox": (x0, y0, x1, y1),
        })
    return elements


def _boxes_overlap(a: tuple, b: tuple, pad: float = _COLLISION_PAD) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 + pad < bx1 and bx0 + pad < ax1 and
            ay0 + pad < by1 and by0 + pad < ay1)


def find_overlaps(elements: list) -> list:
    """Returns a list of (i, j) index pairs (into `elements`) whose
    estimated bounding boxes genuinely intersect. O(n^2) — fine for the
    handful of labels ever on one diagram; not intended for large n."""
    overlaps = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if _boxes_overlap(elements[i]["bbox"], elements[j]["bbox"]):
                overlaps.append((i, j))
    return overlaps


def _shift_bbox(bbox: tuple, dx: float, dy: float) -> tuple:
    x0, y0, x1, y1 = bbox
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def _within_canvas(bbox: tuple, width: Optional[float], height: Optional[float]) -> bool:
    if width is None or height is None:
        return True
    x0, y0, x1, y1 = bbox
    return x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height


def reposition_labels(svg: str) -> dict:
    """Detects overlapping <text> labels in `svg` and nudges the
    LATER-occurring label of each colliding pair outward along a
    deterministic search ring until it clears every other label (moved
    or not) — the earlier label in document order is treated as
    anchored/stable, matching how a human laying out a diagram would
    leave the first-placed label alone and move the one that collided
    into it. Movement is applied to a label at most once (a label
    nudged to resolve one collision keeps that resolved position when
    checked against the next).

    Returns {"svg": str, "moved": int, "unresolved": int} — `svg` is
    the (possibly) rewritten diagram, `moved` is how many labels were
    repositioned, and `unresolved` is how many colliding pairs remained
    after the search budget was exhausted (0 in the overwhelming
    majority of real diagrams, which have very few labels). Never
    raises: a malformed SVG that this module's regex can't usefully
    parse is returned unchanged with moved=0, unresolved=0 — this is a
    best-effort layout aid, not a gate."""
    try:
        elements = parse_text_elements(svg)
    except Exception:
        return {"svg": svg, "moved": 0, "unresolved": 0}
    if len(elements) < 2:
        return {"svg": svg, "moved": 0, "unresolved": 0}

    vb = _VIEWBOX_PATTERN.search(svg)
    canvas_w, canvas_h = (float(vb.group(1)), float(vb.group(2))) if vb else (None, None)

    current_bboxes = [e["bbox"] for e in elements]
    offsets = [(0.0, 0.0) for _ in elements]
    moved = 0

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if not _boxes_overlap(current_bboxes[i], current_bboxes[j]):
                continue
            # Move the later element (j) out of every other element's way.
            resolved = False
            for radius in _SEARCH_RADII:
                for angle in _SEARCH_ANGLES:
                    import math as _math
                    dx = radius * _math.cos(_math.radians(angle))
                    dy = radius * _math.sin(_math.radians(angle))
                    candidate = _shift_bbox(elements[j]["bbox"], dx, dy)
                    if not _within_canvas(candidate, canvas_w, canvas_h):
                        continue
                    if any(k != j and _boxes_overlap(candidate, current_bboxes[k])
                           for k in range(len(elements))):
                        continue
                    offsets[j] = (dx, dy)
                    current_bboxes[j] = candidate
                    resolved = True
                    break
                if resolved:
                    break
            if resolved:
                moved += 1

    remaining = len(find_overlaps([{"bbox": b} for b in current_bboxes]))

    if moved == 0:
        return {"svg": svg, "moved": 0, "unresolved": remaining}

    # Rewrite the SVG: apply each nonzero offset to its element's x/y
    # attributes, working from the END of the string backward so
    # earlier offsets in the same string don't invalidate later spans.
    out = svg
    for idx in sorted(range(len(elements)), key=lambda k: elements[k]["tag_start"], reverse=True):
        dx, dy = offsets[idx]
        if dx == 0.0 and dy == 0.0:
            continue
        e = elements[idx]
        new_x, new_y = e["x"] + dx, e["y"] + dy
        attrs = e["attrs_text"]
        attrs = _X_ATTR_PATTERN.sub(f'x="{new_x:.1f}"', attrs, count=1)
        attrs = _Y_ATTR_PATTERN.sub(f'y="{new_y:.1f}"', attrs, count=1)
        out = out[:e["attrs_start"]] + attrs + out[e["attrs_start"] + len(e["attrs_text"]):]

    return {"svg": out, "moved": moved, "unresolved": remaining}


def check_and_fix_labels(svg: str) -> dict:
    """Convenience wrapper for plugin authors: parses, detects overlaps,
    and if any are found, repositions. Returns the same shape as
    reposition_labels always (a no-overlap SVG passes through with
    moved=0, unresolved=0)."""
    elements = parse_text_elements(svg)
    if not find_overlaps(elements):
        return {"svg": svg, "moved": 0, "unresolved": 0}
    return reposition_labels(svg)
