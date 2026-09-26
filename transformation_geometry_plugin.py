"""
transformation_geometry_plugin.py — diagram_type #14: "transformation".

Covers school-level transformation geometry: reflection, rotation, and
translation of a polygon on the Cartesian plane.

Same split as every other diagram type in this project (see
diagram_renderer.py's module docstring, and diagram_plugin_registry.py's
own docstring for the plugin extension point): Gemini supplies the
ORIGINAL shape's coordinates and the transformation's parameters
(reflection axis, rotation center/angle, translation vector) — it never
supplies the IMAGE's coordinates itself. This renderer computes every
image point deterministically from the original points and the stated
transformation, the same way geometry_solver.py computes triangle
vertices from side/angle constraints rather than trusting Gemini's own
arithmetic. This is a real correctness win, not just a style choice: a
model asked to reflect (3, -2) over the x-axis can and does
occasionally get the sign wrong; a fixed formula applied in Python
never does.

Wired in via diagram_plugin_registry.py, exactly like
polynomial_division_plugin.py — none of the 12 built-in diagram types
are touched by this module at all.
"""
import math

import diagram_plugin_registry

LINE_COLOR = "#1a4d8f"
IMAGE_COLOR = "#b5442e"
AXIS_COLOR = "#666666"
GRID_COLOR = "#dddddd"
LABEL_COLOR = "#111111"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='14' font-weight='600'"
MARGIN = 30
PX_PER_UNIT = 24
MAX_VERTICES = 8

_VALID_TYPES = {"reflection", "rotation", "translation"}
_VALID_AXES = {"x-axis", "y-axis", "y=x", "y=-x", "line"}


def _is_number(v) -> bool:
    # V37 hardening pass: math.isfinite explicitly excludes NaN/inf.
    # isinstance-only checks let NaN through, and every subsequent
    # "<= 0 is invalid" comparison against NaN is silently False in
    # Python, so NaN previously bypassed validation undetected.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate_transformation_spec(spec: dict):
    """Structural validation for diagram_type == 'transformation'. Checks
    SHAPE only (every field the renderer needs is present and numeric) —
    like every other plugin's schema_check, it cannot judge whether the
    transformation is the mathematically "interesting" one for the
    question, only that the renderer has everything it needs to compute
    the image correctly."""
    issues = []

    shape = spec.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        issues.append("'shape' must be a list of at least 2 points")
        shape = []
    elif len(shape) > MAX_VERTICES:
        issues.append(f"'shape' has {len(shape)} points — more than {MAX_VERTICES} suggests "
                       f"a malformed spec rather than a real school-level polygon")

    for i, p in enumerate(shape):
        if not isinstance(p, dict) or not p.get("id") or not _is_number(p.get("x")) or not _is_number(p.get("y")):
            issues.append(f"shape[{i}] must be an object with a non-empty 'id' and numeric 'x'/'y'")

    transform = spec.get("transformation")
    if not isinstance(transform, dict):
        issues.append("'transformation' must be an object")
        return (not issues, issues)

    ttype = transform.get("type")
    if ttype not in _VALID_TYPES:
        issues.append(f"transformation.type must be one of {sorted(_VALID_TYPES)}, got {ttype!r}")
        return (not issues, issues)

    if ttype == "reflection":
        axis = transform.get("axis")
        if axis not in _VALID_AXES:
            issues.append(f"transformation.axis must be one of {sorted(_VALID_AXES)}, got {axis!r}")
        elif axis == "line":
            through = transform.get("through")
            if not isinstance(through, list) or len(through) != 2 \
                    or not all(isinstance(pt, list) and len(pt) == 2 and _is_number(pt[0]) and _is_number(pt[1])
                               for pt in through):
                issues.append("transformation.through must be two [x, y] points when axis is 'line'")
    elif ttype == "rotation":
        center = transform.get("center")
        if not isinstance(center, list) or len(center) != 2 or not all(_is_number(c) for c in center):
            issues.append("transformation.center must be a numeric [x, y] pair")
        if not _is_number(transform.get("angle_deg")):
            issues.append("transformation.angle_deg must be numeric")
        if transform.get("direction") not in ("clockwise", "counterclockwise"):
            issues.append("transformation.direction must be 'clockwise' or 'counterclockwise'")
    elif ttype == "translation":
        vector = transform.get("vector")
        if not isinstance(vector, list) or len(vector) != 2 or not all(_is_number(v) for v in vector):
            issues.append("transformation.vector must be a numeric [dx, dy] pair")

    return (not issues, issues)


def _reflect(x: float, y: float, transform: dict) -> tuple:
    axis = transform["axis"]
    if axis == "x-axis":
        return x, -y
    if axis == "y-axis":
        return -x, y
    if axis == "y=x":
        return y, x
    if axis == "y=-x":
        return -y, -x
    # axis == "line": reflect (x, y) across the line through two given points
    (x1, y1), (x2, y2) = transform["through"]
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return x, y
    a = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    proj_x, proj_y = x1 + a * dx, y1 + a * dy
    return 2 * proj_x - x, 2 * proj_y - y


def _rotate(x: float, y: float, transform: dict) -> tuple:
    cx, cy = transform["center"]
    angle = math.radians(transform["angle_deg"])
    if transform["direction"] == "clockwise":
        angle = -angle
    dx, dy = x - cx, y - cy
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


def _translate(x: float, y: float, transform: dict) -> tuple:
    dx, dy = transform["vector"]
    return x + dx, y + dy


def _apply_transform(x: float, y: float, transform: dict) -> tuple:
    ttype = transform["type"]
    if ttype == "reflection":
        return _reflect(x, y, transform)
    if ttype == "rotation":
        return _rotate(x, y, transform)
    return _translate(x, y, transform)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_transformation(spec: dict) -> str:
    shape = spec["shape"]
    transform = spec["transformation"]

    originals = [(p["id"], float(p["x"]), float(p["y"])) for p in shape]
    images = [(pid, *_apply_transform(x, y, transform)) for pid, x, y in originals]

    all_x = [x for _, x, y in originals] + [x for _, x, y in images]
    all_y = [y for _, x, y in originals] + [y for _, x, y in images]
    if transform["type"] == "rotation":
        all_x.append(transform["center"][0])
        all_y.append(transform["center"][1])
    lo_x, hi_x = math.floor(min(all_x) - 1), math.ceil(max(all_x) + 1)
    lo_y, hi_y = math.floor(min(all_y) - 1), math.ceil(max(all_y) + 1)

    canvas_w = (hi_x - lo_x) * PX_PER_UNIT + 2 * MARGIN
    canvas_h = (hi_y - lo_y) * PX_PER_UNIT + 2 * MARGIN

    def to_px(x: float, y: float) -> tuple:
        # math-space (y-up) -> SVG-space (y-down)
        return (MARGIN + (x - lo_x) * PX_PER_UNIT, canvas_h - MARGIN - (y - lo_y) * PX_PER_UNIT)

    body = []

    # Grid (light) at every integer unit
    for gx in range(lo_x, hi_x + 1):
        px, _ = to_px(gx, 0)
        body.append(f'<line x1="{px:.1f}" y1="{MARGIN:.1f}" x2="{px:.1f}" y2="{canvas_h - MARGIN:.1f}" '
                    f'stroke="{GRID_COLOR}" stroke-width="1"/>')
    for gy in range(lo_y, hi_y + 1):
        _, py = to_px(0, gy)
        body.append(f'<line x1="{MARGIN:.1f}" y1="{py:.1f}" x2="{canvas_w - MARGIN:.1f}" y2="{py:.1f}" '
                    f'stroke="{GRID_COLOR}" stroke-width="1"/>')

    # Axes (darker), only drawn where they actually cross this canvas
    if lo_x <= 0 <= hi_x:
        px, _ = to_px(0, 0)
        body.append(f'<line x1="{px:.1f}" y1="{MARGIN:.1f}" x2="{px:.1f}" y2="{canvas_h - MARGIN:.1f}" '
                    f'stroke="{AXIS_COLOR}" stroke-width="1.6"/>')
    if lo_y <= 0 <= hi_y:
        _, py = to_px(0, 0)
        body.append(f'<line x1="{MARGIN:.1f}" y1="{py:.1f}" x2="{canvas_w - MARGIN:.1f}" y2="{py:.1f}" '
                    f'stroke="{AXIS_COLOR}" stroke-width="1.6"/>')

    def draw_polygon(points_xy, color, dashed):
        pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in points_xy)
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        return (f'<polygon points="{pts_str}" fill="{color}" fill-opacity="0.08" '
                f'stroke="{color}" stroke-width="2"{dash}/>')

    orig_px = [to_px(x, y) for _, x, y in originals]
    img_px = [to_px(x, y) for _, x, y in images]

    body.append(draw_polygon(orig_px, LINE_COLOR, dashed=False))
    body.append(draw_polygon(img_px, IMAGE_COLOR, dashed=True))

    # Correspondence lines (thin dotted) between each original vertex and its image
    for (px1, py1), (px2, py2) in zip(orig_px, img_px):
        body.append(f'<line x1="{px1:.1f}" y1="{py1:.1f}" x2="{px2:.1f}" y2="{py2:.1f}" '
                    f'stroke="#999999" stroke-width="1" stroke-dasharray="2,3"/>')

    # Point dots + labels
    for (pid, x, y), (px, py) in zip(originals, orig_px):
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{LINE_COLOR}"/>')
        body.append(f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" fill="{LABEL_COLOR}" {FONT}>{_escape(pid)}</text>')
    for (pid, x, y), (px, py) in zip(images, img_px):
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{IMAGE_COLOR}"/>')
        body.append(f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" fill="{IMAGE_COLOR}" {FONT}>{_escape(pid)}\u2032</text>')

    # Rotation center marker, if applicable
    if transform["type"] == "rotation":
        cx, cy = transform["center"]
        px, py = to_px(cx, cy)
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{AXIS_COLOR}"/>')
        body.append(f'<text x="{px + 6:.1f}" y="{py - 6:.1f}" fill="{AXIS_COLOR}" {FONT}>O</text>')

    return (f'<svg viewBox="0 0 {canvas_w:.1f} {canvas_h:.1f}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:280px">' + "".join(body) + "</svg>")


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="transformation",
    renderer=_render_transformation,
    schema_check=_validate_transformation_spec,
    description="Reflection/rotation/translation of a polygon on the Cartesian plane; "
                "image coordinates are always computed by the renderer, never trusted from the spec.",
))
