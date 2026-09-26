"""
circle_line_plugin.py — diagram_type "circle_line_intersection"
(GENUINE NEW GAP: the existing built-in "circle" type, per
diagram_renderer.py's _validate_circle_spec, only supports a
tangent-from-an-external-point construction; there was no general
"an arbitrary line meets this circle" construction anywhere in the
codebase before this file — confirmed by reading _render_circle and
_validate_circle_spec directly.)

Given a circle (center + radius) and a line (either two points it
passes through, or a point + direction angle), computes the EXACT
intersection of the line and circle by substituting the line's
parametric form into the circle equation and solving the resulting
quadratic with the quadratic formula — never estimating an
intersection point by eye. The discriminant classifies the case:
  > 0  -> secant  (two intersection points)
  = 0  -> tangent (one intersection point, and the radius to it is
           verified perpendicular to the line as a post-render check)
  < 0  -> no intersection (line drawn, circle drawn, no points marked)

Wired in via diagram_plugin_registry.py; does not touch the existing
built-in "circle" type or its renderer/validator in any way.
"""
import math

import diagram_plugin_registry
import label_layout

LINE_COLOR = "#1a4d8f"
AUX_COLOR = "#c0392b"
LABEL_COLOR = "#111111"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='600'"
SMALL_FONT = "font-family='Noto Sans, Arial' font-size='11'"

W, H = 280, 240
MARGIN = 36

_EPS = 1e-9


def _clip_segment_to_rect(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """Liang-Barsky line-segment clipping against an axis-aligned
    rectangle. Returns the (possibly shortened) segment's endpoints
    clamped to lie within [xmin,xmax] x [ymin,ymax], or None if the
    segment doesn't cross the rectangle at all.

    This exists specifically so a "line extended across the whole
    canvas" construction produces coordinates that are ACTUALLY within
    the SVG's own viewBox, rather than relying on the browser/PDF
    renderer's overflow:hidden default to crop it visually. The two
    are visually identical, but only the former passes diagram_final_
    check.py's own coordinate-bounds scan — and failing that scan
    means the ENTIRE diagram gets silently omitted from the final PDF,
    not just the line. See this module's docstring for the production
    impact this specific gap had before being fixed."""
    dx, dy = x2 - x1, y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < _EPS:
            if qi < 0:
                return None  # parallel and outside on this side
            continue
        t = qi / pi
        if pi < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
    if t0 > t1:
        return None
    return (x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy)


def _is_number(v) -> bool:
    # V37 hardening pass: math.isfinite explicitly excludes NaN/inf.
    # isinstance-only checks let NaN through, and every subsequent
    # "<= 0 is invalid" comparison against NaN is silently False in
    # Python, so NaN previously bypassed validation undetected.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _validate_circle_line_spec(spec: dict):
    issues = []
    center = spec.get("center") or {"x": 0, "y": 0}
    if not (_is_number(center.get("x")) and _is_number(center.get("y"))):
        issues.append("center.x and center.y must both be numbers")
    radius = spec.get("radius")
    if not _is_number(radius) or radius <= 0:
        issues.append(f"radius must be a positive number, got {radius!r}")

    line = spec.get("line")
    if not isinstance(line, dict):
        issues.append("'line' must be an object with either 'through' (two points) "
                       "or 'point' + 'angle_deg'")
    else:
        through = line.get("through")
        has_through = isinstance(through, list) and len(through) == 2 and all(
            isinstance(p, dict) and _is_number(p.get("x")) and _is_number(p.get("y")) for p in through)
        has_point_angle = (isinstance(line.get("point"), dict)
                            and _is_number(line["point"].get("x")) and _is_number(line["point"].get("y"))
                            and _is_number(line.get("angle_deg")))
        if not has_through and not has_point_angle:
            issues.append("'line' must supply either 'through': [pointA, pointB] or "
                           "'point': {x,y} + 'angle_deg'")
        if has_through:
            (ax, ay), (bx, by) = (through[0]["x"], through[0]["y"]), (through[1]["x"], through[1]["y"])
            if math.hypot(bx - ax, by - ay) < _EPS:
                issues.append("'line.through' points must be distinct")

    return (len(issues) == 0), issues


def _line_points(spec: dict) -> tuple:
    """Returns two distinct points (A, B) the line passes through,
    normalized from either input form."""
    line = spec["line"]
    through = line.get("through")
    if through:
        return (through[0]["x"], through[0]["y"]), (through[1]["x"], through[1]["y"])
    p = line["point"]
    ang = math.radians(float(line["angle_deg"]))
    return (p["x"], p["y"]), (p["x"] + math.cos(ang), p["y"] + math.sin(ang))


def solve_circle_line_intersection(spec: dict) -> dict:
    """Pure math, exact quadratic solve. Returns {'case': 'secant'|
    'tangent'|'none', 'points': [(x,y), ...], 'center': (cx,cy),
    'radius': r, 'line_points': (A, B)}. Never estimates: substitutes
    P(t) = A + t*(B-A) into (x-cx)^2 + (y-cy)^2 = r^2 and solves the
    resulting a*t^2 + b*t + c = 0 for t with the quadratic formula."""
    center = spec.get("center") or {"x": 0, "y": 0}
    cx, cy = float(center["x"]), float(center["y"])
    r = float(spec["radius"])
    A, B = _line_points(spec)
    ax, ay = A
    dx, dy = B[0] - A[0], B[1] - A[1]

    fx, fy = ax - cx, ay - cy
    a_coef = dx * dx + dy * dy
    b_coef = 2 * (fx * dx + fy * dy)
    c_coef = fx * fx + fy * fy - r * r

    discriminant = b_coef * b_coef - 4 * a_coef * c_coef
    result = {"center": (cx, cy), "radius": r, "line_points": (A, B)}

    if discriminant < -_EPS:
        result["case"] = "none"
        result["points"] = []
    elif abs(discriminant) <= _EPS:
        t = -b_coef / (2 * a_coef)
        result["case"] = "tangent"
        result["points"] = [(ax + t * dx, ay + t * dy)]
    else:
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b_coef - sqrt_d) / (2 * a_coef)
        t2 = (-b_coef + sqrt_d) / (2 * a_coef)
        result["case"] = "secant"
        result["points"] = [(ax + t1 * dx, ay + t1 * dy), (ax + t2 * dx, ay + t2 * dy)]
    return result


def verify_circle_line_construction(solved: dict) -> tuple:
    """Post-render math validation: every reported intersection point
    must lie EXACTLY on the circle (distance from center == radius).
    For the tangent case, additionally verifies the radius to the
    tangent point is perpendicular to the line direction (dot product
    == 0), which is the defining property of a tangent — not merely
    asserted, actually checked."""
    issues = []
    cx, cy = solved["center"]
    r = solved["radius"]
    for (px, py) in solved["points"]:
        dist = math.hypot(px - cx, py - cy)
        if abs(dist - r) > 1e-6:
            issues.append(f"reported intersection point ({px:.4f},{py:.4f}) is distance {dist:.6f} "
                          f"from center, expected radius {r}")
    if solved["case"] == "tangent" and solved["points"]:
        (px, py) = solved["points"][0]
        A, B = solved["line_points"]
        line_dx, line_dy = B[0] - A[0], B[1] - A[1]
        radius_dx, radius_dy = px - cx, py - cy
        dot = line_dx * radius_dx + line_dy * radius_dy
        norm = math.hypot(line_dx, line_dy) * math.hypot(radius_dx, radius_dy)
        if norm > _EPS and abs(dot / norm) > 1e-6:
            issues.append(f"tangent radius is not perpendicular to the line (cos(angle)={dot/norm:.6f}, "
                          f"expected 0)")
    return (len(issues) == 0), issues


def _render_circle_line(spec: dict) -> str:
    solved = solve_circle_line_intersection(spec)
    ok, issues = verify_circle_line_construction(solved)
    if not ok:
        return ""

    cx_m, cy_m = solved["center"]
    r_m = solved["radius"]
    A, B = solved["line_points"]

    # the point on the line closest to the circle's center — this is
    # the geometrically relevant point for BOTH scaling the view and
    # anchoring the drawn line's extension (see below). Computed once,
    # up front, so both uses stay consistent.
    dx0, dy0 = B[0] - A[0], B[1] - A[1]
    norm0 = math.hypot(dx0, dy0) or 1.0
    ux, uy = dx0 / norm0, dy0 / norm0
    t_foot = (cx_m - A[0]) * ux + (cy_m - A[1]) * uy
    foot_m = (A[0] + ux * t_foot, A[1] + uy * t_foot)

    # figure out a drawing scale that fits BOTH the circle AND the
    # line's closest approach to it — NOT the circle alone. A line
    # whose closest point to the circle is farther away than the
    # circle's own radius (a real "no intersection" case, not just a
    # near-miss) would otherwise never enter the visible canvas at any
    # extension length, since the view was being sized purely from
    # r_m with no regard for where the line actually is.
    half_w_needed = max(r_m, abs(foot_m[0] - cx_m)) * 1.15
    half_h_needed = max(r_m, abs(foot_m[1] - cy_m)) * 1.15
    scale = (W - 2 * MARGIN) / (2 * half_w_needed)
    scale = min(scale, (H - 2 * MARGIN) / (2 * half_h_needed))

    def to_svg(p):
        return (W / 2 + (p[0] - cx_m) * scale, H / 2 - (p[1] - cy_m) * scale)

    center_svg = to_svg((cx_m, cy_m))
    r_px = r_m * scale

    body = []
    body.append(f'<circle cx="{center_svg[0]:.1f}" cy="{center_svg[1]:.1f}" r="{r_px:.1f}" fill="none" '
                f'stroke="{LINE_COLOR}" stroke-width="2" data-role="circle"/>')
    body.append(f'<circle cx="{center_svg[0]:.1f}" cy="{center_svg[1]:.1f}" r="1.8" fill="{LINE_COLOR}" '
                f'data-role="center-point"/>')
    center_label = ((spec.get("center") or {}).get("label") or "O")
    body.append(f'<text x="{center_svg[0]+6:.1f}" y="{center_svg[1]-6:.1f}" fill="{LABEL_COLOR}" {SMALL_FONT} '
                f'data-role="center-label">{_escape(center_label)}</text>')

    # extend the line across the full canvas so it visibly "passes
    # through" rather than stopping at the two input points. Extend
    # from the point on the line CLOSEST TO THE CIRCLE'S CENTER (already
    # computed above, and the scale above was sized to guarantee this
    # point is actually within view) — not from input point A itself,
    # since real questions often state points that define the line's
    # direction but sit far from the circle.
    extend = math.hypot(W, H) / 2.0 / scale
    line_a_m = (foot_m[0] - ux * extend, foot_m[1] - uy * extend)
    line_b_m = (foot_m[0] + ux * extend, foot_m[1] + uy * extend)
    la, lb = to_svg(line_a_m), to_svg(line_b_m)
    clipped = _clip_segment_to_rect(la[0], la[1], lb[0], lb[1], 3, 3, W - 3, H - 3)
    if clipped is not None:
        la, lb = (clipped[0], clipped[1]), (clipped[2], clipped[3])
    body.append(f'<line x1="{la[0]:.1f}" y1="{la[1]:.1f}" x2="{lb[0]:.1f}" y2="{lb[1]:.1f}" '
                f'stroke="{AUX_COLOR}" stroke-width="1.8" data-role="line"/>')

    labels = spec.get("labels") or {}
    prefix = labels.get("intersection_prefix", "P")
    for i, pt_m in enumerate(solved["points"]):
        pt_svg = to_svg(pt_m)
        body.append(f'<circle cx="{pt_svg[0]:.1f}" cy="{pt_svg[1]:.1f}" r="2.5" fill="{AUX_COLOR}" '
                    f'data-role="intersection-point"/>')
        lbl = f"{prefix}{i+1}" if len(solved["points"]) > 1 else prefix
        dy_lbl = -8 if pt_svg[1] < H / 2 else 14
        body.append(f'<text x="{pt_svg[0]:.1f}" y="{(pt_svg[1]+dy_lbl):.1f}" text-anchor="middle" '
                    f'fill="{LABEL_COLOR}" {FONT} data-role="point-label">{_escape(lbl)}</text>')

    case_text = {"secant": "line is a secant (2 points)",
                 "tangent": "line is a tangent (1 point)",
                 "none": "line does not meet the circle"}[solved["case"]]
    body.append(f'<text x="{W/2:.1f}" y="{H-10:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                f'{SMALL_FONT} data-role="case-label">{_escape(case_text)}</text>')

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="max-width:260px">' + "".join(body) + "</svg>")
    fixed = label_layout.check_and_fix_labels(svg)
    return fixed["svg"]


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="circle_line_intersection",
    renderer=_render_circle_line,
    schema_check=_validate_circle_line_spec,
    description="A circle (center+radius) and an arbitrary line (two points, or point+angle), with "
                "intersection point(s) computed by substituting the line's parametric form into the "
                "circle equation and solving the exact quadratic (never estimated). Classifies secant "
                "(2 points) / tangent (1 point, radius perpendicularity independently verified) / no "
                "intersection, and independently re-checks every reported point lies exactly on the "
                "circle before returning any SVG.",
))
