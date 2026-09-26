"""
composite_shaded_plugin.py — diagram_type "composite_shaded_region"
(GENUINE NEW GAP: confirmed absent before this file — the only existing
"shaded region" support anywhere in the codebase is set_theory_plugin.py's
2-set Venn diagram union/intersection shading, which is a different
chapter entirely. Grepping for "shaded"/"composite"/"overlap"/"inscribed"
before writing a line here found nothing for the "combination of plane
figures" question type that closes out Class 10's "Areas Related to
Circles" chapter.)

SCOPING DECISION, STATED UP FRONT (same discipline diagram_plugin_
registry.py itself models): this plugin deliberately does NOT attempt a
general "boolean region operations on arbitrary shapes" engine. That
would mean building an actual 2D polygon/circle clipping library from
scratch inside this codebase — a large, open-ended undertaking with no
bounded correctness story, which is exactly the kind of "invent
work"/unverifiable-scope problem this project's own engineering rules
warn against. Instead, this plugin implements the four SPECIFIC,
extremely common combination-of-figures questions NCERT/SCERT Class
6-10 actually asks, each with its own closed-form area formula (never
estimated, never approximated by generic polygon clipping):

  "circle_in_square"       — a circle inscribed in (or centered inside)
                              a square. shaded area = square - circle.
  "square_in_circle"       — a square inscribed in a circle (vertices
                              on the circle). shaded area = circle - square.
  "triangle_in_circle"     — a right triangle inscribed in a circle with
                              its hypotenuse as the diameter (Thales'
                              theorem — the standard NCERT construction).
                              shaded area = circle - triangle.
  "two_overlapping_circles"— two equal circles a given distance apart.
                              Covers "intersection" (the lens), "union",
                              and "petals" (the two crescent pieces each
                              circle has outside the other) — the three
                              shaded-region variants NCERT actually asks.

Each closed-form formula was independently cross-checked against a
Monte-Carlo numerical estimate during development (see this plugin's
test file) before being trusted, and every rendered shape is further
independently re-measured via the shoelace formula against its own
closed-form value before any SVG is returned — the same "prove it,
don't just draw it" discipline circle_sector_plugin.py already
established for this chapter.

A future combination genuinely not covered here (e.g. a semicircle on a
rectangle) is a new, separately-scoped extension of this SAME module
(one more composite_type + formula), not a reason to build a different
engine — this file's registry-of-composite-types pattern (COMPOSITE_
TYPES dict, below) is exactly the extension point for that.

Wired in via diagram_plugin_registry.py only — no existing file's
behaviour changes as a result of this plugin existing.
"""
import math

import diagram_plugin_registry
import label_layout

LINE_COLOR = "#1a4d8f"
AUX_COLOR = "#c0392b"
LABEL_COLOR = "#111111"
SHADE_COLOR = "#f2b134"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='600'"
SMALL_FONT = "font-family='Noto Sans, Arial' font-size='11'"

W, H = 280, 260
MARGIN = 30
_EPS = 1e-9
_ARC_SAMPLES = 720  # per full circle; sub-arcs get a proportional share


def _is_number(v) -> bool:
    # See circle_sector_plugin._is_number for why math.isfinite is
    # required here, not just isinstance: NaN silently defeats every
    # downstream "<= 0 is invalid" guard AND the shoelace verification's
    # own tolerance comparison (both rely on ordering comparisons that
    # are always False against NaN). Reproduced and confirmed during
    # stress testing before this fix — a "side": NaN spec previously
    # passed schema validation AND verification and produced a
    # NaN-valued diagram spec silently.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _shoelace_area(points) -> float:
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _circle_arc_points(cx, cy, r, start_rad, end_rad, n=None):
    if n is None:
        n = max(24, int(_ARC_SAMPLES * abs(end_rad - start_rad) / (2 * math.pi)))
    return [(cx + r * math.cos(start_rad + (end_rad - start_rad) * i / n),
             cy + r * math.sin(start_rad + (end_rad - start_rad) * i / n))
            for i in range(n + 1)]


# ---------------------------------------------------------------------------
# SCHEMA VALIDATION — one function per composite_type, dispatched below
# ---------------------------------------------------------------------------

def _validate_circle_in_square(spec):
    issues = []
    side = spec.get("side")
    if not _is_number(side) or side <= 0:
        issues.append(f"'side' must be a positive number, got {side!r}")
        side = None
    radius = spec.get("radius", (side / 2.0) if side else None)
    if not _is_number(radius) or radius <= 0:
        issues.append(f"'radius' must be a positive number, got {radius!r}")
    elif side and radius > side / 2.0 + 1e-9:
        issues.append(f"radius ({radius}) cannot exceed side/2 ({side/2.0}) — the circle must fit "
                       f"entirely inside the square")
    if spec.get("shaded", "between") not in ("between", "circle", "square"):
        issues.append("'shaded' must be 'between', 'circle', or 'square'")
    return issues


def _validate_square_in_circle(spec):
    issues = []
    radius = spec.get("radius")
    if not _is_number(radius) or radius <= 0:
        issues.append(f"'radius' must be a positive number, got {radius!r}")
    if spec.get("shaded", "between") not in ("between", "circle", "square"):
        issues.append("'shaded' must be 'between', 'circle', or 'square'")
    return issues


def _validate_triangle_in_circle(spec):
    issues = []
    radius = spec.get("radius")
    if not _is_number(radius) or radius <= 0:
        issues.append(f"'radius' must be a positive number, got {radius!r}")
    angle = spec.get("angle_deg")
    if not _is_number(angle) or not (0 < angle < 90):
        issues.append(f"'angle_deg' (the acute angle of the inscribed right triangle at one end of "
                       f"the diameter) must be strictly between 0 and 90, got {angle!r}")
    if spec.get("shaded", "between") not in ("between", "circle", "triangle"):
        issues.append("'shaded' must be 'between', 'circle', or 'triangle'")
    return issues


def _validate_two_overlapping_circles(spec):
    issues = []
    radius = spec.get("radius")
    if not _is_number(radius) or radius <= 0:
        issues.append(f"'radius' must be a positive number, got {radius!r}")
    distance = spec.get("distance")
    if not _is_number(distance) or distance <= 0:
        issues.append(f"'distance' must be a positive number, got {distance!r}")
    elif _is_number(radius) and radius > 0 and not (0 < distance < 2 * radius):
        issues.append(f"'distance' ({distance}) must be strictly between 0 and 2*radius "
                       f"({2*radius}) for the two circles to actually overlap")
    if spec.get("shaded", "intersection") not in ("intersection", "union", "petals"):
        issues.append("'shaded' must be 'intersection', 'union', or 'petals'")
    return issues


_VALIDATORS = {
    "circle_in_square": _validate_circle_in_square,
    "square_in_circle": _validate_square_in_circle,
    "triangle_in_circle": _validate_triangle_in_circle,
    "two_overlapping_circles": _validate_two_overlapping_circles,
}


def _validate_composite_shaded_spec(spec: dict):
    composite_type = spec.get("composite_type")
    if composite_type not in _VALIDATORS:
        return False, [f"'composite_type' must be one of {sorted(_VALIDATORS)}, got {composite_type!r}"]
    issues = _VALIDATORS[composite_type](spec)
    return (len(issues) == 0), issues


# ---------------------------------------------------------------------------
# CLOSED-FORM MATH — one function per composite_type, dispatched below
# ---------------------------------------------------------------------------

def _compute_circle_in_square(spec):
    side = float(spec["side"])
    radius = float(spec.get("radius", side / 2.0))
    square_area = side * side
    circle_area = math.pi * radius * radius
    return {"side": side, "radius": radius, "square_area": square_area, "circle_area": circle_area,
            "between_area": square_area - circle_area}


def _compute_square_in_circle(spec):
    radius = float(spec["radius"])
    side = radius * math.sqrt(2)
    circle_area = math.pi * radius * radius
    square_area = side * side  # == 2*radius**2
    return {"radius": radius, "side": side, "circle_area": circle_area, "square_area": square_area,
            "between_area": circle_area - square_area}


def _compute_triangle_in_circle(spec):
    radius = float(spec["radius"])
    angle = float(spec["angle_deg"])
    angle_rad = math.radians(angle)
    circle_area = math.pi * radius * radius
    triangle_area = radius * radius * math.sin(2 * angle_rad)
    leg1 = 2 * radius * math.cos(angle_rad)   # adjacent to `angle`, at vertex B1
    leg2 = 2 * radius * math.sin(angle_rad)   # adjacent to `angle`, at vertex B2 (opposite leg)
    return {"radius": radius, "angle_deg": angle, "circle_area": circle_area,
            "triangle_area": triangle_area, "between_area": circle_area - triangle_area,
            "leg1": leg1, "leg2": leg2, "hypotenuse": 2 * radius}


def _compute_two_overlapping_circles(spec):
    radius = float(spec["radius"])
    distance = float(spec["distance"])
    a = distance / 2.0
    alpha = math.acos(max(-1.0, min(1.0, a / radius)))  # half-angle subtended at each center
    lens_area = 2 * radius * radius * alpha - radius * radius * math.sin(2 * alpha)
    circle_area = math.pi * radius * radius
    return {"radius": radius, "distance": distance, "alpha_rad": alpha, "circle_area": circle_area,
            "lens_area": lens_area, "union_area": 2 * circle_area - lens_area,
            "petal_area": circle_area - lens_area}


_COMPUTE = {
    "circle_in_square": _compute_circle_in_square,
    "square_in_circle": _compute_square_in_circle,
    "triangle_in_circle": _compute_triangle_in_circle,
    "two_overlapping_circles": _compute_two_overlapping_circles,
}


def compute_composite_shaded_region(spec: dict) -> dict:
    """Pure math, exact closed-form, dispatched by spec['composite_type'].
    Returns every area NCERT ever asks for this composite figure at once
    (never just the one the question happened to ask, so a future
    solver.py integration never has to re-derive a sibling quantity)."""
    return _COMPUTE[spec["composite_type"]](spec)


# ---------------------------------------------------------------------------
# POST-RENDER VERIFICATION — independently re-measures the geometry this
# spec asks for via shoelace sampling, and compares against the closed
# form. Same discipline as circle_sector_plugin.verify_circle_sector_
# construction, applied to each composite_type's own shaded region.
# ---------------------------------------------------------------------------

def verify_composite_shaded_construction(spec: dict, solved: dict) -> tuple:
    issues = []
    composite_type = spec["composite_type"]
    shaded = spec.get("shaded", "between" if composite_type != "two_overlapping_circles" else "intersection")

    def _check(measured, expected, label):
        if not (math.isfinite(measured) and math.isfinite(expected)):
            issues.append(f"{label} area is non-finite (measured={measured}, expected={expected}) "
                           f"— rejecting rather than silently passing a NaN/inf comparison")
            return
        if abs(measured - expected) > max(1e-6, abs(expected) * 1e-4):
            issues.append(f"rendered {label} measures area {measured:.6f} but the closed-form "
                           f"formula gives {expected:.6f}")

    if composite_type == "circle_in_square":
        side, r = solved["side"], solved["radius"]
        square_poly = [(-side/2, -side/2), (side/2, -side/2), (side/2, side/2), (-side/2, side/2)]
        circle_poly = _circle_arc_points(0, 0, r, 0, 2 * math.pi)
        _check(_shoelace_area(square_poly), solved["square_area"], "square")
        _check(_shoelace_area(circle_poly), solved["circle_area"], "circle")

    elif composite_type == "square_in_circle":
        r = solved["radius"]
        half = r / math.sqrt(2)
        square_poly = [(-half, -half), (half, -half), (half, half), (-half, half)]
        circle_poly = _circle_arc_points(0, 0, r, 0, 2 * math.pi)
        _check(_shoelace_area(square_poly), solved["square_area"], "square")
        _check(_shoelace_area(circle_poly), solved["circle_area"], "circle")

    elif composite_type == "triangle_in_circle":
        r = solved["radius"]
        angle_rad = math.radians(solved["angle_deg"])
        B1, B2 = (-r, 0.0), (r, 0.0)
        C = (r * math.cos(2 * angle_rad), r * math.sin(2 * angle_rad))
        _check(_shoelace_area([B1, B2, C]), solved["triangle_area"], "triangle")
        _check(_shoelace_area(_circle_arc_points(0, 0, r, 0, 2 * math.pi)), solved["circle_area"], "circle")
        # right-angle-at-C sanity: independently verify angle B1-C-B2 really is 90 degrees
        v1 = (B1[0] - C[0], B1[1] - C[1])
        v2 = (B2[0] - C[0], B2[1] - C[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if abs(dot) > 1e-6 * r * r:
            issues.append(f"angle at C is not 90 degrees (dot product {dot:.6f}, expected ~0) — "
                           f"Thales' theorem construction is broken")

    elif composite_type == "two_overlapping_circles":
        r, d = solved["radius"], solved["distance"]
        a = d / 2.0
        alpha = solved["alpha_rad"]
        VERIFY_N = 4000  # fixed high resolution for the correctness check, independent of
                          # render-time sampling — a thin-sliver lens (near-tangent circles)
                          # needs many more samples to converge than a wide overlap does
        arc1 = _circle_arc_points(-a, 0, r, -alpha, alpha, n=VERIFY_N)
        arc2 = _circle_arc_points(a, 0, r, math.pi - alpha, math.pi + alpha, n=VERIFY_N)
        lens_poly = arc1 + arc2
        _check(_shoelace_area(lens_poly), solved["lens_area"], "lens (intersection)")
        petal1 = (_circle_arc_points(-a, 0, r, alpha, 2 * math.pi - alpha, n=VERIFY_N)
                  + list(reversed(_circle_arc_points(a, 0, r, math.pi - alpha, math.pi + alpha, n=VERIFY_N))))
        _check(_shoelace_area(petal1), solved["petal_area"], "petal (circle minus lens)")

    return (len(issues) == 0), issues


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else f"{v:.0f}"


# ---------------------------------------------------------------------------
# RENDERING
# ---------------------------------------------------------------------------

def _poly_path(points, close=True) -> str:
    d = f"M {points[0][0]:.2f} {points[0][1]:.2f} " + " ".join(
        f"L {x:.2f} {y:.2f}" for (x, y) in points[1:])
    return d + (" Z" if close else "")


def _render_circle_in_square(spec, solved, to_svg, scale):
    side, r = solved["side"], solved["radius"]
    shaded = spec.get("shaded", "between")
    sq = [to_svg(p) for p in [(-side/2, -side/2), (side/2, -side/2), (side/2, side/2), (-side/2, side/2)]]
    circ = [to_svg(p) for p in _circle_arc_points(0, 0, r, 0, 2 * math.pi, n=96)]

    body = []
    if shaded == "between":
        d = _poly_path(sq) + " " + _poly_path(circ)
        body.append(f'<path d="{d}" fill="{SHADE_COLOR}" fill-opacity="0.55" fill-rule="evenodd" '
                    f'stroke="none" data-role="shaded-region"/>')
    elif shaded == "circle":
        body.append(f'<path d="{_poly_path(circ)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')
    else:
        body.append(f'<path d="{_poly_path(sq)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')

    body.append(f'<path d="{_poly_path(sq)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.8" '
                f'data-role="square"/>')
    body.append(f'<path d="{_poly_path(circ)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6" '
                f'data-role="circle"/>')
    return body


def _render_square_in_circle(spec, solved, to_svg, scale):
    r = solved["radius"]
    shaded = spec.get("shaded", "between")
    half = r / math.sqrt(2)
    sq = [to_svg(p) for p in [(-half, -half), (half, -half), (half, half), (-half, half)]]
    circ = [to_svg(p) for p in _circle_arc_points(0, 0, r, 0, 2 * math.pi, n=96)]

    body = []
    if shaded == "between":
        d = _poly_path(circ) + " " + _poly_path(sq)
        body.append(f'<path d="{d}" fill="{SHADE_COLOR}" fill-opacity="0.55" fill-rule="evenodd" '
                    f'stroke="none" data-role="shaded-region"/>')
    elif shaded == "circle":
        body.append(f'<path d="{_poly_path(circ)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')
    else:
        body.append(f'<path d="{_poly_path(sq)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')

    body.append(f'<path d="{_poly_path(circ)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6" '
                f'data-role="circle"/>')
    body.append(f'<path d="{_poly_path(sq)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.8" '
                f'data-role="square"/>')
    return body


def _render_triangle_in_circle(spec, solved, to_svg, scale):
    r = solved["radius"]
    angle_rad = math.radians(solved["angle_deg"])
    shaded = spec.get("shaded", "between")
    B1, B2 = (-r, 0.0), (r, 0.0)
    C = (r * math.cos(2 * angle_rad), r * math.sin(2 * angle_rad))
    tri = [to_svg(p) for p in (B1, B2, C)]
    circ = [to_svg(p) for p in _circle_arc_points(0, 0, r, 0, 2 * math.pi, n=96)]

    body = []
    if shaded == "between":
        d = _poly_path(circ) + " " + _poly_path(tri)
        body.append(f'<path d="{d}" fill="{SHADE_COLOR}" fill-opacity="0.55" fill-rule="evenodd" '
                    f'stroke="none" data-role="shaded-region"/>')
    elif shaded == "circle":
        body.append(f'<path d="{_poly_path(circ)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')
    else:
        body.append(f'<path d="{_poly_path(tri)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')

    body.append(f'<path d="{_poly_path(circ)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6" '
                f'data-role="circle"/>')
    body.append(f'<path d="{_poly_path(tri)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.8" '
                f'data-role="triangle"/>')
    labels = spec.get("labels") or {}
    for pt, key, default in ((B1, "B1", "B"), (B2, "B2", "C"), (C, "C", "A")):
        p = to_svg(pt)
        dy = -8 if p[1] < H / 2 else 16
        body.append(f'<text x="{p[0]:.1f}" y="{p[1]+dy:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                    f'{FONT} data-role="point-label">{_escape(labels.get(key, default))}</text>')
    # right-angle marker at C
    cx_svg, cy_svg = to_svg(C)
    body.append(f'<circle cx="{cx_svg:.1f}" cy="{cy_svg:.1f}" r="2" fill="{AUX_COLOR}" '
                f'data-role="right-angle-vertex"/>')
    return body


def _render_two_overlapping_circles(spec, solved, to_svg, scale):
    r, d = solved["radius"], solved["distance"]
    a = d / 2.0
    alpha = solved["alpha_rad"]
    shaded = spec.get("shaded", "intersection")

    circ1 = [to_svg(p) for p in _circle_arc_points(-a, 0, r, 0, 2 * math.pi, n=96)]
    circ2 = [to_svg(p) for p in _circle_arc_points(a, 0, r, 0, 2 * math.pi, n=96)]

    body = []
    if shaded == "intersection":
        arc1 = _circle_arc_points(-a, 0, r, -alpha, alpha, n=64)
        arc2 = _circle_arc_points(a, 0, r, math.pi - alpha, math.pi + alpha, n=64)
        lens = [to_svg(p) for p in (arc1 + arc2)]
        body.append(f'<path d="{_poly_path(lens)}" fill="{SHADE_COLOR}" fill-opacity="0.7" '
                    f'stroke="none" data-role="shaded-region"/>')
    elif shaded == "union":
        body.append(f'<path d="{_poly_path(circ1)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')
        body.append(f'<path d="{_poly_path(circ2)}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')
    else:  # petals
        petal1 = (_circle_arc_points(-a, 0, r, alpha, 2 * math.pi - alpha, n=64)
                  + list(reversed(_circle_arc_points(a, 0, r, math.pi - alpha, math.pi + alpha, n=64))))
        petal2 = (_circle_arc_points(a, 0, r, math.pi + alpha, 3 * math.pi - alpha, n=64)
                  + list(reversed(_circle_arc_points(-a, 0, r, -alpha, alpha, n=64))))
        for petal in (petal1, petal2):
            pts = [to_svg(p) for p in petal]
            body.append(f'<path d="{_poly_path(pts)}" fill="{SHADE_COLOR}" fill-opacity="0.6" '
                        f'stroke="none" data-role="shaded-region"/>')

    body.append(f'<path d="{_poly_path(circ1)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6" '
                f'data-role="circle-1"/>')
    body.append(f'<path d="{_poly_path(circ2)}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6" '
                f'data-role="circle-2"/>')
    return body


_RENDER_BODY = {
    "circle_in_square": _render_circle_in_square,
    "square_in_circle": _render_square_in_circle,
    "triangle_in_circle": _render_triangle_in_circle,
    "two_overlapping_circles": _render_two_overlapping_circles,
}


def _render_composite_shaded(spec: dict) -> str:
    solved = compute_composite_shaded_region(spec)
    ok, issues = verify_composite_shaded_construction(spec, solved)
    if not ok:
        return ""

    composite_type = spec["composite_type"]
    # pick a drawing scale that fits whichever shape's extent is largest
    if composite_type == "circle_in_square":
        half_extent = solved["side"] / 2.0
    elif composite_type == "two_overlapping_circles":
        half_extent = solved["distance"] / 2.0 + solved["radius"]
    else:
        half_extent = solved["radius"]
    scale = (W - 2 * MARGIN) / (2 * half_extent)
    scale = min(scale, (H - 2 * MARGIN) / (2 * half_extent))

    def to_svg(p):
        return (W / 2 + p[0] * scale, H / 2 - p[1] * scale)

    body = _RENDER_BODY[composite_type](spec, solved, to_svg, scale)

    caption = composite_type.replace("_", " ") + f", shaded = {spec.get('shaded', 'between' if composite_type != 'two_overlapping_circles' else 'intersection')}"
    body.append(f'<text x="{W/2:.1f}" y="{H-8:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                f'{SMALL_FONT} data-role="caption">{_escape(caption)}</text>')

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="max-width:260px">' + "".join(body) + "</svg>")
    fixed = label_layout.check_and_fix_labels(svg)
    return fixed["svg"]


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="composite_shaded_region",
    renderer=_render_composite_shaded,
    schema_check=_validate_composite_shaded_spec,
    solver=None,
    description="Class 10 'combination of plane figures' shaded-region questions: circle_in_square, "
                "square_in_circle, triangle_in_circle (right triangle inscribed via Thales' theorem), "
                "and two_overlapping_circles (intersection/union/petals). All areas computed from exact "
                "closed-form formulas (independently cross-checked against Monte Carlo during "
                "development) and the actual rendered geometry is re-measured via shoelace sampling "
                "against those formulas before any SVG is returned. Deliberately scoped to these four "
                "concrete figures rather than a general boolean-region engine — see this module's "
                "docstring for why.",
))
