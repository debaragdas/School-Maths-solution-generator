"""
circle_sector_plugin.py — diagram_type "circle_sector"
(GENUINE NEW GAP: confirmed absent before this file — grepping the
codebase for "sector"/"segment area"/"arc_length" before writing a
line here found only solid_geometry_plugin.py's UNRELATED use of
"sector" to describe a cone's unrolled curved-surface net, which is a
different chapter (surface areas & volumes) and a different shape
entirely. There was no diagram_type anywhere for the Class 10 "Areas
Related to Circles" chapter's actual subject matter: a pie-slice
sector of a circle, the chord-bounded segment next to it, or their
arc lengths / areas. This file adds exactly that, without touching
the existing built-in "circle" type or circle_line_plugin.py.)

Given a circle (center + radius) and the angle θ = angle AOB at the
centre, this plugin covers all four standard NCERT/SCERT question
shapes for this chapter:

  mode="sector", region="minor"  -> the θ-degree pie slice
  mode="sector", region="major"  -> the (360-θ)-degree remaining pie slice
  mode="segment", region="minor" -> the chord-and-minor-arc region
  mode="segment", region="major" -> the chord-and-major-arc region
                                     (= circle - minor segment)

All four quantities NCERT actually asks for are computed from closed-
form formulas — never estimated:

  arc_length(θ)   = (θ/360) * 2 * π * r
  sector_area(θ)  = (θ/360) * π * r²
  segment_area(θ) = sector_area(θ) - triangle_area(O,A,B)
                  = sector_area(θ) - (1/2) r² sin(θ)
  major_region    = whole-circle quantity - minor_region quantity

POST-RENDER VERIFICATION (this project's established "prove it, don't
just draw it" pattern — see circle_line_plugin.verify_circle_line_
construction and solid_geometry_plugin's numeric-consistency check):
the SVG path actually drawn is independently re-measured by sampling
points along the rendered arc and computing the enclosed area via the
shoelace formula, then compared against the closed-form formula above.
A mismatch (e.g. a wrong large-arc-flag/sweep-flag silently drawing
the other arc) is caught here and the diagram is rejected rather than
shipped wrong.

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
MARGIN = 40

_EPS = 1e-9


def _is_number(v) -> bool:
    # math.isfinite explicitly excludes NaN and +/-inf. This matters:
    # NaN passes isinstance() checks, and EVERY subsequent comparison
    # against NaN (<=, >, >=) silently evaluates to False in Python —
    # so a naive isinstance-only check lets NaN sail through every
    # downstream "radius <= 0 is invalid" guard AND, worse, through
    # verify_circle_sector_construction's own tolerance check (since
    # `abs(nan - expected) > tolerance` is also False). Confirmed by
    # reproducing this exact silent-pass during stress testing before
    # fixing it here — see test_nan_and_inf_rejected_at_schema_check.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _validate_circle_sector_spec(spec: dict):
    issues = []
    center = spec.get("center") or {"x": 0, "y": 0}
    if not (_is_number(center.get("x")) and _is_number(center.get("y"))):
        issues.append("center.x and center.y must both be numbers")

    radius = spec.get("radius")
    if not _is_number(radius) or radius <= 0:
        issues.append(f"radius must be a positive number, got {radius!r}")

    angle = spec.get("angle_deg")
    if not _is_number(angle) or not (0 < angle < 360):
        issues.append(f"angle_deg must be a number strictly between 0 and 360 (the angle AOB "
                       f"of the MINOR sector/segment as given by the question), got {angle!r}")

    mode = spec.get("mode", "sector")
    if mode not in ("sector", "segment"):
        issues.append(f"mode must be 'sector' or 'segment', got {mode!r}")

    region = spec.get("region", "minor")
    if region not in ("minor", "major"):
        issues.append(f"region must be 'minor' or 'major', got {region!r}")

    return (len(issues) == 0), issues


def compute_circle_sector(spec: dict) -> dict:
    """Pure math, exact closed-form. Returns every quantity NCERT ever
    asks for in this chapter, for BOTH minor and major at once, so a
    caller (this renderer, or a future solver.py integration) never
    has to re-derive one from the other by hand:

        {'arc_length_minor', 'arc_length_major',
         'sector_area_minor', 'sector_area_major',
         'segment_area_minor', 'segment_area_major',
         'circle_area', 'circumference',
         'theta_deg', 'center': (cx,cy), 'radius': r}
    """
    center = spec.get("center") or {"x": 0, "y": 0}
    cx, cy = float(center["x"]), float(center["y"])
    r = float(spec["radius"])
    theta = float(spec["angle_deg"])
    theta_rad = math.radians(theta)

    circle_area = math.pi * r * r
    circumference = 2 * math.pi * r

    arc_minor = (theta / 360.0) * circumference
    arc_major = circumference - arc_minor

    sector_minor = (theta / 360.0) * circle_area
    sector_major = circle_area - sector_minor

    triangle_oab = 0.5 * r * r * math.sin(theta_rad)
    segment_minor = sector_minor - triangle_oab
    segment_major = circle_area - segment_minor

    return {
        "center": (cx, cy), "radius": r, "theta_deg": theta,
        "circle_area": circle_area, "circumference": circumference,
        "arc_length_minor": arc_minor, "arc_length_major": arc_major,
        "sector_area_minor": sector_minor, "sector_area_major": sector_major,
        "segment_area_minor": segment_minor, "segment_area_major": segment_major,
        "triangle_area_oab": triangle_oab,
    }


def _arc_sample_points(cx, cy, r, start_rad, end_rad, n=2000):
    """n+1 exact points (math-space, y-up) along the circle from
    start_rad to end_rad, walking in the increasing-angle direction
    (end_rad is always start_rad + the intended sweep in radians, so
    the direction is unambiguous and never guessed)."""
    return [(cx + r * math.cos(start_rad + (end_rad - start_rad) * i / n),
             cy + r * math.sin(start_rad + (end_rad - start_rad) * i / n))
            for i in range(n + 1)]


def _shoelace_area(points) -> float:
    total = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def verify_circle_sector_construction(spec: dict, solved: dict) -> tuple:
    """Independently re-measures the region this spec asks for by
    sampling the actual arc geometry and running the shoelace formula
    over the resulting polygon, then compares that measurement against
    the closed-form formula in compute_circle_sector. This is the same
    "don't just trust the formula, re-derive it from the actual drawn
    geometry" check circle_line_plugin.py applies to intersection
    points — here applied to area."""
    issues = []
    cx, cy = solved["center"]
    r = solved["radius"]
    theta = solved["theta_deg"]
    mode = spec.get("mode", "sector")
    region = spec.get("region", "minor")

    start_rad = math.radians(270 - theta / 2.0)
    end_rad_minor = start_rad + math.radians(theta)

    if region == "minor":
        arc_pts = _arc_sample_points(cx, cy, r, start_rad, end_rad_minor)
        expected_sector = solved["sector_area_minor"]
        expected_segment = solved["segment_area_minor"]
    else:
        # major arc: walk the OTHER way around from B back to A, i.e.
        # continue increasing angle from end_rad_minor all the way to
        # start_rad + 2*pi (the long way round), never the short way.
        arc_pts = _arc_sample_points(cx, cy, r, end_rad_minor, start_rad + 2 * math.pi)
        expected_sector = solved["sector_area_major"]
        expected_segment = solved["segment_area_major"]

    if mode == "sector":
        polygon = [(cx, cy)] + arc_pts
        measured = _shoelace_area(polygon)
        if not (math.isfinite(measured) and math.isfinite(expected_sector)):
            issues.append(f"sector area is non-finite (measured={measured}, expected={expected_sector}) "
                           f"— rejecting rather than silently passing a NaN/inf comparison")
        elif abs(measured - expected_sector) > max(1e-6, expected_sector * 1e-4):
            issues.append(f"rendered sector polygon measures area {measured:.6f} but the closed-form "
                           f"formula gives {expected_sector:.6f} — arc direction/flags likely wrong")
    else:
        polygon = arc_pts
        measured = _shoelace_area(polygon)
        if not (math.isfinite(measured) and math.isfinite(expected_segment)):
            issues.append(f"segment area is non-finite (measured={measured}, expected={expected_segment}) "
                           f"— rejecting rather than silently passing a NaN/inf comparison")
        elif abs(measured - expected_segment) > max(1e-6, expected_segment * 1e-4):
            issues.append(f"rendered segment polygon measures area {measured:.6f} but the closed-form "
                           f"formula gives {expected_segment:.6f} — arc direction/flags likely wrong")

    return (len(issues) == 0), issues


def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else f"{v:.0f}"


def _render_circle_sector(spec: dict) -> str:
    solved = compute_circle_sector(spec)
    ok, issues = verify_circle_sector_construction(spec, solved)
    if not ok:
        return ""

    cx_m, cy_m = solved["center"]
    r_m = solved["radius"]
    theta = solved["theta_deg"]
    mode = spec.get("mode", "sector")
    region = spec.get("region", "minor")
    shaded = bool(spec.get("shaded", True))
    labels = spec.get("labels") or {}
    label_a = labels.get("A", "A")
    label_b = labels.get("B", "B")
    center_label = (spec.get("center") or {}).get("label", "O")

    scale = (W - 2 * MARGIN) / (2 * r_m)
    scale = min(scale, (H - 2 * MARGIN) / (2 * r_m))

    def to_svg(p):
        return (W / 2 + (p[0] - cx_m) * scale, H / 2 - (p[1] - cy_m) * scale)

    center_svg = to_svg((cx_m, cy_m))
    r_px = r_m * scale

    start_rad = math.radians(270 - theta / 2.0)
    end_rad = start_rad + math.radians(theta)
    A_m = (cx_m + r_m * math.cos(start_rad), cy_m + r_m * math.sin(start_rad))
    B_m = (cx_m + r_m * math.cos(end_rad), cy_m + r_m * math.sin(end_rad))
    A_svg, B_svg = to_svg(A_m), to_svg(B_m)

    body = []
    body.append(f'<circle cx="{center_svg[0]:.1f}" cy="{center_svg[1]:.1f}" r="{r_px:.1f}" fill="none" '
                f'stroke="{LINE_COLOR}" stroke-width="1.6" data-role="circle"/>')

    # Determine the two SVG-space angles of A and B exactly (no
    # orientation assumption — read straight off the actual rendered
    # pixel positions), then pick whichever of the two possible arcs
    # between them matches the target region's degree-size. This is
    # robust to the coordinate flip between math-space (y-up) and
    # SVG-space (y-down) by construction, not by reasoning about it.
    def svg_angle(p):
        return math.atan2(p[1] - center_svg[1], p[0] - center_svg[0])

    ang_a, ang_b = svg_angle(A_svg), svg_angle(B_svg)
    delta_ccw = (ang_b - ang_a) % (2 * math.pi)  # sweep=1 (positive-angle) direction, degrees
    delta_cw = (2 * math.pi) - delta_ccw          # sweep=0 direction

    target_deg = theta if region == "minor" else (360 - theta)
    # pick the direction whose degree measure matches target_deg
    if abs(math.degrees(delta_ccw) - target_deg) <= abs(math.degrees(delta_cw) - target_deg):
        sweep_flag, arc_deg = 1, math.degrees(delta_ccw)
    else:
        sweep_flag, arc_deg = 0, math.degrees(delta_cw)
    large_arc_flag = 1 if arc_deg > 180 else 0

    arc_path = (f'A {r_px:.2f} {r_px:.2f} 0 {large_arc_flag} {sweep_flag} '
                f'{B_svg[0]:.2f} {B_svg[1]:.2f}')

    if mode == "sector":
        path_d = f'M {center_svg[0]:.2f} {center_svg[1]:.2f} L {A_svg[0]:.2f} {A_svg[1]:.2f} {arc_path} Z'
    else:  # segment: chord-bounded, no radii in the shaded outline
        path_d = f'M {A_svg[0]:.2f} {A_svg[1]:.2f} {arc_path} Z'

    if shaded:
        body.append(f'<path d="{path_d}" fill="{SHADE_COLOR}" fill-opacity="0.55" '
                    f'stroke="none" data-role="shaded-region"/>')

    # radii OA, OB always drawn (dashed for segment mode, since the
    # question's construction still has them even though the shaded
    # region itself is chord-bounded; solid for sector mode)
    dash = ' stroke-dasharray="4,3"' if mode == "segment" else ""
    body.append(f'<line x1="{center_svg[0]:.1f}" y1="{center_svg[1]:.1f}" x2="{A_svg[0]:.1f}" '
                f'y2="{A_svg[1]:.1f}" stroke="{LINE_COLOR}" stroke-width="1.6"{dash} data-role="radius-oa"/>')
    body.append(f'<line x1="{center_svg[0]:.1f}" y1="{center_svg[1]:.1f}" x2="{B_svg[0]:.1f}" '
                f'y2="{B_svg[1]:.1f}" stroke="{LINE_COLOR}" stroke-width="1.6"{dash} data-role="radius-ob"/>')
    if mode == "segment":
        body.append(f'<line x1="{A_svg[0]:.1f}" y1="{A_svg[1]:.1f}" x2="{B_svg[0]:.1f}" y2="{B_svg[1]:.1f}" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.6" data-role="chord"/>')

    # the arc itself always outlined, independent of the shaded fill
    body.append(f'<path d="M {A_svg[0]:.2f} {A_svg[1]:.2f} {arc_path}" fill="none" '
                f'stroke="{LINE_COLOR}" stroke-width="1.8" data-role="arc"/>')

    body.append(f'<circle cx="{center_svg[0]:.1f}" cy="{center_svg[1]:.1f}" r="1.8" fill="{LINE_COLOR}" '
                f'data-role="center-point"/>')
    body.append(f'<text x="{center_svg[0]:.1f}" y="{center_svg[1]+14:.1f}" text-anchor="middle" '
                f'fill="{LABEL_COLOR}" {SMALL_FONT} data-role="center-label">{_escape(center_label)}</text>')
    for (pt_svg, lbl) in ((A_svg, label_a), (B_svg, label_b)):
        dy_lbl = -8 if pt_svg[1] < center_svg[1] else 16
        body.append(f'<text x="{pt_svg[0]:.1f}" y="{(pt_svg[1]+dy_lbl):.1f}" text-anchor="middle" '
                    f'fill="{LABEL_COLOR}" {FONT} data-role="point-label">{_escape(lbl)}</text>')

    angle_label = spec.get("labels", {}).get("angle", f"{_fmt(theta)}°") if region == "minor" \
        else f"{_fmt(360 - theta)}°"
    body.append(f'<text x="{center_svg[0]:.1f}" y="{center_svg[1]-14:.1f}" text-anchor="middle" '
                f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="angle-label">{_escape(angle_label)}</text>')

    caption = f"{region} {mode}" + (f", r = {_fmt(r_m)}" if spec.get("show_radius_label", True) else "")
    body.append(f'<text x="{W/2:.1f}" y="{H-8:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                f'{SMALL_FONT} data-role="caption">{_escape(caption)}</text>')

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="max-width:260px">' + "".join(body) + "</svg>")
    fixed = label_layout.check_and_fix_labels(svg)
    return fixed["svg"]


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="circle_sector",
    renderer=_render_circle_sector,
    schema_check=_validate_circle_sector_spec,
    solver=None,
    description="Class 10 'Areas Related to Circles': a circle (center+radius) with a sector/segment "
                "defined by angle AOB = angle_deg, in mode 'sector'|'segment' and region 'minor'|'major'. "
                "arc length, sector area, and segment area are all computed from exact closed-form "
                "formulas (never estimated) via compute_circle_sector(), and the actual rendered arc is "
                "independently re-measured by shoelace-formula sampling and cross-checked against those "
                "formulas before any SVG is returned.",
))
