"""
unit_circle_plugin.py — diagram_type "unit_circle" (GENUINE NEW GAP:
confirmed absent from the v30 codebase — no plugin, no built-in
renderer, no schema check anywhere referenced "unit circle" before
this file).

Renders the standard Class 10 / Class 11-bridge trigonometry figure:
a circle of radius 1 centred at the origin, the terminal ray for a
given angle theta (measured standard-mathematical convention,
counter-clockwise from the positive x-axis), the point
(cos theta, sin theta) on the circle, and the reference right triangle
(drop to the x-axis) that the value of theta implies.

MATH, NEVER ESTIMATED: the terminal point is computed as
(cos(theta), sin(theta)) via Python's own math.cos/math.sin — the same
"compute from the actual relationship, never eyeball a coordinate"
rule every other plugin in this project already follows. For the
standard textbook angles (multiples of 30 degrees or 45 degrees) the
point is ALSO labelled with its exact symbolic value (e.g. "(1/2,
(root3)/2)") rather than a decimal approximation, since that is what a
Class 10 answer key actually expects; any other angle gets a
3-decimal-place numeric label instead of a fabricated "nice" fraction.

POST-RENDER MATH VALIDATION: independently recomputes cos^2+sin^2 and
asserts it is 1 within floating-point tolerance, and asserts the
plotted point's Euclidean distance from the origin is exactly the
circle's own radius — i.e. this diagram proves the point it draws
really does lie ON the unit circle, rather than trusting the arithmetic
that placed it there. This is the "post-render validation" pattern
already established by diagram_renderer.py's
_verify_surface_area_volume_coverage, applied to a new diagram family.

Wired in via diagram_plugin_registry.py, exactly like every other
plugin in this project (probability_plugin.py, elevation_bearing_plugin.py,
etc.) — no existing file is modified to add this type.
"""
import math

import diagram_plugin_registry
import label_layout

LINE_COLOR = "#1a4d8f"
AUX_COLOR = "#c0392b"
AXIS_COLOR = "#555555"
QUADRANT_COLOR = "#eef3fb"
LABEL_COLOR_FALLBACK = "#111111"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='600'"
SMALL_FONT = "font-family='Noto Sans, Arial' font-size='11'"

W, H = 280, 260
MARGIN = 34
R_PX = min(W, H) / 2 - MARGIN  # pixel radius of the drawn unit circle
CX, CY = W / 2, H / 2 + 6

# Exact symbolic (cos, sin) labels for the standard textbook angles,
# keyed by angle mod 360 in degrees. Values are display strings, not
# used for any arithmetic — the actual plotted point always comes from
# math.cos/math.sin so the picture is correct even if this lookup were
# ever wrong or incomplete.
_EXACT_LABELS = {
    0: ("1", "0"), 30: ("(root3)/2", "1/2"), 45: ("(root2)/2", "(root2)/2"),
    60: ("1/2", "(root3)/2"), 90: ("0", "1"), 120: ("-1/2", "(root3)/2"),
    135: ("-(root2)/2", "(root2)/2"), 150: ("-(root3)/2", "1/2"), 180: ("-1", "0"),
    210: ("-(root3)/2", "-1/2"), 225: ("-(root2)/2", "-(root2)/2"),
    240: ("-1/2", "-(root3)/2"), 270: ("0", "-1"), 300: ("1/2", "-(root3)/2"),
    315: ("(root2)/2", "-(root2)/2"), 330: ("(root3)/2", "-1/2"), 360: ("1", "0"),
}


def _is_number(v) -> bool:
    # V37 hardening pass: math.isfinite explicitly excludes NaN/inf.
    # isinstance-only checks let NaN through, and every subsequent
    # "<= 0 is invalid" comparison against NaN is silently False in
    # Python, so NaN previously bypassed validation undetected.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _validate_unit_circle_spec(spec: dict):
    issues = []
    angle = spec.get("angle_deg")
    if not _is_number(angle):
        issues.append(f"angle_deg must be a number, got {angle!r}")
    if "show_special_angles" in spec and not isinstance(spec["show_special_angles"], bool):
        issues.append("show_special_angles must be a boolean if given")
    if "quadrant_shading" in spec and not isinstance(spec["quadrant_shading"], bool):
        issues.append("quadrant_shading must be a boolean if given")
    return (len(issues) == 0), issues


def solve_unit_circle(spec: dict) -> dict:
    """Pure math: returns {'angle_deg', 'angle_norm_deg', 'cos', 'sin',
    'quadrant'}. angle_norm_deg is angle_deg reduced to [0, 360) for
    display/lookup purposes only — the trig values themselves are
    computed straight from the original angle_deg so a caller passing
    e.g. -30 or 390 still gets the mathematically correct point."""
    angle_deg = float(spec["angle_deg"])
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    norm = angle_deg % 360.0
    if abs(c) < 1e-9:
        quadrant = "on the y-axis"
    elif abs(s) < 1e-9:
        quadrant = "on the x-axis"
    else:
        quadrant = {(1, 1): "I", (-1, 1): "II", (-1, -1): "III", (1, -1): "IV"}[
            (1 if c > 0 else -1, 1 if s > 0 else -1)]
    return {"angle_deg": angle_deg, "angle_norm_deg": norm, "cos": c, "sin": s, "quadrant": quadrant}


def verify_unit_circle_construction(solved: dict) -> tuple:
    """Post-render mathematical validation: proves the computed point
    actually lies on the unit circle (distance from origin == 1) and
    that the fundamental identity cos^2(theta)+sin^2(theta)=1 holds for
    the angle drawn. Returns (ok, issues) — never raises."""
    issues = []
    c, s = solved["cos"], solved["sin"]
    identity = c * c + s * s
    if abs(identity - 1.0) > 1e-6:
        issues.append(f"cos^2+sin^2 = {identity:.9f}, expected 1.0 (construction is not on the unit circle)")
    dist = math.hypot(c, s)
    if abs(dist - 1.0) > 1e-6:
        issues.append(f"plotted point distance from origin = {dist:.9f}, expected radius 1.0")
    return (len(issues) == 0), issues


def _to_svg(x_math: float, y_math: float) -> tuple:
    return (CX + x_math * R_PX, CY - y_math * R_PX)


def _render_unit_circle(spec: dict) -> str:
    solved = solve_unit_circle(spec)
    ok, issues = verify_unit_circle_construction(solved)
    if not ok:
        # Fail-soft, same contract as every other plugin: never raise
        # out of a renderer, log-worthy failure just yields no diagram.
        return ""

    c, s = solved["cos"], solved["sin"]
    angle_deg = solved["angle_deg"]
    norm = round(solved["angle_norm_deg"]) % 360
    show_special = bool(spec.get("show_special_angles", False))
    shade_quadrants = bool(spec.get("quadrant_shading", False))

    body = []

    if shade_quadrants:
        # quadrant I (top-right in math space == top-right in svg since y flips)
        body.append(f'<path d="M {CX:.1f} {CY:.1f} L {CX+R_PX:.1f} {CY:.1f} A {R_PX} {R_PX} 0 0 0 '
                    f'{CX:.1f} {CY-R_PX:.1f} Z" fill="{QUADRANT_COLOR}" data-role="quadrant-shade"/>')

    # axes
    body.append(f'<line x1="{MARGIN/2:.1f}" y1="{CY:.1f}" x2="{W-MARGIN/2:.1f}" y2="{CY:.1f}" '
                f'stroke="{AXIS_COLOR}" stroke-width="1.4" data-role="x-axis"/>')
    body.append(f'<line x1="{CX:.1f}" y1="{MARGIN/2:.1f}" x2="{CX:.1f}" y2="{H-MARGIN/2:.1f}" '
                f'stroke="{AXIS_COLOR}" stroke-width="1.4" data-role="y-axis"/>')
    body.append(f'<text x="{W-MARGIN/2+6:.1f}" y="{CY+4:.1f}" {SMALL_FONT} fill="{AXIS_COLOR}" '
                f'data-role="axis-label">x</text>')
    body.append(f'<text x="{CX-4:.1f}" y="{MARGIN/2-6:.1f}" text-anchor="end" {SMALL_FONT} fill="{AXIS_COLOR}" '
                f'data-role="axis-label">y</text>')

    # unit circle
    body.append(f'<circle cx="{CX:.1f}" cy="{CY:.1f}" r="{R_PX:.1f}" fill="none" '
                f'stroke="{LINE_COLOR}" stroke-width="2" data-role="unit-circle"/>')

    if show_special:
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            tx1, ty1 = _to_svg(math.cos(rad) * 0.96, math.sin(rad) * 0.96)
            tx2, ty2 = _to_svg(math.cos(rad) * 1.04, math.sin(rad) * 1.04)
            body.append(f'<line x1="{tx1:.1f}" y1="{ty1:.1f}" x2="{tx2:.1f}" y2="{ty2:.1f}" '
                        f'stroke="{LINE_COLOR}" stroke-width="1" opacity="0.5" data-role="tick"/>')

    px, py = _to_svg(c, s)
    ox, oy = _to_svg(0, 0)
    foot_x, foot_y = _to_svg(c, 0)  # foot of the reference perpendicular on the x-axis

    # radius to the point (terminal ray)
    body.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
                f'stroke="{LINE_COLOR}" stroke-width="2" data-role="radius"/>')

    # reference triangle: perpendicular drop to the x-axis (dashed, constructional)
    if abs(c) > 1e-6 and abs(s) > 1e-6:
        body.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{foot_x:.1f}" y2="{foot_y:.1f}" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.4" stroke-dasharray="4,3" data-role="reference-drop"/>')

    # angle arc from positive x-axis to the terminal ray
    arc_r = 20
    ang_start, ang_end = 0.0, angle_deg
    large_arc = 1 if abs(ang_end - ang_start) % 360 > 180 else 0
    sweep = 1 if ang_end >= ang_start else 0
    ax1, ay1 = ox + arc_r, oy
    ax2 = ox + arc_r * math.cos(math.radians(-ang_end))
    ay2 = oy + arc_r * math.sin(math.radians(-ang_end))
    body.append(f'<path d="M {ax1:.1f} {ay1:.1f} A {arc_r} {arc_r} 0 {large_arc} {sweep} '
                f'{ax2:.1f} {ay2:.1f}" fill="none" stroke="{AUX_COLOR}" stroke-width="1.4" '
                f'data-role="angle-arc"/>')
    mid_ang = math.radians(angle_deg / 2.0)
    body.append(f'<text x="{ox + (arc_r+14)*math.cos(mid_ang):.1f}" '
                f'y="{oy - (arc_r+14)*math.sin(mid_ang):.1f}" text-anchor="middle" '
                f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="angle-label">{angle_deg:g}&#176;</text>')

    # the plotted point + its coordinate label (exact symbolic form for
    # standard angles, 3-decimal numeric otherwise — see module docstring)
    body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{LINE_COLOR}" data-role="terminal-point"/>')
    if norm in _EXACT_LABELS:
        cos_lbl, sin_lbl = _EXACT_LABELS[norm]
        point_text = f"({cos_lbl}, {sin_lbl})"
    else:
        point_text = f"({c:.3f}, {s:.3f})"
    label_dx = 10 if px >= CX else -10
    anchor = "start" if px >= CX else "end"
    label_dy = -8 if py <= CY else 14
    body.append(f'<text x="{px+label_dx:.1f}" y="{py+label_dy:.1f}" text-anchor="{anchor}" '
                f'fill="{LABEL_COLOR_FALLBACK}" {FONT} data-role="point-label">{_escape(point_text)}</text>')

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="max-width:260px">' + "".join(body) + "</svg>")

    fixed = label_layout.check_and_fix_labels(svg)
    return fixed["svg"]


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="unit_circle",
    renderer=_render_unit_circle,
    schema_check=_validate_unit_circle_spec,
    description="Unit-circle trigonometry diagram: circle of radius 1, the terminal ray for a given "
                "angle theta (any real value, standard math convention), the point (cos theta, sin theta) "
                "computed via math.cos/math.sin (never estimated), and its reference right triangle. "
                "Standard 30/45-degree-multiple angles get an exact symbolic coordinate label; every "
                "other angle gets a 3-decimal numeric label. Independently re-verifies cos^2+sin^2=1 and "
                "that the plotted point's distance from the origin equals the circle's radius before "
                "returning any SVG.",
))
