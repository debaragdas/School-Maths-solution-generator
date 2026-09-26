"""
elevation_bearing_plugin.py — diagram_type #16 ("elevation_depression")
and #17 ("bearing").

Covers two Class 9-10 SEBA/SCERT trigonometry-application figure
families the existing 12 built-in types + earlier plugins do NOT
handle:

  * "elevation_depression" — the right-triangle-with-a-horizontal-
    reference-line diagram used for "angle of elevation of the top of
    a tower..." / "angle of depression from the top of a cliff..."
    questions. Distinct from the existing "trigonometry" type (which
    is a bare labelled right triangle with no horizontal-of-sight
    semantics and no notion of which vertex is the ground and which
    is elevated).

  * "bearing" — a sequence of navigation/survey legs measured as a
    compass bearing (0°-360°, clockwise from North) and a distance
    from a start point, the diagram family used for "a ship sails on
    a bearing of 060°..." questions.

Same split as every other diagram type in this project: Gemini
supplies the SHAPE (angle, which side is known, bearing + distance per
leg) — it never supplies (x, y) coordinates itself. Every coordinate
below is computed here from trigonometry, and the *unknown* side is
still computed internally (so the picture's proportions are always
geometrically correct) even though its numeric value is withheld from
the rendered label by default, so a diagram never spoils the exercise
it illustrates.

Wired in via diagram_plugin_registry.py, exactly like
function_graph_plugin.py / probability_plugin.py / set_theory_plugin.py.
"""
import math

import diagram_plugin_registry

LINE_COLOR = "#1a4d8f"
AUX_COLOR = "#c0392b"      # dashed sight-lines / horizontal references / arcs
GROUND_COLOR = "#555555"
LABEL_COLOR = "#111111"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='600'"
SMALL_FONT = "font-family='Noto Sans, Arial' font-size='11'"

W, H = 280, 220
MARGIN = 42


def _is_number(v) -> bool:
    # V37 hardening pass: math.isfinite explicitly excludes NaN/inf.
    # isinstance-only checks let NaN through, and every subsequent
    # "<= 0 is invalid" comparison against NaN is silently False in
    # Python, so NaN previously bypassed validation undetected.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ------------------------------------------------------------------
# ELEVATION / DEPRESSION
# ------------------------------------------------------------------
_ELEV_KNOWN_SIDES = {"horizontal", "vertical"}
_ELEV_MODES = {"elevation", "depression"}


def _validate_elevation_depression_spec(spec: dict):
    issues = []

    mode = spec.get("mode")
    if mode not in _ELEV_MODES:
        issues.append(f"mode must be one of {sorted(_ELEV_MODES)}, got {mode!r}")

    points = spec.get("points")
    if not isinstance(points, dict):
        issues.append("'points' must be an object with 'eye', 'target', 'foot' ids")
    else:
        ids = {}
        for role in ("eye", "target", "foot"):
            pid = points.get(role)
            if not pid or not isinstance(pid, str):
                issues.append(f"points.{role} must be a non-empty string id")
            else:
                ids[role] = pid
        if len(ids) == 3 and len(set(ids.values())) != 3:
            issues.append("points.eye, points.target, points.foot must all be distinct")

    angle = spec.get("angle_deg")
    if not _is_number(angle) or not (0 < angle < 90):
        issues.append(f"angle_deg must be a number strictly between 0 and 90, got {angle!r}")

    known_side = spec.get("known_side")
    if known_side not in _ELEV_KNOWN_SIDES:
        issues.append(f"known_side must be one of {sorted(_ELEV_KNOWN_SIDES)}, got {known_side!r}")

    known_value = spec.get("known_value")
    if not _is_number(known_value) or known_value <= 0:
        issues.append(f"known_value must be a positive number, got {known_value!r}")

    if "show_unknown_value" in spec and not isinstance(spec["show_unknown_value"], bool):
        issues.append("show_unknown_value must be a boolean if given")

    return (len(issues) == 0), issues


def _solve_elevation_depression(spec: dict) -> dict:
    """Pure math: returns {'horizontal', 'vertical', 'hypotenuse',
    'angle_deg'} computed from tan(angle) = vertical / horizontal — the
    same relationship regardless of mode (see module docstring / the
    engineering report for the alternate-angle justification of why
    depression uses the identical ratio drawn at a different vertex).
    Never estimates: both legs are always derivable from one leg +
    the angle.
    """
    angle_deg = float(spec["angle_deg"])
    angle_rad = math.radians(angle_deg)
    known_side = spec["known_side"]
    known_value = float(spec["known_value"])

    if known_side == "horizontal":
        horizontal = known_value
        vertical = horizontal * math.tan(angle_rad)
    else:
        vertical = known_value
        horizontal = vertical / math.tan(angle_rad)

    hypotenuse = math.hypot(horizontal, vertical)
    return {"horizontal": horizontal, "vertical": vertical,
            "hypotenuse": hypotenuse, "angle_deg": angle_deg}


def _fmt_len(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return f"{round(v):g}"
    return f"{v:.2f}"


def _arc_path(vertex, r, start_deg, end_deg):
    """SVG arc path between two angles (degrees, standard math convention,
    0 = +x axis, counterclockwise) around vertex, at radius r, in SVG
    pixel space (y grows downward, so we negate the y component of the
    unit vector when converting from math-angle to pixel offset)."""
    def pt(deg):
        rad = math.radians(deg)
        return vertex[0] + r * math.cos(rad), vertex[1] - r * math.sin(rad)
    x1, y1 = pt(start_deg)
    x2, y2 = pt(end_deg)
    large_arc = 1 if abs(end_deg - start_deg) > 180 else 0
    sweep = 1 if end_deg < start_deg else 0
    return f'M {x1:.1f} {y1:.1f} A {r} {r} 0 {large_arc} {sweep} {x2:.1f} {y2:.1f}'


def _render_elevation_depression(spec: dict) -> str:
    mode = spec["mode"]
    points = spec["points"]
    solved = _solve_elevation_depression(spec)
    horizontal, vertical = solved["horizontal"], solved["vertical"]
    angle_deg = solved["angle_deg"]
    show_unknown = bool(spec.get("show_unknown_value", False))
    unknown_label = spec.get("unknown_label", "?")

    # --- math-space coordinates (y-up), then flip to SVG (y-down). ---
    if mode == "elevation":
        eye_m = (0.0, 0.0)
        foot_m = (horizontal, 0.0)
        target_m = (horizontal, vertical)
        ground_extra = (-horizontal * 0.28, 0.0)  # decorative ground extension past eye
    else:  # depression
        eye_m = (0.0, vertical)
        foot_m = (0.0, 0.0)
        target_m = (horizontal, 0.0)
        ground_extra = (horizontal * 1.15, vertical)  # horizontal reference ray past eye

    pts_m = [eye_m, foot_m, target_m, ground_extra]
    xs = [p[0] for p in pts_m]
    ys = [p[1] for p in pts_m]
    span_x = max(max(xs) - min(xs), 1.0)
    span_y = max(max(ys) - min(ys), 1.0)
    scale = min((W - 2 * MARGIN) / span_x, (H - 2 * MARGIN) / span_y)
    min_x, min_y = min(xs), min(ys)

    def to_svg(p):
        return (MARGIN + (p[0] - min_x) * scale,
                (H - MARGIN) - (p[1] - min_y) * scale)

    eye, foot, target = to_svg(eye_m), to_svg(foot_m), to_svg(target_m)

    body = []

    # ground / pole legs (solid — these are physical edges)
    body.append(f'<line x1="{foot[0]:.1f}" y1="{foot[1]:.1f}" x2="{target[0]:.1f}" y2="{target[1]:.1f}" '
                f'stroke="{GROUND_COLOR}" stroke-width="2.4" data-role="elev-leg"/>')
    body.append(f'<line x1="{eye[0]:.1f}" y1="{eye[1]:.1f}" x2="{foot[0]:.1f}" y2="{foot[1]:.1f}" '
                f'stroke="{GROUND_COLOR}" stroke-width="2" data-role="elev-leg"/>')

    # right-angle mark at foot (both legs axis-aligned by construction)
    s = 8
    dir_to_eye = (1 if eye[0] > foot[0] else (-1 if eye[0] < foot[0] else 0),
                  1 if eye[1] > foot[1] else (-1 if eye[1] < foot[1] else 0))
    dir_to_target = (1 if target[0] > foot[0] else (-1 if target[0] < foot[0] else 0),
                      1 if target[1] > foot[1] else (-1 if target[1] < foot[1] else 0))
    p1 = (foot[0] + dir_to_eye[0] * s, foot[1] + dir_to_eye[1] * s)
    p2 = (foot[0] + dir_to_eye[0] * s + dir_to_target[0] * s, foot[1] + dir_to_eye[1] * s + dir_to_target[1] * s)
    p3 = (foot[0] + dir_to_target[0] * s, foot[1] + dir_to_target[1] * s)
    body.append(f'<polyline points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}" '
                f'fill="none" stroke="{GROUND_COLOR}" stroke-width="1.3" data-role="right-angle"/>')

    # line of sight (dashed — not a physical edge)
    body.append(f'<line x1="{eye[0]:.1f}" y1="{eye[1]:.1f}" x2="{target[0]:.1f}" y2="{target[1]:.1f}" '
                f'stroke="{AUX_COLOR}" stroke-width="1.6" stroke-dasharray="4,3" data-role="sight-line"/>')

    # angle at eye + its horizontal reference
    if mode == "elevation":
        # the leg eye->foot IS the horizontal; arc sits between it and the sight line
        # compute svg-space angle of eye->foot and eye->target explicitly, robust to layout
        ang_foot = math.degrees(math.atan2(-(foot[1] - eye[1]), foot[0] - eye[0]))
        ang_target = math.degrees(math.atan2(-(target[1] - eye[1]), target[0] - eye[0]))
        arc_r = 22
        body.append(f'<path d="{_arc_path(eye, arc_r, ang_foot, ang_target)}" fill="none" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.4" data-role="angle-arc"/>')
        label_deg = (ang_foot + ang_target) / 2
    else:
        # dashed horizontal reference ray at eye, extending toward target's side
        far = to_svg(ground_extra)
        body.append(f'<line x1="{eye[0]:.1f}" y1="{eye[1]:.1f}" x2="{far[0]:.1f}" y2="{far[1]:.1f}" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.4" stroke-dasharray="3,2" data-role="horizontal-reference"/>')
        ang_ref = math.degrees(math.atan2(-(far[1] - eye[1]), far[0] - eye[0]))
        ang_target = math.degrees(math.atan2(-(target[1] - eye[1]), target[0] - eye[0]))
        arc_r = 22
        body.append(f'<path d="{_arc_path(eye, arc_r, ang_ref, ang_target)}" fill="none" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.4" data-role="angle-arc"/>')
        label_deg = (ang_ref + ang_target) / 2

    lab_rad = math.radians(label_deg)
    lx = eye[0] + (arc_r + 12) * math.cos(lab_rad)
    ly = eye[1] - (arc_r + 12) * math.sin(lab_rad)
    body.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="{AUX_COLOR}" {SMALL_FONT} '
                f'data-role="angle-label">{angle_deg:g}&#176;</text>')

    # point labels
    for (px, py), role in ((eye, "eye"), (foot, "foot"), (target, "target")):
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{LINE_COLOR}"/>')
        dy = -8 if py < H / 2 else 14
        body.append(f'<text x="{px:.1f}" y="{(py+dy):.1f}" text-anchor="middle" fill="{LABEL_COLOR}" {FONT} '
                    f'data-role="point-label">{_escape(points[role])}</text>')

    # dimension labels — known side always numeric, unknown side withheld
    # unless the caller explicitly opts in to showing the computed value.
    hmid = ((eye[0] + foot[0]) / 2 if mode == "elevation" else (foot[0] + target[0]) / 2,
            (eye[1] + foot[1]) / 2 if mode == "elevation" else (foot[1] + target[1]) / 2)
    vmid = ((foot[0] + target[0]) / 2 if mode == "elevation" else (eye[0] + foot[0]) / 2,
            (foot[1] + target[1]) / 2 if mode == "elevation" else (eye[1] + foot[1]) / 2)

    h_text = _fmt_len(horizontal) if (spec["known_side"] == "horizontal" or show_unknown) else unknown_label
    v_text = _fmt_len(vertical) if (spec["known_side"] == "vertical" or show_unknown) else unknown_label
    body.append(f'<text x="{hmid[0]:.1f}" y="{(hmid[1]+14):.1f}" text-anchor="middle" fill="{AUX_COLOR}" '
                f'{SMALL_FONT} data-role="dimension-label">{_escape(h_text)}</text>')
    body.append(f'<text x="{(vmid[0]-8):.1f}" y="{vmid[1]:.1f}" text-anchor="end" dominant-baseline="middle" '
                f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">{_escape(v_text)}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:260px">' + "".join(body) + "</svg>")


# ------------------------------------------------------------------
# BEARING
# ------------------------------------------------------------------
def _validate_bearing_spec(spec: dict):
    issues = []

    start = spec.get("start")
    if not isinstance(start, dict) or not start.get("id"):
        issues.append("'start' must be an object with a non-empty 'id'")
        known_ids = set()
    else:
        known_ids = {start["id"]}

    legs = spec.get("legs")
    if not isinstance(legs, list) or len(legs) == 0:
        issues.append("'legs' must be a non-empty list")
        return (len(issues) == 0), issues

    prev_id = start.get("id") if isinstance(start, dict) else None
    for i, leg in enumerate(legs):
        if not isinstance(leg, dict):
            issues.append(f"legs[{i}] must be an object")
            continue
        to = leg.get("to")
        if not isinstance(to, dict) or not to.get("id"):
            issues.append(f"legs[{i}].to must be an object with a non-empty 'id'")
            continue
        if to["id"] in known_ids:
            issues.append(f"legs[{i}].to.id '{to['id']}' duplicates an earlier point id")
        from_id = leg.get("from_id", prev_id)
        if from_id not in known_ids:
            issues.append(f"legs[{i}].from_id '{from_id}' is not a known point yet")
        bearing = leg.get("bearing_deg")
        if not _is_number(bearing) or not (0 <= bearing < 360):
            issues.append(f"legs[{i}].bearing_deg must be a number in [0, 360), got {bearing!r}")
        dist = leg.get("distance")
        if not _is_number(dist) or dist <= 0:
            issues.append(f"legs[{i}].distance must be a positive number, got {dist!r}")
        known_ids.add(to["id"])
        prev_id = to["id"]

    return (len(issues) == 0), issues


def _solve_bearing(spec: dict) -> dict:
    """Pure math: returns {point_id: (x, y)} in math-space (y-up, North
    = +y, East = +x), computed from each leg's bearing (clockwise from
    North) and distance. Never estimates a position."""
    start = spec["start"]
    coords = {start["id"]: (0.0, 0.0)}
    prev_id = start["id"]
    for leg in spec["legs"]:
        from_id = leg.get("from_id", prev_id)
        fx, fy = coords[from_id]
        brg = math.radians(float(leg["bearing_deg"]))
        dist = float(leg["distance"])
        nx = fx + dist * math.sin(brg)
        ny = fy + dist * math.cos(brg)
        coords[leg["to"]["id"]] = (nx, ny)
        prev_id = leg["to"]["id"]
    return coords


def _render_bearing(spec: dict) -> str:
    coords_m = _solve_bearing(spec)
    ids_in_order = [spec["start"]["id"]] + [leg["to"]["id"] for leg in spec["legs"]]
    labels = {spec["start"]["id"]: spec["start"].get("label", spec["start"]["id"])}
    for leg in spec["legs"]:
        labels[leg["to"]["id"]] = leg["to"].get("label", leg["to"]["id"])

    xs = [c[0] for c in coords_m.values()]
    ys = [c[1] for c in coords_m.values()]
    compass_pad = 26  # room for the little N arrow above the topmost point
    span_x = max(max(xs) - min(xs), 1.0)
    span_y = max(max(ys) - min(ys), 1.0) + compass_pad / 30.0
    scale = min((W - 2 * MARGIN) / span_x, (H - 2 * MARGIN - compass_pad) / span_y)
    min_x, min_y = min(xs), min(ys)

    def to_svg(p):
        return (MARGIN + (p[0] - min_x) * scale,
                (H - MARGIN) - (p[1] - min_y) * scale)

    coords_svg = {pid: to_svg(c) for pid, c in coords_m.items()}

    body = []
    prev_id = spec["start"]["id"]
    seen_compass = set()
    for i, leg in enumerate(spec["legs"]):
        from_id = leg.get("from_id", prev_id)
        to_id = leg["to"]["id"]
        fx, fy = coords_svg[from_id]
        tx, ty = coords_svg[to_id]

        if from_id not in seen_compass:
            body.append(f'<line x1="{fx:.1f}" y1="{(fy+16):.1f}" x2="{fx:.1f}" y2="{(fy-16):.1f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.2" stroke-dasharray="3,2" data-role="north-line"/>')
            body.append(f'<polygon points="{fx:.1f},{(fy-20):.1f} {(fx-3):.1f},{(fy-14):.1f} {(fx+3):.1f},{(fy-14):.1f}" '
                        f'fill="{AUX_COLOR}" data-role="north-arrow"/>')
            body.append(f'<text x="{fx:.1f}" y="{(fy-22):.1f}" text-anchor="middle" fill="{AUX_COLOR}" '
                        f'{SMALL_FONT} data-role="north-label">N</text>')
            seen_compass.add(from_id)

        body.append(f'<line x1="{fx:.1f}" y1="{fy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="bearing-leg"/>')

        # angle arc from due-north (svg-space angle 90) clockwise to the leg direction
        arc_r = 18
        # clockwise from north (90) to ang_leg: sweep flag chosen so the
        # arc always draws the bearing's own swept angle, not its 360-complement
        bearing = float(leg["bearing_deg"])
        start_a, end_a = 90.0, 90.0 - bearing  # clockwise decrease in standard-math-angle terms
        large_arc = 1 if bearing > 180 else 0
        path = (f'M {fx + arc_r*math.cos(math.radians(start_a)):.1f} '
                f'{fy - arc_r*math.sin(math.radians(start_a)):.1f} '
                f'A {arc_r} {arc_r} 0 {large_arc} 1 '
                f'{fx + arc_r*math.cos(math.radians(end_a)):.1f} '
                f'{fy - arc_r*math.sin(math.radians(end_a)):.1f}')
        body.append(f'<path d="{path}" fill="none" stroke="{AUX_COLOR}" stroke-width="1.2" data-role="angle-arc"/>')

        mid_a = math.radians((start_a + end_a) / 2 if bearing <= 180 else (start_a + end_a) / 2 - 180)
        blx = fx + (arc_r + 14) * math.cos(mid_a)
        bly = fy - (arc_r + 14) * math.sin(mid_a)
        body.append(f'<text x="{blx:.1f}" y="{bly:.1f}" text-anchor="middle" fill="{AUX_COLOR}" {SMALL_FONT} '
                    f'data-role="bearing-label">{bearing:03.0f}&#176;</text>')

        body.append(f'<text x="{((fx+tx)/2):.1f}" y="{((fy+ty)/2 - 6):.1f}" text-anchor="middle" '
                    f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">{_fmt_len(float(leg["distance"]))}</text>')

        prev_id = to_id

    for pid in ids_in_order:
        px, py = coords_svg[pid]
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{LINE_COLOR}"/>')
        body.append(f'<text x="{px:.1f}" y="{(py+16):.1f}" text-anchor="middle" fill="{LABEL_COLOR}" {FONT} '
                    f'data-role="point-label">{_escape(labels[pid])}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:260px">' + "".join(body) + "</svg>")


# ------------------------------------------------------------------
# TWO-POINT ELEVATION — height of a tower/building from two ground
# points with two different angles of elevation. Distinct from
# "elevation_depression" above (which is a single-angle, single-triangle
# diagram): this is the classic NCERT Class 10 Ch.9 "two points on the
# ground, X metres apart, angles of elevation alpha and beta" problem,
# which needs TWO simultaneous tan() equations, not one, and a diagram
# with two sight-lines / two angle arcs from two distinct ground points
# to the same top. Neither the single-angle elevation_depression plugin
# nor the plain "trigonometry" built-in type can represent this without
# either faking a second triangle or losing the shared-height constraint
# that makes the picture (and the underlying algebra) correct.
# ------------------------------------------------------------------
_ELEV2_LAYOUTS = {"same_side", "opposite_sides"}


def _validate_elevation_two_point_spec(spec: dict):
    issues = []

    layout = spec.get("layout")
    if layout not in _ELEV2_LAYOUTS:
        issues.append(f"layout must be one of {sorted(_ELEV2_LAYOUTS)}, got {layout!r}")

    points = spec.get("points")
    if not isinstance(points, dict):
        issues.append("'points' must be an object with 'near', 'far', 'foot', 'target' ids")
    else:
        ids = {}
        for role in ("near", "far", "foot", "target"):
            pid = points.get(role)
            if not pid or not isinstance(pid, str):
                issues.append(f"points.{role} must be a non-empty string id")
            else:
                ids[role] = pid
        if len(ids) == 4 and len(set(ids.values())) != 4:
            issues.append("points.near, points.far, points.foot, points.target must all be distinct")

    near_angle = spec.get("near_angle_deg")
    if not _is_number(near_angle) or not (0 < near_angle < 90):
        issues.append(f"near_angle_deg must be a number strictly between 0 and 90, got {near_angle!r}")

    far_angle = spec.get("far_angle_deg")
    if not _is_number(far_angle) or not (0 < far_angle < 90):
        issues.append(f"far_angle_deg must be a number strictly between 0 and 90, got {far_angle!r}")

    if layout == "same_side" and _is_number(near_angle) and _is_number(far_angle):
        # "near" must genuinely be nearer the foot than "far" — otherwise the
        # tan-difference below goes non-positive and there is no valid triangle.
        if near_angle <= far_angle:
            issues.append("for layout='same_side', near_angle_deg must be strictly greater than "
                          f"far_angle_deg (the closer point always sees the greater angle), got "
                          f"near={near_angle!r}, far={far_angle!r}")

    distance_between = spec.get("distance_between")
    if not _is_number(distance_between) or distance_between <= 0:
        issues.append(f"distance_between must be a positive number, got {distance_between!r}")

    if "show_height" in spec and not isinstance(spec["show_height"], bool):
        issues.append("show_height must be a boolean if given")

    return (len(issues) == 0), issues


def _solve_elevation_two_point(spec: dict) -> dict:
    """Pure math, both layouts solved from two simultaneous tan() equations
    (never estimated):

    same_side (near and far both on the same side of the foot, in line):
        tan(near) = h / x            tan(far) = h / (x + p)
        => h = p / (cot(far) - cot(near)),  x = h / tan(near)

    opposite_sides (foot is between the two observation points):
        tan(near) = h / x            tan(far) = h / (p - x)
        => h = p / (cot(near) + cot(far)),  x = h / tan(near)

    Returns {'height', 'near_dist', 'far_dist'} — near_dist/far_dist are
    each point's ground distance from the foot.
    """
    layout = spec["layout"]
    near_deg = float(spec["near_angle_deg"])
    far_deg = float(spec["far_angle_deg"])
    p = float(spec["distance_between"])
    cot_near = 1.0 / math.tan(math.radians(near_deg))
    cot_far = 1.0 / math.tan(math.radians(far_deg))

    if layout == "same_side":
        height = p / (cot_far - cot_near)
        near_dist = height * cot_near
        far_dist = near_dist + p
    else:  # opposite_sides
        height = p / (cot_near + cot_far)
        near_dist = height * cot_near
        far_dist = p - near_dist

    return {"height": height, "near_dist": near_dist, "far_dist": far_dist,
            "near_angle_deg": near_deg, "far_angle_deg": far_deg}


def verify_elevation_two_point(solved: dict) -> tuple:
    """Post-render check: confirms both tan() equations are satisfied by the
    solved (height, near_dist, far_dist) to within floating-point tolerance
    — the defining constraint of a valid two-point elevation diagram."""
    issues = []
    h = solved["height"]
    for label, dist, angle in (("near", solved["near_dist"], solved["near_angle_deg"]),
                                ("far", solved["far_dist"], solved["far_angle_deg"])):
        expected_h = dist * math.tan(math.radians(angle))
        if abs(expected_h - h) > 1e-6 * max(1.0, abs(h)):
            issues.append(f"{label} point: tan({angle}) * {dist} = {expected_h:.6f}, "
                          f"expected height {h:.6f}")
    if h <= 0:
        issues.append(f"solved height must be positive, got {h!r}")
    return (len(issues) == 0), issues


def _render_elevation_two_point(spec: dict) -> str:
    layout = spec["layout"]
    points = spec["points"]
    solved = _solve_elevation_two_point(spec)
    height, near_dist, far_dist = solved["height"], solved["near_dist"], solved["far_dist"]
    show_height = bool(spec.get("show_height", False))

    # math-space (y-up): foot at origin, target straight up, near/far along +x.
    foot_m = (0.0, 0.0)
    target_m = (0.0, height)
    if layout == "same_side":
        near_m = (near_dist, 0.0)
        far_m = (far_dist, 0.0)
    else:  # opposite_sides — near on +x, far on -x, foot between them
        near_m = (near_dist, 0.0)
        far_m = (-far_dist, 0.0)

    pts_m = [foot_m, target_m, near_m, far_m]
    xs = [p[0] for p in pts_m]
    ys = [p[1] for p in pts_m]
    span_x = max(max(xs) - min(xs), 1.0)
    span_y = max(max(ys) - min(ys), 1.0)
    scale = min((W - 2 * MARGIN) / span_x, (H - 2 * MARGIN) / span_y)
    min_x, min_y = min(xs), min(ys)

    def to_svg(p):
        return (MARGIN + (p[0] - min_x) * scale, (H - MARGIN) - (p[1] - min_y) * scale)

    foot, target, near, far = to_svg(foot_m), to_svg(target_m), to_svg(near_m), to_svg(far_m)

    body = []
    # ground line spanning both observation points
    gx1, gx2 = (min(near[0], far[0]) - 10), (max(near[0], far[0]) + 10)
    body.append(f'<line x1="{gx1:.1f}" y1="{foot[1]:.1f}" x2="{gx2:.1f}" y2="{foot[1]:.1f}" '
                f'stroke="{GROUND_COLOR}" stroke-width="1.4" data-role="ground-line"/>')
    # tower
    body.append(f'<line x1="{foot[0]:.1f}" y1="{foot[1]:.1f}" x2="{target[0]:.1f}" y2="{target[1]:.1f}" '
                f'stroke="{GROUND_COLOR}" stroke-width="2.4" data-role="elev-leg"/>')
    # right-angle mark at foot
    s = 8
    body.append(f'<polyline points="{foot[0]-s:.1f},{foot[1]:.1f} {foot[0]-s:.1f},{foot[1]-s:.1f} '
                f'{foot[0]:.1f},{foot[1]-s:.1f}" fill="none" stroke="{GROUND_COLOR}" '
                f'stroke-width="1.3" data-role="right-angle"/>')

    for (px, py), pid, angle_deg in ((near, points["near"], solved["near_angle_deg"]),
                                      (far, points["far"], solved["far_angle_deg"])):
        body.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{target[0]:.1f}" y2="{target[1]:.1f}" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.4" stroke-dasharray="4,3" '
                    f'data-role="sight-line"/>')
        ang_ground = 0.0 if px < target[0] else 180.0
        ang_sight = math.degrees(math.atan2(-(target[1] - py), target[0] - px))
        arc_r = 16
        body.append(f'<path d="{_arc_path((px, py), arc_r, ang_ground, ang_sight)}" fill="none" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.2" data-role="angle-arc"/>')
        lab_rad = math.radians((ang_ground + ang_sight) / 2)
        lx, ly = px + (arc_r + 11) * math.cos(lab_rad), py - (arc_r + 11) * math.sin(lab_rad)
        body.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="{AUX_COLOR}" '
                    f'{SMALL_FONT} data-role="angle-label">{angle_deg:g}&#176;</text>')
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{LINE_COLOR}"/>')
        body.append(f'<text x="{px:.1f}" y="{(py+16):.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                    f'{FONT} data-role="point-label">{_escape(pid)}</text>')

    for (px, py), pid in ((foot, points["foot"]), (target, points["target"])):
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{LINE_COLOR}"/>')
        dy = 16 if py < H / 2 else -8
        body.append(f'<text x="{px:.1f}" y="{(py+dy):.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                    f'{FONT} data-role="point-label">{_escape(pid)}</text>')

    dist_mid = ((near[0] + far[0]) / 2, foot[1] + 14)
    body.append(f'<text x="{dist_mid[0]:.1f}" y="{dist_mid[1]:.1f}" text-anchor="middle" '
                f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">'
                f'{_fmt_len(float(spec["distance_between"]))}</text>')
    if show_height:
        body.append(f'<text x="{(target[0]-8):.1f}" y="{((foot[1]+target[1])/2):.1f}" text-anchor="end" '
                    f'dominant-baseline="middle" fill="{AUX_COLOR}" {SMALL_FONT} '
                    f'data-role="dimension-label">{_fmt_len(height)}</text>')

    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:260px">' + "".join(body) + "</svg>")


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="elevation_two_point",
    renderer=_render_elevation_two_point,
    schema_check=_validate_elevation_two_point_spec,
    description="Height of a tower/building from TWO ground points with two different angles of "
                "elevation ('same_side' or 'opposite_sides' of the foot) — the two-simultaneous-"
                "equations NCERT problem family that the single-angle elevation_depression type "
                "cannot represent. Height and both ground distances are always solved from the "
                "two tan() equations, never estimated.",
))

diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="elevation_depression",
    renderer=_render_elevation_depression,
    schema_check=_validate_elevation_depression_spec,
    description="Angle-of-elevation / angle-of-depression right-triangle diagrams with an explicit "
                "horizontal reference and sight-line; both legs are always computed from the given "
                "angle + one known side via tan(), never estimated.",
))

diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="bearing",
    renderer=_render_bearing,
    schema_check=_validate_bearing_spec,
    description="Compass-bearing navigation/survey diagrams: a sequence of legs, each a bearing "
                "(0-360, clockwise from North) + distance from a named point; every point's position "
                "is computed trigonometrically from the chain of legs, never estimated.",
))
