"""
diagram_renderer.py — turns a structured diagram_spec (which shapes,
which points, which sides are marked equal, where the right angle is)
into an actual SVG diagram. Gemini never writes markup here — it only
supplies the geometric *relationships*; every coordinate, stroke, and
label position is computed deterministically below. This is what keeps
diagrams visually consistent across every question in every book.
"""
import math
import re

import geometry_solver
import diagram_plugin_registry
import label_layout
import math_sanitizer
import polynomial_division_plugin  # noqa: F401 — import-time side effect: registers "polynomial_long_division"
import transformation_geometry_plugin  # noqa: F401 — registers "transformation"
import function_graph_plugin  # noqa: F401 — registers "function_graph"
import probability_plugin  # noqa: F401 — registers "probability"
import set_theory_plugin  # noqa: F401 — registers "venn_diagram"
import elevation_bearing_plugin  # noqa: F401 — registers "elevation_depression", "bearing"
import unit_circle_plugin  # noqa: F401 — registers "unit_circle"
import circle_line_plugin  # noqa: F401 — registers "circle_line_intersection"
import solid_geometry_plugin  # noqa: F401 — registers "solid_net", "cross_section"
import circle_sector_plugin  # noqa: F401 — registers "circle_sector"
import composite_shaded_plugin  # noqa: F401 — registers "composite_shaded_region"
import rectilinear_composite_plugin  # noqa: F401 — registers "rectilinear_composite"
import successive_magnification_plugin  # noqa: F401 — registers "successive_magnification"

W, H = 260, 220
LINE_COLOR = "#1a4d8f"
LABEL_COLOR = "#111111"
FONT = "font-family='Hind Siliguri Regular' font-size='15' font-weight='600'"

# Matches a "চিত্ৰ 7.33" / "Figure 5.3" / "Fig. 8.21" style figure
# citation. Deliberately self-contained (not imported from utils.py)
# to keep this module's existing zero-dependency design intact.
_FIGURE_CITATION_PATTERN = re.compile(r"(চিত্ৰ|figure|fig\.?)\s*[\d০-৯]", re.IGNORECASE)


def _sanitize_extra_label(text: str) -> str:
    """Blocks a fake book-figure citation (e.g. "চিত্ৰ 7.33") from ever
    appearing in an AI-GENERATED diagram's caption.

    BUG FIX (found from a real generated PDF — user-reported,
    reproduced): the diagram_spec schema's own example for
    "extra_labels" used to be the literal word "চিত্ৰ" ("figure"),
    which invited Gemini to write an actual figure citation like
    "চিত্ৰ 7.33" into a GENERATED diagram's caption — making it look
    exactly like a verbatim textbook figure when it wasn't one at all.
    The schema example itself is fixed (see prompts.py), but this is a
    second, independent line of defense: even if a future prompt
    change or a model deviation reintroduces this pattern, the
    renderer itself now refuses to draw it, consistent with this
    project's established "never trust the model's exact output,
    defend at the renderer" philosophy applied everywhere else (point
    ids sent as one-element lists, coordinate text embedded in point
    labels, etc.)."""
    if _FIGURE_CITATION_PATTERN.search(text or ""):
        return ""
    return text


def _svg(body: str, width: int = W, height: int = H) -> str:
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:220px">{body}</svg>')


def _line(p1, p2, color=LINE_COLOR, width=2):
    return f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="{color}" stroke-width="{width}"/>'


def _label(point, text, dx=0, dy=-8):
    x, y = point[0] + dx, point[1] + dy
    return f'<text x="{x}" y="{y}" fill="{LABEL_COLOR}" {FONT}>{text}</text>'


def _tick_marks(p1, p2, count=1, role=None):
    """Small perpendicular tick(s) at the segment's midpoint — the
    standard textbook notation for 'these two sides are equal'.

    `role`, when given, adds a data-role="<role>" attribute to each tick
    <line> so newer renderers can count them precisely via
    verify_marker_coverage the same way coordinate_plot/angle/
    parallel_lines already do, instead of the older triangle renderer's
    stroke-width-substring-count heuristic (still used, unchanged, for
    "triangle" itself — see verify_marker_coverage's triangle branch).
    Defaults to None so every EXISTING caller's output is byte-for-byte
    unchanged (backward compatible)."""
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    perp = angle + math.pi / 2
    marks = []
    spacing = 5
    start_offset = -(count - 1) * spacing / 2
    role_attr = f' data-role="{role}"' if role else ""
    for i in range(count):
        offset = start_offset + i * spacing
        cx = mx + offset * math.cos(angle)
        cy = my + offset * math.sin(angle)
        x1 = cx + 5 * math.cos(perp)
        y1 = cy + 5 * math.sin(perp)
        x2 = cx - 5 * math.cos(perp)
        y2 = cy - 5 * math.sin(perp)
        marks.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{LINE_COLOR}" stroke-width="1.5"{role_attr}/>')
    return "".join(marks)


def _right_angle_box(vertex, dir1, dir2, size=10, role=None):
    """Small square marker showing a right angle at `vertex`, between the
    two directions (unit vectors) `dir1` and `dir2`. `role` behaves the
    same as in _tick_marks above — optional, backward compatible."""
    x, y = vertex
    x1, y1 = x + size * dir1[0], y + size * dir1[1]
    x2, y2 = x + size * (dir1[0] + dir2[0]), y + size * (dir1[1] + dir2[1])
    x3, y3 = x + size * dir2[0], y + size * dir2[1]
    role_attr = f' data-role="{role}"' if role else ""
    return (f'<polyline points="{x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f} {x3:.1f},{y3:.1f}" '
            f'fill="none" stroke="{LINE_COLOR}" stroke-width="1.5"{role_attr}/>')


def _unit(p_from, p_to):
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    dist = math.hypot(dx, dy) or 1
    return (dx / dist, dy / dist)


def _tick_count_for(spec: dict, side_key: str) -> int:
    """A side marked equal in more than one equal_marks group gets extra
    ticks — mirrors how textbooks distinguish AB=AC (single tick) from a
    second, unrelated equal pair (double tick) in the same figure."""
    groups = spec.get("equal_marks", []) or []
    count = 0
    for i, group in enumerate(groups):
        if side_key in group:
            count = i + 1
    return max(count, 1)


def _points_by_id(points):
    return {p["id"]: p for p in points}


# ------------------------------------------------------------------
# TRIANGLE (covers the large majority of Class 9 geometry proofs:
# congruence, isosceles proofs, medians, perpendicular bisectors)
# ------------------------------------------------------------------
def _foot_of_perpendicular(p, a, b):
    """Returns the point on line segment a-b closest to p (the standard
    vector-projection foot of a perpendicular), clamped to the segment
    itself so a foot is never drawn sailing off past the segment's own
    endpoints even for an obtuse triangle."""
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return a
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    return (ax + t * abx, ay + t * aby)


def _render_triangle(spec: dict) -> str:
    points = spec.get("points", [])
    ids = [p["id"] for p in points]
    n = len(ids)

    # ---- GEOMETRY SOLVER INTEGRATION (Phase 1) ----
    # If the question stated actual measurements (side_lengths and/or
    # angles_deg — both OPTIONAL, additive fields; see geometry_solver.py
    # and prompts.py's schema), solve for real, to-scale vertex
    # positions instead of the fixed apex/base_left/base_right layout
    # below. `solved` is None for every spec that doesn't populate
    # those new fields — i.e. every spec generated before this change —
    # in which case every line below runs EXACTLY as it always has.
    # This is the actual backward-compatibility guarantee: old behavior
    # IS the `solved is None` branch, not a separately preserved copy.
    solved = None
    solve_reason = "no side_lengths/angles_deg on this spec"
    if n >= 3:
        solved, solve_reason = geometry_solver.solve_triangle(
            ids[:3],
            side_lengths=spec.get("side_lengths"),
            angles_deg=spec.get("angles_deg"),
            right_angle_at=(spec.get("right_angle_at") if isinstance(spec.get("right_angle_at"), str)
                             else (spec.get("right_angle_at")[0] if spec.get("right_angle_at") else None)),
            canvas=(W, H),
        )
        if solved is not None:
            from utils import logger
            logger.info(f"📐 Triangle {ids[:3]}: geometry solver produced to-scale coordinates ({solve_reason}).")
        elif spec.get("side_lengths") or spec.get("angles_deg"):
            # measurements WERE given but couldn't be solved (under-
            # constrained combination, or self-contradictory — e.g. a
            # stated right_angle_at that doesn't match the stated
            # sides). Log why, then fall back to the schematic layout
            # below exactly as if no measurements had been given —
            # "never guess" applies to the solver's own output too.
            from utils import logger
            logger.warning(f"⚠️ Triangle {ids[:3]}: geometry solver could not use the given "
                            f"measurements ({solve_reason}) — falling back to schematic placement.")

    apex = (W / 2, 25)
    base_left = (40, H - 35)
    base_right = (W - 40, H - 35)
    coords = {}

    if n >= 3:
        if solved is not None:
            coords[ids[0]] = solved[ids[0]]
            coords[ids[1]] = solved[ids[1]]
            coords[ids[2]] = solved[ids[2]]
            apex, base_left, base_right = coords[ids[0]], coords[ids[1]], coords[ids[2]]
        else:
            coords[ids[0]] = apex
            coords[ids[1]] = base_left
            coords[ids[2]] = base_right

    body = []
    if n >= 3:
        body.append(_line(apex, base_left))
        body.append(_line(apex, base_right))
        body.append(_line(base_left, base_right))

    # ALTITUDES — a perpendicular dropped from a vertex onto the
    # OPPOSITE side (not necessarily the base). Each altitude's own
    # foot id gets its EXACT geometric position (real perpendicular
    # projection onto that side's line, clamped to the segment) rather
    # than the old "just space extra points evenly along the base"
    # placeholder, which was geometrically meaningless for anything
    # other than a cevian from the apex to the base — e.g. it could
    # never correctly represent "altitude BE⟂AC" (foot E lies on AC,
    # not on the base BC at all).
    altitude_foot_ids = set()
    for alt in spec.get("altitudes", []) or []:
        frm = alt.get("from")
        to_side = alt.get("to_side") or alt.get("to") or ""
        foot = alt.get("foot")
        if not (frm and foot and len(to_side) == 2):
            continue
        s0, s1 = to_side[0], to_side[1]
        if frm not in coords or s0 not in coords or s1 not in coords:
            continue
        foot_xy = _foot_of_perpendicular(coords[frm], coords[s0], coords[s1])
        coords[foot] = foot_xy
        altitude_foot_ids.add(foot)
        body.append(_line(coords[frm], foot_xy, color="#c0392b"))
        # right-angle mark at the foot, between the direction back up
        # to the vertex the altitude came from and along the side it
        # landed on
        d1 = _unit(foot_xy, coords[frm])
        d2 = _unit(foot_xy, coords[s1] if foot_xy != coords[s1] else coords[s0])
        body.append(_right_angle_box(foot_xy, d1, d2))

    # CEVIANS — a general (not-necessarily-perpendicular) line from a
    # vertex to a point on one of the OTHER two sides — for angle
    # bisectors, medians, and other constructions "altitudes" can't
    # represent (an altitude always draws a right-angle mark at its
    # foot, which would be flatly wrong for a bisector/median foot).
    #
    # BUG FIX (found from a real generated PDF — user-reported,
    # reproduced): a proof about the angle bisectors from B and C
    # meeting at an interior point O was rendered with points D/E
    # force-placed on the BASE with cevians drawn from the APEX — the
    # only fallback this renderer had for any point beyond the first
    # 3 — which doesn't match a bisector-from-B/bisector-from-C
    # construction at all (their feet lie on the two other sides, not
    # the base, and the cevians originate from B and C, not A). This
    # diagram is inherently schematic (not to-scale, like the rest of
    # this renderer), so each foot is placed at the midpoint of its
    # target side — a defensible, deterministic choice that correctly
    # shows WHICH side the cevian lands on and WHERE it starts from,
    # which is what these proofs actually depend on geometrically.
    cevian_foot_ids = set()
    cevian_segments = {}  # foot_id -> (from_vertex_id, to_side) for later use
    for cev in spec.get("cevians", []) or []:
        frm = cev.get("from")
        to_side = cev.get("to_side") or cev.get("to") or ""
        foot = cev.get("foot")
        if not (frm and foot and len(to_side) == 2):
            continue
        s0, s1 = to_side[0], to_side[1]
        if frm not in coords or s0 not in coords or s1 not in coords:
            continue
        p0, p1 = coords[s0], coords[s1]
        foot_xy = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        coords[foot] = foot_xy
        cevian_foot_ids.add(foot)
        cevian_segments[foot] = (frm, foot_xy)
        body.append(_line(coords[frm], foot_xy, color="#c0392b"))

    # Labeled intersection of exactly two cevians (e.g. "O, where the
    # two angle bisectors meet") — computed as the REAL geometric
    # intersection of the two cevian segments actually drawn above, so
    # it's always visually consistent with them, never a separate
    # guessed position.
    intersection_label = spec.get("cevian_intersection_label")
    valid_cevians = list(cevian_segments.values())
    intersection_xy = None
    if intersection_label and len(valid_cevians) == 2:
        (from1, foot1), (from2, foot2) = valid_cevians
        intersection_xy = _segment_intersection(coords[from1], foot1, coords[from2], foot2)
        if intersection_xy is not None:
            ix, iy = intersection_xy
            body.append(f'<circle cx="{ix:.1f}" cy="{iy:.1f}" r="2.5" fill="{LINE_COLOR}" data-role="cevian-intersection"/>')
            body.append(_label((ix, iy), str(intersection_label), dx=8, dy=-8))

            # OPTIONAL: join a named vertex to this intersection point
            # (e.g. "join A to O with a line") — only drawn once the
            # intersection point itself has an actual position.
            join_vertex = spec.get("join_vertex_to_intersection")
            if join_vertex and join_vertex in coords:
                body.append(f'<line x1="{coords[join_vertex][0]:.1f}" y1="{coords[join_vertex][1]:.1f}" '
                            f'x2="{ix:.1f}" y2="{iy:.1f}" stroke="{LINE_COLOR}" stroke-width="1.5" '
                            f'data-role="intersection-join"/>')

    # any further points NOT already placed by an altitude or a cevian
    # are treated as lying on the base segment, evenly spaced (generic
    # cevian feet / bisector points with no more specific placement
    # given) — with a cevian drawn from the apex to each.
    extra_ids = [pid for pid in ids[3:] if pid not in altitude_foot_ids and pid not in cevian_foot_ids]
    for i, pid in enumerate(extra_ids, start=1):
        t = i / (len(extra_ids) + 1)
        x = base_left[0] + t * (base_right[0] - base_left[0])
        coords[pid] = (x, base_left[1])
        body.append(_line(apex, coords[pid]))

    equal_groups = spec.get("equal_marks", []) or []
    for group in equal_groups:
        for side in group:
            if len(side) == 2 and side[0] in coords and side[1] in coords:
                body.append(_tick_marks(coords[side[0]], coords[side[1]], _tick_count_for(spec, side)))

    # right_angle_at: a right angle AT one of the figure's own points
    # (typically a main vertex — e.g. "right-angled at B") that is NOT
    # already an altitude foot (those get their mark automatically,
    # above). Accepts a single id (legacy) or a list of ids, since a
    # figure can need more than one such mark.
    right_angle_ids = spec.get("right_angle_at")
    if isinstance(right_angle_ids, str):
        right_angle_ids = [right_angle_ids]
    for rid in right_angle_ids or []:
        if rid in altitude_foot_ids or rid not in coords:
            continue
        v = coords[rid]
        neighbours = [c for pid, c in coords.items() if pid != rid]
        if len(neighbours) >= 2:
            d1 = _unit(v, neighbours[0])
            d2 = _unit(v, neighbours[-1])
            body.append(_right_angle_box(v, d1, d2))

    for pid, (x, y) in coords.items():
        dy = -10 if y < H / 2 else 16
        body.append(_label((x, y), pid, dy=dy))
        body.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{LINE_COLOR}"/>')

    for extra in spec.get("extra_labels", []) or []:
        extra = _sanitize_extra_label(extra)
        if not extra:
            continue
        body.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" fill="#888" font-size="11">{extra}</text>')

    return _svg("".join(body))


# ------------------------------------------------------------------
# CIRCLE
# ------------------------------------------------------------------
def _render_circle(spec: dict) -> str:
    cx, cy, r = W / 2, H / 2, min(W, H) / 2 - 25
    points = spec.get("points", [])

    # ------------------------------------------------------------------
    # PRODUCTION-AUDIT FIX: the circle diagram type previously had NO
    # concept of a "center" point at all — every declared point,
    # including one meant to represent the circle's center (the single
    # most common phrase in nearly every real circle question: "O is
    # the centre of a circle..."), was placed evenly spaced AROUND the
    # circumference like every other point. A center point was
    # therefore drawn ON the circle's edge instead of at its middle —
    # a wrong, misleading diagram for the most common circle-question
    # pattern in the curriculum. Confirmed via a real rendered PDF.
    #
    # Fix: an OPTIONAL "center_id" field names which declared point (if
    # any) is the center; that point renders at the true center, and
    # every other point is spaced around the circumference as before.
    # Absent center_id, behavior is 100% unchanged (every point on the
    # circumference, exactly as it always has been) — this is strictly
    # additive.
    # ------------------------------------------------------------------
    center_id = spec.get("center_id")
    center_id = center_id if isinstance(center_id, str) and any(p.get("id") == center_id for p in points) else None

    tangent = spec.get("tangent_from")
    tangent_special_ids = set()
    dist_ratio = None
    if center_id and isinstance(tangent, dict):
        ext_id = tangent.get("external_point")
        if ext_id and any(p.get("id") == ext_id for p in points):
            tangent_special_ids.add(ext_id)
            for tid in (tangent.get("tangent_points") or [])[:2]:
                if tid:
                    tangent_special_ids.add(tid)
            dist_ratio = 1.8
            dims = spec.get("dimensions")
            if isinstance(dims, dict) and _is_number(dims.get("distance")) and _is_number(dims.get("radius")):
                real_r = float(dims["radius"])
                real_dist = float(dims["distance"])
                if real_r > 0 and real_dist > real_r:
                    dist_ratio = real_dist / real_r
                elif real_r > 0:
                    # PRODUCTION-AUDIT FIX: distance <= radius means the
                    # "external" point would actually be ON or INSIDE the
                    # circle -- no tangent line can exist. Silently
                    # clamping this to "just barely outside" would draw a
                    # confident-looking tangent construction that
                    # contradicts the question's own given numbers (a
                    # mis-transcribed or self-contradictory problem).
                    # Decline the tangent construction entirely instead —
                    # same "never draw something the numbers don't
                    # support" rule geometry_solver.py already follows —
                    # excluded_ids below is now empty for these ids, so
                    # they fall back to plain circumference placement.
                    from utils import logger
                    logger.warning(f"⚠️ Circle tangent construction: given distance ({real_dist}) is not "
                                    f"greater than the given radius ({real_r}) — a tangent from this point "
                                    f"is mathematically impossible, so this construction is skipped.")
                    dist_ratio = None
                    tangent_special_ids = set()

    # PRODUCTION-AUDIT FIX: a realistic radius:distance ratio (e.g. the
    # very common 5:13 right-triangle case) previously placed the
    # external point — and therefore its tangent lines/points — well
    # outside the 260x220 canvas entirely (confirmed: one real case
    # landed at y=-46, x=286, both off-canvas). check_valid_viewbox
    # only proves the viewBox ATTRIBUTE is well-formed, not that every
    # drawn element is actually inside it, so this shipped silently.
    # Fix: when a tangent construction is present, the circle's own
    # schematic radius is capped so that center-to-external-point
    # distance (r * dist_ratio) always fits inside the canvas with
    # margin — i.e. the WHOLE construction is scaled to fit, rather
    # than drawing the circle at a fixed size and letting the external
    # point overflow whatever that implies.
    if dist_ratio:
        max_half_extent = min(W, H) / 2 - 20
        r = min(r, max_half_extent / dist_ratio)

    excluded_ids = {center_id} | tangent_special_ids
    circumference_points = [p for p in points if p.get("id") not in excluded_ids]
    n = max(len(circumference_points), 1)
    body = [f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{LINE_COLOR}" stroke-width="2"/>']

    coords = {}
    if center_id:
        coords[center_id] = (cx, cy)
    for i, p in enumerate(circumference_points):
        angle = -math.pi / 2 + (2 * math.pi * i / n)
        x, y = cx + r * math.cos(angle), cy + r * math.sin(angle)
        coords[p["id"]] = (x, y)

    for pid, (x, y) in coords.items():
        body.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{LINE_COLOR}"/>')
        body.append(_label((x, y), pid, dy=-10 if y < cy else 16))

    # chords/radii between consecutive-and-all pairs (typical for
    # inscribed-angle/chord questions; when center_id is set, each
    # center-to-circumference pair drawn here is exactly a radius line,
    # which is the correct, desired construction — no special-casing
    # needed, the existing all-pairs logic already produces it).
    ids = list(coords.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            body.append(_line(coords[ids[i]], coords[ids[j]], width=1.5))

    # ------------------------------------------------------------------
    # OPTIONAL: tangent-from-an-external-point construction — the other
    # extremely common real circle-question pattern this diagram type
    # previously had no way to represent at all ("P is a point outside
    # the circle... find the length of the tangent PA"). Gated behind
    # its own explicit field so it never interferes with the plain
    # points-on-a-circle case above. Requires center_id (a tangent is
    # only meaningful relative to a real center).
    # ------------------------------------------------------------------
    if center_id and isinstance(tangent, dict):
        ext_id = tangent.get("external_point")
        tangent_ids = tangent.get("tangent_points") or []
        ext_declared = any(p.get("id") == ext_id for p in points) if ext_id else False
        if ext_id and ext_declared and ext_id not in coords and len(tangent_ids) >= 1 and dist_ratio:
            ext_dist = r * dist_ratio
            # place P along a fixed, visually clean direction (upper-right)
            ext_angle = -math.pi / 4
            px, py = cx + ext_dist * math.cos(ext_angle), cy + ext_dist * math.sin(ext_angle)
            coords[ext_id] = (px, py)
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{LINE_COLOR}"/>')
            body.append(_label((px, py), ext_id, dy=-10 if py < cy else 16))
            body.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.2" stroke-dasharray="3,3" data-role="tangent-op"/>')

            # Real Thales'-theorem tangent-point construction: the two
            # tangent points from P are exactly where the circle with
            # diameter OP intersects the original circle — reusing the
            # SAME _circle_intersection helper the construction diagram
            # type already relies on, just called for both roots.
            mid = ((cx + px) / 2, (cy + py) / 2)
            mid_r = math.hypot(px - cx, py - cy) / 2
            t1 = _circle_intersection((cx, cy), r, mid, mid_r)
            # second root: reflect across the O-P line
            t2 = None
            if t1:
                opx, opy = px - cx, py - cy
                op_len_sq = opx * opx + opy * opy
                if op_len_sq > 0:
                    t1x, t1y = t1[0] - cx, t1[1] - cy
                    proj = (t1x * opx + t1y * opy) / op_len_sq
                    foot = (cx + proj * opx, cy + proj * opy)
                    t2 = (2 * foot[0] - t1[0], 2 * foot[1] - t1[1])

            drawn_tangent_ids = [tid for tid in tangent_ids[:2] if tid]
            for tid, tpoint in zip(drawn_tangent_ids, [t1, t2]):
                if tpoint is None or tid in coords:
                    continue
                coords[tid] = tpoint
                tx, ty = tpoint
                body.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="2.5" fill="{LINE_COLOR}"/>')
                body.append(_label((tx, ty), tid, dy=-10 if ty < cy else 16))
                body.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
                            f'stroke="{LINE_COLOR}" stroke-width="1.5" data-role="tangent-line"/>')
                body.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
                            f'stroke="{LINE_COLOR}" stroke-width="1.2" data-role="tangent-radius"/>')
                # right-angle mark: tangent ⟂ radius at the tangent point (a core
                # textbook convention -- the diagram must show WHY the triangle
                # formed is right-angled, not just assert it)
                dir_to_center = _unit(tpoint, (cx, cy))
                dir_to_ext = _unit(tpoint, (px, py))
                body.append(_right_angle_box(tpoint, dir_to_center, dir_to_ext, size=8, role="tangent-right-angle"))

    return _svg("".join(body))


# ------------------------------------------------------------------
# ANGLE
# ------------------------------------------------------------------
# The previous renderer drew one hard-coded pair of rays at a fixed
# fake angle and one hard-coded arc, regardless of what the question
# actually said — every "angle" diagram looked identical whether the
# question was about a single 40° angle, a linear pair, vertically
# opposite angles, or an angle bisector. Root-caused during the
# renderer audit; replaced with a real construction driven by each
# ray's actual direction (in degrees, 0 = pointing right, increasing
# counter-clockwise as on paper) so the arcs and rays drawn actually
# match the question's own numbers.
#
# direction_deg uses the standard on-paper convention; SVG's y-axis
# points down, so converting to screen coordinates negates the angle.
def _angle_dir_to_xy(vertex, direction_deg, length):
    rad = -math.radians(direction_deg)
    return (vertex[0] + length * math.cos(rad), vertex[1] + length * math.sin(rad))


def _render_angle(spec: dict) -> str:
    vertex_id = spec.get("vertex")
    rays = spec.get("rays", []) or []
    ray_dirs = {}
    for r in rays:
        if not isinstance(r, dict):
            continue
        rid = _coerce_id(r.get("id"))
        try:
            deg = float(r.get("direction_deg"))
        except (TypeError, ValueError):
            continue
        if rid:
            ray_dirs[rid] = deg % 360

    cx, cy = W / 2, H / 2 + 8
    vertex = (cx, cy)
    ray_len = 82

    body = []
    for rid, deg in ray_dirs.items():
        end = _angle_dir_to_xy(vertex, deg, ray_len)
        body.append(f'<line x1="{vertex[0]:.1f}" y1="{vertex[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="ray"/>')
        label_pt = _angle_dir_to_xy(vertex, deg, ray_len + 12)
        body.append(_label(label_pt, rid, dx=-4, dy=0))
        body.append(f'<circle cx="{end[0]:.1f}" cy="{end[1]:.1f}" r="2" fill="{LINE_COLOR}"/>')

    body.append(f'<circle cx="{vertex[0]:.1f}" cy="{vertex[1]:.1f}" r="2.5" fill="{LINE_COLOR}"/>')
    body.append(_label(vertex, _coerce_id(vertex_id) or "O", dx=-14, dy=14))

    # angle_marks: one arc per declared angle, nested at increasing
    # radius (by list order) so several angles sharing the same vertex
    # (a linear pair, angles all the way around a point, ...) don't
    # draw their arcs on top of one another.
    for i, mark in enumerate(spec.get("angle_marks", []) or []):
        between = mark.get("between") or []
        if len(between) != 2 or between[0] not in ray_dirs or between[1] not in ray_dirs:
            continue
        d1, d2 = ray_dirs[between[0]], ray_dirs[between[1]]
        # always sweep the SHORTER way round from d1 to d2
        diff = (d2 - d1) % 360
        if diff > 180:
            d1, d2 = d2, d1
            diff = 360 - diff
        arc_r = 20 + 11 * i
        p1 = _angle_dir_to_xy(vertex, d1, arc_r)
        p2 = _angle_dir_to_xy(vertex, d1 + diff, arc_r)
        large_arc = 1 if diff > 180 else 0
        body.append(f'<path d="M {p1[0]:.1f},{p1[1]:.1f} A {arc_r},{arc_r} 0 {large_arc},0 '
                    f'{p2[0]:.1f},{p2[1]:.1f}" fill="none" stroke="{LINE_COLOR}" '
                    f'stroke-width="1.3" data-role="angle-arc"/>')
        label = mark.get("label")
        if label:
            mid_pt = _angle_dir_to_xy(vertex, d1 + diff / 2, arc_r + 13)
            body.append(_label(mid_pt, str(label), dx=-8, dy=0))

    # OPTIONAL bisector: a dashed ray splitting angle_marks-independent
    # pair "of" exactly down the middle, landing on a new labelled
    # point "to" (not one of the original rays) — the direction is the
    # arithmetic mean of the two ray directions (shorter-arc side).
    bisector = spec.get("bisector")
    if isinstance(bisector, dict):
        of = bisector.get("of") or []
        to_label = _coerce_id(bisector.get("to"))
        if len(of) == 2 and of[0] in ray_dirs and of[1] in ray_dirs and to_label:
            d1, d2 = ray_dirs[of[0]], ray_dirs[of[1]]
            diff = (d2 - d1) % 360
            if diff > 180:
                d1 = d2
                diff = 360 - diff
            mid_deg = d1 + diff / 2
            end = _angle_dir_to_xy(vertex, mid_deg, ray_len)
            body.append(f'<line x1="{vertex[0]:.1f}" y1="{vertex[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.6" stroke-dasharray="5,3" data-role="bisector"/>')
            label_pt = _angle_dir_to_xy(vertex, mid_deg, ray_len + 12)
            body.append(_label(label_pt, to_label, dx=-6, dy=0))

    return _svg("".join(body))


# ------------------------------------------------------------------
# PARALLEL LINES
# ------------------------------------------------------------------
# The previous renderer drew two fixed horizontal lines and one fixed
# transversal with no angle information at all — every parallel-lines
# question (corresponding angles, alternate angles, co-interior
# angles, "find x") produced the exact same picture with no numbers on
# it. Replaced with a construction that actually places a label in the
# correct one of the 4 angle "quadrants" at each line's crossing with
# the transversal, which is what these questions are actually about.
def _render_parallel_lines(spec: dict) -> str:
    y1, y2 = 58, H - 58
    x_left, x_right = 22, W - 22
    # transversal drawn as a slanted line so all 4 angles at each
    # crossing are visually distinguishable (a vertical transversal
    # would make left/right pairs look identical)
    t_top, t_bottom = (46, 18), (W - 46, H - 18)
    cross_x1 = t_top[0] + (t_bottom[0] - t_top[0]) * (y1 - t_top[1]) / (t_bottom[1] - t_top[1])
    cross_x2 = t_top[0] + (t_bottom[0] - t_top[0]) * (y2 - t_top[1]) / (t_bottom[1] - t_top[1])
    cross1, cross2 = (cross_x1, y1), (cross_x2, y2)

    body = [
        _line((x_left, y1), (x_right, y1)),
        _line((x_left, y2), (x_right, y2)),
        _line(t_top, t_bottom, color=AUX_COLOR),
    ]
    # parallel-mark arrowheads (">") on both lines, at their own
    # right-hand end — the standard "these two lines are parallel"
    # textbook notation, independent of any angle_marks data.
    for y in (y1, y2):
        body.append(f'<polygon points="{x_right-10},{y-5} {x_right},{y} {x_right-10},{y+5}" '
                    f'fill="{LINE_COLOR}" data-role="parallel-tick"/>')

    lines = spec.get("lines") or []
    line_ids = [_coerce_id(v) for v in lines if _coerce_id(v)]
    id1 = line_ids[0] if len(line_ids) > 0 else "l1"
    id2 = line_ids[1] if len(line_ids) > 1 else "l2"
    body.append(_label((x_left, y1), id1, dx=-14, dy=4))
    body.append(_label((x_left, y2), id2, dx=-14, dy=4))
    t_label = spec.get("transversal_label")
    if t_label:
        body.append(_label(t_bottom, str(t_label), dx=6, dy=6))

    # angle_marks: each mark names which line it's at and which of the
    # 4 quadrants around that crossing (top_left/top_right/bottom_left/
    # bottom_right, in the usual sense of "above/below the horizontal
    # line, left/right of the transversal") — the label is placed
    # directly in that quadrant, near the crossing point.
    quadrant_offsets = {
        "top_left": (-24, -10),
        "top_right": (14, -10),
        "bottom_left": (-24, 16),
        "bottom_right": (14, 16),
    }
    for mark in spec.get("angle_marks", []) or []:
        line_id = _coerce_id(mark.get("line"))
        position = mark.get("position")
        label = mark.get("label")
        if line_id not in (id1, id2) or position not in quadrant_offsets or not label:
            continue
        cross = cross1 if line_id == id1 else cross2
        dx, dy = quadrant_offsets[position]
        body.append(f'<text x="{cross[0] + dx:.1f}" y="{cross[1] + dy:.1f}" fill="{AUX_COLOR}" '
                    f'{FONT} data-role="angle-label">{label}</text>')

    return _svg("".join(body))


# ------------------------------------------------------------------
# COORDINATE GEOMETRY (diagram_type "coordinate_plot")
# ------------------------------------------------------------------
# Canvas is larger than the other renderers' 260x220 — a Cartesian grid
# with axis labels, distance annotations, and dashed projections needs
# more room to stay legible at print size, and this renderer is the
# only one that scales its own coordinate system rather than using
# fixed geometric positions.
CW, CH = 320, 280
GRID_COLOR = "#c8d3e0"
AUX_COLOR = "#c0392b"   # dashed projections / distance labels / midpoints —
                        # the same "construction" red used by number_line and
                        # the triangle renderer's altitudes, so a student learns
                        # one consistent colour convention across every diagram
                        # type in the book, not a different palette per renderer.


def _cg_coerce_id(value) -> str:
    """Local alias — defined before _coerce_id() below so this section
    can be read top-to-bottom independent of definition order elsewhere
    in the file; delegates to the same coercion once _coerce_id exists."""
    if isinstance(value, list):
        return "".join(str(v) for v in value) if value else ""
    return str(value) if value is not None else ""


def _coord_point_specs(spec: dict) -> list[dict]:
    """Every declared point that has a usable numeric (x, y) — anything
    without one is dropped (defensively) rather than crashing the whole
    diagram, same policy as every other renderer in this file."""
    out = []
    for p in spec.get("points", []) or []:
        if not isinstance(p, dict):
            continue
        try:
            x, y = float(p.get("x")), float(p.get("y"))
        except (TypeError, ValueError):
            continue
        out.append({
            "id": _cg_coerce_id(p.get("id")),
            "label": str(p.get("label", p.get("id", ""))),
            "x": x, "y": y,
            "show_projection": bool(p.get("show_projection", False)),
        })
    return out


def _get_nice_steps(range_min, range_max, max_ticks=8):
    """Selects a 'nice' step size for axis ticks (e.g., 1, 2, 5, 10, 20)
    that results in a reasonable number of ticks for the given range."""
    span = range_max - range_min
    if span == 0:
        return [float(range_min)]
    raw_step = span / max_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    # BUG FIX (V8 polish): use a wider set of multipliers to find a step
    # that produces FEWER ticks on dense plots, avoiding overlap.
    for mult in (1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        step = mult * magnitude
        if span / step <= max_ticks: break
    start = math.floor(range_min / step) * step
    return [v for v in (start + i * step for i in range(int(max_ticks * 2))) if range_min - 1e-9 <= v <= range_max + 1e-9]


def _render_coordinate_plot(spec: dict) -> str:
    """Plots each declared point at its ACTUAL (x, y) coordinate, to
    scale, on a labelled Cartesian grid — this is the Case B fallback
    renderer (used only when no original book figure exists/matched for
    a coordinate-geometry question; see book_diagram_extractor.py and
    solver.py's _attach_book_diagrams for the Case A path that should
    be preferred whenever the book already has the figure).

    Beyond exact-scale point plotting (the original fix here), this now
    supports the full coordinate-geometry feature set: a full-canvas
    grid, arrowed axes that actually extend in all four directions
    (x/x'/y/y'), line SEGMENTS between two declared points (optionally
    dashed, with an optional midpoint marker and/or a computed distance
    label), and full straight LINES through two declared points
    extended to the edge of the plot (for "find the equation of the
    line through..." style questions). See DIAGRAM_SPEC_SCHEMA in
    prompts.py for the exact spec shape Gemini is asked to produce.

    Every auxiliary element (projection, midpoint, distance label,
    line) is tagged with a `data-role="..."` attribute purely so
    verify_marker_coverage() below can count, after rendering, whether
    every relationship the spec asked for actually made it into the
    final SVG — the same "don't just trust the renderer ran" policy
    already used for the triangle renderer's tick/right-angle counts.
    """
    coords = _coord_point_specs(spec)
    by_id = {p["id"]: p for p in coords if p["id"]}

    xs = [p["x"] for p in coords] or [0.0]
    ys = [p["y"] for p in coords] or [0.0]
    # BUG FIX (found via diagram_final_check.py's post-render clipping
    # check, run against a real coordinate_plot spec whose points never
    # crossed y=0 — e.g. P(2,3), Q(-1,1)): the origin must ALWAYS be
    # part of the plotted range, exactly like a real textbook Cartesian
    # plane always shows both axes crossing at O regardless of where
    # the actual data points happen to sit. Without this, min_y/max_y
    # were derived from the data alone, so the origin (where the drawn
    # x-axis and its arrowheads/labels/"O" live) could land well below
    # the canvas — a genuinely clipped axis, not a data point.
    min_x, max_x = min(min(xs), 0.0), max(max(xs), 0.0)
    min_y, max_y = min(min(ys), 0.0), max(max(ys), 0.0)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    # Dynamic canvas sizing based on data aspect ratio.
    aspect_ratio = span_x / span_y if span_y > 0 else 1
    canvas_w, canvas_h = spec.get("canvas_size", [320, 280])
    if aspect_ratio > 1: # wider than tall
        canvas_h = max(180, int(canvas_w / aspect_ratio))
    else: # taller than wide
        canvas_w = max(240, int(canvas_h * aspect_ratio))

    pad = 40
    plot_w, plot_h = canvas_w - 2 * pad, canvas_h - 2 * pad

    scale_x = plot_w / span_x
    scale_y = plot_h / span_y
    ox, oy = pad - min_x * scale_x, canvas_h - pad + min_y * scale_y

    def to_svg(x, y):
        return ox + x * scale_x, oy - y * scale_y

    body = []

    # --- full background grid (faint), one line per integer step, the
    # whole way across the canvas — a real textbook grid, not just tick
    # marks at the axes. Uses "nice" steps.
    x_ticks = _get_nice_steps(min_x, max_x, 5)
    y_ticks = _get_nice_steps(min_y, max_y, 5)

    for val in x_ticks:
        gx, _ = to_svg(val, 0)
        if pad - 1 <= gx <= canvas_w - pad + 1:
            body.append(f'<line x1="{gx:.1f}" y1="{pad}" x2="{gx:.1f}" y2="{canvas_h - pad}" '
                        f'stroke="{GRID_COLOR}" stroke-width="1"/>')
    for val in y_ticks:
        _, gy = to_svg(0, val)
        if pad - 1 <= gy <= canvas_h - pad + 1:
            body.append(f'<line x1="{pad}" y1="{gy:.1f}" x2="{canvas_w - pad}" y2="{gy:.1f}" '
                        f'stroke="{GRID_COLOR}" stroke-width="1"/>')

    # --- axes, extended to the canvas edge with arrowheads at BOTH ends
    # of each (x' <-> x and y' <-> y) — the standard NCERT/SEBA
    # coordinate-geometry convention that the axes extend indefinitely
    # in both directions.
    def _arrow(tip, direction):
        ang = math.atan2(direction[1], direction[0])
        a1, a2 = ang + math.radians(150), ang - math.radians(150)
        p1 = (tip[0] + 7 * math.cos(a1), tip[1] + 7 * math.sin(a1))
        p2 = (tip[0] + 7 * math.cos(a2), tip[1] + 7 * math.sin(a2))
        return f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{LINE_COLOR}"/>'

    x_axis_l, x_axis_r = (pad - 10, oy), (canvas_w - pad + 10, oy)
    y_axis_t, y_axis_b = (ox, pad - 10), (ox, canvas_h - pad + 10)
    body.append(_line(x_axis_l, x_axis_r))
    body.append(_line(y_axis_t, y_axis_b))
    body.append(_arrow(x_axis_r, (1, 0)))
    body.append(_arrow(x_axis_l, (-1, 0)))
    body.append(_arrow(y_axis_t, (0, -1)))
    body.append(_arrow(y_axis_b, (0, 1)))
    body.append(_label((x_axis_r[0] - 4, x_axis_r[1]), "x", dx=6, dy=-6))
    body.append(_label((x_axis_l[0] + 4, x_axis_l[1]), "x'", dx=-14, dy=-6))
    body.append(_label((y_axis_t[0], y_axis_t[1] + 4), "y", dx=8, dy=2))
    body.append(_label((y_axis_b[0], y_axis_b[1] - 4), "y'", dx=8, dy=10))
    body.append(_label((ox, oy), "O", dx=-14, dy=14))

    # --- integer ticks + numeric labels on both axes (the actual scale)
    for val in x_ticks:
        if val == 0: continue
        tx, _ = to_svg(val, 0)
        body.append(f'<line x1="{tx:.1f}" y1="{oy - 3}" x2="{tx:.1f}" y2="{oy + 3}" stroke="{LINE_COLOR}" stroke-width="1"/>')
        body.append(f'<text x="{tx:.1f}" y="{oy + 14}" text-anchor="middle" fill="#555" font-family="Noto Sans, Arial" font-size="9">{val:g}</text>')
    for val in y_ticks:
        if val == 0: continue
        _, ty = to_svg(0, val)
        body.append(f'<line x1="{ox - 3}" y1="{ty:.1f}" x2="{ox + 3}" y2="{ty:.1f}" stroke="{LINE_COLOR}" stroke-width="1"/>')
        body.append(f'<text x="{ox - 7}" y="{ty:.1f}" text-anchor="end" dominant-baseline="middle" fill="#555" font-family="Noto Sans, Arial" font-size="9">{val:g}</text>')

    # --- full straight LINES through two declared points, extended to
    # the plot's own boundary (for "find the equation of the line
    # through A and B" / "show A, B, C are collinear" style questions).
    def _param_clip(p1, p2):
        """Extend the infinite line through svg-space p1,p2 out to the
        plot rectangle's boundary (Liang-Barsky clipping against the
        plot's pad rectangle), returning the two clipped endpoints."""
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        t0, t1 = -1e9, 1e9
        for p, q in ((-dx, x1 - pad), (dx, (CW - pad) - x1),
                     (-dy, y1 - pad), (dy, (canvas_h - pad) - y1)):
            if p == 0:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if t0 > t1:
            return None
        return (x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy)

    for ln in spec.get("lines", []) or []:
        through = ln.get("through") or []
        if len(through) != 2 or through[0] not in by_id or through[1] not in by_id:
            continue
        p1, p2 = by_id[through[0]], by_id[through[1]]
        sp1, sp2 = to_svg(p1["x"], p1["y"]), to_svg(p2["x"], p2["y"])
        clipped = _param_clip(sp1, sp2)
        if not clipped:
            continue
        e1, e2 = clipped
        body.append(f'<line x1="{e1[0]:.1f}" y1="{e1[1]:.1f}" x2="{e2[0]:.1f}" y2="{e2[1]:.1f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="1.8" data-role="line"/>')
        if ln.get("label"):
            body.append(_label(e2, str(ln["label"]), dx=-10, dy=-8))

    # --- line SEGMENTS between two declared points — dashed helper
    # projections, midpoints, and distance annotations all attach here.
    for seg in spec.get("segments", []) or []:
        a, b = _cg_coerce_id(seg.get("from")), _cg_coerce_id(seg.get("to"))
        if a not in by_id or b not in by_id:
            continue
        pa, pb = by_id[a], by_id[b]
        sa, sb = to_svg(pa["x"], pa["y"]), to_svg(pb["x"], pb["y"])
        dashed = bool(seg.get("dashed"))
        body.append(f'<line x1="{sa[0]:.1f}" y1="{sa[1]:.1f}" x2="{sb[0]:.1f}" y2="{sb[1]:.1f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2"'
                    + (' stroke-dasharray="5,3"' if dashed else '')
                    + ' data-role="segment"/>')

        if seg.get("show_midpoint"):
            mx, my = (sa[0] + sb[0]) / 2, (sa[1] + sb[1]) / 2
            body.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="3" fill="{AUX_COLOR}" data-role="midpoint"/>')
            # one tick on each half — a midpoint means the two halves
            # are equal, marked exactly like an equal_marks pair
            # elsewhere in this file, just recoloured/retagged so
            # verify_marker_coverage can count it separately.
            tick1 = _tick_marks(sa, (mx, my), 1).replace(f'stroke="{LINE_COLOR}"', f'stroke="{AUX_COLOR}" data-role="midpoint-tick"')
            tick2 = _tick_marks((mx, my), sb, 1).replace(f'stroke="{LINE_COLOR}"', f'stroke="{AUX_COLOR}" data-role="midpoint-tick"')
            body.append(tick1)
            body.append(tick2)
            mid_label = seg.get("midpoint_label", "M")
            if mid_label:
                body.append(_label((mx, my), str(mid_label), dy=-8))

        if seg.get("show_length"):
            dist = math.hypot(pb["x"] - pa["x"], pb["y"] - pa["y"])
            lx, ly = (sa[0] + sb[0]) / 2, (sa[1] + sb[1]) / 2
            perp = math.atan2(sb[1] - sa[1], sb[0] - sa[0]) + math.pi / 2
            lx += 14 * math.cos(perp)
            ly += 14 * math.sin(perp)
            body.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{AUX_COLOR}" data-role="length-label" '
                        f'font-family="Noto Sans, Arial" font-size="10" text-anchor="middle">{dist:.2f}</text>')

    # --- dashed projections from a point down to the x-axis and across
    # to the y-axis — the classic "drop a perpendicular to read off the
    # coordinates" picture.
    for p in coords:
        if not p.get("show_projection"):
            continue
        sx, sy = to_svg(p["x"], p["y"])
        foot_x, foot_y = to_svg(p["x"], 0)
        foot_x2, foot_y2 = to_svg(0, p["y"])
        if abs(sy - oy) > 0.5:
            body.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{foot_x:.1f}" y2="{foot_y:.1f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1" stroke-dasharray="3,2" data-role="projection"/>')
        if abs(sx - ox) > 0.5:
            body.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{foot_x2:.1f}" y2="{foot_y2:.1f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1" stroke-dasharray="3,2" data-role="projection"/>')

    # --- the points themselves, drawn last so they sit on top of every
    # grid line / segment / projection.
    for p in coords:
        sx, sy = to_svg(p["x"], p["y"])
        body.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{LINE_COLOR}"/>')
        dy = -8 if p["y"] >= 0 else 16
        # BUG FIX (found from real generated PDFs — reported by the
        # user, reproduced, confirmed): Gemini sometimes sends "label"
        # already containing the coordinate text itself, e.g.
        # label="A(-2, 8)" instead of just "A" — plausible since book
        # questions themselves are often written that way. This
        # renderer always appends its OWN "(x, y)" after the label
        # unconditionally, so an already-coordinate-bearing label
        # produced a visible doubled "A(-2, 8)(-2, 8)" on the actual
        # rendered diagram. Strip any trailing coordinate-looking
        # "(num, num)" from the label before appending the renderer's
        # own (authoritative, spec-validated) x/y — this also protects
        # against a label whose embedded coordinate doesn't even match
        # the point's real x/y, which would otherwise show two
        # different, contradictory coordinate pairs on the same point.
        clean_label = _COORD_LABEL_SUFFIX.sub("", str(p["label"])).strip() or p["id"]
        body.append(_label((sx, sy), f'{clean_label}({p["x"]:g}, {p["y"]:g})', dx=6, dy=dy))

    # --- Legend
    legend = spec.get("legend")
    if isinstance(legend, list) and legend:
        legend_y = pad
        for item in legend:
            if isinstance(item, dict) and "label" in item:
                body.append(f'<text x="{canvas_w - pad}" y="{legend_y}" text-anchor="end" fill="{LABEL_COLOR}" font-size="10" data-role="legend-item">{item["label"]}</text>')
                legend_y += 14

    return _svg("".join(body), width=canvas_w, height=canvas_h)


_COORD_LABEL_SUFFIX = re.compile(r"\s*\(\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\)\s*$")


def _coerce_id(value) -> str:
    """Gemini occasionally sends a point/vertex id as a one-element list
    (["D"]) instead of a bare string ("D") — this is what caused the
    'unhashable type: list' crash (a list can't be looked up as a dict
    key via `in`). Coerce anything list-like down to a single string."""
    if isinstance(value, list):
        return "".join(str(v) for v in value) if value else ""
    return str(value) if value is not None else ""


def _normalize_spec(spec: dict) -> dict:
    """Defensively coerces the diagram_spec's known trouble spots before
    any renderer sees it, so a schema deviation from the model degrades
    to 'a slightly different diagram' rather than an exception that
    throws away an entire exercise's already-solved questions."""
    spec = dict(spec)  # don't mutate the caller's dict

    if "right_angle_at" in spec:
        raw = spec["right_angle_at"]
        if isinstance(raw, list):
            spec["right_angle_at"] = [_coerce_id(v) for v in raw if _coerce_id(v)] or None
        else:
            spec["right_angle_at"] = _coerce_id(raw) or None

    if "points" in spec and isinstance(spec["points"], list):
        fixed_points = []
        for p in spec["points"]:
            if isinstance(p, dict):
                pid = _coerce_id(p.get("id"))
                if pid:
                    fixed_point = {"id": pid, "label": p.get("label", pid)}
                    # BUG FIX (found during the coordinate-geometry audit):
                    # this used to build a brand-new {"id", "label"} dict
                    # and drop every other key — including "x"/"y" and
                    # "show_projection" — before the renderer ever saw
                    # the point. That silently blanked EVERY
                    # coordinate_plot diagram's actual coordinates
                    # regardless of what Gemini supplied, since
                    # render_diagram() always normalizes before
                    # rendering. Preserve x/y/show_projection through
                    # normalization so coordinate_plot specs actually
                    # reach the renderer intact.
                    if "x" in p:
                        fixed_point["x"] = p["x"]
                    if "y" in p:
                        fixed_point["y"] = p["y"]
                    if "show_projection" in p:
                        fixed_point["show_projection"] = p["show_projection"]
                    fixed_points.append(fixed_point)
            elif isinstance(p, str) and p:
                fixed_points.append({"id": p, "label": p})
        spec["points"] = fixed_points

    if "equal_marks" in spec and isinstance(spec["equal_marks"], list):
        fixed_groups = []
        for group in spec["equal_marks"]:
            if not isinstance(group, list):
                continue
            fixed_group = [_coerce_id(side) for side in group]
            fixed_groups.append([s for s in fixed_group if s])
        spec["equal_marks"] = fixed_groups

    if "altitudes" in spec and isinstance(spec["altitudes"], list):
        fixed_alts = []
        for alt in spec["altitudes"]:
            if not isinstance(alt, dict):
                continue
            frm = _coerce_id(alt.get("from"))
            to_side = _coerce_id(alt.get("to_side") or alt.get("to"))
            foot = _coerce_id(alt.get("foot"))
            if frm and to_side and foot:
                fixed_alts.append({"from": frm, "to_side": to_side, "foot": foot})
        spec["altitudes"] = fixed_alts

    # coordinate_plot's segments/lines reference point ids the same way
    # altitudes does above — coerce defensively for the same reason
    # (Gemini occasionally sends a one-element list instead of a bare
    # string id).
    if "segments" in spec and isinstance(spec["segments"], list):
        fixed_segs = []
        for seg in spec["segments"]:
            if not isinstance(seg, dict):
                continue
            frm, to = _coerce_id(seg.get("from")), _coerce_id(seg.get("to"))
            if frm and to:
                fixed_segs.append({**seg, "from": frm, "to": to})
        spec["segments"] = fixed_segs

    diagram_type = spec.get("diagram_type")

    # NOTE: "lines" means two different shapes depending on diagram_type
    # — coordinate_plot's lines are {"through": [id, id]} objects, while
    # parallel_lines' lines are just the two plain line-id strings
    # ("l1", "l2"). Branching on diagram_type here (rather than one
    # generic coercion) is what stops parallel_lines' lines[] from
    # being silently emptied by a coercion written for the other shape.
    if "lines" in spec and isinstance(spec["lines"], list):
        if diagram_type == "parallel_lines":
            spec["lines"] = [_coerce_id(v) for v in spec["lines"] if _coerce_id(v)]
        else:
            fixed_lines = []
            for ln in spec["lines"]:
                if not isinstance(ln, dict):
                    continue
                through = ln.get("through")
                if isinstance(through, list) and len(through) == 2:
                    fixed_through = [_coerce_id(v) for v in through]
                    if all(fixed_through):
                        fixed_lines.append({**ln, "through": fixed_through})
            spec["lines"] = fixed_lines

    # angle renderer: rays[] and bisector.of/to ids
    if diagram_type == "angle":
        if "vertex" in spec:
            spec["vertex"] = _coerce_id(spec["vertex"])
        if "rays" in spec and isinstance(spec["rays"], list):
            fixed_rays = []
            for r in spec["rays"]:
                if not isinstance(r, dict):
                    continue
                rid = _coerce_id(r.get("id"))
                if rid and "direction_deg" in r:
                    fixed_rays.append({"id": rid, "direction_deg": r["direction_deg"]})
            spec["rays"] = fixed_rays
        if "angle_marks" in spec and isinstance(spec["angle_marks"], list):
            fixed_marks = []
            for m in spec["angle_marks"]:
                if not isinstance(m, dict):
                    continue
                between = m.get("between")
                if isinstance(between, list) and len(between) == 2:
                    fixed_between = [_coerce_id(v) for v in between]
                    if all(fixed_between):
                        fixed_marks.append({**m, "between": fixed_between})
            spec["angle_marks"] = fixed_marks
        bisector = spec.get("bisector")
        if isinstance(bisector, dict):
            of = bisector.get("of")
            if isinstance(of, list) and len(of) == 2:
                spec["bisector"] = {**bisector, "of": [_coerce_id(v) for v in of],
                                     "to": _coerce_id(bisector.get("to"))}

    # parallel_lines renderer: angle_marks[].line ids
    if diagram_type == "parallel_lines" and "angle_marks" in spec and isinstance(spec["angle_marks"], list):
        fixed_marks = []
        for m in spec["angle_marks"]:
            if not isinstance(m, dict):
                continue
            fixed_marks.append({**m, "line": _coerce_id(m.get("line"))})
        spec["angle_marks"] = fixed_marks

    return spec


def _validate_coordinate_plot_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == "coordinate_plot":
    every declared point must carry usable numeric x/y (a coordinate
    diagram with an un-plottable point is worse than no diagram at
    all), and every segment/line must reference point ids that are
    actually declared with coordinates — the coordinate-geometry
    equivalent of the triangle validator's id-consistency check above.
    """
    issues = []
    raw_points = spec.get("points") or []
    declared_ids = set()
    for p in raw_points:
        if not isinstance(p, dict):
            issues.append(f"point entry '{p}' is not an object with id/x/y")
            continue
        pid = p.get("id")
        try:
            float(p.get("x"))
            float(p.get("y"))
        except (TypeError, ValueError):
            issues.append(f"point '{pid}' is missing a numeric x/y coordinate")
            continue
        if pid:
            declared_ids.add(pid)

    for seg in spec.get("segments", []) or []:
        if not isinstance(seg, dict):
            issues.append(f"segment entry '{seg}' is not an object")
            continue
        a, b = seg.get("from"), seg.get("to")
        if a not in declared_ids or b not in declared_ids:
            issues.append(f"segment '{a}-{b}' references a point not declared with x/y in points[]")

    for ln in spec.get("lines", []) or []:
        if not isinstance(ln, dict):
            issues.append(f"line entry '{ln}' is not an object")
            continue
        through = ln.get("through") or []
        if len(through) != 2 or through[0] not in declared_ids or through[1] not in declared_ids:
            issues.append(f"line 'through' {through} must list exactly 2 point ids declared in points[]")

    return (len(issues) == 0), issues


def _validate_angle_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == "angle": every ray must
    carry a numeric direction_deg, and every angle_marks/bisector entry
    must reference ray ids that are actually declared.

    PRODUCTION-AUDIT FIX: previously had no minimum-ray requirement at
    all — a spec with an empty/missing rays[] passed with zero issues
    and rendered a lone vertex with no angle depicted whatsoever (see
    _render_angle: with ray_dirs empty, nothing is drawn but the point
    itself). An "angle" diagram is definitionally at least 2 rays from
    a shared vertex; require that minimum the same way triangle/
    quadrilateral/trigonometry already require their own minimum point
    counts."""
    issues = []
    ray_ids = set()
    valid_ray_count = 0
    for r in spec.get("rays", []) or []:
        if not isinstance(r, dict):
            issues.append(f"ray entry '{r}' is not an object")
            continue
        rid = r.get("id")
        try:
            float(r.get("direction_deg"))
        except (TypeError, ValueError):
            issues.append(f"ray '{rid}' is missing a numeric direction_deg")
            continue
        valid_ray_count += 1
        if rid:
            ray_ids.add(rid)

    if valid_ray_count < 2:
        issues.append(f"angle requires at least 2 rays with a numeric direction_deg, found {valid_ray_count}")
        return False, issues  # nothing else is meaningfully checkable without real rays

    for mark in spec.get("angle_marks", []) or []:
        between = mark.get("between") or []
        if len(between) != 2 or between[0] not in ray_ids or between[1] not in ray_ids:
            issues.append(f"angle_marks 'between' {between} must list exactly 2 ray ids declared in rays[]")

    bisector = spec.get("bisector")
    if isinstance(bisector, dict):
        of = bisector.get("of") or []
        if len(of) != 2 or of[0] not in ray_ids or of[1] not in ray_ids:
            issues.append(f"bisector 'of' {of} must list exactly 2 ray ids declared in rays[]")
        if not bisector.get("to"):
            issues.append("bisector is missing its 'to' label")

    return (len(issues) == 0), issues


def _validate_parallel_lines_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == "parallel_lines": every
    angle_marks entry must reference one of the (at most 2) declared
    line ids and a valid quadrant position."""
    issues = []
    lines = spec.get("lines") or []
    line_ids = {_coerce_id(v) for v in lines if _coerce_id(v)} or {"l1", "l2"}
    valid_positions = {"top_left", "top_right", "bottom_left", "bottom_right"}

    for mark in spec.get("angle_marks", []) or []:
        line_id = _coerce_id(mark.get("line"))
        position = mark.get("position")
        if line_id not in line_ids:
            issues.append(f"angle_marks 'line' \"{line_id}\" is not one of the declared lines[] ids")
        if position not in valid_positions:
            issues.append(f"angle_marks 'position' \"{position}\" must be one of {sorted(valid_positions)}")
        if not mark.get("label"):
            issues.append("angle_marks entry is missing its 'label'")

    return (len(issues) == 0), issues


def _validate_circle_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == "circle".

    PRODUCTION-AUDIT FINDING: "circle" was never dispatched to any
    dedicated validator here at all — validate_diagram_spec fell
    straight through to its generic "return True, []" default (the
    branch meant for genuinely unimplemented/plugin types), so a
    circle spec with a garbage center_id, a tangent_from referencing
    undeclared points, or duplicate point ids previously passed
    structural validation with zero issues raised — the exact same
    class of gap the triangle-validator fix above closed. _render_circle
    already defends itself against most of this at render time (see its
    own inline checks), but that is a render-time fallback, not a
    validation-time rejection with a clear audit reason — this closes
    the gap the way every other type's validator already does.
    """
    issues = []
    points = spec.get("points") or []
    ids = []
    for p in points:
        if not isinstance(p, dict) or not p.get("id"):
            issues.append(f"point entry '{p}' is not an object with an 'id'")
            continue
        ids.append(p["id"])
    if len(ids) != len(set(ids)):
        issues.append(f"points[] contains duplicate ids: {ids}")
    id_set = set(ids)

    center_id = spec.get("center_id")
    if center_id is not None and center_id not in id_set:
        issues.append(f"center_id '{center_id}' is not one of the declared points {sorted(id_set)}")

    tangent = spec.get("tangent_from")
    if isinstance(tangent, dict):
        if center_id is None or center_id not in id_set:
            issues.append("tangent_from is set but center_id is missing/invalid — a tangent "
                           "construction requires a known circle center")
        ext_id = tangent.get("external_point")
        if not ext_id or ext_id not in id_set:
            issues.append(f"tangent_from.external_point '{ext_id}' is not a declared point")
        tangent_points = tangent.get("tangent_points") or []
        if not tangent_points:
            issues.append("tangent_from.tangent_points must list at least one declared point")
        for tid in tangent_points:
            if tid not in id_set:
                issues.append(f"tangent_from.tangent_points entry '{tid}' is not a declared point")

    dims = spec.get("dimensions")
    if isinstance(dims, dict) and dims.get("radius") is not None:
        try:
            if float(dims["radius"]) <= 0:
                issues.append(f"dimensions.radius must be positive, got {dims['radius']}")
        except (TypeError, ValueError):
            issues.append(f"dimensions.radius '{dims['radius']}' is not numeric")

    return (len(issues) == 0), issues


def _validate_number_line_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == "number_line".

    PRODUCTION-AUDIT FINDING: same dispatch-table gap as "circle" —
    never validated at all before this fix. _render_number_line is
    fairly defensive with defaults (range_min/range_max), so this
    focuses on the one class of error defaults can't paper over: a
    marked_point or construction value that's non-numeric or sits
    outside the given range, which would draw a point in a visibly
    wrong location relative to the number line's own printed scale.
    """
    issues = []
    range_min, range_max = spec.get("range_min"), spec.get("range_max")
    for name, val in (("range_min", range_min), ("range_max", range_max)):
        if val is not None:
            try:
                float(val)
            except (TypeError, ValueError):
                issues.append(f"{name} '{val}' is not numeric")
    if range_min is not None and range_max is not None:
        try:
            if float(range_max) <= float(range_min):
                issues.append(f"range_max ({range_max}) must be greater than range_min ({range_min})")
        except (TypeError, ValueError):
            pass  # already flagged as non-numeric above

    def _check_value_in_range(value, field_name):
        try:
            v = float(value)
        except (TypeError, ValueError):
            issues.append(f"{field_name} '{value}' is not numeric")
            return
        lo = float(range_min) if range_min is not None else 0
        hi = float(range_max) if range_max is not None else max(lo + 4, 4)
        if not (lo <= v <= hi):
            issues.append(f"{field_name} ({v}) falls outside the declared range [{lo}, {hi}]")

    marked = spec.get("marked_point")
    if isinstance(marked, dict) and marked.get("value") is not None:
        _check_value_in_range(marked["value"], "marked_point.value")

    construction = spec.get("construction")
    if isinstance(construction, dict):
        if construction.get("base_point") is not None:
            _check_value_in_range(construction["base_point"], "construction.base_point")
        perp = construction.get("perpendicular_length")
        if perp is not None:
            try:
                if float(perp) <= 0:
                    issues.append(f"construction.perpendicular_length must be positive, got {perp}")
            except (TypeError, ValueError):
                issues.append(f"construction.perpendicular_length '{perp}' is not numeric")

    return (len(issues) == 0), issues


def _validate_square_root_spiral_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == "square_root_spiral".

    PRODUCTION-AUDIT FINDING: same dispatch-table gap. _render_square_root_spiral
    already clamps `steps` into [2, 10] defensively, but a clamp is a
    silent behavior change, not a validation rejection — if a question
    explicitly needs √11 shown and Gemini sent steps=11, silently
    clamping to 10 would draw a spiral that stops one step short of
    what the question actually asked for, with no audit trail at all.
    Reject out-of-range/non-numeric steps here instead, so that
    mismatch is visible in the logs rather than silently truncated.
    """
    issues = []
    steps = spec.get("steps")
    if steps is not None:
        try:
            steps_int = int(steps)
        except (TypeError, ValueError):
            issues.append(f"steps '{steps}' is not an integer")
        else:
            if not (2 <= steps_int <= 10):
                issues.append(f"steps ({steps_int}) must be between 2 and 10 "
                               f"(the renderer's supported range) — reduce/adjust the question's "
                               f"required √n count or this diagram will not accurately represent it")
    return (len(issues) == 0), issues


def validate_diagram_spec(spec: dict | None) -> tuple[bool, list[str]]:
    """Deterministic STRUCTURAL validation only — every point id referenced
    by equal_marks / right_angle_at / an altitude's from+to_side+foot must
    actually be declared in "points" (or, for a foot, be the altitude's own
    declared foot). This catches malformed/inconsistent specs (a mark or
    altitude pointing at a letter that doesn't exist in the figure) before
    they reach the renderer, which is what the "reject and don't show a
    wrong diagram" requirement can actually be automated end-to-end here.

    IMPORTANT LIMITATION, stated plainly rather than glossed over: this
    does NOT verify that the spec is semantically faithful to the natural-
    language question (e.g. that a question saying "equal altitudes"
    actually produced an equal_marks entry covering those two altitude
    segments). That is an open-ended NLU-grounding problem, not something
    a deterministic script can check reliably — catching that class of
    error still depends on the prompt's own instructions to Gemini and on
    human spot-checking, same as any other correctness property of the
    solved text. What this function guarantees is narrower but real: a
    spec that passes it cannot render a dangling/incorrect mark from a
    typo'd or missing point id.
    """
    if not spec:
        return True, []

    if spec.get("diagram_type") == "coordinate_plot":
        return _validate_coordinate_plot_spec(spec)
    if spec.get("diagram_type") == "angle":
        return _validate_angle_spec(spec)
    if spec.get("diagram_type") == "parallel_lines":
        return _validate_parallel_lines_spec(spec)
    if spec.get("diagram_type") == "quadrilateral":
        return _validate_quadrilateral_spec(spec)
    if spec.get("diagram_type") == "trigonometry":
        return _validate_trigonometry_spec(spec)
    if spec.get("diagram_type") == "statistics":
        return _validate_statistics_spec(spec)
    if spec.get("diagram_type") == "surface_area_volume":
        return _validate_surface_area_volume_spec(spec)
    if spec.get("diagram_type") == "construction":
        return _validate_construction_spec(spec)
    if spec.get("diagram_type") == "circle":
        return _validate_circle_spec(spec)
    if spec.get("diagram_type") == "number_line":
        return _validate_number_line_spec(spec)
    if spec.get("diagram_type") == "square_root_spiral":
        return _validate_square_root_spiral_spec(spec)

    plugin_result = diagram_plugin_registry.validate_via_plugin(spec)
    if plugin_result is not None:
        return plugin_result

    if spec.get("diagram_type") != "triangle":
        return True, []  # structural check only implemented for the types dispatched above

    issues = []
    raw_points = spec.get("points") or []
    declared_ids = {p.get("id") for p in raw_points if isinstance(p, dict) and p.get("id")}

    # BUG FIX (found via test_diagram_safety_net.py while building the
    # diagram safety-net recovery pass — a recovered spec of just
    # {"diagram_type": "triangle"} with NO points at all previously
    # sailed through this function with zero issues, because every
    # check below only cross-references altitudes/equal_marks/
    # right_angle_at IDs against declared_ids — with nothing else in
    # the spec, there was nothing to cross-reference, so an empty
    # points[] was silently accepted as structurally valid. That let a
    # triangle with no actual geometry reach the renderer, which then
    # has to fall back to an arbitrary generic placement — exactly the
    # "not tied to the real question" failure this whole validation
    # layer exists to prevent. A triangle is three named vertices by
    # definition; require at least 3, with unique, non-empty ids,
    # mirroring _validate_quadrilateral_spec's own "exactly 4 points"
    # requirement for quadrilaterals just above.
    if len(declared_ids) < 3 or len(raw_points) < 3:
        issues.append(f"triangle requires at least 3 uniquely-identified points, found "
                       f"{len(raw_points)} point entr{'y' if len(raw_points) == 1 else 'ies'} "
                       f"({len(declared_ids)} with a usable id)")
        return False, issues  # nothing else is checkable without real points

    altitude_feet = set()
    for alt in spec.get("altitudes", []) or []:
        frm, to_side, foot = alt.get("from"), alt.get("to_side"), alt.get("foot")
        if frm not in declared_ids:
            issues.append(f"altitude 'from' point '{frm}' is not in points[]")
        if not to_side or len(to_side) != 2 or to_side[0] not in declared_ids or to_side[1] not in declared_ids:
            issues.append(f"altitude 'to_side' \"{to_side}\" references a point not in points[]")
        if foot not in declared_ids:
            issues.append(f"altitude foot '{foot}' is not in points[] (it must be declared there too)")
        else:
            altitude_feet.add(foot)

    known_ids = declared_ids | altitude_feet
    for group in spec.get("equal_marks", []) or []:
        for side in group:
            if len(side) == 2 and (side[0] not in known_ids or side[1] not in known_ids):
                issues.append(f"equal_marks segment '{side}' references a point not in points[]")

    right_angle_ids = spec.get("right_angle_at")
    if isinstance(right_angle_ids, str):
        right_angle_ids = [right_angle_ids]
    for rid in right_angle_ids or []:
        if rid not in known_ids:
            issues.append(f"right_angle_at '{rid}' is not in points[]")

    cevian_feet = set()
    for cev in spec.get("cevians", []) or []:
        frm, to_side, foot = cev.get("from"), cev.get("to_side"), cev.get("foot")
        if frm not in declared_ids:
            issues.append(f"cevian 'from' point '{frm}' is not in points[]")
        if not to_side or len(to_side) != 2 or to_side[0] not in declared_ids or to_side[1] not in declared_ids:
            issues.append(f"cevian 'to_side' \"{to_side}\" references a point not in points[]")
        if foot not in declared_ids:
            issues.append(f"cevian foot '{foot}' is not in points[] (it must be declared there too)")
        else:
            cevian_feet.add(foot)
    known_ids |= cevian_feet

    intersection_label = spec.get("cevian_intersection_label")
    if intersection_label:
        valid_cevian_count = sum(
            1 for cev in (spec.get("cevians") or [])
            if cev.get("from") in declared_ids and cev.get("foot") in declared_ids
            and cev.get("to_side") and len(cev.get("to_side")) == 2
            and cev["to_side"][0] in declared_ids and cev["to_side"][1] in declared_ids
        )
        if valid_cevian_count != 2:
            issues.append(
                f"cevian_intersection_label '{intersection_label}' requires EXACTLY 2 valid "
                f"'cevians' entries to compute an intersection from, found {valid_cevian_count}"
            )

    join_vertex = spec.get("join_vertex_to_intersection")
    if join_vertex and join_vertex not in known_ids:
        issues.append(f"join_vertex_to_intersection '{join_vertex}' is not in points[]")

    return (len(issues) == 0), issues


def _verify_coordinate_plot_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """Post-render check for coordinate_plot: every segment/line the
    spec asked for, and every midpoint/distance-label/projection a
    segment or point asked for, must actually appear in the rendered
    SVG. Every element the renderer draws for these relationships
    carries a `data-role="..."` attribute specifically so this can be
    counted precisely (see _render_coordinate_plot above), the same
    "count what was actually drawn, don't just trust the renderer ran"
    policy already used for the triangle renderer's ticks/right-angles.
    Only relationships whose endpoints are actually declared with
    numeric x/y are counted as "expected" — an id that doesn't resolve
    to a real point was already flagged by validate_diagram_spec and
    is correctly excluded here too, or every legitimate diagram would
    also fail this check for an unrelated typo.
    """
    declared_ids = set()
    for p in spec.get("points") or []:
        if not isinstance(p, dict):
            continue
        try:
            float(p.get("x"))
            float(p.get("y"))
        except (TypeError, ValueError):
            continue
        if p.get("id"):
            declared_ids.add(p["id"])

    expected_segments = 0
    expected_midpoints = 0
    expected_lengths = 0
    for seg in spec.get("segments", []) or []:
        if not isinstance(seg, dict):
            continue
        if seg.get("from") in declared_ids and seg.get("to") in declared_ids:
            expected_segments += 1
            if seg.get("show_midpoint"):
                expected_midpoints += 1
            if seg.get("show_length"):
                expected_lengths += 1

    expected_lines = 0
    for ln in spec.get("lines", []) or []:
        if not isinstance(ln, dict):
            continue
        through = ln.get("through") or []
        if len(through) == 2 and through[0] in declared_ids and through[1] in declared_ids:
            expected_lines += 1

    expected_projections = 0
    for p in spec.get("points") or []:
        if isinstance(p, dict) and p.get("id") in declared_ids and p.get("show_projection"):
            expected_projections += 1

    actual_segments = svg.count('data-role="segment"')
    actual_midpoints = svg.count('data-role="midpoint"')
    actual_lengths = svg.count('data-role="length-label"')
    actual_lines = svg.count('data-role="line"')
    # each projection point can draw up to 2 dashed legs (to x-axis and
    # to y-axis); a point sitting exactly ON an axis only needs the one
    # leg that isn't degenerate, so "at least 1 per expected point" is
    # the correct floor to check, not "== 2 * expected_projections".
    actual_projection_legs = svg.count('data-role="projection"')

    issues = []
    if actual_segments < expected_segments:
        issues.append(f"expected {expected_segments} segment(s), only {actual_segments} rendered")
    if actual_midpoints < expected_midpoints:
        issues.append(f"expected {expected_midpoints} midpoint marker(s), only {actual_midpoints} rendered")
    if actual_lengths < expected_lengths:
        issues.append(f"expected {expected_lengths} distance label(s), only {actual_lengths} rendered")
    if actual_lines < expected_lines:
        issues.append(f"expected {expected_lines} extended line(s), only {actual_lines} rendered")
    if expected_projections > 0 and actual_projection_legs < expected_projections:
        issues.append(f"expected at least {expected_projections} projection leg(s), only {actual_projection_legs} rendered")

    return (len(issues) == 0), issues


def _verify_angle_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """Post-render check for 'angle': every declared ray must actually
    be drawn, every valid angle_marks entry must have its arc drawn,
    and a valid bisector must have its dashed ray drawn — counted via
    the data-role tags _render_angle attaches to each element."""
    ray_ids = set()
    for r in spec.get("rays", []) or []:
        if isinstance(r, dict) and r.get("id"):
            try:
                float(r.get("direction_deg"))
                ray_ids.add(r["id"])
            except (TypeError, ValueError):
                pass

    expected_rays = len(ray_ids)
    expected_arcs = sum(
        1 for m in (spec.get("angle_marks") or [])
        if len((m.get("between") or [])) == 2
        and m["between"][0] in ray_ids and m["between"][1] in ray_ids
    )
    bisector = spec.get("bisector")
    expected_bisector = 0
    if isinstance(bisector, dict):
        of = bisector.get("of") or []
        if len(of) == 2 and of[0] in ray_ids and of[1] in ray_ids and bisector.get("to"):
            expected_bisector = 1

    actual_rays = svg.count('data-role="ray"')
    actual_arcs = svg.count('data-role="angle-arc"')
    actual_bisector = svg.count('data-role="bisector"')

    issues = []
    if actual_rays < expected_rays:
        issues.append(f"expected {expected_rays} ray(s), only {actual_rays} rendered")
    if actual_arcs < expected_arcs:
        issues.append(f"expected {expected_arcs} angle arc(s), only {actual_arcs} rendered")
    if actual_bisector < expected_bisector:
        issues.append(f"expected a bisector ray, none rendered")

    return (len(issues) == 0), issues


def _verify_parallel_lines_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """Post-render check for 'parallel_lines': every valid angle_marks
    entry must have its label actually drawn, counted via the
    data-role="angle-label" tag _render_parallel_lines attaches."""
    lines = spec.get("lines") or []
    line_ids = {_coerce_id(v) for v in lines if _coerce_id(v)} or {"l1", "l2"}
    valid_positions = {"top_left", "top_right", "bottom_left", "bottom_right"}

    expected_labels = sum(
        1 for m in (spec.get("angle_marks") or [])
        if _coerce_id(m.get("line")) in line_ids and m.get("position") in valid_positions and m.get("label")
    )
    actual_labels = svg.count('data-role="angle-label"')

    issues = []
    if actual_labels < expected_labels:
        issues.append(f"expected {expected_labels} angle label(s), only {actual_labels} rendered")

    return (len(issues) == 0), issues


def verify_marker_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """POST-render semantic validation gate — this is deliberately a
    DIFFERENT check from validate_diagram_spec() above, which only
    confirms the spec's point ids are internally consistent BEFORE
    anything is drawn. This function looks at the actual SVG markup
    that was produced and counts whether every declared mathematical
    relationship really got a visible marker, which is the literal
    "Equal lengths -> Equal tick marks / Perpendicular -> Right-angle
    symbols" requirement: a rendered diagram is only accepted if the
    number of tick-mark <line> elements and right-angle <polyline>
    elements in the output SVG matches what the spec's own
    equal_marks / altitudes / right_angle_at entries require — not
    just "did the renderer run without an exception", which says
    nothing about whether a mark was actually drawn where the math
    needs one.

    Deliberately scoped to "triangle" for now (the renderer with real
    per-relationship markers to count); other diagram types return
    (True, []) unchanged rather than a false failure — see the
    _RENDERERS docstring for how to extend this alongside a new
    renderer.
    """
    if not spec:
        return True, []

    if spec.get("diagram_type") == "coordinate_plot":
        return _verify_coordinate_plot_coverage(spec, svg)
    if spec.get("diagram_type") == "angle":
        return _verify_angle_coverage(spec, svg)
    if spec.get("diagram_type") == "parallel_lines":
        return _verify_parallel_lines_coverage(spec, svg)
    if spec.get("diagram_type") == "quadrilateral":
        return _verify_quadrilateral_coverage(spec, svg)
    if spec.get("diagram_type") == "trigonometry":
        return _verify_trigonometry_coverage(spec, svg)
    if spec.get("diagram_type") == "statistics":
        return _verify_statistics_coverage(spec, svg)
    if spec.get("diagram_type") == "surface_area_volume":
        return _verify_surface_area_volume_coverage(spec, svg)

    if spec.get("diagram_type") != "triangle":
        return True, []

    declared_ids = {p.get("id") for p in (spec.get("points") or []) if isinstance(p, dict)}
    altitude_feet = set()
    valid_altitudes = 0
    for alt in spec.get("altitudes", []) or []:
        frm, to_side, foot = alt.get("from"), alt.get("to_side"), alt.get("foot")
        if frm in declared_ids and to_side and len(to_side) == 2 \
                and to_side[0] in declared_ids and to_side[1] in declared_ids and foot:
            altitude_feet.add(foot)
            valid_altitudes += 1
    known_ids = declared_ids | altitude_feet

    expected_ticks = 0
    for group in spec.get("equal_marks", []) or []:
        for side in group:
            if len(side) == 2 and side[0] in known_ids and side[1] in known_ids:
                expected_ticks += _tick_count_for(spec, side)

    right_angle_ids = spec.get("right_angle_at")
    if isinstance(right_angle_ids, str):
        right_angle_ids = [right_angle_ids]
    extra_right_angles = sum(1 for rid in (right_angle_ids or [])
                              if rid in known_ids and rid not in altitude_feet)
    expected_right_angles = valid_altitudes + extra_right_angles

    # Tick marks are always emitted as `<line ... stroke-width="1.5"/>`
    # (see _tick_marks); right-angle boxes are always `<polyline ...>`
    # (see _right_angle_box) — these two element shapes never overlap
    # with the triangle's own edges/cevians, which use stroke-width="2"
    # or the altitude/median color #c0392b, so a plain substring count
    # is exact, not a heuristic, for markup this renderer itself wrote.
    actual_ticks = svg.count('stroke-width="1.5"/>') - svg.count('<polyline')
    actual_right_angles = svg.count('<polyline')

    issues = []
    if actual_ticks < expected_ticks:
        issues.append(f"expected {expected_ticks} equal-mark tick(s), only {actual_ticks} rendered")
    if actual_right_angles < expected_right_angles:
        issues.append(f"expected {expected_right_angles} right-angle symbol(s), only {actual_right_angles} rendered")

    intersection_label = spec.get("cevian_intersection_label")
    if intersection_label:
        valid_cevian_count = sum(
            1 for cev in (spec.get("cevians") or [])
            if cev.get("from") in declared_ids and cev.get("foot") in declared_ids
            and cev.get("to_side") and len(cev.get("to_side")) == 2
            and cev["to_side"][0] in declared_ids and cev["to_side"][1] in declared_ids
        )
        if valid_cevian_count == 2 and svg.count('data-role="cevian-intersection"') < 1:
            issues.append(f"expected the cevian-intersection point '{intersection_label}' to be rendered, none found")
        join_vertex = spec.get("join_vertex_to_intersection")
        if join_vertex and join_vertex in known_ids and valid_cevian_count == 2 \
                and svg.count('data-role="intersection-join"') < 1:
            issues.append(f"expected a joining segment from '{join_vertex}' to "
                           f"'{intersection_label}', none found")

    return (len(issues) == 0), issues


# ------------------------------------------------------------------
# QUADRILATERAL — parallelograms, rectangles, rhombi, squares,
# trapeziums, and general quadrilaterals. Same philosophy as the
# triangle renderer above: Gemini supplies only the RELATIONSHIPS
# (shape family, which sides are marked equal/parallel, where a right
# angle sits, which diagonals to draw) — every coordinate is computed
# deterministically here from the declared `shape`, never freely
# drawn. Points are placed A, B, C, D going around the quadrilateral
# in order (so edges are AB, BC, CD, DA), matching how these are
# always labelled in a Class 8/9 textbook figure.
# ------------------------------------------------------------------
_QUAD_SHAPES = {"parallelogram", "rectangle", "rhombus", "square", "trapezium", "general"}


def _quad_layout(shape: str) -> list[tuple[float, float]]:
    """Returns 4 (x, y) canvas coordinates for A, B, C, D (in that
    order, going around the shape) for the given shape family — the
    ONE place shape geometry is decided, so every quadrilateral of a
    given family always looks the same regardless of which question
    asked for it."""
    cx, cy = W / 2, H / 2
    if shape == "rectangle" or shape == "square":
        hw, hh = (70, 70) if shape == "square" else (85, 55)
        return [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
    if shape == "rhombus":
        hw, hh = 60, 80
        return [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
    if shape == "trapezium":
        # AB is the longer parallel side (bottom), DC the shorter (top) —
        # the standard textbook orientation for trapezium area questions.
        return [(cx - 90, cy + 60), (cx + 90, cy + 60), (cx + 50, cy - 60), (cx - 50, cy - 60)]
    if shape == "parallelogram":
        skew = 35
        return [(cx - 85 + skew, cy + 55), (cx + 85 + skew, cy + 55), (cx + 85 - skew, cy - 55), (cx - 85 - skew, cy - 55)]
    # "general" — a plausible irregular-but-simple (non-self-intersecting)
    # quadrilateral, since Gemini gives us relationships, not coordinates.
    return [(cx - 80, cy + 65), (cx + 90, cy + 40), (cx + 60, cy - 65), (cx - 70, cy - 35)]


def _has_both_diagonals(ids: list, diagonals: list) -> bool:
    """True only if `diagonals` contains BOTH of the quadrilateral's
    actual diagonals (opposite-corner pairs) — e.g. for ids=[A,B,C,D],
    that's an "AC"-type entry AND a "BD"-type entry (either character
    order). A single diagonal, or two entries that are really the same
    diagonal twice, does not count — the intersection point is only
    geometrically well-defined once both real diagonals are drawn."""
    if len(ids) != 4:
        return False
    diag_a = {ids[0], ids[2]}
    diag_b = {ids[1], ids[3]}
    declared = [set(str(d)) for d in (diagonals or []) if len(str(d)) == 2]
    return any(d == diag_a for d in declared) and any(d == diag_b for d in declared)


def _segment_intersection(p1, p2, p3, p4):
    """Standard 2D line-line intersection (treating each pair as an
    infinite line through the two given points) — exact and
    deterministic, no approximation. Returns None only for the
    degenerate case of parallel diagonals (shouldn't occur for a
    genuine simple quadrilateral's two diagonals, but never raises)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def _render_quadrilateral(spec: dict) -> str:
    points = spec.get("points", []) or []
    ids = [p["id"] for p in points][:4]
    shape = spec.get("shape") if spec.get("shape") in _QUAD_SHAPES else "general"
    layout = _quad_layout(shape)
    coords = {pid: layout[i] for i, pid in enumerate(ids)} if len(ids) == 4 else {}

    body = []
    edge_pairs = list(zip(ids, ids[1:] + ids[:1])) if len(ids) == 4 else []
    for a, b in edge_pairs:
        body.append(f'<line x1="{coords[a][0]:.1f}" y1="{coords[a][1]:.1f}" '
                     f'x2="{coords[b][0]:.1f}" y2="{coords[b][1]:.1f}" '
                     f'stroke="{LINE_COLOR}" stroke-width="2" data-role="edge"/>')

    diagonals = spec.get("diagonals", []) or []
    for diag in diagonals:
        d = str(diag)
        if len(d) == 2 and d[0] in coords and d[1] in coords:
            body.append(f'<line x1="{coords[d[0]][0]:.1f}" y1="{coords[d[0]][1]:.1f}" '
                         f'x2="{coords[d[1]][0]:.1f}" y2="{coords[d[1]][1]:.1f}" '
                         f'stroke="{AUX_COLOR}" stroke-width="1.3" stroke-dasharray="4,3" data-role="diagonal"/>')

    # Diagonal-intersection point (e.g. "O") — see prompts.py schema.
    # BUG FIX (found from a real generated PDF — user-reported): a
    # proof about "O, the intersection of the diagonals" previously had
    # no way to have O appear in the diagram at all, since the
    # renderer only ever knew about the 4 declared corner points.
    intersection_label = spec.get("diagonal_intersection_label")
    if intersection_label and len(ids) == 4 and _has_both_diagonals(ids, diagonals):
        ix, iy = _segment_intersection(coords[ids[0]], coords[ids[2]], coords[ids[1]], coords[ids[3]])
        if ix is not None:
            body.append(f'<circle cx="{ix:.1f}" cy="{iy:.1f}" r="2.5" fill="{LINE_COLOR}" data-role="diagonal-intersection"/>')
            body.append(_label((ix, iy), str(intersection_label), dx=8, dy=-8))

    for group in spec.get("equal_marks", []) or []:
        for side in group or []:
            if len(side) == 2 and side[0] in coords and side[1] in coords:
                body.append(_tick_marks(coords[side[0]], coords[side[1]],
                                         _tick_count_for(spec, side), role="equal-tick"))

    right_angle_ids = spec.get("right_angle_at")
    if isinstance(right_angle_ids, str):
        right_angle_ids = [right_angle_ids]
    for rid in right_angle_ids or []:
        if rid not in coords:
            continue
        idx = ids.index(rid)
        prev_id, next_id = ids[idx - 1], ids[(idx + 1) % len(ids)]
        d1 = _unit(coords[rid], coords[prev_id])
        d2 = _unit(coords[rid], coords[next_id])
        body.append(_right_angle_box(coords[rid], d1, d2, role="right-angle"))

    for pid, (x, y) in coords.items():
        dx = -14 if x < cx_default(coords) else 8
        dy = -8 if y < H / 2 else 16
        body.append(_label((x, y), pid, dx=dx if x < W / 2 else 8, dy=dy))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{LINE_COLOR}"/>')

    for extra in spec.get("extra_labels", []) or []:
        extra = _sanitize_extra_label(extra)
        if not extra:
            continue
        body.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" fill="#888" font-size="11">{extra}</text>')

    return _svg("".join(body))


def cx_default(coords):
    """Small helper so quadrilateral vertex labels lean outward from the
    shape's own centroid rather than a fixed canvas midpoint — keeps
    labels legible for the skewed parallelogram/trapezium layouts."""
    xs = [c[0] for c in coords.values()]
    return sum(xs) / len(xs) if xs else W / 2


def _validate_quadrilateral_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == 'quadrilateral':
    exactly 4 declared points (a quadrilateral with 3 or 5+ points is
    not a quadrilateral), a recognized shape family, and every
    equal_marks/diagonal/right_angle_at id must reference one of those
    4 declared points."""
    issues = []
    points = spec.get("points") or []
    ids = []
    for p in points:
        if not isinstance(p, dict) or not p.get("id"):
            issues.append(f"point entry '{p}' is not an object with an 'id'")
            continue
        ids.append(p["id"])
    if len(ids) != 4:
        issues.append(f"quadrilateral requires exactly 4 points, found {len(ids)}")
        return False, issues  # nothing else is checkable without exactly 4 points

    shape = spec.get("shape")
    if shape is not None and shape not in _QUAD_SHAPES:
        issues.append(f"shape '{shape}' must be one of {sorted(_QUAD_SHAPES)}")

    id_set = set(ids)
    for group in spec.get("equal_marks", []) or []:
        for side in group or []:
            if len(side) != 2 or side[0] not in id_set or side[1] not in id_set:
                issues.append(f"equal_marks side '{side}' references a point not in points[]")

    for diag in spec.get("diagonals", []) or []:
        d = str(diag)
        if len(d) != 2 or d[0] not in id_set or d[1] not in id_set:
            issues.append(f"diagonal '{diag}' references a point not in points[]")

    right_angle_ids = spec.get("right_angle_at")
    if isinstance(right_angle_ids, str):
        right_angle_ids = [right_angle_ids]
    for rid in right_angle_ids or []:
        if rid not in id_set:
            issues.append(f"right_angle_at '{rid}' is not in points[]")

    intersection_label = spec.get("diagonal_intersection_label")
    if intersection_label and not _has_both_diagonals(ids, spec.get("diagonals") or []):
        issues.append(
            f"diagonal_intersection_label '{intersection_label}' requires BOTH diagonals "
            f"(an '{ids[0]}{ids[2]}'-type entry AND a '{ids[1]}{ids[3]}'-type entry) to be "
            f"listed in 'diagonals' — the intersection point isn't defined otherwise"
        )

    return (len(issues) == 0), issues


def _verify_quadrilateral_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """Post-render check for 'quadrilateral': all 4 edges must be
    drawn, every declared diagonal/equal-mark/right-angle must have
    its data-role element actually present in the output SVG, and the
    diagonal-intersection point (if requested) must have rendered."""
    points = spec.get("points") or []
    ids = [p["id"] for p in points if isinstance(p, dict) and p.get("id")][:4]
    id_set = set(ids)

    expected_edges = 4 if len(ids) == 4 else 0
    expected_diagonals = sum(1 for d in (spec.get("diagonals") or [])
                              if len(str(d)) == 2 and str(d)[0] in id_set and str(d)[1] in id_set)
    expected_ticks = sum(1 for group in (spec.get("equal_marks") or [])
                          for side in (group or [])
                          if len(side) == 2 and side[0] in id_set and side[1] in id_set)
    right_angle_ids = spec.get("right_angle_at")
    if isinstance(right_angle_ids, str):
        right_angle_ids = [right_angle_ids]
    expected_right_angles = sum(1 for rid in (right_angle_ids or []) if rid in id_set)

    issues = []
    actual_edges = svg.count('data-role="edge"')
    actual_diagonals = svg.count('data-role="diagonal"')
    actual_ticks = svg.count('data-role="equal-tick"')
    actual_right_angles = svg.count('data-role="right-angle"')

    if actual_edges < expected_edges:
        issues.append(f"expected {expected_edges} edge(s), only {actual_edges} rendered")
    if actual_diagonals < expected_diagonals:
        issues.append(f"expected {expected_diagonals} diagonal(s), only {actual_diagonals} rendered")
    if actual_ticks < expected_ticks:
        issues.append(f"expected {expected_ticks} equal-mark tick(s), only {actual_ticks} rendered")
    if actual_right_angles < expected_right_angles:
        issues.append(f"expected {expected_right_angles} right-angle symbol(s), only {actual_right_angles} rendered")


    if spec.get("diagonal_intersection_label") and _has_both_diagonals(ids, spec.get("diagonals") or []):
        if svg.count('data-role="diagonal-intersection"') < 1:
            issues.append(f"expected the diagonal-intersection point "
                           f"'{spec['diagonal_intersection_label']}' to be rendered, none found")

    return (len(issues) == 0), issues


# ------------------------------------------------------------------
# TRIGONOMETRY — a right triangle purpose-built for "find sin/cos/tan
# of angle X" style questions: a fixed right-angle vertex, one marked
# acute angle (numeric or symbolic), and explicit opposite/adjacent/
# hypotenuse side labels. Deliberately a SEPARATE renderer from the
# generic "triangle" type (rather than overloading it) because trig
# questions need a specific, consistent right-angle-at-bottom-left
# orientation and side-role labelling that generic triangle proofs
# don't use.
# ------------------------------------------------------------------
def _render_trigonometry(spec: dict) -> str:
    points = spec.get("points", []) or []
    ids = [p["id"] for p in points][:3]
    right_id = spec.get("right_angle_at")
    dims = spec.get("dimensions") or {}

    # Construct the right triangle TO SCALE from the given side lengths
    # (dimensions), not a fixed decorative shape. Right angle is placed
    # at the bottom-left for consistency (standard textbook orientation).
    adj = float(dims.get("adjacent", 1))
    opp = float(dims.get("opposite", 1))
    hyp = math.hypot(adj, opp)

    scale = min((W - 100) / adj, (H - 80) / opp)
    adj_s, opp_s = adj * scale, opp * scale

    A = (60, H - 40)              # right-angle vertex (bottom-left)
    B = (A[0] + adj_s, A[1])      # adjacent vertex (bottom-right)
    C = (A[0], A[1] - opp_s)      # opposite vertex (top-left)

    id_to_xy = {}
    if len(ids) == 3 and right_id in ids and right_id in {ids[0], ids[1], ids[2]}:
        # Assign A,B,C to the three point IDs based on which one is the
        # right angle, preserving the to-scale geometry computed above.
        # The other two points are assigned to B,C based on which one
        # is the marked angle_at (if any), otherwise arbitrarily.
        angle_id = spec.get("angle_at")
        others = [i for i in ids if i != right_id and i != angle_id]
        id_to_xy[right_id] = A
        if angle_id and angle_id in ids and angle_id != right_id:
            id_to_xy[angle_id] = B
            if others:
                id_to_xy[others[0]] = C
        else:
            id_to_xy[ids[(ids.index(right_id) + 1) % 3]] = B
            id_to_xy[ids[(ids.index(right_id) + 2) % 3]] = C

    body = []
    if len(id_to_xy) == 3:
        body.append(_line(id_to_xy[ids[0]], id_to_xy[ids[1]]))
        body.append(_line(id_to_xy[ids[1]], id_to_xy[ids[2]]))
        body.append(_line(id_to_xy[ids[2]], id_to_xy[ids[0]]))
        d1, d2 = _unit(A, B), _unit(A, C)
        body.append(_right_angle_box(A, d1, d2, role="right-angle"))

        if angle_id in id_to_xy:
            vertex = id_to_xy[angle_id]
            # arc opens toward the OTHER two vertices from this one
            others_xy = [xy for pid, xy in id_to_xy.items() if pid != angle_id]
            a1 = math.degrees(math.atan2(-(others_xy[0][1] - vertex[1]), others_xy[0][0] - vertex[0]))
            a2 = math.degrees(math.atan2(-(others_xy[1][1] - vertex[1]), others_xy[1][0] - vertex[0]))
            radius = 22
            sx, sy = vertex[0] + radius * math.cos(math.radians(a1)), vertex[1] - radius * math.sin(math.radians(a1))
            ex, ey = vertex[0] + radius * math.cos(math.radians(a2)), vertex[1] - radius * math.sin(math.radians(a2))
            body.append(f'<path d="M {sx:.1f} {sy:.1f} A {radius} {radius} 0 0 0 {ex:.1f} {ey:.1f}" '
                        f'fill="none" stroke="{LINE_COLOR}" stroke-width="1.3" data-role="angle-arc"/>')
            label_text = spec.get("angle_label") or (f'{spec.get("angle_value_deg")}°' if spec.get("angle_value_deg") is not None else "θ")
            mid_ang = math.radians((a1 + a2) / 2)
            lx, ly = vertex[0] + (radius + 12) * math.cos(mid_ang), vertex[1] - (radius + 12) * math.sin(mid_ang)
            body.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{LABEL_COLOR}" {FONT} '
                        f'text-anchor="middle" data-role="angle-label">{label_text}</text>')

        for pid, (x, y) in id_to_xy.items():
            dy = 18 if y >= H - 45 else -8
            dx = -12 if x < 90 else 6
            body.append(_label((x, y), pid, dx=dx, dy=dy))
            body.append(f'<circle cx="{x}" cy="{y}" r="2.5" fill="{LINE_COLOR}"/>')

        side_labels = spec.get("side_labels") or {}
        for side, text in side_labels.items():
            side_set = set(str(side))
            if len(side_set) == 2 and side_set.issubset(id_to_xy.keys()) and text:
                p1, p2 = id_to_xy[list(side_set)[0]], id_to_xy[list(side_set)[1]]
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                perp = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.pi / 2
                off = 14 if right_id in side_set else -14 # Push labels outwards from hypotenuse
                lx, ly = mx + off * math.cos(perp), my + off * math.sin(perp)
                body.append(f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{LABEL_COLOR}" {FONT} '
                            f'text-anchor="middle" data-role="side-label">{text}</text>')
        
        # Which point id landed at each computed to-scale vertex, so the
        # dimension labels below can be placed on the correct side
        # regardless of which of A/B/C ended up as the adjacent vs.
        # opposite vertex.
        b_id = next((pid for pid, xy in id_to_xy.items() if xy == B), None)
        c_id = next((pid for pid, xy in id_to_xy.items() if xy == C), None)
        adj_ids = {right_id, b_id} if b_id is not None else None
        opp_ids = {right_id, c_id} if c_id is not None else None
        hyp_ids = {b_id, c_id} if b_id is not None and c_id is not None else None

        # Add dimension labels from 'dimensions'
        if adj_ids:
             p1, p2 = id_to_xy[list(adj_ids)[0]], id_to_xy[list(adj_ids)[1]]
             mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
             body.append(f'<text x="{mx}" y="{my+16}" text-anchor="middle" fill="{AUX_COLOR}" font-size="11" data-role="dimension-label">{dims["adjacent"]:g}</text>')
        if opp_ids:
             p1, p2 = id_to_xy[list(opp_ids)[0]], id_to_xy[list(opp_ids)[1]]
             mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
             body.append(f'<text x="{mx-10}" y="{my}" text-anchor="end" dominant-baseline="middle" fill="{AUX_COLOR}" font-size="11" data-role="dimension-label">{dims["opposite"]:g}</text>')
        p1, p2 = id_to_xy[list(hyp_ids)[0]], id_to_xy[list(hyp_ids)[1]]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        body.append(f'<text x="{mx-10}" y="{my-10}" text-anchor="end" fill="{AUX_COLOR}" font-size="11" data-role="dimension-label">{hyp:.1f}</text>')

    if spec.get("show_ratio"):
        ratio_text = {"sin": "sin θ = বিপ / অতিভুজ (opposite / hypotenuse)",
                      "cos": "cos θ = সন্নিহিত / অতিভুজ (adjacent / hypotenuse)",
                      "tan": "tan θ = বিপ / সন্নিহিত (opposite / adjacent)"}.get(spec["show_ratio"])
        if ratio_text:
            body.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" fill="#888" '
                        f'font-size="10" data-role="ratio-note">{ratio_text}</text>')

    return _svg("".join(body))


def _validate_trigonometry_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == 'trigonometry': exactly
    3 declared points, right_angle_at and angle_at (if given) must both
    reference declared points and must be different points, and
    show_ratio (if given) must be one of sin/cos/tan."""
    issues = []
    points = spec.get("points") or []
    ids = [p.get("id") for p in points if isinstance(p, dict) and p.get("id")]
    if len(ids) != 3:
        issues.append(f"trigonometry requires exactly 3 points, found {len(ids)}")
        return False, issues

    right_id = spec.get("right_angle_at")
    if right_id not in ids:
        issues.append(f"right_angle_at '{right_id}' is not one of the declared points {ids}")

    dims = spec.get("dimensions") or {}
    if not all(_is_number(dims.get(k)) for k in ("adjacent", "opposite")):
        issues.append("trigonometry requires numeric 'adjacent' and 'opposite' side lengths in 'dimensions'")

    ratio = spec.get("show_ratio")
    if ratio is not None and ratio not in ("sin", "cos", "tan"):
        issues.append(f"show_ratio '{ratio}' must be one of sin/cos/tan")

    id_set = set(ids)
    for side in (spec.get("side_labels") or {}):
        side = str(side)
        if len(side) != 2 or side[0] not in id_set or side[1] not in id_set:
            issues.append(f"side_labels key '{side}' references a point not in points[]")

    return (len(issues) == 0), issues


def _verify_trigonometry_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """Post-render check for 'trigonometry': the right-angle mark must
    be present, the angle arc+label must be present if angle_at was
    declared, and every valid side_labels entry must have its label
    actually drawn."""
    points = spec.get("points") or []
    ids = [p.get("id") for p in points if isinstance(p, dict) and p.get("id")]
    id_set = set(ids)

    expected_right_angle = 1 if spec.get("right_angle_at") in id_set else 0
    expected_angle_mark = 1 if (spec.get("angle_at") in id_set and spec.get("angle_at") != spec.get("right_angle_at")) else 0
    expected_side_labels = sum(1 for side, text in (spec.get("side_labels") or {}).items()
                                if text and len(str(side)) == 2 and str(side)[0] in id_set and str(side)[1] in id_set)
    expected_dims = sum(1 for k in ("adjacent", "opposite", "hypotenuse")
                         if k in (spec.get("dimensions") or {}))

    issues = []
    if svg.count('data-role="right-angle"') < expected_right_angle:
        issues.append("expected the right-angle symbol, none rendered")
    if svg.count('data-role="angle-arc"') < expected_angle_mark:
        issues.append("expected the marked angle's arc, none rendered")
    actual_side_labels = svg.count('data-role="side-label"')
    if actual_side_labels < expected_side_labels:
        issues.append(f"expected {expected_side_labels} side label(s), only "
                       f"{actual_side_labels} rendered")
    actual_dims = svg.count('data-role="dimension-label"') # This now checks for adjacent, opposite, and hypotenuse labels
    if actual_dims < expected_dims:
        issues.append(f"expected {expected_dims} dimension label(s), only "
                       f"{actual_dims} rendered")

    return (len(issues) == 0), issues


# ------------------------------------------------------------------
# STATISTICS — bar chart / histogram for frequency-distribution
# questions (Class 8-10: "draw a bar graph for the following data").
# Now also supports pie charts, frequency polygons, and ogives.
# Every bar height is computed to scale from the declared numeric
# values — nothing here is eyeballed or decorative.
# ------------------------------------------------------------------
def _estimate_text_width(text: str, font_size: float) -> float:
    """Rough glyph-width estimate for this renderer's font stack (Noto
    Sans / Arial fallback). Bengali/Assamese glyphs run noticeably
    wider than Latin at the same font-size (conjuncts, matras), so a
    single average-width-per-character factor tuned for Latin (~0.5)
    undercounts real Assamese label width. 0.62 is a conservative
    estimate that errs toward rotating a label that would have JUST
    fit horizontally, rather than leaving a genuinely too-long one
    unrotated and overlapping its neighbor."""
    return len(text) * font_size * 0.62


def _parse_class_interval(cat: str):
    """Parses a class-interval label like "20-40" or "20 - 40" into
    (lower, upper) floats, or None if it doesn't look like one. Used
    to detect whether a histogram's classes have unequal widths, in
    which case bar width/height must be computed from the real
    interval boundaries (see _render_bar_or_histogram's own docstring
    note below) rather than treated as N equal slots."""
    if not isinstance(cat, str):
        return None
    m = re.match(r'^\s*(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)\s*$', cat)
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    return (lo, hi) if hi > lo else None


def _render_bar_or_histogram(spec: dict) -> str:
    categories = [str(c) for c in (spec.get("categories") or [])]
    values = spec.get("values") or []
    chart_type = spec.get("chart_type")

    pad_l, pad_r, pad_t = 40, 20, 24
    n = len(values)
    body = []

    if n == 0:
        return _svg("")

    # ------------------------------------------------------------------
    # PRODUCTION-AUDIT FIX: a histogram's classes are not always equal
    # width (e.g. "0-10, 10-20, 20-40, 40-50" — a real, standard Class
    # 9/10 topic). The mathematically correct convention (the entire
    # point of a histogram, as opposed to a bar chart) is that a bar's
    # AREA — not its height alone — represents its class's frequency:
    # width = the class's real interval width, height = frequency
    # DENSITY (frequency / width), never raw frequency at a uniform
    # width. Before this fix, every class was drawn at an equal slot
    # width regardless of its real interval, with height set directly
    # from the raw frequency — for a class twice as wide as the others,
    # this both misrepresents its visual weight (area) and its height
    # (shows the full frequency at the narrow width's scale, exaggerating
    # its apparent size). Detected here by parsing every category label
    # as a "lower-upper" numeric interval; if ALL of them parse AND the
    # widths are not all equal, real interval-based geometry is used.
    # Any category that doesn't parse as an interval, a bar chart
    # (chart_type != "histogram"), or classes that already happen to be
    # equal width all fall through to the exact original equal-slot
    # rendering — for the equal-width case this produces numerically
    # identical bar geometry either way (proportional density === 
    # proportional frequency when every width is the same), so there is
    # no behavior change for the common case, only for genuinely unequal
    # class widths, which were rendered wrong before this fix.
    # ------------------------------------------------------------------
    intervals = [_parse_class_interval(c) for c in categories] if chart_type == "histogram" else []
    use_density = (
        chart_type == "histogram" and len(intervals) == n and all(iv is not None for iv in intervals)
        and len({round(iv[1] - iv[0], 9) for iv in intervals}) > 1  # genuinely unequal widths present
    )

    if use_density:
        widths = [iv[1] - iv[0] for iv in intervals]
        global_lo, global_hi = intervals[0][0], intervals[-1][1]
        span = max(global_hi - global_lo, 1e-9)
        densities = []
        for v, w in zip(values, widths):
            try:
                densities.append(float(v) / w if w > 0 else 0.0)
            except (TypeError, ValueError):
                densities.append(0.0)
        max_val = max(1.0, max(densities))
    else:
        max_val = max(1.0, max(float(v) for v in values if _is_number(v)))

    # "nice" step for the y-axis gridlines/labels (1, 2, 5, 10, 20, 50, ...)
    raw_step = max_val / 4
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    for mult in (1, 2, 5, 10):
        step = mult * magnitude
        if step >= raw_step:
            break
    top = step * math.ceil(max_val / step)

    plot_w_for_slots = W - pad_l - pad_r
    slot_w = plot_w_for_slots / n
    label_font_size = 8

    # ------------------------------------------------------------------
    # BUG FIX (V8 polishing-phase audit): x-axis category labels used to
    # always render as a single horizontal <text> centered under each
    # bar with NO width check at all. Confirmed on a real generated PDF
    # — a WHO causes-of-death chart with 8 long category names (e.g.
    # "প্ৰসৱকালীন স্বাস্থ্যৰ অৱস্থা") packed into narrow bar slots
    # produced completely unreadable overlapping/interleaved text along
    # the x-axis. Fix: any label estimated wider than its own bar slot
    # (with margin) is rotated -40° and right-anchored at the bar's
    # center instead — the standard textbook/chart convention for long
    # category names — and the bottom padding is enlarged up front to
    # give rotated labels room, rather than letting them clip off the
    # bottom of the figure.
    # ------------------------------------------------------------------
    any_rotated = any(
        _estimate_text_width(cat, label_font_size) > slot_w * 1.15
        for cat in categories
    ) if categories else False
    pad_b = 78 if any_rotated else 46

    plot_h = H - pad_t - pad_b

    origin = (pad_l, H - pad_b)
    body.append(f'<line x1="{origin[0]}" y1="{pad_t}" x2="{origin[0]}" y2="{origin[1]}" '
                f'stroke="{LINE_COLOR}" stroke-width="1.5" data-role="axis"/>')
    body.append(f'<line x1="{origin[0]}" y1="{origin[1]}" x2="{W - pad_r}" y2="{origin[1]}" '
                f'stroke="{LINE_COLOR}" stroke-width="1.5" data-role="axis"/>')

    y_ticks = int(round(top / step))
    for i in range(y_ticks + 1):
        val = i * step
        y = origin[1] - (val / top) * plot_h
        body.append(f'<line x1="{origin[0]-3}" y1="{y:.1f}" x2="{origin[0]}" y2="{y:.1f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="1"/>')
        body.append(f'<text x="{origin[0]-6}" y="{y+3:.1f}" text-anchor="end" fill="#555" '
                    f'font-family="Noto Sans, Arial" font-size="8" data-role="y-tick">{val:g}</text>')

    bar_gap = 0 if chart_type == "histogram" else slot_w * 0.25
    bar_w_default = slot_w - bar_gap
    for i, (cat, val) in enumerate(zip(categories or [""] * n, values)):
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if use_density:
            lo, hi = intervals[i]
            x = origin[0] + ((lo - global_lo) / span) * plot_w_for_slots
            bar_w = ((hi - lo) / span) * plot_w_for_slots
            bar_h = (densities[i] / top) * plot_h if top else 0
        else:
            bar_w = bar_w_default
            x = origin[0] + i * slot_w + bar_gap / 2
            bar_h = (v / top) * plot_h if top else 0
        y = origin[1] - bar_h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                    f'fill="#5b8fd6" stroke="{LINE_COLOR}" stroke-width="1" data-role="bar"/>')
        if cat:
            cx = x + bar_w / 2
            if _estimate_text_width(cat, label_font_size) > slot_w * 1.15:
                body.append(f'<text x="{cx:.1f}" y="{origin[1]+12}" text-anchor="end" '
                            f'fill="#333" font-family="Noto Sans, Arial" font-size="{label_font_size}" '
                            f'transform="rotate(-40 {cx:.1f} {origin[1]+12})" data-role="x-tick-rotated">{cat}</text>')
            else:
                body.append(f'<text x="{cx:.1f}" y="{origin[1]+12}" text-anchor="middle" '
                            f'fill="#333" font-family="Noto Sans, Arial" font-size="{label_font_size}" '
                            f'data-role="x-tick">{cat}</text>')

    if spec.get("x_label"):
        body.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" fill="#555" font-size="9">{spec["x_label"]}</text>')
    if spec.get("y_label"):
        body.append(f'<text x="10" y="{H/2}" text-anchor="middle" fill="#555" font-size="9" '
                    f'transform="rotate(-90 10 {H/2})">{spec["y_label"]}</text>')
    elif use_density:
        # Standard textbook convention: an unequal-class-width histogram's
        # y-axis is labelled "Frequency Density" (কম্পাংক ঘনত্ব), not
        # "Frequency" — the axis no longer reads as raw frequency once
        # density scaling is in effect, and a reader needs that label to
        # correctly interpret the chart, exactly like a real textbook would.
        body.append(f'<text x="10" y="{H/2}" text-anchor="middle" fill="#555" font-size="9" '
                    f'transform="rotate(-90 10 {H/2})" data-role="y-label-auto">Frequency Density</text>')

    return _svg("".join(body))


def _render_pie_chart(spec: dict) -> str:
    values = [float(v) for v in spec.get("values", []) if _is_number(v)]
    categories = [str(c) for c in spec.get("categories", [])]
    total = sum(values)
    if not total:
        return _svg("")

    cx, cy, r = W / 2, H / 2, min(W, H) / 2 - 40
    body = []
    start_angle = -90
    colors = ["#5b8fd6", "#d65b8f", "#8fd65b", "#d68f5b", "#5bd6d6", "#d6d65b"]

    label_positions = []
    for i, value in enumerate(values):
        angle = (value / total) * 360
        end_angle = start_angle + angle
        mid_angle_rad = math.radians((start_angle + end_angle) / 2)

        sx, sy = cx + r * math.cos(math.radians(start_angle)), cy + r * math.sin(math.radians(start_angle))
        ex, ey = cx + r * math.cos(math.radians(end_angle)), cy + r * math.sin(math.radians(end_angle))
        large_arc = 1 if angle > 180 else 0
        path = f'M {cx},{cy} L {sx},{sy} A {r},{r} 0 {large_arc},1 {ex},{ey} Z'
        body.append(f'<path d="{path}" fill="{colors[i % len(colors)]}" data-role="pie-slice"/>')

        # Label placement logic
        label_r = r + 15
        lx, ly = cx + label_r * math.cos(mid_angle_rad), cy + label_r * math.sin(mid_angle_rad)
        label_positions.append({'x': lx, 'y': ly, 'text': categories[i] if i < len(categories) else f"{value:g}"})
        start_angle = end_angle

    # Adjust label positions to prevent overlap
    for i in range(len(label_positions)):
        for j in range(i + 1, len(label_positions)):
            p1, p2 = label_positions[i], label_positions[j]
            dist = math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])
            if dist < 20: # Overlap detected
                # Push labels apart along the line connecting them
                dx, dy = p2['x'] - p1['x'], p2['y'] - p1['y']
                overlap = (20 - dist) / 2
                p1['x'] -= overlap * dx / dist
                p1['y'] -= overlap * dy / dist
                p2['x'] += overlap * dx / dist
                p2['y'] += overlap * dy / dist

    for pos in label_positions:
        anchor = "middle"
        if pos['x'] > cx + 10: anchor = "start"
        if pos['x'] < cx - 10: anchor = "end"
        body.append(f'<text x="{pos["x"]:.1f}" y="{pos["y"]:.1f}" text-anchor="{anchor}" dominant-baseline="middle" fill="{LABEL_COLOR}" font-size="10" data-role="pie-label">{pos["text"]}</text>')

    return _svg("".join(body))


def _render_line_chart(spec: dict) -> str:
    """Renders frequency polygons and ogives."""
    categories = spec.get("categories", [])
    values = [float(v) for v in spec.get("values", []) if _is_number(v)]

    if not values or not categories or len(values) != len(categories):
        return _svg("")

    pad_l, pad_r, pad_t, pad_b = 50, 20, 24, 46
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    origin = (pad_l, H - pad_b)

    # X-axis represents category midpoints or boundaries
    x_coords = [float(c) for c in categories]
    min_x, max_x = min(x_coords), max(x_coords)
    span_x = max_x - min_x or 1

    # Y-axis represents frequency or cumulative frequency
    min_y, max_y = 0, max(1.0, max(values))
    span_y = max_y - min_y

    def to_svg(x, y):
        svg_x = origin[0] + ((x - min_x) / span_x) * plot_w
        svg_y = origin[1] - ((y - min_y) / span_y) * plot_h
        return svg_x, svg_y

    body = []
    # Axes and grid
    body.append(f'<line x1="{origin[0]}" y1="{pad_t}" x2="{origin[0]}" y2="{origin[1]}" stroke="{LINE_COLOR}" stroke-width="1.5" data-role="axis"/>')
    body.append(f'<line x1="{origin[0]}" y1="{origin[1]}" x2="{W - pad_r}" y2="{origin[1]}" stroke="{LINE_COLOR}" stroke-width="1.5" data-role="axis"/>')

    # Ticks
    x_ticks = _get_nice_steps(min_x, max_x, 6)
    y_ticks = _get_nice_steps(min_y, max_y, 5)
    for val in x_ticks:
        x, _ = to_svg(val, 0)
        body.append(f'<line x1="{x:.1f}" y1="{origin[1]}" x2="{x:.1f}" y2="{origin[1]+4}" stroke="{LINE_COLOR}" stroke-width="1"/>')
        body.append(f'<text x="{x:.1f}" y="{origin[1]+16}" text-anchor="middle" font-size="9">{val:g}</text>')
    for val in y_ticks:
        _, y = to_svg(0, val)
        body.append(f'<line x1="{origin[0]-4}" y1="{y:.1f}" x2="{origin[0]}" y2="{y:.1f}" stroke="{LINE_COLOR}" stroke-width="1"/>')
        body.append(f'<text x="{origin[0]-8}" y="{y+3:.1f}" text-anchor="end" font-size="9">{val:g}</text>')

    # Plot line
    points_str = " ".join(f"{to_svg(x, y)[0]:.1f},{to_svg(x, y)[1]:.1f}" for x, y in zip(x_coords, values))
    body.append(f'<polyline points="{points_str}" fill="none" stroke="{AUX_COLOR}" stroke-width="2" data-role="line-plot"/>')

    return _svg("".join(body))


def _validate_statistics_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == 'statistics': values[]
    must be a non-empty list of non-negative numbers, and if
    categories[] is given it must be the same length as values[]."""
    issues = []
    values = spec.get("values")
    if not isinstance(values, list) or not values:
        issues.append("statistics requires a non-empty 'values' list")
        return False, issues

    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            issues.append(f"value '{v}' is not numeric")
            continue
        if fv < 0:
            issues.append(f"value '{v}' is negative — frequencies/heights cannot be negative")

    categories = spec.get("categories")
    if categories is not None:
        if not isinstance(categories, list):
            issues.append("'categories' must be a list")
        elif len(categories) != len(values):
            issues.append(f"'categories' has {len(categories)} entries but 'values' has {len(values)}")

    chart_type = spec.get("chart_type")
    valid_types = ("bar", "histogram", "pie", "frequency_polygon", "ogive")
    if chart_type not in valid_types:
        issues.append(f"chart_type '{chart_type}' must be one of {valid_types}")

    if chart_type in ("frequency_polygon", "ogive") and not all(_is_number(c) for c in categories):
        issues.append(f"chart_type '{chart_type}' requires numeric categories (class marks or boundaries)")

    return (len(issues) == 0), issues


def _verify_statistics_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """Post-render check for 'statistics': every numeric value must
    have produced exactly one rendered bar, and the axes must both be
    present."""
    values = spec.get("values") or []
    expected_bars = sum(1 for v in values if _is_number(v))

    issues = []
    chart_type = spec.get("chart_type")
    if chart_type in ("bar", "histogram"):
        actual_bars = svg.count('data-role="bar"')
        actual_axes = svg.count('data-role="axis"')
        if actual_bars < expected_bars:
            issues.append(f"expected {expected_bars} bar(s), only {actual_bars} rendered")
        if actual_axes < 2:
            issues.append(f"expected 2 axes (x and y), only {actual_axes} rendered")
    elif chart_type == "pie":
        actual_slices = svg.count('data-role="pie-slice"')
        if actual_slices < expected_bars:
            issues.append(f"expected {expected_bars} pie slice(s), only {actual_slices} rendered")
    elif chart_type in ("frequency_polygon", "ogive"):
        if svg.count('data-role="line-plot"') < 1:
            issues.append(f"expected a line plot, none rendered")
            
    return (len(issues) == 0), issues


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------
# SURFACE AREA & VOLUME — a labelled pseudo-3D projection of a solid
# (cuboid/cube/cylinder/cone/sphere), with every dimension the spec
# declares drawn as an actual labelled edge, not just printed as
# floating text. The projection geometry (isometric-style skew for the
# cuboid, ellipse foreshortening for the round solids) is fixed and
# deterministic — Gemini supplies only which solid and its dimensions.
# ------------------------------------------------------------------
_SOLID_REQUIRED_DIMS = {
    "cuboid": ("length", "width", "height"),
    "cube": ("side",),
    "cylinder": ("radius", "height"),
    "cone": ("radius", "height"),
    "sphere": ("radius",),
}


def _render_surface_area_volume(spec: dict) -> str:
    solid = spec.get("solid")
    dims = spec.get("dimensions") or {}
    labels = spec.get("labels") or {}
    body = []
    cx, cy = W / 2, H / 2 + 10

    def dim_label(key, fallback):
        return labels.get(key, fallback)

    if solid in ("cuboid", "cube"):
        if solid == "cube":
            s = float(dims.get("side", 90))
            l = w = h = s
        else:
            l, w, h = float(dims.get("length", 100)), float(dims.get("width", 55)), float(dims.get("height", 75))
        scale = min((W - 120) / (l + w*0.4), (H - 100) / (h + w*0.3))
        l, w, h = l*scale, w*scale, h*scale
        skew_x, skew_y = w * 0.4, w * 0.3
        # front-bottom-left corner
        flb = (cx - l / 2, cy + h / 2)
        frb = (flb[0] + l, flb[1])
        flt = (flb[0], flb[1] - h)
        frt = (frb[0], frb[1] - h)
        blb = (flb[0] + skew_x, flb[1] - skew_y)
        brb = (frb[0] + skew_x, frb[1] - skew_y)
        blt = (flt[0] + skew_x, flt[1] - skew_y)
        brt = (frt[0] + skew_x, frt[1] - skew_y)

        # front face solid, top + side face lighter (classic textbook
        # "see-through" cuboid so the hidden back edges are still visible)
        body.append(f'<polygon points="{flb[0]:.0f},{flb[1]:.0f} {frb[0]:.0f},{frb[1]:.0f} '
                    f'{frt[0]:.0f},{frt[1]:.0f} {flt[0]:.0f},{flt[1]:.0f}" '
                    f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<polygon points="{flt[0]:.0f},{flt[1]:.0f} {frt[0]:.0f},{frt[1]:.0f} '
                    f'{brt[0]:.0f},{brt[1]:.0f} {blt[0]:.0f},{blt[1]:.0f}" '
                    f'fill="#d7e4f7" stroke="{LINE_COLOR}" stroke-width="1.5" data-role="solid-outline"/>')
        body.append(f'<polygon points="{frb[0]:.0f},{frb[1]:.0f} {brb[0]:.0f},{brb[1]:.0f} '
                    f'{brt[0]:.0f},{brt[1]:.0f} {frt[0]:.0f},{frt[1]:.0f}" '
                    f'fill="#cddaf0" stroke="{LINE_COLOR}" stroke-width="1.5" data-role="solid-outline"/>')
        for p1, p2 in ((blb, blt), (blb, brb), (blb, flb)):
            body.append(f'<line x1="{p1[0]:.0f}" y1="{p1[1]:.0f}" x2="{p2[0]:.0f}" y2="{p2[1]:.0f}" '
                        f'stroke="{LINE_COLOR}" stroke-width="1" stroke-dasharray="3,2"/>')

        if solid == "cube":
            body.append(f'<text x="{(flb[0]+frb[0])/2:.0f}" y="{flb[1]+16:.0f}" text-anchor="middle" fill="{LABEL_COLOR}" {FONT} font-size="11" data-role="dimension-label">{dim_label("side", f"s={h_val(dims, 'side', s/scale)}")}</text>')
        else:
            body.append(f'<text x="{(flb[0]+frb[0])/2:.0f}" y="{flb[1]+16:.0f}" text-anchor="middle" fill="{LABEL_COLOR}" {FONT} font-size="11" data-role="dimension-label">{dim_label("length", f"l={h_val(dims, 'length', l/scale)}")}</text>')
            body.append(f'<text x="{flb[0]-8:.0f}" y="{(flb[1]+flt[1])/2:.0f}" text-anchor="end" fill="{LABEL_COLOR}" {FONT} font-size="11" data-role="dimension-label">{dim_label("height", f"h={h_val(dims, 'height', h/scale)}")}</text>')
            body.append(f'<text x="{(frb[0]+brb[0])/2+6:.0f}" y="{(frb[1]+brb[1])/2-2:.0f}" fill="{LABEL_COLOR}" {FONT} font-size="11" data-role="dimension-label">{dim_label("width", f"w={h_val(dims, 'width', w/scale)}")}</text>')

    elif solid == "cylinder":
        r_val, h_val_dim = float(dims.get("radius", 55)), float(dims.get("height", 110))
        scale = min((W - 80) / (2 * r_val), (H - 100) / h_val_dim)
        r, h = r_val * scale, h_val_dim * scale
        top_c = (cx, cy - h / 2)
        bot_c = (cx, cy + h / 2)
        ry = r * 0.32
        body.append(f'<line x1="{cx-r:.0f}" y1="{top_c[1]:.0f}" x2="{cx-r:.0f}" y2="{bot_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<line x1="{cx+r:.0f}" y1="{top_c[1]:.0f}" x2="{cx+r:.0f}" y2="{bot_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<ellipse cx="{bot_c[0]:.0f}" cy="{bot_c[1]:.0f}" rx="{r}" ry="{ry:.0f}" '
                    f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<path d="M {cx-r:.0f} {top_c[1]:.0f} A {r} {ry:.0f} 0 0 0 {cx+r:.0f} {top_c[1]:.0f}" '
                    f'fill="none" stroke="{LINE_COLOR}" stroke-width="1" stroke-dasharray="3,2"/>')
        body.append(f'<path d="M {cx-r:.0f} {top_c[1]:.0f} A {r} {ry:.0f} 0 0 1 {cx+r:.0f} {top_c[1]:.0f}" '
                    f'fill="none" stroke="{LINE_COLOR}" stroke-width="1.5"/>')
        body.append(f'<line x1="{cx:.0f}" y1="{top_c[1]:.0f}" x2="{cx:.0f}" y2="{bot_c[1]:.0f}" '
                    f'stroke="{AUX_COLOR}" stroke-width="1" stroke-dasharray="2,2"/>')
        height_text = dim_label("height", f"h={h_val(dims, 'height', h_val_dim)}")
        radius_text = dim_label("radius", f"r={h_val(dims, 'radius', r_val)}")
        body.append(f'<text x="{cx+8:.0f}" y="{cy:.0f}" fill="{LABEL_COLOR}" {FONT} font-size="11" '
                    f'data-role="dimension-label">{height_text}</text>')
        body.append(f'<text x="{cx:.0f}" y="{bot_c[1]+ry+16:.0f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                    f'{FONT} font-size="11" data-role="dimension-label">{radius_text}</text>')

    elif solid == "cone":
        r_val, h_val_dim = float(dims.get("radius", 55)), float(dims.get("height", 110))
        scale = min((W - 80) / (2 * r_val), (H - 100) / h_val_dim)
        r, h = r_val * scale, h_val_dim * scale
        apex = (cx, cy - h / 2)
        base_c = (cx, cy + h / 2)
        ry = r * 0.32
        body.append(f'<line x1="{apex[0]:.0f}" y1="{apex[1]:.0f}" x2="{cx-r:.0f}" y2="{base_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<line x1="{apex[0]:.0f}" y1="{apex[1]:.0f}" x2="{cx+r:.0f}" y2="{base_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<ellipse cx="{base_c[0]:.0f}" cy="{base_c[1]:.0f}" rx="{r}" ry="{ry:.0f}" '
                    f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<line x1="{apex[0]:.0f}" y1="{apex[1]:.0f}" x2="{cx:.0f}" y2="{base_c[1]:.0f}" '
                    f'stroke="{AUX_COLOR}" stroke-width="1" stroke-dasharray="2,2"/>')
        height_text = dim_label("height", f"h={h_val(dims, 'height', h_val_dim)}")
        radius_text = dim_label("radius", f"r={h_val(dims, 'radius', r_val)}")
        body.append(f'<text x="{cx+8:.0f}" y="{cy:.0f}" fill="{LABEL_COLOR}" {FONT} font-size="11" '
                    f'data-role="dimension-label">{height_text}</text>')
        body.append(f'<text x="{cx:.0f}" y="{base_c[1]+ry+16:.0f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                    f'{FONT} font-size="11" data-role="dimension-label">{radius_text}</text>')

    elif solid == "sphere":
        r_val = float(dims.get("radius", 70))
        r = min(r_val, (min(W, H) - 80) / 2)
        body.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="#eaf1fb" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        ry = r * 0.32
        body.append(f'<ellipse cx="{cx:.0f}" cy="{cy:.0f}" rx="{r}" ry="{ry:.0f}" '
                    f'fill="none" stroke="{LINE_COLOR}" stroke-width="1" stroke-dasharray="3,2"/>')
        body.append(f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{cx+r:.0f}" y2="{cy:.0f}" '
                    f'stroke="{AUX_COLOR}" stroke-width="1.3"/>')
        radius_text = dim_label("radius", f"r={h_val(dims, 'radius', r_val)}")
        body.append(f'<text x="{cx+r/2:.0f}" y="{cy-6:.0f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                    f'{FONT} font-size="11" data-role="dimension-label">{radius_text}</text>')

    return _svg("".join(body))


def h_val(dims: dict, key: str, fallback):
    """Formats a declared numeric dimension for its label (e.g. 'h=7'),
    falling back to the renderer's own fixed drawing size only if the
    spec genuinely didn't supply that dimension (should not normally
    happen — _validate_surface_area_volume_spec requires it)."""
    v = dims.get(key)
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return f"{fallback:g}"


def _validate_surface_area_volume_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == 'surface_area_volume':
    `solid` must be one of the 5 recognized families, and every
    dimension that family requires must be present as a positive
    number in `dimensions`."""
    issues = []
    solid = spec.get("solid")
    if solid not in _SOLID_REQUIRED_DIMS:
        issues.append(f"solid '{solid}' must be one of {sorted(_SOLID_REQUIRED_DIMS)}")
        return False, issues

    dims = spec.get("dimensions")
    if not isinstance(dims, dict):
        issues.append("'dimensions' must be an object")
        return False, issues

    for key in _SOLID_REQUIRED_DIMS[solid]:
        val = dims.get(key)
        try:
            fv = float(val)
        except (TypeError, ValueError):
            issues.append(f"solid '{solid}' requires a numeric '{key}' in dimensions")
            continue
        if fv <= 0:
            issues.append(f"dimension '{key}'={val} must be positive")

    return (len(issues) == 0), issues


def _verify_surface_area_volume_coverage(spec: dict, svg: str) -> tuple[bool, list[str]]:
    """Post-render check for 'surface_area_volume': the solid outline
    must have actually been drawn, and every required dimension for
    this solid must have produced a visible label."""
    solid = spec.get("solid")
    required = _SOLID_REQUIRED_DIMS.get(solid, ())
    issues = []
    if svg.count('data-role="solid-outline"') < 1:
        issues.append(f"expected a rendered outline for solid '{solid}', none found")
    actual_labels = svg.count('data-role="dimension-label"')
    if actual_labels < len(required):
        issues.append(f"expected {len(required)} dimension label(s) for '{solid}', only {actual_labels} rendered")
    return (len(issues) == 0), issues


# ------------------------------------------------------------------
# CONSTRUCTION — a step-by-step compass-and-ruler diagram (draw a base
# segment, swing arcs from known points, join the resulting apex) for
# CONSTRUCTION (অংকন) questions, as distinct from a finished-figure
# "triangle" diagram: this shows the actual physical build sequence
# the student performs, matching how a textbook's construction chapter
# is illustrated (see prompts.py's CONSTRUCTION QUESTIONS rule for the
# corresponding textual "steps" requirement).
# ------------------------------------------------------------------

def _circle_intersection(c1, r1, c2, r2):
    """Returns one of the two intersection points of the circles
    (c1, r1) and (c2, r2), or None if they don't intersect. Picks the
    point with the larger y (i.e. "above" the line joining the two
    centers), matching how a constructed triangle's apex is drawn
    above its base — a modelling choice, not a uniqueness guarantee."""
    x1, y1 = c1
    x2, y2 = c2
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)
    if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h_sq = r1 ** 2 - a ** 2
    if h_sq < 0:
        return None
    h = math.sqrt(h_sq)
    xm, ym = x1 + a * dx / d, y1 + a * dy / d
    xs1, ys1 = xm + h * dy / d, ym - h * dx / d
    xs2, ys2 = xm - h * dy / d, ym + h * dx / d
    return (xs1, ys1) if ys1 >= ys2 else (xs2, ys2)


def _validate_construction_spec(spec: dict) -> tuple[bool, list[str]]:
    """Structural validation for diagram_type == 'construction': at
    least one initial_point with numeric x/y, at least one
    construction_step, and every point id a step references
    ('from'/'to'/'center') must already be known at that point in the
    sequence — either an initial_point, or a point introduced earlier
    by an arc step's own 'intersection_id'. This mirrors the project's
    general "reject a dangling reference before it ever reaches the
    renderer" structural-validation philosophy used for every other
    diagram type above."""
    issues = []
    initial_points = spec.get("initial_points") or []
    known = {p.get("id") for p in initial_points
             if isinstance(p, dict) and p.get("id") and _is_number(p.get("x")) and _is_number(p.get("y"))}
    if not known:
        issues.append("construction requires at least one initial_point with a numeric x/y")
        return False, issues

    steps = spec.get("construction_steps") or []
    if not steps:
        issues.append("construction requires at least one construction_step")
        return False, issues

    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            issues.append(f"construction_steps[{i}] must be an object")
            continue
        stype = s.get("type")
        if stype not in ("line_segment", "join", "arc"):
            issues.append(f"construction_steps[{i}]: unknown step type '{stype}'")
            continue
        if stype in ("line_segment", "join"):
            frm, to = s.get("from"), s.get("to")
            if frm not in known:
                issues.append(f"construction_steps[{i}]: 'from' point '{frm}' is not yet known")
            if to not in known:
                issues.append(f"construction_steps[{i}]: 'to' point '{to}' is not yet known")
        elif stype == "arc":
            center = s.get("center")
            if center not in known:
                issues.append(f"construction_steps[{i}]: 'center' point '{center}' is not yet known")
            if not _is_number(s.get("radius")) or float(s.get("radius", 0)) <= 0:
                issues.append(f"construction_steps[{i}]: arc requires a positive numeric 'radius'")
            new_id = s.get("intersection_id")
            if new_id:
                known.add(new_id)  # becomes known for any later step
    return (len(issues) == 0), issues


def _render_construction(spec: dict) -> str:
    """Renders the compass-and-ruler build sequence: initial_points are
    placed to scale, 'line_segment'/'join' steps draw a straight line
    between two already-known points, and 'arc' steps draw the compass
    swing from a center at a given radius. When two arc steps share the
    same intersection_id (the standard "two arcs from the two known
    endpoints" way a textbook constructs a third vertex), that point's
    position is solved as the circle-circle intersection and becomes
    available to any later step, e.g. the 'join' strokes that draw the
    two new sides."""
    initial_points = spec.get("initial_points") or []
    steps = spec.get("construction_steps") or []

    raw = {p["id"]: (float(p["x"]), float(p["y"])) for p in initial_points
           if isinstance(p, dict) and p.get("id")}
    xs = [c[0] for c in raw.values()] or [0.0]
    ys = [c[1] for c in raw.values()] or [0.0]
    max_radius = max([float(s.get("radius", 0)) for s in steps if s.get("type") == "arc"] or [0.0])
    span_x = max(max(xs) - min(xs), 2 * max_radius, 1.0)
    span_y = max(max(ys) - min(ys), 2 * max_radius, 1.0)
    scale = min((W - 80) / span_x, (H - 80) / span_y)
    x0, y0 = min(xs), min(ys)
    ox, oy = 40, H - 40

    def to_svg(x, y):
        return ox + (x - x0) * scale, oy - (y - y0) * scale

    known_xy = {pid: xy for pid, xy in raw.items()}      # math-space, for geometry
    known = {pid: to_svg(*xy) for pid, xy in raw.items()}  # svg-space, for drawing

    # Pre-solve every intersection point defined by (at least) two arcs
    # sharing an intersection_id, before drawing anything, so a later
    # 'join' step can always find it regardless of step order.
    #
    # PRODUCTION-AUDIT FIX: if two arcs that are SUPPOSED to define a
    # new vertex genuinely can't intersect (e.g. Gemini reported side
    # lengths that violate the triangle inequality, or any other
    # geometrically inconsistent construction), that vertex — and
    # therefore every 'join'/'line_segment' step that depends on it —
    # can never be drawn correctly. The old behavior silently skipped
    # just that one point and its dependent steps while still
    # returning a "successful" SVG showing the base and two dangling,
    # meaningless arcs with no triangle at all — passing every
    # existing validation gate (none of which checked "did every
    # declared point actually get drawn") and shipping a misleading
    # diagram. A construction step whose own geometry cannot be
    # solved is exactly the "deterministic rendering cannot guarantee
    # correctness" case — the whole diagram must be omitted, not
    # partially drawn.
    unresolved_intersections = []
    arc_defs: dict = {}
    for s in steps:
        if s.get("type") == "arc" and s.get("intersection_id"):
            arc_defs.setdefault(s["intersection_id"], []).append((s.get("center"), s.get("radius")))
    for new_id, defs in arc_defs.items():
        if new_id in known_xy or len(defs) < 2:
            continue
        (c1, r1), (c2, r2) = defs[0], defs[1]
        if c1 in known_xy and c2 in known_xy and _is_number(r1) and _is_number(r2):
            p = _circle_intersection(known_xy[c1], float(r1), known_xy[c2], float(r2))
            if p:
                known_xy[new_id] = p
                known[new_id] = to_svg(*p)
            else:
                unresolved_intersections.append(new_id)

    if unresolved_intersections:
        from utils import logger
        logger.warning(f"⚠️ Construction diagram: point(s) {unresolved_intersections} could not be "
                        f"located — the arcs meant to define them don't actually intersect (the "
                        f"given radii/centers are geometrically inconsistent, e.g. a triangle "
                        f"inequality violation) — omitting this diagram rather than publishing an "
                        f"incomplete construction missing its own vertex.")
        return ""

    body = []
    missing_step_points = []
    for s in steps:
        stype = s.get("type")
        if stype in ("line_segment", "join"):
            frm, to = s.get("from"), s.get("to")
            if frm in known and to in known:
                role = "construction-line" if stype == "line_segment" else "construction-join"
                (x1, y1), (x2, y2) = known[frm], known[to]
                body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                            f'stroke="{LINE_COLOR}" stroke-width="2" data-role="{role}"/>')
            else:
                missing_step_points.append((frm, to))
        elif stype == "arc":
            center, radius = s.get("center"), s.get("radius")
            if center in known and _is_number(radius) and float(radius) > 0:
                cx, cy = known[center]
                r_px = float(radius) * scale
                target_id = s.get("intersection_id")
                ang = (math.atan2(known[target_id][1] - cy, known[target_id][0] - cx)
                       if target_id in known else -math.pi / 4)
                a1, a2 = ang - 0.6, ang + 0.6
                sx, sy = cx + r_px * math.cos(a1), cy + r_px * math.sin(a1)
                ex, ey = cx + r_px * math.cos(a2), cy + r_px * math.sin(a2)
                body.append(f'<path d="M {sx:.1f} {sy:.1f} A {r_px:.1f} {r_px:.1f} 0 0 1 {ex:.1f} {ey:.1f}" '
                            f'fill="none" stroke="{LINE_COLOR}" stroke-width="1.3" data-role="construction-arc"/>')

    if missing_step_points:
        # Same class of defect as the unresolved-intersection case above:
        # a side the construction was supposed to draw silently vanished
        # instead of the whole diagram being omitted. Whatever the
        # specific cause (should already be rare — _validate_construction_
        # spec's structural check catches a step referencing an id that
        # was never declared at all), a construction missing one of its
        # own drawn sides is not safe to publish as "complete".
        from utils import logger
        logger.warning(f"⚠️ Construction diagram: step(s) referencing point(s) "
                        f"{missing_step_points} could not be drawn (an endpoint was never "
                        f"resolved) — omitting this diagram rather than publishing an "
                        f"incomplete construction missing one of its own sides.")
        return ""

    for pid, (x, y) in known.items():
        dy_off = 18 if y >= oy - 5 else -8
        body.append(_label((x, y), pid, dx=-4, dy=dy_off))
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{LINE_COLOR}"/>')

    return _svg("".join(body))


# ------------------------------------------------------------------
# NUMBER LINE — plain point placement, or a Pythagorean construction
# (e.g. "show √5 on the number line": O -> A (integer) -> perpendicular
# to B -> arc back down to the line at P).
# V8.2 FIX: now supports decimal ranges and constructions.
# ------------------------------------------------------------------
def _render_number_line(spec: dict) -> str:
    range_min = spec.get("range_min", 0)
    range_max = spec.get("range_max", max(range_min + 4, 4))
    if range_max <= range_min:
        range_max = range_min + 4

    pad_l, pad_r = 26, 26
    line_y = H - 45
    span = range_max - range_min
    px_per_unit = (W - pad_l - pad_r) / span

    def x_of(value):
        return pad_l + (value - range_min) * px_per_unit

    body = []
    x_start, x_end = x_of(range_min), x_of(range_max)
    body.append(_line((x_start - 8, line_y), (x_end + 8, line_y)))
    # arrowheads
    body.append(f'<polygon points="{x_end+8-6},{line_y-4} {x_end+8},{line_y} {x_end+8-6},{line_y+4}" fill="{LINE_COLOR}"/>')
    body.append(f'<polygon points="{x_start-8+6},{line_y-4} {x_start-8},{line_y} {x_start-8+6},{line_y+4}" fill="{LINE_COLOR}"/>')

    ticks = _get_nice_steps(range_min, range_max, 10)
    for val in ticks:
        x = x_of(val)
        body.append(_line((x, line_y - 4), (x, line_y + 4)))
        body.append(_label((x, line_y), f"{val:g}", dy=16))

    construction = spec.get("construction")
    marked = spec.get("marked_point", {})
    if construction:
        base_val = construction.get("base_point", 0)
        perp_len = construction.get("perpendicular_length", 1)
        o_x = x_of(0)
        a_x = x_of(base_val)
        b_y = line_y - perp_len * px_per_unit

        body.append(_line((a_x, line_y), (a_x, b_y), color="#c0392b"))     # A-B (perpendicular leg)
        body.append(_line((o_x, line_y), (a_x, b_y), color=LINE_COLOR))    # O-B (hypotenuse)
        body.append(f'<rect x="{a_x-6}" y="{line_y-6}" width="6" height="6" '
                     f'fill="none" stroke="{LINE_COLOR}" stroke-width="1.2"/>')  # right-angle mark at A

        radius = math.hypot(a_x - o_x, b_y - line_y)
        # arc from B down to the number line, centered at O
        landing_x = o_x + radius
        body.append(f'<path d="M {a_x:.1f},{b_y:.1f} A {radius:.1f},{radius:.1f} 0 0,1 '
                     f'{landing_x:.1f},{line_y:.1f}" fill="none" stroke="#c0392b" '
                     f'stroke-width="1.5" stroke-dasharray="4,2"/>')

        body.append(_label((o_x, line_y), "O", dy=-8))
        body.append(_label((a_x, b_y), "A" if base_val != 0 else "B", dy=-8))
        if perp_len:
            body.append(_label((a_x, (line_y + b_y) / 2), str(perp_len), dx=10))
        body.append(f'<circle cx="{landing_x:.1f}" cy="{line_y}" r="3" fill="#c0392b"/>')
        body.append(_label((landing_x, line_y), marked.get("label", "P"), dy=-8, dx=6))
    elif marked:
        mx = x_of(marked.get("value", (range_min + range_max) / 2))
        body.append(f'<circle cx="{mx:.1f}" cy="{line_y}" r="3.5" fill="#c0392b"/>')
        body.append(_label((mx, line_y), marked.get("label", ""), dy=-10))

    return _svg("".join(body))


# ------------------------------------------------------------------
# SQUARE ROOT SPIRAL — chained right triangles from an origin, each
# hypotenuse OP_n = sqrt(n), used for the classic "construct a square
# root spiral" Class 9 activity.
# ------------------------------------------------------------------
def _render_square_root_spiral(spec: dict) -> str:
    steps = max(2, min(int(spec.get("steps", 6)), 10))

    pts = [(0.0, 0.0), (1.0, 0.0)]  # O = P0, P1 one unit along the x-axis
    for i in range(1, steps):
        px, py = pts[i]
        theta = math.atan2(py, px)
        nxt = theta + math.pi / 2
        pts.append((px + math.cos(nxt), py + math.sin(nxt)))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 34
    scale = min((W - 2 * pad) / max(maxx - minx, 0.001), (H - 2 * pad) / max(maxy - miny, 0.001))

    def to_canvas(p):
        # flip y (SVG y grows downward; our construction used math y-up)
        x = pad + (p[0] - minx) * scale
        y = H - pad - (p[1] - miny) * scale
        return (x, y)

    coords = [to_canvas(p) for p in pts]
    origin = coords[0]

    body = []
    for i in range(1, len(coords)):
        body.append(_line(coords[i - 1], coords[i]))                 # unit leg
        if i >= 2:
            body.append(f'<line x1="{origin[0]}" y1="{origin[1]}" x2="{coords[i][0]}" y2="{coords[i][1]}" '
                         f'stroke="#c0392b" stroke-width="1" stroke-dasharray="3,2"/>')
            mx, my = (origin[0] + coords[i][0]) / 2, (origin[1] + coords[i][1]) / 2
            body.append(f'<text x="{mx:.1f}" y="{my:.1f}" fill="#c0392b" '
                         f'font-family="Noto Sans, Arial" font-size="9">\u221a{i}</text>')

    for i, (x, y) in enumerate(coords):
        label = "O" if i == 0 else f"P{i}"
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{LINE_COLOR}"/>')
        body.append(_label((x, y), label, dy=-7))

    return _svg("".join(body))


def _render_statistics(spec: dict) -> str:
    chart_type = spec.get("chart_type")
    if chart_type in ("bar", "histogram"):
        return _render_bar_or_histogram(spec)
    if chart_type == "pie":
        return _render_pie_chart(spec)
    if chart_type in ("frequency_polygon", "ogive"):
        return _render_line_chart(spec)
    return ""


_RENDERERS = {
    "triangle": _render_triangle,
    "circle": _render_circle,
    "angle": _render_angle,
    "parallel_lines": _render_parallel_lines,
    "coordinate_plot": _render_coordinate_plot,
    "quadrilateral": _render_quadrilateral,
    "trigonometry": _render_trigonometry,
    "statistics": _render_statistics,
    "surface_area_volume": _render_surface_area_volume,
    "construction": _render_construction,
    "number_line": None,          # registered below, after their definitions
    "square_root_spiral": None,
}

_RENDERERS["number_line"] = _render_number_line
_RENDERERS["square_root_spiral"] = _render_square_root_spiral


def render_diagram(spec: dict | None) -> str:
    """Returns an <svg>...</svg> string, or "" if spec is None/unrecognized
    /fails to render. A diagram is a nice-to-have on top of an already
    fully-solved question — it must NEVER be able to take the whole
    exercise down with it. Any renderer exception is caught here and
    logged, degrading to 'no diagram for this one question' instead of
    losing every other already-solved question in the exercise."""
    if not spec:
        return ""
    diagram_type = spec.get("diagram_type")
    renderer = _RENDERERS.get(diagram_type)
    if not renderer and not diagram_plugin_registry.is_plugin_type(diagram_type):
        return ""

    from utils import logger
    normalized = _normalize_spec(spec)
    ok, issues = validate_diagram_spec(normalized)
    if not ok:
        logger.warning(f"⚠️ diagram_spec for type '{diagram_type}' failed structural validation "
                        f"({'; '.join(issues)}) — omitting rather than showing a wrong diagram.")
        return ""
    try:
        if renderer:
            svg = renderer(normalized)
        else:
            svg = diagram_plugin_registry.render_via_plugin(normalized) or ""
    except Exception as e:
        logger.warning(f"⚠️ Diagram rendering failed for type '{diagram_type}' ({e}) — omitting this diagram.")
        return ""

    # UNIVERSAL SMART LABEL LAYOUT — applied here, at the single choke
    # point every built-in renderer AND every plugin's output passes
    # through, so automatic label-overlap repositioning covers every
    # diagram type, not only the three plugins (unit_circle,
    # circle_line_intersection, solid_net/cross_section) that happened
    # to call label_layout.check_and_fix_labels themselves before
    # returning. Those three still call it internally too (harmless —
    # check_and_fix_labels is idempotent: a second pass over an
    # already-collision-free SVG finds nothing to move), but every
    # other built-in shape (triangle, circle, angle, parallel_lines,
    # coordinate_plot, quadrilateral, trigonometry, statistics,
    # surface_area_volume, construction, number_line,
    # square_root_spiral) and every other plugin previously got none of
    # this. Only <text> elements are ever touched (see label_layout.py),
    # never the <line>/<polyline> tick-mark or right-angle markers that
    # verify_marker_coverage below counts, so this can never affect that
    # check's result. Never raises and never blocks rendering — a
    # layout pass that fails for any reason simply leaves the SVG as
    # produced by the renderer, exactly the same fail-open guarantee
    # label_layout.py itself documents.
    try:
        layout_result = label_layout.check_and_fix_labels(svg)
        svg = layout_result.get("svg", svg)
        if layout_result.get("moved"):
            logger.info(f"🏷️ diagram_type '{diagram_type}': repositioned "
                        f"{layout_result['moved']} overlapping label(s).")
        if layout_result.get("unresolved"):
            logger.warning(f"⚠️ diagram_type '{diagram_type}': "
                            f"{layout_result['unresolved']} label overlap(s) could not be "
                            f"resolved within the search budget — shipped best-effort.")
    except Exception as e:
        logger.warning(f"⚠️ Smart label layout pass failed for diagram_type '{diagram_type}' "
                        f"({e}) — continuing with the renderer's original label placement.")

    # MATH SANITIZATION FOR DIAGRAM LABELS (math_sanitizer.py) — the
    # SVG-side counterpart to the sanitize_math_text() gateway every
    # text field passes through. Runs here, at the single choke point
    # every built-in renderer AND every plugin's output already passes
    # through (see the label-layout pass above), so it covers every
    # diagram type with no per-renderer/per-plugin opt-in required.
    # Guarantees a leaked "\frac{1}{2}" / stray "\circ" / unbalanced
    # brace can never reach a <text> element in the printed page,
    # exactly like validate_and_finalize_spans guarantees for prose —
    # never raises (see sanitize_svg_labels's own docstring).
    svg = math_sanitizer.sanitize_svg_labels(svg)

    coverage_ok, coverage_issues = verify_marker_coverage(normalized, svg)
    if not coverage_ok:
        logger.warning(f"⚠️ diagram_spec for type '{diagram_type}' rendered but failed the post-render "
                        f"marker-coverage check ({'; '.join(coverage_issues)}) — a required equal-mark or "
                        f"right-angle symbol is missing from the actual output, so this diagram is being "
                        f"rejected rather than published incomplete.")
        return ""
    return svg
