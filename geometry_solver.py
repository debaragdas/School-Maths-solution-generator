"""
geometry_solver.py — THE GEOMETRY CONSTRAINT SOLVER.

Sits strictly between diagram_decision.py's approved DiagramSpec and
diagram_renderer.py's SVG drawing. Its only job: given whatever actual
geometric measurements (side lengths, angle values) a question stated
and Gemini transcribed into the spec, compute PROVABLY CONSISTENT,
to-scale (x, y) coordinates for the figure's points — or, if the given
measurements don't uniquely and validly determine one, return None and
let the caller fall back to the existing schematic placement.

This module makes NO judgement about what the question means (that is
Stage 1's job, done once, in prompts.py/solver.py) and draws NOTHING
(that is diagram_renderer.py's job). It only does arithmetic: the same
role a hand calculator plays for a student who already knows which
formula to use — it never decides which triangle the question is
about, it only solves the one it's handed.

Design note on WHY this exists (see the architecture design doc,
Section 5): diagram_renderer._render_triangle previously placed every
triangle's vertices at three FIXED screen positions regardless of the
question's actual side lengths or angles — correct for generic
"prove AB=AC"-style congruence sketches, but geometrically wrong
whenever a diagram needs to reflect real measurements (a 30-60-90
triangle, a specific angle value, a to-scale construction).
_render_trigonometry already proved the fix works for ONE fixed shape
(a right triangle from adjacent/opposite) — this module generalizes
that exact technique (compute real vertex positions, then fit-to-scale
to the canvas) to any triangle solvable by SSS / SAS / ASA / AAS.

BACKWARD COMPATIBILITY: every function here is purely additive. No
existing diagram_spec (none of which ever populated the new optional
"side_lengths"/"angles_deg" fields, since they didn't exist before
this module) can ever reach a code path in this module that returns
anything other than (None, "..."). The caller (diagram_renderer.py)
treats a None result as "use the existing schematic fallback, exactly
as before" — see the integration point in _render_triangle.
"""
import math

# Small floating-point tolerances — NOT fuzziness knobs for "close
# enough geometry", only slack for the accumulation of float error
# across a handful of trig operations (a few ULPs, not a modeling
# choice). Any real disagreement (a degree or more, a real triangle-
# inequality violation) is still rejected.
_ANGLE_SUM_TOLERANCE_DEG = 0.05
_RIGHT_ANGLE_TOLERANCE_DEG = 0.5
_TRIANGLE_INEQUALITY_TOLERANCE = 1e-6
_MIN_ANGLE_DEG = 0.01  # a "triangle" with a ~0 degree angle is degenerate

# Used only for the angles-only (no side length given) case, purely to
# fix a canvas scale — this does NOT claim metric accuracy for that
# case; see solve_triangle's own docstring for how that is surfaced.
_DEFAULT_BASE_LENGTH = 10.0


def _normalize_side_key(key: str) -> frozenset:
    """"AB" and "BA" must refer to the same side. A malformed key (not
    exactly 2 characters) normalizes to an empty frozenset, which will
    simply never match a real side lookup — never raises."""
    key = str(key)
    return frozenset(key) if len(key) == 2 and key[0] != key[1] else frozenset()


def _side_lengths_by_pair(ids: list, side_lengths: dict | None) -> dict:
    """Returns {frozenset({id1, id2}): length} for every side_lengths
    entry that refers to two of THIS triangle's own declared ids. Any
    entry referencing an unknown id, or with a non-numeric/non-positive
    length, is silently dropped (defensive — same philosophy as every
    other "never trust the model's exact shape" coercion elsewhere in
    this project) rather than raising and losing the whole diagram."""
    id_set = set(ids)
    result = {}
    for key, value in (side_lengths or {}).items():
        pair = _normalize_side_key(key)
        if len(pair) != 2 or not pair.issubset(id_set):
            continue
        try:
            length = float(value)
        except (TypeError, ValueError):
            continue
        if length > 0:
            result[pair] = length
    return result


def _angles_by_vertex(ids: list, angles_deg: dict | None) -> dict:
    """Returns {vertex_id: degrees} for every angles_deg entry that
    names one of this triangle's own declared ids with a plausible
    (0, 180) degree value. Same defensive-drop philosophy as above."""
    id_set = set(ids)
    result = {}
    for key, value in (angles_deg or {}).items():
        vid = str(key)
        if vid not in id_set:
            continue
        try:
            deg = float(value)
        except (TypeError, ValueError):
            continue
        if _MIN_ANGLE_DEG < deg < 180 - _MIN_ANGLE_DEG:
            result[vid] = deg
    return result


def _law_of_cosines_angle(opposite: float, adj1: float, adj2: float) -> float | None:
    """Returns the angle (degrees) opposite side `opposite`, given the
    other two sides adj1/adj2 — or None if the inputs can't form a
    valid triangle (division by zero, or the arccos argument falls
    outside [-1, 1], which happens exactly when the triangle inequality
    is violated)."""
    if adj1 <= 0 or adj2 <= 0:
        return None
    cos_val = (adj1 ** 2 + adj2 ** 2 - opposite ** 2) / (2 * adj1 * adj2)
    if cos_val < -1 - 1e-9 or cos_val > 1 + 1e-9:
        return None
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


def _validate_triangle_inequality(a: float, b: float, c: float) -> bool:
    """Strict triangle inequality on all three sides, with a small
    tolerance for float accumulation only."""
    return (a + b > c - _TRIANGLE_INEQUALITY_TOLERANCE and
            b + c > a - _TRIANGLE_INEQUALITY_TOLERANCE and
            c + a > b - _TRIANGLE_INEQUALITY_TOLERANCE)


def _place_from_angle_at_origin(origin_len_ab: float, angle_at_a_deg: float,
                                 len_ac: float) -> tuple:
    """Places A at (0,0), B at (origin_len_ab, 0), and returns C given
    the angle at A (between AB and AC) and the length AC. Pure math-
    space (y-up); caller flips/scales to screen space afterward."""
    rad = math.radians(angle_at_a_deg)
    return (len_ac * math.cos(rad), len_ac * math.sin(rad))


def _solve_coords_math_space(ids: list, sides: dict, angles: dict) -> tuple[dict | None, str]:
    """Core solver: given whatever normalized sides/angles were
    supplied (already filtered to this triangle's own 3 ids), attempts
    SSS -> SAS -> ASA/AAS -> angles-only, in that priority order, and
    returns {id: (x, y)} in arbitrary math-space units (A at origin,
    B along +x), or (None, reason).

    Every branch below runs its own consistency check on ITS OWN
    output before returning it — a solver that silently returns
    geometrically inconsistent coordinates would be worse than the
    schematic fallback it's meant to improve on, so nothing here is
    trusted without re-derivation."""
    a_id, b_id, c_id = ids[0], ids[1], ids[2]

    def pair(x, y):
        return frozenset({x, y})

    ab = sides.get(pair(a_id, b_id))
    bc = sides.get(pair(b_id, c_id))
    ca = sides.get(pair(c_id, a_id))

    # ---- SSS: all three sides given ----
    if ab is not None and bc is not None and ca is not None:
        if not _validate_triangle_inequality(ab, bc, ca):
            return None, f"SSS constraints violate the triangle inequality (AB={ab}, BC={bc}, CA={ca})"
        angle_a = _law_of_cosines_angle(opposite=bc, adj1=ab, adj2=ca)
        if angle_a is None:
            return None, "SSS: could not resolve a valid angle at the first vertex"
        A = (0.0, 0.0)
        B = (ab, 0.0)
        C = _place_from_angle_at_origin(ab, angle_a, ca)
        return {a_id: A, b_id: B, c_id: C}, "solved by SSS (all three sides given)"

    # ---- SAS: exactly two sides + the angle BETWEEN them ----
    angle_a = angles.get(a_id)
    angle_b = angles.get(b_id)
    angle_c = angles.get(c_id)

    if ab is not None and ca is not None and angle_a is not None:
        A, B = (0.0, 0.0), (ab, 0.0)
        C = _place_from_angle_at_origin(ab, angle_a, ca)
        return {a_id: A, b_id: B, c_id: C}, "solved by SAS (sides AB, CA + included angle A)"
    if ab is not None and bc is not None and angle_b is not None:
        # angle at B is between BA and BC; place B at origin instead.
        A_local = (ab, 0.0)
        rad = math.radians(angle_b)
        C_local = (bc * math.cos(rad), bc * math.sin(rad))
        # re-origin so A is at (0,0) for a consistent return convention
        ox, oy = A_local
        B_pt = (0.0 - ox, 0.0 - oy)
        A_pt = (0.0, 0.0)
        C_pt = (C_local[0] - ox, C_local[1] - oy)
        return {a_id: A_pt, b_id: B_pt, c_id: C_pt}, "solved by SAS (sides AB, BC + included angle B)"
    if bc is not None and ca is not None and angle_c is not None:
        C_pt = (0.0, 0.0)
        rad = math.radians(angle_c)
        A_local = (ca, 0.0)
        B_local = (bc * math.cos(rad), bc * math.sin(rad))
        return {a_id: A_local, b_id: B_local, c_id: C_pt}, "solved by SAS (sides CA, BC + included angle C)"

    # ---- ASA / AAS: two angles + at least one side ----
    known_angles = {k: v for k, v in (("A", angle_a), ("B", angle_b), ("C", angle_c)) if v is not None}
    if len(known_angles) >= 2:
        # derive the third angle deterministically
        a_deg, b_deg, c_deg = angle_a, angle_b, angle_c
        if angle_a is not None and angle_b is not None:
            c_deg = 180 - angle_a - angle_b
        elif angle_b is not None and angle_c is not None:
            a_deg = 180 - angle_b - angle_c
        elif angle_a is not None and angle_c is not None:
            b_deg = 180 - angle_a - angle_c
        if None in (a_deg, b_deg, c_deg) or min(a_deg, b_deg, c_deg) <= _MIN_ANGLE_DEG:
            return None, f"ASA/AAS: derived angles are degenerate (A={a_deg:.2f}, B={b_deg:.2f}, C={c_deg:.2f})"

        base_length = ab if ab is not None else (bc if bc is not None else (ca if ca is not None else None))
        if base_length is not None:
            # use law of sines to size the whole triangle from whichever
            # single side was actually given, whatever pair it belongs to
            if ab is not None:
                scale = ab / math.sin(math.radians(c_deg))
            elif bc is not None:
                scale = bc / math.sin(math.radians(a_deg))
            else:
                scale = ca / math.sin(math.radians(b_deg))
            solved_ab = scale * math.sin(math.radians(c_deg))
            solved_ca = scale * math.sin(math.radians(b_deg))
            reason = "solved by ASA/AAS (two angles + one side)"
        else:
            solved_ab = _DEFAULT_BASE_LENGTH
            solved_ca = _DEFAULT_BASE_LENGTH * math.sin(math.radians(b_deg)) / math.sin(math.radians(c_deg))
            reason = ("solved by angle-only constraints (no side length given) — angles and "
                      "relative proportions are accurate; absolute size is an arbitrary reference "
                      "scale since the question gave no measurement to fix it")

        A = (0.0, 0.0)
        B = (solved_ab, 0.0)
        C = _place_from_angle_at_origin(solved_ab, a_deg, solved_ca)
        return {a_id: A, b_id: B, c_id: C}, reason

    return None, "under-constrained: fewer than 3 sides, or fewer than (2 sides + included angle), or fewer than 2 angles were given"


def _fit_to_canvas(coords_math_space: dict, ids: list, canvas: tuple, margin: float) -> dict:
    """Uniformly scales + translates math-space coordinates (y-up,
    arbitrary units) to fit inside `canvas` with `margin` on every
    side, and flips y for SVG's y-down convention. Same technique
    diagram_renderer._render_trigonometry already uses inline for its
    own to-scale right triangle — extracted here so every future
    solvable type reuses one proven implementation instead of each
    reinventing it."""
    width, height = canvas
    xs = [coords_math_space[i][0] for i in ids]
    ys = [coords_math_space[i][1] for i in ids]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    avail_w = max(width - 2 * margin, 1.0)
    avail_h = max(height - 2 * margin, 1.0)
    scale = min(avail_w / span_x, avail_h / span_y)

    # center the scaled figure in the canvas
    scaled_w = span_x * scale
    scaled_h = span_y * scale
    offset_x = (width - scaled_w) / 2
    offset_y = (height - scaled_h) / 2

    screen = {}
    for pid in ids:
        x, y = coords_math_space[pid]
        sx = offset_x + (x - min_x) * scale
        # flip y: math-space "up" (larger y) must render nearer the
        # TOP of the canvas (smaller screen y)
        sy = offset_y + (max_y - y) * scale
        screen[pid] = (sx, sy)
    return screen


def solve_triangle(ids: list, side_lengths: dict | None = None, angles_deg: dict | None = None,
                    right_angle_at=None, canvas: tuple = (260, 220), margin: float = 34.0
                    ) -> tuple[dict | None, str]:
    """Returns (coords, reason).

    coords is {id: (x, y)} in SCREEN space (matching diagram_renderer's
    existing W=260,H=220 canvas / point-coordinate convention exactly)
    when — and only when — the given side_lengths/angles_deg uniquely
    and validly determine a triangle. Otherwise coords is None and
    `reason` explains why (logged by the caller, never surfaced as an
    error to the person using the product — an under-constrained spec
    is an entirely normal, expected case: most proof-style questions
    never state numeric measurements at all, by design).

    `ids` must have at least 3 entries (only the first 3 are used —
    matching _render_triangle's own existing "first 3 = main vertices"
    convention for any 4th+ point, which stays on the schematic-cevian
    fallback path regardless of what this function returns).

    Never raises. Any unexpected internal error is caught and reported
    as (None, "..."), same fail-safe-to-schematic guarantee as every
    other failure mode here.
    """
    try:
        if not ids or len(ids) < 3:
            return None, "fewer than 3 declared points — nothing to solve"
        tri_ids = ids[:3]

        sides = _side_lengths_by_pair(tri_ids, side_lengths)
        angles = _angles_by_vertex(tri_ids, angles_deg)

        # PRODUCTION-AUDIT FIX (final pre-launch round) — root cause of
        # "an exact 90° angle is never actually drawn": right_angle_at
        # used to be consulted ONLY further down, as a post-hoc check
        # that an ALREADY-solved triangle's angle at that vertex came
        # out to ~90° — it was never itself treated as a constraint that
        # could help SOLVE the triangle. So a question stating nothing
        # but "angle A = 90°" (right_angle_at="A", no side_lengths, no
        # angles_deg — extremely common for proof-style questions,
        # which rarely give numeric measurements at all) hit the
        # "no side_lengths or angles_deg given" line below and fell all
        # the way back to the generic, non-right schematic triangle
        # shape — the ONE geometric fact the question actually stated
        # was silently dropped from the drawing.
        #
        # Merging it in here (unless angles_deg already gave that exact
        # vertex a value, in which case that explicit value wins and
        # this is a no-op) means right_angle_at now participates in
        # solving through the SAME already-tested SAS/ASA/AAS/angle-only
        # paths below, not a separate code path.
        if right_angle_at:
            rid = str(right_angle_at)
            if rid in tri_ids and rid not in angles:
                angles[rid] = 90.0

        # Narrow, explicitly-scoped default: right_angle_at was the
        # ONLY geometric fact given anywhere (no side_lengths, no
        # angles_deg at all — so nothing above could possibly
        # contradict this), meaning even after the merge just above
        # there's still only ONE known angle, which is one short of
        # what the angle-only ASA/AAS path below needs. Rather than
        # give up and draw a triangle that ISN'T right-angled at all,
        # assume the other two angles are equal (45°/45°) — the plain,
        # most-generic possible right triangle — purely for this
        # SCHEMATIC drawing's shape. This never contradicts anything
        # the question stated (nothing else was stated to contradict)
        # and is the same kind of harmless, clearly-not-to-scale
        # default already used elsewhere in this function (see the
        # "solved by angle-only constraints" branch below, which
        # already pins an arbitrary absolute size for the exact same
        # reason). The one fact the question DID state — where the
        # right angle is — is now actually drawn correctly, instead of
        # silently dropped.
        if not side_lengths and not angles_deg and right_angle_at and len(angles) == 1:
            for vid in tri_ids:
                if vid not in angles:
                    angles[vid] = 45.0

        if not sides and not angles:
            return None, "no side_lengths or angles_deg given — using schematic placement"

        coords_math, reason = _solve_coords_math_space(tri_ids, sides, angles)
        if coords_math is None:
            return None, reason

        # ---- post-solve validation: never trust our own output blindly ----
        a_id, b_id, c_id = tri_ids
        A, B, C = coords_math[a_id], coords_math[b_id], coords_math[c_id]
        if A == B or B == C or C == A:
            return None, "solver produced a degenerate triangle (two coincident vertices)"

        def _angle_between(p_vertex, p1, p2):
            v1 = (p1[0] - p_vertex[0], p1[1] - p_vertex[1])
            v2 = (p2[0] - p_vertex[0], p2[1] - p_vertex[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 == 0 or n2 == 0:
                return None
            cos_val = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            cos_val = max(-1.0, min(1.0, cos_val))
            return math.degrees(math.acos(cos_val))

        ang_a = _angle_between(A, B, C)
        ang_b = _angle_between(B, A, C)
        ang_c = _angle_between(C, A, B)
        if None in (ang_a, ang_b, ang_c):
            return None, "solver could not re-derive interior angles from its own output (degenerate)"
        if abs((ang_a + ang_b + ang_c) - 180) > _ANGLE_SUM_TOLERANCE_DEG:
            return None, (f"solver's own output failed the 180-degree sanity check "
                           f"(A={ang_a:.3f}, B={ang_b:.3f}, C={ang_c:.3f}) — rejecting rather "
                           f"than publishing an internally-inconsistent diagram")
        if min(ang_a, ang_b, ang_c) <= _MIN_ANGLE_DEG:
            return None, "solver produced a degenerate (near-zero-angle) triangle"

        if right_angle_at:
            solved_angle_at_rid = {a_id: ang_a, b_id: ang_b, c_id: ang_c}.get(str(right_angle_at))
            if solved_angle_at_rid is not None and abs(solved_angle_at_rid - 90) > _RIGHT_ANGLE_TOLERANCE_DEG:
                return None, (f"right_angle_at='{right_angle_at}' contradicts the solved geometry "
                               f"(computed {solved_angle_at_rid:.2f}° there, not ~90°) — the "
                               f"question's own constraints are inconsistent with each other")

        screen_coords = _fit_to_canvas(coords_math, tri_ids, canvas, margin)
        return screen_coords, reason
    except Exception as e:  # a solver bug must degrade to schematic, never crash a whole exercise
        return None, f"geometry solver raised an unexpected error ({e}) — falling back to schematic placement"
