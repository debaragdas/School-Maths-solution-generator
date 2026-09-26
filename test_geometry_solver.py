"""
Regression tests for geometry_solver.py (Phase 1 of the diagram-engine
architecture upgrade — see diagram_engine_architecture_design.md).

Pure-stdlib module (only `math`), so these tests need no external deps.
Run: python3 -m pytest test_geometry_solver.py -v
"""
import math
import unittest

import geometry_solver as gs


def _dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _angle_at(vertex, p1, p2):
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    cos_val = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.degrees(math.acos(cos_val))


class TestBackwardCompatibility(unittest.TestCase):
    """The single most important guarantee: no side_lengths/angles_deg
    given -> None, so diagram_renderer's schematic fallback runs
    completely unchanged for every spec generated before this change."""

    def test_no_constraints_returns_none(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"])
        self.assertIsNone(coords)

    def test_empty_dicts_return_none(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={}, angles_deg={})
        self.assertIsNone(coords)

    def test_fewer_than_three_ids_returns_none(self):
        coords, reason = gs.solve_triangle(["A", "B"], side_lengths={"AB": 5})
        self.assertIsNone(coords)

    def test_unrelated_side_key_ignored_returns_none(self):
        # side references a point not in this triangle -> dropped, not used
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"XY": 5})
        self.assertIsNone(coords)

    def test_never_raises_on_garbage_input(self):
        coords, reason = gs.solve_triangle(
            ["A", "B", "C"],
            side_lengths={"AB": "not a number", "BC": None, "weird_key": 5},
            angles_deg={"A": "also not a number", "Z": 40},
        )
        self.assertIsNone(coords)
        self.assertIsInstance(reason, str)


class TestSSS(unittest.TestCase):
    def test_valid_3_4_5_right_triangle(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 4, "BC": 5, "CA": 3})
        self.assertIsNotNone(coords, reason)
        # side ratios must be preserved after fit-to-canvas scaling
        ab, bc, ca = _dist(coords["A"], coords["B"]), _dist(coords["B"], coords["C"]), _dist(coords["C"], coords["A"])
        scale = ab / 4
        self.assertAlmostEqual(bc / scale, 5, places=3)
        self.assertAlmostEqual(ca / scale, 3, places=3)
        # right angle must actually be at A (opposite the hypotenuse BC)
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 90, delta=0.5)

    def test_side_key_order_independent(self):
        c1, _ = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 4, "BC": 5, "CA": 3})
        c2, _ = gs.solve_triangle(["A", "B", "C"], side_lengths={"BA": 4, "CB": 5, "AC": 3})
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)

    def test_triangle_inequality_violation_returns_none(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 1, "BC": 1, "CA": 10})
        self.assertIsNone(coords)
        self.assertIn("triangle inequality", reason)

    def test_only_two_of_three_sides_falls_through_to_underconstrained(self):
        # SSS needs all 3; with only 2 sides and no angle, nothing solves
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 4, "BC": 5})
        self.assertIsNone(coords)


class TestSAS(unittest.TestCase):
    def test_two_sides_plus_included_angle_at_A(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 5, "CA": 5}, angles_deg={"A": 60})
        self.assertIsNotNone(coords, reason)
        ab = _dist(coords["A"], coords["B"])
        ca = _dist(coords["C"], coords["A"])
        self.assertAlmostEqual(ab, ca, places=3)  # isosceles as constructed
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 60, delta=0.5)

    def test_two_sides_plus_included_angle_at_B(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 6, "BC": 4}, angles_deg={"B": 90})
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["B"], coords["A"], coords["C"]), 90, delta=0.5)

    def test_two_sides_plus_included_angle_at_C(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"BC": 6, "CA": 4}, angles_deg={"C": 45})
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["C"], coords["A"], coords["B"]), 45, delta=0.5)


class TestASA_AAS(unittest.TestCase):
    def test_two_angles_plus_one_side(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], angles_deg={"A": 60, "B": 60}, side_lengths={"AB": 6})
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 60, delta=0.5)
        self.assertAlmostEqual(_angle_at(coords["B"], coords["A"], coords["C"]), 60, delta=0.5)
        self.assertAlmostEqual(_angle_at(coords["C"], coords["A"], coords["B"]), 60, delta=0.5)

    def test_angles_summing_over_180_rejected(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], angles_deg={"A": 120, "B": 100}, side_lengths={"AB": 5})
        self.assertIsNone(coords)

    def test_angles_only_no_side_still_solves_shape(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], angles_deg={"A": 50, "B": 60})
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 50, delta=0.5)
        self.assertAlmostEqual(_angle_at(coords["B"], coords["A"], coords["C"]), 60, delta=0.5)
        self.assertIn("angle-only", reason)


class TestRightAngleCrossCheck(unittest.TestCase):
    def test_right_angle_at_matches_solved_geometry(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 4, "BC": 5, "CA": 3},
                                            right_angle_at="A")
        self.assertIsNotNone(coords, reason)

    def test_right_angle_at_contradicts_solved_geometry_rejected(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 4, "BC": 5, "CA": 3},
                                            right_angle_at="B")
        self.assertIsNone(coords)
        self.assertIn("contradicts", reason)


class TestRightAngleAtAsSolvingConstraint(unittest.TestCase):
    """PRODUCTION-AUDIT FIX (final pre-launch round) — root cause of "an
    exact 90° angle is never actually drawn": right_angle_at used to be
    consulted ONLY as a post-hoc check on an ALREADY-solved triangle —
    never as a constraint that could help solve one in the first place.
    A question stating nothing but "angle A = 90°" (right_angle_at
    alone, no side_lengths, no angles_deg — very common for proof-style
    questions) used to fall all the way back to the generic, non-right
    schematic shape. It now solves to an actual 90° angle at the
    stated vertex."""

    def test_right_angle_at_alone_produces_an_actual_ninety_degree_angle(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], right_angle_at="A")
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 90, delta=0.5)

    def test_right_angle_at_alone_is_schematic_not_claimed_to_scale(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], right_angle_at="A")
        self.assertIn("arbitrary", reason)

    def test_right_angle_at_plus_one_other_angle_solves_exactly(self):
        coords, reason = gs.solve_triangle(["A", "B", "C"], angles_deg={"B": 30}, right_angle_at="A")
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 90, delta=0.5)
        self.assertAlmostEqual(_angle_at(coords["B"], coords["A"], coords["C"]), 30, delta=0.5)
        self.assertAlmostEqual(_angle_at(coords["C"], coords["A"], coords["B"]), 60, delta=0.5)

    def test_explicit_angles_deg_for_the_right_angle_vertex_takes_precedence_and_is_unaffected(self):
        # angles_deg already stating 90° at A directly must behave
        # identically whether or not right_angle_at also names A — the
        # merge is a no-op here, not a second, conflicting write.
        coords, reason = gs.solve_triangle(["A", "B", "C"], angles_deg={"A": 90, "B": 30}, right_angle_at="A")
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 90, delta=0.5)

    def test_right_angle_at_with_only_one_side_given_stays_under_constrained(self):
        # One leg + the right angle genuinely isn't enough to fix a
        # right triangle's other leg — this must stay None (schematic
        # fallback), never silently invent the missing leg length.
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 5}, right_angle_at="A")
        self.assertIsNone(coords)

    def test_right_angle_at_with_full_sss_still_uses_real_solve_not_the_fortyfive_default(self):
        # SSS (all three sides) must take priority over the lone-
        # right-angle 45/45 default — that default only ever applies
        # when NO side_lengths were given at all.
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 3, "BC": 5, "CA": 4},
                                            right_angle_at="A")
        self.assertIsNotNone(coords, reason)
        self.assertAlmostEqual(_angle_at(coords["A"], coords["B"], coords["C"]), 90, delta=0.5)
        self.assertNotIn("arbitrary reference scale", reason)

    def test_no_right_angle_at_and_no_measurements_still_returns_none(self):
        # Must not accidentally start solving every unconstrained
        # triangle now — only right_angle_at (or real measurements)
        # trigger solving.
        coords, reason = gs.solve_triangle(["A", "B", "C"])
        self.assertIsNone(coords)


class TestCanvasFit(unittest.TestCase):
    def test_solved_coords_stay_within_canvas_bounds(self):
        canvas = (260, 220)
        coords, reason = gs.solve_triangle(["A", "B", "C"], side_lengths={"AB": 40, "BC": 50, "CA": 30},
                                            canvas=canvas, margin=30)
        self.assertIsNotNone(coords, reason)
        for x, y in coords.values():
            self.assertGreaterEqual(x, -1e-6)
            self.assertLessEqual(x, canvas[0] + 1e-6)
            self.assertGreaterEqual(y, -1e-6)
            self.assertLessEqual(y, canvas[1] + 1e-6)


class TestRendererIntegration(unittest.TestCase):
    """End-to-end through diagram_renderer.render_diagram — proves the
    integration point, not just the solver in isolation."""

    def test_triangle_without_measurements_matches_legacy_schematic_layout(self):
        import diagram_renderer as dr
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]]}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        # legacy fixed apex position must still appear verbatim
        self.assertIn(f'x1="{dr.W / 2}"', svg)

    def test_triangle_with_measurements_renders_to_scale_and_passes_gates(self):
        import diagram_renderer as dr
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 4, "BC": 5, "CA": 3}, "right_angle_at": "A"}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)
        # the legacy fixed apex position must NOT appear — proves the
        # solved-coordinate branch actually ran, not the schematic one
        self.assertNotIn(f'x1="{dr.W / 2}" y1="25"', svg)

    def test_triangle_with_self_contradictory_measurements_falls_back_to_schematic(self):
        import diagram_renderer as dr
        # right_angle_at contradicts the given sides -> solver returns
        # None -> must fall back to the exact legacy schematic layout,
        # never crash, never omit the diagram entirely.
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 4, "BC": 5, "CA": 3}, "right_angle_at": "B"}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        self.assertIn(f'x1="{dr.W / 2}"', svg)  # legacy apex position present


if __name__ == "__main__":
    unittest.main()
