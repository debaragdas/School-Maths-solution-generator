"""
Regression tests for circle_line_plugin.py.

Run: python3 -m unittest test_circle_line_plugin -v
"""
import math
import re
import unittest

import diagram_plugin_registry
import circle_line_plugin as clp


def _spec(**overrides):
    base = {
        "diagram_type": "circle_line_intersection",
        "center": {"x": 0, "y": 0},
        "radius": 5,
        "line": {"through": [{"x": -10, "y": 0}, {"x": 10, "y": 0}]},
    }
    base.update(overrides)
    return base


class TestSchemaValidation(unittest.TestCase):
    def test_valid_spec_passes(self):
        ok, issues = clp._validate_circle_line_spec(_spec())
        self.assertTrue(ok)

    def test_missing_radius_rejected(self):
        spec = _spec()
        del spec["radius"]
        ok, issues = clp._validate_circle_line_spec(spec)
        self.assertFalse(ok)

    def test_negative_radius_rejected(self):
        ok, issues = clp._validate_circle_line_spec(_spec(radius=-1))
        self.assertFalse(ok)

    def test_line_missing_both_forms_rejected(self):
        ok, issues = clp._validate_circle_line_spec(_spec(line={}))
        self.assertFalse(ok)

    def test_point_angle_form_accepted(self):
        ok, issues = clp._validate_circle_line_spec(
            _spec(line={"point": {"x": 0, "y": 0}, "angle_deg": 45}))
        self.assertTrue(ok)

    def test_duplicate_through_points_rejected(self):
        ok, issues = clp._validate_circle_line_spec(
            _spec(line={"through": [{"x": 1, "y": 1}, {"x": 1, "y": 1}]}))
        self.assertFalse(ok)


class TestSolveExactIntersection(unittest.TestCase):
    def test_diameter_line_gives_two_points_at_radius(self):
        solved = clp.solve_circle_line_intersection(_spec())
        self.assertEqual(solved["case"], "secant")
        self.assertEqual(len(solved["points"]), 2)
        xs = sorted(p[0] for p in solved["points"])
        self.assertAlmostEqual(xs[0], -5.0, places=9)
        self.assertAlmostEqual(xs[1], 5.0, places=9)

    def test_tangent_line_gives_exactly_one_point(self):
        # horizontal line y=5 is tangent to a circle of radius 5 at (0,5)
        spec = _spec(line={"through": [{"x": -10, "y": 5}, {"x": 10, "y": 5}]})
        solved = clp.solve_circle_line_intersection(spec)
        self.assertEqual(solved["case"], "tangent")
        self.assertEqual(len(solved["points"]), 1)
        self.assertAlmostEqual(solved["points"][0][0], 0.0, places=6)
        self.assertAlmostEqual(solved["points"][0][1], 5.0, places=6)

    def test_line_missing_circle_entirely(self):
        spec = _spec(line={"through": [{"x": -10, "y": 20}, {"x": 10, "y": 20}]})
        solved = clp.solve_circle_line_intersection(spec)
        self.assertEqual(solved["case"], "none")
        self.assertEqual(solved["points"], [])

    def test_off_center_circle(self):
        spec = _spec(center={"x": 3, "y": 4}, radius=5,
                     line={"through": [{"x": -20, "y": 4}, {"x": 20, "y": 4}]})
        solved = clp.solve_circle_line_intersection(spec)
        self.assertEqual(solved["case"], "secant")
        for (px, py) in solved["points"]:
            self.assertAlmostEqual(math.hypot(px - 3, py - 4), 5.0, places=6)

    def test_point_angle_form_solves_correctly(self):
        spec = _spec(line={"point": {"x": -10, "y": 0}, "angle_deg": 0})
        solved = clp.solve_circle_line_intersection(spec)
        self.assertEqual(solved["case"], "secant")


class TestVerification(unittest.TestCase):
    def test_valid_construction_passes(self):
        solved = clp.solve_circle_line_intersection(_spec())
        ok, issues = clp.verify_circle_line_construction(solved)
        self.assertTrue(ok)

    def test_tangent_perpendicularity_holds(self):
        spec = _spec(line={"through": [{"x": -10, "y": 5}, {"x": 10, "y": 5}]})
        solved = clp.solve_circle_line_intersection(spec)
        ok, issues = clp.verify_circle_line_construction(solved)
        self.assertTrue(ok)

    def test_corrupted_point_fails_verification(self):
        solved = clp.solve_circle_line_intersection(_spec())
        solved["points"][0] = (999.0, 999.0)
        ok, issues = clp.verify_circle_line_construction(solved)
        self.assertFalse(ok)


class TestRendering(unittest.TestCase):
    def test_secant_renders_two_intersection_points(self):
        svg = clp._render_circle_line(_spec())
        self.assertEqual(svg.count('data-role="intersection-point"'), 2)

    def test_tangent_renders_one_intersection_point(self):
        spec = _spec(line={"through": [{"x": -10, "y": 5}, {"x": 10, "y": 5}]})
        svg = clp._render_circle_line(spec)
        self.assertEqual(svg.count('data-role="intersection-point"'), 1)

    def test_no_intersection_renders_zero_points(self):
        spec = _spec(line={"through": [{"x": -10, "y": 20}, {"x": 10, "y": 20}]})
        svg = clp._render_circle_line(spec)
        self.assertEqual(svg.count('data-role="intersection-point"'), 0)
        self.assertIn("does not meet", svg)


class TestPluginRegistration(unittest.TestCase):
    def test_registered(self):
        self.assertIn("circle_line_intersection", diagram_plugin_registry.list_plugins())

    def test_dispatch(self):
        svg = diagram_plugin_registry.render_via_plugin(_spec())
        self.assertTrue(svg.startswith("<svg"))


class TestLineSpansCanvasWithoutClipping(unittest.TestCase):
    """V38 visual-QA regression, two related bugs found and fixed:

    (1) The line was originally extended +/-3*radius from input point A
        itself, not from the point on the line nearest the circle. When
        a question states points far from the circle (a realistic
        phrasing — "line through (-10, 2) and (10, 2)" for a small
        circle at the origin), this made the drawn segment overshoot
        wildly on one side and stop awkwardly mid-canvas on the other.

    (2) MORE SERIOUSLY: the drawing scale was computed from the
        circle's radius alone, with no regard for where the line
        actually is. A line whose closest approach to the circle is
        farther away than the circle's own radius (a real "no
        intersection" case with genuine separation, not a near-miss)
        could then NEVER enter the visible canvas at any extension
        length, because extending along the line's own direction can't
        fix a perpendicular offset the scale never accounted for.
        Confirmed this was a real, PRE-EXISTING production bug (not
        introduced this pass): diagram_final_check.final_pre_pdf_check
        treats any out-of-viewBox coordinate as fatal and unconditionally
        OMITS the whole diagram from the generated PDF/HTML — so this
        wasn't just a cosmetic overshoot, every circle_line_intersection
        diagram whose line has real separation from the circle was being
        silently dropped from production output entirely, before this
        fix. Fixed by (a) sizing the scale to fit both the circle and
        the line's closest approach to it, and (b) clipping the drawn
        line segment to the canvas rectangle (Liang-Barsky) so its
        emitted coordinates are always genuinely within the viewBox
        rather than relying on SVG's overflow:hidden to crop it."""

    def _line_endpoints(self, spec):
        svg = clp._render_circle_line(spec)
        m = re.search(r'<line x1="([\-\d.]+)" y1="([\-\d.]+)" x2="([\-\d.]+)" y2="([\-\d.]+)"', svg)
        self.assertIsNotNone(m, svg)
        return tuple(float(g) for g in m.groups())

    def _assert_within_viewbox_and_spans_canvas(self, spec):
        import diagram_final_check as dfc
        svg = clp._render_circle_line(spec)
        self.assertTrue(svg.startswith("<svg"))
        ok, msg = dfc._check_no_clipped_content(svg)
        self.assertTrue(ok, msg)
        x1, y1, x2, y2 = self._line_endpoints(spec)
        for v in (x1, x2):
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, clp.W)
        for v in (y1, y2):
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, clp.H)
        # spans a meaningful portion of the canvas (not shrunk to a
        # degenerate stub) — segment length relative to the canvas
        # diagonal, robust regardless of which pair of edges a
        # diagonal line happens to clip against (e.g. a 45-degree line
        # near a corner may exit via two adjacent edges, not opposite
        # ones, which is correct clipping behavior, not a defect)
        seg_len = math.hypot(x2 - x1, y2 - y1)
        canvas_diag = math.hypot(clp.W, clp.H)
        self.assertGreater(seg_len, canvas_diag * 0.3,
                           f"line segment ({seg_len:.1f}px) should span a meaningful portion of "
                           f"the canvas ({canvas_diag:.1f}px diagonal), not be a tiny stub")

    def test_far_away_input_points_still_pass_final_check(self):
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=5, line={"through": [{"x": -10, "y": 2}, {"x": 10, "y": 2}]}))

    def test_close_input_points_still_pass_final_check(self):
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=5, line={"through": [{"x": -10, "y": 0}, {"x": 10, "y": 0}]}))

    def test_tangent_case_passes_final_check(self):
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=5, line={"through": [{"x": -20, "y": 5}, {"x": 20, "y": 5}]}))

    def test_genuine_no_intersection_with_real_separation_now_visible(self):
        # the deeper bug: a line 10 units from a radius-3 circle
        # previously could never enter the visible canvas at all
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=3, line={"through": [{"x": -10, "y": 10}, {"x": 10, "y": 10}]}))

    def test_line_far_from_origin_via_point_and_angle(self):
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=3, line={"point": {"x": 50, "y": -50}, "angle_deg": 45}))

    def test_steep_near_vertical_line(self):
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=4, line={"point": {"x": 1, "y": -30}, "angle_deg": 89}))

    def test_vertical_line(self):
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=3, line={"through": [{"x": 2, "y": -20}, {"x": 2, "y": 20}]}))

    def test_offset_circle_center(self):
        self._assert_within_viewbox_and_spans_canvas(
            _spec(radius=4, center={"x": 5, "y": 5},
                  line={"through": [{"x": -5, "y": 5}, {"x": 15, "y": 5}]}))

    def test_full_production_gate_accepts_all_these_cases(self):
        # exercises the ACTUAL production gate (final_pre_pdf_check),
        # not just the standalone clip-check function, for the case
        # that most directly reproduces the found production bug
        import diagram_final_check as dfc
        spec = _spec(radius=3, line={"through": [{"x": -10, "y": 10}, {"x": 10, "y": 10}]})
        svg = clp._render_circle_line(spec)
        q = {"diagram_decision": "GENERATED_DIAGRAM", "diagram_spec": spec, "question_number": 1}
        result = dfc.final_pre_pdf_check(q, svg)
        self.assertEqual(result["final_decision"], "GENERATED_DIAGRAM", result["final_reason"])


if __name__ == "__main__":
    unittest.main()
