"""
Regression tests for vision_validation.py (Phase 5 — deterministic
Vision Validation Engine) and its wiring into
diagram_final_check.final_pre_pdf_check.

Run: python3 -m pytest test_vision_validation.py -v
"""
import unittest

import vision_validation as vv
import diagram_final_check as dfc
import diagram_renderer as dr


class TestNoInvalidNumbers(unittest.TestCase):
    def test_normal_svg_passes(self):
        svg = '<svg viewBox="0 0 260 220"><line x1="10" y1="20" x2="30" y2="40"/></svg>'
        ok, reason = vv.check_no_invalid_numbers(svg)
        self.assertTrue(ok, reason)

    def test_nan_coordinate_fails(self):
        svg = '<svg viewBox="0 0 260 220"><line x1="nan" y1="20" x2="30" y2="40"/></svg>'
        ok, reason = vv.check_no_invalid_numbers(svg)
        self.assertFalse(ok)

    def test_infinity_coordinate_fails(self):
        svg = '<svg viewBox="0 0 260 220"><circle cx="inf" cy="20" r="5"/></svg>'
        ok, reason = vv.check_no_invalid_numbers(svg)
        self.assertFalse(ok)

    def test_non_numeric_but_expected_attr_form_does_not_crash(self):
        # a malformed attribute value should be ignored, not raise
        svg = '<svg><rect width="abc" height="10"/></svg>'
        ok, reason = vv.check_no_invalid_numbers(svg)
        self.assertTrue(ok)


class TestValidViewbox(unittest.TestCase):
    def test_valid_viewbox_passes(self):
        ok, reason = vv.check_valid_viewbox('<svg viewBox="0 0 260 220">...</svg>')
        self.assertTrue(ok, reason)

    def test_missing_viewbox_passes_by_default(self):
        ok, reason = vv.check_valid_viewbox('<svg>...</svg>')
        self.assertTrue(ok)

    def test_zero_width_viewbox_fails(self):
        ok, reason = vv.check_valid_viewbox('<svg viewBox="0 0 0 220">...</svg>')
        self.assertFalse(ok)

    def test_malformed_viewbox_fails(self):
        ok, reason = vv.check_valid_viewbox('<svg viewBox="0 0 abc 220">...</svg>')
        self.assertFalse(ok)


class TestTriangleGeometryReproducible(unittest.TestCase):
    def test_spec_without_measurements_is_not_applicable(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        ok, reason = vv.check_triangle_geometry_reproducible(spec)
        self.assertTrue(ok)

    def test_spec_with_valid_measurements_is_reproducible(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 4, "BC": 5, "CA": 3}}
        ok, reason = vv.check_triangle_geometry_reproducible(spec)
        self.assertTrue(ok, reason)

    def test_non_triangle_type_is_not_applicable(self):
        spec = {"diagram_type": "circle", "side_lengths": {"AB": 4}}
        ok, reason = vv.check_triangle_geometry_reproducible(spec)
        self.assertTrue(ok)

    def test_under_constrained_measurements_still_reproducible_none(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 4}}
        ok, reason = vv.check_triangle_geometry_reproducible(spec)
        self.assertTrue(ok, reason)  # both runs consistently return None -> legitimately reproducible


class TestChartElementCount(unittest.TestCase):
    def test_bar_chart_matching_count_passes(self):
        spec = {"diagram_type": "statistics", "chart_type": "bar", "values": [1, 2, 3]}
        svg = '<rect data-role="bar"/><rect data-role="bar"/><rect data-role="bar"/>'
        ok, reason = vv.check_chart_element_count_matches_data(spec, svg)
        self.assertTrue(ok, reason)

    def test_bar_chart_mismatched_count_fails(self):
        spec = {"diagram_type": "statistics", "chart_type": "bar", "values": [1, 2, 3]}
        svg = '<rect data-role="bar"/><rect data-role="bar"/>'  # only 2 drawn, 3 expected
        ok, reason = vv.check_chart_element_count_matches_data(spec, svg)
        self.assertFalse(ok)

    def test_pie_chart_matching_count_passes(self):
        spec = {"diagram_type": "statistics", "chart_type": "pie", "values": [10, 20, 30]}
        svg = ''.join(f'<path data-role="pie-slice"/>' for _ in range(3))
        ok, reason = vv.check_chart_element_count_matches_data(spec, svg)
        self.assertTrue(ok, reason)

    def test_non_chart_type_is_not_applicable(self):
        spec = {"diagram_type": "triangle"}
        ok, reason = vv.check_chart_element_count_matches_data(spec, "<svg></svg>")
        self.assertTrue(ok)

    def test_line_chart_types_not_checked_by_this_rule(self):
        spec = {"diagram_type": "statistics", "chart_type": "ogive", "values": [1, 2, 3]}
        ok, reason = vv.check_chart_element_count_matches_data(spec, "<svg></svg>")
        self.assertTrue(ok)


class TestEndToEndRealRenders(unittest.TestCase):
    """Feeds ACTUAL diagram_renderer.py output through the full
    validation entry point — proves the checks work against real
    output, not just hand-crafted SVG snippets."""

    def test_real_rendered_triangle_passes_vision_validation(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 4, "BC": 5, "CA": 3}, "right_angle_at": "A"}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        ok, reason = vv.validate_generated_diagram(spec, svg)
        self.assertTrue(ok, reason)

    def test_real_rendered_bar_chart_passes_vision_validation(self):
        spec = {"diagram_type": "statistics", "chart_type": "bar",
                "categories": ["A", "B", "C"], "values": [4, 7, 2]}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        ok, reason = vv.validate_generated_diagram(spec, svg)
        self.assertTrue(ok, reason)


class TestWiringIntoFinalPreCheck(unittest.TestCase):
    def test_final_check_still_passes_a_clean_generated_diagram(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        svg = dr.render_diagram(spec)
        question = {"question_number": "1", "diagram_decision": dfc.GENERATED_DIAGRAM, "diagram_spec": spec}
        result = dfc.final_pre_pdf_check(question, svg)
        self.assertEqual(result["final_decision"], dfc.GENERATED_DIAGRAM)
        self.assertTrue(result["diagram_svg"])

    def test_final_check_rejects_a_diagram_with_invalid_numbers(self):
        question = {"question_number": "1", "diagram_decision": dfc.GENERATED_DIAGRAM,
                    "diagram_spec": {"diagram_type": "triangle"}}
        bad_svg = '<svg viewBox="0 0 260 220"><line x1="nan" y1="20" x2="30" y2="40"/></svg>'
        result = dfc.final_pre_pdf_check(question, bad_svg)
        self.assertEqual(result["final_decision"], dfc.NO_DIAGRAM)
        self.assertIn("vision validation", result["final_reason"])


if __name__ == "__main__":
    unittest.main()
