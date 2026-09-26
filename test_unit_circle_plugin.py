"""
Regression tests for unit_circle_plugin.py.

Run: python3 -m unittest test_unit_circle_plugin -v
"""
import math
import unittest

import diagram_plugin_registry
import unit_circle_plugin as ucp


class TestSchemaValidation(unittest.TestCase):
    def test_valid_spec_passes(self):
        ok, issues = ucp._validate_unit_circle_spec({"angle_deg": 30})
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_missing_angle_rejected(self):
        ok, issues = ucp._validate_unit_circle_spec({})
        self.assertFalse(ok)

    def test_non_numeric_angle_rejected(self):
        ok, issues = ucp._validate_unit_circle_spec({"angle_deg": "thirty"})
        self.assertFalse(ok)

    def test_non_boolean_flags_rejected(self):
        ok, issues = ucp._validate_unit_circle_spec({"angle_deg": 30, "show_special_angles": "yes"})
        self.assertFalse(ok)


class TestSolveUnitCircle(unittest.TestCase):
    def test_90_degrees_is_pure_y(self):
        solved = ucp.solve_unit_circle({"angle_deg": 90})
        self.assertAlmostEqual(solved["cos"], 0.0, places=9)
        self.assertAlmostEqual(solved["sin"], 1.0, places=9)

    def test_45_degrees_symmetric(self):
        solved = ucp.solve_unit_circle({"angle_deg": 45})
        self.assertAlmostEqual(solved["cos"], solved["sin"], places=9)
        self.assertAlmostEqual(solved["cos"], math.sqrt(2) / 2, places=9)

    def test_quadrant_classification(self):
        self.assertEqual(ucp.solve_unit_circle({"angle_deg": 30})["quadrant"], "I")
        self.assertEqual(ucp.solve_unit_circle({"angle_deg": 120})["quadrant"], "II")
        self.assertEqual(ucp.solve_unit_circle({"angle_deg": 210})["quadrant"], "III")
        self.assertEqual(ucp.solve_unit_circle({"angle_deg": 300})["quadrant"], "IV")

    def test_negative_angle_computed_correctly(self):
        solved = ucp.solve_unit_circle({"angle_deg": -90})
        self.assertAlmostEqual(solved["sin"], -1.0, places=9)

    def test_angle_beyond_360_computed_correctly(self):
        solved = ucp.solve_unit_circle({"angle_deg": 390})
        ref = ucp.solve_unit_circle({"angle_deg": 30})
        self.assertAlmostEqual(solved["cos"], ref["cos"], places=9)
        self.assertAlmostEqual(solved["sin"], ref["sin"], places=9)


class TestVerification(unittest.TestCase):
    def test_valid_construction_passes(self):
        solved = ucp.solve_unit_circle({"angle_deg": 37})
        ok, issues = ucp.verify_unit_circle_construction(solved)
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_corrupted_point_fails_verification(self):
        solved = ucp.solve_unit_circle({"angle_deg": 37})
        solved["cos"] = 5.0  # deliberately break the identity
        ok, issues = ucp.verify_unit_circle_construction(solved)
        self.assertFalse(ok)
        self.assertTrue(issues)


class TestRendering(unittest.TestCase):
    def test_renders_valid_svg(self):
        svg = ucp._render_unit_circle({"angle_deg": 60})
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("unit-circle", svg)
        self.assertIn("terminal-point", svg)

    def test_special_angle_gets_exact_symbolic_label(self):
        svg = ucp._render_unit_circle({"angle_deg": 60})
        self.assertIn("(root3)/2", svg)

    def test_non_special_angle_gets_numeric_label(self):
        svg = ucp._render_unit_circle({"angle_deg": 37})
        solved = ucp.solve_unit_circle({"angle_deg": 37})
        self.assertIn(f"{solved['cos']:.3f}", svg)

    def test_show_special_angles_adds_ticks(self):
        svg = ucp._render_unit_circle({"angle_deg": 60, "show_special_angles": True})
        self.assertIn('data-role="tick"', svg)

    def test_quadrant_shading_flag(self):
        svg = ucp._render_unit_circle({"angle_deg": 60, "quadrant_shading": True})
        self.assertIn("quadrant-shade", svg)

    def test_no_overlapping_labels_in_output(self):
        import label_layout
        svg = ucp._render_unit_circle({"angle_deg": 33})
        elements = label_layout.parse_text_elements(svg)
        self.assertEqual(label_layout.find_overlaps(elements), [])


class TestPluginRegistration(unittest.TestCase):
    def test_registered_under_unit_circle(self):
        self.assertIn("unit_circle", diagram_plugin_registry.list_plugins())

    def test_render_via_plugin_dispatch(self):
        svg = diagram_plugin_registry.render_via_plugin({"diagram_type": "unit_circle", "angle_deg": 45})
        self.assertTrue(svg.startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
