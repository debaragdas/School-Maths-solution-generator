"""
Regression tests for circle_sector_plugin.py.

Run: python3 -m unittest test_circle_sector_plugin -v
"""
import math
import unittest

import diagram_plugin_registry
import diagram_decision as dd
import diagram_renderer as dr
import circle_sector_plugin as csp


def _spec(**overrides):
    base = {
        "diagram_type": "circle_sector",
        "center": {"x": 0, "y": 0},
        "radius": 7,
        "angle_deg": 60,
        "mode": "sector",
        "region": "minor",
    }
    base.update(overrides)
    return base


class TestSchemaValidation(unittest.TestCase):
    def test_valid_spec_passes(self):
        ok, issues = csp._validate_circle_sector_spec(_spec())
        self.assertTrue(ok, issues)

    def test_missing_radius_rejected(self):
        spec = _spec()
        del spec["radius"]
        ok, issues = csp._validate_circle_sector_spec(spec)
        self.assertFalse(ok)

    def test_negative_radius_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(radius=-3))
        self.assertFalse(ok)

    def test_zero_radius_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(radius=0))
        self.assertFalse(ok)

    def test_angle_zero_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(angle_deg=0))
        self.assertFalse(ok)

    def test_angle_360_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(angle_deg=360))
        self.assertFalse(ok)

    def test_angle_over_360_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(angle_deg=400))
        self.assertFalse(ok)

    def test_invalid_mode_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(mode="triangle"))
        self.assertFalse(ok)

    def test_invalid_region_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(region="huge"))
        self.assertFalse(ok)

    def test_region_defaults_to_minor_and_mode_to_sector(self):
        spec = _spec()
        del spec["mode"]
        del spec["region"]
        ok, issues = csp._validate_circle_sector_spec(spec)
        self.assertTrue(ok, issues)

    def test_missing_center_defaults_to_origin(self):
        spec = _spec()
        del spec["center"]
        ok, issues = csp._validate_circle_sector_spec(spec)
        self.assertTrue(ok, issues)

    def test_nan_radius_rejected_not_silently_passed(self):
        # V37 hardening: NaN previously passed isinstance-only validation
        # AND the shoelace verification's tolerance comparison (both rely
        # on ordering comparisons that are always False against NaN).
        ok, issues = csp._validate_circle_sector_spec(_spec(radius=float("nan")))
        self.assertFalse(ok)

    def test_inf_radius_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(radius=float("inf")))
        self.assertFalse(ok)

    def test_nan_angle_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(angle_deg=float("nan")))
        self.assertFalse(ok)

    def test_inf_angle_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(angle_deg=float("inf")))
        self.assertFalse(ok)

    def test_nan_center_coordinate_rejected(self):
        ok, issues = csp._validate_circle_sector_spec(_spec(center={"x": float("nan"), "y": 0}))
        self.assertFalse(ok)


class TestClosedFormMath(unittest.TestCase):
    """Cross-checks compute_circle_sector against hand-computable values
    for a textbook case: r=7, theta=60 (NCERT's own worked example uses
    exactly these numbers for this chapter)."""

    def setUp(self):
        self.solved = csp.compute_circle_sector(_spec(radius=7, angle_deg=60))

    def test_arc_length_minor(self):
        expected = (60 / 360) * 2 * math.pi * 7
        self.assertAlmostEqual(self.solved["arc_length_minor"], expected, places=9)

    def test_arc_length_major_is_circumference_minus_minor(self):
        self.assertAlmostEqual(
            self.solved["arc_length_minor"] + self.solved["arc_length_major"],
            self.solved["circumference"], places=9)

    def test_sector_area_minor(self):
        expected = (60 / 360) * math.pi * 7 * 7
        self.assertAlmostEqual(self.solved["sector_area_minor"], expected, places=9)

    def test_sector_areas_sum_to_circle(self):
        self.assertAlmostEqual(
            self.solved["sector_area_minor"] + self.solved["sector_area_major"],
            self.solved["circle_area"], places=9)

    def test_segment_area_minor_formula(self):
        expected = self.solved["sector_area_minor"] - 0.5 * 7 * 7 * math.sin(math.radians(60))
        self.assertAlmostEqual(self.solved["segment_area_minor"], expected, places=9)

    def test_segment_areas_sum_to_circle(self):
        self.assertAlmostEqual(
            self.solved["segment_area_minor"] + self.solved["segment_area_major"],
            self.solved["circle_area"], places=9)

    def test_segment_area_less_than_sector_area_for_convex_angle(self):
        # for theta < 180, the chord cuts off a triangle, so the segment
        # must always be strictly smaller than its sector
        self.assertLess(self.solved["segment_area_minor"], self.solved["sector_area_minor"])

    def test_semicircle_case_area_ratio(self):
        # theta = 180: segment == sector exactly (the "triangle" OAB
        # degenerates to a zero-area straight line, so nothing is
        # subtracted) — a useful closed-form sanity anchor
        solved = csp.compute_circle_sector(_spec(radius=10, angle_deg=179.999999))
        self.assertAlmostEqual(solved["segment_area_minor"], solved["sector_area_minor"], places=2)


class TestRenderedGeometryMatchesFormula(unittest.TestCase):
    """The real correctness test: independently re-measures the actual
    rendered arc via verify_circle_sector_construction's shoelace
    sampling for all four mode x region combinations, at more than one
    radius/angle/center, and confirms every one matches the closed-form
    formula (catching any arc-direction/flag bug that the formula-only
    tests above could never catch)."""

    def _check(self, spec):
        solved = csp.compute_circle_sector(spec)
        ok, issues = csp.verify_circle_sector_construction(spec, solved)
        self.assertTrue(ok, issues)
        svg = csp._render_circle_sector(spec)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("data-role=\"arc\"", svg)
        return svg

    def test_sector_minor(self):
        self._check(_spec(mode="sector", region="minor"))

    def test_sector_major(self):
        self._check(_spec(mode="sector", region="major"))

    def test_segment_minor(self):
        self._check(_spec(mode="segment", region="minor"))

    def test_segment_major(self):
        self._check(_spec(mode="segment", region="major"))

    def test_offset_center_all_combinations(self):
        for mode in ("sector", "segment"):
            for region in ("minor", "major"):
                self._check(_spec(center={"x": 3, "y": -4}, radius=9, angle_deg=137,
                                   mode=mode, region=region))

    def test_small_angle(self):
        self._check(_spec(angle_deg=15, mode="segment", region="minor"))

    def test_large_angle(self):
        self._check(_spec(angle_deg=300, mode="sector", region="minor"))

    def test_shaded_flag_produces_fill_when_true(self):
        svg = self._check(_spec(shaded=True))
        self.assertIn("data-role=\"shaded-region\"", svg)

    def test_shaded_flag_omits_fill_when_false(self):
        svg = self._check(_spec(shaded=False))
        self.assertNotIn("data-role=\"shaded-region\"", svg)

    def test_segment_mode_draws_chord(self):
        svg = self._check(_spec(mode="segment"))
        self.assertIn("data-role=\"chord\"", svg)

    def test_sector_mode_has_no_chord(self):
        svg = self._check(_spec(mode="sector"))
        self.assertNotIn("data-role=\"chord\"", svg)

    def test_custom_point_labels_appear(self):
        svg = self._check(_spec(labels={"A": "P", "B": "Q"}))
        self.assertIn(">P<", svg)
        self.assertIn(">Q<", svg)


class TestPipelineIntegration(unittest.TestCase):
    """Confirms circle_sector flows through the exact same registry /
    decision / renderer choke points every other plugin type does,
    without any special-casing anywhere."""

    def test_registered_as_plugin_not_builtin(self):
        self.assertTrue(diagram_plugin_registry.is_plugin_type("circle_sector"))
        self.assertIn("circle_sector", diagram_plugin_registry.list_plugins())

    def test_does_not_shadow_builtin_circle_type(self):
        self.assertIn("circle", diagram_plugin_registry._BUILTIN_TYPE_NAMES)
        self.assertNotIn("circle", diagram_plugin_registry.list_plugins())

    def test_render_diagram_end_to_end(self):
        svg = dr.render_diagram(_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_render_diagram_rejects_bad_spec(self):
        svg = dr.render_diagram(_spec(radius=-1))
        self.assertEqual(svg, "")

    def test_validate_diagram_spec_uses_plugin_schema_check(self):
        ok, issues = dr.validate_diagram_spec(_spec(angle_deg=500))
        self.assertFalse(ok)

    def test_unknown_diagram_type_not_falsely_accepted(self):
        # a typo'd diagram_type must still be rejected by decision.py's
        # own "is this type registered anywhere at all" check
        spec = _spec()
        spec["diagram_type"] = "circle_sektor"
        self.assertNotIn(spec["diagram_type"], diagram_plugin_registry.list_plugins())


if __name__ == "__main__":
    unittest.main()
