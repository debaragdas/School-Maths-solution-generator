"""
Regression tests for composite_shaded_plugin.py.

Run: python3 -m unittest test_composite_shaded_plugin -v
"""
import math
import random
import unittest

import diagram_plugin_registry
import diagram_renderer as dr
import composite_shaded_plugin as csp


def _mc_area(inside_fn, xmin, xmax, ymin, ymax, trials=250000, seed=0):
    """Independent, formula-free Monte Carlo area estimate — used only
    to cross-check the closed-form formulas below, never used by the
    plugin itself (which is exact, not sampled)."""
    rng = random.Random(seed)
    box_area = (xmax - xmin) * (ymax - ymin)
    hits = sum(1 for _ in range(trials)
               if inside_fn(rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)))
    return hits / trials * box_area


class TestClosedFormAgainstMonteCarlo(unittest.TestCase):
    """These are the load-bearing correctness tests: each closed-form
    formula is checked against an INDEPENDENT numerical estimate that
    shares no code with the formula being tested. A 1.5% relative
    tolerance accommodates Monte Carlo sampling noise at 250k trials,
    not formula slack."""

    def test_circle_in_square_between_area(self):
        side, r = 10.0, 4.0
        solved = csp.compute_composite_shaded_region(
            {"composite_type": "circle_in_square", "side": side, "radius": r})
        mc = _mc_area(lambda x, y: (abs(x) <= side/2 and abs(y) <= side/2) and (x*x+y*y > r*r),
                      -side/2, side/2, -side/2, side/2)
        self.assertLess(abs(mc - solved["between_area"]) / solved["between_area"], 0.02)

    def test_square_in_circle_between_area(self):
        r = 8.0
        solved = csp.compute_composite_shaded_region({"composite_type": "square_in_circle", "radius": r})
        half = r / math.sqrt(2)
        mc = _mc_area(lambda x, y: (x*x+y*y <= r*r) and not (abs(x) <= half and abs(y) <= half),
                      -r, r, -r, r)
        self.assertLess(abs(mc - solved["between_area"]) / solved["between_area"], 0.02)

    def test_triangle_in_circle_between_area(self):
        r, angle = 6.0, 40.0
        solved = csp.compute_composite_shaded_region(
            {"composite_type": "triangle_in_circle", "radius": r, "angle_deg": angle})
        ar = math.radians(angle)
        B1, B2, C = (-r, 0.0), (r, 0.0), (r*math.cos(2*ar), r*math.sin(2*ar))

        def in_triangle(x, y):
            # barycentric sign test
            def sign(p1, p2, p3):
                return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
            p = (x, y)
            d1, d2, d3 = sign(p, B1, B2), sign(p, B2, C), sign(p, C, B1)
            has_neg, has_pos = (d1 < 0 or d2 < 0 or d3 < 0), (d1 > 0 or d2 > 0 or d3 > 0)
            return not (has_neg and has_pos)

        mc = _mc_area(lambda x, y: (x*x+y*y <= r*r) and not in_triangle(x, y), -r, r, -r, r)
        self.assertLess(abs(mc - solved["between_area"]) / solved["between_area"], 0.03)

    def test_two_overlapping_circles_lens_area(self):
        r, d = 5.0, 6.0
        solved = csp.compute_composite_shaded_region(
            {"composite_type": "two_overlapping_circles", "radius": r, "distance": d})
        a = d / 2.0
        mc = _mc_area(lambda x, y: (x+a)**2+y*y <= r*r and (x-a)**2+y*y <= r*r,
                      -a-r-1, a+r+1, -r-1, r+1)
        self.assertLess(abs(mc - solved["lens_area"]) / solved["lens_area"], 0.03)

    def test_petal_plus_lens_equals_circle(self):
        solved = csp.compute_composite_shaded_region(
            {"composite_type": "two_overlapping_circles", "radius": 5.0, "distance": 6.0})
        self.assertAlmostEqual(solved["petal_area"] + solved["lens_area"], solved["circle_area"], places=9)

    def test_union_formula_consistency(self):
        solved = csp.compute_composite_shaded_region(
            {"composite_type": "two_overlapping_circles", "radius": 5.0, "distance": 6.0})
        self.assertAlmostEqual(solved["union_area"], 2 * solved["circle_area"] - solved["lens_area"], places=9)


class TestSchemaValidation(unittest.TestCase):
    def test_circle_in_square_valid(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "circle_in_square", "side": 10})
        self.assertTrue(ok, issues)

    def test_circle_in_square_radius_too_big_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "circle_in_square", "side": 10, "radius": 6})
        self.assertFalse(ok)

    def test_square_in_circle_missing_radius_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec({"composite_type": "square_in_circle"})
        self.assertFalse(ok)

    def test_triangle_in_circle_angle_out_of_range_rejected(self):
        for bad_angle in (0, 90, -5, 120):
            ok, issues = csp._validate_composite_shaded_spec(
                {"composite_type": "triangle_in_circle", "radius": 5, "angle_deg": bad_angle})
            self.assertFalse(ok, f"angle_deg={bad_angle} should be rejected")

    def test_two_overlapping_circles_non_overlapping_distance_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "two_overlapping_circles", "radius": 5, "distance": 11})
        self.assertFalse(ok)

    def test_two_overlapping_circles_zero_distance_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "two_overlapping_circles", "radius": 5, "distance": 0})
        self.assertFalse(ok)

    def test_unknown_composite_type_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec({"composite_type": "nonsense", "radius": 5})
        self.assertFalse(ok)

    def test_invalid_shaded_value_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "circle_in_square", "side": 10, "shaded": "everything"})
        self.assertFalse(ok)

    def test_nan_side_rejected_not_silently_passed(self):
        # V37 hardening: this exact spec previously passed BOTH schema
        # validation AND verify_composite_shaded_construction (NaN
        # comparisons are always False in Python, so the tolerance check
        # `abs(nan - nan) > tolerance` silently evaluated to False and
        # let a NaN-valued diagram through as "verified correct").
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "circle_in_square", "side": float("nan")})
        self.assertFalse(ok)

    def test_inf_radius_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "two_overlapping_circles", "radius": float("inf"), "distance": 5})
        self.assertFalse(ok)

    def test_nan_distance_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "two_overlapping_circles", "radius": 5, "distance": float("nan")})
        self.assertFalse(ok)

    def test_nan_angle_deg_rejected(self):
        ok, issues = csp._validate_composite_shaded_spec(
            {"composite_type": "triangle_in_circle", "radius": 5, "angle_deg": float("nan")})
        self.assertFalse(ok)

    def test_render_diagram_end_to_end_rejects_nan_spec(self):
        # exercises the full pipeline choke point, not just the
        # standalone schema-check function
        svg = dr.render_diagram({"diagram_type": "composite_shaded_region",
                                  "composite_type": "circle_in_square", "side": float("nan")})
        self.assertEqual(svg, "")


class TestRenderedGeometryMatchesFormula(unittest.TestCase):
    """Exercises verify_composite_shaded_construction's shoelace
    cross-check for every composite_type x shaded-mode combination."""

    def _check(self, spec):
        solved = csp.compute_composite_shaded_region(spec)
        ok, issues = csp.verify_composite_shaded_construction(spec, solved)
        self.assertTrue(ok, issues)
        svg = csp._render_composite_shaded(spec)
        self.assertTrue(svg.startswith("<svg"))
        return svg

    def test_circle_in_square_all_shaded_modes(self):
        for shaded in ("between", "circle", "square"):
            self._check({"composite_type": "circle_in_square", "side": 10, "shaded": shaded})

    def test_circle_in_square_explicit_smaller_radius(self):
        self._check({"composite_type": "circle_in_square", "side": 12, "radius": 4})

    def test_square_in_circle_all_shaded_modes(self):
        for shaded in ("between", "circle", "square"):
            self._check({"composite_type": "square_in_circle", "radius": 7, "shaded": shaded})

    def test_triangle_in_circle_all_shaded_modes(self):
        for shaded in ("between", "circle", "triangle"):
            self._check({"composite_type": "triangle_in_circle", "radius": 6, "angle_deg": 35, "shaded": shaded})

    def test_triangle_in_circle_various_angles(self):
        for angle in (10, 30, 45, 60, 85):
            self._check({"composite_type": "triangle_in_circle", "radius": 5, "angle_deg": angle})

    def test_two_overlapping_circles_all_shaded_modes(self):
        for shaded in ("intersection", "union", "petals"):
            self._check({"composite_type": "two_overlapping_circles", "radius": 5, "distance": 6, "shaded": shaded})

    def test_two_overlapping_circles_various_distances(self):
        for distance in (1, 4, 7, 9.5):
            self._check({"composite_type": "two_overlapping_circles", "radius": 5, "distance": distance,
                         "shaded": "petals"})

    def test_shaded_region_fill_present_in_svg(self):
        svg = self._check({"composite_type": "circle_in_square", "side": 10})
        self.assertIn('data-role="shaded-region"', svg)


class TestPipelineIntegration(unittest.TestCase):
    def test_registered_as_plugin_not_builtin(self):
        self.assertTrue(diagram_plugin_registry.is_plugin_type("composite_shaded_region"))
        self.assertIn("composite_shaded_region", diagram_plugin_registry.list_plugins())

    def test_render_diagram_end_to_end(self):
        svg = dr.render_diagram({"diagram_type": "composite_shaded_region",
                                  "composite_type": "circle_in_square", "side": 10})
        self.assertTrue(svg.startswith("<svg"))

    def test_render_diagram_rejects_bad_spec(self):
        svg = dr.render_diagram({"diagram_type": "composite_shaded_region",
                                  "composite_type": "circle_in_square", "side": 10, "radius": 100})
        self.assertEqual(svg, "")

    def test_does_not_shadow_builtin_types(self):
        self.assertNotIn("composite_shaded_region", diagram_plugin_registry._BUILTIN_TYPE_NAMES)


if __name__ == "__main__":
    unittest.main()
