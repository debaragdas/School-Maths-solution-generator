"""
Regression tests for solid_geometry_plugin.py — "solid_net" and
"cross_section" diagram types.

Run: python3 -m unittest test_solid_geometry_plugin -v
"""
import math
import unittest

import diagram_plugin_registry
import solid_geometry_plugin as sgp


# ---------------------------------------------------------------------
# SOLID_NET
# ---------------------------------------------------------------------
class TestNetSchemaValidation(unittest.TestCase):
    def test_valid_cube_spec(self):
        ok, _ = sgp._validate_solid_net_spec({"solid": "cube", "dimensions": {"side": 4}})
        self.assertTrue(ok)

    def test_invalid_solid_rejected(self):
        ok, _ = sgp._validate_solid_net_spec({"solid": "sphere", "dimensions": {}})
        self.assertFalse(ok)

    def test_cuboid_missing_dimension_rejected(self):
        ok, _ = sgp._validate_solid_net_spec({"solid": "cuboid", "dimensions": {"length": 5, "width": 3}})
        self.assertFalse(ok)

    def test_cylinder_valid(self):
        ok, _ = sgp._validate_solid_net_spec({"solid": "cylinder", "dimensions": {"radius": 3, "height": 7}})
        self.assertTrue(ok)

    def test_cone_valid(self):
        ok, _ = sgp._validate_solid_net_spec({"solid": "cone", "dimensions": {"radius": 3, "slant_height": 5}})
        self.assertTrue(ok)

    def test_negative_dimension_rejected(self):
        ok, _ = sgp._validate_solid_net_spec({"solid": "cube", "dimensions": {"side": -2}})
        self.assertFalse(ok)


class TestNetMath(unittest.TestCase):
    def test_cylinder_rectangle_width_is_exact_circumference(self):
        solved = sgp.solve_solid_net({"solid": "cylinder", "dimensions": {"radius": 4, "height": 10}})
        self.assertAlmostEqual(solved["rect_width"], 2 * math.pi * 4, places=9)

    def test_cone_sector_angle_gives_correct_arc_length(self):
        solved = sgp.solve_solid_net({"solid": "cone", "dimensions": {"radius": 3, "slant_height": 9}})
        arc_len = solved["slant_height"] * math.radians(solved["sector_angle_deg"])
        self.assertAlmostEqual(arc_len, 2 * math.pi * 3, places=9)

    def test_cube_net_has_six_equal_faces(self):
        solved = sgp.solve_solid_net({"solid": "cube", "dimensions": {"side": 5}})
        self.assertEqual(len(solved["faces"]), 6)
        self.assertTrue(all(f == (5.0, 5.0) for f in solved["faces"]))

    def test_verification_passes_for_valid_construction(self):
        solved = sgp.solve_solid_net({"solid": "cylinder", "dimensions": {"radius": 4, "height": 10}})
        ok, issues = sgp.verify_solid_net_construction(solved)
        self.assertTrue(ok)

    def test_verification_catches_corrupted_rect_width(self):
        solved = sgp.solve_solid_net({"solid": "cylinder", "dimensions": {"radius": 4, "height": 10}})
        solved["rect_width"] = 999.0
        ok, issues = sgp.verify_solid_net_construction(solved)
        self.assertFalse(ok)


class TestNetRendering(unittest.TestCase):
    def test_cuboid_net_renders(self):
        svg = sgp._render_solid_net({"solid": "cuboid", "dimensions": {"length": 6, "width": 4, "height": 3}})
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg.count("net-face-"), 6)

    def test_cylinder_net_renders(self):
        svg = sgp._render_solid_net({"solid": "cylinder", "dimensions": {"radius": 3, "height": 8}})
        self.assertIn("net-curved-surface", svg)
        self.assertIn("net-top-circle", svg)
        self.assertIn("net-bottom-circle", svg)

    def test_cone_net_renders(self):
        svg = sgp._render_solid_net({"solid": "cone", "dimensions": {"radius": 3, "slant_height": 9}})
        self.assertIn("net-sector", svg)
        self.assertIn("net-base-circle", svg)


# ---------------------------------------------------------------------
# CROSS_SECTION
# ---------------------------------------------------------------------
class TestSectionSchemaValidation(unittest.TestCase):
    def test_valid_cuboid_spec(self):
        ok, _ = sgp._validate_cross_section_spec(
            {"solid": "cuboid", "cut": "parallel_to_face", "dimensions": {"length": 6, "width": 4, "height": 3}})
        self.assertTrue(ok)

    def test_invalid_cut_for_solid_rejected(self):
        ok, _ = sgp._validate_cross_section_spec(
            {"solid": "cuboid", "cut": "through_axis", "dimensions": {"length": 6, "width": 4, "height": 3}})
        self.assertFalse(ok)

    def test_cone_parallel_to_base_requires_cut_height(self):
        ok, _ = sgp._validate_cross_section_spec(
            {"solid": "cone", "cut": "parallel_to_base", "dimensions": {"radius": 3, "height": 10}})
        self.assertFalse(ok)

    def test_cone_cut_height_must_be_less_than_full_height(self):
        ok, _ = sgp._validate_cross_section_spec(
            {"solid": "cone", "cut": "parallel_to_base", "dimensions": {"radius": 3, "height": 10},
             "cut_height_from_apex": 12})
        self.assertFalse(ok)


class TestSectionMath(unittest.TestCase):
    def test_cone_parallel_to_base_uses_similar_triangles(self):
        solved = sgp.solve_cross_section(
            {"solid": "cone", "cut": "parallel_to_base", "dimensions": {"radius": 6, "height": 12},
             "cut_height_from_apex": 4})
        # at 1/3 of the way down from the apex, section radius should be 1/3 of full radius
        self.assertAlmostEqual(solved["radius"], 2.0, places=9)

    def test_cylinder_through_axis_is_rectangle_2r_by_h(self):
        solved = sgp.solve_cross_section(
            {"solid": "cylinder", "cut": "through_axis", "dimensions": {"radius": 5, "height": 20}})
        self.assertEqual(solved["shape"], "rectangle")
        self.assertAlmostEqual(solved["width"], 10.0)
        self.assertAlmostEqual(solved["height"], 20.0)

    def test_cuboid_parallel_to_face_is_l_by_h(self):
        solved = sgp.solve_cross_section(
            {"solid": "cuboid", "cut": "parallel_to_face", "dimensions": {"length": 7, "width": 3, "height": 5}})
        self.assertEqual(solved["shape"], "rectangle")

    def test_verification_passes_valid_cone_section(self):
        solved = sgp.solve_cross_section(
            {"solid": "cone", "cut": "parallel_to_base", "dimensions": {"radius": 6, "height": 12},
             "cut_height_from_apex": 4})
        ok, issues = sgp.verify_cross_section_construction(solved)
        self.assertTrue(ok)

    def test_verification_catches_corrupted_radius(self):
        solved = sgp.solve_cross_section(
            {"solid": "cone", "cut": "parallel_to_base", "dimensions": {"radius": 6, "height": 12},
             "cut_height_from_apex": 4})
        solved["radius"] = 999.0
        ok, issues = sgp.verify_cross_section_construction(solved)
        self.assertFalse(ok)


class TestSectionRendering(unittest.TestCase):
    def test_cuboid_section_renders_with_hidden_edges(self):
        svg = sgp._render_cross_section(
            {"solid": "cuboid", "cut": "parallel_to_face", "dimensions": {"length": 6, "width": 4, "height": 3}})
        self.assertIn("hidden-edge", svg)
        self.assertIn("section-shape", svg)

    def test_cylinder_parallel_to_base_section_is_circle(self):
        svg = sgp._render_cross_section(
            {"solid": "cylinder", "cut": "parallel_to_base", "dimensions": {"radius": 4, "height": 10}})
        self.assertIn("section-shape", svg)

    def test_cone_through_apex_axis_section_is_triangle(self):
        svg = sgp._render_cross_section(
            {"solid": "cone", "cut": "through_apex_axis", "dimensions": {"radius": 4, "height": 10}})
        self.assertIn("<polygon", svg)


class TestPluginRegistration(unittest.TestCase):
    def test_both_registered(self):
        plugins = diagram_plugin_registry.list_plugins()
        self.assertIn("solid_net", plugins)
        self.assertIn("cross_section", plugins)

    def test_net_dispatch(self):
        svg = diagram_plugin_registry.render_via_plugin(
            {"diagram_type": "solid_net", "solid": "cube", "dimensions": {"side": 4}})
        self.assertTrue(svg.startswith("<svg"))

    def test_section_dispatch(self):
        svg = diagram_plugin_registry.render_via_plugin(
            {"diagram_type": "cross_section", "solid": "cylinder", "cut": "through_axis",
             "dimensions": {"radius": 4, "height": 10}})
        self.assertTrue(svg.startswith("<svg"))


class TestCuboidNetDoesNotClip(unittest.TestCase):
    """V38 visual-QA regression: the cuboid net's middle row is
    [side][front][side][back], total width 2*(l+w) in math units, but
    the scale calculation previously divided by (l + 2*w) — always
    underestimating true width by exactly l units and computing too
    generous a scale, clipping the net off the canvas whenever l was
    reasonably large relative to w. Found via diagram_final_check.
    _check_no_clipped_content on a representative (l=6, w=4, h=3)
    cuboid, confirmed via manual scale-formula derivation, and fixed
    by dividing by the correct 2*(l+w)."""

    def _no_clip(self, spec):
        import diagram_final_check as dfc
        svg = sgp._render_solid_net(spec)
        ok, msg = dfc._check_no_clipped_content(svg)
        self.assertTrue(ok, msg)

    def test_original_failing_case(self):
        self._no_clip({"solid": "cuboid", "dimensions": {"length": 6, "width": 4, "height": 3}})

    def test_long_thin_cuboid(self):
        self._no_clip({"solid": "cuboid", "dimensions": {"length": 15, "width": 2, "height": 4}})

    def test_wide_short_cuboid(self):
        self._no_clip({"solid": "cuboid", "dimensions": {"length": 3, "width": 12, "height": 3}})

    def test_cube_unaffected(self):
        self._no_clip({"solid": "cube", "dimensions": {"side": 5}})

    def test_various_proportions(self):
        for l, w, h in [(4, 4, 4), (10, 1, 1), (1, 10, 10), (8, 6, 2), (2, 2, 20)]:
            self._no_clip({"solid": "cuboid", "dimensions": {"length": l, "width": w, "height": h}})


if __name__ == "__main__":
    unittest.main()
