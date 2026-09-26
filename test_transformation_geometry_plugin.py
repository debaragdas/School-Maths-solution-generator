"""
Regression tests for transformation_geometry_plugin.py (diagram_type #14:
"transformation").

Run: python3 -m pytest test_transformation_geometry_plugin.py -v
"""
import math
import re
import unittest

import diagram_plugin_registry as registry
import diagram_renderer as dr
import transformation_geometry_plugin as plugin


def _triangle_shape():
    return [
        {"id": "A", "x": 1, "y": 1},
        {"id": "B", "x": 4, "y": 1},
        {"id": "C", "x": 1, "y": 3},
    ]


def _reflection_spec(axis="x-axis"):
    return {
        "diagram_type": "transformation",
        "shape": _triangle_shape(),
        "transformation": {"type": "reflection", "axis": axis},
    }


def _rotation_spec(angle=90, direction="counterclockwise", center=(0, 0)):
    return {
        "diagram_type": "transformation",
        "shape": _triangle_shape(),
        "transformation": {"type": "rotation", "center": list(center), "angle_deg": angle,
                            "direction": direction},
    }


def _translation_spec(vector=(3, -2)):
    return {
        "diagram_type": "transformation",
        "shape": _triangle_shape(),
        "transformation": {"type": "translation", "vector": list(vector)},
    }


class TestRegistration(unittest.TestCase):
    def test_registered_as_a_plugin_type(self):
        self.assertTrue(registry.is_plugin_type("transformation"))

    def test_not_one_of_the_12_builtin_types(self):
        self.assertNotIn("transformation", registry._BUILTIN_TYPE_NAMES)

    def test_appears_in_list_plugins(self):
        self.assertIn("transformation", registry.list_plugins())


class TestSchemaValidation(unittest.TestCase):
    def test_valid_reflection_spec_passes(self):
        ok, issues = plugin._validate_transformation_spec(_reflection_spec())
        self.assertTrue(ok, issues)

    def test_valid_rotation_spec_passes(self):
        ok, issues = plugin._validate_transformation_spec(_rotation_spec())
        self.assertTrue(ok, issues)

    def test_valid_translation_spec_passes(self):
        ok, issues = plugin._validate_transformation_spec(_translation_spec())
        self.assertTrue(ok, issues)

    def test_too_few_shape_points_rejected(self):
        spec = _translation_spec()
        spec["shape"] = [{"id": "A", "x": 0, "y": 0}]
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)

    def test_too_many_shape_points_rejected(self):
        spec = _translation_spec()
        spec["shape"] = [{"id": str(i), "x": i, "y": i} for i in range(plugin.MAX_VERTICES + 1)]
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)

    def test_shape_point_missing_coordinates_rejected(self):
        spec = _translation_spec()
        spec["shape"][0] = {"id": "A"}
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)

    def test_unknown_transformation_type_rejected(self):
        spec = _translation_spec()
        spec["transformation"] = {"type": "shear"}
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)

    def test_reflection_missing_axis_rejected(self):
        spec = _reflection_spec()
        del spec["transformation"]["axis"]
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)

    def test_reflection_over_arbitrary_line_requires_through(self):
        spec = _reflection_spec(axis="line")
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)
        spec["transformation"]["through"] = [[0, 0], [1, 1]]
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertTrue(ok, issues)

    def test_rotation_missing_direction_rejected(self):
        spec = _rotation_spec()
        del spec["transformation"]["direction"]
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)

    def test_rotation_missing_angle_rejected(self):
        spec = _rotation_spec()
        del spec["transformation"]["angle_deg"]
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)

    def test_translation_missing_vector_rejected(self):
        spec = _translation_spec()
        del spec["transformation"]["vector"]
        ok, issues = plugin._validate_transformation_spec(spec)
        self.assertFalse(ok)


class TestMathematicalCorrectness(unittest.TestCase):
    """The renderer computes every image point itself — these tests
    verify the actual arithmetic, independent of any SVG output."""

    def test_reflect_over_x_axis(self):
        self.assertEqual(plugin._reflect(3, 5, {"axis": "x-axis"}), (3, -5))

    def test_reflect_over_y_axis(self):
        self.assertEqual(plugin._reflect(3, 5, {"axis": "y-axis"}), (-3, 5))

    def test_reflect_over_y_equals_x(self):
        self.assertEqual(plugin._reflect(3, 5, {"axis": "y=x"}), (5, 3))

    def test_reflect_over_y_equals_neg_x(self):
        self.assertEqual(plugin._reflect(3, 5, {"axis": "y=-x"}), (-5, -3))

    def test_reflect_over_arbitrary_line_is_involution(self):
        # Reflecting twice over the same line must return the original point.
        transform = {"axis": "line", "through": [[0, 0], [2, 1]]}
        x, y = plugin._reflect(4, -3, transform)
        x2, y2 = plugin._reflect(x, y, transform)
        self.assertAlmostEqual(x2, 4, places=6)
        self.assertAlmostEqual(y2, -3, places=6)

    def test_rotate_90_ccw_about_origin(self):
        transform = {"center": [0, 0], "angle_deg": 90, "direction": "counterclockwise"}
        x, y = plugin._rotate(1, 0, transform)
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, 1, places=6)

    def test_rotate_90_cw_about_origin(self):
        transform = {"center": [0, 0], "angle_deg": 90, "direction": "clockwise"}
        x, y = plugin._rotate(1, 0, transform)
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, -1, places=6)

    def test_rotate_180_about_nonorigin_center(self):
        transform = {"center": [2, 2], "angle_deg": 180, "direction": "clockwise"}
        x, y = plugin._rotate(3, 2, transform)
        self.assertAlmostEqual(x, 1, places=6)
        self.assertAlmostEqual(y, 2, places=6)

    def test_translate(self):
        self.assertEqual(plugin._translate(1, 1, {"vector": [3, -2]}), (4, -1))


class TestRenderer(unittest.TestCase):
    def test_renders_valid_svg_wrapper(self):
        svg = plugin._render_transformation(_reflection_spec())
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_contains_original_and_image_point_labels(self):
        svg = plugin._render_transformation(_translation_spec())
        for pid in ("A", "B", "C"):
            self.assertIn(f">{pid}<", svg)
            self.assertIn(f">{pid}\u2032<", svg)

    def test_two_polygons_are_drawn(self):
        svg = plugin._render_transformation(_reflection_spec())
        self.assertEqual(svg.count("<polygon"), 2)

    def test_correspondence_lines_match_vertex_count(self):
        svg = plugin._render_transformation(_translation_spec())
        # 3 vertices -> 3 dotted correspondence lines
        self.assertEqual(len(re.findall(r'stroke-dasharray="2,3"', svg)), 3)

    def test_rotation_center_marker_present(self):
        svg = plugin._render_transformation(_rotation_spec())
        self.assertIn(">O<", svg)

    def test_translation_has_no_rotation_center_marker(self):
        svg = plugin._render_transformation(_translation_spec())
        self.assertNotIn(">O<", svg)

    def test_canvas_grows_to_contain_translated_image(self):
        near = plugin._render_transformation(_translation_spec(vector=(1, 1)))
        far = plugin._render_transformation(_translation_spec(vector=(20, 20)))
        near_w = float(re.search(r'viewBox="0 0 ([\d.]+)', near).group(1))
        far_w = float(re.search(r'viewBox="0 0 ([\d.]+)', far).group(1))
        self.assertGreater(far_w, near_w)


class TestEndToEndThroughRenderDiagram(unittest.TestCase):
    def test_valid_spec_renders_through_the_real_dispatch(self):
        svg = dr.render_diagram(_reflection_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_invalid_spec_is_rejected_not_rendered(self):
        spec = _reflection_spec()
        del spec["transformation"]["axis"]
        svg = dr.render_diagram(spec)
        self.assertEqual(svg, "")

    def test_none_spec_returns_empty_string(self):
        self.assertEqual(dr.render_diagram(None), "")
