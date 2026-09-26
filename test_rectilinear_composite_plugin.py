"""Tests for rectilinear_composite_plugin — the Class 6-8 compound
rectilinear mensuration engine (L-shapes, T-shapes, step figures).

Core contract: the model supplies ONLY non-overlapping rectangle pieces
in question units; this plugin derives everything else EXACTLY (union
outline, area Σw×h, perimeter), re-measures what it drew with the
shoelace formula, and refuses to render anything it cannot prove
(holes, overlaps, malformed specs, open traces)."""
import math
import unittest

import rectilinear_composite_plugin as rcp


class TestValidation(unittest.TestCase):
    def test_valid_l_shape_passes(self):
        ok, issues = rcp.validate_rectilinear_composite({"pieces": [
            {"x": 0, "y": 0, "width": 8, "height": 2},
            {"x": 0, "y": 2, "width": 3, "height": 4}]})
        self.assertTrue(ok, issues)

    def test_overlapping_pieces_rejected(self):
        ok, issues = rcp.validate_rectilinear_composite({"pieces": [
            {"x": 0, "y": 0, "width": 4, "height": 4},
            {"x": 2, "y": 2, "width": 4, "height": 4}]})
        self.assertFalse(ok)
        self.assertTrue(any("overlap" in i.lower() for i in issues))

    def test_edge_touching_is_allowed_not_overlap(self):
        ok, issues = rcp.validate_rectilinear_composite({"pieces": [
            {"x": 0, "y": 0, "width": 3, "height": 6},
            {"x": 3, "y": 0, "width": 5, "height": 2}]})
        self.assertTrue(ok, issues)

    def test_nan_and_bad_fields_rejected(self):
        for bad in (
            {"pieces": [{"x": 0, "y": 0, "width": float("nan"), "height": 2}]},
            {"pieces": [{"x": 0, "y": 0, "width": -1, "height": 2}]},
            {"pieces": [{"x": 0, "y": 0, "width": 1}]},          # missing height
            {"pieces": "nope"},
            {"pieces": []},
        ):
            ok, _ = rcp.validate_rectilinear_composite(bad)
            self.assertFalse(ok, bad)

    def test_shaded_pieces_index_bounds(self):
        base = [{"x": 0, "y": 0, "width": 2, "height": 2}]
        ok, _ = rcp.validate_rectilinear_composite(
            {"pieces": base, "shaded_pieces": [0]})
        self.assertTrue(ok)
        ok, _ = rcp.validate_rectilinear_composite(
            {"pieces": base, "shaded_pieces": [5]})
        self.assertFalse(ok)


class TestExactGeometry(unittest.TestCase):
    def test_l_shape_area_28_perimeter_28(self):
        # L: base 8x2 + upright 3x4 -> area 16+12=28; outline
        # (0,0)->(8,0)->(8,2)->(3,2)->(3,6)->(0,6)->close = 8+2+5+4+3+6 = 28
        solved = rcp.compute_rectilinear_union({"pieces": [
            {"x": 0, "y": 0, "width": 8, "height": 2},
            {"x": 0, "y": 2, "width": 3, "height": 4}]})
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved["total_area"], 28.0, places=9)
        self.assertAlmostEqual(solved["perimeter"], 28.0, places=9)
        self.assertEqual(len(solved["outline"]), 6)  # one vertex per true corner

    def test_t_shape_area_and_outline_single_loop(self):
        solved = rcp.compute_rectilinear_union({"pieces": [
            {"x": 0, "y": 0, "width": 2, "height": 5},
            {"x": 2, "y": 3, "width": 4, "height": 2}]})
        self.assertIsNotNone(solved)
        # T: stem 2x5 = 10, crossbar 4x2 = 8 -> 18
        self.assertAlmostEqual(solved["total_area"], 18.0, places=9)
        pts = solved["outline"]
        perim = sum(math.dist(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts)))
        self.assertAlmostEqual(perim, 22.0, places=9)
        self.assertEqual(round(perim, 9), round(solved["perimeter"], 9))

    def test_hole_figure_rejected_not_guessed(self):
        # a proper NON-overlapping ring tiling (corners belong to one strip
        # only): structurally valid per the schema, but encloses a hole
        ring = [
            {"x": 0, "y": 0, "width": 7, "height": 1},
            {"x": 0, "y": 4, "width": 7, "height": 1},
            {"x": 0, "y": 1, "width": 1, "height": 3},
            {"x": 6, "y": 1, "width": 1, "height": 3},
        ]
        solved = rcp.compute_rectilinear_union({"pieces": ring})
        self.assertIsNone(solved)
        ok, issues = rcp.validate_rectilinear_composite({"pieces": ring})
        self.assertTrue(ok, issues)  # structurally valid tiles, but...

    def test_fully_enclosed_piece_still_one_outline_ok(self):
        # two stacked strips sharing a full edge: single outline, seam interior
        solved = rcp.compute_rectilinear_union({"pieces": [
            {"x": 0, "y": 0, "width": 5, "height": 2},
            {"x": 0, "y": 2, "width": 5, "height": 3}]})
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved["total_area"], 25.0, places=9)
        self.assertAlmostEqual(solved["perimeter"], 20.0, places=9)


class TestRendering(unittest.TestCase):
    def test_render_produces_svg_with_labels_for_valid_spec(self):
        svg = rcp._render_rectilinear_composite({"pieces": [
            {"x": 0, "y": 0, "width": 8, "height": 2},
            {"x": 0, "y": 2, "width": 3, "height": 4}],
            "unit": "cm"})
        self.assertIn("<svg", svg)
        self.assertIn("union-outline", svg)
        self.assertIn("28", svg)       # exact area caption
        self.assertIn("8 cm", svg)     # boundary dimension label

    def test_render_returns_empty_string_on_unprovable_spec(self):
        self.assertEqual(rcp._render_rectilinear_composite(
            {"pieces": [{"x": 0, "y": 0, "width": float("nan"), "height": 2}]}), "")

    def test_registered_in_plugin_registry(self):
        import diagram_plugin_registry as reg
        plugin = reg.get_plugin("rectilinear_composite")
        self.assertIsNotNone(plugin)

    def test_end_to_end_through_diagram_renderer(self):
        import diagram_renderer
        svg = diagram_renderer.render_diagram({
            "diagram_type": "rectilinear_composite",
            "pieces": [{"x": 0, "y": 0, "width": 6, "height": 2},
                       {"x": 0, "y": 2, "width": 2, "height": 3}],
            "unit": "cm",
        })
        self.assertIn("<svg", svg)


if __name__ == "__main__":
    unittest.main()
