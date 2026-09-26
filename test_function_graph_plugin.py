"""
Regression tests for function_graph_plugin.py (diagram_type #15:
"function_graph").

Run: python3 -m pytest test_function_graph_plugin.py -v
"""
import math
import re
import unittest

import diagram_plugin_registry as registry
import diagram_renderer as dr
import function_graph_plugin as plugin


def _exp_spec(base=2, a=1, c=0):
    return {"diagram_type": "function_graph",
            "function": {"kind": "exponential", "base": base, "a": a, "c": c}}


def _log_spec(base=2):
    return {"diagram_type": "function_graph",
            "function": {"kind": "logarithmic", "base": base}}


def _abs_spec(a=1, h=0, k=0):
    return {"diagram_type": "function_graph",
            "function": {"kind": "absolute_value", "a": a, "h": h, "k": k}}


def _piecewise_spec():
    return {"diagram_type": "function_graph",
            "function": {"kind": "piecewise", "pieces": [
                {"expr": "linear", "a": 1, "b": 0, "domain": [-5, 0]},
                {"expr": "constant", "c": 2, "domain": [0, 5]},
            ]}}


class TestRegistration(unittest.TestCase):
    def test_registered_as_a_plugin_type(self):
        self.assertTrue(registry.is_plugin_type("function_graph"))

    def test_not_one_of_the_12_builtin_types(self):
        self.assertNotIn("function_graph", registry._BUILTIN_TYPE_NAMES)


class TestSchemaValidation(unittest.TestCase):
    def test_valid_exponential_passes(self):
        ok, issues = plugin._validate_function_graph_spec(_exp_spec())
        self.assertTrue(ok, issues)

    def test_valid_logarithmic_passes(self):
        ok, issues = plugin._validate_function_graph_spec(_log_spec())
        self.assertTrue(ok, issues)

    def test_valid_absolute_value_passes(self):
        ok, issues = plugin._validate_function_graph_spec(_abs_spec())
        self.assertTrue(ok, issues)

    def test_valid_piecewise_passes(self):
        ok, issues = plugin._validate_function_graph_spec(_piecewise_spec())
        self.assertTrue(ok, issues)

    def test_exponential_base_1_rejected(self):
        spec = _exp_spec(base=1)
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_exponential_negative_base_rejected(self):
        spec = _exp_spec(base=-2)
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_logarithmic_missing_base_rejected(self):
        spec = _log_spec()
        del spec["function"]["base"]
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_absolute_value_missing_a_rejected(self):
        spec = _abs_spec()
        del spec["function"]["a"]
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_piecewise_empty_pieces_rejected(self):
        spec = _piecewise_spec()
        spec["function"]["pieces"] = []
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_piecewise_too_many_pieces_rejected(self):
        spec = _piecewise_spec()
        spec["function"]["pieces"] = [
            {"expr": "constant", "c": i, "domain": [i, i + 1]} for i in range(plugin.MAX_PIECES + 1)
        ]
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_piecewise_bad_domain_rejected(self):
        spec = _piecewise_spec()
        spec["function"]["pieces"][0]["domain"] = [5, -5]
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_unknown_kind_rejected(self):
        spec = _exp_spec()
        spec["function"]["kind"] = "trigonometric"
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)

    def test_bad_x_range_rejected(self):
        spec = _exp_spec()
        spec["x_range"] = [5, -5]
        ok, issues = plugin._validate_function_graph_spec(spec)
        self.assertFalse(ok)


class TestMathematicalCorrectness(unittest.TestCase):
    def test_exponential_evaluates_correctly(self):
        self.assertAlmostEqual(plugin._eval_exponential(3, {"base": 2, "a": 1, "c": 0}), 8)
        self.assertAlmostEqual(plugin._eval_exponential(0, {"base": 5, "a": 2, "c": 1}), 3)

    def test_logarithmic_evaluates_correctly(self):
        self.assertAlmostEqual(plugin._eval_logarithmic(8, {"base": 2, "a": 1, "c": 0}), 3)

    def test_logarithmic_undefined_for_nonpositive_x(self):
        self.assertIsNone(plugin._eval_logarithmic(0, {"base": 2}))
        self.assertIsNone(plugin._eval_logarithmic(-3, {"base": 2}))

    def test_absolute_value_evaluates_correctly(self):
        self.assertEqual(plugin._eval_absolute_value(-4, {"a": 2, "h": 1, "k": 3}), 2 * abs(-4 - 1) + 3)


class TestRenderer(unittest.TestCase):
    def test_renders_valid_svg_wrapper(self):
        svg = plugin._render_function_graph(_exp_spec())
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_exponential_curve_is_monotonic_increasing(self):
        svg = plugin._render_function_graph(_exp_spec(base=2))
        pts = re.search(r'<polyline points="([^"]+)"', svg).group(1).split()
        ys = [float(p.split(",")[1]) for p in pts]
        # SVG y grows downward, so an increasing function's y-pixel values decrease
        self.assertTrue(all(ys[i] >= ys[i + 1] - 1e-6 for i in range(len(ys) - 1)))

    def test_logarithmic_breaks_polyline_at_domain_boundary(self):
        spec = _log_spec()
        spec["x_range"] = [-2, 8]
        svg = plugin._render_function_graph(spec)
        # x <= 0 is undefined, so only one polyline (for x > 0) should be drawn
        self.assertEqual(svg.count("<polyline"), 1)

    def test_piecewise_produces_one_polyline_per_piece(self):
        svg = plugin._render_function_graph(_piecewise_spec())
        self.assertEqual(svg.count("<polyline"), 2)

    def test_absolute_value_vertex_is_the_minimum(self):
        spec = _abs_spec(a=1, h=2, k=-1)
        svg = plugin._render_function_graph(spec)
        pts = re.search(r'<polyline points="([^"]+)"', svg).group(1).split()
        ys_px = [float(p.split(",")[1]) for p in pts]
        # V-shape opening upward: in SVG space (y grows downward) the vertex
        # is the point with the LARGEST y-pixel value.
        vertex_idx = ys_px.index(max(ys_px))
        self.assertNotEqual(vertex_idx, 0)
        self.assertNotEqual(vertex_idx, len(ys_px) - 1)


class TestEndToEndThroughRenderDiagram(unittest.TestCase):
    def test_valid_spec_renders_through_the_real_dispatch(self):
        svg = dr.render_diagram(_exp_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_invalid_spec_is_rejected_not_rendered(self):
        spec = _exp_spec()
        spec["function"]["kind"] = "bogus"
        self.assertEqual(dr.render_diagram(spec), "")
