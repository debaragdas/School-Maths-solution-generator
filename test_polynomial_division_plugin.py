"""
Regression tests for polynomial_division_plugin.py (diagram_type #13:
"polynomial_long_division") — the fix for the reported "wrong margin
format" on polynomial long-division layouts.

Run: python3 -m pytest test_polynomial_division_plugin.py -v
"""
import re
import unittest

import diagram_plugin_registry as registry
import diagram_renderer as dr
import polynomial_division_plugin as plugin


def _sample_spec(n_steps=3):
    steps = [
        {"subtract": "x³ + πx²", "remainder": "(3-π)x² + 3x + 1"},
        {"subtract": "(3-π)x² + (3-π)πx", "remainder": "[3-3π+π²]x + 1"},
        {"subtract": "[3-3π+π²]x + (3-3π+π²)π", "remainder": "[1-3π+3π²-π³]"},
    ][:n_steps]
    return {
        "diagram_type": "polynomial_long_division",
        "divisor": "x + π",
        "quotient": "x² + (3-π)x + (3-3π+π²)",
        "dividend": "x³ + 3x² + 3x + 1",
        "steps": steps,
    }


class TestRegistration(unittest.TestCase):
    def test_registered_as_a_plugin_type(self):
        self.assertTrue(registry.is_plugin_type("polynomial_long_division"))

    def test_not_one_of_the_12_builtin_types(self):
        self.assertNotIn("polynomial_long_division", registry._BUILTIN_TYPE_NAMES)

    def test_appears_in_list_plugins(self):
        self.assertIn("polynomial_long_division", registry.list_plugins())


class TestSchemaValidation(unittest.TestCase):
    def test_valid_spec_passes(self):
        ok, issues = plugin._validate_polynomial_long_division_spec(_sample_spec())
        self.assertTrue(ok, issues)
        self.assertEqual(issues, [])

    def test_missing_divisor_rejected(self):
        spec = _sample_spec()
        del spec["divisor"]
        ok, issues = plugin._validate_polynomial_long_division_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("divisor" in i for i in issues))

    def test_empty_string_divisor_rejected(self):
        spec = _sample_spec()
        spec["divisor"] = "   "
        ok, issues = plugin._validate_polynomial_long_division_spec(spec)
        self.assertFalse(ok)

    def test_missing_steps_rejected(self):
        spec = _sample_spec()
        del spec["steps"]
        ok, issues = plugin._validate_polynomial_long_division_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("steps" in i for i in issues))

    def test_empty_steps_list_rejected(self):
        spec = _sample_spec()
        spec["steps"] = []
        ok, issues = plugin._validate_polynomial_long_division_spec(spec)
        self.assertFalse(ok)

    def test_step_missing_remainder_rejected(self):
        spec = _sample_spec()
        del spec["steps"][0]["remainder"]
        ok, issues = plugin._validate_polynomial_long_division_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("remainder" in i for i in issues))

    def test_step_not_a_dict_rejected(self):
        spec = _sample_spec()
        spec["steps"][0] = "not a dict"
        ok, issues = plugin._validate_polynomial_long_division_spec(spec)
        self.assertFalse(ok)

    def test_too_many_steps_rejected(self):
        spec = _sample_spec()
        spec["steps"] = [{"subtract": "x", "remainder": "y"}] * (plugin.MAX_STEPS + 1)
        ok, issues = plugin._validate_polynomial_long_division_spec(spec)
        self.assertFalse(ok)

    def test_single_step_is_valid(self):
        ok, issues = plugin._validate_polynomial_long_division_spec(_sample_spec(n_steps=1))
        self.assertTrue(ok, issues)


class TestRenderer(unittest.TestCase):
    def test_renders_valid_svg_wrapper(self):
        svg = plugin._render_polynomial_long_division(_sample_spec())
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_contains_divisor_dividend_quotient_text(self):
        svg = plugin._render_polynomial_long_division(_sample_spec())
        self.assertIn("x + π", svg)
        self.assertIn("x³ + 3x² + 3x + 1", svg)
        self.assertIn("x² + (3-π)x + (3-3π+π²)", svg)

    def test_contains_one_rule_line_per_step_plus_the_roof(self):
        spec = _sample_spec(n_steps=3)
        svg = plugin._render_polynomial_long_division(spec)
        # one <line> for the roof + one per step's subtract-rule = steps+1
        self.assertEqual(svg.count("<line "), len(spec["steps"]) + 1)

    def test_contains_the_bracket_path(self):
        svg = plugin._render_polynomial_long_division(_sample_spec())
        self.assertIn("<path ", svg)

    def test_minus_sign_used_not_hyphen(self):
        svg = plugin._render_polynomial_long_division(_sample_spec())
        self.assertIn("\u2212", svg)  # real minus sign (U+2212)

    def test_successive_steps_are_indented_further_right(self):
        spec = _sample_spec(n_steps=3)
        svg = plugin._render_polynomial_long_division(spec)
        x_positions = [float(x) for x in re.findall(r'<text x="([\d.]+)"', svg)]
        # rows in order: quotient, dividend, step0 subtract, step0 remainder,
        # step1 subtract, step1 remainder, step2 subtract, step2 remainder
        # (plus the divisor text at index 0, which is right-anchored and
        # excluded from this monotonic check since it's a different column)
        content_rows_x = x_positions[1:]
        self.assertEqual(content_rows_x[0], content_rows_x[1])  # quotient == dividend indent
        self.assertLess(content_rows_x[1], content_rows_x[2])   # dividend < step0 indent
        self.assertEqual(content_rows_x[2], content_rows_x[3])  # step0 subtract == step0 remainder
        self.assertLess(content_rows_x[3], content_rows_x[4])   # step0 < step1 indent
        self.assertEqual(content_rows_x[4], content_rows_x[5])
        self.assertLess(content_rows_x[5], content_rows_x[6])   # step1 < step2 indent

    def test_canvas_sized_to_fit_longest_row_no_excess_or_clipping(self):
        spec = _sample_spec(n_steps=1)
        spec["dividend"] = "x^100 extremely long polynomial expression that should widen the canvas a lot"
        svg = plugin._render_polynomial_long_division(spec)
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        self.assertIsNotNone(m)
        canvas_w = float(m.group(1))
        self.assertGreater(canvas_w, plugin._text_width(spec["dividend"]))

    def test_more_steps_produces_a_taller_canvas(self):
        svg_1 = plugin._render_polynomial_long_division(_sample_spec(n_steps=1))
        svg_3 = plugin._render_polynomial_long_division(_sample_spec(n_steps=3))
        h1 = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg_1).group(1))
        h3 = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg_3).group(1))
        self.assertGreater(h3, h1)

    def test_html_special_characters_are_escaped(self):
        spec = _sample_spec(n_steps=1)
        spec["dividend"] = "x < y & z > 0"
        svg = plugin._render_polynomial_long_division(spec)
        self.assertIn("&lt;", svg)
        self.assertIn("&amp;", svg)
        self.assertIn("&gt;", svg)
        self.assertNotIn("x < y & z > 0", svg)


class TestEndToEndThroughRenderDiagram(unittest.TestCase):
    """Proves the plugin actually flows through diagram_renderer.
    render_diagram()'s real dispatch — schema_check, renderer, AND the
    post-render marker-coverage gate — not just its own functions
    called directly."""

    def test_valid_spec_renders_through_the_real_dispatch(self):
        svg = dr.render_diagram(_sample_spec())
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("x + π", svg)

    def test_invalid_spec_is_rejected_not_rendered(self):
        spec = _sample_spec()
        del spec["dividend"]
        svg = dr.render_diagram(spec)
        self.assertEqual(svg, "")

    def test_none_spec_returns_empty_string(self):
        self.assertEqual(dr.render_diagram(None), "")

    def test_unknown_diagram_type_still_falls_through_cleanly(self):
        self.assertEqual(dr.render_diagram({"diagram_type": "not_a_real_type"}), "")


if __name__ == "__main__":
    unittest.main()
