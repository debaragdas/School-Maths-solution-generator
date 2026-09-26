"""
Regression tests for successive_magnification_plugin.py.

Run: python3 -m unittest test_successive_magnification_plugin -v
"""
import unittest
from decimal import Decimal

import diagram_plugin_registry
import diagram_renderer as dr
import successive_magnification_plugin as smp


class TestSchemaValidation(unittest.TestCase):
    def test_valid_three_decimal_value(self):
        ok, issues = smp._validate_successive_magnification_spec(
            {"diagram_type": "successive_magnification", "value": "3.765"})
        self.assertTrue(ok, issues)

    def test_valid_four_decimal_value(self):
        ok, issues = smp._validate_successive_magnification_spec({"value": "4.2626"})
        self.assertTrue(ok, issues)

    def test_json_number_rejected_not_string(self):
        # the whole point of requiring a string: a JSON number risks
        # floating-point digit corruption
        ok, issues = smp._validate_successive_magnification_spec({"value": 3.765})
        self.assertFalse(ok)

    def test_whole_number_rejected(self):
        ok, issues = smp._validate_successive_magnification_spec({"value": "5"})
        self.assertFalse(ok)

    def test_negative_value_rejected(self):
        ok, issues = smp._validate_successive_magnification_spec({"value": "-2.5"})
        self.assertFalse(ok)

    def test_non_numeric_string_rejected(self):
        ok, issues = smp._validate_successive_magnification_spec({"value": "abc"})
        self.assertFalse(ok)

    def test_too_many_decimal_digits_rejected(self):
        ok, issues = smp._validate_successive_magnification_spec({"value": "1.2345678"})
        self.assertFalse(ok)

    def test_single_decimal_digit_accepted(self):
        ok, issues = smp._validate_successive_magnification_spec({"value": "0.3"})
        self.assertTrue(ok, issues)

    def test_six_decimal_digits_accepted(self):
        ok, issues = smp._validate_successive_magnification_spec({"value": "9.999999"})
        self.assertTrue(ok, issues)


class TestExactDigitReconstruction(unittest.TestCase):
    """The load-bearing correctness property: every row's chosen digit,
    reconstructed back into a number, must exactly equal the input —
    using Decimal throughout so there's no floating-point risk of an
    off-by-one digit."""

    def _check(self, value_str):
        solved = smp.compute_successive_magnification({"value": value_str})
        ok, issues = smp.verify_successive_magnification_construction(solved)
        self.assertTrue(ok, issues)
        self.assertEqual(solved["reconstructed_value"], Decimal(value_str))
        self.assertEqual(solved["final_point_value"], Decimal(value_str))
        return solved

    def test_reported_example_3_765(self):
        solved = self._check("3.765")
        self.assertEqual(solved["d"], 3)
        self.assertEqual(len(solved["rows"]), 4)  # overview + 3 decimal rows
        self.assertEqual(solved["decimal_digits"], [7, 6, 5])

    def test_reported_example_4_2626(self):
        solved = self._check("4.2626")
        self.assertEqual(solved["d"], 4)
        self.assertEqual(len(solved["rows"]), 5)

    def test_single_decimal_digit(self):
        solved = self._check("0.3")
        self.assertEqual(len(solved["rows"]), 2)  # overview + 1 decimal row

    def test_max_six_decimal_digits(self):
        solved = self._check("9.999999")
        self.assertEqual(len(solved["rows"]), 7)

    def test_trailing_nines_carry_correctly(self):
        # a case where every digit is 9 — easy to get an off-by-one on
        # the row boundaries wrong here specifically
        self._check("2.999")

    def test_leading_zero_digits(self):
        self._check("4.001")

    def test_double_digit_integer_part(self):
        solved = self._check("10.5")
        self.assertEqual(solved["integer_part"], Decimal(10))

    def test_zero_integer_part(self):
        self._check("0.123456")

    def test_each_row_highlight_matches_next_row_range_exactly(self):
        # exercises verify_successive_magnification_construction's own
        # cross-row consistency check directly
        solved = smp.compute_successive_magnification({"value": "3.765"})
        ok, issues = smp.verify_successive_magnification_construction(solved)
        self.assertTrue(ok, issues)


class TestRendering(unittest.TestCase):
    def _check(self, value_str):
        spec = {"diagram_type": "successive_magnification", "value": value_str}
        svg = smp._render_successive_magnification(spec)
        self.assertTrue(svg.startswith("<svg"), svg)
        return svg

    def test_renders_for_all_decimal_lengths(self):
        for v in ("0.3", "3.765", "4.2626", "9.99999", "9.999999"):
            self._check(v)

    def test_final_point_label_shows_exact_value(self):
        svg = self._check("3.765")
        self.assertIn(">3.765<", svg)

    def test_row_count_matches_decimal_digits_plus_one(self):
        svg = self._check("4.2626")
        self.assertEqual(svg.count('data-role="row-title"'), 5)

    def test_final_row_has_point_not_highlighted_segment(self):
        svg = self._check("3.765")
        # 4 rows total (0..3): rows 0,1,2 highlighted segments, row 3 a point
        self.assertEqual(svg.count('data-role="highlighted-segment"'), 3)
        self.assertEqual(svg.count('data-role="final-point"'), 1)

    def test_passes_objective_clip_and_overlap_checks(self):
        import diagram_final_check as dfc
        for v in ("0.3", "3.765", "4.2626", "9.999999"):
            svg = self._check(v)
            ok_o, msg_o = dfc._check_no_overlapping_labels(svg)
            ok_c, msg_c = dfc._check_no_clipped_content(svg)
            self.assertTrue(ok_o, msg_o)
            self.assertTrue(ok_c, msg_c)

    def test_passes_real_production_gate(self):
        import diagram_final_check as dfc
        spec = {"diagram_type": "successive_magnification", "value": "4.2626"}
        svg = smp._render_successive_magnification(spec)
        q = {"diagram_decision": "GENERATED_DIAGRAM", "diagram_spec": spec, "question_number": 1}
        result = dfc.final_pre_pdf_check(q, svg)
        self.assertEqual(result["final_decision"], "GENERATED_DIAGRAM", result["final_reason"])

    def test_invalid_spec_rejected_by_render_diagram(self):
        svg = dr.render_diagram({"diagram_type": "successive_magnification", "value": 3.765})
        self.assertEqual(svg, "")


class TestPipelineIntegration(unittest.TestCase):
    def test_registered_as_plugin_not_builtin(self):
        self.assertTrue(diagram_plugin_registry.is_plugin_type("successive_magnification"))
        self.assertIn("successive_magnification", diagram_plugin_registry.list_plugins())

    def test_render_diagram_end_to_end(self):
        svg = dr.render_diagram({"diagram_type": "successive_magnification", "value": "3.765"})
        self.assertTrue(svg.startswith("<svg"))


if __name__ == "__main__":
    unittest.main()
