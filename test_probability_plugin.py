"""
Regression tests for probability_plugin.py (diagram_type #16:
"probability").

Run: python3 -m pytest test_probability_plugin.py -v
"""
import re
import unittest
from fractions import Fraction

import diagram_plugin_registry as registry
import diagram_renderer as dr
import probability_plugin as plugin


def _coin_toss_tree_spec(n_tosses=2):
    stage = {"branches": [{"label": "H", "probability": "1/2"}, {"label": "T", "probability": "1/2"}]}
    return {"diagram_type": "probability", "mode": "tree", "stages": [stage] * n_tosses}


def _dice_sample_space_spec(combine="sum"):
    return {
        "diagram_type": "probability", "mode": "sample_space",
        "rows": ["1", "2", "3", "4", "5", "6"], "cols": ["1", "2", "3", "4", "5", "6"],
        "combine": combine,
    }


class TestRegistration(unittest.TestCase):
    def test_registered_as_a_plugin_type(self):
        self.assertTrue(registry.is_plugin_type("probability"))

    def test_not_one_of_the_12_builtin_types(self):
        self.assertNotIn("probability", registry._BUILTIN_TYPE_NAMES)


class TestSchemaValidation(unittest.TestCase):
    def test_valid_tree_spec_passes(self):
        ok, issues = plugin._validate_probability_spec(_coin_toss_tree_spec())
        self.assertTrue(ok, issues)

    def test_valid_sample_space_spec_passes(self):
        ok, issues = plugin._validate_probability_spec(_dice_sample_space_spec())
        self.assertTrue(ok, issues)

    def test_missing_mode_rejected(self):
        spec = _coin_toss_tree_spec()
        del spec["mode"]
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)

    def test_tree_empty_stages_rejected(self):
        spec = _coin_toss_tree_spec()
        spec["stages"] = []
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)

    def test_tree_too_many_stages_rejected(self):
        spec = _coin_toss_tree_spec(n_tosses=plugin.MAX_STAGES + 1)
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)

    def test_tree_branch_missing_probability_rejected(self):
        spec = _coin_toss_tree_spec()
        del spec["stages"][0]["branches"][0]["probability"]
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)

    def test_tree_branch_non_fraction_probability_rejected(self):
        spec = _coin_toss_tree_spec()
        spec["stages"][0]["branches"][0]["probability"] = "half"
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)

    def test_sample_space_missing_rows_rejected(self):
        spec = _dice_sample_space_spec()
        del spec["rows"]
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)

    def test_sample_space_grid_too_large_rejected(self):
        spec = _dice_sample_space_spec()
        spec["rows"] = [str(i) for i in range(plugin.MAX_GRID_SIZE + 1)]
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)

    def test_sample_space_explicit_cells_accepted_without_combine(self):
        spec = {"diagram_type": "probability", "mode": "sample_space",
                "rows": ["H", "T"], "cols": ["H", "T"],
                "cells": [["HH", "HT"], ["TH", "TT"]]}
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertTrue(ok, issues)

    def test_sample_space_mismatched_cells_rejected(self):
        spec = {"diagram_type": "probability", "mode": "sample_space",
                "rows": ["H", "T"], "cols": ["H", "T"],
                "cells": [["HH", "HT"]]}
        ok, issues = plugin._validate_probability_spec(spec)
        self.assertFalse(ok)


class TestTreeMathematicalCorrectness(unittest.TestCase):
    def test_two_coin_tosses_have_four_leaves_each_probability_quarter(self):
        svg = plugin._render_tree(_coin_toss_tree_spec(2))
        for outcome in ("HH", "HT", "TH", "TT"):
            self.assertIn(f"{outcome}: 1/4", svg)

    def test_unequal_probabilities_multiply_correctly(self):
        spec = {
            "diagram_type": "probability", "mode": "tree",
            "stages": [
                {"branches": [{"label": "Red", "probability": "2/5"}, {"label": "Blue", "probability": "3/5"}]},
                {"branches": [{"label": "Red", "probability": "1/4"}, {"label": "Blue", "probability": "3/4"}]},
            ],
        }
        svg = plugin._render_tree(spec)
        # Red then Red = 2/5 * 1/4 = 1/10
        self.assertIn(f"RedRed: {Fraction(2, 5) * Fraction(1, 4)}", svg)

    def test_three_stage_tree_has_eight_leaves(self):
        svg = plugin._render_tree(_coin_toss_tree_spec(3))
        self.assertEqual(len(re.findall(r": 1/8", svg)), 8)


class TestSampleSpaceRenderer(unittest.TestCase):
    def test_renders_valid_svg_wrapper(self):
        svg = plugin._render_sample_space(_dice_sample_space_spec())
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_sum_combine_computes_correct_cell(self):
        svg = plugin._render_sample_space(_dice_sample_space_spec(combine="sum"))
        # row "6", col "6" -> sum 12 must appear
        self.assertIn(">12<", svg)

    def test_product_combine_computes_correct_cell(self):
        svg = plugin._render_sample_space(_dice_sample_space_spec(combine="product"))
        self.assertIn(">36<", svg)

    def test_explicit_cells_used_verbatim(self):
        spec = {"diagram_type": "probability", "mode": "sample_space",
                "rows": ["H", "T"], "cols": ["H", "T"],
                "cells": [["HH", "HT"], ["TH", "TT"]]}
        svg = plugin._render_sample_space(spec)
        for outcome in ("HH", "HT", "TH", "TT"):
            self.assertIn(f">{outcome}<", svg)

    def test_grid_cell_count_matches_rows_times_cols(self):
        svg = plugin._render_sample_space(_dice_sample_space_spec())
        # 6x6 = 36 inner white cells (+ headers, but only inner cells fill="white")
        self.assertEqual(svg.count('fill="white"'), 36)


class TestEndToEndThroughRenderDiagram(unittest.TestCase):
    def test_valid_tree_spec_renders_through_the_real_dispatch(self):
        svg = dr.render_diagram(_coin_toss_tree_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_valid_sample_space_spec_renders_through_the_real_dispatch(self):
        svg = dr.render_diagram(_dice_sample_space_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_invalid_spec_is_rejected_not_rendered(self):
        spec = _coin_toss_tree_spec()
        spec["mode"] = "bogus"
        self.assertEqual(dr.render_diagram(spec), "")
