"""
Regression tests for set_theory_plugin.py (diagram_type #17:
"venn_diagram").

Run: python3 -m pytest test_set_theory_plugin.py -v
"""
import unittest

import diagram_plugin_registry as registry
import diagram_renderer as dr
import set_theory_plugin as plugin


def _two_set_spec(shaded=None, universal=False):
    spec = {
        "diagram_type": "venn_diagram",
        "sets": [
            {"id": "A", "label": "A", "elements": ["1", "2", "3", "4"]},
            {"id": "B", "label": "B", "elements": ["3", "4", "5", "6"]},
        ],
        "shaded_region": shaded,
    }
    if universal:
        spec["universal_set"] = {"label": "U", "elements": ["1", "2", "3", "4", "5", "6", "7"]}
    return spec


def _three_set_spec():
    return {
        "diagram_type": "venn_diagram",
        "sets": [
            {"id": "A", "label": "A", "elements": ["1", "2", "3"]},
            {"id": "B", "label": "B", "elements": ["3", "4", "5"]},
            {"id": "C", "label": "C", "elements": ["3", "6", "7"]},
        ],
    }


class TestRegistration(unittest.TestCase):
    def test_registered_as_a_plugin_type(self):
        self.assertTrue(registry.is_plugin_type("venn_diagram"))

    def test_not_one_of_the_12_builtin_types(self):
        self.assertNotIn("venn_diagram", registry._BUILTIN_TYPE_NAMES)


class TestSchemaValidation(unittest.TestCase):
    def test_valid_two_set_spec_passes(self):
        ok, issues = plugin._validate_venn_spec(_two_set_spec())
        self.assertTrue(ok, issues)

    def test_valid_three_set_spec_passes(self):
        ok, issues = plugin._validate_venn_spec(_three_set_spec())
        self.assertTrue(ok, issues)

    def test_single_set_rejected(self):
        spec = _two_set_spec()
        spec["sets"] = [spec["sets"][0]]
        ok, issues = plugin._validate_venn_spec(spec)
        self.assertFalse(ok)

    def test_four_sets_rejected(self):
        spec = _three_set_spec()
        spec["sets"].append({"id": "D", "label": "D", "elements": ["9"]})
        ok, issues = plugin._validate_venn_spec(spec)
        self.assertFalse(ok)

    def test_set_missing_label_rejected(self):
        spec = _two_set_spec()
        del spec["sets"][0]["label"]
        ok, issues = plugin._validate_venn_spec(spec)
        self.assertFalse(ok)

    def test_too_many_elements_rejected(self):
        spec = _two_set_spec()
        spec["sets"][0]["elements"] = [str(i) for i in range(plugin.MAX_ELEMENTS_PER_SET + 1)]
        ok, issues = plugin._validate_venn_spec(spec)
        self.assertFalse(ok)

    def test_invalid_shaded_region_rejected(self):
        spec = _two_set_spec(shaded="everything")
        ok, issues = plugin._validate_venn_spec(spec)
        self.assertFalse(ok)

    def test_valid_shaded_region_accepted(self):
        for shade in ("union", "intersection", "A_only", "B_only", "complement_A",
                      "complement_B", "symmetric_difference"):
            with self.subTest(shade=shade):
                ok, issues = plugin._validate_venn_spec(_two_set_spec(shaded=shade))
                self.assertTrue(ok, issues)

    def test_shaded_region_rejected_for_three_sets(self):
        spec = _three_set_spec()
        spec["shaded_region"] = "union"
        ok, issues = plugin._validate_venn_spec(spec)
        self.assertFalse(ok)


class TestRegionComputation(unittest.TestCase):
    def test_element_only_in_a(self):
        sets = _two_set_spec()["sets"]
        self.assertEqual(plugin._region_for_element("1", sets), frozenset({"A"}))

    def test_element_only_in_b(self):
        sets = _two_set_spec()["sets"]
        self.assertEqual(plugin._region_for_element("6", sets), frozenset({"B"}))

    def test_element_in_both(self):
        sets = _two_set_spec()["sets"]
        self.assertEqual(plugin._region_for_element("3", sets), frozenset({"A", "B"}))

    def test_element_in_none(self):
        sets = _two_set_spec()["sets"]
        self.assertEqual(plugin._region_for_element("99", sets), frozenset())

    def test_element_in_all_three(self):
        sets = _three_set_spec()["sets"]
        self.assertEqual(plugin._region_for_element("3", sets), frozenset({"A", "B", "C"}))


class TestRenderer(unittest.TestCase):
    def test_two_set_renders_valid_svg_wrapper(self):
        svg = plugin._render_venn(_two_set_spec())
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_three_set_renders_valid_svg_wrapper(self):
        svg = plugin._render_venn(_three_set_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_two_set_has_exactly_two_circles(self):
        svg = plugin._render_venn(_two_set_spec())
        self.assertEqual(svg.count("<circle"), 2)

    def test_three_set_has_exactly_three_circles(self):
        svg = plugin._render_venn(_three_set_spec())
        self.assertEqual(svg.count("<circle"), 3)

    def test_shared_element_appears_once_not_duplicated_per_set(self):
        svg = plugin._render_venn(_two_set_spec())
        # "3" and "4" are shared between A and B — each should appear exactly once
        self.assertEqual(svg.count(">3<"), 1)
        self.assertEqual(svg.count(">4<"), 1)

    def test_universal_set_rectangle_drawn_when_present(self):
        without = plugin._render_venn(_two_set_spec(universal=False))
        with_u = plugin._render_venn(_two_set_spec(universal=True))
        self.assertEqual(without.count("<rect"), 0)
        self.assertEqual(with_u.count("<rect"), 1)

    def test_shaded_region_adds_fill_opacity_markup(self):
        unshaded = plugin._render_venn(_two_set_spec(shaded=None))
        shaded = plugin._render_venn(_two_set_spec(shaded="intersection"))
        self.assertNotIn("fill-opacity", unshaded)
        self.assertIn("fill-opacity", shaded)

    def test_all_elements_from_both_sets_present(self):
        svg = plugin._render_venn(_two_set_spec())
        for e in ("1", "2", "3", "4", "5", "6"):
            self.assertIn(f">{e}<", svg)


class TestEndToEndThroughRenderDiagram(unittest.TestCase):
    def test_valid_spec_renders_through_the_real_dispatch(self):
        svg = dr.render_diagram(_two_set_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_invalid_spec_is_rejected_not_rendered(self):
        spec = _two_set_spec()
        spec["sets"] = [spec["sets"][0]]
        self.assertEqual(dr.render_diagram(spec), "")
