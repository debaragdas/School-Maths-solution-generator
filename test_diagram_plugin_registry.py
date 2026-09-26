"""
Regression tests for diagram_plugin_registry.py (Phase 4 — Plugin-based
Diagram Engine). Proves two things: (1) the 12 built-in types are
completely unaffected by the registry existing, and (2) a brand new
type can be registered and flow correctly through decide_diagram ->
render_diagram -> validate_diagram_spec without touching any of the
three files that used to require hand-editing for a new type.

Run: python3 -m pytest test_diagram_plugin_registry.py -v
"""
import unittest

import diagram_plugin_registry as registry
import diagram_renderer as dr
import diagram_decision as dd


def _dummy_renderer(spec):
    return dr._svg(f'<circle cx="50" cy="50" r="10" data-role="dummy-plugin"/>')


def _dummy_schema_check(spec):
    if not spec.get("value"):
        return False, ["'value' is required for the test_dummy_type"]
    return True, []


class TestRegistryCore(unittest.TestCase):
    def tearDown(self):
        registry.unregister_plugin("test_dummy_type")
        registry.unregister_plugin("test_dummy_type_2")

    def test_register_and_retrieve(self):
        plugin = registry.DiagramPlugin(type_name="test_dummy_type", renderer=_dummy_renderer)
        registry.register_plugin(plugin)
        self.assertTrue(registry.is_plugin_type("test_dummy_type"))
        self.assertIs(registry.get_plugin("test_dummy_type"), plugin)
        self.assertIn("test_dummy_type", registry.list_plugins())

    def test_unknown_type_not_a_plugin(self):
        self.assertFalse(registry.is_plugin_type("nonexistent_type_xyz"))
        self.assertIsNone(registry.get_plugin("nonexistent_type_xyz"))

    def test_missing_type_name_raises(self):
        with self.assertRaises(ValueError):
            registry.register_plugin(registry.DiagramPlugin(type_name="", renderer=_dummy_renderer))

    def test_non_callable_renderer_raises(self):
        with self.assertRaises(ValueError):
            registry.register_plugin(registry.DiagramPlugin(type_name="test_dummy_type_2", renderer="not callable"))

    def test_cannot_shadow_a_builtin_type(self):
        with self.assertRaises(ValueError):
            registry.register_plugin(registry.DiagramPlugin(type_name="triangle", renderer=_dummy_renderer))

    def test_render_via_plugin_returns_none_for_non_plugin_type(self):
        self.assertIsNone(registry.render_via_plugin({"diagram_type": "triangle"}))

    def test_render_via_plugin_catches_exceptions(self):
        def bad_renderer(spec):
            raise RuntimeError("boom")
        registry.register_plugin(registry.DiagramPlugin(type_name="test_dummy_type", renderer=bad_renderer))
        result = registry.render_via_plugin({"diagram_type": "test_dummy_type"})
        self.assertEqual(result, "")  # fail-soft, never raises up to the caller

    def test_validate_via_plugin_uses_default_when_no_schema_check(self):
        registry.register_plugin(registry.DiagramPlugin(type_name="test_dummy_type", renderer=_dummy_renderer))
        ok, issues = registry.validate_via_plugin({"diagram_type": "test_dummy_type"})
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_validate_via_plugin_uses_provided_schema_check(self):
        registry.register_plugin(registry.DiagramPlugin(
            type_name="test_dummy_type", renderer=_dummy_renderer, schema_check=_dummy_schema_check))
        ok, issues = registry.validate_via_plugin({"diagram_type": "test_dummy_type"})
        self.assertFalse(ok)
        ok, issues = registry.validate_via_plugin({"diagram_type": "test_dummy_type", "value": 5})
        self.assertTrue(ok)


class TestBuiltinTypesUnaffected(unittest.TestCase):
    """The single most important guarantee for Phase 4: registering
    plugins must never change behavior for the 12 existing types."""

    def test_triangle_still_renders_via_builtin_dispatch(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        self.assertIn("<svg", svg)

    def test_circle_still_renders(self):
        spec = {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "A"}, {"id": "B"}]}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)


class TestEndToEndPluginFlow(unittest.TestCase):
    """Proves a NEW type flows through decide_diagram -> render_diagram
    -> validate_diagram_spec purely via registration, with zero edits
    to diagram_renderer.py/diagram_decision.py's hardcoded dispatch."""

    def setUp(self):
        registry.register_plugin(registry.DiagramPlugin(
            type_name="test_dummy_type", renderer=_dummy_renderer, schema_check=_dummy_schema_check,
            description="test-only plugin"))

    def tearDown(self):
        registry.unregister_plugin("test_dummy_type")

    def test_render_diagram_uses_the_plugin(self):
        svg = dr.render_diagram({"diagram_type": "test_dummy_type", "value": 5})
        self.assertIn("dummy-plugin", svg)

    def test_render_diagram_rejects_via_plugin_schema_check(self):
        svg = dr.render_diagram({"diagram_type": "test_dummy_type"})  # missing 'value'
        self.assertEqual(svg, "")

    def test_decide_diagram_recognizes_the_plugin_type_as_registered(self):
        question = {
            "question_number": "1",
            "diagram_spec": {"diagram_type": "test_dummy_type", "value": 5},
            "has_book_diagram": False,
        }
        verdict = dd.decide_diagram(question)
        self.assertEqual(verdict["decision"], dd.GENERATED_DIAGRAM)

    def test_decide_diagram_still_rejects_a_truly_unregistered_type(self):
        question = {
            "question_number": "1",
            "diagram_spec": {"diagram_type": "totally_made_up_type_xyz"},
            "has_book_diagram": False,
        }
        verdict = dd.decide_diagram(question)
        self.assertEqual(verdict["decision"], dd.NO_DIAGRAM)


class TestPromptsDocumentationStaysInSync(unittest.TestCase):
    """A plugin the renderer can handle but Gemini's own prompt never
    mentions is effectively dead code in production — the model will
    simply never emit that diagram_type. This guards against that drift:
    every *real* (non-test-fixture) plugin type registered at import time
    must have its own '"diagram_type": "<name>"' example block in
    prompts.py."""

    def test_every_real_plugin_type_is_documented_in_prompts(self):
        import elevation_bearing_plugin  # noqa: F401
        import unit_circle_plugin  # noqa: F401
        import circle_line_plugin  # noqa: F401
        import solid_geometry_plugin  # noqa: F401
        import function_graph_plugin  # noqa: F401
        import probability_plugin  # noqa: F401
        import set_theory_plugin  # noqa: F401
        import transformation_geometry_plugin  # noqa: F401
        import polynomial_division_plugin  # noqa: F401
        import prompts

        undocumented = []
        for type_name in registry.list_plugins():
            if type_name.startswith("test_dummy_type"):
                continue  # this file's own throwaway fixtures, not real product types
            needle = f'"diagram_type": "{type_name}"'
            if needle not in prompts.DIAGRAM_SPEC_SCHEMA:
                undocumented.append(type_name)
        self.assertEqual(undocumented, [],
                         f"registered plugin type(s) missing from prompts.py: {undocumented}")


if __name__ == "__main__":
    unittest.main()
