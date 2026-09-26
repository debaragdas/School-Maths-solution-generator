"""
test_prompt_plugin_drift_guard.py — PROMPT <-> PLUGIN/REGISTRY DRIFT GUARD.

GENUINE GAP THIS FILLS (confirmed absent by grepping the whole test
suite before writing this file — nothing anywhere cross-checks
prompts.py's documented diagram_type schema against what's actually
registered, in either direction).

Two failure modes this protects against, permanently, for every
diagram_type that will ever exist (not just the ones present today):

  1. A new plugin gets registered (diagram_plugin_registry.register_
     plugin) but nobody documents its schema in prompts.py. Gemini can
     never be told to use it, so it's dead code the AI will never
     produce a spec for.

  2. prompts.py documents a diagram_type that no longer exists in
     either the built-in renderer dict or the plugin registry (e.g. a
     type gets renamed or removed from the registry but the prompt
     text is forgotten). Gemini would be instructed to emit specs for
     a type that render_diagram silently can't render.

Both directions are checked by parsing prompts.py's actual text (not
by hand-maintaining a duplicate list here, which would itself drift)
and comparing the resulting set against diagram_plugin_registry's own
live state.

Run: python3 -m unittest test_prompt_plugin_drift_guard -v
"""
import re
import unittest

import diagram_plugin_registry as registry

# Importing diagram_renderer triggers every plugin module's import-time
# registration side effect (see diagram_renderer.py's own "# noqa: F401"
# import block) — this is the SAME mechanism the real pipeline relies
# on, so this test exercises the real registration path, not a mock of it.
import diagram_renderer  # noqa: F401


def _diagram_types_documented_in_prompts() -> set:
    """Parses prompts.py's own text for every diagram_type string it
    documents, in EITHER of the two forms actually used there:

      (a) most types, one schema block each:
              "diagram_type": "probability"
      (b) the 12 original built-in types, documented together as a
          single pipe-separated enum on one line:
              "diagram_type": "triangle" | "circle" | "angle" | ...
    """
    with open("prompts.py", "r", encoding="utf-8") as f:
        text = f.read()

    found = set(re.findall(r'"diagram_type":\s*"([a-zA-Z_]+)"', text))

    enum_match = re.search(r'"diagram_type":\s*((?:"[a-zA-Z_]+"\s*\|\s*)+"[a-zA-Z_]+")', text)
    if enum_match:
        found |= set(re.findall(r'"([a-zA-Z_]+)"', enum_match.group(1)))

    return found


def _diagram_types_actually_registered() -> set:
    return set(registry.list_plugins()) | set(registry._BUILTIN_TYPE_NAMES)


class TestPromptPluginDriftGuard(unittest.TestCase):
    def test_prompts_parser_finds_a_sane_nonempty_set(self):
        # a basic sanity floor: if this ever returns empty or tiny, the
        # PARSER broke (prompts.py's format changed), not the product —
        # fail loudly rather than have every other assertion below
        # vacuously pass against an empty set.
        documented = _diagram_types_documented_in_prompts()
        self.assertGreaterEqual(len(documented), 20, documented)

    def test_every_registered_type_is_documented_in_prompts(self):
        documented = _diagram_types_documented_in_prompts()
        registered = _diagram_types_actually_registered()
        missing = registered - documented
        self.assertEqual(missing, set(),
                          f"These diagram types are registered (renderable) but prompts.py "
                          f"never documents their schema, so Gemini can never be instructed "
                          f"to produce one: {sorted(missing)}")

    def test_every_documented_type_is_actually_registered(self):
        documented = _diagram_types_documented_in_prompts()
        registered = _diagram_types_actually_registered()
        orphaned = documented - registered
        self.assertEqual(orphaned, set(),
                          f"prompts.py documents a schema for these diagram_type(s), but "
                          f"nothing in diagram_renderer.py's built-ins or the plugin registry "
                          f"can actually render them: {sorted(orphaned)}")

    def test_circle_sector_specifically_present_both_sides(self):
        # explicit anchor for the type this specific engineering pass
        # added, so a future refactor that silently drops either side
        # fails immediately and specifically, not just as part of the
        # general set-difference assertions above.
        self.assertIn("circle_sector", _diagram_types_actually_registered())
        self.assertIn("circle_sector", _diagram_types_documented_in_prompts())


if __name__ == "__main__":
    unittest.main()
