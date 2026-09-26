"""
test_nan_inf_hardening.py — V37 VERIFICATION & HARDENING PASS.

GENUINE BUG THIS FILE GUARDS AGAINST (found during this pass's edge-case
stress testing, not hypothetical): every plugin's `_is_number()` helper
checked `isinstance(v, (int, float))` only. NaN passes that check, and
in Python every ordering comparison against NaN (`<=`, `>`, `>=`) is
silently `False` — so a `"radius": float('nan')` spec sailed straight
through schema validation (`not _is_number(radius) or radius <= 0`
evaluates to `False`, i.e. "valid") and, for the two newest plugins,
even through the shoelace-formula verification step (`abs(measured -
expected) > tolerance` is also `False` against NaN), producing a
diagram that had been marked "verified correct" while actually holding
NaN coordinates throughout.

Reproduced directly before fixing, for both a brand-new plugin
(composite_shaded_plugin, this pass) and a plugin that predates this
entire engineering effort (circle_line_plugin) — confirming this was a
systemic, pre-existing pattern across the codebase, not a one-off typo
in new code. Fixed by requiring `math.isfinite(v)` in every `_is_number`
in all 8 affected files.

This test intentionally builds one MINIMAL VALID spec per plugin
(copying each plugin's own real required fields) and pokes exactly one
required numeric field with NaN, confirming schema_check rejects it.
"""
import unittest

import diagram_plugin_registry as registry
import diagram_renderer as dr  # noqa: F401 — triggers all plugin registration


def _check_type(diagram_type, base_spec, poison_field_path):
    """poison_field_path: e.g. ('radius',) or ('dimensions', 'length')."""
    import copy
    plugin = registry.get_plugin(diagram_type)
    for bad in (float("nan"), float("inf"), float("-inf")):
        spec = copy.deepcopy(base_spec)
        target = spec
        for key in poison_field_path[:-1]:
            target = target[key]
        target[poison_field_path[-1]] = bad
        ok, issues = plugin.schema_check(spec)
        if ok:
            return False, f"{diagram_type} accepted {poison_field_path}={bad}"
    return True, None


class TestNanInfRejectedAcrossAllPlugins(unittest.TestCase):
    def test_circle_line_intersection(self):
        ok, msg = _check_type("circle_line_intersection",
                              {"radius": 5, "line": {"point": {"x": 0, "y": 0}, "angle_deg": 0}},
                              ("radius",))
        self.assertTrue(ok, msg)

    def test_circle_sector(self):
        ok, msg = _check_type("circle_sector", {"radius": 5, "angle_deg": 60}, ("radius",))
        self.assertTrue(ok, msg)

    def test_composite_shaded_region(self):
        ok, msg = _check_type("composite_shaded_region",
                              {"composite_type": "circle_in_square", "side": 10}, ("side",))
        self.assertTrue(ok, msg)

    def test_elevation_depression(self):
        ok, msg = _check_type("elevation_depression",
                              {"mode": "elevation",
                               "points": {"eye": "A", "target": "B", "foot": "C"},
                               "angle_deg": 30},
                              ("angle_deg",))
        self.assertTrue(ok, msg)

    def test_unit_circle(self):
        ok, msg = _check_type("unit_circle", {"angle_deg": 45}, ("angle_deg",))
        self.assertTrue(ok, msg)

    def test_solid_net_cuboid(self):
        ok, msg = _check_type("solid_net",
                              {"solid": "cuboid", "dimensions": {"length": 4, "width": 3, "height": 2}},
                              ("dimensions", "length"))
        self.assertTrue(ok, msg)

    def test_cross_section_cylinder(self):
        ok, msg = _check_type("cross_section",
                              {"solid": "cylinder", "cut": "horizontal",
                               "dimensions": {"radius": 3, "height": 6}},
                              ("dimensions", "radius"))
        self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
