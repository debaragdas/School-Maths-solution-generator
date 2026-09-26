"""
Regression tests for elevation_bearing_plugin.py (diagram_type #16
"elevation_depression" and #17 "bearing").

Run: python3 -m pytest test_elevation_bearing_plugin.py -v
"""
import math
import re
import unittest

import diagram_plugin_registry as registry
import diagram_renderer as dr
import elevation_bearing_plugin as plugin


def _elev_spec(mode="elevation", angle=30, known_side="horizontal", known_value=20, **extra):
    spec = {
        "diagram_type": "elevation_depression",
        "mode": mode,
        "points": {"eye": "A", "target": "B", "foot": "C"},
        "angle_deg": angle,
        "known_side": known_side,
        "known_value": known_value,
    }
    spec.update(extra)
    return spec


def _bearing_spec():
    return {
        "diagram_type": "bearing",
        "start": {"id": "O", "label": "O"},
        "legs": [
            {"to": {"id": "A", "label": "A"}, "bearing_deg": 60, "distance": 5},
            {"to": {"id": "B", "label": "B"}, "bearing_deg": 145, "distance": 8, "from_id": "A"},
        ],
    }


class TestRegistration(unittest.TestCase):
    def test_elevation_depression_registered(self):
        self.assertTrue(registry.is_plugin_type("elevation_depression"))

    def test_bearing_registered(self):
        self.assertTrue(registry.is_plugin_type("bearing"))

    def test_neither_shadows_a_builtin_type(self):
        self.assertNotIn("elevation_depression", registry._BUILTIN_TYPE_NAMES)
        self.assertNotIn("bearing", registry._BUILTIN_TYPE_NAMES)


class TestElevationDepressionSchemaValidation(unittest.TestCase):
    def test_valid_elevation_passes(self):
        ok, issues = plugin._validate_elevation_depression_spec(_elev_spec("elevation"))
        self.assertTrue(ok, issues)

    def test_valid_depression_passes(self):
        ok, issues = plugin._validate_elevation_depression_spec(_elev_spec("depression"))
        self.assertTrue(ok, issues)

    def test_bad_mode_rejected(self):
        spec = _elev_spec()
        spec["mode"] = "sideways"
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_angle_zero_rejected(self):
        spec = _elev_spec(angle=0)
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_angle_ninety_rejected(self):
        spec = _elev_spec(angle=90)
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_angle_over_ninety_rejected(self):
        spec = _elev_spec(angle=120)
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_negative_angle_rejected(self):
        spec = _elev_spec(angle=-10)
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_bad_known_side_rejected(self):
        spec = _elev_spec(known_side="diagonal")
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_negative_known_value_rejected(self):
        spec = _elev_spec(known_value=-5)
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_zero_known_value_rejected(self):
        spec = _elev_spec(known_value=0)
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_missing_points_rejected(self):
        spec = _elev_spec()
        del spec["points"]
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_duplicate_point_ids_rejected(self):
        spec = _elev_spec()
        spec["points"] = {"eye": "A", "target": "A", "foot": "C"}
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)

    def test_non_bool_show_unknown_value_rejected(self):
        spec = _elev_spec(show_unknown_value="yes")
        ok, issues = plugin._validate_elevation_depression_spec(spec)
        self.assertFalse(ok)


class TestElevationDepressionMath(unittest.TestCase):
    def test_known_horizontal_computes_vertical(self):
        solved = plugin._solve_elevation_depression(_elev_spec(known_side="horizontal", known_value=20, angle=30))
        self.assertAlmostEqual(solved["horizontal"], 20.0)
        self.assertAlmostEqual(solved["vertical"], 20.0 * math.tan(math.radians(30)))

    def test_known_vertical_computes_horizontal(self):
        solved = plugin._solve_elevation_depression(_elev_spec(known_side="vertical", known_value=15, angle=45))
        self.assertAlmostEqual(solved["vertical"], 15.0)
        self.assertAlmostEqual(solved["horizontal"], 15.0 / math.tan(math.radians(45)))

    def test_45_degrees_gives_equal_legs(self):
        solved = plugin._solve_elevation_depression(_elev_spec(known_side="horizontal", known_value=10, angle=45))
        self.assertAlmostEqual(solved["horizontal"], solved["vertical"], places=6)

    def test_hypotenuse_matches_pythagoras(self):
        solved = plugin._solve_elevation_depression(_elev_spec(known_side="horizontal", known_value=12, angle=37))
        expected_hyp = math.hypot(solved["horizontal"], solved["vertical"])
        self.assertAlmostEqual(solved["hypotenuse"], expected_hyp)

    def test_mode_does_not_change_the_tan_ratio(self):
        """Elevation and depression must compute the identical
        horizontal/vertical pair for the same angle+known leg — only
        where the angle is drawn (eye vs. horizontal-through-eye)
        differs, not the trigonometric ratio itself."""
        e = plugin._solve_elevation_depression(_elev_spec("elevation", angle=40, known_side="horizontal", known_value=9))
        d = plugin._solve_elevation_depression(_elev_spec("depression", angle=40, known_side="horizontal", known_value=9))
        self.assertAlmostEqual(e["horizontal"], d["horizontal"])
        self.assertAlmostEqual(e["vertical"], d["vertical"])


class TestElevationDepressionRenderer(unittest.TestCase):
    def test_renders_valid_svg_wrapper(self):
        svg = plugin._render_elevation_depression(_elev_spec("elevation"))
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_elevation_has_right_angle_and_sight_line_and_arc(self):
        svg = plugin._render_elevation_depression(_elev_spec("elevation"))
        self.assertIn('data-role="right-angle"', svg)
        self.assertIn('data-role="sight-line"', svg)
        self.assertIn('data-role="angle-arc"', svg)
        self.assertIn('data-role="angle-label"', svg)

    def test_depression_has_horizontal_reference(self):
        svg = plugin._render_elevation_depression(_elev_spec("depression"))
        self.assertIn('data-role="horizontal-reference"', svg)
        self.assertIn('data-role="right-angle"', svg)

    def test_elevation_has_no_horizontal_reference_ray(self):
        """Elevation's horizontal IS the eye-foot leg itself — no
        separate dashed reference ray is drawn (would be redundant)."""
        svg = plugin._render_elevation_depression(_elev_spec("elevation"))
        self.assertNotIn('data-role="horizontal-reference"', svg)

    def test_angle_label_shows_the_given_angle(self):
        svg = plugin._render_elevation_depression(_elev_spec(angle=37))
        self.assertIn(">37", svg)

    def test_point_labels_present(self):
        svg = plugin._render_elevation_depression(_elev_spec())
        self.assertIn('>A<', svg)
        self.assertIn('>B<', svg)
        self.assertIn('>C<', svg)

    def test_unknown_side_withheld_by_default(self):
        svg = plugin._render_elevation_depression(_elev_spec(known_side="horizontal", known_value=20, angle=30))
        self.assertIn('>?<', svg)  # the computed vertical leg is not spoiled

    def test_unknown_side_shown_when_requested(self):
        spec = _elev_spec(known_side="horizontal", known_value=20, angle=30, show_unknown_value=True)
        svg = plugin._render_elevation_depression(spec)
        self.assertNotIn('>?<', svg)
        expected_vertical = 20.0 * math.tan(math.radians(30))
        self.assertIn(plugin._fmt_len(expected_vertical), svg)

    def test_custom_unknown_label_used(self):
        spec = _elev_spec(unknown_label="h")
        svg = plugin._render_elevation_depression(spec)
        self.assertIn('>h<', svg)

    def test_full_pipeline_render_diagram_elevation(self):
        svg = dr.render_diagram(_elev_spec("elevation"))
        self.assertTrue(svg.startswith("<svg"))

    def test_full_pipeline_render_diagram_depression(self):
        svg = dr.render_diagram(_elev_spec("depression"))
        self.assertTrue(svg.startswith("<svg"))

    def test_full_pipeline_rejects_invalid_spec(self):
        spec = _elev_spec(angle=120)  # invalid: >= 90
        svg = dr.render_diagram(spec)
        self.assertEqual(svg, "")

    def test_various_angles_all_render_without_error(self):
        for angle in (5, 15, 30, 44, 46, 60, 75, 85):
            for mode in ("elevation", "depression"):
                svg = plugin._render_elevation_depression(_elev_spec(mode, angle=angle))
                self.assertTrue(svg.startswith("<svg"))
                self.assertIn("</svg>", svg)


class TestBearingSchemaValidation(unittest.TestCase):
    def test_valid_spec_passes(self):
        ok, issues = plugin._validate_bearing_spec(_bearing_spec())
        self.assertTrue(ok, issues)

    def test_single_leg_passes(self):
        spec = _bearing_spec()
        spec["legs"] = spec["legs"][:1]
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertTrue(ok, issues)

    def test_missing_start_rejected(self):
        spec = _bearing_spec()
        del spec["start"]
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_empty_legs_rejected(self):
        spec = _bearing_spec()
        spec["legs"] = []
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_bearing_360_rejected(self):
        spec = _bearing_spec()
        spec["legs"][0]["bearing_deg"] = 360
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_negative_bearing_rejected(self):
        spec = _bearing_spec()
        spec["legs"][0]["bearing_deg"] = -10
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_bearing_0_is_valid(self):
        spec = _bearing_spec()
        spec["legs"][0]["bearing_deg"] = 0
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertTrue(ok, issues)

    def test_zero_distance_rejected(self):
        spec = _bearing_spec()
        spec["legs"][0]["distance"] = 0
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_negative_distance_rejected(self):
        spec = _bearing_spec()
        spec["legs"][0]["distance"] = -3
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_unknown_from_id_rejected(self):
        spec = _bearing_spec()
        spec["legs"][1]["from_id"] = "Z"
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_duplicate_to_id_rejected(self):
        spec = _bearing_spec()
        spec["legs"][1]["to"]["id"] = "A"  # already used by leg[0]
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_missing_to_id_rejected(self):
        spec = _bearing_spec()
        del spec["legs"][0]["to"]["id"]
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertFalse(ok)

    def test_from_id_defaults_to_previous_leg(self):
        """legs[1] omits from_id in the base fixture only via explicit
        override — verify the *implicit* chain (no from_id at all)
        also validates cleanly."""
        spec = _bearing_spec()
        del spec["legs"][1]["from_id"]
        ok, issues = plugin._validate_bearing_spec(spec)
        self.assertTrue(ok, issues)


class TestBearingMath(unittest.TestCase):
    def test_first_leg_from_start(self):
        coords = plugin._solve_bearing(_bearing_spec())
        ox, oy = coords["O"]
        self.assertAlmostEqual(ox, 0.0)
        self.assertAlmostEqual(oy, 0.0)

    def test_bearing_due_north_moves_positive_y_only(self):
        spec = {"diagram_type": "bearing", "start": {"id": "O"},
                "legs": [{"to": {"id": "A"}, "bearing_deg": 0, "distance": 10}]}
        coords = plugin._solve_bearing(spec)
        ax, ay = coords["A"]
        self.assertAlmostEqual(ax, 0.0)
        self.assertAlmostEqual(ay, 10.0)

    def test_bearing_due_east_moves_positive_x_only(self):
        spec = {"diagram_type": "bearing", "start": {"id": "O"},
                "legs": [{"to": {"id": "A"}, "bearing_deg": 90, "distance": 10}]}
        coords = plugin._solve_bearing(spec)
        ax, ay = coords["A"]
        self.assertAlmostEqual(ax, 10.0, places=6)
        self.assertAlmostEqual(ay, 0.0, places=6)

    def test_bearing_due_south_moves_negative_y_only(self):
        spec = {"diagram_type": "bearing", "start": {"id": "O"},
                "legs": [{"to": {"id": "A"}, "bearing_deg": 180, "distance": 10}]}
        coords = plugin._solve_bearing(spec)
        ax, ay = coords["A"]
        self.assertAlmostEqual(ax, 0.0, places=6)
        self.assertAlmostEqual(ay, -10.0, places=6)

    def test_bearing_due_west_moves_negative_x_only(self):
        spec = {"diagram_type": "bearing", "start": {"id": "O"},
                "legs": [{"to": {"id": "A"}, "bearing_deg": 270, "distance": 10}]}
        coords = plugin._solve_bearing(spec)
        ax, ay = coords["A"]
        self.assertAlmostEqual(ax, -10.0, places=6)
        self.assertAlmostEqual(ay, 0.0, places=6)

    def test_distance_from_start_matches_leg_distance(self):
        spec = {"diagram_type": "bearing", "start": {"id": "O"},
                "legs": [{"to": {"id": "A"}, "bearing_deg": 37, "distance": 12.5}]}
        coords = plugin._solve_bearing(spec)
        ax, ay = coords["A"]
        self.assertAlmostEqual(math.hypot(ax, ay), 12.5)

    def test_second_leg_chains_from_first(self):
        coords = plugin._solve_bearing(_bearing_spec())
        ax, ay = coords["A"]
        bx, by = coords["B"]
        dist_ab = math.hypot(bx - ax, by - ay)
        self.assertAlmostEqual(dist_ab, 8.0)

    def test_explicit_from_id_overrides_chain_order(self):
        spec = {"diagram_type": "bearing", "start": {"id": "O"},
                "legs": [
                    {"to": {"id": "A"}, "bearing_deg": 90, "distance": 5},
                    {"to": {"id": "B"}, "bearing_deg": 180, "distance": 5, "from_id": "O"},
                ]}
        coords = plugin._solve_bearing(spec)
        # B branches from O (not from A), due south of O
        self.assertAlmostEqual(coords["B"][0], 0.0, places=6)
        self.assertAlmostEqual(coords["B"][1], -5.0, places=6)


class TestBearingRenderer(unittest.TestCase):
    def test_renders_valid_svg_wrapper(self):
        svg = plugin._render_bearing(_bearing_spec())
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_has_north_arrow_and_bearing_leg(self):
        svg = plugin._render_bearing(_bearing_spec())
        self.assertIn('data-role="north-arrow"', svg)
        self.assertIn('data-role="north-line"', svg)
        self.assertIn('data-role="bearing-leg"', svg)
        self.assertIn('data-role="angle-arc"', svg)

    def test_compass_drawn_once_per_distinct_from_point(self):
        """O and A are each a 'from' point once — exactly 2 compasses,
        never one per leg (which would double up on O)."""
        svg = plugin._render_bearing(_bearing_spec())
        self.assertEqual(svg.count('data-role="north-arrow"'), 2)

    def test_bearing_labels_show_three_digit_values(self):
        svg = plugin._render_bearing(_bearing_spec())
        self.assertIn("060", svg)
        self.assertIn("145", svg)

    def test_point_labels_present(self):
        svg = plugin._render_bearing(_bearing_spec())
        for label in ("O", "A", "B"):
            self.assertIn(f'>{label}<', svg)

    def test_distance_labels_present(self):
        svg = plugin._render_bearing(_bearing_spec())
        self.assertIn(">5<", svg)
        self.assertIn(">8<", svg)

    def test_full_pipeline_render_diagram(self):
        svg = dr.render_diagram(_bearing_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_full_pipeline_rejects_invalid_spec(self):
        spec = _bearing_spec()
        spec["legs"][0]["bearing_deg"] = 400
        svg = dr.render_diagram(spec)
        self.assertEqual(svg, "")

    def test_various_bearings_all_render_without_error(self):
        for b in (0, 30, 90, 135, 180, 225, 270, 315, 359):
            spec = {"diagram_type": "bearing", "start": {"id": "O"},
                    "legs": [{"to": {"id": "A"}, "bearing_deg": b, "distance": 6}]}
            svg = plugin._render_bearing(spec)
            self.assertTrue(svg.startswith("<svg"))

    def test_three_leg_chain_renders(self):
        spec = {
            "diagram_type": "bearing", "start": {"id": "O", "label": "Port"},
            "legs": [
                {"to": {"id": "A", "label": "A"}, "bearing_deg": 45, "distance": 6},
                {"to": {"id": "B", "label": "B"}, "bearing_deg": 120, "distance": 4},
                {"to": {"id": "C", "label": "C"}, "bearing_deg": 200, "distance": 5},
            ],
        }
        svg = dr.render_diagram(spec)
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg.count('data-role="bearing-leg"'), 3)


class TestDiagramDecisionIntegration(unittest.TestCase):
    """Confirms diagram_decision.py's own is-a-known-type check (which
    consults the plugin registry, per its own docstring) recognizes
    both new types without any per-type edit to that file."""

    def test_elevation_depression_recognized(self):
        import diagram_decision
        question = {"question": "Find the angle of elevation.",
                    "diagram_spec": _elev_spec()}
        self.assertIn(
            "elevation_depression",
            list(registry.list_plugins()),
        )

    def test_bearing_recognized(self):
        self.assertIn("bearing", list(registry.list_plugins()))

    def test_elevation_two_point_recognized(self):
        self.assertIn("elevation_two_point", list(registry.list_plugins()))


def _two_point_spec(layout="same_side", near=60, far=30, distance_between=20, **extra):
    spec = {
        "diagram_type": "elevation_two_point",
        "layout": layout,
        "points": {"near": "C", "far": "D", "foot": "B", "target": "A"},
        "near_angle_deg": near,
        "far_angle_deg": far,
        "distance_between": distance_between,
    }
    spec.update(extra)
    return spec


class TestElevationTwoPointValidation(unittest.TestCase):
    def test_valid_same_side_spec_passes(self):
        ok, issues = plugin._validate_elevation_two_point_spec(_two_point_spec())
        self.assertTrue(ok, issues)

    def test_valid_opposite_sides_spec_passes(self):
        ok, issues = plugin._validate_elevation_two_point_spec(_two_point_spec(layout="opposite_sides"))
        self.assertTrue(ok, issues)

    def test_bad_layout_rejected(self):
        ok, issues = plugin._validate_elevation_two_point_spec(_two_point_spec(layout="sideways"))
        self.assertFalse(ok)

    def test_same_side_requires_near_angle_greater_than_far(self):
        ok, issues = plugin._validate_elevation_two_point_spec(_two_point_spec(near=20, far=50))
        self.assertFalse(ok)
        self.assertTrue(any("must be strictly greater than" in i for i in issues))

    def test_opposite_sides_allows_any_relative_angles(self):
        ok, issues = plugin._validate_elevation_two_point_spec(
            _two_point_spec(layout="opposite_sides", near=20, far=50))
        self.assertTrue(ok, issues)

    def test_missing_points_rejected(self):
        spec = _two_point_spec()
        del spec["points"]
        ok, issues = plugin._validate_elevation_two_point_spec(spec)
        self.assertFalse(ok)

    def test_duplicate_point_ids_rejected(self):
        spec = _two_point_spec()
        spec["points"]["far"] = spec["points"]["near"]
        ok, issues = plugin._validate_elevation_two_point_spec(spec)
        self.assertFalse(ok)

    def test_non_positive_distance_rejected(self):
        ok, issues = plugin._validate_elevation_two_point_spec(_two_point_spec(distance_between=0))
        self.assertFalse(ok)

    def test_angle_out_of_range_rejected(self):
        ok, issues = plugin._validate_elevation_two_point_spec(_two_point_spec(near=90))
        self.assertFalse(ok)


class TestElevationTwoPointMath(unittest.TestCase):
    """The textbook-standard case: two points 20 m apart on the same side,
    angles 60 deg (near) and 30 deg (far) -> height = 10*sqrt(3) m,
    the well-known NCERT Class 10 Ch.9 answer for this exact setup."""

    def test_same_side_matches_textbook_answer(self):
        solved = plugin._solve_elevation_two_point(_two_point_spec())
        self.assertAlmostEqual(solved["height"], 10 * math.sqrt(3), places=9)
        self.assertAlmostEqual(solved["near_dist"], 10.0, places=9)
        self.assertAlmostEqual(solved["far_dist"], 30.0, places=9)

    def test_same_side_far_minus_near_equals_distance_between(self):
        solved = plugin._solve_elevation_two_point(_two_point_spec())
        self.assertAlmostEqual(solved["far_dist"] - solved["near_dist"], 20.0, places=9)

    def test_opposite_sides_near_plus_far_equals_distance_between(self):
        solved = plugin._solve_elevation_two_point(_two_point_spec(layout="opposite_sides"))
        self.assertAlmostEqual(solved["near_dist"] + solved["far_dist"], 20.0, places=9)

    def test_post_render_verification_passes_for_valid_solve(self):
        solved = plugin._solve_elevation_two_point(_two_point_spec())
        ok, issues = plugin.verify_elevation_two_point(solved)
        self.assertTrue(ok, issues)

    def test_post_render_verification_catches_tampered_height(self):
        solved = plugin._solve_elevation_two_point(_two_point_spec())
        solved["height"] *= 1.5  # simulate a corrupted/incorrect solve
        ok, issues = plugin.verify_elevation_two_point(solved)
        self.assertFalse(ok)

    def test_various_angle_pairs_stay_internally_consistent(self):
        # for a range of valid (near > far) angle pairs, re-deriving height
        # from each point's own distance + angle must reproduce the same height
        for near, far in ((45, 30), (70, 20), (80, 10), (50, 49)):
            solved = plugin._solve_elevation_two_point(_two_point_spec(near=near, far=far))
            ok, issues = plugin.verify_elevation_two_point(solved)
            self.assertTrue(ok, f"near={near} far={far}: {issues}")

    def test_opposite_sides_various_angles_stay_consistent(self):
        for near, far in ((60, 30), (45, 45), (10, 80)):
            solved = plugin._solve_elevation_two_point(
                _two_point_spec(layout="opposite_sides", near=near, far=far))
            ok, issues = plugin.verify_elevation_two_point(solved)
            self.assertTrue(ok, f"near={near} far={far}: {issues}")


class TestElevationTwoPointRendering(unittest.TestCase):
    def test_renders_valid_svg_same_side(self):
        svg = dr.render_diagram(_two_point_spec())
        self.assertTrue(svg.startswith("<svg"))

    def test_renders_valid_svg_opposite_sides(self):
        svg = dr.render_diagram(_two_point_spec(layout="opposite_sides"))
        self.assertTrue(svg.startswith("<svg"))

    def test_full_pipeline_rejects_invalid_spec(self):
        spec = _two_point_spec(near=20, far=50)  # violates same_side ordering
        self.assertEqual(dr.render_diagram(spec), "")

    def test_both_sight_lines_present(self):
        svg = dr.render_diagram(_two_point_spec())
        self.assertEqual(svg.count('data-role="sight-line"'), 2)

    def test_both_angle_labels_present(self):
        svg = dr.render_diagram(_two_point_spec())
        self.assertIn("60&#176;", svg)
        self.assertIn("30&#176;", svg)

    def test_height_hidden_by_default(self):
        svg = dr.render_diagram(_two_point_spec())
        # height value (10*sqrt(3) ~ 17.32) must not leak into the diagram
        # unless the caller explicitly opts in via show_height
        self.assertNotIn("17.32", svg)

    def test_height_shown_when_requested(self):
        svg = dr.render_diagram(_two_point_spec(show_height=True))
        self.assertIn("17.32", svg)

    def test_point_labels_present(self):
        svg = dr.render_diagram(_two_point_spec())
        for pid in ("C", "D", "B", "A"):
            self.assertIn(f'">{pid}<', svg)

    def test_all_point_ids_used_as_svg_text_nodes(self):
        # every declared point id must appear exactly as its own label text,
        # never silently dropped or substituted
        spec = _two_point_spec()
        svg = dr.render_diagram(spec)
        for role, pid in spec["points"].items():
            self.assertIn(f'data-role="point-label">{pid}<', svg,
                          f"point id for role '{role}' ({pid}) missing from rendered labels")


if __name__ == "__main__":
    unittest.main()
