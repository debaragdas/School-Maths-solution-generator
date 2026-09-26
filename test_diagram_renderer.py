"""
Regression tests for diagram_renderer.py.

diagram_renderer.py is pure-stdlib (only `math`), so these tests need no
external dependencies and no live Gemini/PyMuPDF/Playwright access — they
can run anywhere, including CI with zero extra installs.

Run: python3 -m unittest test_diagram_renderer -v
(pytest also works if it's installed: python3 -m pytest test_diagram_renderer.py -v)

Coverage, by section:
  - EXISTING renderer types (triangle/circle/angle/parallel_lines/
    coordinate_plot/number_line/square_root_spiral): smoke + fail-safe
    tests locking in current (unchanged) behavior as a regression guard
    for any future edit to diagram_renderer.py.
  - NEW renderer types added in this pass (quadrilateral/trigonometry/
    statistics/surface_area_volume): full renderer + structural
    validator + post-render coverage tests, both valid-spec (accept)
    and malformed-spec (reject) cases, per the brief's requirement that
    every new renderer ship with its own regression tests.
"""
import unittest
from unittest import mock

import diagram_renderer as dr
import label_layout


class TestExistingRenderersUnchanged(unittest.TestCase):
    """Locks in current behavior of the renderers that existed BEFORE
    this pass, so future changes to shared helpers (_tick_marks,
    _right_angle_box, etc.) can't silently break them."""

    def test_triangle_valid_spec_renders_and_passes_gates(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]], "right_angle_at": "B"}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_triangle_dangling_altitude_point_rejected(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "altitudes": [{"from": "A", "to_side": "BZ", "foot": "E"}]}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("BZ" in i for i in issues))

    def test_triangle_cevian_intersection_point_renders(self):
        """Regression test for a real bug found in a user-generated PDF:
        a proof about two angle bisectors meeting at O was rendered
        with the bisector feet incorrectly forced onto the base with
        cevians from the apex (the only fallback that existed before),
        and O never appeared in the diagram at all."""
        spec = {
            "diagram_type": "triangle",
            "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}, {"id": "E"}],
            "equal_marks": [["AB", "AC"]],
            "cevians": [
                {"from": "B", "to_side": "AC", "foot": "D"},
                {"from": "C", "to_side": "AB", "foot": "E"},
            ],
            "cevian_intersection_label": "O",
            "join_vertex_to_intersection": "A",
        }
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertIn('data-role="cevian-intersection"', svg)
        self.assertIn('data-role="intersection-join"', svg)
        self.assertIn(">O<", svg)
        # bisector feet must NOT get a spurious right-angle mark --
        # only altitudes (perpendicular cevians) do
        self.assertEqual(svg.count("<polyline"), 0)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_triangle_cevian_without_intersection_label_no_extra_marks(self):
        spec = {"diagram_type": "triangle",
                "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
                "cevians": [{"from": "B", "to_side": "AC", "foot": "D"}]}
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertTrue(ok)
        svg = dr.render_diagram(spec)
        self.assertNotIn('data-role="cevian-intersection"', svg)

    def test_cevian_intersection_label_requires_exactly_two_cevians(self):
        spec = {"diagram_type": "triangle",
                "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
                "cevians": [{"from": "B", "to_side": "AC", "foot": "D"}],
                "cevian_intersection_label": "O"}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("cevian_intersection_label" in i for i in issues))

    def test_join_vertex_to_intersection_must_be_declared(self):
        spec = {"diagram_type": "triangle",
                "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}, {"id": "E"}],
                "cevians": [{"from": "B", "to_side": "AC", "foot": "D"},
                            {"from": "C", "to_side": "AB", "foot": "E"}],
                "cevian_intersection_label": "O",
                "join_vertex_to_intersection": "Z"}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("join_vertex_to_intersection" in i for i in issues))

    def test_extra_points_without_altitude_or_cevian_still_fall_back_to_base(self):
        """Backward-compat: a plain extra point with no altitudes/cevians
        entry at all must still use the old base-line fallback
        unchanged (this is deliberately NOT a bug -- only points that
        are supposed to be a bisector/median foot need 'cevians')."""
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}]}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)  # renders without error, same as before this change

    def test_fake_figure_citation_in_extra_labels_is_blocked(self):
        """Regression test for a real bug found in a user-generated PDF:
        an AI-GENERATED diagram displayed a specific 'চিত্ৰ 7.33'-style
        caption in extra_labels, making it look exactly like the book's
        own verbatim figure when it was not."""
        for fake in ["চিত্ৰ 7.33", "Figure 5.3", "Fig. 8.21", "figure 12"]:
            spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                    "extra_labels": [fake]}
            svg = dr.render_diagram(spec)
            self.assertNotIn(fake, svg, f"fake citation '{fake}' should have been blocked")

    def test_legitimate_extra_label_note_still_allowed(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "extra_labels": ["not to scale"]}
        svg = dr.render_diagram(spec)
        self.assertIn("not to scale", svg)

    def test_circle_renders(self):
        spec = {"diagram_type": "circle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        svg = dr.render_diagram(spec)
        self.assertIn("<circle", svg)


class TestCircleCenterAndTangent(unittest.TestCase):
    """PRODUCTION-AUDIT REGRESSION TESTS: found via adversarial testing
    against realistic circle questions ("O is the centre...", "a
    tangent is drawn from external point P..."). Before this fix, the
    circle diagram type had NO concept of a center point at all —
    every declared point, including one meant to be the center, was
    placed evenly spaced AROUND the circumference like every other
    point, and there was no way to represent a tangent-from-an-
    external-point construction (one of the most common real Class 10
    circle-question patterns) at all."""

    def _center_x_y(self, svg):
        import re
        m = re.search(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="2.5"', svg)
        return float(m.group(1)), float(m.group(2))

    def test_no_center_id_behaves_exactly_as_before(self):
        """Backward compatibility: omitting center_id must produce the
        exact same all-points-on-circumference rendering as always."""
        spec = {"diagram_type": "circle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        svg = dr.render_diagram(spec)
        x, y = self._center_x_y(svg)
        # first point is placed at the TOP of the circumference (schematic
        # placement, 12-o'clock position) -- its y differs from the true
        # center's y (110); only x coincidentally matches since "top" is
        # directly above center.
        self.assertNotAlmostEqual(y, 110.0, delta=0.5)

    def test_center_id_places_that_point_at_the_true_center(self):
        spec = {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "A"}, {"id": "B"}],
                "center_id": "O"}
        svg = dr.render_diagram(spec)
        x, y = self._center_x_y(svg)  # first drawn point marker is the center (drawn first)
        self.assertAlmostEqual(x, 130.0, delta=0.5)
        self.assertAlmostEqual(y, 110.0, delta=0.5)

    def test_unknown_center_id_is_rejected_not_silently_dropped(self):
        """PRODUCTION-AUDIT DESIGN CHANGE: previously, an unknown center_id
        was silently ignored at render time — the diagram still rendered,
        just with every point (including the one meant to be the center)
        placed on the circumference instead. That is exactly the kind of
        misleading output this project's own stated philosophy rejects
        everywhere else (see the triangle/quadrilateral/book-citation
        validators): if a question says "O is the centre" and the spec's
        own center_id doesn't match any declared point, that is a real,
        detectable inconsistency in Gemini's output, not something to
        silently paper over. _validate_circle_spec now catches this before
        rendering, so render_diagram correctly omits the diagram entirely
        (empty string) rather than publishing one with a lost "this is the
        center" semantic. The renderer's own internal center_id fallback
        (in _render_circle) is UNCHANGED and still exists as defense in
        depth for any caller that bypasses validate_diagram_spec directly."""
        spec = {"diagram_type": "circle", "points": [{"id": "A"}, {"id": "B"}], "center_id": "DOES_NOT_EXIST"}
        svg = dr.render_diagram(spec)  # must not crash
        self.assertEqual(svg, "")

        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("center_id" in issue for issue in issues))

    def test_valid_center_id_still_renders_normally(self):
        spec = {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "A"}, {"id": "B"}], "center_id": "O"}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)

    def test_tangent_construction_is_geometrically_exact(self):
        """The tangent line at each tangent point must be exactly
        perpendicular to that point's radius — the actual mathematical
        definition of tangency, not just a plausible-looking picture."""
        import re, math
        spec = {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "P"}, {"id": "A"}, {"id": "B"}],
                "center_id": "O", "tangent_from": {"external_point": "P", "tangent_points": ["A", "B"]},
                "dimensions": {"radius": 5, "distance": 13}}
        svg = dr.render_diagram(spec)
        self.assertEqual(svg.count('data-role="tangent-line"'), 2)
        self.assertEqual(svg.count('data-role="tangent-right-angle"'), 2)
        lines = re.findall(r'<line x1="([\d.\-]+)" y1="([\d.\-]+)" x2="([\d.\-]+)" y2="([\d.\-]+)"[^>]*'
                            r'data-role="tangent-line"', svg)
        radii = re.findall(r'<line x1="([\d.\-]+)" y1="([\d.\-]+)" x2="([\d.\-]+)" y2="([\d.\-]+)"[^>]*'
                            r'data-role="tangent-radius"', svg)
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(radii), 2)
        for tl, rd in zip(lines, radii):
            v1 = (float(tl[2]) - float(tl[0]), float(tl[3]) - float(tl[1]))
            v2 = (float(rd[2]) - float(rd[0]), float(rd[3]) - float(rd[1]))
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag = math.hypot(*v1) * math.hypot(*v2)
            angle = math.degrees(math.acos(max(-1, min(1, dot / mag))))
            self.assertAlmostEqual(angle, 90.0, delta=0.5)

    def test_tangent_construction_realistic_ratio_stays_within_canvas(self):
        """PRODUCTION-AUDIT REGRESSION TEST: a realistic radius:distance
        ratio (5:13, a very common textbook right-triangle case)
        previously placed the external point and its tangent points
        WELL outside the 260x220 canvas (confirmed: one real case
        landed at (286, -46)). Every drawn point must now stay inside."""
        import re
        spec = {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "P"}, {"id": "A"}, {"id": "B"}],
                "center_id": "O", "tangent_from": {"external_point": "P", "tangent_points": ["A", "B"]},
                "dimensions": {"radius": 5, "distance": 13}}
        svg = dr.render_diagram(spec)
        coords = re.findall(r'cx="([\d.\-]+)" cy="([\d.\-]+)" r="2.5"', svg)
        self.assertEqual(len(coords), 4)  # O, P, A, B
        for x, y in coords:
            self.assertGreaterEqual(float(x), 0)
            self.assertLessEqual(float(x), dr.W)
            self.assertGreaterEqual(float(y), 0)
            self.assertLessEqual(float(y), dr.H)

    def test_impossible_tangent_distance_declines_gracefully(self):
        """PRODUCTION-AUDIT REGRESSION TEST: if the given distance is
        not greater than the given radius, the external point would be
        ON or INSIDE the circle -- no tangent can exist. Must decline
        the tangent construction (never silently clamp to 'just
        outside' and draw a construction that contradicts the given
        numbers), and must never crash."""
        spec = {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "P"}, {"id": "A"}, {"id": "B"}],
                "center_id": "O", "tangent_from": {"external_point": "P", "tangent_points": ["A", "B"]},
                "dimensions": {"radius": 10, "distance": 5}}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        self.assertEqual(svg.count('data-role="tangent-line"'), 0)

    def test_malformed_tangent_from_never_crashes(self):
        for bad_tangent in ["not a dict", 123, None, {}, {"external_point": "GHOST"}]:
            spec = {"diagram_type": "circle", "points": [{"id": "O"}, {"id": "P"}],
                    "center_id": "O", "tangent_from": bad_tangent}
            svg = dr.render_diagram(spec)  # must not raise
            self.assertIsInstance(svg, str)

    def test_angle_valid_spec(self):
        spec = {"diagram_type": "angle", "vertex": "O",
                "rays": [{"id": "r1", "direction_deg": 0}, {"id": "r2", "direction_deg": 40}],
                "angle_marks": [{"between": ["r1", "r2"], "label": "40°"}]}
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertTrue(ok)
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_parallel_lines_valid_spec(self):
        spec = {"diagram_type": "parallel_lines", "lines": ["l1", "l2"],
                "angle_marks": [{"line": "l1", "position": "top_left", "label": "x"}]}
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_coordinate_plot_valid_spec(self):
        spec = {"diagram_type": "coordinate_plot",
                "points": [{"id": "A", "x": 2, "y": 3}, {"id": "B", "x": -1, "y": 1}],
                "segments": [{"from": "A", "to": "B"}]}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_coordinate_plot_background_grid_not_id_checked(self):
        """Regression test for the specific bug scenario described in the
        engineering brief: background grid lines are drawn from raw
        numeric coordinates (not declared point ids) and must NOT be
        rejected by the structural validator. This documents/locks in
        that the reported failure mode does not exist in this code."""
        spec = {"diagram_type": "coordinate_plot", "points": [{"id": "A", "x": 1, "y": 0}]}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)

    def test_coordinate_plot_axis_never_clipped_when_data_excludes_origin(self):
        """BUG FIX (found via diagram_final_check.py's post-render
        clipping check, run against this exact spec): min_x/max_x/
        min_y/max_y used to be derived from the data points alone, so
        when every point sits on one side of an axis (e.g. all y >= 1,
        never reaching y=0), the drawn x-axis — which must always cross
        at y=0 — landed outside the canvas entirely, a genuinely
        clipped axis. The origin must always be part of the plotted
        range, exactly like a real Cartesian plane always shows both
        axes regardless of where the data happens to sit."""
        from diagram_final_check import _check_no_clipped_content
        for points in (
            [{"id": "P", "x": 2, "y": 3}, {"id": "Q", "x": -1, "y": 1}],   # data never reaches y=0
            [{"id": "P", "x": 5, "y": 5}, {"id": "Q", "x": 8, "y": 9}],    # first quadrant, far from origin
            [{"id": "P", "x": -5, "y": -5}, {"id": "Q", "x": -8, "y": -9}],  # third quadrant, far from origin
        ):
            spec = {"diagram_type": "coordinate_plot", "points": points}
            svg = dr.render_diagram(spec)
            self.assertTrue(svg)
            ok, reason = _check_no_clipped_content(svg)
            self.assertTrue(ok, f"{reason} for points={points}")

    def test_coordinate_plot_dense_ticks_are_thinned(self):
        # V8 polish fix: a dense range like [0, 7] should not produce 8
        # overlapping ticks [0,1,2,3,4,5,6,7]; it should pick a wider step.
        spec = {"diagram_type": "coordinate_plot",
                "points": [{"id": "A", "x": 0, "y": 0}, {"id": "B", "x": 7, "y": 0}]}
        svg = dr.render_diagram(spec)
        self.assertLess(svg.count('text-anchor="middle"'), 7) # Should be ~4 ticks (0,2,4,6), not 8.

    def test_number_line_renders(self):
        svg = dr.render_diagram({"diagram_type": "number_line", "points": [{"value": 2}, {"value": 5}]})
        self.assertTrue(svg)

    def test_square_root_spiral_renders(self):
        svg = dr.render_diagram({"diagram_type": "square_root_spiral", "n": 5})
        self.assertTrue(svg)

    def test_unknown_type_fails_safe_to_empty(self):
        self.assertEqual(dr.render_diagram({"diagram_type": "not_a_real_type"}), "")

    def test_none_spec_fails_safe_to_empty(self):
        self.assertEqual(dr.render_diagram(None), "")


class TestQuadrilateralRenderer(unittest.TestCase):
    def _valid_spec(self, **overrides):
        spec = {"diagram_type": "quadrilateral", "shape": "rectangle",
                "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
                "equal_marks": [["AB", "CD"], ["BC", "DA"]],
                "right_angle_at": ["A", "B"], "diagonals": ["AC"]}
        spec.update(overrides)
        return spec

    def test_valid_spec_passes_all_gates(self):
        spec = self._valid_spec()
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_all_shape_families_render(self):
        for shape in ("parallelogram", "rectangle", "rhombus", "square", "trapezium", "general"):
            spec = self._valid_spec(shape=shape, equal_marks=[], right_angle_at=[], diagonals=[])
            svg = dr.render_diagram(spec)
            self.assertTrue(svg, f"shape {shape} produced empty svg")
            self.assertEqual(svg.count('data-role="edge"'), 4, f"shape {shape} should have 4 edges")

    def test_wrong_point_count_rejected(self):
        spec = self._valid_spec(points=[{"id": "A"}, {"id": "B"}, {"id": "C"}])
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_invalid_shape_rejected(self):
        spec = self._valid_spec(shape="hexagon")
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_dangling_equal_marks_id_rejected(self):
        spec = self._valid_spec(equal_marks=[["AB", "ZZ"]])
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_dangling_diagonal_rejected(self):
        spec = self._valid_spec(diagonals=["AZ"])
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_coverage_check_catches_missing_diagonal(self):
        # spec declares a diagonal but we render a version of the SVG
        # with that data-role stripped out, to prove the coverage check
        # actually looks at the rendered output rather than trusting the
        # spec blindly.
        spec = self._valid_spec()
        svg = dr.render_diagram(spec)
        tampered = svg.replace('data-role="diagonal"', 'data-role="not-a-diagonal"')
        cov_ok, issues = dr.verify_marker_coverage(spec, tampered)
        self.assertFalse(cov_ok)
        self.assertTrue(any("diagonal" in i for i in issues))

    def test_fake_figure_citation_in_extra_labels_is_blocked(self):
        spec = self._valid_spec(extra_labels=["চিত্ৰ 7.33"])
        svg = dr.render_diagram(spec)
        self.assertNotIn("চিত্ৰ 7.33", svg)

    def test_diagonal_intersection_point_renders_with_both_diagonals(self):
        """Regression test for a real bug found in a user-generated PDF:
        a proof referencing 'O, the intersection of the diagonals' had
        no way to make O appear in the diagram at all."""
        spec = self._valid_spec(diagonals=["AC", "BD"], diagonal_intersection_label="O",
                                 equal_marks=[], right_angle_at=[])
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertIn('data-role="diagonal-intersection"', svg)
        self.assertIn(">O<", svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_diagonal_intersection_label_requires_both_true_diagonals(self):
        # only ONE diagonal declared -- intersection point isn't defined
        spec = self._valid_spec(diagonals=["AC"], diagonal_intersection_label="O",
                                 equal_marks=[], right_angle_at=[])
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("diagonal_intersection_label" in i for i in issues))

    def test_diagonal_intersection_label_absent_renders_nothing_extra(self):
        spec = self._valid_spec(diagonals=["AC", "BD"], equal_marks=[], right_angle_at=[])
        svg = dr.render_diagram(spec)
        self.assertNotIn('data-role="diagonal-intersection"', svg)

    def test_coverage_check_catches_missing_intersection_point(self):
        spec = self._valid_spec(diagonals=["AC", "BD"], diagonal_intersection_label="O",
                                 equal_marks=[], right_angle_at=[])
        svg = dr.render_diagram(spec)
        tampered = svg.replace('data-role="diagonal-intersection"', 'data-role="stripped"')
        cov_ok, issues = dr.verify_marker_coverage(spec, tampered)
        self.assertFalse(cov_ok)


class TestTrigonometryRenderer(unittest.TestCase):
    def _valid_spec(self, **overrides):
        spec = {"diagram_type": "trigonometry",
                "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "right_angle_at": "A", "dimensions": {"adjacent": 4, "opposite": 3},
                "angle_at": "B", "angle_value_deg": 36.87, "show_ratio": "sin"}
        spec.update(overrides)
        return spec

    def test_valid_spec_passes_all_gates(self):
        spec = self._valid_spec()
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_right_angle_mark_always_present(self):
        spec = self._valid_spec()
        svg = dr.render_diagram(spec)
        self.assertEqual(svg.count('data-role="right-angle"'), 1)

    def test_triangle_is_drawn_to_scale_from_dimensions(self):
        # A 3-4-5 triangle should have its adjacent side visibly longer
        # than its opposite side.
        spec = self._valid_spec(dimensions={"adjacent": 4, "opposite": 3})
        svg = dr.render_diagram(spec)
        # This is a proxy for checking the rendered line lengths; a more
        # robust test would parse the SVG and compare coordinates.
        self.assertIn('data-role="dimension-label"', svg)

    def test_wrong_point_count_rejected(self):
        spec = self._valid_spec(points=[{"id": "A"}, {"id": "B"}])
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_invalid_ratio_rejected(self):
        spec = self._valid_spec(show_ratio="sec")
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_coverage_check_catches_missing_angle_arc(self):
        spec = self._valid_spec()
        svg = dr.render_diagram(spec)
        tampered = svg.replace('data-role="angle-arc"', 'data-role="stripped"')
        cov_ok, issues = dr.verify_marker_coverage(spec, tampered)
        self.assertFalse(cov_ok)


class TestStatisticsRenderer(unittest.TestCase):
    def _valid_spec(self, **overrides):
        spec = {"diagram_type": "statistics", "chart_type": "bar",
                "categories": ["A", "B", "C", "D"], "values": [10, 25, 15, 30],
                "x_label": "Category", "y_label": "Frequency"}
        spec.update(overrides)
        return spec

    def test_valid_spec_passes_all_gates(self):
        spec = self._valid_spec()
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_bar_count_matches_values(self):
        spec = self._valid_spec(values=[3, 7, 2, 9, 5], categories=["a", "b", "c", "d", "e"])
        svg = dr.render_diagram(spec)
        self.assertEqual(svg.count('data-role="bar"'), 5)

    def test_bars_are_scaled_not_uniform(self):
        """The largest value must produce the tallest bar — proves bars
        are computed from the actual numbers, not decorative/uniform."""
        spec = self._valid_spec(values=[5, 50], categories=["small", "big"], x_label=None, y_label=None)
        svg = dr.render_diagram(spec)
        import re
        heights = [float(h) for h in re.findall(r'height="([\d.]+)"', svg)]
        self.assertEqual(len(heights), 2)
        self.assertGreater(max(heights), min(heights) * 5)  # 50 vs 5 -> roughly 10x, allow margin

    def test_empty_values_rejected(self):
        ok, _ = dr.validate_diagram_spec(self._valid_spec(values=[]))
        self.assertFalse(ok)

    def test_histogram_unequal_class_widths_uses_proportional_bar_width(self):
        """PRODUCTION-AUDIT REGRESSION TEST: found via a real rendered
        PDF for a standard Class 9/10 unequal-class-width histogram
        question (0-10, 10-20, 20-40, 40-50). Before this fix, every
        class was drawn at an equal slot width regardless of its real
        interval width, with bar height set directly from raw
        frequency — mathematically wrong (a histogram's bar AREA, not
        height alone, must represent frequency; a class twice as wide
        must be drawn twice as wide, not just as tall as its raw
        frequency implies). This is the actual mathematical point of a
        histogram as distinct from a bar chart."""
        import re
        spec = {"diagram_type": "statistics", "chart_type": "histogram",
                "categories": ["0-10", "10-20", "20-40", "40-50"], "values": [5, 8, 18, 7]}
        svg = dr.render_diagram(spec)
        rects = re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)
        self.assertEqual(len(rects), 4)
        widths = [float(w) for _, _, w, _ in rects]
        # the "20-40" class (index 2) is twice as wide (20) as every
        # other class (10) -- its drawn bar must be twice as wide too
        self.assertAlmostEqual(widths[2], widths[0] * 2, places=1)
        self.assertAlmostEqual(widths[2], widths[1] * 2, places=1)
        self.assertAlmostEqual(widths[2], widths[3] * 2, places=1)
        # bars must be contiguous (share edges), a core histogram convention
        xs = [float(x) for x, _, _, _ in rects]
        for i in range(3):
            self.assertAlmostEqual(xs[i] + widths[i], xs[i + 1], places=1)
        # heights must reflect frequency DENSITY (freq/width), not raw
        # frequency -- class 2 has the highest raw frequency (18) but
        # NOT the highest density (18/20=0.9 vs class 1's 8/10=0.8 --
        # close, but class 1 must not be dramatically shorter despite
        # having a much lower raw frequency)
        heights = [float(h) for _, _, _, h in rects]
        density0, density1, density2, density3 = 5/10, 8/10, 18/20, 7/10
        # heights should be in the same ratio as densities
        scale = heights[0] / density0
        for h, d in zip(heights, (density0, density1, density2, density3)):
            self.assertAlmostEqual(h, d * scale, places=1)
        self.assertIn("Frequency Density", svg, "unequal-width histograms must auto-label the y-axis")

    def test_histogram_equal_class_widths_unaffected_by_the_fix(self):
        """Backward-compatibility guarantee: when class widths ARE all
        equal (the common case), bar geometry must be byte-identical
        to what the pre-fix equal-slot renderer always produced."""
        import re
        spec = {"diagram_type": "statistics", "chart_type": "histogram",
                "categories": ["0-10", "10-20", "20-30", "30-40"], "values": [5, 8, 18, 7]}
        svg = dr.render_diagram(spec)
        rects = re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)
        widths = [float(w) for _, _, w, _ in rects]
        self.assertEqual(len(set(widths)), 1, "all bars must remain equal width when classes are equal width")
        self.assertNotIn("Frequency Density", svg, "must not auto-label density when widths are already equal")

    def test_histogram_non_interval_categories_fall_back_unaffected(self):
        """Categories that don't parse as 'lower-upper' intervals (e.g.
        a histogram mislabeled with plain names) must fall back to the
        exact original equal-slot rendering, never crash."""
        import re
        spec = {"diagram_type": "statistics", "chart_type": "histogram",
                "categories": ["Group A", "Group B", "Group C"], "values": [3, 5, 2]}
        svg = dr.render_diagram(spec)
        rects = re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)
        widths = [float(w) for _, _, w, _ in rects]
        self.assertEqual(len(set(widths)), 1)

    def test_negative_value_rejected(self):
        ok, _ = dr.validate_diagram_spec(self._valid_spec(values=[5, -2, 3], categories=["a", "b", "c"]))
        self.assertFalse(ok)

    def test_category_length_mismatch_rejected(self):
        ok, _ = dr.validate_diagram_spec(self._valid_spec(categories=["a", "b"], values=[1, 2, 3]))
        self.assertFalse(ok)

    def test_invalid_chart_type_rejected(self):
        ok, _ = dr.validate_diagram_spec(self._valid_spec(chart_type="line"))
        self.assertFalse(ok, "a genuinely invalid chart_type like 'line' must be rejected")

    def test_histogram_bars_touch(self):
        spec = self._valid_spec(chart_type="histogram")
        svg = dr.render_diagram(spec)
        self.assertEqual(svg.count('data-role="bar"'), 4)

    def test_long_category_labels_are_rotated_not_left_overlapping(self):
        # Regression test for a real garbled chart found in production:
        # the WHO causes-of-death bar chart (Exercise 14.3 Q1) packed 8
        # long Assamese category names into narrow bar slots and every
        # label rendered horizontally, overlapping its neighbors into
        # unreadable text. Long labels must now rotate instead.
        long_categories = [
            "প্ৰসৱকালীন স্বাস্থ্যৰ অৱস্থা", "হৃদৰোগ", "শ্বাস-প্ৰশ্বাসৰ সমস্যা",
            "সংক্ৰামক ৰোগ", "আঘাত আৰু দুৰ্ঘটনা", "কৰ্কট ৰোগ",
            "মানসিক স্বাস্থ্যজনিত সমস্যা", "অন্যান্য কাৰণ",
        ]
        spec = self._valid_spec(values=[32, 25, 13, 5, 5, 5, 4, 22], categories=long_categories)
        svg = dr.render_diagram(spec)
        self.assertIn('data-role="x-tick-rotated"', svg,
                      "long category labels must rotate instead of overlapping horizontally")
        # Every category must still appear exactly once, just rotated.
        self.assertEqual(svg.count('data-role="x-tick-rotated"'), len(long_categories))
        self.assertEqual(svg.count('data-role="bar"'), len(long_categories))

    def test_short_category_labels_stay_horizontal(self):
        # Short single-letter labels (e.g. political-party bar chart A-F)
        # should NOT be rotated — that would look worse for the common
        # case where labels genuinely fit.
        spec = self._valid_spec(values=[75, 55, 38, 28, 12, 38], categories=["A", "B", "C", "D", "E", "F"])
        svg = dr.render_diagram(spec)
        self.assertNotIn('data-role="x-tick-rotated"', svg)
        self.assertEqual(svg.count('data-role="x-tick"'), 6)

    def test_pie_chart_valid_spec(self):
        spec = self._valid_spec(chart_type="pie", categories=["A", "B", "C"], values=[30, 50, 20])
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)
        self.assertEqual(svg.count('data-role="pie-slice"'), 3)

    def test_frequency_polygon_valid_spec(self):
        spec = {
            "diagram_type": "statistics", "chart_type": "frequency_polygon",
            "categories": [5, 15, 25, 35], "values": [3, 9, 10, 5]
        }
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        self.assertIn('data-role="line-plot"', svg)

    def test_ogive_valid_spec(self):
        spec = {
            "diagram_type": "statistics", "chart_type": "ogive", "ogive_type": "less_than",
            "categories": [10, 20, 30, 40], "values": [5, 13, 25, 30]
        }
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        self.assertIn('data-role="line-plot"', svg)

    def test_frequency_polygon_requires_numeric_categories(self):
        spec = {"diagram_type": "statistics", "chart_type": "frequency_polygon", "categories": ["A", "B"], "values": [5, 10]}
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

class TestSurfaceAreaVolumeRenderer(unittest.TestCase):
    def test_cuboid_valid_spec(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "cuboid",
                "dimensions": {"length": 10, "width": 5, "height": 7}}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)
        self.assertEqual(svg.count('data-role="dimension-label"'), 3)

    def test_cuboid_is_drawn_to_scale_from_dimensions(self):
        # A long, short, flat cuboid should look visibly different from a tall, narrow one.
        spec_long = {"diagram_type": "surface_area_volume", "solid": "cuboid",
                     "dimensions": {"length": 200, "width": 50, "height": 20}}
        svg_long = dr.render_diagram(spec_long)
        spec_tall = {"diagram_type": "surface_area_volume", "solid": "cuboid",
                     "dimensions": {"length": 50, "width": 50, "height": 200}}
        svg_tall = dr.render_diagram(spec_tall)
        self.assertNotEqual(svg_long, svg_tall)

    def test_cube_valid_spec(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "cube", "dimensions": {"side": 8}}
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)
        self.assertEqual(svg.count('data-role="dimension-label"'), 1)

    def test_cylinder_valid_spec(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "cylinder",
                "dimensions": {"radius": 4, "height": 12}}
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)
        self.assertEqual(svg.count('data-role="dimension-label"'), 2)

    def test_cone_valid_spec(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "cone",
                "dimensions": {"radius": 3, "height": 9}}
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)
        self.assertEqual(svg.count('data-role="dimension-label"'), 2)

    def test_sphere_valid_spec(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "sphere", "dimensions": {"radius": 6}}
        svg = dr.render_diagram(spec)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)
        self.assertEqual(svg.count('data-role="dimension-label"'), 1)

    def test_unknown_solid_rejected(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "pyramid", "dimensions": {}}
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_missing_required_dimension_rejected(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "cylinder", "dimensions": {"radius": 5}}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("height" in i for i in issues))

    def test_negative_dimension_rejected(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "cube", "dimensions": {"side": -3}}
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_non_numeric_dimension_rejected(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "sphere", "dimensions": {"radius": "big"}}
        ok, _ = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_coverage_check_catches_missing_outline(self):
        spec = {"diagram_type": "surface_area_volume", "solid": "sphere", "dimensions": {"radius": 6}}
        svg = dr.render_diagram(spec)
        tampered = svg.replace('data-role="solid-outline"', 'data-role="stripped"')
        cov_ok, issues = dr.verify_marker_coverage(spec, tampered)
        self.assertFalse(cov_ok)


class TestConstructionRenderer(unittest.TestCase):
    def _triangle_construction_spec(self):
        return {
            "diagram_type": "construction",
            "initial_points": [
                {"id": "B", "x": 0, "y": 0},
                {"id": "C", "x": 7, "y": 0}
            ],
            "construction_steps": [
                {"type": "line_segment", "from": "B", "to": "C", "label": "7 cm"},
                {"type": "arc", "center": "B", "radius": 5, "intersection_id": "A"},
                {"type": "arc", "center": "C", "radius": 6, "intersection_id": "A"},
                {"type": "join", "from": "A", "to": "B"},
                {"type": "join", "from": "A", "to": "C"}
            ]
        }

    def test_valid_triangle_construction_spec_passes_all_gates(self):
        spec = self._triangle_construction_spec()
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        cov_ok, cov_issues = dr.verify_marker_coverage(spec, svg)
        self.assertTrue(cov_ok, cov_issues)

    def test_renders_correct_number_of_elements(self):
        spec = self._triangle_construction_spec()
        svg = dr.render_diagram(spec)
        self.assertEqual(svg.count('data-role="construction-line"'), 1)
        self.assertEqual(svg.count('data-role="construction-arc"'), 2)
        self.assertEqual(svg.count('data-role="construction-join"'), 2)
        self.assertIn(">A<", svg)
        self.assertIn(">B<", svg)
        self.assertIn(">C<", svg)

    def test_missing_initial_points_rejected(self):
        spec = self._triangle_construction_spec()
        spec["initial_points"] = []
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_step_with_undefined_point_rejected(self):
        spec = self._triangle_construction_spec()
        spec["construction_steps"][0]["from"] = "Z" # Point Z is not in initial_points
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_geometrically_impossible_arcs_reject_whole_diagram_not_partial(self):
        """PRODUCTION-AUDIT REGRESSION TEST: a construction spec that
        passes STRUCTURAL validation (every step references a known
        point id, in valid order) can still be numerically unsolvable —
        e.g. BC=10 with two radius-1 arcs from B and C, which can never
        meet since neither reaches even a quarter of the way across.
        Before this fix, render_diagram() silently returned a
        "successful" SVG showing only the base line and two dangling,
        non-intersecting arcs — no vertex A, no sides AB/AC — passing
        every existing validation gate despite being a mathematically
        incomplete construction diagram for a triangle question. The
        fix must reject the WHOLE diagram (empty string) so
        final_pre_pdf_check correctly reports NO_DIAGRAM rather than a
        misleadingly "complete" one missing its own apex."""
        spec = {
            "diagram_type": "construction",
            "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 10, "y": 0}],
            "construction_steps": [
                {"type": "line_segment", "from": "B", "to": "C", "label": "10 cm"},
                {"type": "arc", "center": "B", "radius": 1, "intersection_id": "A"},
                {"type": "arc", "center": "C", "radius": 1, "intersection_id": "A"},
                {"type": "join", "from": "A", "to": "B"},
                {"type": "join", "from": "A", "to": "C"},
            ],
        }
        # structural validation alone (no numeric solving) has no way
        # to know these arcs won't meet -- this spec legitimately passes it
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)
        svg = dr.render_diagram(spec)
        self.assertEqual(svg, "", "an unsolvable construction must render to nothing, never a partial diagram")

    def test_partially_solvable_multi_point_construction_also_rejected(self):
        """Same defect, different shape: even if SOME points in a
        multi-step construction resolve fine, one unsolvable
        intersection anywhere in the sequence must still reject the
        entire diagram, not just omit that one step."""
        spec = {
            "diagram_type": "construction",
            "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 4, "y": 0}],
            "construction_steps": [
                {"type": "line_segment", "from": "B", "to": "C", "label": "4 cm"},
                {"type": "arc", "center": "B", "radius": 3, "intersection_id": "A"},
                {"type": "arc", "center": "C", "radius": 3, "intersection_id": "A"},  # this DOES intersect
                {"type": "join", "from": "A", "to": "B"},
                {"type": "join", "from": "A", "to": "C"},
                {"type": "arc", "center": "B", "radius": 1, "intersection_id": "E"},
                {"type": "arc", "center": "C", "radius": 1, "intersection_id": "E"},  # this does NOT
                {"type": "join", "from": "E", "to": "B"},
            ],
        }
        svg = dr.render_diagram(spec)
        self.assertEqual(svg, "", "one unsolvable point anywhere must reject the whole construction")


class TestPreviouslyUnvalidatedTypesNowValidated(unittest.TestCase):
    """PRODUCTION AUDIT: validate_diagram_spec's dispatch table never
    included "circle", "number_line", or "square_root_spiral" at all —
    each fell through to the generic "return True, []" default meant
    for genuinely unimplemented plugin types, so garbage input for
    these three types previously passed structural validation with
    zero issues raised. This class locks in the fix."""

    def test_circle_duplicate_point_ids_rejected(self):
        spec = {"diagram_type": "circle", "points": [{"id": "A"}, {"id": "A"}]}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)
        self.assertTrue(any("duplicate" in i for i in issues))

    def test_circle_tangent_without_center_id_rejected(self):
        spec = {"diagram_type": "circle", "points": [{"id": "P"}, {"id": "A"}, {"id": "B"}],
                "tangent_from": {"external_point": "P", "tangent_points": ["A", "B"]}}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_circle_negative_radius_rejected(self):
        spec = {"diagram_type": "circle", "points": [{"id": "A"}], "dimensions": {"radius": -5}}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_circle_well_formed_spec_accepted(self):
        spec = {"diagram_type": "circle",
                "points": [{"id": "O"}, {"id": "P"}, {"id": "A"}, {"id": "B"}],
                "center_id": "O",
                "tangent_from": {"external_point": "P", "tangent_points": ["A", "B"]},
                "dimensions": {"radius": 5, "distance": 13}}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)

    def test_number_line_range_max_not_greater_than_min_rejected(self):
        spec = {"diagram_type": "number_line", "range_min": 5, "range_max": 2}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_number_line_marked_point_outside_range_rejected(self):
        spec = {"diagram_type": "number_line", "range_min": 0, "range_max": 5,
                "marked_point": {"value": 99, "label": "P"}}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_number_line_well_formed_spec_accepted(self):
        spec = {"diagram_type": "number_line", "range_min": 0, "range_max": 5,
                "marked_point": {"value": 3, "label": "P"}}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)

    def test_square_root_spiral_steps_out_of_range_rejected(self):
        spec = {"diagram_type": "square_root_spiral", "steps": 25}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_square_root_spiral_non_numeric_steps_rejected(self):
        spec = {"diagram_type": "square_root_spiral", "steps": "many"}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertFalse(ok)

    def test_square_root_spiral_well_formed_spec_accepted(self):
        spec = {"diagram_type": "square_root_spiral", "steps": 6}
        ok, issues = dr.validate_diagram_spec(spec)
        self.assertTrue(ok, issues)


class TestUniversalLabelLayout(unittest.TestCase):
    """render_diagram() is the single choke point every built-in renderer
    AND every plugin's SVG passes through before reaching a question. Before
    this pass, label_layout.check_and_fix_labels() was only ever called
    directly by three plugins (unit_circle_plugin, circle_line_plugin,
    solid_geometry_plugin); every other built-in shape and every other
    plugin shipped with zero automatic label-overlap repositioning. These
    tests lock in that render_diagram() now runs the label-layout pass for
    ANY diagram_type, using mock.patch (rather than hunting for a specific
    geometric spec that happens to produce a colliding layout) so the test
    is a direct, unambiguous assertion about the integration point itself
    and won't rot if a renderer's internal label geometry changes later."""

    def test_render_diagram_invokes_label_layout_for_a_builtin_type(self):
        """'triangle' is a built-in renderer that has never itself called
        label_layout — proves the hook fires for renderer types beyond the
        three plugins that already called it manually."""
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        with mock.patch.object(label_layout, "check_and_fix_labels",
                                wraps=label_layout.check_and_fix_labels) as spy:
            svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        spy.assert_called_once()
        # The SVG handed to label_layout must be exactly the renderer's own
        # output (order of operations: render -> layout -> coverage check).
        self.assertIn("<svg", spy.call_args[0][0])

    def test_render_diagram_invokes_label_layout_for_every_registered_builtin_renderer(self):
        """Sweeps every type in dr._RENDERERS (minus the ones needing a
        nontrivial spec to render at all) to confirm none of them are
        silently skipped by the universal hook."""
        minimal_specs = {
            "triangle": {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
            "circle": {"diagram_type": "circle", "points": [{"id": "O"}], "dimensions": {"radius": 5}},
            "number_line": {"diagram_type": "number_line", "range_min": 0, "range_max": 5},
            "square_root_spiral": {"diagram_type": "square_root_spiral", "steps": 4},
        }
        for dtype, spec in minimal_specs.items():
            with self.subTest(diagram_type=dtype):
                with mock.patch.object(label_layout, "check_and_fix_labels",
                                        wraps=label_layout.check_and_fix_labels) as spy:
                    svg = dr.render_diagram(spec)
                self.assertTrue(svg, f"{dtype} failed to render at all")
                spy.assert_called_once()

    def test_label_layout_actually_resolves_a_real_collision_end_to_end(self):
        """Not just 'was it called' — an actual overlapping-label SVG fed
        straight through render_diagram()'s post-processing must come out
        with fewer/no overlaps, proving the hook does real work and isn't
        just wired up as a no-op passthrough."""
        colliding_svg = ('<svg viewBox="0 0 260 220" xmlns="http://www.w3.org/2000/svg">'
                          '<text x="100" y="100" font-size="13">A</text>'
                          '<text x="102" y="101" font-size="13">B</text>'
                          '</svg>')
        before = label_layout.find_overlaps(label_layout.parse_text_elements(colliding_svg))
        self.assertTrue(before, "test fixture itself must start out colliding")
        result = label_layout.check_and_fix_labels(colliding_svg)
        after = label_layout.find_overlaps(label_layout.parse_text_elements(result["svg"]))
        self.assertEqual(after, [])
        self.assertGreaterEqual(result["moved"], 1)

    def test_label_layout_pass_never_disturbs_marker_coverage(self):
        """Repositioning <text> labels must never touch the <line>/<polyline>
        tick-mark / right-angle markers verify_marker_coverage counts —
        this is what makes the universal hook safe to add without
        redesigning the existing post-render validation gate."""
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]], "right_angle_at": "B"}
        svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        ok, issues = dr.verify_marker_coverage(dr._normalize_spec(spec), svg)
        self.assertTrue(ok, issues)

    def test_label_layout_failure_never_blocks_rendering(self):
        """label_layout.py documents a fail-open guarantee (never raises);
        render_diagram()'s own wrapper around the call must honor that too
        — a broken label-layout pass degrades to the renderer's original
        output, never to an empty/failed diagram."""
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        with mock.patch.object(label_layout, "check_and_fix_labels",
                                side_effect=RuntimeError("boom")):
            svg = dr.render_diagram(spec)
        self.assertTrue(svg)
        self.assertIn("<svg", svg)


if __name__ == "__main__":
    unittest.main()
