"""
Regression tests for book_intelligence.py (Book Intelligence Layer:
figure embedding/search/similarity, canonical library, constraint
graph, ranking, confidence scoring, self-verification).

Run: python3 -m pytest test_book_intelligence.py -v
"""
import io
import unittest

import book_intelligence as bi


def _make_image(shape="circle", fill="white", size=(200, 200)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, fill)
    d = ImageDraw.Draw(img)
    if shape == "circle":
        d.ellipse([20, 20, size[0] - 20, size[1] - 20], outline="black", width=3)
    elif shape == "triangle":
        d.polygon([(size[0] // 2, 10), (10, size[1] - 10), (size[0] - 10, size[1] - 10)],
                   outline="black", width=3)
    elif shape == "square":
        d.rectangle([20, 20, size[0] - 20, size[1] - 20], outline="black", width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestFigureEmbeddingDatabase(unittest.TestCase):
    def test_identical_content_scores_1(self):
        img = _make_image("circle")
        self.assertEqual(bi.image_similarity(img, img), 1.0)

    def test_different_shapes_score_meaningfully_lower(self):
        circle, triangle = _make_image("circle"), _make_image("triangle")
        score = bi.image_similarity(circle, triangle)
        self.assertLess(score, 0.9)
        self.assertGreaterEqual(score, 0.0)

    def test_corrupt_bytes_return_none_not_zero(self):
        self.assertIsNone(bi.image_similarity(b"not an image", _make_image()))
        self.assertIsNone(bi.image_avg_hash(b"garbage"))

    def test_hamming_distance_zero_for_identical_hashes(self):
        h = bi.image_avg_hash(_make_image("square"))
        self.assertEqual(bi.hamming_distance(h, h), 0)

    def test_embedding_database_caches_hashes(self):
        figures = [{"bytes": _make_image("circle"), "figure_ref": "1.1"},
                   {"bytes": _make_image("triangle"), "figure_ref": "1.2"}]
        db = bi.FigureEmbeddingDatabase(figures)
        self.assertEqual(len(db), 2)
        entries = db.all_entries()
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(h is not None for _, _, h in entries))

    def test_embedding_database_handles_corrupt_figure_gracefully(self):
        figures = [{"bytes": b"corrupt", "figure_ref": "1.1"}]
        db = bi.FigureEmbeddingDatabase(figures)
        entries = db.all_entries()
        self.assertIsNone(entries[0][2])  # hash is None, not a crash


class TestSemanticFigureSearch(unittest.TestCase):
    def test_identical_text_scores_1(self):
        self.assertAlmostEqual(bi.text_similarity("চিত্ৰ 7.14", "চিত্ৰ 7.14"), 1.0)

    def test_disjoint_text_scores_0(self):
        self.assertEqual(bi.text_similarity("abc def", "xyz uvw"), 0.0)

    def test_empty_text_scores_0_never_raises(self):
        self.assertEqual(bi.text_similarity("", "something"), 0.0)
        self.assertEqual(bi.text_similarity("something", ""), 0.0)
        self.assertEqual(bi.text_similarity("", ""), 0.0)

    def test_partial_overlap_scores_between_0_and_1(self):
        score = bi.text_similarity("the quick brown fox", "the quick red fox")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_search_ranks_matching_figure_first(self):
        figures = [{"figure_ref": "7.20"}, {"figure_ref": "7.14"}, {"figure_ref": "3.1"}]
        results = bi.search_figures_by_text("চিত্ৰ 7.14 অনুসৰি", figures, top_k=3)
        self.assertEqual(results[0]["figure"]["figure_ref"], "7.14")

    def test_search_never_used_to_auto_attach_wrong_figures(self):
        """Documents the scope boundary: this is a ranking aid only."""
        figures = [{"figure_ref": "9.9"}]
        results = bi.search_figures_by_text("completely unrelated text", figures)
        # low-scoring results are still returned (for QA visibility),
        # never filtered out as "the answer" -- caller decides what to do
        self.assertEqual(len(results), 1)


class TestCanonicalDiagramLibrary(unittest.TestCase):
    def test_all_canonical_specs_pass_real_structural_validation(self):
        import diagram_renderer as dr
        for topic in bi.list_canonical_topics():
            spec = bi.get_canonical_spec(topic)
            spec.pop("description", None)
            ok, issues = dr.validate_diagram_spec(spec)
            self.assertTrue(ok, f"{topic}: {issues}")

    def test_all_canonical_specs_actually_render(self):
        import diagram_renderer as dr
        for topic in bi.list_canonical_topics():
            spec = bi.get_canonical_spec(topic)
            spec.pop("description", None)
            svg = dr.render_diagram(spec)
            self.assertTrue(svg, f"{topic} failed to render")

    def test_unknown_topic_returns_none(self):
        self.assertIsNone(bi.get_canonical_spec("not_a_real_topic"))

    def test_returned_spec_is_a_copy_not_the_original(self):
        spec1 = bi.get_canonical_spec("tangent_from_external_point")
        spec1["points"] = "mutated"
        spec2 = bi.get_canonical_spec("tangent_from_external_point")
        self.assertNotEqual(spec2["points"], "mutated")


class TestConstraintGraph(unittest.TestCase):
    def test_consistent_equal_marks_and_side_lengths(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]], "side_lengths": {"AB": 5, "AC": 5}}
        graph = bi.ConstraintGraph.from_triangle_spec(spec)
        ok, conflicts = graph.check_consistency(spec)
        self.assertTrue(ok, conflicts)

    def test_contradictory_equal_marks_and_side_lengths_detected(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]], "side_lengths": {"AB": 5, "AC": 9}}
        graph = bi.ConstraintGraph.from_triangle_spec(spec)
        ok, conflicts = graph.check_consistency(spec)
        self.assertFalse(ok)
        self.assertEqual(len(conflicts), 1)
        self.assertIn("AB", conflicts[0])
        self.assertIn("AC", conflicts[0])

    def test_contradictory_right_angle_and_angles_deg_detected(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "right_angle_at": "A", "angles_deg": {"A": 60}}
        graph = bi.ConstraintGraph.from_triangle_spec(spec)
        ok, conflicts = graph.check_consistency(spec)
        self.assertFalse(ok)

    def test_consistent_right_angle_and_angles_deg(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "right_angle_at": "A", "angles_deg": {"A": 90}}
        graph = bi.ConstraintGraph.from_triangle_spec(spec)
        ok, conflicts = graph.check_consistency(spec)
        self.assertTrue(ok, conflicts)

    def test_no_numeric_constraints_nothing_to_check(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "equal_marks": [["AB", "AC"]]}
        graph = bi.ConstraintGraph.from_triangle_spec(spec)
        ok, conflicts = graph.check_consistency(spec)
        self.assertTrue(ok)
        self.assertEqual(conflicts, [])

    def test_right_angle_at_as_list_handled(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}], "right_angle_at": ["A", "B"]}
        graph = bi.ConstraintGraph.from_triangle_spec(spec)
        self.assertEqual(graph.right_angles, {"A", "B"})

    def test_malformed_input_never_crashes(self):
        spec = {"points": "not a list", "equal_marks": "not a list", "right_angle_at": 12345}
        graph = bi.ConstraintGraph.from_triangle_spec(spec)
        ok, conflicts = graph.check_consistency(spec)  # must not raise
        self.assertIsInstance(ok, bool)


class TestFigureRanking(unittest.TestCase):
    def test_exact_match_ranks_first(self):
        figures = [{"figure_ref": "7.20", "source": "raster"}, {"figure_ref": "7.14", "source": "vector"}]
        ranked = bi.rank_figure_candidates("চিত্ৰ 7.14", "7.14", figures)
        self.assertEqual(ranked[0]["figure"]["figure_ref"], "7.14")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_no_cited_ref_still_ranks_by_text_similarity(self):
        figures = [{"figure_ref": "7.14", "source": "raster"}, {"figure_ref": "9.1", "source": "raster"}]
        ranked = bi.rank_figure_candidates("figure 7.14 shown", None, figures)
        self.assertEqual(ranked[0]["figure"]["figure_ref"], "7.14")

    def test_empty_figures_list_returns_empty(self):
        self.assertEqual(bi.rank_figure_candidates("text", "1.1", []), [])


class TestConfidenceScoring(unittest.TestCase):
    def test_all_signals_pass_scores_100(self):
        signals = {k: True for k in bi._CONFIDENCE_WEIGHTS}
        result = bi.confidence_score(signals)
        self.assertEqual(result["score"], 100.0)
        self.assertFalse(result["needs_review"])

    def test_no_signals_scores_0_and_needs_review(self):
        result = bi.confidence_score({})
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(result["needs_review"])

    def test_missing_signal_treated_as_failing_not_ignored(self):
        signals = {k: True for k in list(bi._CONFIDENCE_WEIGHTS)[:-1]}  # omit the last one
        result = bi.confidence_score(signals)
        self.assertLess(result["score"], 100.0)

    def test_proven_conflict_forces_needs_review_even_with_high_score(self):
        """PRODUCTION-AUDIT REGRESSION TEST: found while building this
        module. A spec scoring 80/100 (above the 70 threshold) but
        carrying a PROVEN contradiction was NOT flagged before this
        fix -- a hard, detected conflict must never be masked by
        enough other passing signals."""
        signals = {k: True for k in bi._CONFIDENCE_WEIGHTS}
        signals["constraint_consistent"] = False  # -20 points -> 80/100, still above threshold
        result = bi.confidence_score(signals, has_conflicts=True)
        self.assertEqual(result["score"], 80.0)
        self.assertTrue(result["needs_review"], "a proven conflict must force review regardless of score")

    def test_without_has_conflicts_flag_80_does_not_need_review(self):
        """Confirms the threshold-only path still behaves sensibly when
        there ISN'T a proven conflict (just a lower soft score)."""
        signals = {k: True for k in bi._CONFIDENCE_WEIGHTS}
        signals["constraint_consistent"] = False
        result = bi.confidence_score(signals, has_conflicts=False)
        self.assertFalse(result["needs_review"])


class TestSelfVerify(unittest.TestCase):
    def test_clean_generated_triangle_scores_high_no_conflicts(self):
        import diagram_decision as dd
        import diagram_renderer as dr
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 6, "BC": 12, "CA": 6 * 3 ** 0.5}, "right_angle_at": "A"}
        q = {"question_number": "1", "question_text": "...", "diagram_spec": spec}
        svg = dr.render_diagram(spec)
        report = bi.self_verify(q, dd.GENERATED_DIAGRAM, svg)
        self.assertEqual(report["confidence"]["score"], 100.0)
        self.assertFalse(report["confidence"]["needs_review"])
        self.assertEqual(report["conflicts"], [])

    def test_contradictory_spec_flagged(self):
        import diagram_decision as dd
        import diagram_renderer as dr
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]], "side_lengths": {"AB": 5, "AC": 9}}
        q = {"question_number": "2", "question_text": "...", "diagram_spec": spec}
        svg = dr.render_diagram(spec)
        report = bi.self_verify(q, dd.GENERATED_DIAGRAM, svg)
        self.assertTrue(report["confidence"]["needs_review"])
        self.assertTrue(len(report["conflicts"]) >= 1)

    def test_no_diagram_case_handled(self):
        import diagram_decision as dd
        q = {"question_number": "3", "diagram_spec": None}
        report = bi.self_verify(q, dd.NO_DIAGRAM, "")
        self.assertEqual(report["final_decision"], dd.NO_DIAGRAM)
        self.assertIsInstance(report["confidence"], dict)

    def test_legitimately_no_diagram_needed_is_not_flagged(self):
        """PRODUCTION-AUDIT FIX: previously EVERY NO_DIAGRAM question was
        flagged needs_review=True unconditionally, including the
        overwhelming majority-case of plain algebra/arithmetic that
        never needed a diagram at all — flooding reviewers with false
        positives. A question whose reason is the plain "doesn't need
        one" message, with no safety-net flag, must NOT be flagged."""
        import diagram_decision as dd
        q = {"question_number": "4", "question_text": "সৰল কৰা: 3/4 + 1/2",
             "diagram_spec": None,
             "diagram_decision_reason": "no diagram_spec — question doesn't need one",
             "diagram_safety_net_flagged": False}
        report = bi.self_verify(q, dd.NO_DIAGRAM, "")
        self.assertFalse(report["confidence"]["needs_review"])
        self.assertEqual(report["confidence"]["score"], 100.0)
        self.assertEqual(report["conflicts"], [])

    def test_safety_net_flagged_no_diagram_is_still_flagged(self):
        """A question the safety net could NOT confirm was safe to skip
        must still be flagged for review — this is exactly the audit
        trail diagram_safety_net.py exists to create."""
        import diagram_decision as dd
        q = {"question_number": "5", "question_text": "ত্ৰিভুজ PQR ...",
             "diagram_spec": None,
             "diagram_decision_reason": "no diagram_spec — question doesn't need one",
             "diagram_safety_net_flagged": True}
        report = bi.self_verify(q, dd.NO_DIAGRAM, "")
        self.assertTrue(report["confidence"]["needs_review"])
        self.assertTrue(any("safety-net" in c for c in report["conflicts"]))

    def test_rejected_spec_no_diagram_is_still_flagged(self):
        """A question that DID have a diagram_spec, which was rejected
        by validation, is real lost content — must stay flagged even
        though the final decision is NO_DIAGRAM."""
        import diagram_decision as dd
        q = {"question_number": "6", "question_text": "ত্ৰিভুজ ABC ...",
             "diagram_spec": {"diagram_type": "triangle", "points": []},
             "diagram_decision_reason": "triangle requires at least 3 uniquely-identified points, found 0"}
        report = bi.self_verify(q, dd.NO_DIAGRAM, "")
        self.assertTrue(report["confidence"]["needs_review"])
        self.assertTrue(any("rejected" in c for c in report["conflicts"]))

    def test_book_diagram_case_handled(self):
        import diagram_decision as dd
        q = {"question_number": "4", "question_text": "চিত্ৰ 7.14 চাওক",
             "book_diagram_base64": "somebase64data", "book_diagram_figure_ref": "7.14",
             "diagram_spec": None}
        report = bi.self_verify(q, dd.BOOK_DIAGRAM, "")
        self.assertEqual(report["final_decision"], dd.BOOK_DIAGRAM)
        self.assertFalse(report["confidence"]["needs_review"])

    def test_never_raises_on_malformed_question(self):
        import diagram_decision as dd
        q = {"diagram_spec": "not a dict"}
        report = bi.self_verify(q, dd.GENERATED_DIAGRAM, "")  # must not raise
        self.assertIsInstance(report, dict)


if __name__ == "__main__":
    unittest.main()
