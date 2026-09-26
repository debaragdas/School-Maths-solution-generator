"""
Regression tests for diagram_decision.py — the deterministic STAGE 1
decision layer between Gemini's raw output and diagram_renderer.py's
rendering.

Run: python3 -m unittest test_diagram_decision -v
"""
import unittest

import diagram_decision as dd


class TestBookDiagramPrecedence(unittest.TestCase):
    def test_book_diagram_present_drops_generated_spec(self):
        q = {"book_diagram_base64": "xxxx",
             "diagram_spec": {"diagram_type": "triangle", "points": [{"id": "A"}]},
             "question_text": "চিত্ৰ 3.14 চাই প্ৰমাণ কৰা", "given": ""}
        result = dd.decide_diagram(q)
        self.assertIsNone(result["diagram_spec"])
        self.assertEqual(result["decision"], dd.BOOK_DIAGRAM)

    def test_no_diagram_spec_passes_through_none(self):
        result = dd.decide_diagram({"diagram_spec": None})
        self.assertIsNone(result["diagram_spec"])
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)


class TestStage2BookCitationReverification(unittest.TestCase):
    """New in this revision: book_diagram_base64 is no longer trusted on
    its mere presence — the citation is independently re-derived from
    the question's own text."""

    def test_book_diagram_without_any_citation_in_text_is_rejected(self):
        # book_diagram_base64 is set (as if upstream attached one), but
        # the question's own text has no figure citation at all — this
        # should never happen if _attach_book_diagrams is working, but
        # the decision engine must not trust it blindly regardless.
        q = {"book_diagram_base64": "xxxx", "book_diagram_figure_ref": "3.14",
             "question_text": "প্ৰমাণ কৰা যে AB = AC", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_book_diagram_figure_ref_mismatch_is_rejected(self):
        # The recorded figure ref disagrees with what's actually in the
        # question's own text — a genuine inconsistency, not proof of a
        # verified citation.
        q = {"book_diagram_base64": "xxxx", "book_diagram_figure_ref": "9.99",
             "question_text": "চিত্ৰ 3.14 চাই প্ৰমাণ কৰা", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_book_diagram_with_matching_citation_is_accepted(self):
        q = {"book_diagram_base64": "xxxx", "book_diagram_figure_ref": "3.14",
             "question_text": "চিত্ৰ 3.14 চাই প্ৰমাণ কৰা", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.BOOK_DIAGRAM)


class TestUnrecognizedDiagramType(unittest.TestCase):
    """New in this revision: closes a real gap where
    diagram_renderer.validate_diagram_spec silently returns (True, [])
    for any diagram_type it has no dedicated structural check for —
    including a typo'd or entirely hallucinated type string."""

    def test_typo_diagram_type_rejected(self):
        q = {"diagram_spec": {"diagram_type": "traingle",  # typo, not a real type
                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
             "question_text": "ত্ৰিভুজ ABC আকি লওক।", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_completely_unknown_diagram_type_rejected(self):
        q = {"diagram_spec": {"diagram_type": "decorative_illustration",
                               "points": [{"id": "A"}]},
             "question_text": "কিবা এটা প্ৰশ্ন", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_known_type_still_accepted(self):
        q = {"diagram_spec": {"diagram_type": "triangle",
                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
             "question_text": "ত্ৰিভুজ ABC আকি লওক।", "given": "", "required": "", "steps": [], "final_answer": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.GENERATED_DIAGRAM)


class TestStage3TypeMismatch(unittest.TestCase):
    """New in this revision: a spec's own signature fields must agree
    with its declared diagram_type — a schema-footprint check, not a
    keyword search of the question text."""

    def test_solid_field_with_wrong_type_rejected(self):
        q = {"diagram_spec": {"diagram_type": "triangle", "solid": "cylinder",
                               "dimensions": {"radius": 4, "height": 12}},
             "question_text": "চোঙৰ আয়তন নিৰ্ণয় কৰা", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_chart_type_field_with_wrong_type_rejected(self):
        q = {"diagram_spec": {"diagram_type": "triangle", "chart_type": "bar",
                               "categories": ["A", "B"], "values": [1, 2]},
             "question_text": "দণ্ড লেখ অংকন কৰা", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_diagonals_field_with_wrong_type_rejected(self):
        q = {"diagram_spec": {"diagram_type": "triangle",
                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
                               "diagonals": ["AC"]},
             "question_text": "ABCD এটা বৰ্গক্ষেত্ৰ", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_matching_solid_type_accepted(self):
        q = {"diagram_spec": {"diagram_type": "surface_area_volume", "solid": "cylinder",
                               "dimensions": {"radius": 4, "height": 12}},
             "question_text": "চোঙৰ আয়তন নিৰ্ণয় কৰা যাৰ ব্যাসাৰ্ধ 4 আৰু উচ্চতা 12", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.GENERATED_DIAGRAM)

    def test_shared_points_field_is_not_flagged_as_mismatch(self):
        # "points" alone is legitimately shared by triangle/circle/
        # quadrilateral — must never be treated as a signature mismatch.
        q = {"diagram_spec": {"diagram_type": "circle",
                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
             "question_text": "বৃত্তত অন্তর্লিখিত ত্ৰিভুজ ABC", "given": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.GENERATED_DIAGRAM)


class TestStage5ReusesStructuralValidator(unittest.TestCase):
    """New in this revision: an unsafe/incomplete spec is rejected by
    reusing diagram_renderer.validate_diagram_spec BEFORE rendering is
    ever attempted, not discovered mid-render."""

    def test_construction_without_any_steps_rejected(self):
        q = {"construction_instruments": ["কম্পাছ"],
             "diagram_spec": {"diagram_type": "construction",
                               "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 7, "y": 0}]},
             "question_text": "B আৰু C ৰ মাজত ৰেখাখণ্ড BC অংকন কৰা।"}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)


class TestAngleGroundingBugfix(unittest.TestCase):
    """Regression test for a real bug found during this same audit: the
    previous version's grounding check only ever looked at a "points"-
    shaped field, so "angle" specs (which use "vertex"/"rays", not
    "points") were silently never grounding-checked at all."""

    def test_angle_spec_with_ungrounded_vertex_and_rays_rejected(self):
        q = {"diagram_spec": {"diagram_type": "angle", "vertex": "Q",
                               "rays": [{"id": "R", "direction_deg": 0}, {"id": "S", "direction_deg": 60}],
                               "angle_marks": [{"between": ["R", "S"], "label": "60°"}]},
             "question_text": "সৰল কৰা: 3/4 + 1/2", "given": "", "required": "", "steps": [], "final_answer": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.NO_DIAGRAM)

    def test_angle_spec_with_grounded_vertex_and_rays_accepted(self):
        q = {"diagram_spec": {"diagram_type": "angle", "vertex": "O",
                               "rays": [{"id": "A", "direction_deg": 0}, {"id": "B", "direction_deg": 60}],
                               "angle_marks": [{"between": ["A", "B"], "label": "60°"}]},
             "question_text": "O বিন্দুত OA আৰু OB ৰশ্মিৰে গঠিত ∠AOB = 60° নিৰ্ণয় কৰা।",
             "given": "", "required": "", "steps": [], "final_answer": ""}
        result = dd.decide_diagram(q)
        self.assertEqual(result["decision"], dd.GENERATED_DIAGRAM)


class TestInternalConsistency(unittest.TestCase):
    def test_instruments_set_but_wrong_diagram_type_rejected(self):
        q = {"construction_instruments": ["কম্পাছ", "স্কেল"],
             "diagram_spec": {"diagram_type": "triangle",
                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
             "question_text": "Construct triangle ABC with AB=5cm"}
        result = dd.decide_diagram(q)
        self.assertIsNone(result["diagram_spec"])

    def test_construction_type_without_instruments_rejected(self):
        q = {"construction_instruments": None,
             "diagram_spec": {"diagram_type": "construction",
                               "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 7, "y": 0}]},
             "question_text": "Prove BC = 7"}
        result = dd.decide_diagram(q)
        self.assertIsNone(result["diagram_spec"])

    def test_construction_type_with_instruments_is_consistent(self):
        q = {"construction_instruments": ["কম্পাছ", "স্কেল"],
             "diagram_spec": {"diagram_type": "construction",
                               "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 7, "y": 0}],
                               "construction_steps": [{"type": "line_segment", "from": "B", "to": "C"}]},
             "question_text": "B আৰু C ৰ মাজত ৰেখাখণ্ড BC অংকন কৰা।"}
        result = dd.decide_diagram(q)
        self.assertIsNotNone(result["diagram_spec"])


class TestTextualGrounding(unittest.TestCase):
    def test_ungrounded_points_rejected(self):
        q = {"diagram_spec": {"diagram_type": "triangle",
                               "points": [{"id": "X"}, {"id": "Y"}, {"id": "Z"}]},
             "question_text": "সমকোণী ত্ৰিভুজৰ কৰ্ণ নিৰ্ণয় কৰা।",
             "given": "", "required": "", "steps": [], "final_answer": ""}
        result = dd.decide_diagram(q)
        self.assertIsNone(result["diagram_spec"])

    def test_grounded_points_pass(self):
        q = {"diagram_spec": {"diagram_type": "triangle",
                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
             "question_text": "ত্ৰিভুজ ABC ত AB = AC, প্ৰমাণ কৰা যে angle B = angle C",
             "given": "", "required": "", "steps": [], "final_answer": ""}
        result = dd.decide_diagram(q)
        self.assertIsNotNone(result["diagram_spec"])

    def test_grounding_only_needs_one_point_present(self):
        # Point ids may legitimately be introduced across question_text/
        # given/required/steps rather than all in one field.
        q = {"diagram_spec": {"diagram_type": "triangle",
                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
             "question_text": "প্ৰমাণ কৰা যে ত্ৰিভুজটোৰ কোণসমূহ সমান।",
             "given": "AB = AC = BC", "required": "", "steps": [], "final_answer": ""}
        result = dd.decide_diagram(q)
        self.assertIsNotNone(result["diagram_spec"])

    def test_non_point_types_skip_grounding_check(self):
        # "statistics" has no letter-labelled points, so grounding is
        # a no-op — this must never reject a valid chart.
        q = {"diagram_spec": {"diagram_type": "statistics", "chart_type": "bar",
                               "categories": ["A", "B"], "values": [1, 2]},
             "question_text": "নিম্নলিখিত তথ্যৰ ভিত্তিত এটা দণ্ড লেখ অংকন কৰা।"}
        result = dd.decide_diagram(q)
        self.assertIsNotNone(result["diagram_spec"])


class TestFullyValidSpec(unittest.TestCase):
    def test_passes_through_unchanged(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        q = {"diagram_spec": spec,
             "question_text": "ত্ৰিভুজ ABC আকি লওক।",
             "given": "", "required": "", "steps": [], "final_answer": ""}
        result = dd.decide_diagram(q)
        # Stage 5 normalizes the spec (e.g. fills in default point labels),
        # so it is an equal-in-substance but no longer identical object —
        # check the decision and that the meaningful content survived.
        self.assertEqual(result["decision"], dd.GENERATED_DIAGRAM)
        self.assertIsNotNone(result["diagram_spec"])
        self.assertEqual(result["diagram_spec"]["diagram_type"], "triangle")
        self.assertEqual([p["id"] for p in result["diagram_spec"]["points"]], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
