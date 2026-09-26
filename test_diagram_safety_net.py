"""
Regression tests for diagram_safety_net.py.

Run: python3 -m unittest test_diagram_safety_net -v
"""
import unittest
from unittest.mock import patch

import diagram_safety_net as dsn
from diagram_decision import NO_DIAGRAM, BOOK_DIAGRAM, GENERATED_DIAGRAM


class TestGeometryLikeDetection(unittest.TestCase):
    def test_triangle_vocabulary_flagged(self):
        q = {"question_text": "ত্ৰিভুজ ABC ত AB = AC হ'লে প্ৰমাণ কৰা যে", "given": "", "required": ""}
        self.assertTrue(dsn._looks_like_geometry_question(q))

    def test_english_construct_flagged(self):
        q = {"question_text": "Construct a triangle with sides 5, 6, 7 cm.", "given": "", "required": ""}
        self.assertTrue(dsn._looks_like_geometry_question(q))

    def test_vertex_list_pattern_flagged(self):
        q = {"question_text": "In quadrilateral P, Q, R and S, PQ is parallel to RS.", "given": "", "required": ""}
        self.assertTrue(dsn._looks_like_geometry_question(q))

    def test_pure_arithmetic_not_flagged(self):
        q = {"question_text": "3/4 + 1/2 সৰল কৰা।", "given": "", "required": ""}
        self.assertFalse(dsn._looks_like_geometry_question(q))

    def test_empty_text_not_flagged(self):
        self.assertFalse(dsn._looks_like_geometry_question({}))


class TestRecoverableGate(unittest.TestCase):
    def test_no_diagram_with_missing_spec_reason_is_recoverable(self):
        q = {"diagram_decision": NO_DIAGRAM,
             "diagram_decision_reason": "no diagram_spec — question doesn't need one",
             "book_diagram_base64": None}
        self.assertTrue(dsn._is_recoverable(q))

    def test_book_diagram_question_never_recoverable(self):
        q = {"diagram_decision": NO_DIAGRAM,
             "diagram_decision_reason": "no diagram_spec — question doesn't need one",
             "book_diagram_base64": "xxxx"}
        self.assertFalse(dsn._is_recoverable(q))

    def test_rejected_spec_not_recoverable(self):
        # A real diagram_spec existed and was correctly rejected by
        # diagram_decision's own Stage 3/4/5 checks — this module must
        # never re-guess over an already-reasoned rejection.
        q = {"diagram_decision": NO_DIAGRAM,
             "diagram_decision_reason": "diagram_type 'blah' is not a recognized/renderable type",
             "book_diagram_base64": None}
        self.assertFalse(dsn._is_recoverable(q))

    def test_generated_diagram_not_recoverable(self):
        q = {"diagram_decision": GENERATED_DIAGRAM,
             "diagram_decision_reason": "passed every decision-engine stage; forwarded to the renderer",
             "book_diagram_base64": None}
        self.assertFalse(dsn._is_recoverable(q))

    def test_book_diagram_decision_not_recoverable(self):
        q = {"diagram_decision": BOOK_DIAGRAM,
             "diagram_decision_reason": "verified textbook citation '3.14' takes precedence",
             "book_diagram_base64": "xxxx"}
        self.assertFalse(dsn._is_recoverable(q))


class TestQuestionTextBlobIncludesWorkedSolution(unittest.TestCase):
    """PRODUCTION-AUDIT FIX (final pre-launch round): the rebuild/recovery
    call must see the SAME text diagram_decision._is_textually_grounded
    later validates against — question_text/given/required/steps/
    final_answer — not just the first three."""

    def test_steps_and_final_answer_are_included(self):
        q = {"question_text": "Prove BC^2 = AB^2 + AC^2.", "given": "right angle at A",
             "required": "prove", "steps": ["Drop perpendicular AD from A to BC."],
             "final_answer": "Hence proved using point D."}
        blob = dsn._question_text_blob(q)
        self.assertIn("Prove BC", blob)
        self.assertIn("perpendicular AD from A to BC", blob)
        self.assertIn("Hence proved using point D", blob)

    def test_missing_fields_do_not_raise(self):
        self.assertEqual(dsn._question_text_blob({}), "")

    def test_non_list_steps_field_handled_gracefully(self):
        q = {"question_text": "q", "steps": "single step string"}
        blob = dsn._question_text_blob(q)
        self.assertIn("single step string", blob)


class TestConstructionInstrumentsThreading(unittest.TestCase):
    """PRODUCTION-AUDIT FIX (final pre-launch round): reproduces and
    fixes the bug where a recovered diagram_type="construction" spec
    was rejected by decide_diagram's Stage 4 consistency check because
    construction_instruments was never threaded from the recovery
    response onto the question before decide_diagram ran."""

    def _flagged_question(self):
        return {
            "question_number": "5", "sub_part": None,
            "question_text": "Construct triangle ABC with AB=5cm, BC=6cm.",
            "given": "", "required": "", "construction_instruments": None,
            "diagram_decision": NO_DIAGRAM,
            "diagram_decision_reason": "no diagram_spec — question doesn't need one",
            "book_diagram_base64": None, "diagram_spec": None,
        }

    @patch("diagram_safety_net._call_gemini_recovery")
    def test_construction_spec_with_matching_instructions_is_accepted(self, mock_call):
        mock_call.return_value = {
            "needs_diagram": True,
            "diagram_spec": {
                "diagram_type": "construction",
                "initial_points": [{"id": "A", "x": 0, "y": 0}, {"id": "B", "x": 5, "y": 0}],
                "construction_steps": [{"type": "line_segment", "from": "A", "to": "B", "label": "5 cm"}],
            },
            "construction_instruments": ["Draw AB = 5cm with a ruler."],
        }
        q = self._flagged_question()
        dsn.apply_diagram_safety_net([q], class_name=9, exercise_label="7.1")
        self.assertIsNotNone(q["diagram_spec"], "must not be rejected for missing construction_instruments")
        self.assertEqual(q["construction_instruments"], ["Draw AB = 5cm with a ruler."])
        self.assertEqual(q["diagram_decision"], GENERATED_DIAGRAM)

    @patch("diagram_safety_net._call_gemini_recovery")
    def test_stale_construction_instruments_do_not_poison_a_later_ordinary_spec(self, mock_call):
        mock_call.return_value = {
            "needs_diagram": True,
            "diagram_spec": {"diagram_type": "triangle",
                              "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]},
            "construction_instruments": None,
        }
        q = self._flagged_question()
        q["construction_instruments"] = ["stale leftover from a previous, now-discarded diagram"]
        dsn.apply_diagram_safety_net([q], class_name=9, exercise_label="7.1")
        self.assertIsNotNone(q["diagram_spec"])
        self.assertIsNone(q["construction_instruments"])

    @patch("diagram_safety_net._call_gemini_recovery")
    def test_rejected_recovery_clears_construction_instruments_too(self, mock_call):
        mock_call.return_value = {
            "needs_diagram": True,
            "diagram_spec": {"diagram_type": "triangle"},  # missing required fields -> rejected
            "construction_instruments": None,
        }
        q = self._flagged_question()
        q["construction_instruments"] = ["stale"]
        dsn.apply_diagram_safety_net([q], class_name=9, exercise_label="7.1")
        self.assertIsNone(q["diagram_spec"])
        self.assertIsNone(q["construction_instruments"])


class TestApplySafetyNet(unittest.TestCase):
    def _flagged_question(self):
        return {
            "question_number": "5", "sub_part": None,
            "question_text": "ত্ৰিভুজ ABC ত ∠A = 90° হ'লে প্ৰমাণ কৰা যে BC^2 = AB^2 + AC^2",
            "given": "", "required": "",
            "diagram_decision": NO_DIAGRAM,
            "diagram_decision_reason": "no diagram_spec — question doesn't need one",
            "book_diagram_base64": None,
            "diagram_spec": None,
        }

    def test_no_candidates_is_a_no_op(self):
        questions = [{
            "question_number": "1", "question_text": "3/4 + 1/2 সৰল কৰা।",
            "diagram_decision": NO_DIAGRAM,
            "diagram_decision_reason": "no diagram_spec — question doesn't need one",
            "book_diagram_base64": None,
        }]
        dsn.apply_diagram_safety_net(questions, class_name=9, exercise_label="7.1")
        self.assertNotIn("diagram_safety_net_flagged", questions[0])

    @patch("diagram_safety_net._call_gemini_recovery")
    def test_recovery_call_failure_fails_open(self, mock_call):
        mock_call.side_effect = RuntimeError("network down")
        q = self._flagged_question()
        dsn.apply_diagram_safety_net([q], class_name=9, exercise_label="7.1")
        self.assertIsNone(q["diagram_spec"])
        self.assertTrue(q["diagram_safety_net_flagged"])

    @patch("diagram_safety_net._call_gemini_recovery")
    def test_second_opinion_confirms_no_diagram_needed(self, mock_call):
        mock_call.return_value = {"needs_diagram": False, "diagram_spec": None}
        q = self._flagged_question()
        dsn.apply_diagram_safety_net([q], class_name=9, exercise_label="7.1")
        self.assertIsNone(q["diagram_spec"])
        self.assertFalse(q["diagram_safety_net_flagged"])

    @patch("diagram_safety_net._call_gemini_recovery")
    def test_recovered_spec_that_fails_validation_is_discarded(self, mock_call):
        mock_call.return_value = {
            "needs_diagram": True,
            # Missing required fields -> should fail validate_diagram_spec
            # inside decide_diagram, and be discarded rather than published.
            "diagram_spec": {"diagram_type": "triangle"},
        }
        q = self._flagged_question()
        dsn.apply_diagram_safety_net([q], class_name=9, exercise_label="7.1")
        self.assertIsNone(q["diagram_spec"])
        self.assertTrue(q["diagram_safety_net_flagged"])

    @patch("diagram_safety_net._call_gemini_recovery")
    def test_valid_recovered_spec_is_accepted_via_decide_diagram(self, mock_call):
        mock_call.return_value = {
            "needs_diagram": True,
            "diagram_spec": {
                "diagram_type": "triangle",
                "points": [{"id": "A", "label": "A"}, {"id": "B", "label": "B"}, {"id": "C", "label": "C"}],
            },
        }
        q = self._flagged_question()
        dsn.apply_diagram_safety_net([q], class_name=9, exercise_label="7.1")
        self.assertFalse(q["diagram_safety_net_flagged"])
        self.assertIsNotNone(q["diagram_spec"])
        self.assertEqual(q["diagram_decision"], GENERATED_DIAGRAM)
        self.assertIn("safety-net recovery", q["diagram_decision_reason"])


if __name__ == "__main__":
    unittest.main()
