"""Tests for the post-attach figure-caption cross-check
(solver._verify_figure_caption_matches + its wiring inside
_attach_book_diagrams, config.FIGURE_MATCH_VERIFICATION gate) and the
review webapp's exposure of the resulting flags."""
import base64
import json
import unittest
from unittest import mock

import config
import solver


class _FakeResp:
    def __init__(self, payload):
        self.text = json.dumps(payload)


def _img(ref="7.36", source="vision"):
    return {"bytes": b"PNG", "ext": "png", "figure_ref": ref, "source": source}


def _attached_q(ref="7.36"):
    return {
        "question_number": "5",
        "book_diagram_base64": base64.b64encode(b"PNG").decode("ascii"),
        "book_diagram_figure_ref": ref,
    }


class TestCaptionCrossCheck(unittest.TestCase):
    def setUp(self):
        self._old_gate = config.FIGURE_MATCH_VERIFICATION
        config.FIGURE_MATCH_VERIFICATION = True

    def tearDown(self):
        config.FIGURE_MATCH_VERIFICATION = self._old_gate

    def test_exact_agreement_marks_verified(self):
        q = _attached_q("7.36")
        with mock.patch.object(solver, "_call_gemini_caption_read",
                               return_value=_FakeResp({"figure_ref": "7.36"})):
            solver._verify_figure_caption_matches([(q, _img())], "7.5")
        self.assertTrue(q.get("diagram_match_verified"))
        self.assertFalse(q.get("diagram_match_low_confidence", False))

    def test_disagreement_flags_low_confidence_and_keeps_attachment(self):
        q = _attached_q("7.36")
        with mock.patch.object(solver, "_call_gemini_caption_read",
                               return_value=_FakeResp({"figure_ref": "7.38"})):
            solver._verify_figure_caption_matches([(q, _img())], "7.5")
        self.assertTrue(q.get("diagram_match_low_confidence"))
        # fail-open: attachment must survive a flagged check
        self.assertTrue(q.get("book_diagram_base64"))
        notes = " ".join(q.get("review_notes", []))
        self.assertIn("mismatch", notes.lower())

    def test_unreadable_caption_flags_low_confidence(self):
        q = _attached_q("7.36")
        with mock.patch.object(solver, "_call_gemini_caption_read",
                               return_value=_FakeResp({"figure_ref": None})):
            solver._verify_figure_caption_matches([(q, _img())], "7.5")
        self.assertTrue(q.get("diagram_match_low_confidence"))

    def test_assamese_digit_reread_normalizes_to_ascii_agreement(self):
        q = _attached_q("7.36")
        with mock.patch.object(solver, "_call_gemini_caption_read",
                               return_value=_FakeResp({"figure_ref": "৭.৩৬"})):
            solver._verify_figure_caption_matches([(q, _img())], "7.5")
        self.assertTrue(q.get("diagram_match_verified"))

    def test_deterministic_sources_are_never_rechecked(self):
        q = _attached_q()
        with mock.patch.object(solver, "_call_gemini_caption_read") as call:
            solver._verify_figure_caption_matches(
                [(q, _img(source="raster")), (q, _img(source="vector_multipage"))], "7.5")
        call.assert_not_called()
        self.assertFalse(q.get("diagram_match_low_confidence", False))

    def test_call_failure_fails_open_without_crash_or_flag(self):
        q = _attached_q()
        with mock.patch.object(solver, "_call_gemini_caption_read",
                               side_effect=RuntimeError("429 quota")):
            solver._verify_figure_caption_matches([(q, _img())], "7.5")
        self.assertFalse(q.get("diagram_match_low_confidence", False))
        self.assertFalse(q.get("diagram_match_verified", False))
        # NOTE: the config.FIGURE_MATCH_VERIFICATION on/off decision lives
        # in _attach_book_diagrams (the only production caller), covered by
        # TestWiringInAttach below — this function always checks when asked.


class TestWiringInAttach(unittest.TestCase):
    """_attach_book_diagrams must actually route matched pairs into the
    cross-check when the gate is on (and not when off)."""

    def setUp(self):
        self._old_gate = config.FIGURE_MATCH_VERIFICATION

    def tearDown(self):
        config.FIGURE_MATCH_VERIFICATION = self._old_gate

    def _run_attach(self, images, questions):
        with mock.patch.object(solver, "get_verified_figures", return_value=images), \
             mock.patch.object(solver, "_call_gemini_caption_read",
                               return_value=_FakeResp({"figure_ref": "3.14"})) as cap:
            solver._attach_book_diagrams(questions, "ex.pdf", "3.1", 9, 3)
        return cap

    def test_vision_sourced_attachment_is_cross_checked(self):
        questions = [_attached_q("3.14")]
        questions[0]["question_text"] = "চিত্ৰ 3.14 চোৱা।"
        cap = self._run_attach([_img("3.14")], questions)
        self.assertEqual(cap.call_count, 1)
        self.assertTrue(questions[0].get("diagram_match_verified"))

    def test_gate_off_skips_check(self):
        config.FIGURE_MATCH_VERIFICATION = False
        questions = [_attached_q("3.14")]
        questions[0]["question_text"] = "চিত্ৰ 3.14 চোৱা।"
        cap = self._run_attach([_img("3.14")], questions)
        cap.assert_not_called()


class TestWebappSummaryFields(unittest.TestCase):
    def test_summary_exposes_both_flags(self):
        import review_webapp
        state = {"question_acceptance": {}}
        q = {"question_number": "1", "diagram_match_verified": True,
             "book_diagram_base64": "x"}
        summary = review_webapp._question_summary(state, q)
        self.assertTrue(summary["diagram_match_verified"])
        self.assertFalse(summary["diagram_match_low_confidence"])


if __name__ == "__main__":
    unittest.main()
