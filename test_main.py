"""
Regression tests for main.py's process_exercise() — specifically the
NEW review-aware skip/resume logic: an exercise is only ever skipped
once it's fully APPROVED; one with an existing draft but no approval
yet is left alone for Stage 2's interactive review instead of being
re-solved (which would waste a Gemini call and could discard reviewer
progress); a pre-existing PDF with no review state at all (predates
this feature) is reported as unreviewable rather than silently treated
as forever-approved.

`fitz` / `google.genai` are stubbed exactly as the other test files do.

Run: python3 -m pytest test_main.py -v
"""
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


def _install_fakes():
    if "fitz" not in sys.modules:
        fake_fitz = types.ModuleType("fitz")
        fake_fitz.open = mock.Mock()
        sys.modules["fitz"] = fake_fitz
    if "google" not in sys.modules:
        google_pkg = types.ModuleType("google")
        genai_mod = types.ModuleType("google.genai")
        genai_mod.Client = mock.Mock()
        types_mod = types.ModuleType("google.genai.types")
        types_mod.Part = mock.Mock()
        types_mod.GenerateContentConfig = mock.Mock()
        types_mod.ThinkingConfig = mock.Mock()
        google_pkg.genai = genai_mod
        sys.modules["google"] = google_pkg
        sys.modules["google.genai"] = genai_mod
        sys.modules["google.genai.types"] = types_mod


_install_fakes()

import config
import review_state
import main


class TestProcessExerciseResumability(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_output_dir = config.OUTPUT_DIR
        config.OUTPUT_DIR = os.path.join(self.tmpdir, "output")
        self.exercise = {"label": "7.1", "path": "/fake/exercise_7.1.pdf"}
        self.output_path = os.path.join(config.OUTPUT_DIR, "class_9", "chapter_7", "ex_7_1.pdf")

    def tearDown(self):
        config.OUTPUT_DIR = self._orig_output_dir
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_pdf(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "wb") as f:
            f.write(b"%PDF fake" + b"X" * 2000)

    def test_approved_exercise_is_skipped_without_resolving(self):
        self._write_pdf()
        review_state.init_state(self.output_path, {"exercise_label": "7.1", "questions": []},
                                 9, 7, "ch", "7.1")
        review_state.approve(self.output_path)

        with mock.patch("main.solve_exercise") as mock_solve:
            result = main.process_exercise(self.exercise, 9, 7, "ch")
        mock_solve.assert_not_called()
        self.assertEqual(result, "skipped: 7.1")

    def test_pending_review_exercise_is_left_for_stage_2_without_resolving(self):
        self._write_pdf()
        review_state.init_state(self.output_path, {"exercise_label": "7.1", "questions": []},
                                 9, 7, "ch", "7.1")
        # never approved — still pending_review

        with mock.patch("main.solve_exercise") as mock_solve:
            result = main.process_exercise(self.exercise, 9, 7, "ch")
        mock_solve.assert_not_called()
        self.assertEqual(result, "awaiting_review: 7.1")

    def test_legacy_pdf_with_no_review_state_is_reported_not_silently_approved(self):
        self._write_pdf()
        # No review_state.init_state call at all — simulates a PDF from
        # before this feature existed.
        with mock.patch("main.solve_exercise") as mock_solve:
            result = main.process_exercise(self.exercise, 9, 7, "ch")
        mock_solve.assert_not_called()
        self.assertEqual(result, "unreviewable_legacy_pdf: 7.1")

    def _fake_render_pdf(self, html, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"%PDF" + b"X" * 2000)

    def test_brand_new_exercise_solves_and_initializes_review_state(self):
        solved = {"exercise_label": "7.1", "questions": [
            {"question_number": "1", "sub_part": None, "final_answer": "42"},
        ]}
        with mock.patch("main.solve_exercise", return_value=solved) as mock_solve, \
             mock.patch("main.verify_solution", return_value=(True, [])), \
             mock.patch("main.render_exercise_html", return_value="<html></html>"), \
             mock.patch("main.render_pdf", side_effect=self._fake_render_pdf):
            result = main.process_exercise(self.exercise, 9, 7, "ch")
        mock_solve.assert_called_once()
        self.assertEqual(result, "done: 7.1")
        state = review_state.load_state(self.output_path)
        self.assertIsNotNone(state)
        self.assertEqual(state["status"], review_state.STATUS_PENDING)


class TestExerciseFilter(unittest.TestCase):
    """Regression tests for config.EXERCISE_FILTER — solving just one
    (or a few) exercises instead of the whole chapter."""

    def _exercises(self):
        return [{"label": "7.1"}, {"label": "7.2"}, {"label": "7.4"}]

    def test_none_filter_returns_every_exercise_unchanged(self):
        result = main._apply_exercise_filter(self._exercises(), None, chapter=7)
        self.assertEqual(result, self._exercises())

    def test_empty_list_filter_returns_every_exercise_unchanged(self):
        result = main._apply_exercise_filter(self._exercises(), [], chapter=7)
        self.assertEqual(result, self._exercises())

    def test_single_label_narrows_to_just_that_exercise(self):
        result = main._apply_exercise_filter(self._exercises(), ["7.4"], chapter=7)
        self.assertEqual(result, [{"label": "7.4"}])

    def test_multiple_labels_narrows_to_just_those(self):
        result = main._apply_exercise_filter(self._exercises(), ["7.1", "7.4"], chapter=7)
        self.assertEqual(result, [{"label": "7.1"}, {"label": "7.4"}])

    def test_unknown_label_warns_and_returns_empty_for_that_entry(self):
        with mock.patch("main.logger") as mock_logger:
            result = main._apply_exercise_filter(self._exercises(), ["7.99"], chapter=7)
        self.assertEqual(result, [])
        mock_logger.warning.assert_called_once()
        self.assertIn("7.99", mock_logger.warning.call_args[0][0])

    def test_mix_of_known_and_unknown_labels_returns_only_known_and_warns(self):
        with mock.patch("main.logger") as mock_logger:
            result = main._apply_exercise_filter(self._exercises(), ["7.4", "7.99"], chapter=7)
        self.assertEqual(result, [{"label": "7.4"}])
        mock_logger.warning.assert_called_once()

    def test_label_type_coercion_int_vs_str(self):
        # split_exercises() labels are strings ("7.4"); a config author
        # typing a bare number should still work.
        result = main._apply_exercise_filter(self._exercises(), [7.4], chapter=7)
        self.assertEqual(result, [{"label": "7.4"}])


class TestSolveSingleExerciseToggle(unittest.TestCase):
    """Regression tests for the explicit config.SOLVE_SINGLE_EXERCISE
    True/False switch — EXERCISE_FILTER is only ever consulted when
    this is True; when it's False (the default), the whole chapter is
    solved regardless of whatever EXERCISE_FILTER happens to be set to."""

    class _FakeConfig:
        pass

    def test_false_ignores_exercise_filter_entirely(self):
        cfg = self._FakeConfig()
        cfg.SOLVE_SINGLE_EXERCISE = False
        cfg.EXERCISE_FILTER = ["7.4"]
        self.assertIsNone(main._effective_exercise_filter(cfg))

    def test_true_uses_exercise_filter(self):
        cfg = self._FakeConfig()
        cfg.SOLVE_SINGLE_EXERCISE = True
        cfg.EXERCISE_FILTER = ["7.4"]
        self.assertEqual(main._effective_exercise_filter(cfg), ["7.4"])

    def test_missing_attributes_default_to_whole_chapter(self):
        # A config.py that predates this feature (neither attribute set)
        # must behave exactly like SOLVE_SINGLE_EXERCISE = False.
        cfg = self._FakeConfig()
        self.assertIsNone(main._effective_exercise_filter(cfg))

    def test_true_with_no_exercise_filter_set_yields_none(self):
        cfg = self._FakeConfig()
        cfg.SOLVE_SINGLE_EXERCISE = True
        self.assertIsNone(main._effective_exercise_filter(cfg))


if __name__ == "__main__":
    unittest.main()
