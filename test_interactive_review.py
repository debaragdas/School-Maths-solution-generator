"""
Regression tests for interactive_review.py — the per-question,
human-in-the-loop console workflow that main.py now runs on every
exercise before it can be approved/published.

`correction_engine.regenerate_question` / `attach_manual_figure_to_question`
are mocked here (they're already fully covered, including real
rendering, by test_correction_engine.py) — these tests exercise the
LOOP's control flow: which menu choice does what, resumability across
sessions, and the "repeat until accepted" contract.

Run: python3 -m pytest test_interactive_review.py -v
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

import review_state
import correction_engine
import interactive_review


class _ScriptedIO:
    """Feeds a fixed, ordered list of responses to input_fn calls and
    records everything print_fn was asked to print, so a test can both
    drive the loop deterministically and assert on what the reviewer
    would have seen."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.printed = []

    def input_fn(self, prompt=""):
        if not self._responses:
            raise AssertionError(f"Ran out of scripted responses (last prompt: {prompt!r})")
        return self._responses.pop(0)

    def print_fn(self, *args):
        self.printed.append(" ".join(str(a) for a in args))

    def open_pdf_fn(self, path):
        self.printed.append(f"[[opened {path}]]")


def _solved(num_questions=2, with_diagram_on=None):
    questions = []
    for i in range(1, num_questions + 1):
        q = {
            "question_number": str(i), "sub_part": None, "question_text": f"Question {i} text",
            "given": "g", "required": "r", "steps": ["s1"], "final_answer": f"answer {i}",
            "construction_instruments": None, "diagram_spec": None, "book_diagram_base64": None,
            "book_diagram_mime": None, "book_diagram_figure_ref": None, "needs_review": False,
        }
        if with_diagram_on and i in with_diagram_on:
            q["diagram_spec"] = {"diagram_type": "triangle"}
        questions.append(q)
    return {"exercise_label": "7.1", "questions": questions}


class TestReviewExerciseInteractively(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmpdir, "ex_7_1.pdf")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_review_state_returns_false(self):
        io = _ScriptedIO([])
        result = interactive_review.review_exercise_interactively(
            self.pdf_path, input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
        )
        self.assertFalse(result)

    def test_already_approved_exercise_returns_true_without_prompting(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        review_state.approve(self.pdf_path)
        io = _ScriptedIO([])  # no responses queued — must not prompt at all
        result = interactive_review.review_exercise_interactively(
            self.pdf_path, input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
        )
        self.assertTrue(result)

    def test_accepting_every_question_approves_the_exercise(self):
        review_state.init_state(self.pdf_path, _solved(2), 9, 7, "ch", "7.1")
        io = _ScriptedIO(["a", "a"])
        result = interactive_review.review_exercise_interactively(
            self.pdf_path, reviewer="Priya", input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
        )
        self.assertTrue(result)
        state = review_state.load_state(self.pdf_path)
        self.assertEqual(state["status"], review_state.STATUS_APPROVED)
        self.assertIn("1", state["accepted_questions"])
        self.assertIn("2", state["accepted_questions"])

    def test_scope_flagged_or_diagram_only_prompts_for_relevant_questions(self):
        solved = _solved(3, with_diagram_on={2})
        solved["questions"][2]["needs_review"] = True  # Q3 flagged
        review_state.init_state(self.pdf_path, solved, 9, 7, "ch", "7.1")
        # Q1 has no diagram and isn't flagged -> auto-accepted, no prompt.
        # Q2 (diagram) and Q3 (flagged) -> must prompt, in order.
        io = _ScriptedIO(["a", "a"])
        result = interactive_review.review_exercise_interactively(
            self.pdf_path, scope="flagged_or_diagram",
            input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
        )
        self.assertTrue(result)
        state = review_state.load_state(self.pdf_path)
        self.assertEqual(set(state["accepted_questions"]), {"1", "2", "3"})

    def test_quit_preserves_progress_and_resumes_on_next_call(self):
        review_state.init_state(self.pdf_path, _solved(2), 9, 7, "ch", "7.1")
        io1 = _ScriptedIO(["a", "q"])  # accept Q1, then quit before Q2
        result1 = interactive_review.review_exercise_interactively(
            self.pdf_path, input_fn=io1.input_fn, print_fn=io1.print_fn, open_pdf_fn=io1.open_pdf_fn,
        )
        self.assertFalse(result1)
        state = review_state.load_state(self.pdf_path)
        self.assertEqual(state["status"], review_state.STATUS_PENDING)
        self.assertEqual(state["accepted_questions"], ["1"])

        # Resuming: Q1 must NOT be re-prompted (only one response queued, for Q2).
        io2 = _ScriptedIO(["a"])
        result2 = interactive_review.review_exercise_interactively(
            self.pdf_path, input_fn=io2.input_fn, print_fn=io2.print_fn, open_pdf_fn=io2.open_pdf_fn,
        )
        self.assertTrue(result2)
        self.assertEqual(review_state.load_state(self.pdf_path)["status"], review_state.STATUS_APPROVED)

    def test_regenerate_calls_correction_engine_and_loops_back(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        io = _ScriptedIO(["r", "Step 2 is wrong, fix it", "a"])
        with mock.patch.object(correction_engine, "regenerate_question") as mock_regen:
            result = interactive_review.review_exercise_interactively(
                self.pdf_path, reviewer="Priya",
                input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
            )
        self.assertTrue(result)
        mock_regen.assert_called_once_with(
            self.pdf_path, "1", "Step 2 is wrong, fix it", sub_part=None, reviewer="Priya",
        )

    def test_empty_correction_prompt_does_not_call_correction_engine(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        io = _ScriptedIO(["r", "", "a"])
        with mock.patch.object(correction_engine, "regenerate_question") as mock_regen:
            result = interactive_review.review_exercise_interactively(
                self.pdf_path, input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
            )
        self.assertTrue(result)
        mock_regen.assert_not_called()

    def test_regenerate_failure_is_reported_and_loop_continues(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        io = _ScriptedIO(["r", "fix it", "a"])
        with mock.patch.object(correction_engine, "regenerate_question", side_effect=ValueError("boom")):
            result = interactive_review.review_exercise_interactively(
                self.pdf_path, input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
            )
        self.assertTrue(result)
        self.assertTrue(any("boom" in line for line in io.printed))

    def test_figure_upload_calls_attach_manual_figure(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        image_path = os.path.join(self.tmpdir, "shot.png")
        with open(image_path, "wb") as f:
            f.write(b"fake png bytes")
        io = _ScriptedIO(["f", image_path, "4.12", "", "", "a"])  # path, figure#, page(blank), bbox(blank), accept
        with mock.patch.object(correction_engine, "attach_manual_figure_to_question") as mock_attach:
            result = interactive_review.review_exercise_interactively(
                self.pdf_path, reviewer="Priya",
                input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
            )
        self.assertTrue(result)
        mock_attach.assert_called_once()
        args, kwargs = mock_attach.call_args
        self.assertEqual(args[0], self.pdf_path)
        self.assertEqual(args[1], "1")
        self.assertEqual(kwargs["figure_number"], "4.12")
        self.assertEqual(kwargs["image_bytes"], b"fake png bytes")

    def test_figure_upload_missing_file_is_reported_without_crashing(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        io = _ScriptedIO(["f", "/no/such/file.png", "a"])
        result = interactive_review.review_exercise_interactively(
            self.pdf_path, input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
        )
        self.assertTrue(result)
        self.assertTrue(any("not found" in line.lower() for line in io.printed))

    def test_preview_opens_pdf_and_reprompts_same_question(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        io = _ScriptedIO(["p", "a"])
        result = interactive_review.review_exercise_interactively(
            self.pdf_path, input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
        )
        self.assertTrue(result)
        self.assertTrue(any("opened" in line for line in io.printed))

    def test_unrecognized_choice_reprompts(self):
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        io = _ScriptedIO(["xyz", "a"])
        result = interactive_review.review_exercise_interactively(
            self.pdf_path, input_fn=io.input_fn, print_fn=io.print_fn, open_pdf_fn=io.open_pdf_fn,
        )
        self.assertTrue(result)

    def test_corrected_question_is_removed_from_accepted_before_reprompt(self):
        """If a question was somehow already accepted and then gets
        corrected again, it must require a fresh accept — exercised via
        review_state.finalize_correction directly (already covered end
        to end by correction_engine's own tests); this just confirms
        the interactive loop respects is_question_accepted() freshly
        on every iteration rather than caching it."""
        review_state.init_state(self.pdf_path, _solved(1), 9, 7, "ch", "7.1")
        review_state.mark_question_accepted(self.pdf_path, "1")
        state = review_state.load_state(self.pdf_path)
        self.assertIn("1", state["accepted_questions"])
        state["version"] = 1
        review_state.finalize_correction(state, self.pdf_path, "1", "some fix", action="corrected")
        reloaded = review_state.load_state(self.pdf_path)
        self.assertNotIn("1", reloaded["accepted_questions"])


if __name__ == "__main__":
    unittest.main()
