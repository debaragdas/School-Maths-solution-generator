"""
Regression tests for correction_engine.py — Human Review workflow
capability 1: "Wrong AI Output Correction".

`fitz` and `google.genai` are stubbed exactly as test_solver.py /
test_html_renderer.py already do (neither is installed/reachable in
this sandbox, and solver.py needs both at import time). The actual
Gemini call (_call_gemini_correction) and the actual HTML/PDF
rendering (render_exercise_html / render_pdf) are mocked — these tests
exercise the CORRECTION/MERGE logic, not the network call or a real
Playwright render.

Run: python3 -m pytest test_correction_engine.py -v
"""
import json
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


def _fake_response(text: str):
    resp = mock.Mock()
    resp.text = text
    return resp


class TestRegenerateQuestion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmpdir, "output", "class_9", "chapter_7", "ex_7_4.pdf")
        solved = {
            "exercise_label": "7.4",
            "questions": [
                {"question_number": "1", "sub_part": None, "question_text": "Q1 text",
                 "given": "g1", "required": "r1", "steps": ["s1", "s2", "s3"],
                 "final_answer": "old wrong answer", "construction_instruments": None,
                 "diagram_spec": None, "book_diagram_base64": None, "book_diagram_mime": None,
                 "book_diagram_figure_ref": None, "answer_kind": "calculation"},
                {"question_number": "2", "sub_part": None, "question_text": "Q2 text",
                 "given": "g2", "required": "r2", "steps": ["only step"],
                 "final_answer": "untouched answer", "construction_instruments": None,
                 "diagram_spec": None, "book_diagram_base64": None, "book_diagram_mime": None,
                 "book_diagram_figure_ref": None, "answer_kind": "calculation"},
            ],
        }
        review_state.init_state(self.pdf_path, solved, class_name=9, chapter=7,
                                 chapter_name="ত্ৰিভুজ", exercise_label="7.4")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_correction(self, corrected_json_text, question_number="1", sub_part=None, reviewer=None):
        def fake_render_pdf(html, output_path):
            with open(output_path, "wb") as f:
                f.write(b"%PDF fake rendered content" + b"Z" * 2000)

        with mock.patch.object(correction_engine, "_call_gemini_correction",
                                return_value=_fake_response(corrected_json_text)), \
             mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf", side_effect=fake_render_pdf) as render_pdf_mock:
            new_state = correction_engine.regenerate_question(
                self.pdf_path, question_number=question_number,
                correction_instruction="Step 3 calculation is incorrect. Recalculate from Step 2 only.",
                sub_part=sub_part, reviewer=reviewer,
            )
            return new_state, render_pdf_mock

    def test_no_review_state_raises(self):
        other_path = os.path.join(self.tmpdir, "output", "class_9", "chapter_8", "ex_8_1.pdf")
        with self.assertRaises(ValueError):
            correction_engine.regenerate_question(other_path, "1", "fix it")

    def test_question_not_found_raises(self):
        with self.assertRaises(ValueError):
            self._run_correction(json.dumps({"final_answer": "x"}), question_number="99")

    def test_happy_path_updates_only_targeted_question(self):
        corrected = {
            "question_number": "1", "sub_part": None, "question_text": "Q1 text",
            "given": "g1", "required": "r1", "steps": ["s1", "s2", "s3 corrected"],
            "final_answer": "corrected final answer", "construction_instruments": None,
            "diagram_spec": None,
        }
        new_state, render_pdf_mock = self._run_correction(json.dumps(corrected))

        q1 = new_state["solved"]["questions"][0]
        q2 = new_state["solved"]["questions"][1]
        self.assertEqual(q1["final_answer"], "corrected final answer")
        self.assertEqual(q2["final_answer"], "untouched answer", "other questions must be byte-for-byte untouched")
        self.assertEqual(new_state["version"], 2)
        self.assertEqual(new_state["status"], review_state.STATUS_PENDING)
        render_pdf_mock.assert_called_once()
        # the underlying render call targets a TEMP path (atomic-write
        # pattern — see correction_engine._atomic_render_pdf), which is
        # then swapped onto the real path only after a full, valid render.
        self.assertEqual(render_pdf_mock.call_args.args[1], self.pdf_path + ".correcting.tmp")
        with open(self.pdf_path, "rb") as f:
            self.assertTrue(f.read().startswith(b"%PDF fake rendered content"))
        self.assertFalse(os.path.exists(self.pdf_path + ".correcting.tmp"))

    def test_component_hint_is_prefixed_into_the_prompt_but_never_changes_defaults(self):
        # component=None (the default, used by every pre-existing caller
        # including interactive_review.py) must send the EXACT instruction
        # text unprefixed — this is the backward-compatibility contract
        # for every test above that doesn't pass component at all.
        corrected = {
            "question_number": "1", "sub_part": None, "question_text": "Q1 text",
            "given": "g1", "required": "r1", "steps": ["fixed"],
            "final_answer": "fixed", "construction_instruments": None, "diagram_spec": None,
        }
        with mock.patch.object(correction_engine, "_call_gemini_correction",
                                return_value=_fake_response(json.dumps(corrected))) as call_mock, \
             mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf",
                                side_effect=lambda html, path: open(path, "wb").write(b"%PDF" + b"Z" * 2000)):
            correction_engine.regenerate_question(
                self.pdf_path, question_number="1",
                correction_instruction="Step 3 is wrong", sub_part=None,
            )
            prompt_sent = call_mock.call_args.args[0]
            self.assertIn("Step 3 is wrong", prompt_sent)
            self.assertNotIn("SCOPE:", prompt_sent)

            call_mock.reset_mock()
            correction_engine.regenerate_question(
                self.pdf_path, question_number="1",
                correction_instruction="Diagram labels wrong", sub_part=None, component="diagram",
            )
            prompt_sent = call_mock.call_args.args[0]
            self.assertIn("Diagram labels wrong", prompt_sent)
            self.assertIn("only the diagram", prompt_sent)

            call_mock.reset_mock()
            correction_engine.regenerate_question(
                self.pdf_path, question_number="1",
                correction_instruction="Altitude missing", sub_part=None, component="solution",
            )
            prompt_sent = call_mock.call_args.args[0]
            self.assertIn("Altitude missing", prompt_sent)
            self.assertIn("only the worked solution", prompt_sent)


        corrected = {
            "question_number": "1", "sub_part": None, "question_text": "Q1 text",
            "given": "g1", "required": "r1", "steps": ["s1", "s2", "fixed"],
            "final_answer": "fixed answer", "construction_instruments": None, "diagram_spec": None,
        }
        new_state, _ = self._run_correction(json.dumps(corrected), reviewer="Priya")
        last_entry = new_state["history"][-1]
        self.assertEqual(last_entry["action"], "corrected")
        self.assertEqual(last_entry["question_number"], "1")
        self.assertEqual(last_entry["reviewer"], "Priya")

    def test_previously_approved_exercise_resets_to_pending_after_correction(self):
        review_state.approve(self.pdf_path, reviewer="Priya")
        corrected = {
            "question_number": "1", "sub_part": None, "question_text": "Q1 text",
            "given": "g1", "required": "r1", "steps": ["fixed"],
            "final_answer": "fixed", "construction_instruments": None, "diagram_spec": None,
        }
        new_state, _ = self._run_correction(json.dumps(corrected))
        self.assertEqual(new_state["status"], review_state.STATUS_PENDING,
                          "a correction to an already-approved exercise must force re-review")

    def test_recovers_from_unescaped_backslash_in_latex(self):
        # a single backslash before 's' is genuinely invalid JSON (unlike
        # '\f', which is one of JSON's OWN valid escapes and would just
        # silently parse to a form-feed character instead of raising —
        # see _VALID_JSON_ESCAPE_CHARS in solver.py) — this must trigger
        # the same _repair_invalid_json_escapes pass solver.py's main
        # solve path already relies on.
        broken = (
            '{"question_number": "1", "sub_part": null, "question_text": "Q1 text", '
            '"given": "g1", "required": "r1", "steps": ["\\sqrt{4} fixed"], '
            '"final_answer": "\\sqrt{4} = 2", "construction_instruments": null, "diagram_spec": null}'
        )
        new_state, _ = self._run_correction(broken)
        self.assertIn("sqrt", new_state["solved"]["questions"][0]["final_answer"])

    def test_gemini_identity_fields_cannot_override_target_question(self):
        # even if the model's response tries to change question_number,
        # the engine must pin it back to the one actually being corrected.
        corrected = {
            "question_number": "999", "sub_part": None, "question_text": "Q1 text",
            "given": "g1", "required": "r1", "steps": ["fixed"],
            "final_answer": "fixed", "construction_instruments": None, "diagram_spec": None,
        }
        new_state, _ = self._run_correction(json.dumps(corrected), question_number="1")
        self.assertEqual(new_state["solved"]["questions"][0]["question_number"], "1")
        self.assertEqual(len(new_state["solved"]["questions"]), 2, "must not add a duplicate question")

    def test_invalid_json_even_after_repair_raises(self):
        with self.assertRaises(ValueError):
            self._run_correction("this is not JSON at all {{{")

    def test_empty_response_raises(self):
        with self.assertRaises(ValueError):
            self._run_correction("")


    def test_failed_rerender_never_destroys_the_previous_good_pdf(self):
        """PRODUCTION-AUDIT REGRESSION TEST: reproduces a Playwright-style
        crash mid-render (writes a truncated file at the final path, then
        raises) and proves the exercise's PREVIOUS, already-approved PDF
        survives byte-for-byte untouched — see correction_engine._atomic_render_pdf."""
        good_bytes = b"%PDF-1.4 REAL APPROVED CONTENT" + b"X" * 2000
        with open(self.pdf_path, "wb") as f:
            f.write(good_bytes)
        review_state.approve(self.pdf_path, reviewer="Priya")

        def crashing_render_pdf(html, output_path):
            with open(output_path, "wb") as f:
                f.write(b"TRUNCATED")
            raise RuntimeError("simulated browser crash during PDF generation")

        corrected = {
            "question_number": "1", "sub_part": None, "question_text": "Q1 text",
            "given": "g1", "required": "r1", "steps": ["fixed"],
            "final_answer": "fixed", "construction_instruments": None, "diagram_spec": None,
        }
        with mock.patch.object(correction_engine, "_call_gemini_correction",
                                return_value=_fake_response(json.dumps(corrected))), \
             mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf", side_effect=crashing_render_pdf):
            with self.assertRaises(RuntimeError):
                correction_engine.regenerate_question(self.pdf_path, "1", "fix it")

        with open(self.pdf_path, "rb") as f:
            surviving = f.read()
        self.assertEqual(surviving, good_bytes, "a failed re-render must never touch the previous good PDF")
        # And no leftover temp file should be left behind either.
        self.assertFalse(os.path.exists(self.pdf_path + ".correcting.tmp"))

        # Review state must also stay consistent — still approved, since
        # the correction never actually completed.
        state = review_state.load_state(self.pdf_path)
        self.assertEqual(state["status"], review_state.STATUS_APPROVED)
        self.assertEqual(state["version"], 1)

    def test_successful_rerender_replaces_the_pdf_atomically(self):
        with open(self.pdf_path, "wb") as f:
            f.write(b"OLD CONTENT")
        corrected = {
            "question_number": "1", "sub_part": None, "question_text": "Q1 text",
            "given": "g1", "required": "r1", "steps": ["fixed"],
            "final_answer": "fixed", "construction_instruments": None, "diagram_spec": None,
        }

        def real_write_render_pdf(html, output_path):
            with open(output_path, "wb") as f:
                f.write(b"NEW REAL PDF CONTENT" + b"Y" * 2000)

        with mock.patch.object(correction_engine, "_call_gemini_correction",
                                return_value=_fake_response(json.dumps(corrected))), \
             mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf", side_effect=real_write_render_pdf):
            correction_engine.regenerate_question(self.pdf_path, "1", "fix it")

        with open(self.pdf_path, "rb") as f:
            content = f.read()
        self.assertTrue(content.startswith(b"NEW REAL PDF CONTENT"))
        self.assertFalse(os.path.exists(self.pdf_path + ".correcting.tmp"))


    def test_concurrent_corrections_to_different_questions_do_not_lose_either_fix(self):
        """PRODUCTION-AUDIT REGRESSION TEST: reproduces two reviewers
        correcting DIFFERENT questions of the SAME exercise at close to
        the same time. Before the fix, whichever correction's (slow,
        network-bound) Gemini call finished LATER would merge into a
        stale full copy of `solved` loaded before the OTHER correction
        saved, and its save would silently revert that other fix. See
        correction_engine.regenerate_question's locked fresh-reload
        section and review_state.state_lock()."""
        import threading
        import time as time_module

        def fake_render_pdf(html, output_path):
            with open(output_path, "wb") as f:
                f.write(b"%PDF" + b"X" * 2000)

        # ONE shared dispatcher (patched once — mock.patch is not safe to
        # apply concurrently from multiple threads against the same
        # attribute) that answers based on which question is being
        # corrected, with an artificial delay to widen the race window,
        # matching the technique this project's own concurrency tests use
        # (see figure_database.py's test_concurrent_ingestion... test).
        def shared_call(prompt):
            if '"question_number": "1"' in prompt:
                time_module.sleep(0.15)
                qnum, ans = "1", "FIXED1"
            else:
                time_module.sleep(0.05)
                qnum, ans = "2", "FIXED2"
            return _fake_response(json.dumps({
                "question_number": qnum, "sub_part": None, "question_text": f"q{qnum}",
                "given": "g", "required": "r", "steps": ["fixed"], "final_answer": ans,
                "construction_instruments": None, "diagram_spec": None,
            }))

        results = {}
        with mock.patch.object(correction_engine, "_call_gemini_correction", side_effect=shared_call), \
             mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf", side_effect=fake_render_pdf):
            t1 = threading.Thread(target=lambda: results.update(
                q1=correction_engine.regenerate_question(self.pdf_path, "1", "fix q1")))
            t2 = threading.Thread(target=lambda: results.update(
                q2=correction_engine.regenerate_question(self.pdf_path, "2", "fix q2")))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        final = review_state.load_state(self.pdf_path)
        answers = {q["question_number"]: q["final_answer"] for q in final["solved"]["questions"]}
        self.assertEqual(answers.get("1"), "FIXED1", "Question 1's correction must not be lost")
        self.assertEqual(answers.get("2"), "FIXED2", "Question 2's correction must not be lost")
        self.assertEqual(final["version"], 3, "both corrections must be recorded as distinct versions")


class TestDiagramActions(unittest.TestCase):
    """Regression tests for correction_engine.py's diagram-only actions
    — capability 3 (rebuild/generate from scratch), 6 (remove generated
    diagram), 7 (remove book figure)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmpdir, "output", "class_9", "chapter_7", "ex_7_4.pdf")
        self.solved = {
            "exercise_label": "7.4",
            "questions": [
                {"question_number": "1", "sub_part": None,
                 "question_text": "ABC is a triangle with AB=5cm, BC=6cm, AC=7cm. In the figure, find angle B.",
                 "given": "AB=5, BC=6, AC=7", "required": "angle B", "steps": ["s1"],
                 "final_answer": "ans", "construction_instruments": None,
                 "diagram_spec": {"diagram_type": "triangle", "points": [
                     {"id": "A", "x": 0, "y": 0}, {"id": "B", "x": 1, "y": 0}, {"id": "C", "x": 0.5, "y": 1}]},
                 "book_diagram_base64": None, "book_diagram_mime": None,
                 "book_diagram_figure_ref": None, "answer_kind": "calculation",
                 "diagram_decision": "GENERATED_DIAGRAM", "needs_review": False, "review_notes": []},
                # No diagram AND no book figure yet — the "Generate Diagram" case.
                {"question_number": "2", "sub_part": None,
                 "question_text": "Prove that the diagonals of a parallelogram PQRS bisect each other.",
                 "given": "PQRS parallelogram", "required": "prove diagonals bisect", "steps": ["s1"],
                 "final_answer": "proved", "construction_instruments": None, "diagram_spec": None,
                 "book_diagram_base64": None, "book_diagram_mime": None,
                 "book_diagram_figure_ref": None, "answer_kind": "proof",
                 "diagram_decision": "NO_DIAGRAM", "diagram_decision_reason": "no diagram_spec — question doesn't need one",
                 "needs_review": False, "review_notes": []},
                # Has a (possibly wrong) book figure — the "Remove Book Figure" case.
                {"question_number": "3", "sub_part": None,
                 "question_text": "See Figure 7.18 and find the radius.", "given": "g3", "required": "r3",
                 "steps": ["s1"], "final_answer": "ans3", "construction_instruments": None,
                 "diagram_spec": None,
                 "book_diagram_base64": __import__("base64").b64encode(b"fakepng").decode(),
                 "book_diagram_mime": "image/png", "book_diagram_figure_ref": "7.18",
                 "answer_kind": "calculation", "diagram_decision": "BOOK_DIAGRAM", "needs_review": False,
                 "review_notes": []},
            ],
        }
        review_state.init_state(self.pdf_path, self.solved, class_name=9, chapter=7,
                                 chapter_name="ত্ৰিভুজ", exercise_label="7.4")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_with_fakes(self, fn, *args, gemini_result=None, gemini_raises=None, **kwargs):
        def fake_render_pdf(html, output_path):
            with open(output_path, "wb") as f:
                f.write(b"%PDF fake rendered content" + b"Z" * 2000)

        recovery_mock = mock.Mock()
        if gemini_raises is not None:
            recovery_mock.side_effect = gemini_raises
        else:
            recovery_mock.return_value = gemini_result

        with mock.patch("correction_engine.diagram_safety_net.generate_diagram_from_scratch", recovery_mock), \
             mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf", side_effect=fake_render_pdf):
            return fn(*args, **kwargs), recovery_mock

    def test_regenerate_diagram_rebuilds_a_valid_triangle_from_scratch(self):
        new_spec = {"diagram_type": "triangle",
                    "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        new_state, recovery_mock = self._run_with_fakes(
            correction_engine.regenerate_diagram_from_scratch, self.pdf_path, "1",
            gemini_result={"needs_diagram": True, "diagram_spec": new_spec},
            instruction="Right angle at B, not isosceles", reviewer="tester",
        )
        recovery_mock.assert_called_once()
        args, kwargs = recovery_mock.call_args
        self.assertEqual(kwargs.get("extra_instruction") or (args[2] if len(args) > 2 else None),
                          "Right angle at B, not isosceles")
        q1 = new_state["solved"]["questions"][0]
        self.assertIsNotNone(q1["diagram_spec"])
        self.assertFalse(q1["needs_review"])
        self.assertEqual(new_state["version"], 2)
        self.assertEqual(new_state["history"][-1]["action"], "diagram_rebuilt")

    def test_regenerate_diagram_discards_stale_construction_instruments(self):
        # Reproduces the exact bug this round fixes: Q1 previously had a
        # (wrong) diagram_type="construction" diagram, so construction_
        # instruments is still set on disk. A rebuild that correctly
        # produces an ordinary "triangle" spec this time must NOT be
        # rejected by decide_diagram's Stage 4 consistency check just
        # because the OLD construction_instruments is still lying around.
        state = review_state.load_state(self.pdf_path)
        state["solved"]["questions"][0]["construction_instruments"] = ["Draw AB = 5cm with a ruler."]
        review_state.save_state(self.pdf_path, state)

        new_spec = {"diagram_type": "triangle",
                    "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]}
        new_state, _ = self._run_with_fakes(
            correction_engine.regenerate_diagram_from_scratch, self.pdf_path, "1",
            gemini_result={"needs_diagram": True, "diagram_spec": new_spec,
                            "construction_instruments": None},
        )
        q1 = new_state["solved"]["questions"][0]
        self.assertIsNotNone(q1["diagram_spec"], "must not be rejected due to stale construction_instruments")
        self.assertEqual(q1["diagram_decision"], "GENERATED_DIAGRAM")
        self.assertIsNone(q1["construction_instruments"], "the old value must be discarded, not just ignored")

    def test_regenerate_diagram_accepts_a_matching_construction_rebuild(self):
        new_spec = {
            "diagram_type": "construction",
            "initial_points": [{"id": "A", "x": 0, "y": 0}, {"id": "B", "x": 5, "y": 0}],
            "construction_steps": [{"type": "line_segment", "from": "A", "to": "B", "label": "5 cm"}],
        }
        instructions = ["Draw segment AB = 5cm.", "Draw an arc from A."]
        new_state, _ = self._run_with_fakes(
            correction_engine.regenerate_diagram_from_scratch, self.pdf_path, "1",
            gemini_result={"needs_diagram": True, "diagram_spec": new_spec,
                            "construction_instruments": instructions},
        )
        q1 = new_state["solved"]["questions"][0]
        self.assertIsNotNone(q1["diagram_spec"])
        self.assertEqual(q1["construction_instruments"], instructions)

    def test_regenerate_diagram_never_leaves_an_invalid_spec(self):
        # An unregistered diagram_type must fail decide_diagram's own
        # Stage 3 check and result in NO diagram, not a broken one.
        bad_spec = {"diagram_type": "not_a_real_type", "points": []}
        new_state, _ = self._run_with_fakes(
            correction_engine.regenerate_diagram_from_scratch, self.pdf_path, "1",
            gemini_result={"needs_diagram": True, "diagram_spec": bad_spec},
        )
        q1 = new_state["solved"]["questions"][0]
        self.assertIsNone(q1["diagram_spec"])
        self.assertTrue(q1["needs_review"])
        self.assertTrue(any("failed validation" in n for n in q1["review_notes"]))

    def test_regenerate_diagram_call_failure_leaves_no_diagram_and_flags_review(self):
        new_state, _ = self._run_with_fakes(
            correction_engine.regenerate_diagram_from_scratch, self.pdf_path, "1",
            gemini_raises=RuntimeError("Gemini timed out"),
        )
        q1 = new_state["solved"]["questions"][0]
        self.assertIsNone(q1["diagram_spec"])
        self.assertTrue(q1["needs_review"])
        self.assertTrue(any("Gemini timed out" in n for n in q1["review_notes"]))

    def test_generate_diagram_works_for_a_question_with_none_at_all(self):
        new_spec = {"diagram_type": "quadrilateral",
                    "points": [{"id": "P"}, {"id": "Q"}, {"id": "R"}, {"id": "S"}]}
        new_state, _ = self._run_with_fakes(
            correction_engine.regenerate_diagram_from_scratch, self.pdf_path, "2",
            gemini_result={"needs_diagram": True, "diagram_spec": new_spec},
        )
        q2 = new_state["solved"]["questions"][1]
        self.assertIsNotNone(q2["diagram_spec"])
        self.assertEqual(q2["diagram_decision"], "GENERATED_DIAGRAM")

    def test_remove_generated_diagram_clears_spec_but_not_solution(self):
        def fake_render_pdf(html, output_path):
            with open(output_path, "wb") as f:
                f.write(b"%PDF fake rendered content" + b"Z" * 2000)
        with mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf", side_effect=fake_render_pdf):
            new_state = correction_engine.remove_generated_diagram(self.pdf_path, "1", reviewer="tester")
        q1 = new_state["solved"]["questions"][0]
        self.assertIsNone(q1["diagram_spec"])
        self.assertEqual(q1["diagram_decision"], "NO_DIAGRAM")
        self.assertEqual(q1["final_answer"], "ans", "the worked solution must be untouched")
        self.assertEqual(new_state["history"][-1]["action"], "diagram_removed")

    def test_remove_book_figure_clears_figure_but_not_solution(self):
        def fake_render_pdf(html, output_path):
            with open(output_path, "wb") as f:
                f.write(b"%PDF fake rendered content" + b"Z" * 2000)
        with mock.patch.object(correction_engine, "render_exercise_html", return_value="<html></html>"), \
             mock.patch.object(correction_engine, "render_pdf", side_effect=fake_render_pdf):
            new_state = correction_engine.remove_book_figure(self.pdf_path, "3", reviewer="tester")
        q3 = new_state["solved"]["questions"][2]
        self.assertIsNone(q3["book_diagram_base64"])
        self.assertIsNone(q3["book_diagram_figure_ref"])
        self.assertFalse(q3.get("has_book_diagram"))
        self.assertEqual(q3["final_answer"], "ans3", "the worked solution must be untouched")
        self.assertEqual(new_state["history"][-1]["action"], "figure_removed")

    def test_diagram_action_on_unknown_question_raises(self):
        with self.assertRaises(ValueError):
            correction_engine.remove_generated_diagram(self.pdf_path, "99")
        with self.assertRaises(ValueError):
            correction_engine.remove_book_figure(self.pdf_path, "99")


if __name__ == "__main__":
    unittest.main()
