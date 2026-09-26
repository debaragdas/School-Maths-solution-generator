"""
Regression tests for review_state.py — persistent per-exercise review
state and the chapter-level publish gate.

Run: python3 -m pytest test_review_state.py -v
"""
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import review_state


class TestReviewStateLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmpdir, "output", "class_9", "chapter_7", "ex_7_4.pdf")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _solved(self):
        return {"exercise_label": "7.4", "questions": [
            {"question_number": "1", "sub_part": None, "final_answer": "answer 1"},
            {"question_number": "2", "sub_part": None, "final_answer": "answer 2"},
        ]}

    def test_init_state_creates_pending_review_at_version_1(self):
        state = review_state.init_state(self.pdf_path, self._solved(), class_name=9, chapter=7,
                                         chapter_name="ত্ৰিভুজ", exercise_label="7.4")
        self.assertEqual(state["status"], review_state.STATUS_PENDING)
        self.assertEqual(state["version"], 1)
        self.assertEqual(len(state["history"]), 1)
        self.assertTrue(os.path.exists(review_state.state_path(self.pdf_path)))

    def test_load_state_returns_none_when_absent(self):
        self.assertIsNone(review_state.load_state(self.pdf_path))

    def test_approve_sets_status_and_appends_history(self):
        review_state.init_state(self.pdf_path, self._solved(), 9, 7, "ত্ৰিভুজ", "7.4")
        state = review_state.approve(self.pdf_path, reviewer="Priya", note="looks good")
        self.assertEqual(state["status"], review_state.STATUS_APPROVED)
        self.assertEqual(state["history"][-1]["action"], "approved")
        self.assertEqual(len(state["reviewer_notes"]), 1)

    def test_reject_sets_status(self):
        review_state.init_state(self.pdf_path, self._solved(), 9, 7, "ত্ৰিভুজ", "7.4")
        state = review_state.reject(self.pdf_path, reviewer="Priya", note="Q3 wrong")
        self.assertEqual(state["status"], review_state.STATUS_REJECTED)

    def test_approve_without_prior_state_raises(self):
        with self.assertRaises(ValueError):
            review_state.approve(self.pdf_path)

    def test_finalize_correction_bumps_version_and_resets_to_pending(self):
        review_state.init_state(self.pdf_path, self._solved(), 9, 7, "ত্ৰিভুজ", "7.4")
        review_state.approve(self.pdf_path, reviewer="Priya")

        with review_state.state_lock(self.pdf_path):
            state = review_state.load_state(self.pdf_path)
            state["solved"]["questions"][0]["final_answer"] = "corrected answer 1"
            new_state = review_state.finalize_correction(
                state, self.pdf_path, question_number="1", correction_instruction="fix answer 1",
                reviewer="Priya",
            )
        self.assertEqual(new_state["version"], 2)
        self.assertEqual(new_state["status"], review_state.STATUS_PENDING,
                          "a correction must require re-review, even if it was previously approved")
        self.assertEqual(new_state["solved"]["questions"][0]["final_answer"], "corrected answer 1")
        # and it must actually be persisted, not just returned
        reloaded = review_state.load_state(self.pdf_path)
        self.assertEqual(reloaded["version"], 2)

    def test_corrupt_state_file_is_recovered_from_not_fatal(self):
        path = review_state.state_path(self.pdf_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("{not valid json")
        self.assertIsNone(review_state.load_state(self.pdf_path))


class TestReviewFolderLayout(unittest.TestCase):
    """Regression tests for the final pre-launch reorg: the .review.json
    sidecar now lives in a same-level Review/ subfolder next to the PDF
    (keeping the PDF folder itself clean), with backward-compat for any
    sidecar already written the old way (directly beside the PDF)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmpdir, "output", "class_9", "chapter_7", "ex_7_4.pdf")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _solved(self):
        return {"exercise_label": "7.4", "questions": [
            {"question_number": "1", "sub_part": None, "final_answer": "answer 1"},
        ]}

    def test_fresh_exercise_writes_sidecar_under_review_subfolder(self):
        review_state.init_state(self.pdf_path, self._solved(), 9, 7, "ত্ৰিভুজ", "7.4")
        expected = os.path.join(self.tmpdir, "output", "class_9", "chapter_7", "Review", "ex_7_4.review.json")
        self.assertEqual(review_state.state_path(self.pdf_path), expected)
        self.assertTrue(os.path.exists(expected))
        # the PDF's own directory holds ONLY what main.py itself puts there —
        # this module never writes anything directly beside the PDF anymore.
        pdf_dir = os.path.dirname(self.pdf_path)
        self.assertNotIn("ex_7_4.review.json", os.listdir(pdf_dir) if os.path.isdir(pdf_dir) else [])

    def test_legacy_beside_pdf_sidecar_is_still_read_and_written_in_place(self):
        legacy_path = os.path.splitext(self.pdf_path)[0] + review_state.STATE_SUFFIX
        os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
        import json
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump({"exercise_label": "7.4", "class_name": 9, "chapter": 7, "chapter_name": "ত্ৰিভুজ",
                       "status": review_state.STATUS_PENDING, "version": 1,
                       "solved": self._solved(), "history": [], "reviewer_notes": [],
                       "accepted_questions": []}, f)

        self.assertEqual(review_state.state_path(self.pdf_path), legacy_path)
        state = review_state.load_state(self.pdf_path)
        self.assertIsNotNone(state)

        review_state.approve(self.pdf_path, reviewer="tester")
        self.assertTrue(os.path.exists(legacy_path))
        review_subfolder = os.path.join(os.path.dirname(self.pdf_path), "Review")
        self.assertFalse(os.path.isdir(review_subfolder), "an existing legacy sidecar must not also get a new-layout copy")

    def test_list_states_resolves_pdf_path_correctly_under_review_subfolder(self):
        review_state.init_state(self.pdf_path, self._solved(), 9, 7, "ত্ৰিভুজ", "7.4")
        output_root = os.path.join(self.tmpdir, "output")
        states = review_state.list_states(output_root)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["pdf_path"], self.pdf_path)

    def test_list_states_still_resolves_legacy_layout(self):
        legacy_path = os.path.splitext(self.pdf_path)[0] + review_state.STATE_SUFFIX
        os.makedirs(os.path.dirname(legacy_path), exist_ok=True)
        import json
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump({"exercise_label": "7.4", "class_name": 9, "chapter": 7, "chapter_name": "ত্ৰিভুজ",
                       "status": review_state.STATUS_PENDING, "version": 1,
                       "solved": self._solved(), "history": [], "reviewer_notes": [],
                       "accepted_questions": []}, f)
        output_root = os.path.join(self.tmpdir, "output")
        states = review_state.list_states(output_root)
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["pdf_path"], self.pdf_path)

    def test_publish_chapter_still_finds_the_pdf_with_new_layout(self):
        review_state.init_state(self.pdf_path, self._solved(), 9, 7, "ত্ৰিভুজ", "7.4")
        os.makedirs(os.path.dirname(self.pdf_path), exist_ok=True)
        with open(self.pdf_path, "wb") as f:
            f.write(b"%PDF fake" + b"Z" * 1500)
        review_state.approve(self.pdf_path)
        output_root = os.path.join(self.tmpdir, "output")
        published_root = os.path.join(self.tmpdir, "published")
        result = review_state.publish_chapter(9, 7, output_root, published_root)
        self.assertEqual(len(result["published_files"]), 1)
        self.assertTrue(os.path.exists(result["published_files"][0]))


class TestConcurrentReviewActions(unittest.TestCase):
    """PRODUCTION-AUDIT REGRESSION TESTS: reproduces the exact
    last-writer-wins race two internal reviewers can trigger acting on
    the same exercise at close to the same moment — see
    review_state.state_lock()'s docstring."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmpdir, "ex_7_4.pdf")
        review_state.init_state(self.pdf_path, {"exercise_label": "7.4", "questions": []},
                                 9, 7, "ত্ৰিভুজ", "7.4")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_approve_and_reject_never_loses_either_action(self):
        import threading

        orig_load = review_state.load_state

        def slow_load(*a, **kw):
            state = orig_load(*a, **kw)
            time.sleep(0.05)  # widen the race window, same technique figure_database's own test uses
            return state

        with mock.patch.object(review_state, "load_state", side_effect=slow_load):
            t1 = threading.Thread(target=review_state.approve, args=(self.pdf_path,), kwargs={"reviewer": "A"})
            t2 = threading.Thread(target=review_state.reject, args=(self.pdf_path,), kwargs={"reviewer": "B"})
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        final = review_state.load_state(self.pdf_path)
        actions = [h["action"] for h in final["history"]]
        self.assertEqual(actions.count("approved"), 1, "the approve action must not be lost")
        self.assertEqual(actions.count("rejected"), 1, "the reject action must not be lost")


if __name__ == "__main__":
    unittest.main()


class TestChapterPublishGate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_root = os.path.join(self.tmpdir, "output")
        self.published_root = os.path.join(self.tmpdir, "published")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_exercise(self, label, status=None):
        pdf_path = os.path.join(self.output_root, "class_9", "chapter_7", f"ex_{label.replace('.', '_')}.pdf")
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF fake content")
        review_state.init_state(pdf_path, {"exercise_label": label, "questions": []},
                                 class_name=9, chapter=7, chapter_name="ত্ৰিভুজ", exercise_label=label)
        if status == review_state.STATUS_APPROVED:
            review_state.approve(pdf_path)
        elif status == review_state.STATUS_REJECTED:
            review_state.reject(pdf_path)
        return pdf_path

    def test_not_ready_when_any_exercise_is_pending(self):
        self._make_exercise("7.1", status=review_state.STATUS_APPROVED)
        self._make_exercise("7.2")  # still pending_review
        status = review_state.chapter_publish_status(9, 7, self.output_root)
        self.assertFalse(status["ready_to_publish"])
        self.assertIn("7.2", status["pending"])

    def test_ready_when_every_exercise_is_approved(self):
        self._make_exercise("7.1", status=review_state.STATUS_APPROVED)
        self._make_exercise("7.2", status=review_state.STATUS_APPROVED)
        status = review_state.chapter_publish_status(9, 7, self.output_root)
        self.assertTrue(status["ready_to_publish"])
        self.assertEqual(status["pending"], [])

    def test_not_ready_with_zero_exercises(self):
        status = review_state.chapter_publish_status(9, 99, self.output_root)
        self.assertFalse(status["ready_to_publish"])

    def test_publish_chapter_raises_if_not_all_approved(self):
        self._make_exercise("7.1", status=review_state.STATUS_APPROVED)
        self._make_exercise("7.2")
        with self.assertRaises(ValueError):
            review_state.publish_chapter(9, 7, self.output_root, self.published_root)

    def test_publish_chapter_copies_every_approved_pdf(self):
        p1 = self._make_exercise("7.1", status=review_state.STATUS_APPROVED)
        p2 = self._make_exercise("7.2", status=review_state.STATUS_APPROVED)
        result = review_state.publish_chapter(9, 7, self.output_root, self.published_root)
        self.assertEqual(len(result["published_files"]), 2)
        for src in (p1, p2):
            dst = os.path.join(result["dest_dir"], os.path.basename(src))
            self.assertTrue(os.path.exists(dst))

    def test_publish_never_touches_other_chapters(self):
        self._make_exercise("7.1", status=review_state.STATUS_APPROVED)
        other_pdf = os.path.join(self.output_root, "class_9", "chapter_8", "ex_8_1.pdf")
        os.makedirs(os.path.dirname(other_pdf), exist_ok=True)
        with open(other_pdf, "wb") as f:
            f.write(b"%PDF chapter 8")
        review_state.init_state(other_pdf, {"exercise_label": "8.1", "questions": []},
                                 9, 8, "বৃত্ত", "8.1")
        # Do NOT approve chapter 8's exercise.
        result = review_state.publish_chapter(9, 7, self.output_root, self.published_root)
        self.assertEqual(len(result["published_files"]), 1)


if __name__ == "__main__":
    unittest.main()
