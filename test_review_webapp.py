"""
Regression tests for review_webapp.py — the local Human Review browser
app that main.py's Stage 2 launches by default (config.REVIEW_MODE ==
"web").

`fitz` and `google.genai` are stubbed exactly as test_correction_engine.py
/ test_interactive_review.py already do. correction_engine.regenerate_
question / attach_manual_figure_to_question are mocked here (their real
merge/render logic is already fully covered by test_correction_engine.py)
— these tests exercise the FLASK ROUTES: request parsing, response
shape, and that every mutating endpoint really does delegate to the
same review_state.py / correction_engine.py functions the console
workflow uses, never touching a review.json or PDF directly itself.

Run: python3 -m pytest test_review_webapp.py -v
"""
import base64
import io
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
import review_webapp


def _sample_solved():
    return {
        "exercise_label": "7.4",
        "questions": [
            {"question_number": "1", "sub_part": None, "question_text": "Find AB in triangle ABC.",
             "given": "g1", "required": "r1", "steps": ["s1", "s2"],
             "final_answer": "AB = 5 cm", "construction_instruments": None,
             "diagram_spec": {"diagram_type": "triangle", "points": []},
             "book_diagram_base64": None, "book_diagram_mime": None,
             "book_diagram_figure_ref": None, "needs_review": True,
             "review_notes": ["Coordinate self-check could not reach a majority."]},
            {"question_number": "2", "sub_part": "a", "question_text": "Find the missing figure reference.",
             "given": "g2", "required": "r2", "steps": ["only step"],
             "final_answer": "untouched answer", "construction_instruments": None,
             "diagram_spec": None, "book_diagram_base64": base64.b64encode(b"fakepngbytes").decode(),
             "book_diagram_mime": "image/png", "book_diagram_figure_ref": "7.18",
             "needs_review": False, "review_notes": []},
        ],
    }


class ReviewWebappTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.pdf_path = os.path.join(self.output_dir, "class_9", "chapter_7", "ex_7_4.pdf")
        review_state.init_state(self.pdf_path, _sample_solved(), class_name=9, chapter=7,
                                 chapter_name="ত্ৰিভুজ", exercise_label="7.4")
        self.app = review_webapp.create_app(9, 7, output_dir=self.output_dir)
        self.app.testing = True
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestDashboardAndExerciseAPI(ReviewWebappTestBase):
    def test_dashboard_page_loads(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Human Review", resp.data)

    def test_exercise_page_loads(self):
        resp = self.client.get("/exercise/7.4")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Review Exercise", resp.data)

    def test_api_exercises_lists_this_chapter_only(self):
        resp = self.client.get("/api/exercises")
        data = json.loads(resp.data)
        self.assertEqual(len(data["exercises"]), 1)
        self.assertEqual(data["exercises"][0]["exercise_label"], "7.4")
        self.assertEqual(data["exercises"][0]["total_questions"], 2)
        self.assertFalse(data["chapter_status"]["ready_to_publish"])

    def test_api_exercise_detail_includes_question_flags(self):
        resp = self.client.get("/api/exercise/7.4")
        data = json.loads(resp.data)
        self.assertEqual(data["status"], review_state.STATUS_PENDING)
        q1 = data["questions"][0]
        self.assertEqual(q1["key"], "1")
        self.assertTrue(q1["needs_review"])
        self.assertFalse(q1["accepted"])
        q2 = data["questions"][1]
        self.assertEqual(q2["key"], "2(a)")
        self.assertTrue(q2["has_book_figure"])
        self.assertEqual(q2["book_diagram_figure_ref"], "7.18")

    def test_api_exercise_detail_404_for_unknown_exercise(self):
        resp = self.client.get("/api/exercise/7.99")
        self.assertEqual(resp.status_code, 404)

    def test_diagram_preview_renders_svg(self):
        resp = self.client.get("/api/exercise/7.4/diagram/1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "image/svg+xml")

    def test_diagram_preview_404_when_no_diagram(self):
        resp = self.client.get("/api/exercise/7.4/diagram/2(a)")
        self.assertEqual(resp.status_code, 404)

    def test_figure_preview_returns_decoded_bytes(self):
        resp = self.client.get("/api/exercise/7.4/figure/2(a)")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data, b"fakepngbytes")
        self.assertEqual(resp.mimetype, "image/png")

    def test_figure_preview_404_when_missing(self):
        resp = self.client.get("/api/exercise/7.4/figure/1")
        self.assertEqual(resp.status_code, 404)


class TestAcceptAndApprove(ReviewWebappTestBase):
    def test_accept_marks_question_accepted(self):
        resp = self.client.post("/api/exercise/7.4/question/1/accept")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["accepted_count"], 1)
        state = review_state.load_state(self.pdf_path)
        self.assertTrue(review_state.is_question_accepted(state, "1"))

    def test_accept_with_subpart_key(self):
        resp = self.client.post("/api/exercise/7.4/question/2(a)/accept")
        self.assertEqual(resp.status_code, 200)
        state = review_state.load_state(self.pdf_path)
        self.assertTrue(review_state.is_question_accepted(state, "2", "a"))

    def test_approve_blocked_until_every_question_accepted(self):
        self.client.post("/api/exercise/7.4/question/1/accept")
        resp = self.client.post("/api/exercise/7.4/approve", json={})
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("2(a)", data["error"])
        state = review_state.load_state(self.pdf_path)
        self.assertEqual(state["status"], review_state.STATUS_PENDING)

    def test_approve_succeeds_once_all_questions_accepted(self):
        self.client.post("/api/exercise/7.4/question/1/accept")
        self.client.post("/api/exercise/7.4/question/2(a)/accept")
        resp = self.client.post("/api/exercise/7.4/approve", json={"reviewer": "tester"})
        self.assertEqual(resp.status_code, 200)
        state = review_state.load_state(self.pdf_path)
        self.assertEqual(state["status"], review_state.STATUS_APPROVED)

    def test_approve_all_accepts_every_pending_question_then_approves(self):
        # No individual accepts at all beforehand — approve-all must
        # accept everything itself in one request.
        resp = self.client.post("/api/exercise/7.4/approve-all", json={"reviewer": "tester"})
        self.assertEqual(resp.status_code, 200)
        state = review_state.load_state(self.pdf_path)
        self.assertTrue(review_state.is_question_accepted(state, "1"))
        self.assertTrue(review_state.is_question_accepted(state, "2", "a"))
        self.assertEqual(state["status"], review_state.STATUS_APPROVED)

    def test_approve_all_leaves_already_accepted_questions_untouched(self):
        self.client.post("/api/exercise/7.4/question/1/accept")
        resp = self.client.post("/api/exercise/7.4/approve-all", json={})
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["accepted_count"], 2)
        self.assertEqual(data["status"], review_state.STATUS_APPROVED)

    def test_approve_all_404_for_unknown_exercise(self):
        resp = self.client.post("/api/exercise/7.99/approve-all", json={})
        self.assertEqual(resp.status_code, 404)

    def test_individual_accept_and_approve_endpoints_still_work_unchanged(self):
        # Both flows must coexist — approve-all is additive, not a replacement.
        self.client.post("/api/exercise/7.4/question/1/accept")
        self.client.post("/api/exercise/7.4/question/2(a)/accept")
        resp = self.client.post("/api/exercise/7.4/approve", json={})
        self.assertEqual(resp.status_code, 200)


class TestRegenerate(ReviewWebappTestBase):
    @mock.patch("review_webapp.correction_engine.regenerate_question")
    def test_regenerate_requires_instruction(self, mock_regen):
        resp = self.client.post("/api/exercise/7.4/question/1/regenerate",
                                 json={"instruction": "", "component": "solution"})
        self.assertEqual(resp.status_code, 400)
        mock_regen.assert_not_called()

    @mock.patch("review_webapp.correction_engine.regenerate_diagram_from_scratch")
    def test_regenerate_diagram_component_routes_to_from_scratch_rebuild(self, mock_rebuild):
        mock_rebuild.return_value = review_state.load_state(self.pdf_path)
        resp = self.client.post("/api/exercise/7.4/question/1/regenerate",
                                 json={"instruction": "Altitude missing", "component": "diagram"})
        self.assertEqual(resp.status_code, 200)
        mock_rebuild.assert_called_once()
        args, kwargs = mock_rebuild.call_args
        self.assertEqual(args[0], self.pdf_path)
        self.assertEqual(args[1], "1")
        self.assertIsNone(kwargs["sub_part"])
        self.assertEqual(kwargs["instruction"], "Altitude missing")

    @mock.patch("review_webapp.correction_engine.regenerate_diagram_from_scratch")
    def test_regenerate_diagram_component_allows_blank_instruction(self, mock_rebuild):
        # "Generate Diagram" for a question with none yet needs no
        # instruction at all — this must NOT be rejected client-side.
        mock_rebuild.return_value = review_state.load_state(self.pdf_path)
        resp = self.client.post("/api/exercise/7.4/question/2(a)/regenerate",
                                 json={"instruction": "", "component": "diagram"})
        self.assertEqual(resp.status_code, 200)
        mock_rebuild.assert_called_once()
        args, kwargs = mock_rebuild.call_args
        self.assertEqual(kwargs["sub_part"], "a")
        self.assertIsNone(kwargs["instruction"])

    @mock.patch("review_webapp.correction_engine.regenerate_diagram_from_scratch")
    def test_regenerate_diagram_component_failure_returns_500(self, mock_rebuild):
        mock_rebuild.side_effect = ValueError("boom")
        resp = self.client.post("/api/exercise/7.4/question/1/regenerate",
                                 json={"instruction": "", "component": "diagram"})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("boom", json.loads(resp.data)["error"])

    @mock.patch("review_webapp.correction_engine.regenerate_question")
    def test_regenerate_splits_subpart_key(self, mock_regen):
        mock_regen.return_value = review_state.load_state(self.pdf_path)
        self.client.post("/api/exercise/7.4/question/2(a)/regenerate",
                          json={"instruction": "Use textbook method", "component": None})
        args, kwargs = mock_regen.call_args
        self.assertEqual(args[1], "2")
        self.assertEqual(kwargs["sub_part"], "a")

    @mock.patch("review_webapp.correction_engine.regenerate_question")
    def test_regenerate_failure_returns_500_with_message(self, mock_regen):
        mock_regen.side_effect = ValueError("boom")
        resp = self.client.post("/api/exercise/7.4/question/1/regenerate",
                                 json={"instruction": "fix it"})
        self.assertEqual(resp.status_code, 500)
        self.assertIn("boom", json.loads(resp.data)["error"])


class TestFigureUpload(ReviewWebappTestBase):
    @mock.patch("review_webapp.correction_engine.attach_manual_figure_to_question")
    def test_figure_upload_via_json_base64(self, mock_attach):
        mock_attach.return_value = review_state.load_state(self.pdf_path)
        data_url = "data:image/png;base64," + base64.b64encode(b"cropped-bytes").decode()
        resp = self.client.post("/api/exercise/7.4/question/1/figure",
                                 json={"figure_number": "7.18", "page_number": 42,
                                       "image_base64": data_url, "reviewer": "tester"})
        self.assertEqual(resp.status_code, 200)
        mock_attach.assert_called_once()
        args, kwargs = mock_attach.call_args
        self.assertEqual(args[0], self.pdf_path)
        self.assertEqual(args[1], "1")
        self.assertEqual(kwargs["image_bytes"], b"cropped-bytes")
        self.assertEqual(kwargs["figure_number"], "7.18")
        self.assertEqual(kwargs["page_number"], 42)
        self.assertEqual(kwargs["reviewer"], "tester")

    @mock.patch("review_webapp.correction_engine.attach_manual_figure_to_question")
    def test_figure_upload_via_multipart_file(self, mock_attach):
        mock_attach.return_value = review_state.load_state(self.pdf_path)
        resp = self.client.post(
            "/api/exercise/7.4/question/1/figure",
            data={"figure_number": "7.19", "page_number": "10",
                  "file": (io.BytesIO(b"filebytes"), "crop.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 200)
        args, kwargs = mock_attach.call_args
        self.assertEqual(kwargs["image_bytes"], b"filebytes")
        self.assertEqual(kwargs["figure_number"], "7.19")
        self.assertEqual(kwargs["page_number"], 10)

    def test_figure_upload_missing_image_returns_400(self):
        resp = self.client.post("/api/exercise/7.4/question/1/figure",
                                 json={"figure_number": "7.18"})
        self.assertEqual(resp.status_code, 400)

    @mock.patch("review_webapp.correction_engine.attach_manual_figure_to_question")
    def test_figure_upload_with_blank_figure_number_succeeds_as_none(self, mock_attach):
        # V39: figure_number is now optional — a question with no real
        # book-figure citation (e.g. a construction/number-line question
        # that never had one) can still have an image manually attached;
        # it just isn't stored in the citable, format-validated figure
        # database (see correction_engine.attach_manual_figure_to_question's
        # no-citation branch). This must succeed, not 400.
        mock_attach.return_value = review_state.load_state(self.pdf_path)
        data_url = "data:image/png;base64," + base64.b64encode(b"x").decode()
        resp = self.client.post("/api/exercise/7.4/question/1/figure",
                                 json={"image_base64": data_url})
        self.assertEqual(resp.status_code, 200)
        args, kwargs = mock_attach.call_args
        self.assertIsNone(kwargs["figure_number"])


class TestRemoveDiagramAndFigure(ReviewWebappTestBase):
    @mock.patch("review_webapp.correction_engine.remove_generated_diagram")
    def test_remove_diagram_delegates_correctly(self, mock_remove):
        mock_remove.return_value = review_state.load_state(self.pdf_path)
        resp = self.client.delete("/api/exercise/7.4/question/1/diagram")
        self.assertEqual(resp.status_code, 200)
        mock_remove.assert_called_once()
        args, kwargs = mock_remove.call_args
        self.assertEqual(args[0], self.pdf_path)
        self.assertEqual(args[1], "1")
        self.assertIsNone(kwargs["sub_part"])

    @mock.patch("review_webapp.correction_engine.remove_generated_diagram")
    def test_remove_diagram_failure_returns_500(self, mock_remove):
        mock_remove.side_effect = ValueError("no such question")
        resp = self.client.delete("/api/exercise/7.4/question/1/diagram")
        self.assertEqual(resp.status_code, 500)

    @mock.patch("review_webapp.correction_engine.remove_book_figure")
    def test_remove_figure_delegates_correctly(self, mock_remove):
        mock_remove.return_value = review_state.load_state(self.pdf_path)
        resp = self.client.delete("/api/exercise/7.4/question/2(a)/figure")
        self.assertEqual(resp.status_code, 200)
        mock_remove.assert_called_once()
        args, kwargs = mock_remove.call_args
        self.assertEqual(kwargs["sub_part"], "a")

    @mock.patch("review_webapp.correction_engine.remove_book_figure")
    def test_remove_figure_failure_returns_500(self, mock_remove):
        mock_remove.side_effect = ValueError("boom")
        resp = self.client.delete("/api/exercise/7.4/question/2(a)/figure")
        self.assertEqual(resp.status_code, 500)


class TestStop(ReviewWebappTestBase):
    def test_stop_sets_flag(self):
        self.assertFalse(self.app.config["STOP_REQUESTED"])
        resp = self.client.post("/api/stop")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(self.app.config["STOP_REQUESTED"])


class TestRunReviewServerLifecycle(unittest.TestCase):
    """run_review_server() itself is a thin poll loop around create_app()
    + a real werkzeug server — exercised here with the server thread and
    browser-open both faked out, so this test never actually binds a
    socket or opens a browser."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "output")
        self.pdf_path = os.path.join(self.output_dir, "class_9", "chapter_7", "ex_7_4.pdf")
        review_state.init_state(self.pdf_path, _sample_solved(), class_name=9, chapter=7,
                                 chapter_name="ত্ৰিভুজ", exercise_label="7.4")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_false_when_reviewer_stops(self):
        created_apps = []
        real_create_app = review_webapp.create_app

        def _spy_create_app(*args, **kwargs):
            app = real_create_app(*args, **kwargs)
            created_apps.append(app)
            return app

        class _FakeThread:
            def __init__(self, app, host, port):
                self._app = app

            def start(self):
                # Simulate the reviewer clicking "Pause & Exit" immediately.
                self._app.config["STOP_REQUESTED"] = True

            def shutdown(self):
                pass

        with mock.patch.object(review_webapp, "make_server", mock.Mock()), \
             mock.patch.object(review_webapp, "create_app", side_effect=_spy_create_app), \
             mock.patch.object(review_webapp, "_ServerThread", _FakeThread), \
             mock.patch.object(review_webapp.webbrowser, "open"):
            result = review_webapp.run_review_server(
                9, 7, output_dir=self.output_dir, open_browser=False, poll_interval=0.01,
            )

        self.assertFalse(result)
        self.assertEqual(len(created_apps), 1)

    def test_returns_true_once_chapter_fully_approved(self):
        review_state.approve(self.pdf_path)  # only exercise in this chapter -> ready_to_publish

        class _FakeThread:
            def __init__(self, app, host, port):
                pass

            def start(self):
                pass

            def shutdown(self):
                pass

        with mock.patch.object(review_webapp, "make_server", mock.Mock()), \
             mock.patch.object(review_webapp, "_ServerThread", _FakeThread), \
             mock.patch.object(review_webapp.webbrowser, "open"):
            result = review_webapp.run_review_server(
                9, 7, output_dir=self.output_dir, open_browser=False, poll_interval=0.01,
            )

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
