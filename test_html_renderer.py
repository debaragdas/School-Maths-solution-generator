"""
Regression tests for html_renderer.py.

PRODUCTION-AUDIT NOTE: this test file did not exist before — a real,
pre-existing coverage gap that let a genuine bug ship undetected: the
self_verify() integration (wired in for the Book Intelligence Layer)
called logger.info()/logger.warning() without html_renderer.py ever
importing `logger`, so the FIRST question that triggered a
needs_review-worthy confidence report crashed the entire render with
a NameError raised from inside the except-handler itself (the handler
tried to log the original NameError and hit a second one). No test
anywhere in the 367-test suite exercised render_exercise_html() with
data that actually triggers that code path, so it shipped invisibly.
Fixed by importing logger; this file exists so that class of gap
cannot recur silently again.

`fitz` and `google.genai` are stubbed the same way test_solver.py
already does, since solver.py (imported transitively for
_normalize_all_text_fields in a couple of tests) needs both at import
time and neither is installed/reachable in this sandbox.

Run: python3 -m pytest test_html_renderer.py -v
"""
import sys
import types
import unittest
from unittest import mock


def _install_stubs():
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


_install_stubs()

import html_renderer as hr


def _question(**overrides):
    base = {
        "question_number": "1", "sub_part": "",
        "question_text": "A test question.",
        "given": "some given data", "required": "find something",
        "steps": ["step one", "step two"],
        "final_answer": "the answer",
        "diagram_spec": None, "has_book_diagram": False,
    }
    base.update(overrides)
    return base


class TestRenderExerciseHtml(unittest.TestCase):
    def test_basic_render_produces_html(self):
        html = hr.render_exercise_html({"questions": [_question()]}, 9, 7, "ত্ৰিভুজ", "7.1")
        self.assertIn("<html", html.lower())
        self.assertIn("A test question.", html)

    def test_multiple_questions_all_present(self):
        qs = [_question(question_number=str(i), question_text=f"Question {i}") for i in range(1, 6)]
        html = hr.render_exercise_html({"questions": qs}, 9, 7, "X", "7.1")
        for i in range(1, 6):
            self.assertIn(f"Question {i}", html)

    def test_empty_questions_list_does_not_crash(self):
        html = hr.render_exercise_html({"questions": []}, 9, 7, "X", "7.1")
        self.assertIn("<html", html.lower())

    def test_generated_diagram_question_renders_svg(self):
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]]}
        html = hr.render_exercise_html({"questions": [_question(diagram_spec=spec)]}, 9, 7, "X", "7.1")
        self.assertIn("<svg", html)

    def test_book_diagram_question_embeds_image(self):
        import base64
        import io
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (100, 80), "white")
        ImageDraw.Draw(img).ellipse([10, 10, 90, 70], outline="black", width=3)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        fake_png = base64.b64encode(buf.getvalue()).decode("ascii")
        q = _question(diagram_spec=None, has_book_diagram=True,
                       book_diagram_base64=fake_png, book_diagram_mime="image/png",
                       book_diagram_figure_ref="7.14")
        html = hr.render_exercise_html({"questions": [q]}, 9, 7, "X", "7.1")
        self.assertIn(fake_png[:50], html)  # embedded somewhere in an <img> src

    def test_large_diagram_type_gets_stacked_flag(self):
        spec = {"diagram_type": "statistics", "chart_type": "bar",
                "categories": ["A", "B"], "values": [1, 2]}
        html = hr.render_exercise_html({"questions": [_question(diagram_spec=spec)]}, 9, 7, "X", "7.1")
        self.assertIn("diagram-box--large", html)

    # ------------------------------------------------------------------
    # THE BUG THIS FILE EXISTS TO PREVENT FROM RECURRING SILENTLY:
    # ------------------------------------------------------------------

    def test_self_verify_integration_never_crashes_the_render(self):
        """Runs a full render with a spec that self_verify's
        ConstraintGraph will flag as contradictory (equal_marks
        disagrees with side_lengths) -- this is exactly the
        needs_review code path that used to hit the undefined `logger`
        NameError and crash the ENTIRE exercise render, not just skip
        one question's confidence report."""
        spec = {"diagram_type": "triangle", "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "equal_marks": [["AB", "AC"]], "side_lengths": {"AB": 5, "AC": 9}}
        html = hr.render_exercise_html({"questions": [_question(diagram_spec=spec)]}, 9, 7, "X", "7.1")
        # must complete and still contain the question's content --
        # a crash here would raise before ever reaching this assertion
        self.assertIn("<html", html.lower())
        self.assertIn("<svg", html)

    def test_self_verify_logging_actually_uses_a_real_logger(self):
        """Directly proves html_renderer.logger is a real, callable
        logger object (not missing) -- the precise fact whose absence
        caused the crash."""
        self.assertTrue(hasattr(hr, "logger"))
        hr.logger.info("smoke test — must not raise")
        hr.logger.warning("smoke test — must not raise")

    def test_self_verify_failure_itself_is_swallowed_not_fatal(self):
        """If self_verify() raises for some unrelated reason, the
        render must still complete (defense in depth, independent of
        the logger fix above)."""
        with mock.patch.object(hr.book_intelligence, "self_verify", side_effect=RuntimeError("boom")):
            html = hr.render_exercise_html({"questions": [_question()]}, 9, 7, "X", "7.1")
        self.assertIn("<html", html.lower())

    # ------------------------------------------------------------------
    # REGRESSION TESTS FOR TEXT RENDERING FIXES (Assamese, LaTeX, CSS)
    # ------------------------------------------------------------------

    def test_css_includes_font_synthesis_none(self):
        """Verify the CSS includes font-synthesis: none to fix broken
        Assamese conjuncts in bold text (production bug fix)."""
        html = hr.render_exercise_html({"questions": [_question()]}, 9, 7, "X", "7.1")
        self.assertIn("font-synthesis: none", html)

    def test_assamese_text_preserved_in_output(self):
        """Verify Assamese text (including conjuncts) is preserved
        in the rendered HTML without escaping or corruption."""
        assamese_text = "ব্যাসাৰ্ধ চূড়ান্ত উত্তৰ"
        q = _question(question_text=assamese_text, given=assamese_text, final_answer=assamese_text)
        html = hr.render_exercise_html({"questions": [q]}, 9, 7, "ত্ৰিভুজ", "7.1")
        self.assertIn(assamese_text, html)

    def test_latex_delimiters_preserved_in_output(self):
        """Verify LaTeX delimiters are preserved in the rendered HTML
        so MathJax can render them correctly (not escaped)."""
        latex_content = r"Given: \( \sqrt{x^2 + y^2} \) Required: \( \frac{a}{b} \)"
        q = _question(given=latex_content, required=latex_content, steps=[latex_content])
        html = hr.render_exercise_html({"questions": [q]}, 9, 7, "X", "7.1")
        # LaTeX delimiters must be present, not escaped as &amp;#92; etc.
        self.assertIn(r"\(", html)
        self.assertIn(r"\)", html)
        self.assertIn(r"\sqrt", html)
        self.assertIn(r"\frac", html)

    def test_font_face_declaration_present(self):
        """Verify the Tiro Bangla font-face declaration is present
        in the CSS for proper Assamese glyph rendering."""
        html = hr.render_exercise_html({"questions": [_question()]}, 9, 7, "X", "7.1")
        self.assertIn("@font-face", html)
        self.assertIn("Tiro Bangla", html)

    def test_assamese_numerals_converted(self):
        """Verify ASCII numerals are converted to Assamese numerals
        in chapter numbers as expected."""
        html = hr.render_exercise_html({"questions": [_question()]}, 9, 7, "X", "7.1")
        # Chapter 7 should be rendered as Assamese ৭
        self.assertIn("অধ্যায় ৭", html)


if __name__ == "__main__":
    unittest.main()
