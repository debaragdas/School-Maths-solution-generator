"""
AUDIT REGRESSION — real-MathJax render validation check.

Follows this project's existing e2e_*.py convention (see
e2e_hardening_check.py): real Playwright/Chromium against the actual
vendored MathJax bundle, no mocking of the rendering pipeline itself.
Not part of the fast pytest suite (which always mocks render_pdf —
see test_main.py / test_correction_engine.py) because it needs a real
browser; run it directly:

    python3 e2e_render_validation_check.py

Proves two things about the render-time gate added in this audit pass:

  1. Clean, fully-sanitized content still produces a normal PDF (no
     false positives from the new post-typeset error scan).
  2. Content that reaches pdf_generator with an undefined/malformed
     LaTeX macro — simulating a future code path that forgets to call
     sanitize_math_text() — is caught and BLOCKED before any PDF is
     written, with a diagnostic naming the offending question, rather
     than silently shipping a page with visible red MathJax error
     glyphs. This is the failure mode math_sanitizer.py's tightened
     KNOWN_LATEX_MACROS check exists to prevent at the source; this
     script proves the second, independent gate behind it also works.
"""
import os

import html_renderer
import pdf_generator

OUT_DIR = "/tmp/e2e_render_validation"
os.makedirs(OUT_DIR, exist_ok=True)


def check_clean_content_still_publishes():
    solved = {
        "questions": [
            {
                "question_number": 1,
                "question_text": r"\(\frac{1}{2} + \frac{1}{3}\) নিৰ্ণয় কৰা।",
                "given": "", "required": "",
                "steps": [r"\(\frac{1}{2} + \frac{1}{3} = \frac{5}{6}\)"],
                "final_answer": r"\(\frac{5}{6}\)", "answer_kind": "value",
            },
        ]
    }
    html = html_renderer.render_exercise_html(solved, 9, 3, "বীজগণিত", "3.1")
    out_path = os.path.join(OUT_DIR, "clean.pdf")
    pdf_generator.render_pdf(html, out_path)
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0, \
        "clean content should have produced a real PDF file"
    print("PASS: clean content still publishes normally")


def check_bypassed_broken_macro_is_blocked():
    mathjax_js = html_renderer._get_mathjax_js()
    raw_html = """<!DOCTYPE html><html><head>
<script>
window.__mathjaxReady__ = false;
window.MathJax = {tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['$$','$$']]},svg:{fontCache:'local'},
startup:{ready:()=>{MathJax.startup.defaultReady();MathJax.startup.promise.then(()=>{window.__mathjaxReady__=true;});}}};
</script>
<script>%s</script>
</head><body>
<div class="question-block"><div class="question-number">প্ৰশ্ন 9:</div>
<div>Bypassed sanitizer: \\(\\gibberish{x}\\)</div></div>
</body></html>""" % mathjax_js

    out_path = os.path.join(OUT_DIR, "should_not_exist.pdf")
    if os.path.exists(out_path):
        os.remove(out_path)
    try:
        pdf_generator.render_pdf(raw_html, out_path)
        raise AssertionError(
            "render_pdf should have raised RenderValidationError for an "
            "undefined macro but did not — a broken PDF would have shipped"
        )
    except pdf_generator.RenderValidationError as e:
        assert not os.path.exists(out_path), \
            "no PDF file should exist after a blocked render"
        assert "gibberish" in str(e) or "প্ৰশ্ন 9" in str(e), \
            f"error message should point at the offending question, got: {e}"
        print(f"PASS: bypassed broken macro correctly blocked -> {e}")


if __name__ == "__main__":
    check_clean_content_still_publishes()
    check_bypassed_broken_macro_is_blocked()
    print("\nAll render-validation regression checks passed.")
