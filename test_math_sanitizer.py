"""
Automated tests for math_sanitizer.py — the production Math
Sanitization Layer. Pure-Python module with no external dependencies
(no google.genai/fitz stubbing needed, unlike test_solver.py).

Run: python3 -m unittest test_math_sanitizer -v
"""
import unittest

import math_sanitizer as ms


class TestValidContentPassesThroughUnchanged(unittest.TestCase):
    """The sanitizer must never touch already-correct content."""

    def test_simple_fraction(self):
        self.assertEqual(ms.sanitize_math_text("F = \\(\\frac{9}{5}(0)\\) + 32 = 32"),
                          "F = \\(\\frac{9}{5}(0)\\) + 32 = 32")

    def test_vector(self):
        self.assertEqual(ms.sanitize_math_text("\\(\\vec{AB}\\)"), "\\(\\vec{AB}\\)")

    def test_angle_with_degree(self):
        self.assertEqual(ms.sanitize_math_text("\\(\\angle ABC = 90^{\\circ}\\)"),
                          "\\(\\angle ABC = 90^{\\circ}\\)")

    def test_overline(self):
        self.assertEqual(ms.sanitize_math_text("\\(\\overline{AB} = \\overline{CD}\\)"),
                          "\\(\\overline{AB} = \\overline{CD}\\)")

    def test_well_formed_matrix(self):
        s = "\\(\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}\\)"
        self.assertEqual(ms.sanitize_math_text(s), s)

    def test_sqrt_with_index(self):
        self.assertEqual(ms.sanitize_math_text("\\(\\sqrt[3]{27} = 3\\)"),
                          "\\(\\sqrt[3]{27} = 3\\)")

    def test_half_open_interval_mismatched_bracket_chars(self):
        # \left( ... \right] is valid LaTeX — bracket TYPES don't need
        # to match, only the \left/\right pairing itself.
        s = "\\(\\left(0, 1\\right]\\)"
        self.assertEqual(ms.sanitize_math_text(s), s)

    def test_plain_assamese_prose_with_no_math_untouched(self):
        s = "AB = 5 একক, CD = 3 একক"
        self.assertEqual(ms.sanitize_math_text(s), s)

    def test_multiple_separate_spans_untouched(self):
        s = "\\(x = 5\\) আৰু \\(y = 10\\)"
        self.assertEqual(ms.sanitize_math_text(s), s)

    def test_empty_string(self):
        self.assertEqual(ms.sanitize_math_text(""), "")

    def test_non_string_passthrough(self):
        self.assertIsNone(ms.sanitize_math_text(None))
        self.assertEqual(ms.sanitize_math_text(42), 42)
        self.assertEqual(ms.sanitize_math_text([1, 2]), [1, 2])


class TestBareLatexWrapping(unittest.TestCase):
    """Undelimited LaTeX Gemini forgot to wrap must get \\( \\) added."""

    def test_simple_bare_power(self):
        self.assertEqual(ms.sanitize_math_text("32^{\\frac{2}{5}}"),
                          "\\(32^{\\frac{2}{5}}\\)")

    def test_bare_number_left_alone(self):
        # No backslash/^/_ at all -> nothing to wrap.
        self.assertEqual(ms.sanitize_math_text("32 apples"), "32 apples")

    def test_bare_fraction_with_internal_space_not_split(self):
        """THE reported production bug: a bare \\frac whose argument
        contains a space used to get cut mid-brace by the old regex,
        producing a broken '\\(\\frac{-32\\) \\(\\times\\) 5}{9}' with a
        dangling, never-wrapped '}{9}' tail sitting OUTSIDE any \\( \\)."""
        result = ms.sanitize_math_text(
            "C = \\frac{-32 \\times 5}{9} = \\frac{-160}{9} \\approx -17.78")
        self.assertEqual(
            result,
            "C = \\(\\frac{-32 \\times 5}{9}\\) = \\(\\frac{-160}{9}\\) \\(\\approx\\) -17.78")
        # The old bug's exact signature: a "}{9}" sitting right after a
        # closing "\)" — i.e. OUTSIDE math mode — never happens here.
        self.assertNotIn("\\)}{9}", result)
        self.assertNotIn("\\)\\(\\times\\)5}{9}", result)

    def test_bare_parens_with_internal_space_not_split(self):
        """SAME bug class, but for PARENTHESES instead of braces — a
        bare "(7 \\times 8)^{\\frac{1}{2}}" (completely ordinary: a
        product raised to a power) used to get cut at the space inside
        the still-open "(", exactly like the frac/brace case above,
        producing the broken "(7 \\(\\times\\) \\(8)^{\\frac{1}{2}}\\)"
        seen in a real production PDF."""
        result = ms.sanitize_math_text(
            "7^{\\frac{1}{2}} \\cdot 8^{\\frac{1}{2}} = (7 \\times 8)^{\\frac{1}{2}}")
        self.assertEqual(
            result,
            "\\(7^{\\frac{1}{2}}\\) \\(\\cdot\\) \\(8^{\\frac{1}{2}}\\) = \\((7 \\times 8)^{\\frac{1}{2}}\\)")
        self.assertNotIn("(7 \\(", result)
        self.assertNotIn("\\(8)", result)

    def test_nested_parens_with_spaces_stay_together(self):
        result = ms.sanitize_math_text("((1 + 2) \\times 3)^2 = 81")
        self.assertEqual(result, "\\(((1 + 2) \\times 3)^2\\) = 81")

    def test_mixed_already_delimited_and_bare(self):
        s = "already \\(\\frac{9}{5}\\) delimited, then bare \\sqrt{2 + 3} after"
        result = ms.sanitize_math_text(s)
        self.assertIn("\\(\\frac{9}{5}\\)", result)
        self.assertIn("\\(\\sqrt{2 + 3}\\)", result)


class TestJsonEscapeLeakage(unittest.TestCase):
    """Literal '\\n' '\\t' '\\r' '\\b' '\\f' surviving as backslash-letter
    text (vs. a genuine macro for that letter) must be disambiguated
    correctly."""

    def test_n_becomes_br(self):
        self.assertEqual(ms.sanitize_math_text("line1\\nline2"), "line1<br>line2")

    def test_t_becomes_spacing(self):
        self.assertEqual(ms.sanitize_math_text("a\\tb"), "a&nbsp;&nbsp;&nbsp;&nbsp;b")

    def test_r_b_f_dropped(self):
        self.assertEqual(ms.sanitize_math_text("x\\ry\\bz\\fw"), "xyzw")

    def test_genuine_macros_starting_with_leak_letters_untouched(self):
        # These are bare (undelimited) genuine macros: correctly left
        # un-mangled by the escape-leak decoder (not turned into <br>
        # etc.), and then correctly \( \)-wrapped by wrap_bare_latex
        # like any other bare LaTeX — never treated as leaked JSON
        # control characters.
        s = "\\tan(x), \\nabla f, \\theta, \\triangle ABC, \\therefore, \\text{units}"
        result = ms.sanitize_math_text(s)
        self.assertNotIn("<br>", result)
        self.assertNotIn("&nbsp;", result)
        for macro in ("\\tan(x)", "\\nabla", "\\theta", "\\triangle", "\\therefore", "\\text{units}"):
            self.assertIn(macro, result)

    def test_tanx_still_wraps_as_single_bare_run(self):
        # \tan followed directly by 'x' with no space — must not be
        # mistaken for a bad escape, and gets wrapped as one bare run.
        result = ms.sanitize_math_text("\\tan(x) = 1")
        self.assertIn("\\tan(x)", result)
        self.assertNotIn("<br>", result)


class TestUnbalancedBraces(unittest.TestCase):
    def test_missing_closing_brace_repaired(self):
        self.assertEqual(ms.sanitize_math_text("\\(\\frac{1}{2\\)"), "\\(\\frac{1}{2}\\)")

    def test_extra_closing_brace_repaired(self):
        self.assertEqual(ms.sanitize_math_text("\\(\\frac{1}}{2}\\)"), "\\(\\frac{1}{2}\\)")

    def test_deeply_unbalanced_still_produces_valid_output(self):
        result = ms.sanitize_math_text("\\(\\frac{{{1}{2}\\)")
        # Never leaves a raw, still-unbalanced \( \) span.
        inner = result[2:-2] if result.startswith("\\(") and result.endswith("\\)") else result
        self.assertEqual(inner.count("{"), inner.count("}"))


class TestOrphanAndNestedDelimiters(unittest.TestCase):
    def test_orphan_close_dropped(self):
        self.assertEqual(ms.sanitize_math_text("y = 5\\) stray"), "y = 5 stray")

    def test_orphan_open_autocloses_at_end(self):
        result = ms.sanitize_math_text("z = \\(\\frac{1}{2} never closed")
        self.assertTrue(result.count("\\(") == result.count("\\)"))

    def test_nested_open_splits_into_two_spans(self):
        result = ms.sanitize_math_text("x = \\(\\(\\frac{1}{2}\\) + 1\\)")
        self.assertTrue(result.count("\\(") == result.count("\\)"))
        self.assertIn("\\(\\frac{1}{2}\\)", result)

    def test_unclosed_bare_open_never_corrupts_output(self):
        # No matching close anywhere in the whole text.
        result = ms.sanitize_math_text("total = \\(\\frac{1}{2}")
        self.assertEqual(result.count("\\("), result.count("\\)"))
        self.assertNotIn("\\\\((", result)  # the exact corruption pattern once seen


class TestLeftRightMismatch(unittest.TestCase):
    def test_matched_left_right_kept(self):
        s = "\\(\\left(\\frac{1}{2}\\right)\\)"
        self.assertEqual(ms.sanitize_math_text(s), s)

    def test_unmatched_left_stripped_not_broken(self):
        result = ms.sanitize_math_text("\\(\\left(\\frac{1}{2}\\)")
        self.assertNotIn("\\left", result)
        self.assertNotIn("\\right", result)
        # Still valid: braces balanced, degrades gracefully either way.
        self.assertTrue(result.count("\\(") == result.count("\\)"))


class TestMatrixVectorAngleDelimiterConstructs(unittest.TestCase):
    def test_mismatched_matrix_env_names_degrade_to_readable_text(self):
        result = ms.sanitize_math_text("\\(\\begin{matrix}1&2\\\\3&4\\end{pmatrix}\\)")
        self.assertNotIn("\\begin", result)
        self.assertNotIn("\\end", result)
        self.assertNotIn("\\(", result)
        for n in ("1", "2", "3", "4"):
            self.assertIn(n, result)

    def test_well_formed_matrix_row_separator_preserved(self):
        # The \\\\ row separator must survive the escape-collapse step
        # intact (exactly 2 backslashes), not be collapsed to 1.
        result = ms.sanitize_math_text("\\(\\begin{vmatrix}1&0\\\\0&1\\end{vmatrix}\\)")
        self.assertIn("\\\\", result)
        self.assertIn("\\begin{vmatrix}", result)


class TestAssameseInsideMathSpan(unittest.TestCase):
    def test_assamese_run_split_out_of_math_mode(self):
        result = ms.sanitize_math_text("\\(\\frac{1}{2} বাকী অংশ 3x\\)")
        self.assertIn("\\(\\frac{1}{2}\\)", result)
        self.assertIn("বাকী অংশ", result)
        self.assertIn("\\(3x\\)", result)
        # The Assamese text itself must never sit inside \( \).
        self.assertNotIn("\\(\\frac{1}{2} বাকী", result)


class TestMalformedCommandsDegradeSafely(unittest.TestCase):
    def test_stray_backslash_digit_degrades(self):
        result = ms.sanitize_math_text("\\(5 \\9 apples\\)")
        self.assertNotIn("\\9", result)
        self.assertNotIn("\\(", result)
        self.assertIn("5", result)
        self.assertIn("apples", result)

    def test_never_leaves_a_raw_unbalanced_delimiter_visible(self):
        """Broad fuzz-style check: no matter how garbled the input, the
        output must never contain a lone, unmatched \\( or \\)."""
        garbage_inputs = [
            "\\(\\(\\(unbalanced",
            "\\)\\)\\) too many closes",
            "\\(\\frac{{{{1}}2}\\)",
            "\\left\\left\\left(1\\right)",
            "\\(\\begin{matrix}\\end{array}\\)",
            "random \\@ \\% \\# \\9 \\0 garbage \\(",
        ]
        for text in garbage_inputs:
            result = ms.sanitize_math_text(text)
            self.assertEqual(result.count("\\("), result.count("\\)"),
                              f"unbalanced delimiters survived for input: {text!r} -> {result!r}")


class TestFinalDegradeReadability(unittest.TestCase):
    """When a span can't be repaired, the plain-text fallback must
    still be legible math, not raw LaTeX soup."""

    def test_degraded_fraction_reads_as_a_over_b(self):
        # Force a degrade via a genuinely broken environment mismatch
        # wrapping a fraction.
        result = ms.sanitize_math_text(
            "\\(\\begin{matrix}\\frac{1}{2}\\end{array}\\)")
        self.assertIn("(1/2)", result)

    def test_degrade_never_contains_backslash(self):
        result = ms.sanitize_math_text("\\(\\begin{matrix}1&2\\end{array}\\)")
        self.assertNotIn("\\", result)


class TestBareUnicodeMathSymbolsInsideDelimiters(unittest.TestCase):
    """Regression tests for a THIRD, distinct root cause found analyzing
    a real irrational-numbers (surds) exercise: content like
    '\\(2√5 × 2\\)' is perfectly well-STRUCTURED (braces/parens/left-
    right/environments all balanced), so the earlier structural
    checks correctly say "fine" — but MathJax's TeX parser expects
    real LaTeX commands inside math mode, not bare Unicode glyphs, and
    can throw a hard parse error (the "raw text in red" symptom) on a
    literal "√" or "×" character even though nothing is unbalanced."""

    def test_bare_sqrt_of_single_number_converted(self):
        self.assertEqual(ms.sanitize_math_text("\\(2\u221a10\\)"), "\\(2\\sqrt{10}\\)")

    def test_sqrt_with_paren_argument_converted(self):
        self.assertEqual(ms.sanitize_math_text("\\(\u221a(5 \u00d7 2)\\)"),
                          "\\(\\sqrt{5  \\times  2}\\)")

    def test_times_symbol_converted(self):
        result = ms.sanitize_math_text("\\((3 \u00d7 2)\\)")
        self.assertIn("\\times", result)
        self.assertNotIn("\u00d7", result)

    def test_degree_symbol_converted(self):
        self.assertEqual(ms.sanitize_math_text("\\(90\u00b0\\)"), "\\(90^{\\circ}\\)")

    def test_exact_reported_pdf_pattern(self):
        result = ms.sanitize_math_text("5 + \\(2\u221a5 \u00d7 2\\) + 2")
        self.assertNotIn("\u221a", result)
        self.assertNotIn("\u00d7", result)
        self.assertIn("\\sqrt{5}", result)
        self.assertIn("\\times", result)

    def test_conjugate_radical_subtraction_pattern(self):
        result = ms.sanitize_math_text("\\((\u221a5)\u00b2 - (\u221a2)\u00b2\\)")
        self.assertIn("\\sqrt{5}", result)
        self.assertIn("\\sqrt{2}", result)
        self.assertNotIn("\u221a", result)

    def test_never_produces_a_second_unbalanced_delimiter(self):
        for text in ["\\(2\u221a5 \u00d7 2\\)", "\\(\u221a7 - \u221a6\\)", "\\((\u221a5)\u00b2\\)"]:
            out = ms.sanitize_math_text(text)
            self.assertEqual(out.count("\\("), out.count("\\)"))


class TestIdempotence(unittest.TestCase):
    """Running the sanitizer twice (e.g. once at solve-time, again as
    the html_renderer safety net) must never change already-clean
    output further."""

    def test_double_pass_stable(self):
        cases = [
            "C = \\frac{-32 \\times 5}{9} = \\frac{-160}{9} \\approx -17.78",
            "\\(\\begin{matrix}1&2\\\\3&4\\end{pmatrix}\\)",
            "\\(\\left(0, 1\\right]\\)",
            "line1\\nline2 and \\tan(x)",
            "AB = 5 একক",
            "\\(2\u221a5 \u00d7 2\\)",
        ]
        for c in cases:
            once = ms.sanitize_math_text(c)
            twice = ms.sanitize_math_text(once)
            self.assertEqual(once, twice, f"not idempotent for {c!r}: {once!r} -> {twice!r}")


class TestUnknownMacroDegradation(unittest.TestCase):
    """AUDIT REGRESSION (see math_sanitizer.py's _looks_structurally_valid
    comment): confirmed by directly rendering against this project's own
    vendored MathJax bundle that an undefined control sequence does NOT
    raise a normal MathJax parse error — it silently falls back to
    drawing the raw command text as red glyphs, with no <merror> marker
    anywhere in the DOM. Structural validation alone (balanced braces/
    left-right/environments) can never catch this, so unknown
    multi-letter macros must be rejected here and degraded to plain
    text instead of being passed through as "probably fine"."""

    def test_hallucinated_macro_degrades_instead_of_passing_through(self):
        for bad in (r"\gibberish", r"\sinx(x)", r"\vecc{v}", r"\notarealcommand"):
            out = ms.sanitize_math_text(bad)
            # Must not survive as a live, undelimited-as-real-macro span —
            # either it's plain text now, or (if wrapped) the macro name
            # itself is gone.
            self.assertNotIn("\\gibberish", out)
            self.assertNotIn("\\sinx", out)
            self.assertNotIn("\\vecc", out)
            self.assertNotIn("\\notarealcommand", out)

    def test_real_known_macros_still_pass_through_untouched(self):
        for good in (r"\frac{1}{2}", r"\sin(x)", r"\vec{v}", r"\alpha + \beta",
                     r"\uparrow", r"\vdash", r"\mathcal{A}"):
            out = ms.sanitize_math_text(good)
            self.assertIn("\\(", out)
            self.assertIn("\\)", out)


if __name__ == "__main__":
    unittest.main()
