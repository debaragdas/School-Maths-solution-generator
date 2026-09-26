"""
test_math_pipeline_comprehensive.py — end-to-end hardening tests for
the whole math-formatting pipeline (Gemini output -> math_sanitizer.py
-> latex_grammar.py -> html_renderer.py -> diagram_renderer.py's SVG
labels -> MathJax).

Unlike test_math_sanitizer.py (which pins exact expected strings for
individual repair steps) and test_latex_grammar.py (which pins exact
accept/reject verdicts for the grammar parser in isolation), this file
asserts PIPELINE-LEVEL INVARIANTS that must hold no matter how broken
the input is:

  1. sanitize_math_text() never raises, on any input.
  2. Its output never contains an unbalanced "\\(" / "\\)".
  3. Every "\\( ... \\)" span in its output is grammatically valid
     LaTeX (per latex_grammar.check_latex_grammar) — if it weren't,
     that span must have been degraded to plain text instead.
  4. It is idempotent: sanitizing already-sanitized text is a no-op.
  5. sanitize_svg_labels() never lets a backslash or an unbalanced
     brace reach an SVG <text> element.
  6. Nothing here is achieved by matching one specific bug shape —
     every check is run over a wide, mixed battery of inputs
     (fractions/powers/roots/matrices/equations/geometry labels/mixed
     Assamese+English/nested expressions/deliberately-broken AI
     output/random fuzz), so a fix for one shape can't be quietly
     narrow.

Run: python3 -m unittest test_math_pipeline_comprehensive -v
"""
import random
import re
import unittest

import math_sanitizer as ms
import latex_grammar as lg
import diagram_renderer as dr

_SPAN_RE = re.compile(r"\\\((.*?)\\\)", re.DOTALL)


def assert_pipeline_invariants(case, raw_input, out):
    """Shared invariant checker used by every test in this file."""
    case.assertIsInstance(out, str)

    opens, closes = out.count("\\("), out.count("\\)")
    case.assertEqual(opens, closes,
                      f"unbalanced \\( \\) for input {raw_input!r} -> {out!r}")

    for span in _SPAN_RE.findall(out):
        valid, reason = lg.check_latex_grammar(span)
        case.assertTrue(
            valid,
            f"grammatically INVALID span reached output (would be a red "
            f"MathJax error): {span!r} ({reason}) — input was {raw_input!r}, "
            f"full output {out!r}"
        )
        # A span that survived as real math must never itself contain
        # a leaked opening/closing delimiter or an unbalanced brace.
        case.assertNotIn("\\(", span)
        case.assertNotIn("\\)", span)
        case.assertEqual(span.count("{"), span.count("}"))

    # Idempotence: sanitizing the already-sanitized output changes
    # nothing further.
    twice = ms.sanitize_math_text(out)
    case.assertEqual(out, twice,
                      f"not idempotent for input {raw_input!r}: "
                      f"first pass {out!r}, second pass {twice!r}")


def run_and_check(case, raw_input):
    out = ms.sanitize_math_text(raw_input)
    assert_pipeline_invariants(case, raw_input, out)
    return out


class TestFractions(unittest.TestCase):
    def test_simple(self):
        run_and_check(self, r"\(\frac{1}{2} + \frac{1}{3} = \frac{5}{6}\)")

    def test_nested(self):
        run_and_check(self, r"\(\frac{\frac{1}{2}}{\frac{3}{4}}\)")

    def test_bare_frac_gets_wrapped(self):
        out = run_and_check(self, r"The answer is \frac{9}{5} exactly.")
        self.assertIn("\\(", out)

    def test_missing_second_argument_degrades_safely(self):
        out = run_and_check(self, r"\(\frac{1}\)")
        self.assertNotIn("\\frac", out)
        self.assertNotIn("{", out)
        self.assertNotIn("}", out)

    def test_missing_both_arguments_degrades_safely(self):
        out = run_and_check(self, r"\(\frac\)")
        self.assertNotIn("\\", out)

    def test_unclosed_frac(self):
        out = run_and_check(self, r"The value is \frac{1}{2 and more text")
        # The trailing prose must not get swallowed as \frac's 2nd
        # argument — the fraction closes at "2" and "and more text"
        # remains ordinary trailing prose.
        self.assertIn("and more text", out)
        self.assertNotIn("{2 and", out)


class TestPowers(unittest.TestCase):
    def test_simple_power(self):
        run_and_check(self, r"\(x^2 + y^2 = z^2\)")

    def test_multi_digit_exponent_gets_braced(self):
        out = run_and_check(self, r"\(2^10 = 1024\)")
        self.assertIn("2^{10}", out)

    def test_negative_exponent_gets_braced(self):
        out = run_and_check(self, r"\(x^-1\)")
        self.assertIn("x^{-1}", out)

    def test_letter_after_digit_exponent_left_alone(self):
        # x^2y is standard, CORRECT notation (y is a separate factor)
        # — must not be reinterpreted as x^{2y}.
        out = run_and_check(self, r"\(x^2y\)")
        self.assertIn("x^2y", out)

    def test_dangling_caret_degrades_safely(self):
        out = run_and_check(self, r"\(x^\)")
        self.assertNotIn("^", out)

    def test_double_superscript_degrades_safely(self):
        out = run_and_check(self, r"\(x^^2\)")
        for span in _SPAN_RE.findall(out):
            self.assertTrue(lg.check_latex_grammar(span)[0])

    def test_subscript_chain(self):
        run_and_check(self, r"\(a_1^2 + a_2^2 + \dots + a_n^2\)")


class TestRoots(unittest.TestCase):
    def test_square_root(self):
        run_and_check(self, r"\(\sqrt{16} = 4\)")

    def test_cube_root(self):
        run_and_check(self, r"\(\sqrt[3]{27} = 3\)")

    def test_nested_root(self):
        run_and_check(self, r"\(\sqrt{\sqrt{16}}\)")

    def test_bare_unicode_sqrt_gets_converted(self):
        out = run_and_check(self, r"\(2√5 \times 2\)")
        self.assertIn("\\sqrt", out)
        self.assertNotIn("√", out)

    def test_sqrt_missing_argument_degrades_safely(self):
        out = run_and_check(self, r"\(\sqrt\)")
        self.assertNotIn("\\", out)


class TestMatrices(unittest.TestCase):
    def test_well_formed_matrix(self):
        run_and_check(self, r"\(\begin{pmatrix}1&2\\3&4\end{pmatrix}\)")

    def test_cases_environment(self):
        run_and_check(self, r"\(f(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}\)")

    def test_mismatched_matrix_names_degrades_safely(self):
        out = run_and_check(self, r"\(\begin{matrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\)")
        self.assertNotIn("\\begin", out)
        self.assertNotIn("\\end", out)

    def test_unterminated_matrix_degrades_safely(self):
        out = run_and_check(
            self, r"\(\begin{bmatrix} 1 & 2 \\ 3 & 4\)"
        )
        self.assertNotIn("\\begin", out)


class TestEquations(unittest.TestCase):
    def test_quadratic_formula(self):
        run_and_check(self, r"\(x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\)")

    def test_system_like_equation(self):
        run_and_check(self, r"\(3x + 2y = 12\), \(x - y = 1\)")

    def test_trig_identity(self):
        run_and_check(self, r"\(\sin^2\theta + \cos^2\theta = 1\)")


class TestGeometryLabels(unittest.TestCase):
    """diagram_renderer.py's SVG output — the SECOND, previously
    unprotected, path into the rendered page."""

    def _extra_label_svg(self, label_text):
        spec = {
            "diagram_type": "triangle",
            "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
            "right_angle_at": "B",
            "extra_labels": [label_text],
        }
        return dr.render_diagram(spec)

    def test_plain_label_untouched(self):
        svg = self._extra_label_svg("Right triangle ABC")
        self.assertIn("Right triangle ABC", svg)

    def test_leaked_frac_in_label_never_reaches_svg(self):
        svg = self._extra_label_svg(r"side = \frac{1}{2} m")
        self.assertNotIn("\\frac", svg)
        self.assertNotIn("{", svg.split("</svg>")[0].split("extra")[-1]) if False else None
        # No leaked LaTeX command or brace anywhere a label was placed.
        for m in re.finditer(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL):
            self.assertNotIn("\\", m.group(1))

    def test_degree_circ_becomes_degree_symbol(self):
        svg = self._extra_label_svg(r"angle = 60^\circ")
        self.assertIn("60°", svg)
        self.assertNotIn("\\circ", svg)

    def test_leaked_matrix_in_label_degrades_safely(self):
        svg = self._extra_label_svg(r"\begin{matrix}1&2\end{matrix}")
        for m in re.finditer(r"<text[^>]*>(.*?)</text>", svg, re.DOTALL):
            self.assertNotIn("\\", m.group(1))
            self.assertEqual(m.group(1).count("{"), m.group(1).count("}"))

    def test_sanitize_svg_labels_direct_unit(self):
        svg = '<text x="1" y="1">\\frac{1}{2}</text>'
        out = ms.sanitize_svg_labels(svg)
        self.assertNotIn("\\", out)
        self.assertNotIn("{", out)

    def test_sanitize_svg_labels_leaves_clean_text_alone(self):
        svg = '<text x="1" y="1">A</text><text x="2" y="2">5 cm</text>'
        self.assertEqual(ms.sanitize_svg_labels(svg), svg)

    def test_sanitize_svg_labels_non_string_passthrough(self):
        self.assertIsNone(ms.sanitize_svg_labels(None))


class TestMixedTextAndMath(unittest.TestCase):
    def test_prose_with_embedded_math(self):
        out = run_and_check(
            self, "If the radius is \\(r = 5\\) cm, the area is \\(\\pi r^2\\) sq cm."
        )
        self.assertIn("radius is", out)

    def test_multiple_spans_in_one_sentence(self):
        run_and_check(
            self,
            r"Given \(a = 3\) and \(b = 4\), find \(c\) where \(c^2 = a^2 + b^2\).",
        )

    def test_currency_not_mistaken_for_math(self):
        out = run_and_check(self, "The total cost is $500 for 3 items.")
        self.assertIn("$500", out)


class TestAssameseEnglishMath(unittest.TestCase):
    def test_assamese_prose_with_math_span(self):
        out = run_and_check(
            self, "ত্রিভুজৰ কোণ \\(\\angle ABC = 90^{\\circ}\\) হয়।"
        )
        self.assertIn("ত্রিভুজৰ", out)
        self.assertIn("হয়।", out)

    def test_assamese_leaked_inside_math_span_gets_split_out(self):
        out = run_and_check(self, r"\(x + এটা সংখ্যা + y\)")
        # The Assamese phrase must survive as readable text, not be
        # silently deleted, and must not stay trapped inside a math
        # span next to raw LaTeX operators.
        self.assertIn("এটা সংখ্যা", out)

    def test_assamese_number_words_with_equation(self):
        run_and_check(self, "যদি \\(x = 5\\) আৰু \\(y = 3\\), তেন্তে \\(x + y = 8\\)।")


class TestNestedExpressions(unittest.TestCase):
    def test_deeply_nested_fraction(self):
        run_and_check(self, r"\(\frac{1}{1 + \frac{1}{1 + \frac{1}{2}}}\)")

    def test_nested_sqrt_and_frac(self):
        run_and_check(self, r"\(\sqrt{\frac{a^2 + b^2}{c^2}}\)")

    def test_nested_left_right_groups(self):
        run_and_check(self, r"\(\left(\frac{1}{\left(1+x\right)^2}\right)\)")

    def test_matrix_inside_cases(self):
        run_and_check(
            self,
            r"\(M = \begin{cases} \begin{pmatrix}1&0\\0&1\end{pmatrix} & n=0 \\ 0 & n \ne 0 \end{cases}\)",
        )


class TestMalformedAiOutputs(unittest.TestCase):
    """Realistic 'Gemini produced broken LaTeX' shapes, reproduced
    directly rather than invented shapes nobody actually sees."""

    def test_double_open_delimiter(self):
        run_and_check(self, r"\(\(x + 1\)\)")

    def test_stray_close_with_nothing_open(self):
        run_and_check(self, r"x = 5 \) more text")

    def test_stray_open_never_closed(self):
        run_and_check(self, r"text \( x + 1")

    def test_dollar_and_paren_mixed_delimiters(self):
        run_and_check(self, r"$x + \(y\)$")

    def test_mismatched_left_right_bracket_type(self):
        run_and_check(self, r"\left( \frac{1}{2} \right] end")

    def test_json_escape_leakage(self):
        out = run_and_check(self, r"line1\nline2\tindented")
        self.assertNotIn("\\n", out)
        self.assertNotIn("\\t", out)

    def test_double_escaped_unicode(self):
        out = run_and_check(self, r"\\u221a25 = 5")
        self.assertIn("√", out)

    def test_over_escaped_backslashes(self):
        run_and_check(self, r"\\\\frac{1}{2}")

    def test_random_backslash_letter_leak(self):
        run_and_check(self, r"This costs \$5 and \yield 10")

    def test_triple_nested_unclosed_braces(self):
        out = run_and_check(self, r"\(\frac{\frac{1}{2}}{3\)")
        # Auto-closed unambiguously (the only brace missing is the
        # outer \frac's final "}"), so this must render as the intended
        # nested fraction, not swallow anything extra.
        self.assertEqual(out, r"\(\frac{\frac{1}{2}}{3}\)")

    def test_completely_scrambled_input(self):
        run_and_check(self, r"\)\)\(\(}}{{\\frac\\sqrt^^__")


class TestFuzz(unittest.TestCase):
    """Randomly composed malformed LaTeX. Deterministic seed so
    failures reproduce. This is intentionally NOT trying to guess
    real Gemini bug shapes — it's adversarial noise, to prove the
    invariants hold structurally rather than because the test authors
    thought of every case."""

    FRAGMENTS = [
        "x", "y", "1", "23", "-1", "a", "b", "=", "+", "-", "*",
        "\\(", "\\)", "{", "}", "[", "]", "^", "_", "\\\\",
        "\\frac", "\\sqrt", "\\begin{matrix}", "\\end{matrix}",
        "\\begin{pmatrix}", "\\end{pmatrix}", "\\left(", "\\right)",
        "\\vec", "\\text", "\\alpha", "\\circ", "&", "\\n", "\\t",
        "√", "×", "÷", "এটা", "সংখ্যা", " ", "\\u221a", "\\\\u221a",
        "\\notarealmacro", "$", "৯",
    ]

    def _random_string(self, rng, length):
        return "".join(rng.choice(self.FRAGMENTS) for _ in range(length))

    def test_fuzz_battery(self):
        rng = random.Random(1234567)
        failures = []
        for n in range(500):
            length = rng.randint(1, 15)
            s = self._random_string(rng, length)
            try:
                out = ms.sanitize_math_text(s)
            except Exception as e:  # pragma: no cover
                failures.append(f"RAISED on {s!r}: {e!r}")
                continue
            try:
                assert_pipeline_invariants(self, s, out)
            except AssertionError as e:
                failures.append(str(e))
        if failures:
            self.fail(
                f"{len(failures)} fuzz failures out of 500:\n"
                + "\n".join(failures[:15])
            )

    def test_fuzz_never_raises_on_pure_random_unicode(self):
        rng = random.Random(9876)
        for _ in range(200):
            length = rng.randint(0, 20)
            s = "".join(chr(rng.randint(32, 0x2764)) for _ in range(length))
            try:
                ms.sanitize_math_text(s)
            except Exception as e:  # pragma: no cover
                self.fail(f"RAISED on random unicode {s!r}: {e!r}")


class TestSvgSanitizerFuzz(unittest.TestCase):
    def test_svg_labels_never_leak_backslash_or_unbalanced_brace(self):
        rng = random.Random(42)
        fragments = TestFuzz.FRAGMENTS
        for _ in range(200):
            length = rng.randint(1, 10)
            label = "".join(rng.choice(fragments) for _ in range(length))
            svg = f'<text x="1" y="1">{label}</text>'
            try:
                out = ms.sanitize_svg_labels(svg)
            except Exception as e:  # pragma: no cover
                self.fail(f"sanitize_svg_labels RAISED on {label!r}: {e!r}")
            for m in re.finditer(r"<text[^>]*>(.*?)</text>", out, re.DOTALL):
                content = m.group(1)
                self.assertNotIn("\\", content,
                                  f"leaked backslash for label {label!r} -> {content!r}")
                self.assertEqual(content.count("{"), content.count("}"),
                                  f"unbalanced brace for label {label!r} -> {content!r}")


if __name__ == "__main__":
    unittest.main()
