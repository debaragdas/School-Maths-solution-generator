"""
Automated tests for latex_grammar.py — the recursive-descent LaTeX
grammar checker (tokenizer + arity-aware parser).

Run: python3 -m unittest test_latex_grammar -v
"""
import unittest

from latex_grammar import check_latex_grammar, tokenize


def ok(s):
    return check_latex_grammar(s)[0]


class TestValidGrammar(unittest.TestCase):
    """Every one of these is real, MathJax-renderable LaTeX and must be
    accepted."""

    def test_empty_and_plain(self):
        self.assertTrue(ok(""))
        self.assertTrue(ok("x + y = z"))
        self.assertTrue(ok("   "))

    def test_fraction(self):
        self.assertTrue(ok(r"\frac{1}{2}"))
        self.assertTrue(ok(r"\dfrac{a+b}{c}"))
        self.assertTrue(ok(r"\frac{\frac{1}{2}}{3}"))

    def test_powers_and_subscripts(self):
        self.assertTrue(ok(r"x^2 + y_1"))
        self.assertTrue(ok(r"x^{23}"))
        self.assertTrue(ok(r"a_1^2 + b_2^2 = c^2"))
        self.assertTrue(ok(r"x^{-1}"))
        self.assertTrue(ok(r"\sin^2\theta + \cos^2\theta = 1"))
        self.assertTrue(ok(r"x^\circ"))

    def test_roots(self):
        self.assertTrue(ok(r"\sqrt{2}"))
        self.assertTrue(ok(r"\sqrt[3]{8}"))
        self.assertTrue(ok(r"\sqrt{a^2 + b^2}"))

    def test_matrices(self):
        self.assertTrue(ok(r"\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}"))
        self.assertTrue(ok(r"\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}"))
        self.assertTrue(ok(r"\begin{bmatrix} a \\ b \end{bmatrix}"))
        self.assertTrue(ok(r"\begin{cases} x & x \ge 0 \\ -x & x < 0 \end{cases}"))

    def test_nested_environment(self):
        self.assertTrue(ok(
            r"\begin{matrix} \begin{matrix} 1 \end{matrix} & 2 \\ 3 & 4 \end{matrix}"
        ))

    def test_left_right(self):
        self.assertTrue(ok(r"\left( \frac{1}{2} \right]"))
        self.assertTrue(ok(r"\left[ 0, 1 \right)"))
        self.assertTrue(ok(r"\left\{ x \right\}"))
        self.assertTrue(ok(r"\left| x \right|"))
        self.assertTrue(ok(r"\left( x \right."))

    def test_decorations(self):
        self.assertTrue(ok(r"\vec{AB}"))
        self.assertTrue(ok(r"\overline{AB} = \overline{CD}"))
        self.assertTrue(ok(r"\hat{x}"))
        self.assertTrue(ok(r"\text{hello world}"))
        self.assertTrue(ok(r"\mathbb{R}"))

    def test_binom(self):
        self.assertTrue(ok(r"\binom{n}{k}"))
        self.assertTrue(ok(r"A_{1} = \binom{n}{k}"))

    def test_equations_and_sums(self):
        self.assertTrue(ok(r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}"))
        self.assertTrue(ok(r"\lim_{x \to \infty} \frac{1}{x} = 0"))
        self.assertTrue(ok(r"P(A \cup B) = P(A) + P(B) - P(A \cap B)"))

    def test_unknown_command_is_zero_arity_and_accepted(self):
        # A brand-new macro this domain has never seen gets the exact
        # same treatment as any other zero-arity symbol command.
        self.assertTrue(ok(r"\somebrandnewfuturemacro x + y"))


class TestInvalidGrammar(unittest.TestCase):
    """Every one of these is a genuine MathJax parse-error shape and
    must be rejected."""

    def test_frac_missing_argument(self):
        self.assertFalse(ok(r"\frac{1}"))
        self.assertFalse(ok(r"\frac"))

    def test_binom_missing_argument(self):
        self.assertFalse(ok(r"\binom{1}"))

    def test_sqrt_missing_argument(self):
        self.assertFalse(ok(r"\sqrt"))

    def test_dangling_superscript(self):
        self.assertFalse(ok(r"x^"))
        self.assertFalse(ok(r"x_"))

    def test_double_superscript(self):
        self.assertFalse(ok(r"x^^2"))
        self.assertFalse(ok(r"x^_2"))
        self.assertFalse(ok(r"x__2"))

    def test_decoration_missing_argument(self):
        self.assertFalse(ok(r"\vec"))
        self.assertFalse(ok(r"\text"))
        self.assertFalse(ok(r"\overline"))

    def test_mismatched_environment_name(self):
        self.assertFalse(ok(r"\begin{matrix} 1 & 2 \end{pmatrix}"))

    def test_unterminated_environment(self):
        self.assertFalse(ok(r"\begin{matrix} 1 & 2 \\ 3 & 4"))

    def test_orphan_end(self):
        self.assertFalse(ok(r"1 & 2 \end{matrix}"))

    def test_unbalanced_braces(self):
        self.assertFalse(ok(r"\frac{1}{2"))
        self.assertFalse(ok(r"x + 1}"))

    def test_left_without_delimiter(self):
        self.assertFalse(ok(r"\left"))

    def test_dangling_backslash(self):
        self.assertFalse(ok("x + 1 \\"))


class TestTokenizer(unittest.TestCase):
    def test_row_separator_is_one_token(self):
        toks = tokenize(r"1 \\ 2")
        kinds = [t[0] for t in toks]
        self.assertIn("row", kinds)
        self.assertEqual(kinds.count("row"), 1)

    def test_command_name_is_full_word(self):
        toks = tokenize(r"\frac{1}{2}")
        self.assertEqual(toks[0], ("cmd", "frac", 0))

    def test_never_raises_on_garbage(self):
        garbage = [
            "{{{{{{{{",
            "}}}}}}}}",
            "\\" * 50,
            "^^^^^^^^^^",
            "____________",
            "\\begin{\\end{\\begin{",
            "",
        ]
        for g in garbage:
            # Must return a (bool, str) tuple, never raise.
            result = check_latex_grammar(g)
            self.assertIsInstance(result, tuple)
            self.assertIsInstance(result[0], bool)


if __name__ == "__main__":
    unittest.main()
