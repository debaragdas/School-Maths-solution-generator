"""
Regression tests for math_verifier.py — the new deterministic
(sympy-based) answer re-verification tier added to close audit item
E1 (no independent correctness check existed for any non-coordinate
answer type before this).

Pure-stdlib-plus-sympy, no PyMuPDF/genai/network needed.

Run: python3 -m unittest test_math_verifier -v
"""
import unittest

import math_verifier as mv


class TestArithmeticExtraction(unittest.TestCase):
    def test_extracts_simple_expression(self):
        self.assertEqual(mv.extract_arithmetic_expression("Simplify: 3/4 + 1/2"), "3/4 + 1/2")

    def test_extracts_with_evaluate_cue(self):
        self.assertEqual(mv.extract_arithmetic_expression("Evaluate: 2^3 + 5*(4-1)"), "2^3 + 5*(4-1)")

    def test_assamese_digits_normalized(self):
        result = mv.extract_arithmetic_expression("সমাধান কৰা: ৩ + ৪")
        self.assertEqual(result, "3 + 4")

    def test_no_cue_returns_none(self):
        self.assertIsNone(mv.extract_arithmetic_expression("Prove that triangle ABC is isosceles."))

    def test_expression_with_variable_returns_none(self):
        # this is a "solve" case, not "evaluate" -- must not be treated
        # as a bare arithmetic expression
        self.assertIsNone(mv.extract_arithmetic_expression("Simplify: 3x + 4x"))

    def test_expression_with_unsafe_characters_returns_none(self):
        self.assertIsNone(mv.extract_arithmetic_expression("Simplify: 3/4 + √2"))


class TestLinearEquationExtraction(unittest.TestCase):
    def test_extracts_single_variable_equation(self):
        result = mv.extract_linear_equation("Solve for x: 2x + 3 = 11")
        self.assertEqual(result, ("2x + 3", "11"))

    def test_no_solve_cue_returns_none(self):
        self.assertIsNone(mv.extract_linear_equation("If 2x + 3 = 11, find the value of x + 1."))

    def test_two_variables_returns_none(self):
        self.assertIsNone(mv.extract_linear_equation("Solve: x + y = 10"))

    def test_zero_variables_returns_none(self):
        self.assertIsNone(mv.extract_linear_equation("Solve: 3 + 4 = 7"))


class TestFinalAnswerExtraction(unittest.TestCase):
    def test_single_number(self):
        self.assertEqual(mv.extract_single_final_number("The answer is 5/4."), __import__("sympy").Rational(5, 4))

    def test_ambiguous_multiple_numbers_returns_none(self):
        self.assertIsNone(mv.extract_single_final_number("The two roots are 2 and 3."))

    def test_variable_value_extraction(self):
        import sympy
        self.assertEqual(mv.extract_variable_value("Therefore x = 4.", "x"), sympy.Rational(4))

    def test_variable_value_wrong_variable_returns_none(self):
        self.assertIsNone(mv.extract_variable_value("Therefore y = 4.", "x"))


class TestVerifyArithmetic(unittest.TestCase):
    def test_correct_answer_confirmed(self):
        result = mv.verify_question({"question_text": "Simplify: 3/4 + 1/2", "final_answer": "The value is 5/4."})
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["check"], "arithmetic")

    def test_wrong_answer_contradicted(self):
        result = mv.verify_question({"question_text": "Simplify: 3/4 + 1/2", "final_answer": "The value is 2."})
        self.assertEqual(result["status"], "contradicted")
        self.assertIn("5/4", result["detail"])

    def test_power_uses_exponent_not_xor(self):
        """Regression test for the XOR-vs-power parsing bug found while
        writing this module: sympy's default parser treats a bare '^'
        as bitwise XOR, not exponentiation. 2^3 must independently
        recompute to 8 (2**3), not 1 (2 XOR 3)."""
        result = mv.verify_question({"question_text": "Evaluate: 2^3 + 5*(4-1)", "final_answer": "Answer = 23"})
        self.assertEqual(result["status"], "confirmed", result["detail"])
        result_wrong = mv.verify_question({"question_text": "Evaluate: 2^3", "final_answer": "Answer = 1"})
        self.assertEqual(result_wrong["status"], "contradicted")
        self.assertIn("8", result_wrong["detail"])

    def test_negative_and_decimal_numbers(self):
        result = mv.verify_question({"question_text": "Evaluate: 5.5 - 2.25", "final_answer": "3.25"})
        self.assertEqual(result["status"], "confirmed", result["detail"])

    def test_not_applicable_for_proof_question(self):
        result = mv.verify_question({
            "question_text": "Prove that the diagonals of a rhombus bisect each other at right angles.",
            "final_answer": "Hence proved.",
        })
        self.assertEqual(result["status"], "not_applicable")


class TestVerifyLinearEquation(unittest.TestCase):
    def test_correct_solution_confirmed(self):
        result = mv.verify_question({"question_text": "Solve for x: 2x + 3 = 11", "final_answer": "x = 4"})
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["check"], "linear_equation")

    def test_wrong_solution_contradicted_with_correct_value_hint(self):
        result = mv.verify_question({"question_text": "Solve for x: 2x + 3 = 11", "final_answer": "x = 5"})
        self.assertEqual(result["status"], "contradicted")
        self.assertIn("x=4", result["detail"])

    def test_assamese_digits_equation(self):
        result = mv.verify_question({"question_text": "সমাধান কৰা: ২x + ৩ = ১১", "final_answer": "x = ৪"})
        self.assertEqual(result["status"], "confirmed", result["detail"])

    def test_multivariable_system_not_applicable(self):
        result = mv.verify_question({"question_text": "Solve: x + y = 10", "final_answer": "x=4, y=6"})
        self.assertEqual(result["status"], "not_applicable")

    def test_negative_coefficient_equation(self):
        result = mv.verify_question({"question_text": "Solve for x: 5 - x = 2", "final_answer": "x = 3"})
        self.assertEqual(result["status"], "confirmed", result["detail"])


class TestVerifyQuestionsBatch(unittest.TestCase):
    def test_contradicted_question_is_flagged_not_rewritten(self):
        """Matches the fail-safe policy used by the coordinate tie-break
        fix: a contradiction must set needs_review and leave
        final_answer completely untouched — never auto-correct."""
        q = {"question_text": "Simplify: 3/4 + 1/2", "final_answer": "The value is 2."}
        confirmed, contradicted = mv.verify_questions([q])
        self.assertEqual(contradicted, 1)
        self.assertEqual(confirmed, 0)
        self.assertTrue(q["needs_review"])
        self.assertIn("The value is 2.", q["final_answer"])  # untouched
        self.assertTrue(any("5/4" in note for note in q["review_notes"]))

    def test_confirmed_question_is_not_flagged(self):
        q = {"question_text": "Simplify: 3/4 + 1/2", "final_answer": "The value is 5/4."}
        confirmed, contradicted = mv.verify_questions([q])
        self.assertEqual(confirmed, 1)
        self.assertEqual(contradicted, 0)
        self.assertNotIn("needs_review", q)

    def test_not_applicable_question_is_not_flagged(self):
        q = {"question_text": "Prove that triangle ABC is isosceles.", "final_answer": "Hence proved."}
        confirmed, contradicted = mv.verify_questions([q])
        self.assertEqual(confirmed, 0)
        self.assertEqual(contradicted, 0)
        self.assertNotIn("needs_review", q)

    def test_mixed_batch_only_flags_the_contradicted_one(self):
        q_ok = {"question_text": "Simplify: 1/2 + 1/2", "final_answer": "1"}
        q_bad = {"question_text": "Solve for x: 3x = 9", "final_answer": "x = 2"}
        q_na = {"question_text": "Describe the properties of a rhombus.", "final_answer": "..."}
        mv.verify_questions([q_ok, q_bad, q_na])
        self.assertNotIn("needs_review", q_ok)
        self.assertTrue(q_bad.get("needs_review"))
        self.assertNotIn("needs_review", q_na)

    def test_never_raises_on_malformed_question_dict(self):
        # missing keys entirely -- must degrade gracefully, not crash
        # the whole batch
        try:
            mv.verify_questions([{}, {"question_text": None, "final_answer": None}])
        except Exception as e:
            self.fail(f"verify_questions raised on malformed input: {e}")


if __name__ == "__main__":
    unittest.main()
