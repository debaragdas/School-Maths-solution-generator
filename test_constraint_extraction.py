"""
Regression tests for constraint_extraction.py (Phase 3 — Constraint
Extraction Engine's deterministic validation boundary).

Pure-stdlib module, no mocking of fitz/network needed.
Run: python3 -m pytest test_constraint_extraction.py -v
"""
import unittest

import constraint_extraction as ce


class TestValidateConstraints(unittest.TestCase):
    def test_no_constraint_fields_is_valid(self):
        ok, issues = ce.validate_constraints({"diagram_type": "triangle",
                                               "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}]})
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_non_dict_spec_is_valid_noop(self):
        ok, issues = ce.validate_constraints(None)
        self.assertTrue(ok)
        ok, issues = ce.validate_constraints("not a dict")
        self.assertTrue(ok)

    def test_valid_side_lengths_and_angles(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 5, "BC": 6}, "angles_deg": {"A": 60}}
        ok, issues = ce.validate_constraints(spec)
        self.assertTrue(ok, issues)

    def test_non_dict_field_is_invalid(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "side_lengths": "5cm"}
        ok, issues = ce.validate_constraints(spec)
        self.assertFalse(ok)
        self.assertIn("must be an object", issues[0])

    def test_non_numeric_value_is_invalid(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "side_lengths": {"AB": "5 cm"}}
        ok, issues = ce.validate_constraints(spec)
        self.assertFalse(ok)
        self.assertTrue(any("not numeric" in i for i in issues))

    def test_nan_and_infinite_values_are_invalid(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": float("nan"), "BC": float("inf")}}
        ok, issues = ce.validate_constraints(spec)
        self.assertFalse(ok)
        self.assertEqual(len(issues), 2)

    def test_malformed_side_key_is_invalid(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "side_lengths": {"AAA": 5, "A": 5}}
        ok, issues = ce.validate_constraints(spec)
        self.assertFalse(ok)
        self.assertTrue(any("not a valid two-point side identifier" in i for i in issues))

    def test_reference_to_undeclared_point_is_invalid(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "side_lengths": {"AX": 5}}
        ok, issues = ce.validate_constraints(spec)
        self.assertFalse(ok)
        self.assertTrue(any("not declared" in i for i in issues))

    def test_angle_reference_to_undeclared_point_is_invalid(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "angles_deg": {"Z": 60}}
        ok, issues = ce.validate_constraints(spec)
        self.assertFalse(ok)


class TestSanitizeConstraints(unittest.TestCase):
    def test_valid_spec_passes_through_unchanged_in_content(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "side_lengths": {"AB": 5}}
        result = ce.sanitize_constraints(spec)
        self.assertEqual(result["side_lengths"], {"AB": 5.0})

    def test_does_not_mutate_the_input(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "side_lengths": {"AB": "bad", "BC": 5}}
        original_copy = dict(spec["side_lengths"])
        ce.sanitize_constraints(spec)
        self.assertEqual(spec["side_lengths"], original_copy, "input must never be mutated")

    def test_invalid_entries_are_dropped_valid_entries_kept(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 5, "BC": "not a number", "CX": 3}}
        result = ce.sanitize_constraints(spec)
        self.assertEqual(result["side_lengths"], {"AB": 5.0})

    def test_all_invalid_leaves_empty_dict_not_missing_key(self):
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "angles_deg": {"Z": 60}}
        result = ce.sanitize_constraints(spec)
        self.assertEqual(result["angles_deg"], {})

    def test_non_dict_spec_returned_unchanged(self):
        self.assertIsNone(ce.sanitize_constraints(None))
        self.assertEqual(ce.sanitize_constraints("x"), "x")

    def test_sanitized_output_always_solvable_or_none_never_crashes_solver(self):
        """End-to-end proof: whatever sanitize_constraints lets through,
        geometry_solver.py must never choke on it (it shouldn't have
        to — the whole point of this stage is a clean handoff)."""
        import geometry_solver
        spec = {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                "side_lengths": {"AB": 4, "BC": "garbage", "CA": 3, "ZZ": 99}}
        clean = ce.sanitize_constraints(spec)
        coords, reason = geometry_solver.solve_triangle(
            ["A", "B", "C"], side_lengths=clean.get("side_lengths"), angles_deg=clean.get("angles_deg"))
        # AB=4, CA=3 remain; BC was dropped (garbage) and ZZ was dropped
        # (undeclared) -> under-constrained (only 2 of 3 sides, no angle)
        # -> None is the CORRECT, expected outcome, not a crash.
        self.assertIsNone(coords)
        self.assertIsInstance(reason, str)


class TestSanitizeQuestions(unittest.TestCase):
    def test_sanitizes_every_question_in_place(self):
        questions = [
            {"question_number": "1", "diagram_spec": {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                                                        "side_lengths": {"AB": "bad"}}},
            {"question_number": "2", "diagram_spec": {"points": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
                                                        "side_lengths": {"AB": 5}}},
        ]
        ce.sanitize_questions(questions)
        self.assertEqual(questions[0]["diagram_spec"]["side_lengths"], {})
        self.assertEqual(questions[1]["diagram_spec"]["side_lengths"], {"AB": 5.0})

    def test_question_without_diagram_spec_is_untouched(self):
        questions = [{"question_number": "1"}]
        ce.sanitize_questions(questions)  # must not raise
        self.assertNotIn("diagram_spec", questions[0])

    def test_empty_or_none_questions_list_is_safe(self):
        ce.sanitize_questions([])
        ce.sanitize_questions(None)


if __name__ == "__main__":
    unittest.main()
