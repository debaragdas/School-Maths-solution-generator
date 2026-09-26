"""
Regression tests for solver.py: the existing JSON-repair/math-
normalization helpers, and the coordinate-verification tie-break fix
from the V7 engineering audit (item E2 — a single disagreeing vision
call used to overwrite the original answer outright; fixed to a
proper best-of-3 majority vote with a fail-safe "needs_review" flag
on a genuine 3-way split).

`google.genai` and `fitz` are stubbed before import — this sandbox has
neither installed and no network to install them, and solver.py only
needs their names to exist at import time for these tests (which don't
exercise the real network call path — _read_point_coords itself is
monkeypatched to control what each "vote" returns).

Run: python3 -m unittest test_solver -v
"""
import sys
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
import solver  # noqa: E402  (must come after the fakes are installed)


class TestJsonRepair(unittest.TestCase):
    def test_repairs_unescaped_backslash_in_latex(self):
        # a raw, invalid backslash before a non-escape character (LaTeX
        # \frac) must become \\frac so json.loads can parse it
        broken = r'{"final_answer": "\frac{1}{2}"}'
        repaired = solver._repair_invalid_json_escapes(broken)
        import json
        parsed = json.loads(repaired)
        self.assertIn("frac", parsed["final_answer"])

    def test_leaves_genuinely_valid_escapes_alone(self):
        # only '"', '\\', '/' are "valid" by this repair pass's own
        # definition (_VALID_JSON_ESCAPE_CHARS) — \n is DELIBERATELY
        # NOT in that set, so it gets doubled (kept as literal text,
        # not turned into a real newline by json.loads) so the later
        # _normalize_math_text pass can disambiguate a real line break
        # from a LaTeX macro like \neq post-parse. An escaped quote is
        # the case that's genuinely left untouched.
        valid = r'{"text": "she said \"hello\""}'
        import json
        repaired = solver._repair_invalid_json_escapes(valid)
        parsed = json.loads(repaired)
        self.assertEqual(parsed["text"], 'she said "hello"')

    def test_strip_json_fences_removes_code_block_markers(self):
        text = "```json\n{\"a\": 1}\n```"
        self.assertEqual(solver._strip_json_fences(text), '{"a": 1}')


class TestParseGeminiJsonResponse(unittest.TestCase):
    """PRODUCTION-AUDIT FIX (final pre-launch round) — root cause of
    "fractions are broken" / "broken LaTeX" reports: \\b \\f \\n \\r \\t
    are all escapes JSON ITSELF considers valid, so a raw, single
    backslash before a LaTeX macro starting with one of those letters
    (\\frac, \\because, \\boxed, \\beta, \\neq, \\right, \\text, \\tan,
    \\therefore, \\triangle...) never raised a JSONDecodeError — it
    silently parsed into a control character eating the macro's first
    letter, and the old two-stage "repair only after a raised error"
    approach could never catch it because nothing ever raised. These
    tests reproduce that and confirm parse_gemini_json_response (which
    always repairs BEFORE the first parse) fixes it."""

    def test_frac_macro_survives_intact(self):
        raw = r'{"final_answer": "x = \frac{1}{2}"}'
        parsed = solver.parse_gemini_json_response(raw, "test")
        self.assertEqual(parsed["final_answer"], r"x = \frac{1}{2}")

    def test_because_boxed_binom_beta_all_survive(self):
        raw = (r'{"steps": ["\because AB = AC", "\boxed{x=5}", '
               r'"\binom{5}{2}", "\beta + \theta = 90"]}')
        parsed = solver.parse_gemini_json_response(raw, "test")
        joined = " ".join(parsed["steps"])
        for macro in (r"\because", r"\boxed", r"\binom", r"\beta", r"\theta"):
            self.assertIn(macro, joined)

    def test_right_rightarrow_text_therefore_triangle_tan_all_survive(self):
        raw = (r'{"final_answer": "\triangle ABC \rightarrow \therefore '
               r'\tan\theta \text{is defined}"}')
        parsed = solver.parse_gemini_json_response(raw, "test")
        for macro in (r"\triangle", r"\rightarrow", r"\therefore", r"\tan", r"\text"):
            self.assertIn(macro, parsed["final_answer"])

    def test_no_form_feed_or_control_chars_leak_into_output(self):
        raw = r'{"final_answer": "\frac{1}{2} + \frac{1}{3}"}'
        parsed = solver.parse_gemini_json_response(raw, "test")
        self.assertNotIn("\x0c", parsed["final_answer"])  # form feed (old \f corruption)
        self.assertNotIn("\x08", parsed["final_answer"])  # backspace (old \b corruption)

    def test_correctly_double_escaped_input_is_unaffected(self):
        # Gemini writing PROPER JSON (double backslash) must still work
        # identically — the always-on repair pass is a no-op for it.
        raw = r'{"final_answer": "x = \\frac{1}{2}"}'
        parsed = solver.parse_gemini_json_response(raw, "test")
        self.assertEqual(parsed["final_answer"], r"x = \frac{1}{2}")

    def test_genuinely_invalid_json_still_raises_valueerror(self):
        with self.assertRaises(ValueError):
            solver.parse_gemini_json_response("this is not JSON at all {{{", "test")

    def test_full_pipeline_fraction_survives_end_to_end(self):
        # Proves the fix all the way through _normalize_math_text too —
        # not just raw parsing. The fraction must ALSO end up wrapped in
        # \( \) (V39 addition: _wrap_bare_latex) since surviving JSON
        # parsing intact is necessary but not sufficient — MathJax still
        # can't render un-delimited LaTeX at all (see test_solver.py's
        # TestWrapBareLatex class for that fix's own dedicated tests).
        raw = r'{"final_answer": "x = \frac{1}{2}"}'
        parsed = solver.parse_gemini_json_response(raw, "test")
        normalized = solver._normalize_math_text(parsed["final_answer"])
        self.assertEqual(normalized, r"x = \(\frac{1}{2}\)")


class TestMathTextNormalization(unittest.TestCase):
    def test_distinguishes_real_newline_from_latex_macro(self):
        # a literal \n followed by something that ISN'T a lowercase
        # Latin letter (here, digit/punctuation as in a numbered
        # sub-part list) is a real line break and becomes <br>; \neq
        # must NOT be touched (still the intact LaTeX macro, since 'e'
        # immediately after 'n' reads as a continuing macro name).
        text = solver._normalize_math_text(r"solve as follows:\n(ii) x \neq y")
        self.assertIn("<br>", text)
        self.assertIn(r"\neq", text)

    def test_normalize_all_text_fields_recurses_into_steps_list(self):
        parsed = {"questions": [{"steps": [r"step one:\n(2) step two", r"x \neq y"], "final_answer": r"answer:\n5"}]}
        solver._normalize_all_text_fields(parsed)
        q = parsed["questions"][0]
        self.assertIn("<br>", q["steps"][0])
        self.assertIn(r"\neq", q["steps"][1])
        self.assertIn("<br>", q["final_answer"])


class TestTabAndNewlineTableFix(unittest.TestCase):
    """V40 FIX — root cause of raw "\\n"/"\\t" showing up verbatim in
    generated PDFs. Reproduces the exact reported case: a table-shaped
    "given" field where a table cell's variable name ("x", "y") is
    itself a lowercase Latin letter, which the OLD "any letter after
    the backslash = must be a macro" rule mistook for the start of a
    LaTeX macro (like \\nabla) and left completely undecoded."""

    def test_raw_table_example_no_longer_has_literal_backslash_n_or_t(self):
        text = r"\nx\t-2\t-1\t0\t1\t3\ny\t8\t7\t-1.25\t3\t-1"
        result = solver._normalize_math_text(text)
        self.assertNotIn(r"\n", result)
        self.assertNotIn(r"\t", result)
        self.assertIn("<br>", result)
        self.assertIn("&nbsp;", result)

    def test_table_variable_name_x_not_mistaken_for_a_macro(self):
        # Old bug: "\nx" was misread as the start of a LaTeX macro
        # because 'x' is a lowercase letter, so it was never converted.
        result = solver._normalize_math_text(r"তালিকা:\nx\t1\t2")
        self.assertIn("<br>", result)
        self.assertNotIn(r"\nx", result)
        self.assertNotIn(r"\t", result)

    def test_real_tan_text_theta_times_triangle_therefore_macros_survive(self):
        for macro in (r"\tan\theta", r"\text{is defined}", r"\theta = 90",
                      r"\times 2", r"\triangle ABC", r"\therefore x = 5"):
            result = solver._normalize_math_text(macro)
            self.assertIn(macro.split()[0].split("{")[0], result)
            self.assertNotIn("&nbsp;", result)

    def test_ne_macro_still_survives_alongside_generalized_fix(self):
        result = solver._normalize_math_text(r"x \ne y")
        self.assertIn(r"\ne", result)
        self.assertNotIn("<br>", result)

    def test_literal_tab_before_digit_becomes_spacing_not_raw_text(self):
        result = solver._normalize_math_text(r"5\t10\t15")
        self.assertNotIn(r"\t", result)
        self.assertIn("&nbsp;", result)


class TestStrayUnicodeEscapeFix(unittest.TestCase):
    """PRODUCTION-AUDIT FIX (final pre-launch round) — root cause of
    "PDF shows \\u221a / \\u03c0 instead of √ / π": Gemini sometimes
    double-backslashes an already-correct "\\u221a" JSON escape into
    "\\\\u221a", which json.loads() then legitimately decodes into the
    six-character PLAIN-TEXT string "\\u221a" (one backslash + u221a) —
    valid as far as JSON parsing goes, but never re-interpreted as a
    unicode escape by anything downstream. _normalize_math_text now
    catches and decodes any lone \\uXXXX left after its existing
    backslash-collapse pass."""

    def test_stray_sqrt_escape_decoded(self):
        # Simulates the over-escaped raw JSON case: after json.loads(),
        # the Python string literally contains "\\\\u221a" (i.e. two
        # backslash characters followed by u221a) as its own two-
        # backslash run, which the existing collapse reduces to one
        # BEFORE this fix's decode step runs.
        text = solver._normalize_math_text("\\\\u221a x")
        self.assertEqual(text, "√ x")

    def test_stray_pi_escape_decoded(self):
        text = solver._normalize_math_text("ব্যাসাৰ্ধ = \\\\u03c0r")
        self.assertIn("π", text)
        self.assertNotIn("u03c0", text)

    def test_already_correct_single_escape_from_json_parsing_is_untouched_upstream(self):
        # A CORRECTLY single-escaped "\u221a" in raw JSON is decoded by
        # json.loads() itself into the real "√" character LONG before
        # _normalize_math_text ever runs — so by the time text reaches
        # this function in normal operation, it's already "√", not
        # backslash-u-two-two-one-a. This test only documents that
        # normal (non-broken) case is a no-op here.
        self.assertEqual(solver._normalize_math_text("√ x"), "√ x")

    def test_real_latex_macros_starting_with_u_are_never_touched(self):
        for macro in (r"\underline{ABC}", r"\uparrow", r"\uplus", r"\unlhd", r"\upsilon"):
            result = solver._normalize_math_text(macro)
            # V39: bare LaTeX is now also wrapped in \( \) by _wrap_bare_latex
            # — that's correct, separate behavior. What THIS test actually
            # checks is narrower and still holds: the macro text itself
            # must survive completely intact, never decoded/mangled as a
            # stray \uXXXX unicode escape.
            self.assertEqual(result, f"\\({macro}\\)")

    def test_decodes_correctly_even_immediately_followed_by_digit_or_letter(self):
        # A math symbol with no space before the next character is
        # extremely common ("√4", "√x") — must still decode correctly.
        self.assertEqual(solver._normalize_math_text("\\\\u221a4 = 2"), "√4 = 2")
        self.assertEqual(solver._normalize_math_text("\\\\u221ax"), "√x")

    def test_multiple_stray_escapes_in_one_string_all_decoded(self):
        text = solver._normalize_math_text("\\\\u221a4 = 2, \\\\u03c0 \\\\u2248 3.14")
        self.assertIn("√4 = 2", text)
        self.assertIn("π", text)
        self.assertIn("≈", text)

    def test_applies_inside_full_field_normalization_pipeline(self):
        parsed = {"questions": [{"final_answer": "উত্তৰ: \\\\u221a25 = 5"}]}
        solver._normalize_all_text_fields(parsed)
        self.assertEqual(parsed["questions"][0]["final_answer"], "উত্তৰ: √25 = 5")


class TestWrapBareLatex(unittest.TestCase):
    """V39 fix — root cause of 'broken fraction' / 'raw LaTeX shown as
    plain text' reports, DISTINCT from the JSON-escape-corruption bug
    _repair_invalid_json_escapes already fixes (that one mangles the
    backslash itself; this one is Gemini simply forgetting the \\( \\)
    wrapper around an otherwise-perfectly-correct expression, e.g.
    "32^{\\frac{2}{5}}" appearing bare in the middle of a sentence,
    which MathJax then never touches at all)."""

    def test_exact_reported_example(self):
        text = "প্ৰদত্ত অভিব্যক্তিটো হ'ল 32^{\\frac{2}{5}}।"
        result = solver._normalize_math_text(text)
        self.assertEqual(result, "প্ৰদত্ত অভিব্যক্তিটো হ'ল \\(32^{\\frac{2}{5}}\\)।")

    def test_already_delimited_math_is_never_double_wrapped(self):
        text = "চূড়ান্ত উত্তৰ: \\(32^{\\frac{2}{5}}\\) ৰ মান হ'ল 4।"
        self.assertEqual(solver._normalize_math_text(text), text)

    def test_multiple_already_delimited_spans_survive_untouched(self):
        text = "\\(x + y = 10\\) আৰু \\(x - y = 2\\)।"
        self.assertEqual(solver._normalize_math_text(text), text)

    def test_plain_equality_with_no_latex_left_untouched(self):
        # This project intentionally leaves simple equalities like this
        # unwrapped elsewhere by design — must not start forcing them
        # into math mode as a side effect of this fix.
        text = "OA = 2 একক।"
        self.assertEqual(solver._normalize_math_text(text), text)

    def test_plain_number_with_unit_left_untouched(self):
        text = "এটা বৰ্গৰ কালি 81 বৰ্গ চে.মি.।"
        self.assertEqual(solver._normalize_math_text(text), text)

    def test_parenthesized_compound_expression_stays_one_clean_span(self):
        # Regression for a real bug caught during development: without
        # '(' ')' in the allowed run characters, "(2^5)^{\frac{2}{5}}"
        # was split into TWO separately-wrapped fragments with a stray
        # bare ')' left outside — worse than not wrapping at all.
        text = "গতিকে, 32^{\\frac{2}{5}} = (2^5)^{\\frac{2}{5}}।"
        result = solver._normalize_math_text(text)
        self.assertEqual(result,
                         "গতিকে, \\(32^{\\frac{2}{5}}\\) = \\((2^5)^{\\frac{2}{5}}\\)।")

    def test_trailing_comma_placed_outside_the_wrap(self):
        # Regression for a real bug caught during development:
        # "2^5," was being wrapped as "\(2^5,\)" (comma pulled inside)
        # instead of "\(2^5\)," (comma is sentence punctuation, not
        # part of the expression).
        text = "...প্রথমে 2^5, যিটো ঠিক আছে।"
        result = solver._normalize_math_text(text)
        self.assertEqual(result, "...প্রথমে \\(2^5\\), যিটো ঠিক আছে।")

    def test_sentence_with_no_math_at_all_untouched(self):
        text = "কোনো সমীকৰণ নাই, কেৱল সাধাৰণ বাক্য।"
        self.assertEqual(solver._normalize_math_text(text), text)

    def test_non_string_input_passed_through(self):
        self.assertEqual(solver._normalize_math_text(None), None)
        self.assertEqual(solver._normalize_math_text(42), 42)


class TestConstructionInstrumentsStaysAList(unittest.TestCase):
    """Regression test for a real production bug: _coerce_instruments_field
    was documented to return a list[str] but actually returned a single
    ", "-joined STRING. The template does
    `{% for tool in q.construction_instruments %}<span class="instrument-chip">...`
    — iterating a *string* walks it character by character (Jinja/Python
    strings are iterable per-character), so every individual glyph,
    including isolated Bengali/Assamese combining vowel signs that can't
    render standalone, got its own boxed chip — exactly the broken
    dotted-circle boxes seen in a real generated PDF's "প্ৰয়োজনীয় সঁজুলি:"
    (required tools) line."""

    def test_returns_a_list_not_a_joined_string(self):
        result = solver._coerce_instruments_field(
            ["স্কেল", "কম্পাছ", "পেন্সিল", "চাঁদমাৰি"])
        self.assertIsInstance(result, list)
        self.assertEqual(result, ["স্কেল", "কম্পাছ", "পেন্সিল", "চাঁদমাৰি"])

    def test_iterating_the_result_yields_whole_tools_not_characters(self):
        result = solver._coerce_instruments_field(["কম্পাছ", "স্কেল"])
        # This is exactly what the Jinja `{% for tool in ... %}` loop does.
        chips = [tool for tool in result]
        self.assertEqual(chips, ["কম্পাছ", "স্কেল"])
        # None of the old bug's single-character fragments.
        self.assertNotIn("ো", chips)
        self.assertNotIn("্", chips)

    def test_dedupes_and_drops_blanks_still_as_a_list(self):
        result = solver._coerce_instruments_field(["স্কেল", "স্কেল", "  ", None, "কম্পাছ"])
        self.assertEqual(result, ["স্কেল", "কম্পাছ"])

    def test_empty_or_none_stays_none(self):
        self.assertIsNone(solver._coerce_instruments_field(None))
        self.assertIsNone(solver._coerce_instruments_field([]))
        self.assertIsNone(solver._coerce_instruments_field(["", "  "]))

    def test_bare_string_input_still_wrapped_as_single_item_list(self):
        result = solver._coerce_instruments_field("স্কেল")
        self.assertEqual(result, ["স্কেল"])


class TestTextFieldDictLeakageFix(unittest.TestCase):
    """Regression tests for a real bug found in a user-generated PDF:
    Gemini sometimes sends 'given'/'required' as a structured object
    instead of a plain string despite the schema instructing otherwise,
    and the raw Python dict repr (e.g. "{'title': 'প্ৰদত্ত', ...}")
    was rendering directly into the published PDF."""

    def test_exact_reported_shape_produces_no_raw_dict_repr(self):
        parsed = {"questions": [{
            "given": {"title": "প্ৰদত্ত", "statements": [
                "এটা ত্ৰিভুজ ABC ৰ ভূমি BC = 7 চে.মি.",
                "এটা ভূমি কোণ ∠B = 75°",
                "আন দুটা বাহুৰ সমষ্টি AB + AC = 13 চে.মি.",
            ]},
            "required": {"title": "অংকন কৰিবলৈ", "statement": "ABC ত্ৰিভুজটো অংকন কৰা।"},
        }]}
        solver._normalize_all_text_fields(parsed)
        q = parsed["questions"][0]
        self.assertNotIn("{'", q["given"])
        self.assertNotIn("{'", q["required"])
        self.assertIn("BC = 7", q["given"])
        self.assertIn("ABC ত্ৰিভুজটো অংকন কৰা", q["required"])
        # PRODUCTION-AUDIT FIX: the structured object's "title" (e.g.
        # "প্ৰদত্ত") must NOT be baked into the flattened text — the
        # Jinja template already prints its own static "প্ৰদত্ত:" label
        # before this value, so including the title here produced a
        # real, confirmed double-labeled PDF ("প্ৰদত্ত: প্ৰদত্ত। ...").
        self.assertFalse(q["given"].startswith("প্ৰদত্ত"),
                          f"'given' must not start with a duplicated title: {q['given']!r}")
        self.assertFalse(q["required"].startswith("অংকন কৰিবলৈ"),
                          f"'required' must not start with a duplicated title: {q['required']!r}")

    def test_plain_string_fields_are_left_alone(self):
        parsed = {"questions": [{"given": "প্ৰদত্ত: x = 5", "required": "x নিৰ্ণয় কৰা"}]}
        solver._normalize_all_text_fields(parsed)
        q = parsed["questions"][0]
        self.assertEqual(q["given"], "প্ৰদত্ত: x = 5")
        self.assertEqual(q["required"], "x নিৰ্ণয় কৰা")

    def test_bare_list_of_strings_is_joined_not_repr_leaked(self):
        parsed = {"questions": [{"given": ["fact one", "fact two"]}]}
        solver._normalize_all_text_fields(parsed)
        self.assertNotIn("[", parsed["questions"][0]["given"])
        self.assertIn("fact one", parsed["questions"][0]["given"])
        self.assertIn("fact two", parsed["questions"][0]["given"])

    def test_unknown_dict_shape_never_leaks_raw_repr(self):
        """Even a dict shape this module has never seen before must not
        surface a raw {'key': 'value'} repr -- degrade gracefully."""
        parsed = {"questions": [{"given": {"some_unexpected_key": "actual content here"}}]}
        solver._normalize_all_text_fields(parsed)
        result = parsed["questions"][0]["given"]
        self.assertNotIn("{'", result)
        self.assertIn("actual content here", result)

    def test_steps_list_containing_a_dict_item_is_flattened_too(self):
        parsed = {"questions": [{"steps": [{"statement": "step as an object"}, "normal step"]}]}
        solver._normalize_all_text_fields(parsed)
        steps = parsed["questions"][0]["steps"]
        self.assertNotIn("{'", steps[0])
        self.assertIn("step as an object", steps[0])
        self.assertEqual(steps[1], "normal step")

    def test_none_value_becomes_empty_string_not_the_word_none(self):
        parsed = {"questions": [{"given": None}]}
        solver._normalize_all_text_fields(parsed)
        self.assertEqual(parsed["questions"][0]["given"], "")


class TestRetryableErrorDetection(unittest.TestCase):
    def test_429_is_retryable(self):
        class FakeErr(Exception):
            pass
        e = FakeErr("429 RESOURCE_EXHAUSTED: quota exceeded")
        self.assertTrue(solver._is_retryable_api_error(e))

    def test_auth_error_is_not_retryable(self):
        class FakeErr(Exception):
            pass
        e = FakeErr("401 UNAUTHENTICATED: invalid credentials")
        self.assertFalse(solver._is_retryable_api_error(e))


class TestCoordinateVerificationTieBreak(unittest.TestCase):
    """The core regression coverage for the tie-break fix itself."""

    def _question(self, point_label="A", answer_xy="(3,4)"):
        return {
            "book_diagram_base64": "ZmFrZQ==",  # base64("fake")
            "question_text": f"Write the coordinates of {point_label} point shown in the figure.",
            "given": "",
            "final_answer": f"The coordinates of point {point_label} are {answer_xy}.",
            "steps": [f"Reading the graph, point {point_label} is at {answer_xy}."],
        }

    def test_agreement_on_first_check_makes_no_change(self):
        q = self._question(answer_xy="(3,4)")
        with mock.patch("solver._read_point_coords", return_value=(3, 4)) as read_mock:
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        self.assertEqual(read_mock.call_count, 1, "must not fire a tie-break call when the first read already agrees")
        self.assertIn("(3,4)", q["final_answer"])
        self.assertNotIn("needs_review", q)

    def test_tie_break_confirms_reread_corrects_answer(self):
        """original=(3,4) [WRONG], reread=(5,6), tie-break=(5,6) -> 2-of-3
        majority for the reread -> answer IS corrected."""
        q = self._question(answer_xy="(3,4)")
        with mock.patch("solver._read_point_coords", side_effect=[(5, 6), (5, 6)]) as read_mock:
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        self.assertEqual(read_mock.call_count, 2)
        self.assertIn("(5,6)", q["final_answer"])
        self.assertNotIn("(3,4)", q["final_answer"])
        self.assertNotIn("needs_review", q)

    def test_tie_break_confirms_original_does_not_overwrite(self):
        """original=(3,4) [ACTUALLY RIGHT], reread=(5,6) [the outlier],
        tie-break=(3,4) -> 2-of-3 majority for the ORIGINAL -> must NOT
        be overwritten. This is the exact bug the old code had: it would
        have blindly replaced (3,4) with (5,6) on the first disagreement
        alone, with no tie-break at all."""
        q = self._question(answer_xy="(3,4)")
        with mock.patch("solver._read_point_coords", side_effect=[(5, 6), (3, 4)]) as read_mock:
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        self.assertEqual(read_mock.call_count, 2)
        self.assertIn("(3,4)", q["final_answer"])
        self.assertNotIn("needs_review", q)

    def test_three_way_split_flags_for_review_instead_of_guessing(self):
        """original=(3,4), reread=(5,6), tie-break=(7,8) — three genuinely
        different readings, no majority anywhere. Must NOT pick any of
        them automatically; must flag needs_review and leave the
        original untouched (fail-safe, not fail-guess)."""
        q = self._question(answer_xy="(3,4)")
        with mock.patch("solver._read_point_coords", side_effect=[(5, 6), (7, 8)]):
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        self.assertIn("(3,4)", q["final_answer"], "original must be left untouched on a 3-way split")
        self.assertTrue(q.get("needs_review"))
        self.assertTrue(q.get("review_notes"))

    def test_tie_break_call_failing_also_flags_rather_than_guesses(self):
        """If the tie-break call itself fails (returns None), that's
        equivalent to 'no majority reached' — must flag, not guess."""
        q = self._question(answer_xy="(3,4)")
        with mock.patch("solver._read_point_coords", side_effect=[(5, 6), None]):
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        self.assertIn("(3,4)", q["final_answer"])
        self.assertTrue(q.get("needs_review"))

    def test_first_read_call_failing_leaves_answer_untouched_no_flag(self):
        """If even the FIRST verification call fails outright (not a
        disagreement, just a failure), this is the pre-existing
        fail-open behavior: leave the original answer as-is, and don't
        raise or flag — there was no actual disagreement detected."""
        q = self._question(answer_xy="(3,4)")
        with mock.patch("solver._read_point_coords", return_value=None) as read_mock:
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        self.assertEqual(read_mock.call_count, 1)
        self.assertIn("(3,4)", q["final_answer"])
        self.assertNotIn("needs_review", q)

    def test_non_coordinate_question_is_never_touched(self):
        q = {"question_text": "Simplify 3/4 + 1/2", "final_answer": "5/4"}
        with mock.patch("solver._read_point_coords") as read_mock:
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        read_mock.assert_not_called()

    def test_question_without_book_diagram_is_never_touched(self):
        q = self._question()
        del q["book_diagram_base64"]
        with mock.patch("solver._read_point_coords") as read_mock:
            solver._verify_coordinate_answers([q], class_name=9, exercise_label="3.2")
        read_mock.assert_not_called()


class TestSolveExerciseWiresInMathVerifier(unittest.TestCase):
    """Integration-style test proving math_verifier.verify_questions is
    actually called from the real solve_exercise() pipeline (not just
    an orphaned module) — mocks the Gemini call layer to return a
    canned response containing one deliberately-wrong arithmetic
    answer, and checks the returned parsed dict comes back flagged."""

    def test_wrong_arithmetic_answer_gets_flagged_end_to_end(self):
        import json as _json
        fake_response_json = _json.dumps({
            "questions": [
                {"question_number": 1, "question_text": "Simplify: 3/4 + 1/2",
                 "given": "", "required": "the simplified value", "steps": ["Step 1"],
                 "final_answer": "The value is 2."},  # deliberately WRONG (correct is 5/4)
            ]
        })

        class _FakeResp:
            text = fake_response_json

        with mock.patch("solver._call_gemini", return_value=_FakeResp()), \
             mock.patch("solver._attach_book_diagrams"), \
             mock.patch("solver._verify_coordinate_answers"), \
             mock.patch("builtins.open", mock.mock_open(read_data=b"%PDF-fake")):
            result = solver.solve_exercise("fake.pdf", class_name=9, chapter=1, exercise_label="1.1")

        q = result["questions"][0]
        self.assertTrue(q.get("needs_review"), "wrong arithmetic answer must be flagged by the wired-in math_verifier")
        self.assertIn("The value is 2.", q["final_answer"], "must not be silently rewritten")


class TestStrictBookFigureMatching(unittest.TestCase):
    """Regression tests for the V8 strict book-figure-matching policy.

    ROOT CAUSE this replaced: the old page-order fallback attached a
    book image to a question with NO figure-number reference of its
    own, guessing purely from position in the extracted-image pool.
    Confirmed on a real generated PDF: Exercise 14.3 Q6 (comparing two
    sections' frequency polygons — a question with no figure of its
    own to draw from) was matched to Figure 14.9, an unrelated generic
    median-illustration diagram from the chapter's theory section.

    New policy: a book figure is ONLY ever attached when the
    question's own text cites an exact figure number that matches an
    extracted image's own caption. No other path exists."""

    def _img(self, figure_ref, source="raster", tag=b"IMG"):
        return {"bytes": tag, "ext": "png", "figure_ref": figure_ref, "source": source}

    def test_question_with_no_figure_reference_gets_no_book_diagram(self):
        # Simulates Exercise 14.3 Q6: has_book_diagram=True (Gemini saw
        # SOME figure nearby in the source PDF) but the question's own
        # text names no figure number at all.
        questions = [{"question_number": 6,
            "question_text": "একেখন লেখতে উভয় শাখাৰ ছাত্র/ছাত্ৰীৰ নম্বৰসমূহ দুটা বাৰংবাৰতা বহুভুজৰদ্বাৰা "
                              "উপস্থাপন কৰি দেখুওৱা।",
            "given": "দুটা শাখা A আৰু B ৰ ছাত্র-ছাত্ৰীৰ নম্বৰৰ বাৰংবাৰতা বিভাজন তালিকা।",
        }]
        fake_images = [self._img("14.9", tag=b"UNRELATED_THEORY_FIGURE")]
        with mock.patch("solver.get_verified_figures", return_value=fake_images):
            solver._attach_book_diagrams(questions, "fake_exercise.pdf", "14.3", class_name=9)

        self.assertIsNone(questions[0]["book_diagram_base64"],
                           "a question with no figure-number reference of its own must NEVER "
                           "receive a book diagram via positional/page-order guessing")

    def test_exact_figure_number_match_is_still_attached(self):
        questions = [{"question_number": 1,
            "question_text": "চিত্ৰ 7.30 চোৱা। দেখুওৱা যে AB = AC।",
            "given": "",
        }]
        fake_images = [self._img("7.30", tag=b"CORRECT_FIGURE")]
        with mock.patch("solver.get_verified_figures", return_value=fake_images):
            solver._attach_book_diagrams(questions, "fake_exercise.pdf", "7.2", class_name=9)

        self.assertIsNotNone(questions[0]["book_diagram_base64"])
        self.assertEqual(questions[0]["book_diagram_figure_ref"], "7.30")

    def test_subpart_without_ref_gets_no_diagram(self):
        """STRICT POLICY (V8.1): a sub-part that does NOT itself contain
        the figure reference must NOT inherit the diagram from a
        previous part. Each question part stands on its own."""
        questions = [
            {"question_number": 1, "question_text": "চিত্ৰ 14.9 চোৱা। উপৰোক্ত তথ্যক দণ্ডলেখৰ সহায়ত উপস্থাপন কৰা।", "given": ""},
            {"question_number": 1, "question_text": "লেখটোৰপৰা কি সিদ্ধান্তত উপনীত হ'ব পাৰি?", "given": ""},
            # This sub-part does NOT repeat "চিত্ৰ 14.9 চোৱা"
        ]
        fake_images = [self._img("14.9", tag=b"GROUP_FIGURE")]
        with mock.patch("solver.get_verified_figures", return_value=fake_images):
            solver._attach_book_diagrams(questions, "fake_exercise.pdf", "14.3", class_name=9)

        self.assertIsNotNone(questions[0]["book_diagram_base64"])
        self.assertIsNone(questions[1]["book_diagram_base64"],
                           "sub-part without its own figure reference must NOT inherit the diagram")

    def test_figure_reference_cited_only_in_required_field_is_still_found(self):
        # PRODUCTION-AUDIT FIX (final pre-launch round): the citation
        # scan used to only look at question_text + given, missing a
        # question that states its figure reference inside "required"
        # instead (a real, if less common, book phrasing). Purely
        # additive under the strict exact-match policy — can only
        # surface a genuine citation, never fabricate a false match.
        questions = [{"question_number": 3,
            "question_text": "নিম্নলিখিত তথ্যৰ ভিত্তিত এখন লেখ অংকন কৰা।",
            "given": "তালিকাত দিয়া তথ্য।",
            "required": "চিত্ৰ 9.5 ৰ দৰে এখন দণ্ডলেখ অংকন কৰি দেখুওৱা।",
        }]
        fake_images = [self._img("9.5", tag=b"REQUIRED_FIELD_FIGURE")]
        with mock.patch("solver.get_verified_figures", return_value=fake_images):
            solver._attach_book_diagrams(questions, "fake_exercise.pdf", "9.1", class_name=9)

        self.assertIsNotNone(questions[0]["book_diagram_base64"])
        self.assertEqual(questions[0]["book_diagram_figure_ref"], "9.5")


        questions = [
            {"question_number": 1, "question_text": "চিত্ৰ 7.30 চোৱা।", "given": ""},
            {"question_number": 2, "question_text": "AD এডাল উন্নতি য'ত AB = AC।", "given": ""},
        ]
        fake_images = [self._img("7.30", tag=b"Q1_FIGURE")]
        with mock.patch("solver.get_verified_figures", return_value=fake_images):
            solver._attach_book_diagrams(questions, "fake_exercise.pdf", "7.3", class_name=9)

        self.assertIsNotNone(questions[0]["book_diagram_base64"])
        self.assertIsNone(questions[1]["book_diagram_base64"],
                           "a new question_number with no figure reference of its own must not "
                           "silently inherit the previous question's image")
 
class TestAnswerKindClassification(unittest.TestCase):
    """Regression tests for the "✔ প্ৰমাণিত হ'ল" (Proven) badge fix.

    Root cause: templates/base.html.jinja used to print "✔ প্ৰমাণিত
    হ'ল।" under EVERY question unconditionally, including pure
    calculation/data-interpretation questions where nothing was
    proved. Fix: solver._classify_answer_kind() looks at the
    question's own wording (question_text/required) and returns
    "proof" only when it actually asks for a proof/demonstration;
    the template now only shows the Proven badge for that case."""

    def test_prove_that_question_is_classified_as_proof(self):
        q = {"question_text": "দেখুওৱা যে AB = AC", "required": "প্ৰমাণ কৰিব লাগে যে AB = AC"}
        self.assertEqual(solver._classify_answer_kind(q), "proof")

    def test_english_prove_keyword_is_classified_as_proof(self):
        q = {"question_text": "Prove that the triangle is isosceles.", "required": ""}
        self.assertEqual(solver._classify_answer_kind(q), "proof")

    def test_mean_median_mode_question_is_not_a_proof(self):
        # Real case from Exercise 14.4 Q1 — mean/median/mode calculation,
        # previously mis-labelled "Proven" in the generated PDF.
        q = {"question_text": "এই গোলসমূহৰ মাধ্য, মধ্যমা আৰু বহুলক নিৰ্ণয় কৰা।",
             "required": "গোলসমূহৰ মাধ্য, মধ্যমা আৰু বহুলক নিৰ্ণয় কৰিব লাগে।"}
        self.assertEqual(solver._classify_answer_kind(q), "calculation")

    def test_yes_no_justification_question_is_not_a_proof(self):
        # Real case from Exercise 14.3 Q4(iii) — the correct answer is
        # actually "No", so stamping "Proven" under it was actively
        # misleading.
        q = {"question_text": "সৰ্বোচ্চ সংখ্যক পাতৰ দীঘ 153 মিঃমিঃ বুলি মন্তব্য আগবঢ়োৱা কথাটো শুদ্ধ হ'বনে?",
             "required": "সৰ্বাধিক সংখ্যক পাতৰ দীঘ 153 মি.মি. হয় নে নহয় পৰীক্ষা কৰা।"}
        self.assertEqual(solver._classify_answer_kind(q), "calculation")

    def test_normalize_all_text_fields_sets_answer_kind_on_every_question(self):
        parsed = {"questions": [
            {"question_text": "দেখুওৱা যে AB=AC", "required": "প্ৰমাণ কৰা", "given": "", "final_answer": ""},
            {"question_text": "মাধ্য নিৰ্ণয় কৰা", "required": "মাধ্য নিৰ্ণয় কৰিব লাগে", "given": "", "final_answer": ""},
        ]}
        solver._normalize_all_text_fields(parsed)
        self.assertEqual(parsed["questions"][0]["answer_kind"], "proof")
        self.assertEqual(parsed["questions"][1]["answer_kind"], "calculation")


if __name__ == "__main__":
    unittest.main()
