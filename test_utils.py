"""
Regression tests for utils.py. Pure stdlib (logging, re, time,
functools) — no mocking of external dependencies needed.

Run: python3 -m unittest test_utils -v
"""
import time
import threading
import unittest

import utils


class TestExtractFigureReference(unittest.TestCase):
    def test_english_figure_word(self):
        self.assertEqual(utils.extract_figure_reference("See Figure 3.14 below."), "3.14")

    def test_fig_abbreviation(self):
        self.assertEqual(utils.extract_figure_reference("As shown in Fig. 7.39."), "7.39")

    def test_assamese_word_and_digits(self):
        self.assertEqual(utils.extract_figure_reference("চিত্ৰ ৩.১৪ চাওক।"), "3.14")

    def test_mixed_ascii_and_assamese_digits(self):
        self.assertEqual(utils.extract_figure_reference("চিত্ৰ 7.39"), "7.39")

    def test_integer_only_figure_number(self):
        self.assertEqual(utils.extract_figure_reference("Figure 5 shows a triangle."), "5")

    def test_no_reference_returns_none(self):
        self.assertIsNone(utils.extract_figure_reference("Solve for x: 2x + 3 = 11"))

    def test_caption_without_number_returns_none(self):
        # "চিত্ৰ আৰ্হি" (diagram template caption, no number) must NOT match
        self.assertIsNone(utils.extract_figure_reference("চিত্ৰ আৰ্হি ব্যৱহাৰ কৰক"))

    def test_empty_text_returns_none(self):
        self.assertIsNone(utils.extract_figure_reference(""))
        self.assertIsNone(utils.extract_figure_reference(None))

    def test_ocr_dash_separator_normalized_to_dot(self):
        # scanned-caption OCR frequently misreads '.' as '-'
        self.assertEqual(utils.extract_figure_reference("Fig 7-18 shows the triangle."), "7.18")

    def test_ocr_en_dash_separator_normalized_to_dot(self):
        self.assertEqual(utils.extract_figure_reference("Figure 7–18"), "7.18")

    def test_ocr_spaced_dot_separator_normalized(self):
        self.assertEqual(utils.extract_figure_reference("Fig. 7 . 18 below"), "7.18")

    def test_assamese_digits_with_dash_separator(self):
        self.assertEqual(utils.extract_figure_reference("চিত্ৰ ৭-১৮ চাওক।"), "7.18")


class TestVerifySolution(unittest.TestCase):
    def _valid_question(self, number=1):
        return {"question_number": number, "final_answer": "42", "steps": ["step one", "step two"]}

    def test_valid_single_question_passes(self):
        ok, issues = utils.verify_solution({"questions": [self._valid_question()]})
        self.assertTrue(ok, issues)
        self.assertEqual(issues, [])

    def test_no_questions_fails(self):
        ok, issues = utils.verify_solution({"questions": []})
        self.assertFalse(ok)
        self.assertTrue(any("No questions" in i for i in issues))

    def test_missing_question_number_flagged(self):
        ok, issues = utils.verify_solution({"questions": [{"final_answer": "42", "steps": ["s"]}]})
        self.assertFalse(ok)
        self.assertTrue(any("question_number" in i for i in issues))

    def test_empty_final_answer_flagged(self):
        ok, issues = utils.verify_solution({"questions": [{"question_number": 1, "final_answer": "", "steps": ["s"]}]})
        self.assertFalse(ok)
        self.assertTrue(any("empty final_answer" in i for i in issues))

    def test_missing_steps_flagged(self):
        ok, issues = utils.verify_solution({"questions": [{"question_number": 1, "final_answer": "42", "steps": []}]})
        self.assertFalse(ok)
        self.assertTrue(any("no solution steps" in i for i in issues))

    def test_unbalanced_block_math_flagged(self):
        q = self._valid_question()
        q["final_answer"] = "The answer is $$x^2$$ and also $$y^2"  # odd number of $$
        ok, issues = utils.verify_solution({"questions": [q]})
        self.assertFalse(ok)
        self.assertTrue(any("$$" in i for i in issues))

    def test_balanced_block_math_passes(self):
        q = self._valid_question()
        q["final_answer"] = "The answer is $$x^2 + y^2$$."
        ok, issues = utils.verify_solution({"questions": [q]})
        self.assertTrue(ok, issues)

    def test_unbalanced_inline_math_flagged(self):
        q = self._valid_question()
        q["final_answer"] = "The value of $x is 5"  # single unmatched $
        ok, issues = utils.verify_solution({"questions": [q]})
        self.assertFalse(ok)
        self.assertTrue(any("inline $" in i for i in issues))

    def test_inline_dollar_inside_balanced_block_math_not_double_counted(self):
        # a lone $ INSIDE a $$...$$ block must not be mistaken for an
        # unbalanced inline $ once the block content is stripped
        q = self._valid_question()
        q["final_answer"] = r"$$\text{cost} = \$5$$"
        ok, issues = utils.verify_solution({"questions": [q]})
        self.assertTrue(ok, issues)

    def test_contiguous_numbering_passes(self):
        qs = [self._valid_question(1), self._valid_question(2), self._valid_question(3)]
        ok, issues = utils.verify_solution({"questions": qs})
        self.assertTrue(ok, issues)

    def test_numbering_gap_flagged(self):
        qs = [self._valid_question(1), self._valid_question(3)]  # missing 2
        ok, issues = utils.verify_solution({"questions": qs})
        self.assertFalse(ok)
        self.assertTrue(any("gap" in i for i in issues))

    def test_sub_parts_of_same_number_do_not_trigger_gap(self):
        qs = [
            {"question_number": 1, "sub_part": "a", "final_answer": "1", "steps": ["s"]},
            {"question_number": 1, "sub_part": "b", "final_answer": "2", "steps": ["s"]},
            {"question_number": 2, "final_answer": "3", "steps": ["s"]},
        ]
        ok, issues = utils.verify_solution({"questions": qs})
        self.assertTrue(ok, issues)


class TestAlreadyDone(unittest.TestCase):
    def test_nonexistent_path_is_not_done(self):
        self.assertFalse(utils.already_done("/tmp/definitely_does_not_exist_12345.pdf"))

    def test_small_existing_file_is_not_done(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(b"tiny")
            f.flush()
            self.assertFalse(utils.already_done(f.name))

    def test_large_existing_file_is_done(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(b"x" * 2000)
            f.flush()
            self.assertTrue(utils.already_done(f.name))


class TestRetryDecorator(unittest.TestCase):
    def test_succeeds_without_retry_when_first_call_works(self):
        calls = []

        @utils.retry(times=3, delay_seconds=0)
        def fn():
            calls.append(1)
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds(self):
        calls = []

        @utils.retry(times=3, delay_seconds=0)
        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ValueError("transient")
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(len(calls), 2)

    def test_raises_final_exception_after_exhausting_attempts(self):
        @utils.retry(times=2, delay_seconds=0)
        def fn():
            raise ValueError("always fails")

        with self.assertRaises(ValueError):
            fn()


class TestRetryWithBackoff(unittest.TestCase):
    def setUp(self):
        # These tests exercise the CONCURRENCY semaphore / retry-timing
        # behavior of retry_with_backoff in isolation — they must not
        # share the real, process-wide _vertex_ai_rate_limiter's call
        # budget with production code or with each other (a handful of
        # calls here would otherwise eat into the same 8-per-minute
        # window every other test run in this process shares, which is
        # exactly the kind of cross-test coupling that makes a suite
        # slow and order-dependent). Swap in a fresh, effectively
        # unlimited limiter for the duration of this test class only.
        self._real_rate_limiter = utils._vertex_ai_rate_limiter
        utils._vertex_ai_rate_limiter = utils._GlobalRateLimiter(max_calls_per_minute=100000)

    def tearDown(self):
        utils._vertex_ai_rate_limiter = self._real_rate_limiter

    def test_non_retryable_error_raises_immediately_no_wait(self):
        calls = []

        @utils.retry_with_backoff(times=5, base_delay=0.01, retryable_check=lambda e: False)
        def fn():
            calls.append(1)
            raise ValueError("permanent failure")

        start = time.monotonic()
        with self.assertRaises(ValueError):
            fn()
        elapsed = time.monotonic() - start
        self.assertEqual(len(calls), 1, "must not retry a non-retryable error")
        self.assertLess(elapsed, 0.5)

    def test_retryable_error_eventually_succeeds(self):
        calls = []

        @utils.retry_with_backoff(times=4, base_delay=0.001, max_delay=0.01, retryable_check=lambda e: True)
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("429 rate limited")
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(len(calls), 3)

    def test_no_retryable_check_treats_everything_as_retryable(self):
        calls = []

        @utils.retry_with_backoff(times=2, base_delay=0.001, max_delay=0.01)
        def fn():
            calls.append(1)
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            fn()
        self.assertEqual(len(calls), 2)

    def test_only_one_call_in_flight_at_a_time_across_threads(self):
        """The concurrency fix: no matter how many threads call a
        retry_with_backoff-wrapped function at once (e.g. PARALLEL_WORKERS
        exercises each making their own Vertex AI call), the ACTUAL
        underlying calls must never overlap in time — proven directly by
        having each call record enter/exit timestamps and asserting no
        two [enter, exit] windows intersect, not just by asserting a
        final call count."""
        lock_free_calls = []  # (start, end) pairs
        record_lock = threading.Lock()

        @utils.retry_with_backoff(times=1, base_delay=0.001, max_delay=0.01)
        def fn():
            start = time.monotonic()
            time.sleep(0.03)  # long enough that overlap would be reliably observed
            end = time.monotonic()
            with record_lock:
                lock_free_calls.append((start, end))
            return "ok"

        threads = [threading.Thread(target=fn) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(lock_free_calls), 6)
        windows = sorted(lock_free_calls)
        for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
            self.assertLessEqual(
                e1, s2,
                f"two calls overlapped in time: [{s1:.4f}, {e1:.4f}] and "
                f"[{s2:.4f}, {s2:.4f}] — more than one Vertex AI request "
                f"was in flight at once"
            )

    def test_concurrency_limit_does_not_hold_semaphore_during_backoff_sleep(self):
        """The semaphore must be released while a failed call is sleeping
        off its backoff delay — otherwise one thread's retry wait would
        needlessly block every other thread's independent call, which
        would slow the whole pipeline down for no quota-safety benefit."""
        calls = []

        @utils.retry_with_backoff(times=2, base_delay=0.05, max_delay=0.05,
                                   retryable_check=lambda e: True)
        def flaky():
            calls.append(time.monotonic())
            if len(calls) == 1:
                raise ValueError("429 rate limited")
            return "ok"

        @utils.retry_with_backoff(times=1, base_delay=0.001)
        def other():
            return time.monotonic()

        results = {}

        def run_flaky():
            results["flaky"] = flaky()

        t = threading.Thread(target=run_flaky)
        t.start()
        time.sleep(0.01)  # let `flaky` fail once and enter its backoff sleep
        other_call_time = other()  # must not block behind flaky's backoff sleep
        t.join()

        # `other` must have completed well before flaky's ~0.05s backoff
        # sleep finished, proving it wasn't held up by the semaphore.
        self.assertLess(other_call_time - calls[0], 0.045)


if __name__ == "__main__":
    unittest.main()
