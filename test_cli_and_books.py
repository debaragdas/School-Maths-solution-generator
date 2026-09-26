"""Tests for main.py's CLI layer (--class/--chapter/--exercise/--book-url/
--construction/--review-mode/--list-books) and the books.py class 6-10
registry it resolves through.

Contract under test:
- With NO flags, config.py's values are used unchanged (backward compat).
- Every flag overrides the config MODULE for this run only — config.py
  the file is never written.
- --exercise implies single-exercise mode for the run.
- --book-url wins over the books.py registry; without it the registry
  resolves by class; a class with no usable URL falls back to
  config.BOOK_URL with a loud warning, never a crash.
"""
import unittest
from unittest import mock

import books
import config
import main

_CONFIG_KEYS_TOUCHED_BY_CLI = [
    "CLASS", "CHAPTER", "BOOK_URL", "IS_CONSTRUCTION_CHAPTER",
    "REVIEW_MODE", "SOLVE_SINGLE_EXERCISE", "EXERCISE_FILTER",
]


def _parse(argv):
    return main.build_arg_parser().parse_args(argv)


class CLIOVERRIDEBase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(config, k) for k in _CONFIG_KEYS_TOUCHED_BY_CLI}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)


class TestNoFlagsBackwardCompat(CLIOVERRIDEBase):
    def test_no_flags_leaves_every_config_value_untouched(self):
        before = {k: getattr(config, k) for k in _CONFIG_KEYS_TOUCHED_BY_CLI}
        result = main.apply_cli_overrides(config, _parse([]))
        after = {k: getattr(config, k) for k in _CONFIG_KEYS_TOUCHED_BY_CLI}
        self.assertEqual(before, after)
        self.assertEqual(result, (int(config.CLASS), int(config.CHAPTER)))


class TestFlagOverrides(CLIOVERRIDEBase):
    def test_class_and_chapter_override(self):
        main.apply_cli_overrides(config, _parse(["--class", "10", "--chapter", "3"]))
        self.assertEqual(config.CLASS, 10)
        self.assertEqual(config.CHAPTER, 3)

    def test_exercise_flags_enable_single_exercise_mode(self):
        main.apply_cli_overrides(config, _parse(["--chapter", "7", "--exercise", "7.4"]))
        self.assertTrue(config.SOLVE_SINGLE_EXERCISE)
        self.assertEqual(config.EXERCISE_FILTER, ["7.4"])

    def test_multiple_exercise_labels_preserved_in_order(self):
        main.apply_cli_overrides(config, _parse(["--exercise", "10.1", "10.2"]))
        self.assertEqual(config.EXERCISE_FILTER, ["10.1", "10.2"])

    def test_construction_flag_true(self):
        main.apply_cli_overrides(config, _parse(["--chapter", "10", "--construction"]))
        self.assertTrue(config.IS_CONSTRUCTION_CHAPTER)

    def test_no_construction_flag_false(self):
        old = config.IS_CONSTRUCTION_CHAPTER
        try:
            config.IS_CONSTRUCTION_CHAPTER = True
            main.apply_cli_overrides(config, _parse(["--no-construction"]))
            self.assertFalse(config.IS_CONSTRUCTION_CHAPTER)
        finally:
            config.IS_CONSTRUCTION_CHAPTER = old

    def test_review_mode_override(self):
        main.apply_cli_overrides(config, _parse(["--review-mode", "cli"]))
        self.assertEqual(config.REVIEW_MODE, "cli")


class TestBookUrlResolution(CLIOVERRIDEBase):
    def test_explicit_book_url_wins_over_registry(self):
        main.apply_cli_overrides(
            config, _parse(["--class", "9", "--book-url", "https://example.com/my.pdf"]))
        self.assertEqual(config.BOOK_URL, "https://example.com/my.pdf")

    def test_registry_resolves_seba_classes_without_flag(self):
        main.apply_cli_overrides(config, _parse(["--class", "10", "--chapter", "7"]))
        self.assertEqual(config.BOOK_URL,
                         books.SEBA_MATHS_BOOKS[10]["url"])

    def test_unregistered_url_class_falls_back_with_warning_not_crash(self):
        # Class 6 has no stable single-file SCERT URL in the registry:
        # must fall back to whatever config.BOOK_URL holds, loudly.
        config.BOOK_URL = "https://fallback.example/book.pdf"
        with mock.patch.object(main.logger, "warning") as warn:
            main.apply_cli_overrides(config, _parse(["--class", "6", "--chapter", "1"]))
            self.assertTrue(warn.called)
        self.assertEqual(config.BOOK_URL, "https://fallback.example/book.pdf")

    def test_completely_unknown_class_falls_back_with_warning(self):
        config.BOOK_URL = "https://fallback.example/book.pdf"
        with mock.patch.object(main.logger, "warning"):
            main.apply_cli_overrides(config, _parse(["--class", "42", "--chapter", "1"]))
        self.assertEqual(config.BOOK_URL, "https://fallback.example/book.pdf")


class TestListBooks(unittest.TestCase):
    def test_list_books_flag_prints_registry_and_skips_pipeline(self):
        with mock.patch("builtins.print") as fake_print:
            rc = main.main(["--list-books"])
        self.assertIsNone(rc)
        printed = "".join(str(c.args[0]) for c in fake_print.call_args_list if c.args)
        for cls in range(6, 11):
            self.assertIn(f"Class {cls:>2}", printed)


class TestBooksRegistry(unittest.TestCase):
    def test_registered_classes_are_exactly_6_to_10(self):
        self.assertEqual(sorted(books.SEBA_MATHS_BOOKS), [6, 7, 8, 9, 10])

    def test_seba_class_urls_verified_shape(self):
        for cls in (9, 10):
            url = books.book_url_for(cls)
            self.assertTrue(url.startswith("https://"), url)
            self.assertTrue(url.lower().endswith(".pdf"), url)
            self.assertIn(f"class-{cls}", url.replace("%20", " ").lower())

    def test_scert_classes_raise_actionable_error(self):
        for cls in (6, 7, 8):
            with self.assertRaises(books.BookNotRegisteredException) as ctx:
                books.book_url_for(cls)
            msg = str(ctx.exception)
            self.assertIn("--book-url", msg)          # names the fix...
            self.assertIn("scert.assam.gov.in", msg)  # ...and where to look

    def test_unknown_class_raises(self):
        with self.assertRaises(books.BookNotRegisteredException):
            books.book_url_for(11)

    def test_list_books_covers_all_registered_classes(self):
        table = books.list_books()
        for cls in range(6, 11):
            self.assertIn(f"Class {cls:>2}", table)


if __name__ == "__main__":
    unittest.main()
