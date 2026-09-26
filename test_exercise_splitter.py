"""
Regression tests for exercise_splitter.py — both the pre-existing
heading-detection regex logic and the class/chapter temp-filename
namespacing fix from the V7 engineering audit (item A2: two different
chapters/classes' "Exercise 3.2" used to collide on the same temp
path).

This sandbox doesn't have PyMuPDF (fitz) installed and has no network
access to install it, so `fitz` is stubbed with a minimal fake BEFORE
importing exercise_splitter — enough to exercise the module's own
logic (regex matching, page-range math, output filenames) without
needing a real PDF library. This is a test-only stub; production code
is untouched and will use the real PyMuPDF wherever it's installed.

Run: python3 -m unittest test_exercise_splitter -v
"""
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self):
        return self._text


class _FakeDoc:
    """Minimal stand-in for fitz.Document sufficient for
    find_exercise_headings/split_exercises to run end-to-end."""
    def __init__(self, pages_text: dict[int, str] | None = None):
        # pages_text keyed by 1-based page number -> text
        self._pages_text = pages_text or {}
        self.saved_to = None

    def __getitem__(self, zero_based_idx):
        page_num = zero_based_idx + 1
        return _FakePage(self._pages_text.get(page_num, ""))

    def insert_pdf(self, other, from_page, to_page):
        pass  # no-op; we only care about the output PATH, not real PDF bytes

    def save(self, path):
        self.saved_to = path
        with open(path, "wb") as f:
            f.write(b"%PDF-1.4\n%fake")

    def close(self):
        pass


def _install_fake_fitz(pages_text):
    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = mock.Mock(return_value=_FakeDoc(pages_text))
    sys.modules["fitz"] = fake_fitz
    return fake_fitz


class TestExerciseSplitter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Force a clean re-import against whichever fake fitz each test installs.
        sys.modules.pop("exercise_splitter", None)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        sys.modules.pop("fitz", None)
        sys.modules.pop("exercise_splitter", None)

    def test_find_exercise_label_english(self):
        _install_fake_fitz({})
        import exercise_splitter as es
        self.assertEqual(es._find_exercise_label("Exercise 3.2\nSolve the following..."), "3.2")
        self.assertEqual(es._find_exercise_label("EXERCISE: 7.1"), "7.1")

    def test_find_exercise_label_assamese_digits(self):
        _install_fake_fitz({})
        import exercise_splitter as es
        # Assamese digits ৩.২ == "3.2"
        label = es._find_exercise_label("অনুশীলনী ৩.২")
        self.assertEqual(label, "3.2")

    def test_find_exercise_label_none_when_absent(self):
        _install_fake_fitz({})
        import exercise_splitter as es
        self.assertIsNone(es._find_exercise_label("just some regular paragraph text"))

    def test_find_exercise_headings_dedups_consecutive_same_label(self):
        pages = {
            1: "Exercise 3.1\nfirst question here",
            2: "Exercise 3.1\n(running header repeat) more text",  # same label, should dedup
            3: "Exercise 3.2\nnext exercise begins",
        }
        _install_fake_fitz(pages)
        import exercise_splitter as es
        headings = es.find_exercise_headings("fake.pdf", 1, 3)
        self.assertEqual(headings, [(1, "3.1"), (3, "3.2")])

    def test_temp_filenames_are_namespaced_by_class_and_chapter(self):
        """The actual regression test for audit item A2: two different
        (class, chapter) runs producing the SAME exercise label must
        NOT write to the same temp path."""
        pages = {1: "Exercise 3.2\nsome question text here"}
        _install_fake_fitz(pages)
        import exercise_splitter as es

        result_a = es.split_exercises("fake.pdf", 1, 1, self.tmpdir, class_name=9, chapter=3)
        result_b = es.split_exercises("fake.pdf", 1, 1, self.tmpdir, class_name=10, chapter=3)

        self.assertEqual(result_a[0]["label"], "3.2")
        self.assertEqual(result_b[0]["label"], "3.2")
        self.assertNotEqual(result_a[0]["path"], result_b[0]["path"],
                             "Exercise 3.2 from class 9 and class 10 must not share a temp path")
        self.assertTrue(os.path.exists(result_a[0]["path"]))
        self.assertTrue(os.path.exists(result_b[0]["path"]))
        # both files genuinely exist independently (the old bug would have
        # had the second call silently overwrite the first's file)
        self.assertIn("class9_ch3", result_a[0]["path"])
        self.assertIn("class10_ch3", result_b[0]["path"])

    def test_page_range_math_for_multiple_exercises(self):
        pages = {
            1: "Exercise 5.1\nfirst", 2: "still 5.1", 3: "Exercise 5.2\nsecond", 4: "still 5.2",
        }
        _install_fake_fitz(pages)
        import exercise_splitter as es
        result = es.split_exercises("fake.pdf", 1, 4, self.tmpdir, class_name=8, chapter=5)
        self.assertEqual(result[0]["start_page"], 1)
        self.assertEqual(result[0]["end_page"], 2)
        self.assertEqual(result[1]["start_page"], 3)
        self.assertEqual(result[1]["end_page"], 4)


if __name__ == "__main__":
    unittest.main()
