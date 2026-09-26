"""
Regression tests for chapter_detector.py — the pure-logic parts
(chapter-number extraction, title-quality guard, page-range resolution
from TOC/page-scan) using a minimal `fitz` stub, since this sandbox
has no PyMuPDF and no network to install it. Vision-fallback paths
(get_chapter_index_cached / get_chapter_title_via_vision) are also
mocked where exercised — this file tests chapter_detector's own logic,
not vision_ocr's.

Run: python3 -m unittest test_chapter_detector -v
"""
import sys
import types
import unittest
from unittest import mock


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self):
        return self._text


class _FakeDoc:
    def __init__(self, pages_text: dict[int, str] | None = None, toc: list | None = None, page_count: int | None = None):
        self._pages_text = pages_text or {}
        self._toc = toc or []
        self.page_count = page_count if page_count is not None else (max(self._pages_text) if self._pages_text else 0)

    def get_toc(self, simple=True):
        return self._toc

    def __iter__(self):
        for i in range(1, self.page_count + 1):
            yield _FakePage(self._pages_text.get(i, ""))

    def close(self):
        pass


def _install_fake_fitz(doc: _FakeDoc):
    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = mock.Mock(return_value=doc)
    sys.modules["fitz"] = fake_fitz


class TestChapterNumberExtraction(unittest.TestCase):
    def setUp(self):
        _install_fake_fitz(_FakeDoc())
        sys.modules.pop("chapter_detector", None)
        import chapter_detector as cd
        self.cd = cd

    def tearDown(self):
        sys.modules.pop("fitz", None)
        sys.modules.pop("chapter_detector", None)

    def test_english_chapter_word(self):
        self.assertEqual(self.cd._find_chapter_number_in_text("Chapter 7: Triangles"), 7)

    def test_assamese_word_and_digits(self):
        self.assertEqual(self.cd._find_chapter_number_in_text("অধ্যায় ৩: ত্রিভুজ"), 3)

    def test_ordinal_word_requires_chapter_context(self):
        # ordinal word alone (no "chapter"/"অধ্যায়" anywhere) must NOT match —
        # otherwise any incidental use of "তৃতীয়" (third) in body text would
        # misfire as a chapter heading
        self.assertIsNone(self.cd._find_chapter_number_in_text("তৃতীয় ব্যক্তি একজন আহিল"))

    def test_ordinal_word_with_chapter_context_matches(self):
        self.assertEqual(self.cd._find_chapter_number_in_text("তৃতীয় অধ্যায়"), 3)

    def test_no_chapter_reference_returns_none(self):
        self.assertIsNone(self.cd._find_chapter_number_in_text("Solve the following exercise."))


class TestLooksLikeRealTitle(unittest.TestCase):
    def setUp(self):
        _install_fake_fitz(_FakeDoc())
        sys.modules.pop("chapter_detector", None)
        import chapter_detector as cd
        self.cd = cd

    def tearDown(self):
        sys.modules.pop("fitz", None)
        sys.modules.pop("chapter_detector", None)

    def test_real_assamese_title_accepted(self):
        self.assertTrue(self.cd._looks_like_real_title("ত্রিভুজ"))

    def test_stray_single_latin_letter_rejected(self):
        """Regression test for the exact 'অধ্যায় ৬P' OCR-garbage bug
        this guard was built for: a lone stray Latin character is not
        a real title."""
        self.assertFalse(self.cd._looks_like_real_title("P"))

    def test_empty_string_rejected(self):
        self.assertFalse(self.cd._looks_like_real_title(""))

    def test_mostly_latin_text_rejected(self):
        self.assertFalse(self.cd._looks_like_real_title("Random garbled XKQZ"))


class TestChapterPageRange(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("fitz", None)
        sys.modules.pop("chapter_detector", None)

    def test_found_via_toc_bookmarks(self):
        toc = [[1, "Chapter 1: Numbers", 1], [1, "Chapter 2: Triangles", 10], [1, "Chapter 3: Circles", 25]]
        doc = _FakeDoc(toc=toc, page_count=40)
        _install_fake_fitz(doc)
        import chapter_detector as cd
        start, end = cd.get_chapter_page_range("fake.pdf", 2, class_name=9, text_layer_reliable=True)
        self.assertEqual((start, end), (10, 24))

    def test_last_chapter_runs_to_end_of_book(self):
        toc = [[1, "Chapter 1: Numbers", 1], [1, "Chapter 2: Triangles", 10]]
        doc = _FakeDoc(toc=toc, page_count=40)
        _install_fake_fitz(doc)
        import chapter_detector as cd
        start, end = cd.get_chapter_page_range("fake.pdf", 2, class_name=9, text_layer_reliable=True)
        self.assertEqual((start, end), (10, 40))

    def test_falls_back_to_page_scan_when_toc_missing_chapter(self):
        pages = {1: "Chapter 1: Numbers\nsome content", 5: "Chapter 2: Triangles\nmore content", 12: "Chapter 3: Circles"}
        doc = _FakeDoc(pages_text=pages, toc=[], page_count=20)
        _install_fake_fitz(doc)
        import chapter_detector as cd
        start, end = cd.get_chapter_page_range("fake.pdf", 2, class_name=9, text_layer_reliable=True)
        self.assertEqual((start, end), (5, 11))

    def test_heading_in_body_of_page_not_top_third_ignored_by_page_scan(self):
        """A stray mid-page mention of 'Chapter 3' must not be treated
        as chapter 3's heading — only text within the top third of the
        page counts."""
        long_body = "x" * 300
        pages = {1: "Chapter 1: Numbers", 5: long_body + "Chapter 3: Circles (mentioned mid-page)"}
        doc = _FakeDoc(pages_text=pages, toc=[], page_count=10)
        _install_fake_fitz(doc)
        import chapter_detector as cd
        with self.assertRaises(ValueError):
            # chapter 3 heading is buried past the top third of page 5's
            # text, and page-scan/TOC both fail to find it, so this
            # should fall through toward the vision fallback path (which
            # will fail here since vision_ocr isn't mocked/available) —
            # confirms the top-third guard is doing its job rather than
            # the (wrong) alternative of matching it anyway.
            with mock.patch.dict(sys.modules, {"vision_ocr": mock.MagicMock(
                get_chapter_index_cached=mock.Mock(return_value=[]))}):
                cd.get_chapter_page_range("fake.pdf", 3, class_name=9, text_layer_reliable=True)

    def test_unreliable_text_layer_skips_straight_to_vision(self):
        doc = _FakeDoc(toc=[[1, "Chapter 1: Numbers", 1]], page_count=20)
        _install_fake_fitz(doc)
        import chapter_detector as cd
        fake_vision_ocr = mock.MagicMock(
            get_chapter_index_cached=mock.Mock(return_value=[{"page": 8, "label": "2"}, {"page": 15, "label": "3"}])
        )
        with mock.patch.dict(sys.modules, {"vision_ocr": fake_vision_ocr}):
            start, end = cd.get_chapter_page_range("fake.pdf", 2, class_name=9, text_layer_reliable=False)
        self.assertEqual((start, end), (8, 14))
        # must not even attempt the TOC/page-scan path when the text layer is unreliable
        fake_vision_ocr.get_chapter_index_cached.assert_called_once()


if __name__ == "__main__":
    unittest.main()
