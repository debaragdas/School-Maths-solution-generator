"""
Regression tests for review_cli.py and the shared
figure_database.detect_image_ext() helper it (and
correction_engine.attach_manual_figure_to_question) rely on.

Run: python3 -m pytest test_review_cli.py -v
"""
import io
import sys
import types
import unittest
from unittest import mock

from PIL import Image


def _install_fakes():
    if "fitz" not in sys.modules:
        fake_fitz = types.ModuleType("fitz")
        fake_fitz.open = mock.Mock()
        sys.modules["fitz"] = fake_fitz


_install_fakes()

import review_cli
import figure_database


class TestDetectImageExt(unittest.TestCase):
    """detect_image_ext lives in figure_database.py (shared by
    review_cli.py's add-figure command and
    correction_engine.attach_manual_figure_to_question) — see its
    docstring for the mismatched-mime-type bug this fixes."""

    def _bytes(self, fmt):
        img = Image.new("RGB", (50, 50), color="blue")
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_detects_real_format_even_with_mismatched_fallback(self):
        jpeg_bytes = self._bytes("JPEG")
        self.assertEqual(figure_database.detect_image_ext(jpeg_bytes, fallback="png"), "jpg")

    def test_detects_png_correctly(self):
        png_bytes = self._bytes("PNG")
        self.assertEqual(figure_database.detect_image_ext(png_bytes, fallback="jpg"), "png")

    def test_falls_back_on_undetectable_bytes_without_raising(self):
        self.assertEqual(figure_database.detect_image_ext(b"not an image at all", fallback="png"), "png")

    def test_falls_back_on_empty_bytes_without_raising(self):
        self.assertEqual(figure_database.detect_image_ext(b"", fallback="jpg"), "jpg")


class TestReviewCliImportsWithoutHeavyDeps(unittest.TestCase):
    """review_cli.py must stay importable/usable for list/show/approve/
    reject/publish without requiring google-genai or fitz to be loaded
    at import time — only `correct` (correction_engine) and `add-figure`
    (figure_database) should pull in the heavier stack, and only when
    actually invoked. See review_cli.py's lazy imports."""

    def test_heavy_modules_are_not_bound_at_module_level(self):
        self.assertNotIn("correction_engine", vars(review_cli),
                          "correction_engine must be imported lazily inside _cmd_correct, not at module level")
        self.assertNotIn("figure_database", vars(review_cli),
                          "figure_database must be imported lazily inside _cmd_add_figure, not at module level")

    def test_build_parser_works(self):
        parser = review_cli.build_parser()
        self.assertIsNotNone(parser)


if __name__ == "__main__":
    unittest.main()
