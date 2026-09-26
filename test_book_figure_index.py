"""
Regression tests for book_figure_index.py's fingerprint fix (audit
item A2): the cache key changed from a weak path+size+mtime proxy to
a real SHA-256 content hash, so two different PDFs can never collide
on their cache directory regardless of filename/timestamp coincidence.

`fitz` is stubbed before import since book_figure_index imports
book_diagram_extractor, which imports fitz at module level, and this
sandbox has no PyMuPDF and no network to install it. extract_diagram_images
itself is also mocked in these tests since we're testing the CACHE
layer, not the extraction logic (that's book_diagram_extractor's own
test file's job).

Run: python3 -m unittest test_book_figure_index -v
"""
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


def _install_fake_fitz():
    if "fitz" not in sys.modules:
        fake_fitz = types.ModuleType("fitz")
        fake_fitz.open = mock.Mock()
        sys.modules["fitz"] = fake_fitz


class TestFigureIndexFingerprint(unittest.TestCase):
    def setUp(self):
        _install_fake_fitz()
        sys.modules.pop("book_figure_index", None)
        sys.modules.pop("book_diagram_extractor", None)
        import book_figure_index as bfi
        self.bfi = bfi
        self.tmpdir = tempfile.mkdtemp()
        self.bfi.INDEX_ROOT = os.path.join(self.tmpdir, "figure_index")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_pdf(self, path, content: bytes):
        with open(path, "wb") as f:
            f.write(content)

    def test_fingerprint_is_content_based_not_path_or_mtime(self):
        path_a = os.path.join(self.tmpdir, "exercise_3.2.pdf")
        path_b = os.path.join(self.tmpdir, "exercise_3.2.pdf")  # same path, will overwrite
        self._write_pdf(path_a, b"%PDF-1.4 content A")
        fp1 = self.bfi._fingerprint(path_a)
        # overwrite same path with DIFFERENT content but try to force
        # the same mtime, simulating the exact scenario the old
        # path+size+mtime fingerprint was vulnerable to
        self._write_pdf(path_b, b"%PDF-1.4 content B - different!!")
        same_mtime = os.path.getmtime(path_a)
        os.utime(path_b, (same_mtime, same_mtime))
        fp2 = self.bfi._fingerprint(path_b)
        self.assertNotEqual(fp1, fp2, "different content at the same path/mtime must fingerprint differently")

    def test_identical_content_at_different_paths_shares_fingerprint(self):
        """Content hashing means the SAME exercise content reuses its
        cache even if the temp path differs run to run (e.g. before vs.
        after the exercise_splitter namespacing fix) — a nice side
        benefit of content-based caching over path-based."""
        path_a = os.path.join(self.tmpdir, "a.pdf")
        path_b = os.path.join(self.tmpdir, "subdir_b.pdf")
        self._write_pdf(path_a, b"%PDF-1.4 identical content")
        self._write_pdf(path_b, b"%PDF-1.4 identical content")
        self.assertEqual(self.bfi._fingerprint(path_a), self.bfi._fingerprint(path_b))

    def test_get_figure_index_cache_hit_does_not_rescan(self):
        path = os.path.join(self.tmpdir, "exercise_1.1.pdf")
        self._write_pdf(path, b"%PDF-1.4 some exercise content")

        fake_images = [{"bytes": b"pngbytes1", "ext": "png", "figure_ref": "3.14", "source": "raster"}]
        with mock.patch.object(self.bfi, "extract_diagram_images", return_value=fake_images) as extract_mock:
            first = self.bfi.get_figure_index(path, class_name=9)
            self.assertEqual(extract_mock.call_count, 1)
            second = self.bfi.get_figure_index(path, class_name=9)
            # cache hit -> extract_diagram_images must NOT run a second time
            self.assertEqual(extract_mock.call_count, 1)

        self.assertEqual(first[0]["figure_ref"], "3.14")
        self.assertEqual(second[0]["figure_ref"], "3.14")
        self.assertEqual(first[0]["bytes"], second[0]["bytes"])

    def test_different_content_never_shares_a_cached_result(self):
        path1 = os.path.join(self.tmpdir, "exercise_2.1.pdf")
        path2 = os.path.join(self.tmpdir, "exercise_2.1.pdf")  # deliberately same filename
        self._write_pdf(path1, b"%PDF-1.4 first exercise content")

        with mock.patch.object(self.bfi, "extract_diagram_images",
                                return_value=[{"bytes": b"AAA", "ext": "png", "figure_ref": "1.1", "source": "raster"}]):
            result1 = self.bfi.get_figure_index(path1, class_name=9)

        self._write_pdf(path2, b"%PDF-1.4 SECOND totally different exercise content")
        with mock.patch.object(self.bfi, "extract_diagram_images",
                                return_value=[{"bytes": b"BBB", "ext": "png", "figure_ref": "2.2", "source": "raster"}]) as extract_mock2:
            result2 = self.bfi.get_figure_index(path2, class_name=9)
            self.assertEqual(extract_mock2.call_count, 1, "different content at the same path must NOT be served from result1's cache")

        self.assertEqual(result1[0]["figure_ref"], "1.1")
        self.assertEqual(result2[0]["figure_ref"], "2.2")


if __name__ == "__main__":
    unittest.main()
