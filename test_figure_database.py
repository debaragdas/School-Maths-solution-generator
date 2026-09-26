"""
Regression tests for figure_database.py (Phase 2 — Persistent Figure
Database). `fitz` is stubbed exactly as test_book_figure_index.py
already does, since book_diagram_extractor imports it at module level
and this sandbox has neither PyMuPDF nor network access.
extract_diagram_images itself is mocked — these tests exercise the
PERSISTENCE/MERGE layer, not the extraction logic.

Run: python3 -m pytest test_figure_database.py -v
"""
import os
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


def _install_fake_fitz():
    if "fitz" not in sys.modules:
        fake_fitz = types.ModuleType("fitz")
        fake_fitz.open = mock.Mock()
        sys.modules["fitz"] = fake_fitz


class TestFigureDatabase(unittest.TestCase):
    def setUp(self):
        _install_fake_fitz()
        sys.modules.pop("figure_database", None)
        sys.modules.pop("book_diagram_extractor", None)
        import figure_database as fdb
        self.fdb = fdb
        self.tmpdir = tempfile.mkdtemp()
        self.fdb.DB_ROOT = os.path.join(self.tmpdir, "figure_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_pdf(self, name, content: bytes):
        path = os.path.join(self.tmpdir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    # ---- backward compatibility: no chapter given -> old per-run behavior ----

    def test_no_chapter_given_falls_back_to_non_persisted_extraction(self):
        path = self._write_pdf("ex.pdf", b"%PDF content A")
        fake_images = [{"bytes": b"AAA", "ext": "png", "figure_ref": "1.1", "source": "raster"}]
        with mock.patch.object(self.fdb, "extract_diagram_images", return_value=fake_images) as extract_mock:
            result1 = self.fdb.get_verified_figures(path, class_name=9, chapter=None)
            result2 = self.fdb.get_verified_figures(path, class_name=9, chapter=None)
        self.assertEqual(extract_mock.call_count, 2, "no persistence should happen without a chapter")
        self.assertEqual(result1[0]["figure_ref"], "1.1")
        self.assertEqual(result2[0]["figure_ref"], "1.1")

    # ---- core persistence behavior ----

    def test_same_pdf_second_call_does_not_reextract(self):
        path = self._write_pdf("ex_7_1.pdf", b"%PDF content B")
        fake_images = [{"bytes": b"BBB", "ext": "png", "figure_ref": "7.14", "source": "raster"}]
        with mock.patch.object(self.fdb, "extract_diagram_images", return_value=fake_images) as extract_mock:
            first = self.fdb.get_verified_figures(path, class_name=9, chapter=7, book_url="http://x/book.pdf")
            self.assertEqual(extract_mock.call_count, 1)
            second = self.fdb.get_verified_figures(path, class_name=9, chapter=7, book_url="http://x/book.pdf")
            self.assertEqual(extract_mock.call_count, 1, "cache hit must not re-run extraction")
        self.assertEqual(first[0]["figure_ref"], "7.14")
        self.assertEqual(second[0]["figure_ref"], "7.14")
        self.assertEqual(first[0]["bytes"], second[0]["bytes"])

    def test_figures_persist_to_disk_across_a_fresh_module_state(self):
        """The whole point of Phase 2: unlike the old per-run cache,
        this must survive being looked up from a totally fresh
        in-memory state (simulating a new process), as long as the
        same DB_ROOT directory is used."""
        path = self._write_pdf("ex_7_1.pdf", b"%PDF content C")
        fake_images = [{"bytes": b"CCC", "ext": "png", "figure_ref": "7.20", "source": "vector"}]
        with mock.patch.object(self.fdb, "extract_diagram_images", return_value=fake_images):
            self.fdb.get_verified_figures(path, class_name=9, chapter=7, book_url="http://x/book.pdf")

        db_root = self.fdb.DB_ROOT
        sys.modules.pop("figure_database", None)
        import figure_database as fdb2
        fdb2.DB_ROOT = db_root
        with mock.patch.object(fdb2, "extract_diagram_images") as extract_mock2:
            result = fdb2.get_verified_figures(path, class_name=9, chapter=7, book_url="http://x/book.pdf")
            extract_mock2.assert_not_called()
        self.assertEqual(result[0]["figure_ref"], "7.20")
        self.assertEqual(result[0]["bytes"], b"CCC")

    def test_different_exercise_pdfs_in_same_chapter_accumulate(self):
        """This is the capability the old per-run cache structurally
        could not have: exercise 7.1's figure and exercise 7.2's
        figure both end up retrievable together once both have been
        ingested into the same chapter's store."""
        path1 = self._write_pdf("ex_7_1.pdf", b"%PDF exercise 7.1 content")
        path2 = self._write_pdf("ex_7_2.pdf", b"%PDF exercise 7.2 content, totally different")

        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"F14", "ext": "png", "figure_ref": "7.14", "source": "raster"}]):
            self.fdb.get_verified_figures(path1, class_name=9, chapter=7, book_url="http://x/book.pdf")

        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"F39", "ext": "png", "figure_ref": "7.39", "source": "raster"}]):
            result2 = self.fdb.get_verified_figures(path2, class_name=9, chapter=7, book_url="http://x/book.pdf")

        refs = {img["figure_ref"] for img in result2}
        self.assertEqual(refs, {"7.14", "7.39"}, "chapter-level store must accumulate across exercise PDFs")

    def test_different_chapters_are_isolated(self):
        path_ch7 = self._write_pdf("ex_7_1.pdf", b"%PDF chapter 7 content")
        path_ch8 = self._write_pdf("ex_8_1.pdf", b"%PDF chapter 8 content")
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"X", "ext": "png", "figure_ref": "7.1", "source": "raster"}]):
            self.fdb.get_verified_figures(path_ch7, class_name=9, chapter=7, book_url="http://x/book.pdf")
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"Y", "ext": "png", "figure_ref": "8.1", "source": "raster"}]) as extract_mock:
            result = self.fdb.get_verified_figures(path_ch8, class_name=9, chapter=8, book_url="http://x/book.pdf")
        refs = {img["figure_ref"] for img in result}
        self.assertEqual(refs, {"8.1"}, "chapter 8's store must not contain chapter 7's figures")

    def test_different_books_are_isolated(self):
        path_a = self._write_pdf("ex_a.pdf", b"%PDF book A content")
        path_b = self._write_pdf("ex_b.pdf", b"%PDF book B content")
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"X", "ext": "png", "figure_ref": "1.1", "source": "raster"}]):
            self.fdb.get_verified_figures(path_a, class_name=9, chapter=1, book_url="http://x/book_a.pdf")
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"Y", "ext": "png", "figure_ref": "1.1", "source": "raster"}]) as extract_mock:
            self.fdb.get_verified_figures(path_b, class_name=9, chapter=1, book_url="http://x/book_b.pdf")
        self.assertEqual(extract_mock.call_count, 1, "book B must extract independently of book A's cache")

    def test_first_seen_figure_wins_on_duplicate_ref_across_pdfs(self):
        path1 = self._write_pdf("ex_1.pdf", b"%PDF first")
        path2 = self._write_pdf("ex_2.pdf", b"%PDF second, different content")
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"ORIGINAL", "ext": "png", "figure_ref": "7.14", "source": "raster"}]):
            self.fdb.get_verified_figures(path1, class_name=9, chapter=7, book_url="http://x/book.pdf")
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"DUPLICATE", "ext": "png", "figure_ref": "7.14", "source": "raster"}]):
            result = self.fdb.get_verified_figures(path2, class_name=9, chapter=7, book_url="http://x/book.pdf")
        matches = [img for img in result if img["figure_ref"] == "7.14"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["bytes"], b"ORIGINAL")

    def test_captionless_images_are_never_persisted_but_still_returned_this_run(self):
        path = self._write_pdf("ex.pdf", b"%PDF content")
        fake_images = [
            {"bytes": b"REF", "ext": "png", "figure_ref": "7.14", "source": "raster"},
            {"bytes": b"NOREF", "ext": "png", "figure_ref": None, "source": "raster"},
        ]
        with mock.patch.object(self.fdb, "extract_diagram_images", return_value=fake_images):
            result = self.fdb.get_verified_figures(path, class_name=9, chapter=7, book_url="http://x/book.pdf")
        self.assertEqual(len(result), 2)
        chapter_dir = self.fdb._chapter_dir(self.fdb._book_id_from_url("http://x/book.pdf"), 7)
        index = self.fdb._load_index(chapter_dir)
        self.assertEqual(len(index["figures"]), 1, "only the ref-bearing figure should be persisted")

    def test_extraction_failure_degrades_to_empty_list_not_a_crash(self):
        path = self._write_pdf("ex.pdf", b"%PDF content")
        with mock.patch.object(self.fdb, "extract_diagram_images", side_effect=RuntimeError("boom")):
            result = self.fdb.get_verified_figures(path, class_name=9, chapter=7, book_url="http://x/book.pdf")
        self.assertEqual(result, [])

    def test_corrupt_index_json_is_recovered_from_not_fatal(self):
        path = self._write_pdf("ex.pdf", b"%PDF content")
        book_id = self.fdb._book_id_from_url("http://x/book.pdf")
        chapter_dir = self.fdb._chapter_dir(book_id, 7)
        os.makedirs(chapter_dir, exist_ok=True)
        with open(os.path.join(chapter_dir, "index.json"), "w") as f:
            f.write("{not valid json!!!")
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"X", "ext": "png", "figure_ref": "7.1", "source": "raster"}]):
            result = self.fdb.get_verified_figures(path, class_name=9, chapter=7, book_url="http://x/book.pdf")
        self.assertEqual(result[0]["figure_ref"], "7.1")

    def test_concurrent_ingestion_from_different_exercises_never_loses_a_figure(self):
        """PRODUCTION-AUDIT REGRESSION TEST (Phase 7): reproduces the
        exact race main.py's ThreadPoolExecutor can trigger — multiple
        exercises in the SAME chapter calling get_verified_figures()
        at the same moment, each contributing a DIFFERENT new figure.
        Without the _chapter_lock fix, this reliably loses at least one
        thread's figure to a last-writer-wins index.json overwrite;
        with the fix, every figure must survive."""
        import threading

        n_threads = 8
        paths = []
        for i in range(n_threads):
            paths.append(self._write_pdf(f"ex_thread_{i}.pdf", f"%PDF unique content {i}".encode()))

        def fake_extract(pdf_path, class_name=None):
            # derive a unique figure_ref per PDF so we can verify none
            # were lost, and add a tiny sleep to widen the race window
            idx = pdf_path.rsplit("_", 1)[-1].split(".")[0]
            time.sleep(0.01)
            return [{"bytes": f"BYTES{idx}".encode(), "ext": "png",
                     "figure_ref": f"7.{idx}", "source": "raster"}]

        with mock.patch.object(self.fdb, "extract_diagram_images", side_effect=fake_extract):
            threads = [threading.Thread(target=self.fdb.get_verified_figures,
                                        kwargs=dict(pdf_path=p, class_name=9, chapter=7,
                                                    book_url="http://x/book.pdf"))
                       for p in paths]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # after all threads complete, the chapter's persisted index must
        # contain EVERY figure — none silently lost to a lost write.
        book_id = self.fdb._book_id_from_url("http://x/book.pdf")
        chapter_dir = self.fdb._chapter_dir(book_id, 7)
        index = self.fdb._load_index(chapter_dir)
        refs = {e["figure_ref"] for e in index["figures"]}
        expected_refs = {f"7.{i}" for i in range(n_threads)}
        self.assertEqual(refs, expected_refs,
                          f"lost figures under concurrent ingestion: missing {expected_refs - refs}")


class TestManualFigureIngestion(unittest.TestCase):
    """Regression tests for add_manual_figure() — the Human Review
    workflow's 'Missing Book Figure' capability."""

    def setUp(self):
        _install_fake_fitz()
        sys.modules.pop("figure_database", None)
        sys.modules.pop("book_diagram_extractor", None)
        import figure_database as fdb
        self.fdb = fdb
        self.tmpdir = tempfile.mkdtemp()
        self.fdb.DB_ROOT = os.path.join(self.tmpdir, "figure_db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_manual_figure_is_persisted_and_marked_verified(self):
        entry = self.fdb.add_manual_figure(
            image_bytes=b"CROPPED_PNG_BYTES", figure_number="4.12", class_name=9, chapter=4,
            subject="Mathematics", page_number=83, reviewer="Priya", book_url="http://x/book.pdf",
        )
        self.assertEqual(entry["figure_ref"], "4.12")
        self.assertTrue(entry["verified_by_human"])
        self.assertEqual(entry["source"], "manual")
        self.assertEqual(entry["version"], 1)

    def test_manual_figure_is_retrievable_via_get_verified_figures(self):
        """The whole point: solver.py's existing, UNCHANGED exact-ref
        matching must pick this up automatically with zero code
        changes there — verified by calling the real public lookup
        function, not just inspecting the index file."""
        self.fdb.add_manual_figure(
            image_bytes=b"CROPPED_PNG_BYTES", figure_number="4.12", class_name=9, chapter=4,
            book_url="http://x/book.pdf",
        )
        with mock.patch.object(self.fdb, "extract_diagram_images", return_value=[]):
            path = os.path.join(self.tmpdir, "ex_4_2.pdf")
            with open(path, "wb") as f:
                f.write(b"%PDF some exercise 4.2 content")
            result = self.fdb.get_verified_figures(path, class_name=9, chapter=4, book_url="http://x/book.pdf")
        refs = {img["figure_ref"]: img for img in result}
        self.assertIn("4.12", refs)
        self.assertEqual(refs["4.12"]["bytes"], b"CROPPED_PNG_BYTES")

    def test_reuploading_same_figure_number_overwrites_and_bumps_version(self):
        self.fdb.add_manual_figure(
            image_bytes=b"FIRST_CROP", figure_number="4.12", class_name=9, chapter=4,
            book_url="http://x/book.pdf",
        )
        entry2 = self.fdb.add_manual_figure(
            image_bytes=b"CORRECTED_CROP", figure_number="4.12", class_name=9, chapter=4,
            book_url="http://x/book.pdf",
        )
        self.assertEqual(entry2["version"], 2)
        chapter_dir = self.fdb._chapter_dir(self.fdb._book_id_from_url("http://x/book.pdf"), 4)
        index = self.fdb._load_index(chapter_dir)
        matches = [e for e in index["figures"] if e["figure_ref"] == "4.12"]
        self.assertEqual(len(matches), 1, "must overwrite, not duplicate, the same figure_ref")
        with open(os.path.join(chapter_dir, matches[0]["file"]), "rb") as f:
            self.assertEqual(f.read(), b"CORRECTED_CROP")

    def test_manual_figure_overrides_an_automatically_extracted_one_with_the_same_ref(self):
        with mock.patch.object(self.fdb, "extract_diagram_images",
                                return_value=[{"bytes": b"AUTO_EXTRACTED", "ext": "png",
                                                "figure_ref": "4.12", "source": "raster"}]):
            path = os.path.join(self.tmpdir, "ex_4_1.pdf")
            with open(path, "wb") as f:
                f.write(b"%PDF auto content")
            self.fdb.get_verified_figures(path, class_name=9, chapter=4, book_url="http://x/book.pdf")

        self.fdb.add_manual_figure(
            image_bytes=b"HUMAN_VERIFIED_CROP", figure_number="4.12", class_name=9, chapter=4,
            reviewer="Priya", book_url="http://x/book.pdf",
        )
        with mock.patch.object(self.fdb, "extract_diagram_images", return_value=[]):
            result = self.fdb.get_verified_figures(path, class_name=9, chapter=4, book_url="http://x/book.pdf")
        matches = [img for img in result if img["figure_ref"] == "4.12"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["bytes"], b"HUMAN_VERIFIED_CROP")

    def test_normalizes_assamese_digit_figure_numbers(self):
        entry = self.fdb.add_manual_figure(
            image_bytes=b"X", figure_number="৪.১২", class_name=9, chapter=4, book_url="http://x/book.pdf",
        )
        self.assertEqual(entry["figure_ref"], "4.12")

    def test_rejects_empty_bytes(self):
        with self.assertRaises(ValueError):
            self.fdb.add_manual_figure(image_bytes=b"", figure_number="4.12", class_name=9, chapter=4,
                                        book_url="http://x/book.pdf")

    def test_rejects_unusable_figure_number(self):
        with self.assertRaises(ValueError):
            self.fdb.add_manual_figure(image_bytes=b"X", figure_number="not-a-number", class_name=9,
                                        chapter=4, book_url="http://x/book.pdf")

    def test_duplicate_bytes_under_a_different_ref_logs_warning_but_does_not_raise(self):
        self.fdb.add_manual_figure(image_bytes=b"SAME_BYTES", figure_number="4.12", class_name=9,
                                    chapter=4, book_url="http://x/book.pdf")
        # Should not raise even though the bytes are identical to 4.12's.
        entry = self.fdb.add_manual_figure(image_bytes=b"SAME_BYTES", figure_number="4.13", class_name=9,
                                            chapter=4, book_url="http://x/book.pdf")
        self.assertEqual(entry["figure_ref"], "4.13")


if __name__ == "__main__":
    unittest.main()
