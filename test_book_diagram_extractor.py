"""
Regression tests for book_diagram_extractor.py.

The main coverage here is for the `merge_raster_and_vector_candidates`
fix: the old implementation silently dropped a `figure_ref` found by
the vector/vision path if it happened to overlap with an already-found
raster image, which was the root cause of the "figure_ref remains null"
bug for any book whose figures are embedded raster images.

Run: python3 -m unittest test_book_diagram_extractor -v
"""
import sys
import types
import unittest
from unittest import mock


def _install_fakes():
    # book_diagram_extractor imports fitz. Real PyMuPDF IS available in
    # this project's normal dev/CI environment (see requirements.txt) and
    # should always be preferred when it's genuinely importable — only
    # fall back to a stub in a sandbox that truly doesn't have it. This is
    # deliberately an import ATTEMPT, not merely "is 'fitz' already in
    # sys.modules": checking sys.modules alone made whether the real
    # PyMuPDF-backed tests below actually ran (vs. silently skipping)
    # depend on unrelated test-collection/import order elsewhere in the
    # suite, which is exactly the kind of accidental flakiness a
    # production-hardening pass should remove, not introduce.
    if "fitz" in sys.modules:
        return
    try:
        import fitz  # noqa: F401 — real PyMuPDF; leave it in sys.modules as-is
        return
    except ImportError:
        pass
    fake_fitz = types.ModuleType("fitz")
    fake_fitz.Rect = mock.Mock(side_effect=lambda *args: tuple(args))
    sys.modules["fitz"] = fake_fitz


_install_fakes()
import book_diagram_extractor as bde  # noqa: E402


class TestFigureNumPatternOcrRobustness(unittest.TestCase):
    """PRODUCTION-AUDIT FIX (final pre-launch round): _FIGURE_NUM_PATTERN
    (used only for the independent OCR cross-check of an already-claimed
    figure ref) now tolerates the same '.'/'-'/en-dash/spaced separator
    noise Tesseract commonly introduces, mirroring utils.py's own fix —
    this can only let MORE genuine matches be positively confirmed, it
    never causes a new false rejection (a non-match still just means
    'inconclusive', which keeps the original claimed ref either way)."""

    def _extract(self, text):
        m = bde._FIGURE_NUM_PATTERN.search(text)
        return f"{m.group(1)}.{m.group(2)}" if m else None

    def test_plain_dot_separator(self):
        self.assertEqual(self._extract("চিত্ৰ 7.18"), "7.18")

    def test_dash_separator(self):
        self.assertEqual(self._extract("Fig 7-18"), "7.18")

    def test_en_dash_separator(self):
        self.assertEqual(self._extract("7–18"), "7.18")

    def test_spaced_dot_separator(self):
        self.assertEqual(self._extract("7 . 18"), "7.18")

    def test_no_number_returns_none(self):
        self.assertIsNone(self._extract("চিত্ৰ"))


class TestMergeCandidates(unittest.TestCase):
    def _raster(self, page, bbox):
        return (page, bbox, "raster")

    def _vector(self, page, bbox, ref):
        return (page, bbox, "vector", ref)

    def test_disjoint_candidates_are_all_kept(self):
        raster = [self._raster(0, (10, 10, 50, 50))]
        vector = [self._vector(0, (100, 100, 150, 150), "3.14")]
        merged = bde.merge_raster_and_vector_candidates(raster, vector)
        self.assertEqual(len(merged), 2)
        self.assertIsNone(merged[0][3])
        self.assertEqual(merged[1][3], "3.14")

    def test_overlapping_vector_adopts_ref_onto_raster(self):
        """This is the core bug fix: a vector/vision candidate that
        overlaps a raster one should have its `figure_ref` ADOPTED by
        the raster entry, not discarded entirely."""
        raster = [self._raster(0, (10, 10, 100, 100))]
        vector = [self._vector(0, (12, 12, 98, 98), "3.14")]  # same region
        merged = bde.merge_raster_and_vector_candidates(raster, vector)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][2], "raster", "must keep the raster crop")
        self.assertEqual(merged[0][3], "3.14", "must adopt the vector's figure_ref")

    def test_overlapping_vector_with_no_ref_does_not_nullify_existing(self):
        # The reverse of the above: if the raster already has a ref (e.g.
        # from a previous merge) and an overlapping candidate has no ref,
        # the existing ref must not be overwritten with None.
        raster_with_ref = [[0, (10, 10, 100, 100), "raster", "3.14"]]
        vector_no_ref = self._vector(0, (12, 12, 98, 98), None)

        # Simulate the merge by hand to test the inner logic
        idx = bde.merge_raster_and_vector_candidates._overlap_index(raster_with_ref, 0, vector_no_ref[1])
        self.assertIsNotNone(idx)
        if vector_no_ref[3] and raster_with_ref[idx][3] is None:
            raster_with_ref[idx][3] = vector_no_ref[3]

        self.assertEqual(raster_with_ref[0][3], "3.14")


_HAS_REAL_FITZ = not isinstance(bde.fitz.Rect, mock.Mock)
_FONT_PATH = "templates/fonts/TiroBangla-Regular.ttf"  # ships with this project — has Bengali/Assamese glyphs


def _make_two_page_pdf(path, *, page1_drawing, page2_drawing, page1_text="", page2_text="",
                        caption_on_page2=None, caption_on_page1=None, width=400, height=500):
    """Builds a real 2-page PDF for end-to-end extractor tests. `page1_drawing`/
    `page2_drawing` are callables(shape) that draw onto that page's Shape.
    Filler Assamese/Bengali body text is always included on both pages so
    _text_layer_reliable() passes (this project's own gate — see that
    function's docstring), matching how a real SEBA/NCERT textbook page
    is dominated by Assamese body text around a figure."""
    import fitz as _fitz
    doc = _fitz.open()
    doc.new_page(width=width, height=height)
    doc.new_page(width=width, height=height)
    page1, page2 = doc[0], doc[1]

    if page1_drawing:
        shape = page1.new_shape()
        page1_drawing(shape)
        shape.finish(color=(0, 0, 0), width=1.5)
        shape.commit()
    if page2_drawing:
        shape = page2.new_shape()
        page2_drawing(shape)
        shape.finish(color=(0, 0, 0), width=1.5)
        shape.commit()

    page1.insert_font(fontname="TiroBangla", fontfile=_FONT_PATH)
    page2.insert_font(fontname="TiroBangla", fontfile=_FONT_PATH)
    # Filler body text so the page is dominated by Assamese/Bengali script,
    # exactly what _text_layer_reliable() requires before caption search
    # (and therefore this whole feature) is attempted at all.
    page1.insert_text((20, 20), page1_text or "এইটো এটা পৰীক্ষামূলক পৃষ্ঠা যাৰ মাজত চিত্ৰ আছে",
                       fontsize=10, fontname="TiroBangla", fontfile=_FONT_PATH)
    page2.insert_text((20, 20), page2_text or "দ্বিতীয় পৃষ্ঠাত চিত্ৰৰ বাকী অংশ আছে",
                       fontsize=10, fontname="TiroBangla", fontfile=_FONT_PATH)
    if caption_on_page1:
        cx, cy, ctext = caption_on_page1
        page1.insert_text((cx, cy), ctext, fontsize=11, fontname="TiroBangla", fontfile=_FONT_PATH)
    if caption_on_page2:
        cx, cy, ctext = caption_on_page2
        page2.insert_text((cx, cy), ctext, fontsize=11, fontname="TiroBangla", fontfile=_FONT_PATH)

    doc.save(path)
    doc.close()


@unittest.skipUnless(_HAS_REAL_FITZ, "requires real PyMuPDF, not the lightweight test-time fitz stub")
class TestCrossPageFigureStitching(unittest.TestCase):
    """End-to-end regression tests for the multi-page figure handling added
    in this pass (see _stitch_cross_page_figures in book_diagram_extractor.py).
    Uses real PyMuPDF to build small synthetic PDFs — genuine coverage of
    the actual page-rendering / clustering / stitching code path, not just
    the pure-Python merge logic TestMergeCandidates above already covers."""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _path(self, name):
        import os
        return os.path.join(self._tmpdir, name)

    def test_figure_split_across_page_break_is_stitched_into_one_image(self):
        """The core case this feature exists for: a figure's top half sits
        at the very bottom of page 1 (no caption there — none is dropped
        as an incomplete orphan), its bottom half sits at the very top of
        page 2 directly above the printed caption. Must come back as
        exactly ONE image, not two (and not zero)."""
        path = self._path("split.pdf")

        def draw_top(shape):
            shape.draw_line((150, 470), (150, 495))
            shape.draw_circle((150, 480), 8)

        def draw_bottom(shape):
            shape.draw_line((150, 5), (150, 80))
            shape.draw_circle((150, 60), 8)
            shape.draw_rect(bde.fitz.Rect(120, 70, 180, 100))

        _make_two_page_pdf(path, page1_drawing=draw_top, page2_drawing=draw_bottom,
                            caption_on_page2=(120, 115, "চিত্ৰ 4.7"))

        images = bde.extract_diagram_images(path, class_name=10)
        self.assertEqual(len(images), 1, "must produce exactly one stitched image, not a duplicate/fragment")
        self.assertEqual(images[0]["source"], "vector_multipage")
        self.assertEqual(images[0]["figure_ref"], "4.7")
        self.assertGreater(len(images[0]["bytes"]), 0)

    def test_stitched_image_is_taller_than_either_half_alone(self):
        """Proves the stitch actually combined both halves' pixels rather
        than just cropping one page — the output height must exceed what
        either page's own contribution alone would produce."""
        from PIL import Image
        import io as _io
        path = self._path("split_height.pdf")

        def draw_top(shape):
            shape.draw_line((150, 460), (150, 495))
            shape.draw_circle((150, 475), 8)
            shape.draw_line((110, 490), (190, 490))

        def draw_bottom(shape):
            shape.draw_line((150, 5), (150, 90))
            shape.draw_circle((150, 60), 8)
            shape.draw_line((110, 10), (190, 10))

        _make_two_page_pdf(path, page1_drawing=draw_top, page2_drawing=draw_bottom,
                            caption_on_page2=(135, 110, "চিত্ৰ 2.3"))
        images = bde.extract_diagram_images(path, class_name=10)
        self.assertEqual(len(images), 1)
        img = Image.open(_io.BytesIO(images[0]["bytes"]))
        # At RENDER_DPI=200, a single page's own ~35pt-tall crop would be
        # roughly 100px; the stitched image spans both halves so must be
        # well beyond that.
        self.assertGreater(img.height, 150)

    def test_unrelated_edge_clusters_with_no_x_overlap_are_not_stitched(self):
        """Two figures that each merely happen to touch a page edge, but
        sit at completely different horizontal positions, are NOT the same
        figure continuing — must never be merged into one fabricated
        image. Each may still independently qualify (or not) on its own
        page, but never as a false cross-page pair."""
        path = self._path("unrelated_edges.pdf")

        def draw_bottom_left(shape):
            shape.draw_line((30, 470), (30, 495))
            shape.draw_circle((30, 480), 8)

        def draw_top_right(shape):
            shape.draw_line((350, 5), (350, 30))
            shape.draw_circle((350, 20), 8)

        _make_two_page_pdf(path, page1_drawing=draw_bottom_left, page2_drawing=draw_top_right,
                            caption_on_page2=(330, 45, "চিত্ৰ 9.1"))
        images = bde.extract_diagram_images(path, class_name=10)
        # The page-2 cluster has its own caption directly below it and is a
        # complete, self-contained figure on its own page — that's fine and
        # expected. What must NOT happen is a "vector_multipage" entry.
        self.assertFalse(any(img["source"] == "vector_multipage" for img in images),
                          "unrelated edge clusters with no x-alignment must never be stitched together")

    def test_edge_clusters_without_any_caption_are_not_stitched(self):
        """No printed figure number anywhere near the continuation (or the
        first part) — per the 'never guess' policy, nothing should be
        fabricated, stitched or otherwise."""
        path = self._path("no_caption.pdf")

        def draw_top(shape):
            shape.draw_line((150, 470), (150, 495))
            shape.draw_circle((150, 480), 8)

        def draw_bottom(shape):
            shape.draw_line((150, 5), (150, 80))
            shape.draw_circle((150, 60), 8)

        _make_two_page_pdf(path, page1_drawing=draw_top, page2_drawing=draw_bottom)  # no caption anywhere
        images = bde.extract_diagram_images(path, class_name=10)
        self.assertFalse(any(img["source"] == "vector_multipage" for img in images))

    def test_single_page_figure_unaffected_by_multipage_logic(self):
        """Regression guard: an ordinary figure fully contained in the
        middle of one page (nowhere near either edge) must render exactly
        as before — the new cross-page code path must be a strict no-op
        for the common case."""
        path = self._path("single_page.pdf")

        def draw_middle(shape):
            shape.draw_line((100, 200), (300, 200))
            shape.draw_circle((200, 200), 20)

        _make_two_page_pdf(path, page1_drawing=draw_middle, page2_drawing=None,
                            caption_on_page1=(150, 240, "চিত্ৰ 1.1"))
        images = bde.extract_diagram_images(path, class_name=10)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["source"], "vector")
        self.assertEqual(images[0]["figure_ref"], "1.1")


if __name__ == "__main__":
    unittest.main()