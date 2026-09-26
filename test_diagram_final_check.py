"""
Regression tests for diagram_final_check.py — the deterministic FINAL
VALIDATION stage that runs after rendering and before a question's
HTML/PDF is assembled.

Run: python3 -m unittest test_diagram_final_check -v
"""
import base64
import unittest
from io import BytesIO

from PIL import Image

import diagram_final_check as dfc
from diagram_decision import NO_DIAGRAM, BOOK_DIAGRAM, GENERATED_DIAGRAM

_GOOD_SVG = '<svg viewBox="0 0 260 220"><text x="10" y="10">A</text><text x="100" y="100">B</text></svg>'


def _png_base64(size=(100, 100), fill=None, gradient=False) -> str:
    img = Image.new("RGB", size, color=fill or (0, 0, 0))
    if gradient:
        for x in range(size[0]):
            for y in range(size[1]):
                img.putpixel((x, y), (x % 256, y % 256, (x * y) % 256))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestGeneratedDiagramHappyPath(unittest.TestCase):
    def test_good_svg_passes(self):
        r = dfc.final_pre_pdf_check({"diagram_decision": GENERATED_DIAGRAM}, _GOOD_SVG)
        self.assertEqual(r["final_decision"], GENERATED_DIAGRAM)
        self.assertEqual(r["diagram_svg"], _GOOD_SVG)


class TestOverlappingLabels(unittest.TestCase):
    def test_exact_duplicate_text_position_kept_and_flagged_for_review(self):
        # V40: overlap is a heuristic signal, not a technical defect —
        # the diagram is kept (still published) and the question is
        # flagged for Human Review instead of being rejected outright.
        bad_svg = '<svg viewBox="0 0 260 220"><text x="10" y="10">A</text><text x="10" y="10">B</text></svg>'
        q = {"diagram_decision": GENERATED_DIAGRAM}
        r = dfc.final_pre_pdf_check(q, bad_svg)
        self.assertEqual(r["final_decision"], GENERATED_DIAGRAM)
        self.assertEqual(r["diagram_svg"], bad_svg)
        self.assertTrue(q.get("needs_review"))
        self.assertTrue(any("same coordinates" in note for note in q.get("review_notes", [])))

    def test_nearby_but_distinct_positions_not_flagged(self):
        # Conservative by design: only an EXACT coordinate match is a
        # provable overlap; two close-but-distinct labels must never be
        # falsely flagged.
        close_svg = '<svg viewBox="0 0 260 220"><text x="10.0" y="10.0">A</text><text x="10.5" y="10.0">B</text></svg>'
        q = {"diagram_decision": GENERATED_DIAGRAM}
        r = dfc.final_pre_pdf_check(q, close_svg)
        self.assertEqual(r["final_decision"], GENERATED_DIAGRAM)
        self.assertFalse(q.get("needs_review"))


class TestClippedContent(unittest.TestCase):
    def test_coordinate_far_outside_viewbox_kept_and_flagged_for_review(self):
        # V40: clipping is a heuristic signal, not a technical defect —
        # kept and flagged for Human Review instead of rejected.
        clipped_svg = '<svg viewBox="0 0 260 220"><text x="400" y="10">A</text></svg>'
        q = {"diagram_decision": GENERATED_DIAGRAM}
        r = dfc.final_pre_pdf_check(q, clipped_svg)
        self.assertEqual(r["final_decision"], GENERATED_DIAGRAM)
        self.assertEqual(r["diagram_svg"], clipped_svg)
        self.assertTrue(q.get("needs_review"))

    def test_coordinate_within_tolerance_margin_accepted(self):
        # A stroke sitting right at the edge (within the small
        # anti-aliasing/stroke-width tolerance) must not be flagged.
        edge_svg = '<svg viewBox="0 0 260 220"><line x1="0" y1="0" x2="262" y2="220"/></svg>'
        q = {"diagram_decision": GENERATED_DIAGRAM}
        r = dfc.final_pre_pdf_check(q, edge_svg)
        self.assertEqual(r["final_decision"], GENERATED_DIAGRAM)
        self.assertFalse(q.get("needs_review"))


class TestGeneratedDiagramStillRejectsTechnicalDefects(unittest.TestCase):
    """V40: only genuinely technically-invalid SVG (non-finite
    coordinates, a malformed/degenerate viewBox) can still reject a
    GENERATED_DIAGRAM outright — everything else is a heuristic that
    routes to Human Review instead (see TestOverlappingLabels /
    TestClippedContent above)."""

    def test_nan_coordinate_still_rejected(self):
        bad_svg = '<svg viewBox="0 0 260 220"><line x1="nan" y1="20" x2="30" y2="40"/></svg>'
        q = {"diagram_decision": GENERATED_DIAGRAM}
        r = dfc.final_pre_pdf_check(q, bad_svg)
        self.assertEqual(r["final_decision"], NO_DIAGRAM)
        self.assertEqual(r["diagram_svg"], "")

    def test_degenerate_viewbox_still_rejected(self):
        bad_svg = '<svg viewBox="0 0 0 220"><text x="1" y="1">A</text></svg>'
        q = {"diagram_decision": GENERATED_DIAGRAM}
        r = dfc.final_pre_pdf_check(q, bad_svg)
        self.assertEqual(r["final_decision"], NO_DIAGRAM)


class TestMissingRender(unittest.TestCase):
    def test_generated_decision_with_no_actual_svg_rejected(self):
        r = dfc.final_pre_pdf_check({"diagram_decision": GENERATED_DIAGRAM}, "")
        self.assertEqual(r["final_decision"], NO_DIAGRAM)
        self.assertIn("no SVG", r["final_reason"])


class TestBookDiagramCropValidation(unittest.TestCase):
    def test_valid_textured_crop_passes(self):
        b64 = _png_base64(gradient=True)
        q = {"diagram_decision": BOOK_DIAGRAM, "book_diagram_base64": b64, "book_diagram_figure_ref": "3.14"}
        r = dfc.final_pre_pdf_check(q, "")
        self.assertEqual(r["final_decision"], BOOK_DIAGRAM)
        self.assertEqual(r["book_diagram_base64"], b64)
        self.assertFalse(q.get("needs_review"))

    def test_blank_crop_kept_and_flagged_for_review(self):
        # V40: blankness is a heuristic (pixel-stddev threshold) signal,
        # not proof of a technical defect — the crop decodes fine, so
        # it's kept published and flagged for Human Review instead of
        # being rejected outright.
        b64 = _png_base64(fill=(255, 255, 255))
        q = {"diagram_decision": BOOK_DIAGRAM, "book_diagram_base64": b64, "book_diagram_figure_ref": "3.14"}
        r = dfc.final_pre_pdf_check(q, "")
        self.assertEqual(r["final_decision"], BOOK_DIAGRAM)
        self.assertEqual(r["book_diagram_base64"], b64)
        self.assertTrue(q.get("needs_review"))
        self.assertTrue(any("blank" in note for note in q.get("review_notes", [])))

    def test_too_small_crop_kept_and_flagged_for_review(self):
        # V40: crop size is a heuristic threshold, not proof of
        # corruption — kept and flagged for Human Review instead.
        b64 = _png_base64(size=(5, 5))
        q = {"diagram_decision": BOOK_DIAGRAM, "book_diagram_base64": b64, "book_diagram_figure_ref": "3.14"}
        r = dfc.final_pre_pdf_check(q, "")
        self.assertEqual(r["final_decision"], BOOK_DIAGRAM)
        self.assertEqual(r["book_diagram_base64"], b64)
        self.assertTrue(q.get("needs_review"))

    def test_corrupt_bytes_still_rejected(self):
        # V40: a genuine decode failure is a technical defect — there
        # is nothing for a human to review — so this alone still rejects.
        q = {"diagram_decision": BOOK_DIAGRAM,
             "book_diagram_base64": base64.b64encode(b"not a real image").decode()}
        r = dfc.final_pre_pdf_check(q, "")
        self.assertEqual(r["final_decision"], NO_DIAGRAM)

    def test_book_decision_without_bytes_still_rejected(self):
        # V40: no image data at all is a technical defect (nothing to
        # show), so this alone still rejects.
        r = dfc.final_pre_pdf_check({"diagram_decision": BOOK_DIAGRAM}, "")
        self.assertEqual(r["final_decision"], NO_DIAGRAM)


class TestNoDiagramNegativeCase(unittest.TestCase):
    def test_stray_svg_on_no_diagram_decision_is_cleared(self):
        r = dfc.final_pre_pdf_check({"diagram_decision": NO_DIAGRAM}, _GOOD_SVG)
        self.assertEqual(r["final_decision"], NO_DIAGRAM)
        self.assertEqual(r["diagram_svg"], "")

    def test_clean_no_diagram_case_passes(self):
        r = dfc.final_pre_pdf_check({"diagram_decision": NO_DIAGRAM}, "")
        self.assertEqual(r["final_decision"], NO_DIAGRAM)


class TestBackwardCompatibleInference(unittest.TestCase):
    """A caller that never ran diagram_decision.decide_diagram() (no
    "diagram_decision" key at all) must still work sensibly rather than
    defaulting to NO_DIAGRAM and nuking a legitimate diagram."""

    def test_infers_generated_diagram_when_svg_present(self):
        r = dfc.final_pre_pdf_check({}, _GOOD_SVG)
        self.assertEqual(r["final_decision"], GENERATED_DIAGRAM)
        self.assertEqual(r["diagram_svg"], _GOOD_SVG)

    def test_infers_book_diagram_when_bytes_present(self):
        b64 = _png_base64(gradient=True)
        r = dfc.final_pre_pdf_check({"book_diagram_base64": b64}, "")
        self.assertEqual(r["final_decision"], BOOK_DIAGRAM)

    def test_infers_no_diagram_when_nothing_present(self):
        r = dfc.final_pre_pdf_check({}, "")
        self.assertEqual(r["final_decision"], NO_DIAGRAM)


if __name__ == "__main__":
    unittest.main()
