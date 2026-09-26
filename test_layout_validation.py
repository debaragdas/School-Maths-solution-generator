"""
Regression tests for layout_validation.py (Phase 6 — Layout Validation
Engine) and the accompanying templates/style.css prevention fix.

`fitz` is stubbed with a lightweight fake PDF/page/rect object model
(not just a bare mock) since these tests need to exercise real
geometry math (area/overflow calculations), not just prove a function
was called — same spirit as the rest of this project's tests, which
always verify actual behavior over call-counting where the logic
being tested is itself the point.

Run: python3 -m pytest test_layout_validation.py -v
"""
import os
import sys
import types
import unittest
from unittest import mock


class _FakeRect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0


class _FakePage:
    def __init__(self, rect, images=None, text=""):
        self.rect = rect
        self._images = images or []  # list of (xref, [bbox, ...])
        self._text = text

    def get_images(self, full=True):
        return [(xref,) for xref, _ in self._images]

    def get_image_rects(self, xref):
        for x, bboxes in self._images:
            if x == xref:
                return bboxes
        return []

    def get_text(self, mode="text"):
        return self._text


class _FakeDoc:
    def __init__(self, pages):
        self._pages = pages
        self.page_count = len(pages)
        self.closed = False

    def __getitem__(self, i):
        return self._pages[i]

    def close(self):
        self.closed = True


def _install_fake_fitz(doc_to_return=None, raise_on_open=None):
    fake_fitz = types.ModuleType("fitz")
    if raise_on_open:
        fake_fitz.open = mock.Mock(side_effect=raise_on_open)
    else:
        fake_fitz.open = mock.Mock(return_value=doc_to_return)
    sys.modules["fitz"] = fake_fitz
    return fake_fitz


class TestLayoutValidation(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("layout_validation", None)

    def tearDown(self):
        sys.modules.pop("fitz", None)
        sys.modules.pop("layout_validation", None)

    def test_clean_pdf_passes(self):
        page = _FakePage(rect=_FakeRect(0, 0, 600, 800),
                          images=[(1, [_FakeRect(50, 50, 500, 400)])], text="normal text")
        _install_fake_fitz(_FakeDoc([page]))
        import layout_validation as lv
        result = lv.validate_pdf_layout("fake.pdf")
        self.assertTrue(result["ok"], result["issues"])
        self.assertEqual(result["page_count"], 1)

    def test_near_zero_area_image_flagged(self):
        page = _FakePage(rect=_FakeRect(0, 0, 600, 800),
                          images=[(1, [_FakeRect(50, 50, 50.1, 50.1)])])  # tiny sliver
        _install_fake_fitz(_FakeDoc([page]))
        import layout_validation as lv
        result = lv.validate_pdf_layout("fake.pdf")
        self.assertFalse(result["ok"])
        self.assertTrue(any("near-zero rendered area" in i["issue"] for i in result["issues"]))

    def test_boundary_overflow_flagged(self):
        page = _FakePage(rect=_FakeRect(0, 0, 600, 800),
                          images=[(1, [_FakeRect(50, 50, 500, 850)])])  # extends past bottom
        _install_fake_fitz(_FakeDoc([page]))
        import layout_validation as lv
        result = lv.validate_pdf_layout("fake.pdf")
        self.assertFalse(result["ok"])
        self.assertTrue(any("extends" in i["issue"] and "page boundary" in i["issue"] for i in result["issues"]))

    def test_replacement_glyph_flagged(self):
        page = _FakePage(rect=_FakeRect(0, 0, 600, 800), text="সংখ্যা \ufffd\ufffd তথ্য")
        _install_fake_fitz(_FakeDoc([page]))
        import layout_validation as lv
        result = lv.validate_pdf_layout("fake.pdf")
        self.assertFalse(result["ok"])
        self.assertTrue(any("replacement-character" in i["issue"] for i in result["issues"]))

    def test_multiple_pages_all_checked(self):
        good_page = _FakePage(rect=_FakeRect(0, 0, 600, 800), text="fine")
        bad_page = _FakePage(rect=_FakeRect(0, 0, 600, 800), text="bad \ufffd text")
        _install_fake_fitz(_FakeDoc([good_page, bad_page]))
        import layout_validation as lv
        result = lv.validate_pdf_layout("fake.pdf")
        self.assertFalse(result["ok"])
        self.assertEqual(result["issues"][0]["page"], 2)

    def test_unopenable_pdf_does_not_raise_and_reports_ok(self):
        _install_fake_fitz(raise_on_open=RuntimeError("corrupt file"))
        import layout_validation as lv
        result = lv.validate_pdf_layout("fake.pdf")
        self.assertTrue(result["ok"], "an audit-layer failure must never be reported as a layout defect")
        self.assertIn("skipped_reason", result)

    def test_validate_and_log_never_raises_even_on_open_failure(self):
        _install_fake_fitz(raise_on_open=RuntimeError("boom"))
        import layout_validation as lv
        result = lv.validate_and_log("fake.pdf", exercise_label="7.2")  # must not raise
        self.assertTrue(result["ok"])

    def test_validate_and_log_reports_issues(self):
        page = _FakePage(rect=_FakeRect(0, 0, 600, 800), text="bad \ufffd text")
        _install_fake_fitz(_FakeDoc([page]))
        import layout_validation as lv
        result = lv.validate_and_log("fake.pdf", exercise_label="7.2")
        self.assertFalse(result["ok"])


class TestDiagramBoxCssPreventsPageSplit(unittest.TestCase):
    """Part A of Phase 6: proves the CSS root-cause fix is actually in
    the shipped stylesheet, not just described in a design doc."""

    def setUp(self):
        css_path = os.path.join(os.path.dirname(__file__), "templates", "style.css")
        with open(css_path, "r", encoding="utf-8") as f:
            self.css = f.read()

    def _rule_block(self, selector: str) -> str:
        idx = self.css.index(selector)
        end = self.css.index("}", idx)
        return self.css[idx:end + 1]

    def test_diagram_box_has_break_inside_avoid(self):
        block = self._rule_block(".diagram-box {")
        self.assertIn("break-inside: avoid", block)
        self.assertIn("page-break-inside: avoid", block)

    def test_diagram_box_large_has_break_inside_avoid(self):
        block = self._rule_block(".diagram-box--large {")
        self.assertIn("break-inside: avoid", block)
        self.assertIn("page-break-inside: avoid", block)


if __name__ == "__main__":
    unittest.main()
