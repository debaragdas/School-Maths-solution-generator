"""
Regression tests for label_layout.py — the smart label placement /
automatic collision-repositioning module.

Run: python3 -m unittest test_label_layout -v
"""
import unittest

import label_layout as ll


class TestParsing(unittest.TestCase):
    def test_parses_basic_text_elements(self):
        svg = '<svg viewBox="0 0 200 200"><text x="10" y="20">A</text></svg>'
        elements = ll.parse_text_elements(svg)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["x"], 10.0)
        self.assertEqual(elements[0]["y"], 20.0)
        self.assertEqual(elements[0]["text"], "A")

    def test_skips_elements_missing_coordinates(self):
        svg = '<svg viewBox="0 0 200 200"><text>no coords</text></svg>'
        elements = ll.parse_text_elements(svg)
        self.assertEqual(elements, [])

    def test_respects_text_anchor_for_bbox(self):
        svg = ('<svg viewBox="0 0 200 200">'
               '<text x="100" y="20" text-anchor="middle">Hello</text></svg>')
        elements = ll.parse_text_elements(svg)
        x0, y0, x1, y1 = elements[0]["bbox"]
        self.assertLess(x0, 100)
        self.assertGreater(x1, 100)


class TestOverlapDetection(unittest.TestCase):
    def test_detects_exact_duplicate_as_overlap(self):
        svg = ('<svg viewBox="0 0 200 200">'
               '<text x="10" y="10">A</text><text x="10" y="10">B</text></svg>')
        elements = ll.parse_text_elements(svg)
        self.assertEqual(len(ll.find_overlaps(elements)), 1)

    def test_detects_near_but_not_identical_overlap(self):
        # Two labels 2px apart with a 13px font WILL visually overlap
        # even though their coordinates are not identical — this is
        # exactly the class of collision the old exact-match check in
        # diagram_final_check.py could never catch.
        svg = ('<svg viewBox="0 0 200 200">'
               '<text x="10" y="10" font-size="13">Hello</text>'
               '<text x="12" y="10" font-size="13">World</text></svg>')
        elements = ll.parse_text_elements(svg)
        self.assertEqual(len(ll.find_overlaps(elements)), 1)

    def test_far_apart_labels_not_flagged(self):
        svg = ('<svg viewBox="0 0 200 200">'
               '<text x="10" y="10">A</text><text x="150" y="150">B</text></svg>')
        elements = ll.parse_text_elements(svg)
        self.assertEqual(ll.find_overlaps(elements), [])


class TestRepositioning(unittest.TestCase):
    def test_no_overlap_returns_svg_unchanged(self):
        svg = ('<svg viewBox="0 0 200 200">'
               '<text x="10" y="10">A</text><text x="150" y="150">B</text></svg>')
        result = ll.check_and_fix_labels(svg)
        self.assertEqual(result["svg"], svg)
        self.assertEqual(result["moved"], 0)
        self.assertEqual(result["unresolved"], 0)

    def test_overlapping_labels_get_repositioned(self):
        svg = ('<svg viewBox="0 0 300 300">'
               '<text x="100" y="100" font-size="13">Hello</text>'
               '<text x="102" y="100" font-size="13">World</text></svg>')
        result = ll.check_and_fix_labels(svg)
        self.assertGreaterEqual(result["moved"], 1)
        # after fixing, re-parsing the returned SVG must show no residual overlap
        elements_after = ll.parse_text_elements(result["svg"])
        self.assertEqual(ll.find_overlaps(elements_after), [])

    def test_first_label_in_document_order_stays_anchored(self):
        svg = ('<svg viewBox="0 0 300 300">'
               '<text x="100" y="100" font-size="13">Hello</text>'
               '<text x="102" y="100" font-size="13">World</text></svg>')
        result = ll.check_and_fix_labels(svg)
        elements_after = ll.parse_text_elements(result["svg"])
        self.assertEqual(elements_after[0]["x"], 100.0)
        self.assertEqual(elements_after[0]["y"], 100.0)

    def test_repositioning_stays_within_viewbox_when_possible(self):
        svg = ('<svg viewBox="0 0 300 300">'
               '<text x="150" y="150" font-size="13">Hello</text>'
               '<text x="151" y="150" font-size="13">World</text></svg>')
        result = ll.check_and_fix_labels(svg)
        for e in ll.parse_text_elements(result["svg"]):
            x0, y0, x1, y1 = e["bbox"]
            self.assertGreaterEqual(x0, -1)
            self.assertGreaterEqual(y0, -1)

    def test_malformed_svg_returns_unchanged(self):
        bad_svg = "not even valid svg <<<"
        result = ll.check_and_fix_labels(bad_svg)
        self.assertEqual(result["svg"], bad_svg)
        self.assertEqual(result["moved"], 0)

    def test_three_stacked_labels_all_resolved(self):
        svg = ('<svg viewBox="0 0 300 300">'
               '<text x="100" y="100" font-size="13">AAAA</text>'
               '<text x="101" y="100" font-size="13">BBBB</text>'
               '<text x="102" y="101" font-size="13">CCCC</text></svg>')
        result = ll.check_and_fix_labels(svg)
        elements_after = ll.parse_text_elements(result["svg"])
        self.assertEqual(ll.find_overlaps(elements_after), [])


if __name__ == "__main__":
    unittest.main()
