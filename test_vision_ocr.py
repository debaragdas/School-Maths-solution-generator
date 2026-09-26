"""
Regression tests for vision_ocr._extract_list — covers the confirmed
"list has no attribute get" bug.

Root cause: vision_scan_for_headings() and locate_figures_via_vision()
each called `parsed.get(key, [])` directly on the JSON decoded from a
Gemini response. The prompt asks for {"headings": [...]} / {"figures":
[...]}, but a model will sometimes return the bare array instead of the
wrapper object -- at which point `parsed` is a `list`, and `list` has no
`.get` method. That line sat OUTSIDE each call site's try/except, so
instead of skipping just the one bad batch/page (as both docstrings
claimed), an AttributeError propagated out of the whole function,
discarding every result already collected in earlier iterations of the
same call.

Run: python3 -m pytest test_vision_ocr.py -v
     (or plain: python3 test_vision_ocr.py)
"""
from vision_ocr import _extract_list


def test_bug_reproduction_bare_list_used_to_crash():
    """Documents the exact crash: calling list.get() directly raises."""
    bare_list_response = [{"bbox": [0.1, 0.1, 0.4, 0.4], "figure_ref": "3.14"}]
    try:
        bare_list_response.get("figures", [])
        assert False, "expected AttributeError — if this doesn't raise, the repro is wrong"
    except AttributeError:
        pass  # confirmed: this is the exact failure the fix addresses


def test_bare_list_response_is_handled():
    bare_list_response = [{"bbox": [0.1, 0.1, 0.4, 0.4], "figure_ref": "3.14"}]
    out = _extract_list(bare_list_response, "figures")
    assert out == [{"bbox": [0.1, 0.1, 0.4, 0.4], "figure_ref": "3.14"}]


def test_normal_wrapped_response():
    wrapped = {"figures": [{"bbox": [0, 0, 1, 1], "figure_ref": None}]}
    assert _extract_list(wrapped, "figures") == wrapped["figures"]


def test_missing_key_returns_empty():
    assert _extract_list({}, "figures") == []


def test_non_json_shaped_value_returns_empty_not_crash():
    assert _extract_list("not even a list or dict", "figures") == []
    assert _extract_list(None, "figures") == []
    assert _extract_list(42, "figures") == []


def test_non_dict_items_in_list_are_filtered():
    mixed = {"figures": [{"figure_ref": "1.1"}, "garbage", 3, None]}
    assert _extract_list(mixed, "figures") == [{"figure_ref": "1.1"}]


def test_wrong_type_value_for_key_returns_empty():
    assert _extract_list({"figures": "not a list"}, "figures") == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
