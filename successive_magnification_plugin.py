"""
successive_magnification_plugin.py — diagram_type "successive_magnification"

GENUINE, CONFIRMED-RECURRING GAP (not speculative — this exact
diagram_type name, or the near-identical "number_line_magnification",
was observed being requested by Gemini and rejected by decide_diagram
as "not a recognized/renderable type" across multiple independent
review sessions, always for the same underlying question pattern):
Class 9 SEBA "Number System" chapter's standard technique for locating
an irrational or repeating decimal on the number line by repeated
10-way zooming — "সংখ্যাৰেখাত ৪.২৬২৬ ক সূচিত কৰক" ("show 4.2626 on the
number line"), "সংখ্যাৰেখাত কিদৰে পৰিবৰ্ধন সহায়ত ৩.৭৬৫ সংখ্যাটো সূচিত
কৰিব পাৰি দেখুওৱা" ("show how successive magnification locates 3.765"),
etc. Nothing in this codebase rendered this before this file — the
existing "number_line" type is a single, non-zoomed number line and is
a different (also valid, still-supported) diagram family.

THE TECHNIQUE, for a decimal value with d digits after the point (e.g.
3.765, d=3): draw 1+d stacked number-line rows.

  Row 0 (overview): the integers surrounding floor(value), with the
    unit segment [floor(value), floor(value)+1] highlighted.
  Row j, for j = 1..d: the interval that was highlighted in row j-1,
    subdivided into 10 equal tenths (so row j's ticks are exactly one
    more decimal place of precision than row j-1's). The sub-interval
    containing `value` is highlighted (row j < d), except in the FINAL
    row (j == d), where the exact point `value` itself is marked
    instead of a sub-interval, since no further subdivision is needed.

Dashed guide lines connect each highlighted segment's two endpoints
down to the next row's two end-ticks, visually showing the zoom.

MATHEMATICAL EXACTNESS: `value` is accepted as a STRING (e.g. "3.765"),
not a JSON number — deliberately, so every decimal digit is taken
exactly as given, with zero floating-point rounding risk (a JSON
number like 3.765 does not have an exact binary float representation,
and reconstructing "which digit is which" from a rounded float would
risk an off-by-one digit at exactly the wrong moment). All internal
arithmetic uses Python's `decimal.Decimal`, never `float`, until the
final SVG pixel-coordinate step (where sub-pixel precision doesn't
matter). Verified by reconstructing `value` from the exact chosen
digit path and comparing back against the input Decimal — this is the
same "prove it, don't just draw it" discipline circle_sector_plugin.py
established for this project.
"""
import re
from decimal import Decimal, InvalidOperation

import diagram_plugin_registry
import label_layout

LINE_COLOR = "#1a4d8f"
AUX_COLOR = "#c0392b"
LABEL_COLOR = "#111111"
HIGHLIGHT_COLOR = "#1a9e8f"
GUIDE_COLOR = "#888888"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='600'"
SMALL_FONT = "font-family='Noto Sans, Arial' font-size='11'"
TITLE_FONT = "font-family='Noto Sans, Arial' font-size='11' font-weight='600'"

W = 640
ROW_HEIGHT = 92
TOP_MARGIN = 22
PAD_L, PAD_R = 55, 55

_VALUE_PATTERN = re.compile(r"^-?\d+\.\d+$")


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _validate_successive_magnification_spec(spec: dict):
    issues = []
    value = spec.get("value")
    if not isinstance(value, str):
        issues.append(f"'value' must be a STRING (e.g. \"3.765\"), not a JSON number — this avoids "
                      f"floating-point rounding corrupting which decimal digit is which; got {value!r}")
        return False, issues
    if not _VALUE_PATTERN.match(value):
        issues.append(f"'value' must look like a plain decimal with a fractional part, e.g. \"3.765\" "
                      f"or \"4.2626\" (a whole number has nothing to successively magnify); got {value!r}")
        return False, issues
    try:
        decimal_value = Decimal(value)
    except InvalidOperation:
        issues.append(f"'value' ({value!r}) could not be parsed as a decimal")
        return False, issues
    if decimal_value < 0:
        issues.append("'value' must be non-negative (this construction is defined for positive decimals)")
    decimal_digit_count = len(value.split(".")[1])
    if not (1 <= decimal_digit_count <= 6):
        issues.append(f"'value' has {decimal_digit_count} decimal digit(s) — must be between 1 and 6 "
                      f"for a renderable number of magnification rows")
    return (len(issues) == 0), issues


def compute_successive_magnification(spec: dict) -> dict:
    """Pure, exact Decimal arithmetic — no float involved until
    rendering. Returns every row's exact range/ticks/highlighted digit,
    plus a `reconstructed_value` that MUST equal the input (checked by
    verify_successive_magnification, not just asserted here)."""
    value_str = spec["value"]
    decimal_value = Decimal(value_str)
    int_str, frac_str = value_str.split(".")
    integer_part = Decimal(int_str)
    decimal_digits = [int(ch) for ch in frac_str]
    d = len(decimal_digits)

    rows = []
    # Row 0: integer overview, +/-3 around the integer part
    rows.append({
        "level": 0,
        "range_start": integer_part - 3,
        "range_end": integer_part + 3,
        "num_segments": 6,
        "tick_decimals": 0,
        "highlight_index": 3,  # the [integer_part, integer_part+1] segment, 3 ticks in from the left
    })

    prefix = integer_part
    for j in range(1, d + 1):
        step = Decimal(1).scaleb(-j)  # 10^-j, exact
        row_start = prefix
        row_end = prefix + Decimal(1).scaleb(-(j - 1))  # prefix + 10^-(j-1)
        digit = decimal_digits[j - 1]
        rows.append({
            "level": j,
            "range_start": row_start,
            "range_end": row_end,
            "num_segments": 10,
            "tick_decimals": j,
            "highlight_index": digit,
            "is_final": j == d,
        })
        prefix = row_start + digit * step

    reconstructed_value = integer_part + sum(
        (Decimal(digit) * Decimal(1).scaleb(-(i + 1)) for i, digit in enumerate(decimal_digits)),
        Decimal(0),
    )

    return {
        "value_str": value_str, "decimal_value": decimal_value, "integer_part": integer_part,
        "decimal_digits": decimal_digits, "d": d, "rows": rows,
        "final_point_value": prefix, "reconstructed_value": reconstructed_value,
    }


def verify_successive_magnification_construction(solved: dict) -> tuple:
    issues = []
    if solved["reconstructed_value"] != solved["decimal_value"]:
        issues.append(f"digit-path reconstruction gives {solved['reconstructed_value']} but the input "
                      f"value is {solved['decimal_value']} — a digit-extraction/indexing bug")
    if solved["final_point_value"] != solved["decimal_value"]:
        issues.append(f"the final marked point ({solved['final_point_value']}) does not equal the "
                      f"input value ({solved['decimal_value']})")
    # every row's highlighted sub-interval must be an exact subset of
    # the NEXT row's own [range_start, range_end] (the whole point of
    # "zooming in") — checked explicitly rather than assumed
    rows = solved["rows"]
    for i in range(len(rows) - 1):
        row, nxt = rows[i], rows[i + 1]
        seg = (row["range_end"] - row["range_start"]) / row["num_segments"]
        hi_start = row["range_start"] + row["highlight_index"] * seg
        hi_end = hi_start + seg
        if hi_start != nxt["range_start"] or hi_end != nxt["range_end"]:
            issues.append(f"row {row['level']}'s highlighted segment [{hi_start},{hi_end}] does not "
                          f"exactly match row {nxt['level']}'s range [{nxt['range_start']},{nxt['range_end']}]")
    return (len(issues) == 0), issues


def _fmt(value: Decimal, decimals: int) -> str:
    if decimals == 0:
        return str(int(value))
    return f"{value:.{decimals}f}"


def _render_successive_magnification(spec: dict) -> str:
    solved = compute_successive_magnification(spec)
    ok, issues = verify_successive_magnification_construction(solved)
    if not ok:
        return ""

    rows = solved["rows"]
    num_rows = len(rows)
    H = TOP_MARGIN + ROW_HEIGHT * num_rows + 14
    line_width = W - PAD_L - PAD_R

    def row_top(i):
        return TOP_MARGIN + i * ROW_HEIGHT

    def row_line_y(i):
        return row_top(i) + 40

    def tick_x(i, seg_index):
        row = rows[i]
        return PAD_L + line_width * (seg_index / row["num_segments"])

    body = []
    row_endpoints = []  # (left_x, right_x, line_y) per row, for guide lines

    for i, row in enumerate(rows):
        line_y = row_line_y(i)
        n = row["num_segments"]
        step = (row["range_end"] - row["range_start"]) / n
        x_left, x_right = tick_x(i, 0), tick_x(i, n)
        row_endpoints.append((x_left, x_right, line_y))

        body.append(f'<text x="{(x_left + x_right)/2:.1f}" y="{row_top(i)+12:.1f}" text-anchor="middle" '
                    f'fill="{AUX_COLOR}" {TITLE_FONT} data-role="row-title">'
                    f'{_escape(f"বিবৰ্ধন {i+1}")}</text>')

        # base line with arrowheads
        body.append(f'<line x1="{x_left-14:.1f}" y1="{line_y:.1f}" x2="{x_right+14:.1f}" y2="{line_y:.1f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="1.6" data-role="number-line"/>')
        for (ax, direction) in ((x_left - 14, -1), (x_right + 14, 1)):
            tip = ax + direction * 8
            body.append(f'<polygon points="{ax:.1f},{line_y-4:.1f} {ax:.1f},{line_y+4:.1f} {tip:.1f},{line_y:.1f}" '
                        f'fill="{LINE_COLOR}" data-role="arrowhead"/>')

        # ticks + labels
        for s in range(n + 1):
            x = tick_x(i, s)
            body.append(f'<line x1="{x:.1f}" y1="{line_y-5:.1f}" x2="{x:.1f}" y2="{line_y+5:.1f}" '
                        f'stroke="{LINE_COLOR}" stroke-width="1.3" data-role="tick"/>')
            tick_value = row["range_start"] + s * step
            body.append(f'<text x="{x:.1f}" y="{line_y+19:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                        f'{SMALL_FONT} data-role="tick-label">{_fmt(tick_value, row["tick_decimals"])}</text>')

        if not row.get("is_final"):
            hi = row["highlight_index"]
            hx1, hx2 = tick_x(i, hi), tick_x(i, hi + 1)
            body.append(f'<line x1="{hx1:.1f}" y1="{line_y:.1f}" x2="{hx2:.1f}" y2="{line_y:.1f}" '
                        f'stroke="{HIGHLIGHT_COLOR}" stroke-width="5" data-role="highlighted-segment"/>')
        else:
            hi = row["highlight_index"]
            px = tick_x(i, hi)
            body.append(f'<circle cx="{px:.1f}" cy="{line_y:.1f}" r="4" fill="{AUX_COLOR}" '
                        f'data-role="final-point"/>')
            body.append(f'<line x1="{px:.1f}" y1="{line_y+8:.1f}" x2="{px:.1f}" y2="{line_y+22:.1f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.6" data-role="final-point-drop"/>')
            body.append(f'<text x="{px:.1f}" y="{line_y+36:.1f}" text-anchor="middle" fill="{AUX_COLOR}" '
                        f'{FONT} data-role="final-point-label">{_escape(solved["value_str"])}</text>')

    # dashed guide lines connecting each row's highlighted segment to
    # the NEXT row's own two ends (visually: "this is what we zoomed into")
    for i in range(num_rows - 1):
        row = rows[i]
        line_y = row_endpoints[i][2]
        next_left, next_right, next_line_y = row_endpoints[i + 1]
        next_title_y = row_top(i + 1) - 2
        hi = row["highlight_index"]
        hx1, hx2 = tick_x(i, hi), tick_x(i, hi + 1)
        for (hx, nx) in ((hx1, next_left), (hx2, next_right)):
            body.append(f'<line x1="{hx:.1f}" y1="{line_y+6:.1f}" x2="{nx:.1f}" y2="{next_title_y:.1f}" '
                        f'stroke="{GUIDE_COLOR}" stroke-width="1" stroke-dasharray="3,3" '
                        f'data-role="guide-line"/>')

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="max-width:620px">' + "".join(body) + "</svg>")
    fixed = label_layout.check_and_fix_labels(svg)
    return fixed["svg"]


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="successive_magnification",
    renderer=_render_successive_magnification,
    schema_check=_validate_successive_magnification_spec,
    solver=None,
    description="Class 9 'Number System' successive-magnification construction: repeated 10-way "
                "zoom-in number-line rows locating a given decimal (e.g. 3.765 or 4.2626) exactly. "
                "'value' is given as a STRING to avoid floating-point digit corruption; all internal "
                "arithmetic uses exact Decimal math, and the digit path is independently reconstructed "
                "and compared against the input before any SVG is returned.",
))
