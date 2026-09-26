"""
polynomial_division_plugin.py — diagram_type #13: "polynomial_long_division".

PRODUCTION-AUDIT FIX (this round): a polynomial long-division layout
(divisor | dividend, quotient on top, repeated subtract-and-bring-down
rows each under their own horizontal rule) had NO dedicated rendering
support anywhere in this pipeline before now — it was left entirely to
Gemini writing freeform LaTeX (\\require{enclose}/nested \\begin{array}
tricks) directly into a solution step's text, rendered by MathJax. That
combination is notoriously inconsistent for exactly this layout (column
alignment, rule widths, and indentation all depend on MathJax picking
compatible metrics for hand-written array/enclose LaTeX), which is what
produced the "wrong margin format" reported in production.

This plugin instead renders the whole layout as plain, deterministic
SVG — every row's vertical position, every horizontal rule's exact
width, and every column's indentation is computed here in Python, the
exact same "Gemini supplies structure, this renderer computes every
coordinate" split every other diagram type in this project already
uses (see diagram_renderer.py's own module docstring). Math notation
within each row (exponents, pi, fractions) is plain Unicode text (x2,
x3, pi, sqrt, +/-, x, / -- NOT LaTeX) for the same reason every other
diagram type's labels already are (e.g. an angle's "60 degree" label)
-- SVG <text> doesn't run through MathJax, and this project already
has zero LaTeX inside any diagram_spec label anywhere.

Wired in via diagram_plugin_registry.py (Phase 4's extension point for
diagram type #13 onward) rather than editing diagram_renderer.py's own
_RENDERERS dict / validate_diagram_spec dispatch chain directly — so
none of the 12 built-in diagram types are touched by this at all.
"""
import diagram_plugin_registry

LINE_COLOR = "#1a4d8f"
LABEL_COLOR = "#111111"
FONT_SIZE = 15
FONT = f"font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='{FONT_SIZE}' font-weight='600'"
LINE_HEIGHT = 28
RULE_GAP = 6          # vertical gap between a row's text baseline and its own horizontal rule
INDENT_STEP = 22       # how much further right each successive division step's rows sit
MARGIN = 16
MAX_STEPS = 8          # sanity cap — a real Class 9/10 polynomial division never needs more


def _text_width(text: str) -> float:
    """Same rough glyph-width estimate diagram_renderer.py's own
    _estimate_text_width uses for this project's font stack — kept as
    a local copy rather than an import specifically to avoid this
    plugin depending on diagram_renderer.py's internals (plugins are
    meant to be self-contained; see diagram_plugin_registry.py's own
    docstring on why the 12 built-in types are out of scope for this
    module, and symmetrically, why this module shouldn't reach back
    into diagram_renderer.py's private helpers either)."""
    return len(text) * FONT_SIZE * 0.62


def _validate_polynomial_long_division_spec(spec: dict):
    """Structural validation for diagram_type == 'polynomial_long_division'.
    Deliberately permissive about the MATH content (this plugin has no
    way to check that a subtraction is arithmetically correct — that's
    the solver's job, already covered by solver.py's own self-
    verification pass) and strict only about SHAPE: every field the
    renderer needs must actually be present as a non-empty string, and
    the step count must be sane, so a malformed spec fails validation
    (falls back to no diagram) rather than the renderer crashing or
    silently drawing something empty/broken."""
    issues = []

    def _nonempty_str(value, field_name):
        if not isinstance(value, str) or not value.strip():
            issues.append(f"'{field_name}' must be a non-empty string")

    _nonempty_str(spec.get("divisor"), "divisor")
    _nonempty_str(spec.get("dividend"), "dividend")
    _nonempty_str(spec.get("quotient"), "quotient")

    steps = spec.get("steps")
    if not isinstance(steps, list) or not steps:
        issues.append("'steps' must be a non-empty array")
    else:
        if len(steps) > MAX_STEPS:
            issues.append(f"'steps' has {len(steps)} entries — more than {MAX_STEPS} suggests a "
                           f"malformed/looping spec rather than a real division")
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(f"steps[{i}] must be an object")
                continue
            _nonempty_str(step.get("subtract"), f"steps[{i}].subtract")
            _nonempty_str(step.get("remainder"), f"steps[{i}].remainder")

    return (not issues, issues)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_polynomial_long_division(spec: dict) -> str:
    divisor = spec["divisor"]
    dividend = spec["dividend"]
    quotient = spec["quotient"]
    steps = spec["steps"]

    # ---- Layout pass: decide every row's (text, x, y, indent_level) up
    # front, so canvas width/height can be sized exactly around the real
    # content instead of guessing a fixed box and hoping nothing clips
    # or leaves excess blank margin (the literal "wrong margin format"
    # complaint this plugin exists to fix). ----
    divisor_w = _text_width(divisor)
    bar_x = MARGIN + divisor_w + 14  # the long-division vertical bracket sits here
    content_x0 = bar_x + 10          # dividend/quotient start just right of the bracket

    rows = []  # each: {"text": str, "x": float, "y": float, "rule_after": bool, "rule_width": float}
    y = MARGIN + FONT_SIZE
    quotient_y = y
    rows.append({"text": quotient, "x": content_x0, "y": y, "rule_after": False})
    y += LINE_HEIGHT
    rows.append({"text": dividend, "x": content_x0, "y": y, "rule_after": False})

    for i, step in enumerate(steps):
        indent = content_x0 + INDENT_STEP * (i + 1)
        y += LINE_HEIGHT
        subtract_text = f"\u2212 {step['subtract']}"  # real minus sign, not a hyphen
        rows.append({"text": subtract_text, "x": indent, "y": y,
                     "rule_after": True, "rule_width": max(_text_width(subtract_text), 40)})
        y += LINE_HEIGHT
        rows.append({"text": step["remainder"], "x": indent, "y": y, "rule_after": False})

    # roof (horizontal rule under the quotient, over the dividend) spans
    # from the bracket to the widest of the quotient/dividend rows.
    roof_width = max(_text_width(quotient), _text_width(dividend)) + 14
    roof_y = quotient_y + RULE_GAP

    canvas_w = MARGIN + max(
        (r["x"] + _text_width(r["text"]) for r in rows), default=content_x0
    ) + MARGIN
    canvas_w = max(canvas_w, bar_x + roof_width + MARGIN)
    canvas_h = y + MARGIN

    body = []
    # Divisor, vertically centered on the dividend row (not the quotient
    # row) — matching how a long-division bracket is conventionally
    # drawn.
    dividend_row_y = rows[1]["y"]
    body.append(f'<text x="{bar_x - 10:.1f}" y="{dividend_row_y:.1f}" text-anchor="end" '
                f'fill="{LABEL_COLOR}" {FONT}>{_escape(divisor)}</text>')

    # The bracket itself: vertical stroke down the left, horizontal roof
    # across the top — a plain right-angle long-division symbol.
    bracket_top = quotient_y - FONT_SIZE
    bracket_bottom = dividend_row_y + 6
    body.append(f'<path d="M {bar_x:.1f} {bracket_top:.1f} L {bar_x:.1f} {bracket_bottom:.1f}" '
                f'stroke="{LINE_COLOR}" stroke-width="1.6" fill="none"/>')
    body.append(f'<line x1="{bar_x:.1f}" y1="{roof_y:.1f}" x2="{bar_x + roof_width:.1f}" '
                f'y2="{roof_y:.1f}" stroke="{LINE_COLOR}" stroke-width="1.6"/>')

    for r in rows:
        body.append(f'<text x="{r["x"]:.1f}" y="{r["y"]:.1f}" fill="{LABEL_COLOR}" {FONT}>'
                    f'{_escape(r["text"])}</text>')
        if r["rule_after"]:
            rule_y = r["y"] + RULE_GAP
            body.append(f'<line x1="{r["x"]:.1f}" y1="{rule_y:.1f}" '
                        f'x2="{r["x"] + r["rule_width"]:.1f}" y2="{rule_y:.1f}" '
                        f'stroke="{LINE_COLOR}" stroke-width="1.3"/>')

    return (f'<svg viewBox="0 0 {canvas_w:.1f} {canvas_h:.1f}" xmlns="http://www.w3.org/2000/svg">'
            + "".join(body) + "</svg>")


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="polynomial_long_division",
    renderer=_render_polynomial_long_division,
    schema_check=_validate_polynomial_long_division_spec,
    description="Divisor/dividend/quotient long-division layout with per-step subtract+remainder rows.",
))
