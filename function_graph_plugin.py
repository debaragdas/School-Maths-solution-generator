"""
function_graph_plugin.py — diagram_type #15: "function_graph".

Covers the graph families the existing "coordinate_plot" type does NOT
handle: exponential, logarithmic, absolute value, and piecewise
functions (linear/constant pieces over disjoint domains) — the
school-level graph-sketching questions beyond plotting a handful of
discrete points or a straight line.

Same split as every other diagram type in this project: Gemini
supplies the function's SHAPE (kind + numeric parameters + the x-range
to sketch over) — it never supplies sampled (x, y) pairs itself. Every
single plotted point is computed here in Python with the `math`
module, so an exponential curve is guaranteed to actually be
exponential rather than whatever curve-ish shape a model free-hands.

Wired in via diagram_plugin_registry.py, exactly like
polynomial_division_plugin.py.
"""
import math

import diagram_plugin_registry

LINE_COLOR = "#1a4d8f"
AXIS_COLOR = "#666666"
GRID_COLOR = "#dddddd"
LABEL_COLOR = "#111111"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='14' font-weight='600'"
MARGIN = 30
PX_PER_UNIT = 26
SAMPLES_PER_UNIT = 20   # curve smoothness
MAX_PIECES = 6

_VALID_KINDS = {"exponential", "logarithmic", "absolute_value", "piecewise"}
_VALID_PIECE_EXPR = {"linear", "constant"}


def _is_number(v) -> bool:
    # V37 hardening pass: math.isfinite explicitly excludes NaN/inf.
    # isinstance-only checks let NaN through, and every subsequent
    # "<= 0 is invalid" comparison against NaN is silently False in
    # Python, so NaN previously bypassed validation undetected.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate_function_graph_spec(spec: dict):
    """Structural validation for diagram_type == 'function_graph'. Same
    contract as every other plugin's schema_check: checks SHAPE, not
    whether the chosen parameters are the "right" ones for the actual
    question."""
    issues = []

    fn = spec.get("function")
    if not isinstance(fn, dict):
        issues.append("'function' must be an object")
        return (not issues, issues)

    kind = fn.get("kind")
    if kind not in _VALID_KINDS:
        issues.append(f"function.kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}")
        return (not issues, issues)

    if kind == "exponential":
        if not _is_number(fn.get("base")) or fn.get("base") <= 0 or fn.get("base") == 1:
            issues.append("exponential function.base must be a positive number != 1")
        for opt in ("a", "c"):
            if opt in fn and not _is_number(fn[opt]):
                issues.append(f"exponential function.{opt} must be numeric if given")
    elif kind == "logarithmic":
        if not _is_number(fn.get("base")) or fn.get("base") <= 0 or fn.get("base") == 1:
            issues.append("logarithmic function.base must be a positive number != 1")
        for opt in ("a", "c"):
            if opt in fn and not _is_number(fn[opt]):
                issues.append(f"logarithmic function.{opt} must be numeric if given")
    elif kind == "absolute_value":
        if not _is_number(fn.get("a")):
            issues.append("absolute_value function.a must be numeric")
        for opt in ("h", "k"):
            if opt in fn and not _is_number(fn[opt]):
                issues.append(f"absolute_value function.{opt} must be numeric if given")
    elif kind == "piecewise":
        pieces = fn.get("pieces")
        if not isinstance(pieces, list) or not pieces:
            issues.append("piecewise function.pieces must be a non-empty list")
        elif len(pieces) > MAX_PIECES:
            issues.append(f"piecewise function.pieces has {len(pieces)} entries — more than "
                          f"{MAX_PIECES} suggests a malformed spec")
        else:
            for i, piece in enumerate(pieces):
                if not isinstance(piece, dict):
                    issues.append(f"pieces[{i}] must be an object")
                    continue
                if piece.get("expr") not in _VALID_PIECE_EXPR:
                    issues.append(f"pieces[{i}].expr must be one of {sorted(_VALID_PIECE_EXPR)}")
                domain = piece.get("domain")
                if not isinstance(domain, list) or len(domain) != 2 \
                        or not all(_is_number(d) for d in domain) or domain[0] >= domain[1]:
                    issues.append(f"pieces[{i}].domain must be an increasing [lo, hi] pair")
                if piece.get("expr") == "linear" and (not _is_number(piece.get("a")) or not _is_number(piece.get("b"))):
                    issues.append(f"pieces[{i}] (linear) must provide numeric 'a' and 'b' (y = a*x + b)")
                if piece.get("expr") == "constant" and not _is_number(piece.get("c")):
                    issues.append(f"pieces[{i}] (constant) must provide numeric 'c'")

    x_range = spec.get("x_range")
    if x_range is not None:
        if not isinstance(x_range, list) or len(x_range) != 2 or not all(_is_number(v) for v in x_range) \
                or x_range[0] >= x_range[1]:
            issues.append("'x_range' must be an increasing [lo, hi] numeric pair")

    return (not issues, issues)


def _eval_exponential(x: float, fn: dict) -> float:
    a, base, c = fn.get("a", 1), fn["base"], fn.get("c", 0)
    return a * (base ** x) + c


def _eval_logarithmic(x: float, fn: dict):
    if x <= 0:
        return None  # undefined — caller breaks the polyline here
    a, base, c = fn.get("a", 1), fn["base"], fn.get("c", 0)
    return a * (math.log(x) / math.log(base)) + c


def _eval_absolute_value(x: float, fn: dict) -> float:
    a, h, k = fn["a"], fn.get("h", 0), fn.get("k", 0)
    return a * abs(x - h) + k


def _default_x_range(kind: str, fn: dict) -> tuple:
    if kind == "logarithmic":
        return (0.1, 8)
    if kind == "piecewise":
        pieces = fn["pieces"]
        return (min(p["domain"][0] for p in pieces), max(p["domain"][1] for p in pieces))
    return (-5, 5)


def _sample_curve(kind: str, fn: dict, x_lo: float, x_hi: float) -> list:
    """Returns a list of (x, y)-or-None polylines (None marks a break,
    e.g. logarithmic's undefined x <= 0 region or a piecewise gap)."""
    if kind == "piecewise":
        polylines = []
        for piece in fn["pieces"]:
            plo, phi = piece["domain"]
            n = max(2, int((phi - plo) * SAMPLES_PER_UNIT))
            pts = []
            for i in range(n + 1):
                x = plo + (phi - plo) * i / n
                y = piece["a"] * x + piece["b"] if piece["expr"] == "linear" else piece["c"]
                pts.append((x, y))
            polylines.append(pts)
        return polylines

    n = max(2, int((x_hi - x_lo) * SAMPLES_PER_UNIT))
    pts = []
    polylines = []
    for i in range(n + 1):
        x = x_lo + (x_hi - x_lo) * i / n
        if kind == "exponential":
            y = _eval_exponential(x, fn)
        elif kind == "absolute_value":
            y = _eval_absolute_value(x, fn)
        else:
            y = _eval_logarithmic(x, fn)
        if y is None:
            if pts:
                polylines.append(pts)
                pts = []
            continue
        pts.append((x, y))
    if pts:
        polylines.append(pts)
    return polylines


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_function_graph(spec: dict) -> str:
    fn = spec["function"]
    kind = fn["kind"]
    x_lo, x_hi = spec.get("x_range") or _default_x_range(kind, fn)

    polylines = _sample_curve(kind, fn, x_lo, x_hi)
    all_y = [y for line in polylines for _, y in line]
    y_range = spec.get("y_range")
    if y_range:
        y_lo, y_hi = y_range
    elif all_y:
        y_lo, y_hi = min(all_y), max(all_y)
        pad = max((y_hi - y_lo) * 0.1, 1)
        y_lo, y_hi = y_lo - pad, y_hi + pad
    else:
        y_lo, y_hi = -5, 5

    lo_x_grid, hi_x_grid = math.floor(x_lo), math.ceil(x_hi)
    lo_y_grid, hi_y_grid = math.floor(y_lo), math.ceil(y_hi)

    canvas_w = (hi_x_grid - lo_x_grid) * PX_PER_UNIT + 2 * MARGIN
    canvas_h = (hi_y_grid - lo_y_grid) * PX_PER_UNIT + 2 * MARGIN

    def to_px(x: float, y: float) -> tuple:
        px = MARGIN + (x - lo_x_grid) * PX_PER_UNIT
        py = canvas_h - MARGIN - (y - lo_y_grid) * PX_PER_UNIT
        return px, py

    body = []
    for gx in range(lo_x_grid, hi_x_grid + 1):
        px, _ = to_px(gx, 0)
        body.append(f'<line x1="{px:.1f}" y1="{MARGIN:.1f}" x2="{px:.1f}" y2="{canvas_h - MARGIN:.1f}" '
                    f'stroke="{GRID_COLOR}" stroke-width="1"/>')
    for gy in range(lo_y_grid, hi_y_grid + 1):
        _, py = to_px(0, gy)
        body.append(f'<line x1="{MARGIN:.1f}" y1="{py:.1f}" x2="{canvas_w - MARGIN:.1f}" y2="{py:.1f}" '
                    f'stroke="{GRID_COLOR}" stroke-width="1"/>')

    if lo_x_grid <= 0 <= hi_x_grid:
        px, _ = to_px(0, 0)
        body.append(f'<line x1="{px:.1f}" y1="{MARGIN:.1f}" x2="{px:.1f}" y2="{canvas_h - MARGIN:.1f}" '
                    f'stroke="{AXIS_COLOR}" stroke-width="1.6"/>')
    if lo_y_grid <= 0 <= hi_y_grid:
        _, py = to_px(0, 0)
        body.append(f'<line x1="{MARGIN:.1f}" y1="{py:.1f}" x2="{canvas_w - MARGIN:.1f}" y2="{py:.1f}" '
                    f'stroke="{AXIS_COLOR}" stroke-width="1.6"/>')

    for line in polylines:
        pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in (to_px(x, y) for x, y in line))
        body.append(f'<polyline points="{pts_str}" fill="none" stroke="{LINE_COLOR}" stroke-width="2.2"/>')

    label = spec.get("label")
    if label:
        body.append(f'<text x="{MARGIN:.1f}" y="{(MARGIN - 10):.1f}" fill="{LABEL_COLOR}" {FONT}>'
                    f'{_escape(label)}</text>')

    return (f'<svg viewBox="0 0 {canvas_w:.1f} {canvas_h:.1f}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:280px">' + "".join(body) + "</svg>")


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="function_graph",
    renderer=_render_function_graph,
    schema_check=_validate_function_graph_spec,
    description="Exponential/logarithmic/absolute-value/piecewise function graphs; every plotted "
                "point is computed in Python from the stated parameters, never sampled by the model.",
))
