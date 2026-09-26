"""
probability_plugin.py — diagram_type #16: "probability".

Covers the two probability-diagram families school-level questions
actually need: probability TREE diagrams (independent, sequential
events — e.g. two coin tosses, drawing balls with/without
replacement) and SAMPLE-SPACE grids (e.g. the 6x6 grid for two dice).

Same split as every other diagram type in this project: Gemini
supplies each stage's branch labels and probabilities (as plain
fraction strings like "1/2") or each sample-space axis's values —
it never supplies the combined path probabilities or the grid's
computed cell values itself. Every leaf's combined probability
(product of the fractions along its path) and every sample-space
cell that asks to be auto-computed (sum/product of its row and
column) is computed here with Python's `fractions.Fraction`, so the
arithmetic is always exact, never a model's mental-math guess.

Wired in via diagram_plugin_registry.py, exactly like
polynomial_division_plugin.py.
"""
from fractions import Fraction

import diagram_plugin_registry

LINE_COLOR = "#1a4d8f"
LABEL_COLOR = "#111111"
GRID_LINE_COLOR = "#999999"
HEADER_FILL = "#eef3fa"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='600'"
SMALL_FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='11' font-weight='500'"
MARGIN = 16
MAX_STAGES = 4
MAX_BRANCHES_PER_STAGE = 6
MAX_GRID_SIZE = 8

_VALID_MODES = {"tree", "sample_space"}
_VALID_COMBINE = {"sum", "product", None}


def _is_fraction_string(s) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    try:
        Fraction(s)
        return True
    except (ValueError, ZeroDivisionError):
        return False


def _validate_probability_spec(spec: dict):
    """Structural validation for diagram_type == 'probability'. Checks
    SHAPE only, same contract as every other plugin's schema_check."""
    issues = []
    mode = spec.get("mode")
    if mode not in _VALID_MODES:
        issues.append(f"'mode' must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        return (not issues, issues)

    if mode == "tree":
        stages = spec.get("stages")
        if not isinstance(stages, list) or not stages:
            issues.append("'stages' must be a non-empty list")
            return (not issues, issues)
        if len(stages) > MAX_STAGES:
            issues.append(f"'stages' has {len(stages)} entries — more than {MAX_STAGES} would "
                          f"produce an unreadable tree (branch count grows multiplicatively)")
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict) or not isinstance(stage.get("branches"), list) \
                    or not stage["branches"]:
                issues.append(f"stages[{i}] must be an object with a non-empty 'branches' list")
                continue
            if len(stage["branches"]) > MAX_BRANCHES_PER_STAGE:
                issues.append(f"stages[{i}].branches has {len(stage['branches'])} entries — more "
                              f"than {MAX_BRANCHES_PER_STAGE}")
            for j, br in enumerate(stage["branches"]):
                if not isinstance(br, dict) or not br.get("label") or not _is_fraction_string(br.get("probability")):
                    issues.append(f"stages[{i}].branches[{j}] must have a non-empty 'label' and a "
                                  f"'probability' expressed as a fraction string like '1/2'")
    else:  # sample_space
        rows = spec.get("rows")
        cols = spec.get("cols")
        if not isinstance(rows, list) or not rows or len(rows) > MAX_GRID_SIZE:
            issues.append(f"'rows' must be a non-empty list of at most {MAX_GRID_SIZE} values")
        if not isinstance(cols, list) or not cols or len(cols) > MAX_GRID_SIZE:
            issues.append(f"'cols' must be a non-empty list of at most {MAX_GRID_SIZE} values")
        combine = spec.get("combine")
        cells = spec.get("cells")
        if combine not in _VALID_COMBINE and cells is None:
            issues.append(f"'combine' must be one of {sorted(c for c in _VALID_COMBINE if c)} "
                          f"unless explicit 'cells' are given")
        if cells is not None:
            if not isinstance(cells, list) or (rows and len(cells) != len(rows)):
                issues.append("'cells' must be a 2D list with one row per entry in 'rows'")
            elif cols:
                for r in cells:
                    if not isinstance(r, list) or len(r) != len(cols):
                        issues.append("each row of 'cells' must have one entry per entry in 'cols'")
                        break

    return (not issues, issues)


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_tree(spec: dict) -> str:
    stages = spec["stages"]

    # Build every root-to-leaf path first: each path is a list of
    # (label, probability_fraction) pairs, one per stage, in order.
    paths = [[]]
    for stage in stages:
        next_paths = []
        for path in paths:
            for br in stage["branches"]:
                next_paths.append(path + [(br["label"], Fraction(br["probability"]))])
        paths = next_paths

    n_leaves = len(paths)
    n_stages = len(stages)
    row_h = 34
    col_w = 90
    canvas_h = n_leaves * row_h + 2 * MARGIN
    canvas_w = (n_stages + 1) * col_w + 2 * MARGIN + 90  # +90 for the outcome/probability column

    body = []
    root_y = canvas_h / 2
    root_x = MARGIN
    body.append(f'<circle cx="{root_x:.1f}" cy="{root_y:.1f}" r="3" fill="{LINE_COLOR}"/>')

    # node_positions[stage_index] maps a path-prefix (tuple of labels) -> (x, y)
    prev_positions = {(): (root_x, root_y)}
    for s, stage in enumerate(stages):
        new_positions = {}
        # Evenly divide the full vertical span among all leaves, in the
        # order their prefixes were produced, so every node sits at the
        # vertical midpoint of the leaves beneath it.
        slot_h = (canvas_h - 2 * MARGIN) / max(1, n_leaves)
        leaf_cursor = 0
        for prefix in prev_positions.keys():
            px, py = prev_positions[prefix]
            for br in stage["branches"]:
                new_prefix = prefix + ((br["label"], Fraction(br["probability"])),)
                leaves_for_branch = _leaves_under_prefix(paths, new_prefix)
                mid = leaf_cursor + leaves_for_branch / 2
                cy = MARGIN + mid * slot_h
                cx = MARGIN + (s + 1) * col_w
                new_positions[new_prefix] = (cx, cy)
                body.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                            f'stroke="{LINE_COLOR}" stroke-width="1.6"/>')
                mid_x, mid_y = (px + cx) / 2, (py + cy) / 2 - 6
                body.append(f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" '
                            f'fill="{LABEL_COLOR}" {SMALL_FONT}>{_escape(br["probability"])}</text>')
                body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{LINE_COLOR}"/>')
                body.append(f'<text x="{cx + 8:.1f}" y="{cy - 8:.1f}" fill="{LABEL_COLOR}" {FONT}>'
                            f'{_escape(br["label"])}</text>')
                leaf_cursor += leaves_for_branch
        prev_positions = new_positions

    # Outcome + combined-probability column, one row per leaf, evenly spaced
    leaf_positions = sorted(prev_positions.items(), key=lambda kv: kv[1][1])
    for prefix, (cx, cy) in leaf_positions:
        outcome = "".join(label for label, _ in prefix)
        combined = Fraction(1)
        for _, p in prefix:
            combined *= p
        body.append(f'<text x="{cx + 20:.1f}" y="{cy:.1f}" fill="{LABEL_COLOR}" {FONT}>'
                    f'{_escape(outcome)}: {combined}</text>')

    return (f'<svg viewBox="0 0 {canvas_w:.1f} {canvas_h:.1f}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:340px">' + "".join(body) + "</svg>")


def _leaves_under_prefix(paths: list, prefix: tuple) -> int:
    depth = len(prefix)
    return sum(1 for p in paths if tuple(p[:depth]) == prefix)


def _render_sample_space(spec: dict) -> str:
    rows, cols = spec["rows"], spec["cols"]
    combine = spec.get("combine")
    cells = spec.get("cells")

    def cell_value(r, c):
        if cells is not None:
            return cells[rows.index(r)][cols.index(c)]
        if combine == "sum":
            return str(_to_number(r) + _to_number(c))
        if combine == "product":
            return str(_to_number(r) * _to_number(c))
        return f"({r},{c})"

    cell_w, cell_h = 40, 30
    header_w = 40
    canvas_w = header_w + len(cols) * cell_w + 2 * MARGIN
    canvas_h = header_w + len(rows) * cell_h + 2 * MARGIN

    body = []
    # corner + column headers
    body.append(f'<rect x="{MARGIN}" y="{MARGIN}" width="{header_w}" height="{header_w}" '
               f'fill="{HEADER_FILL}" stroke="{GRID_LINE_COLOR}"/>')
    for j, c in enumerate(cols):
        x = MARGIN + header_w + j * cell_w
        body.append(f'<rect x="{x}" y="{MARGIN}" width="{cell_w}" height="{header_w}" '
                   f'fill="{HEADER_FILL}" stroke="{GRID_LINE_COLOR}"/>')
        body.append(f'<text x="{x + cell_w / 2:.1f}" y="{MARGIN + header_w / 2 + 5:.1f}" '
                   f'text-anchor="middle" fill="{LABEL_COLOR}" {FONT}>{_escape(c)}</text>')
    for i, r in enumerate(rows):
        y = MARGIN + header_w + i * cell_h
        body.append(f'<rect x="{MARGIN}" y="{y}" width="{header_w}" height="{cell_h}" '
                   f'fill="{HEADER_FILL}" stroke="{GRID_LINE_COLOR}"/>')
        body.append(f'<text x="{MARGIN + header_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" '
                   f'text-anchor="middle" fill="{LABEL_COLOR}" {FONT}>{_escape(r)}</text>')
        for j, c in enumerate(cols):
            x = MARGIN + header_w + j * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                       f'fill="white" stroke="{GRID_LINE_COLOR}"/>')
            body.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" '
                       f'text-anchor="middle" fill="{LABEL_COLOR}" {SMALL_FONT}>'
                       f'{_escape(cell_value(r, c))}</text>')

    return (f'<svg viewBox="0 0 {canvas_w} {canvas_h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="max-width:340px">' + "".join(body) + "</svg>")


def _to_number(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0


def _render_probability(spec: dict) -> str:
    if spec["mode"] == "tree":
        return _render_tree(spec)
    return _render_sample_space(spec)


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="probability",
    renderer=_render_probability,
    schema_check=_validate_probability_spec,
    description="Probability tree diagrams (combined-path probabilities computed exactly via "
                "fractions.Fraction) and sample-space grids (e.g. two dice), mode-selected.",
))
