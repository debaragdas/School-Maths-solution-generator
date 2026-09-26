"""
set_theory_plugin.py — diagram_type #17: "venn_diagram".

Covers school-level set theory: 2-set or 3-set Venn diagrams, with an
optional universal set rectangle and an optional shaded region (union,
intersection, difference, complement, symmetric difference — 2-set
only, since a shaded region for 3 overlapping sets needs a genuinely
different SVG path per operation and isn't a school-level ask the way
the 2-set case is).

Same split as every other diagram type in this project: Gemini
supplies each set's elements — it never supplies which REGION each
element belongs in. Region membership (only-in-A, only-in-B,
in-both, in-neither) is computed here in Python purely from set
membership (an element string appearing in more than one set's
"elements" list is placed in the overlap), so an element can never
render inside a circle it wasn't actually declared to belong to.

Wired in via diagram_plugin_registry.py, exactly like
polynomial_division_plugin.py.
"""
import diagram_plugin_registry

LINE_COLOR = "#1a4d8f"
LABEL_COLOR = "#111111"
SHADE_COLOR = "#f4b942"
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='16' font-weight='700'"
ELEMENT_FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='500'"
MAX_SETS = 3
MAX_ELEMENTS_PER_SET = 12

_VALID_SHADE = {
    "union", "intersection", "A_only", "B_only",
    "complement_A", "complement_B", "symmetric_difference", None,
}


def _validate_venn_spec(spec: dict):
    """Structural validation for diagram_type == 'venn_diagram'. Checks
    SHAPE only, same contract as every other plugin's schema_check."""
    issues = []
    sets = spec.get("sets")
    if not isinstance(sets, list) or not (2 <= len(sets) <= MAX_SETS):
        issues.append(f"'sets' must be a list of 2 to {MAX_SETS} set objects")
        sets = []
    for i, s in enumerate(sets):
        if not isinstance(s, dict) or not s.get("id") or not s.get("label"):
            issues.append(f"sets[{i}] must be an object with non-empty 'id' and 'label'")
            continue
        elements = s.get("elements", [])
        if not isinstance(elements, list):
            issues.append(f"sets[{i}].elements must be a list")
        elif len(elements) > MAX_ELEMENTS_PER_SET:
            issues.append(f"sets[{i}].elements has {len(elements)} entries — more than "
                          f"{MAX_ELEMENTS_PER_SET} would overflow the diagram")

    shaded = spec.get("shaded_region")
    if shaded not in _VALID_SHADE:
        issues.append(f"'shaded_region' must be one of {sorted(v for v in _VALID_SHADE if v)}, got {shaded!r}")
    if shaded is not None and len(sets) != 2:
        issues.append("'shaded_region' is only supported for exactly 2 sets")

    universal = spec.get("universal_set")
    if universal is not None and not isinstance(universal, dict):
        issues.append("'universal_set' must be an object if given")

    return (not issues, issues)


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _region_for_element(elem: str, sets: list) -> frozenset:
    """Which sets (by id) genuinely declare this element — computed
    purely from membership, never trusted from any separate
    'region' field Gemini might otherwise have tried to supply."""
    return frozenset(s["id"] for s in sets if elem in s.get("elements", []))


def _render_venn_2set(spec: dict) -> str:
    sets = spec["sets"]
    a, b = sets[0], sets[1]
    shaded = spec.get("shaded_region")
    universal = spec.get("universal_set")

    W, H = 360, 240
    r = 80
    cx_a, cy = 150, 120
    cx_b = 210
    rect_pad = 20

    body = []
    if universal is not None:
        body.append(f'<rect x="{rect_pad}" y="{rect_pad}" width="{W - 2 * rect_pad}" '
                    f'height="{H - 2 * rect_pad}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6"/>')
        body.append(f'<text x="{rect_pad + 10}" y="{rect_pad + 20}" fill="{LABEL_COLOR}" {FONT}>'
                    f'{_escape(universal.get("label", "U"))}</text>')

    # Shading is drawn BEFORE the circle outlines so the outlines stay crisp on top.
    if shaded:
        clip_id_a, clip_id_b = "clipA", "clipB"
        body.append(f'<defs><clipPath id="{clip_id_a}"><circle cx="{cx_a}" cy="{cy}" r="{r}"/></clipPath>'
                    f'<clipPath id="{clip_id_b}"><circle cx="{cx_b}" cy="{cy}" r="{r}"/></clipPath></defs>')
        if shaded == "union":
            body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="{SHADE_COLOR}" fill-opacity="0.45"/>')
            body.append(f'<circle cx="{cx_b}" cy="{cy}" r="{r}" fill="{SHADE_COLOR}" fill-opacity="0.45"/>')
        elif shaded == "intersection":
            body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="{SHADE_COLOR}" fill-opacity="0.6" '
                        f'clip-path="url(#{clip_id_b})"/>')
        elif shaded == "A_only":
            body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="{SHADE_COLOR}" fill-opacity="0.5"/>')
            body.append(f'<circle cx="{cx_b}" cy="{cy}" r="{r}" fill="white" clip-path="url(#{clip_id_a})"/>')
        elif shaded == "B_only":
            body.append(f'<circle cx="{cx_b}" cy="{cy}" r="{r}" fill="{SHADE_COLOR}" fill-opacity="0.5"/>')
            body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="white" clip-path="url(#{clip_id_b})"/>')
        elif shaded == "complement_A":
            body.append(f'<rect x="{rect_pad}" y="{rect_pad}" width="{W - 2 * rect_pad}" '
                        f'height="{H - 2 * rect_pad}" fill="{SHADE_COLOR}" fill-opacity="0.45"/>')
            body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="white"/>')
        elif shaded == "complement_B":
            body.append(f'<rect x="{rect_pad}" y="{rect_pad}" width="{W - 2 * rect_pad}" '
                        f'height="{H - 2 * rect_pad}" fill="{SHADE_COLOR}" fill-opacity="0.45"/>')
            body.append(f'<circle cx="{cx_b}" cy="{cy}" r="{r}" fill="white"/>')
        elif shaded == "symmetric_difference":
            body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="{SHADE_COLOR}" fill-opacity="0.5"/>')
            body.append(f'<circle cx="{cx_b}" cy="{cy}" r="{r}" fill="{SHADE_COLOR}" fill-opacity="0.5"/>')
            body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="white" clip-path="url(#{clip_id_b})"/>')

    body.append(f'<circle cx="{cx_a}" cy="{cy}" r="{r}" fill="none" stroke="{LINE_COLOR}" stroke-width="2"/>')
    body.append(f'<circle cx="{cx_b}" cy="{cy}" r="{r}" fill="none" stroke="{LINE_COLOR}" stroke-width="2"/>')
    body.append(f'<text x="{cx_a - r - 6}" y="{cy - r + 10}" fill="{LABEL_COLOR}" {FONT}>{_escape(a["label"])}</text>')
    body.append(f'<text x="{cx_b + r - 12}" y="{cy - r + 10}" fill="{LABEL_COLOR}" {FONT}>{_escape(b["label"])}</text>')

    all_elems = list(dict.fromkeys(list(a.get("elements", [])) + list(b.get("elements", []))))
    only_a, only_b, both = [], [], []
    for e in all_elems:
        region = _region_for_element(e, sets)
        if region == {a["id"], b["id"]}:
            both.append(e)
        elif region == {a["id"]}:
            only_a.append(e)
        elif region == {b["id"]}:
            only_b.append(e)

    def place(elements, cx, top):
        y = top
        for e in elements:
            body.append(f'<text x="{cx:.1f}" y="{y:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                        f'{ELEMENT_FONT}>{_escape(e)}</text>')
            y += 16

    place(only_a, cx_a - 35, cy - (len(only_a) * 16) / 2)
    place(only_b, cx_b + 35, cy - (len(only_b) * 16) / 2)
    place(both, (cx_a + cx_b) / 2, cy - (len(both) * 16) / 2)

    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:340px">'
            + "".join(body) + "</svg>")


def _render_venn_3set(spec: dict) -> str:
    sets = spec["sets"]
    universal = spec.get("universal_set")
    W, H = 380, 300
    r = 90
    # Standard symmetric 3-circle layout
    centers = [(150, 120), (210, 120), (180, 175)]
    rect_pad = 20

    body = []
    if universal is not None:
        body.append(f'<rect x="{rect_pad}" y="{rect_pad}" width="{W - 2 * rect_pad}" '
                    f'height="{H - 2 * rect_pad}" fill="none" stroke="{LINE_COLOR}" stroke-width="1.6"/>')
        body.append(f'<text x="{rect_pad + 10}" y="{rect_pad + 20}" fill="{LABEL_COLOR}" {FONT}>'
                    f'{_escape(universal.get("label", "U"))}</text>')

    label_offsets = [(-r - 6, -r + 20), (r - 14, -r + 20), (-6, r + 20)]
    for (cx, cy), s, (lox, loy) in zip(centers, sets, label_offsets):
        body.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{LINE_COLOR}" '
                    f'stroke-width="2"/>')
        body.append(f'<text x="{cx + lox}" y="{cy + loy}" fill="{LABEL_COLOR}" {FONT}>'
                    f'{_escape(s["label"])}</text>')

    all_elems = list(dict.fromkeys(e for s in sets for e in s.get("elements", [])))
    ids = [s["id"] for s in sets]
    # A small fixed set of representative placement points per region
    # (only-A, only-B, only-C, AB, AC, BC, ABC), tuned for the layout above.
    region_points = {
        frozenset({ids[0]}): (120, 95),
        frozenset({ids[1]}): (240, 95),
        frozenset({ids[2]}): (180, 220),
        frozenset({ids[0], ids[1]}): (180, 90),
        frozenset({ids[0], ids[2]}): (140, 170),
        frozenset({ids[1], ids[2]}): (220, 170),
        frozenset({ids[0], ids[1], ids[2]}): (180, 140),
    }
    buckets = {k: [] for k in region_points}
    for e in all_elems:
        region = _region_for_element(e, sets)
        if region in buckets:
            buckets[region].append(e)

    for region, elems in buckets.items():
        cx, cy = region_points[region]
        y = cy - (len(elems) * 16) / 2
        for e in elems:
            body.append(f'<text x="{cx:.1f}" y="{y:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                        f'{ELEMENT_FONT}>{_escape(e)}</text>')
            y += 16

    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:340px">'
            + "".join(body) + "</svg>")


def _render_venn(spec: dict) -> str:
    if len(spec["sets"]) == 3:
        return _render_venn_3set(spec)
    return _render_venn_2set(spec)


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="venn_diagram",
    renderer=_render_venn,
    schema_check=_validate_venn_spec,
    description="2-set or 3-set Venn diagrams with optional universal-set rectangle and optional "
                "shaded region (2-set only); region membership is always computed from set "
                "membership, never trusted from a separate field.",
))
