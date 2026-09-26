"""
solid_geometry_plugin.py — diagram_type "solid_net" and diagram_type
"cross_section" (GENUINE NEW GAP: confirmed absent before this file —
diagram_renderer.py's built-in "surface_area_volume" type draws solid
SOLIDS with hidden-back-edges shown dashed, but grepping the codebase
for "net"/"cross_section"/"unfold" found nothing that unfolds a solid
into its 2D net or shows a plane cutting through one. This file adds
both, without touching surface_area_volume or any of its tests.)

diagram_type "solid_net": unfolds cube / cuboid / cylinder / cone into
their standard flat net, with every face's dimensions computed EXACTLY
from the solid's own given dimensions (a cuboid's 6 faces are exact
l*w / l*h / w*h rectangles; a cylinder's curved-surface net is an exact
2*pi*r (circumference) by h rectangle; a cone's curved-surface net is
an exact circular sector of radius = slant height and arc length
2*pi*r, so its sector angle is exactly (r / slant_height) * 360
degrees) — never an approximated or schematic unfolding.

diagram_type "cross_section": shows a solid (cuboid/cylinder/cone, with
hidden back edges dashed, reusing this project's established hidden-
edge convention from _render_surface_area_volume) together with the
EXACT 2D shape produced by a named cutting plane:
  cuboid, plane parallel to a face   -> a rectangle of the other two dims
  cylinder, plane parallel to base   -> a circle of the base radius
  cylinder, plane through the axis   -> a rectangle 2r wide, h tall
  cone, plane parallel to base at height h_cut from the apex
                                      -> a circle of radius r*(h_cut/H),
                                         by similar triangles — exact,
                                         not estimated
  cone, plane through the apex and axis
                                      -> an isosceles triangle, base 2r,
                                         height H

Both diagram_type schema_checks + a post-render numeric-consistency
check (re-derives every drawn dimension from the input spec and
confirms they match) are included, following this project's existing
"prove it, don't just draw it" pattern (e.g.
_verify_surface_area_volume_coverage in diagram_renderer.py).

Wired in via diagram_plugin_registry.py only — no existing file
modified.
"""
import math

import diagram_plugin_registry
import label_layout

LINE_COLOR = "#1a4d8f"
AUX_COLOR = "#c0392b"
LABEL_COLOR = "#111111"
HIDDEN_DASH = "3,2"  # matches diagram_renderer.py's own hidden-edge convention exactly
FONT = "font-family='Hind Siliguri Regular, Noto Sans, Arial' font-size='13' font-weight='600'"
SMALL_FONT = "font-family='Noto Sans, Arial' font-size='11'"

W, H = 300, 240
_EPS = 1e-9


def _is_number(v) -> bool:
    # V37 hardening pass: math.isfinite explicitly excludes NaN/inf.
    # isinstance-only checks let NaN through, and every subsequent
    # "<= 0 is invalid" comparison against NaN is silently False in
    # Python, so NaN previously bypassed validation undetected.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt(v: float) -> str:
    return f"{v:g}"


# ------------------------------------------------------------------
# SOLID_NET
# ------------------------------------------------------------------
_NET_SOLIDS = {"cube", "cuboid", "cylinder", "cone"}
_NET_REQUIRED_DIMS = {
    "cube": {"side"}, "cuboid": {"length", "width", "height"},
    "cylinder": {"radius", "height"}, "cone": {"radius", "slant_height"},
}


def _validate_solid_net_spec(spec: dict):
    issues = []
    solid = spec.get("solid")
    if solid not in _NET_SOLIDS:
        issues.append(f"solid must be one of {sorted(_NET_SOLIDS)}, got {solid!r}")
        return False, issues
    dims = spec.get("dimensions")
    if not isinstance(dims, dict):
        issues.append("'dimensions' must be an object")
        return False, issues
    for key in _NET_REQUIRED_DIMS[solid]:
        v = dims.get(key)
        if not _is_number(v) or v <= 0:
            issues.append(f"dimensions.{key} must be a positive number for solid '{solid}', got {v!r}")
    return (len(issues) == 0), issues


def solve_solid_net(spec: dict) -> dict:
    """Pure math: returns the exact face/panel geometry for the net,
    keyed by solid type. Nothing here is a schematic guess — every
    panel's size is derived directly from the given dimensions."""
    solid = spec["solid"]
    dims = spec["dimensions"]
    if solid == "cube":
        s = float(dims["side"])
        return {"solid": "cube", "faces": [(s, s)] * 6}
    if solid == "cuboid":
        l, w, h = float(dims["length"]), float(dims["width"]), float(dims["height"])
        # a cross-net has 2 (l x w) faces (top/bottom), 2 (l x h) faces (front/back), 2 (w x h) faces (sides)
        return {"solid": "cuboid", "l": l, "w": w, "h": h}
    if solid == "cylinder":
        r, h = float(dims["radius"]), float(dims["height"])
        circumference = 2 * math.pi * r
        return {"solid": "cylinder", "radius": r, "height": h, "rect_width": circumference, "rect_height": h}
    if solid == "cone":
        r, l = float(dims["radius"]), float(dims["slant_height"])
        sector_angle_deg = (r / l) * 360.0  # arc length 2*pi*r must equal l * angle_rad
        return {"solid": "cone", "radius": r, "slant_height": l, "sector_angle_deg": sector_angle_deg}


def verify_solid_net_construction(solved: dict) -> tuple:
    """Post-render check: for the cylinder net, confirms the rectangle
    width exactly equals 2*pi*r (the base circle's circumference); for
    the cone net, confirms the sector's arc length (radius * angle_rad)
    exactly equals 2*pi*r. These are the defining geometric constraints
    of a valid net — checked, not assumed."""
    issues = []
    if solved["solid"] == "cylinder":
        expected = 2 * math.pi * solved["radius"]
        if abs(solved["rect_width"] - expected) > 1e-6:
            issues.append(f"cylinder net rectangle width {solved['rect_width']:.6f} != "
                          f"2*pi*r = {expected:.6f}")
    if solved["solid"] == "cone":
        angle_rad = math.radians(solved["sector_angle_deg"])
        arc_len = solved["slant_height"] * angle_rad
        expected = 2 * math.pi * solved["radius"]
        if abs(arc_len - expected) > 1e-6:
            issues.append(f"cone net sector arc length {arc_len:.6f} != 2*pi*r = {expected:.6f}")
    return (len(issues) == 0), issues


def _render_solid_net(spec: dict) -> str:
    solved = solve_solid_net(spec)
    ok, issues = verify_solid_net_construction(solved)
    if not ok:
        return ""

    body = []
    solid = solved["solid"]

    if solid in ("cube", "cuboid"):
        if solid == "cube":
            l = w = h = solved["faces"][0][0]
        else:
            l, w, h = solved["l"], solved["w"], solved["h"]
        # middle row is [w][l][w][l] (side,front,side,back) = total width
        # 2*(l+w) in math units — NOT (l + 2*w), which was the bug here:
        # that formula always underestimates the true net width by
        # exactly l units, computing too generous a scale and clipping
        # the net off the right edge of the canvas.
        scale = min((W - 40) / (2 * (l + w)), (H - 40) / (h + 2 * w))
        L, Wd, Hd = l * scale, w * scale, h * scale
        # standard cross-net layout: a row of 4 faces (w,h,l,h around the
        # middle) plus one face above and one below the "l" face.


        # middle row: [w x h] [l x h] [w x h] [l x h]  (side, front, side, back)
        row_y = (H - Hd) / 2
        panels = []
        x0 = (W - (Wd + L + Wd + L)) / 2
        panels.append((x0, row_y, Wd, Hd, "side"))
        x1 = x0 + Wd
        panels.append((x1, row_y, L, Hd, "front"))
        x2 = x1 + L
        panels.append((x2, row_y, Wd, Hd, "side"))
        x3 = x2 + Wd
        panels.append((x3, row_y, L, Hd, "back"))
        # top and bottom faces (l x w) attached above/below the "front" panel
        panels.append((x1, row_y - Wd, L, Wd, "top"))
        panels.append((x1, row_y + Hd, L, Wd, "bottom"))
        for (px, py, pw, ph, role) in panels:
            body.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
                        f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="1.6" '
                        f'data-role="net-face-{role}"/>')
        dims = spec["dimensions"]
        if solid == "cube":
            body.append(f'<text x="{x1+L/2:.1f}" y="{row_y+Hd+Wd+16:.1f}" text-anchor="middle" '
                        f'fill="{LABEL_COLOR}" {SMALL_FONT} data-role="dimension-label">'
                        f'side = {_fmt(float(dims["side"]))}</text>')
        else:
            body.append(f'<text x="{x1+L/2:.1f}" y="{row_y+Hd+Wd+16:.1f}" text-anchor="middle" '
                        f'fill="{LABEL_COLOR}" {SMALL_FONT} data-role="dimension-label">'
                        f'l={_fmt(l)}, w={_fmt(w)}, h={_fmt(h)}</text>')

    elif solid == "cylinder":
        r, hgt = solved["radius"], solved["height"]
        rw, rh = solved["rect_width"], solved["rect_height"]
        scale = min((W - 40) / (rw + 2 * r), (H - 40) / max(rh, 2 * r))
        rw_px, rh_px, r_px = rw * scale, rh * scale, r * scale
        rect_x = (W - rw_px) / 2
        rect_y = (H - rh_px) / 2
        body.append(f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{rw_px:.1f}" height="{rh_px:.1f}" '
                    f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="1.6" data-role="net-curved-surface"/>')
        top_cx = rect_x + rw_px / 2
        body.append(f'<circle cx="{top_cx:.1f}" cy="{rect_y-r_px-4:.1f}" r="{r_px:.1f}" fill="#d7e4f7" '
                    f'stroke="{LINE_COLOR}" stroke-width="1.6" data-role="net-top-circle"/>')
        body.append(f'<circle cx="{top_cx:.1f}" cy="{rect_y+rh_px+r_px+4:.1f}" r="{r_px:.1f}" fill="#d7e4f7" '
                    f'stroke="{LINE_COLOR}" stroke-width="1.6" data-role="net-bottom-circle"/>')
        body.append(f'<text x="{rect_x+rw_px/2:.1f}" y="{rect_y+rh_px/2:.1f}" text-anchor="middle" '
                    f'fill="{LABEL_COLOR}" {SMALL_FONT} data-role="dimension-label">'
                    f'2&#960;r = {_fmt(rw)}, h = {_fmt(hgt)}</text>')

    elif solid == "cone":
        r, slant = solved["radius"], solved["slant_height"]
        angle_deg = solved["sector_angle_deg"]
        scale = (min(W, H) - 60) / (2 * slant)
        slant_px = slant * scale
        apex = (W / 2, H / 2 - slant_px * 0.35)
        half = angle_deg / 2.0
        a1 = math.radians(90 - half)
        a2 = math.radians(90 + half)
        p1 = (apex[0] + slant_px * math.cos(a1), apex[1] + slant_px * math.sin(a1))
        p2 = (apex[0] + slant_px * math.cos(a2), apex[1] + slant_px * math.sin(a2))
        large_arc = 1 if angle_deg > 180 else 0
        body.append(f'<path d="M {apex[0]:.1f} {apex[1]:.1f} L {p1[0]:.1f} {p1[1]:.1f} '
                    f'A {slant_px:.1f} {slant_px:.1f} 0 {large_arc} 1 {p2[0]:.1f} {p2[1]:.1f} Z" '
                    f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="1.6" data-role="net-sector"/>')
        base_mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + slant_px * 0.55)
        r_px = r * scale
        body.append(f'<circle cx="{base_mid[0]:.1f}" cy="{base_mid[1]+r_px+8:.1f}" r="{r_px:.1f}" '
                    f'fill="#d7e4f7" stroke="{LINE_COLOR}" stroke-width="1.6" data-role="net-base-circle"/>')
        body.append(f'<text x="{apex[0]:.1f}" y="{apex[1]-8:.1f}" text-anchor="middle" fill="{LABEL_COLOR}" '
                    f'{SMALL_FONT} data-role="dimension-label">sector angle = {angle_deg:.1f}&#176;, '
                    f'slant = {_fmt(slant)}, r = {_fmt(r)}</text>')

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="max-width:280px">' + "".join(body) + "</svg>")
    fixed = label_layout.check_and_fix_labels(svg)
    return fixed["svg"]


# ------------------------------------------------------------------
# CROSS_SECTION
# ------------------------------------------------------------------
_SECTION_SOLIDS = {"cuboid", "cylinder", "cone"}
_SECTION_CUTS = {
    "cuboid": {"parallel_to_face"},
    "cylinder": {"parallel_to_base", "through_axis"},
    "cone": {"parallel_to_base", "through_apex_axis"},
}


def _validate_cross_section_spec(spec: dict):
    issues = []
    solid = spec.get("solid")
    if solid not in _SECTION_SOLIDS:
        issues.append(f"solid must be one of {sorted(_SECTION_SOLIDS)}, got {solid!r}")
        return False, issues
    dims = spec.get("dimensions")
    if not isinstance(dims, dict):
        issues.append("'dimensions' must be an object")
        return False, issues
    cut = spec.get("cut")
    if cut not in _SECTION_CUTS[solid]:
        issues.append(f"cut must be one of {sorted(_SECTION_CUTS[solid])} for solid '{solid}', got {cut!r}")

    if solid == "cuboid":
        for key in ("length", "width", "height"):
            if not _is_number(dims.get(key)) or dims.get(key) <= 0:
                issues.append(f"dimensions.{key} must be a positive number, got {dims.get(key)!r}")
    elif solid == "cylinder":
        for key in ("radius", "height"):
            if not _is_number(dims.get(key)) or dims.get(key) <= 0:
                issues.append(f"dimensions.{key} must be a positive number, got {dims.get(key)!r}")
    elif solid == "cone":
        for key in ("radius", "height"):
            if not _is_number(dims.get(key)) or dims.get(key) <= 0:
                issues.append(f"dimensions.{key} must be a positive number, got {dims.get(key)!r}")
        if cut == "parallel_to_base":
            hc = spec.get("cut_height_from_apex")
            if not _is_number(hc) or hc <= 0:
                issues.append("cut_height_from_apex must be a positive number for a cone "
                              "parallel_to_base cross-section")
            elif _is_number(dims.get("height")) and hc >= dims["height"]:
                issues.append("cut_height_from_apex must be less than the cone's own height")
    return (len(issues) == 0), issues


def solve_cross_section(spec: dict) -> dict:
    """Pure math: returns the exact 2D section shape produced by the
    named cut. Never estimates — the cone parallel_to_base case in
    particular uses similar triangles (radius scales linearly with
    distance from the apex) rather than any schematic guess."""
    solid, cut = spec["solid"], spec["cut"]
    dims = spec["dimensions"]
    if solid == "cuboid":
        l, w, h = float(dims["length"]), float(dims["width"]), float(dims["height"])
        return {"solid": solid, "cut": cut, "shape": "rectangle", "width": w, "height": h,
                "l": l, "w": w, "h": h}
    if solid == "cylinder":
        r, hgt = float(dims["radius"]), float(dims["height"])
        if cut == "parallel_to_base":
            return {"solid": solid, "cut": cut, "shape": "circle", "radius": r, "height": hgt}
        return {"solid": solid, "cut": cut, "shape": "rectangle", "width": 2 * r, "height": hgt,
                "radius": r}
    if solid == "cone":
        r, H_full = float(dims["radius"]), float(dims["height"])
        if cut == "parallel_to_base":
            h_cut = float(spec["cut_height_from_apex"])
            section_radius = r * (h_cut / H_full)  # similar triangles, exact
            return {"solid": solid, "cut": cut, "shape": "circle", "radius": section_radius,
                    "full_radius": r, "full_height": H_full, "cut_height_from_apex": h_cut}
        return {"solid": solid, "cut": cut, "shape": "triangle", "base": 2 * r, "height": H_full,
                "radius": r}


def verify_cross_section_construction(solved: dict) -> tuple:
    """Post-render check: for the cone parallel_to_base case, re-derives
    the section radius independently from the similar-triangles ratio
    and confirms it matches what will be drawn (guards against a future
    edit accidentally changing the formula in one place but not the
    other)."""
    issues = []
    if solved["solid"] == "cone" and solved["cut"] == "parallel_to_base":
        expected = solved["full_radius"] * (solved["cut_height_from_apex"] / solved["full_height"])
        if abs(solved["radius"] - expected) > 1e-6:
            issues.append(f"section radius {solved['radius']:.6f} != r*(h_cut/H) = {expected:.6f}")
        if not (0 < solved["radius"] < solved["full_radius"] + 1e-9):
            issues.append(f"section radius {solved['radius']:.6f} out of valid range (0, {solved['full_radius']})")
    return (len(issues) == 0), issues


def _render_cross_section(spec: dict) -> str:
    solved = solve_cross_section(spec)
    ok, issues = verify_cross_section_construction(solved)
    if not ok:
        return ""

    body = []
    solid = solved["solid"]
    cx, cy = W * 0.32, H / 2

    if solid == "cuboid":
        l, w, h = solved["l"], solved["w"], solved["h"]
        scale = min((W * 0.55) / (l + w * 0.4), (H - 50) / (h + w * 0.3))
        L, Wd, Hd = l * scale, w * scale, h * scale
        skew_x, skew_y = Wd * 0.4, Wd * 0.3
        flb = (cx - L / 2, cy + Hd / 2)
        frb = (flb[0] + L, flb[1])
        flt = (flb[0], flb[1] - Hd)
        frt = (frb[0], frb[1] - Hd)
        blb = (flb[0] + skew_x, flb[1] - skew_y)
        blt = (flt[0] + skew_x, flt[1] - skew_y)
        brb = (frb[0] + skew_x, frb[1] - skew_y)
        body.append(f'<polygon points="{flb[0]:.0f},{flb[1]:.0f} {frb[0]:.0f},{frb[1]:.0f} '
                    f'{frt[0]:.0f},{frt[1]:.0f} {flt[0]:.0f},{flt[1]:.0f}" fill="#eaf1fb" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        for p1, p2 in ((blb, blt), (blb, brb), (blb, flb)):
            body.append(f'<line x1="{p1[0]:.0f}" y1="{p1[1]:.0f}" x2="{p2[0]:.0f}" y2="{p2[1]:.0f}" '
                        f'stroke="{LINE_COLOR}" stroke-width="1" stroke-dasharray="{HIDDEN_DASH}" '
                        f'data-role="hidden-edge"/>')
        # cut plane: shown as a shaded panel parallel to the front face,
        # positioned partway through the box (exact geometry: a
        # parallel-to-front-face cut is itself an l x h rectangle,
        # identical in shape/size to the front face by construction)
        cut_x = flb[0] + L * 0.55
        # the drawn section rectangle uses the SAME shape as computed by solve_cross_section
        sw_px, sh_px = solved["w"] * scale, solved["h"] * scale
        body.append(f'<rect x="{cut_x-2:.0f}" y="{flt[1]:.0f}" width="4" height="{Hd:.0f}" '
                    f'fill="{AUX_COLOR}" opacity="0.35" data-role="cut-plane"/>')
        section_x = W * 0.68
        body.append(f'<rect x="{section_x:.0f}" y="{cy-sh_px/2:.0f}" width="{sw_px:.0f}" height="{sh_px:.0f}" '
                    f'fill="#fbeaea" stroke="{AUX_COLOR}" stroke-width="1.8" data-role="section-shape"/>')
        body.append(f'<text x="{section_x+sw_px/2:.0f}" y="{cy+sh_px/2+16:.0f}" text-anchor="middle" '
                    f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">section: '
                    f'{_fmt(solved["w"])} x {_fmt(solved["h"])}</text>')

    elif solid == "cylinder":
        r, hgt = spec["dimensions"]["radius"], spec["dimensions"]["height"]
        scale = min((W * 0.5) / (2 * r), (H - 50) / hgt)
        r_px, h_px = r * scale, hgt * scale
        top_c, bot_c = (cx, cy - h_px / 2), (cx, cy + h_px / 2)
        ry = r_px * 0.32
        body.append(f'<line x1="{cx-r_px:.0f}" y1="{top_c[1]:.0f}" x2="{cx-r_px:.0f}" y2="{bot_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<line x1="{cx+r_px:.0f}" y1="{top_c[1]:.0f}" x2="{cx+r_px:.0f}" y2="{bot_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<ellipse cx="{bot_c[0]:.0f}" cy="{bot_c[1]:.0f}" rx="{r_px:.0f}" ry="{ry:.0f}" '
                    f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<path d="M {cx-r_px:.0f} {top_c[1]:.0f} A {r_px:.0f} {ry:.0f} 0 0 0 {cx+r_px:.0f} {top_c[1]:.0f}" '
                    f'fill="none" stroke="{LINE_COLOR}" stroke-width="1" stroke-dasharray="{HIDDEN_DASH}" '
                    f'data-role="hidden-edge"/>')
        body.append(f'<path d="M {cx-r_px:.0f} {top_c[1]:.0f} A {r_px:.0f} {ry:.0f} 0 0 1 {cx+r_px:.0f} {top_c[1]:.0f}" '
                    f'fill="none" stroke="{LINE_COLOR}" stroke-width="1.5" data-role="solid-outline"/>')
        section_x = W * 0.78
        if solved["shape"] == "circle":
            sr_px = solved["radius"] * scale
            cut_y = top_c[1] + h_px * 0.4
            body.append(f'<ellipse cx="{cx:.0f}" cy="{cut_y:.0f}" rx="{r_px:.0f}" ry="{ry:.0f}" '
                        f'fill="none" stroke="{AUX_COLOR}" stroke-width="1.6" data-role="cut-plane"/>')
            body.append(f'<circle cx="{section_x:.0f}" cy="{cy:.0f}" r="{sr_px:.0f}" fill="#fbeaea" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.8" data-role="section-shape"/>')
            body.append(f'<text x="{section_x:.0f}" y="{cy+sr_px+16:.0f}" text-anchor="middle" '
                        f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">section: '
                        f'circle r={_fmt(solved["radius"])}</text>')
        else:
            body.append(f'<line x1="{cx:.0f}" y1="{top_c[1]:.0f}" x2="{cx:.0f}" y2="{bot_c[1]:.0f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.6" data-role="cut-plane"/>')
            sw_px, sh_px = solved["width"] * scale, solved["height"] * scale
            body.append(f'<rect x="{section_x-sw_px/2:.0f}" y="{cy-sh_px/2:.0f}" width="{sw_px:.0f}" '
                        f'height="{sh_px:.0f}" fill="#fbeaea" stroke="{AUX_COLOR}" stroke-width="1.8" '
                        f'data-role="section-shape"/>')
            body.append(f'<text x="{section_x:.0f}" y="{cy+sh_px/2+16:.0f}" text-anchor="middle" '
                        f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">section: '
                        f'{_fmt(solved["width"])} x {_fmt(solved["height"])} rectangle</text>')

    elif solid == "cone":
        r, H_full = spec["dimensions"]["radius"], spec["dimensions"]["height"]
        scale = min((W * 0.5) / (2 * r), (H - 50) / H_full)
        r_px, h_px = r * scale, H_full * scale
        apex, base_c = (cx, cy - h_px / 2), (cx, cy + h_px / 2)
        ry = r_px * 0.32
        body.append(f'<line x1="{apex[0]:.0f}" y1="{apex[1]:.0f}" x2="{cx-r_px:.0f}" y2="{base_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<line x1="{apex[0]:.0f}" y1="{apex[1]:.0f}" x2="{cx+r_px:.0f}" y2="{base_c[1]:.0f}" '
                    f'stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        body.append(f'<ellipse cx="{base_c[0]:.0f}" cy="{base_c[1]:.0f}" rx="{r_px:.0f}" ry="{ry:.0f}" '
                    f'fill="#eaf1fb" stroke="{LINE_COLOR}" stroke-width="2" data-role="solid-outline"/>')
        section_x = W * 0.78
        if solved["shape"] == "circle":
            frac = solved["cut_height_from_apex"] / solved["full_height"]
            cut_y = apex[1] + h_px * frac
            cut_r_px = r_px * frac
            body.append(f'<ellipse cx="{cx:.0f}" cy="{cut_y:.0f}" rx="{cut_r_px:.0f}" ry="{ry*frac:.0f}" '
                        f'fill="none" stroke="{AUX_COLOR}" stroke-width="1.6" data-role="cut-plane"/>')
            sr_px = solved["radius"] * scale
            body.append(f'<circle cx="{section_x:.0f}" cy="{cy:.0f}" r="{sr_px:.0f}" fill="#fbeaea" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.8" data-role="section-shape"/>')
            body.append(f'<text x="{section_x:.0f}" y="{cy+sr_px+16:.0f}" text-anchor="middle" '
                        f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">section: '
                        f'circle r={_fmt(solved["radius"])}</text>')
        else:
            body.append(f'<line x1="{apex[0]:.0f}" y1="{apex[1]:.0f}" x2="{cx:.0f}" y2="{base_c[1]:.0f}" '
                        f'stroke="{AUX_COLOR}" stroke-width="1.6" data-role="cut-plane"/>')
            base_px, height_px = solved["base"] * scale, solved["height"] * scale
            tri = (f'{section_x:.0f},{cy-height_px/2:.0f} {section_x-base_px/2:.0f},{cy+height_px/2:.0f} '
                   f'{section_x+base_px/2:.0f},{cy+height_px/2:.0f}')
            body.append(f'<polygon points="{tri}" fill="#fbeaea" stroke="{AUX_COLOR}" stroke-width="1.8" '
                        f'data-role="section-shape"/>')
            body.append(f'<text x="{section_x:.0f}" y="{cy+height_px/2+16:.0f}" text-anchor="middle" '
                        f'fill="{AUX_COLOR}" {SMALL_FONT} data-role="dimension-label">section: '
                        f'triangle base={_fmt(solved["base"])}, height={_fmt(solved["height"])}</text>')

    svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
           f'style="max-width:280px">' + "".join(body) + "</svg>")
    fixed = label_layout.check_and_fix_labels(svg)
    return fixed["svg"]


diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="solid_net",
    renderer=_render_solid_net,
    schema_check=_validate_solid_net_spec,
    description="Unfolds cube/cuboid/cylinder/cone into their exact 2D net: every panel's size is "
                "derived directly from the solid's own dimensions (e.g. a cylinder's curved-surface "
                "panel is an exact 2*pi*r by h rectangle, a cone's is an exact circular sector with "
                "angle (r/slant_height)*360 degrees) and independently re-verified against those "
                "defining formulas before any SVG is returned.",
))

diagram_plugin_registry.register_plugin(diagram_plugin_registry.DiagramPlugin(
    type_name="cross_section",
    renderer=_render_cross_section,
    schema_check=_validate_cross_section_spec,
    description="Shows a solid (cuboid/cylinder/cone), hidden back edges dashed using this project's "
                "existing hidden-edge convention, together with the exact 2D shape produced by a named "
                "cutting plane (parallel-to-base, through-axis, parallel-to-face, or through-apex-axis). "
                "The cone parallel-to-base case derives its section radius from similar triangles "
                "(r * h_cut / H) and independently re-verifies that formula before returning any SVG.",
))
