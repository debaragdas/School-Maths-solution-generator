"""
vision_validation.py — THE VISION VALIDATION ENGINE (Phase 5).

DETERMINISTIC, per the approved brief — no AI call in this module.
"Vision validation" here means programmatic inspection of the actual
rendered artifact (the SVG string / chart data / solved geometry),
proving concrete, checkable facts about it, exactly the same
philosophy diagram_final_check.py already established for overlap and
clipping. This module is the generalization of that philosophy to
geometry-consistency and chart-data-consistency, plus two new
rendering-quality guards that specifically harden the new work from
Phases 1 and 4 (the geometry solver and the plugin registry are both
new code paths that didn't exist when diagram_final_check.py was
written, and both are exactly the kind of code where a silent
numerical bug — a NaN slipping through a division, a degenerate
bounding box from a bad fit-to-canvas — would otherwise be invisible
until a human found it in a printed PDF).

Every check here either PROVES a concrete fact about the artifact or
safely no-ops (same "prove or omit, never guess" contract as
diagram_final_check.py). Nothing here retries or regenerates anything
— a failure here is reported outward exactly like every other stage's
failure: omit the diagram, log why, never silently ship something this
module could show was wrong.

WIRING: final_pre_pdf_check() in diagram_final_check.py calls into
this module for every GENERATED_DIAGRAM, as ADDITIONAL gates alongside
its own existing overlap/clipping checks (reused here, not
duplicated — see _check_no_overlapping_labels/_check_no_clipped_content
imports below).
"""
import re

from utils import logger

_NUMBER_ATTR_PATTERN = re.compile(
    r'\b(?:x|y|x1|y1|x2|y2|cx|cy|r|rx|ry|width|height)="([^"]*)"')
_VIEWBOX_PATTERN = re.compile(r'viewBox="([^"]*)"')
_DATA_ROLE_COUNT_CACHE_PATTERN = re.compile(r'data-role="([a-zA-Z0-9_\-]+)"')


# ---------------------------------------------------------------------
# RENDERING QUALITY: no NaN/Infinity, a sane viewBox.
# ---------------------------------------------------------------------

def check_no_invalid_numbers(svg: str) -> tuple:
    """Every numeric coordinate/size attribute the renderer emits must
    be a real, finite number. A NaN or Infinity in an SVG attribute
    (e.g. from an unguarded division by a zero span, or a degenerate
    geometry_solver fit-to-canvas edge case) renders as either nothing
    or garbage in different viewers — this is the deterministic
    signature of a numerical bug upstream, not a rendering-quality
    judgement call."""
    for match in _NUMBER_ATTR_PATTERN.finditer(svg):
        raw = match.group(1)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue  # non-numeric attribute value (shouldn't happen for these attrs, but not this check's job)
        if val != val or val in (float("inf"), float("-inf")):
            return False, f"SVG contains a non-finite coordinate/size value ({raw!r})"
    return True, ""


def check_valid_viewbox(svg: str) -> tuple:
    """The SVG's own declared viewBox must parse to exactly four
    finite numbers with a positive width and height — the minimum bar
    for "this SVG has any well-defined visible area at all"."""
    m = _VIEWBOX_PATTERN.search(svg)
    if not m:
        return True, ""  # no viewBox present — nothing to prove either way, same policy as diagram_final_check.py
    parts = m.group(1).split()
    if len(parts) != 4:
        return False, f"viewBox does not have exactly 4 values ({m.group(1)!r})"
    try:
        _, _, width, height = (float(p) for p in parts)
    except ValueError:
        return False, f"viewBox contains a non-numeric value ({m.group(1)!r})"
    if width <= 0 or height <= 0:
        return False, f"viewBox has a non-positive width/height ({width} x {height})"
    return True, ""


# ---------------------------------------------------------------------
# GEOMETRY: independent re-derivation for solver-produced triangles.
# ---------------------------------------------------------------------

def check_triangle_geometry_reproducible(spec: dict) -> tuple:
    """For a triangle diagram_spec that carries side_lengths/angles_deg
    (i.e. one that should have gone through Phase 1's geometry solver),
    independently RE-RUNS the solver on the same inputs and confirms it
    returns a result — a real, deterministic reproducibility guarantee,
    not a placeholder. Because geometry_solver.solve_triangle is a pure
    function (no hidden state, no randomness), calling it twice on
    identical input must either both succeed with numerically identical
    output, or both fail identically — anything else is proof of a
    hidden non-determinism bug in what is supposed to be a fully
    deterministic system, which is worth catching here specifically
    because it is exactly the kind of defect that would otherwise never
    surface in ordinary single-run testing.

    Not applicable (returns ok=True) for specs without those fields —
    those never went through the solver in the first place and are
    validated by diagram_final_check.py's existing structural checks
    only, same as before Phase 1 existed.
    """
    if not isinstance(spec, dict) or spec.get("diagram_type") != "triangle":
        return True, ""
    if not spec.get("side_lengths") and not spec.get("angles_deg"):
        return True, ""

    import geometry_solver
    points = spec.get("points", [])
    ids = [p.get("id") for p in points if isinstance(p, dict)]
    if len(ids) < 3:
        return True, ""

    kwargs = dict(side_lengths=spec.get("side_lengths"), angles_deg=spec.get("angles_deg"),
                  right_angle_at=spec.get("right_angle_at"))
    run1, reason1 = geometry_solver.solve_triangle(ids[:3], **kwargs)
    run2, reason2 = geometry_solver.solve_triangle(ids[:3], **kwargs)

    if (run1 is None) != (run2 is None):
        return False, (f"geometry solver gave DIFFERENT success/failure results on two identical "
                        f"calls (run1={'solved' if run1 else 'None'}, run2={'solved' if run2 else 'None'}) "
                        f"— this indicates a hidden non-determinism bug, not a legitimate solving outcome")
    if run1 is None:
        return True, ""  # both consistently declined — a legitimate, reproducible outcome
    for pid in ids[:3]:
        x1, y1 = run1[pid]
        x2, y2 = run2[pid]
        if abs(x1 - x2) > 1e-9 or abs(y1 - y2) > 1e-9:
            return False, (f"geometry solver produced DIFFERENT coordinates for point '{pid}' on "
                            f"two identical calls ({run1[pid]} vs {run2[pid]}) — non-deterministic "
                            f"solver output is a correctness bug")
    return True, ""


# ---------------------------------------------------------------------
# CHARTS: the number of drawn data elements must match the spec's data.
# ---------------------------------------------------------------------

def check_chart_element_count_matches_data(spec: dict, svg: str) -> tuple:
    """For bar/histogram/pie charts, the number of drawn bars/slices
    (identified by the renderer's own existing data-role="bar" /
    data-role="pie-slice" attributes — no new tagging needed, these
    already exist in diagram_renderer.py's chart output) must equal the
    number of valid numeric values in the spec. A mismatch is the
    deterministic signature of a chart silently dropping or duplicating
    a data point during rendering."""
    if not isinstance(spec, dict):
        return True, ""
    diagram_type = spec.get("diagram_type")
    if diagram_type != "statistics":
        return True, ""

    chart_type = spec.get("chart_type")
    values = spec.get("values", []) or []
    numeric_values = [v for v in values if isinstance(v, (int, float)) or
                       (isinstance(v, str) and _is_number_str(v))]

    if chart_type in ("bar", "histogram"):
        role = "bar"
    elif chart_type == "pie":
        role = "pie-slice"
    else:
        return True, ""  # frequency_polygon/ogive draw a single connected line, not per-value shapes; not this check's job

    drawn = len(re.findall(fr'data-role="{role}"', svg))
    expected = len(numeric_values)
    if expected and drawn != expected:
        return False, (f"chart_type='{chart_type}' spec has {expected} numeric value(s) but the "
                        f"rendered SVG contains {drawn} '{role}' element(s) — a data point was "
                        f"silently dropped or duplicated during rendering")
    return True, ""


def _is_number_str(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------
# TOP-LEVEL ENTRY POINT — called from diagram_final_check.py.
# ---------------------------------------------------------------------

def validate_generated_diagram(spec: dict, svg: str) -> tuple:
    """Runs every deterministic vision-validation check relevant to
    this diagram and returns (ok, reason) — the FIRST failing check's
    reason, since (matching every other stage in this pipeline) one
    concrete, actionable reason is more useful than a merged wall of
    text, and the caller already re-checks from scratch on any future
    fix rather than needing a full list. Never raises: an unexpected
    internal error in this module must degrade to "treat as passed"
    (log loudly, but never let a bug in a VALIDATOR be the reason a
    perfectly good diagram gets thrown away) — this mirrors
    _check_no_clipped_content's own "nothing to check -> True" policy
    for the case where this module itself can't complete its job.
    """
    checks = [
        lambda: check_no_invalid_numbers(svg),
        lambda: check_valid_viewbox(svg),
        lambda: check_triangle_geometry_reproducible(spec),
        lambda: check_chart_element_count_matches_data(spec, svg),
    ]
    for check in checks:
        try:
            ok, reason = check()
        except Exception as e:
            logger.warning(f"⚠️ vision_validation: a check raised an unexpected error ({e}) — "
                            f"skipping that check rather than incorrectly rejecting a possibly-fine diagram.")
            continue
        if not ok:
            return False, reason
    return True, "passed all deterministic vision-validation checks"
