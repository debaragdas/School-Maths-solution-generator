"""
diagram_decision.py — THE DIAGRAM DECISION ENGINE.

Sits strictly between Gemini's raw understanding of a question
(prompts.py's schema, parsed in solver.py) and diagram_renderer.py's
deterministic RENDERING of a validated DiagramSpec. Nothing in this
module draws anything, computes a coordinate, or invents geometry —
and no AI call is made here. Its only job is to decide, deterministically,
whether a diagram reaches the renderer at all and in what form, so that
"what to draw" and "how to draw it" are two fully separate questions
answered by two separate modules.

    Stage 0 (AI — prompts.py + solver.py's parsing of its response):
        question text -> {diagram_spec, construction_instruments,
        question_text, given, required, steps, book_diagram_base64,
        book_diagram_figure_ref, ...}. This is the ONLY place natural-
        language understanding happens, by design (see the note at the
        bottom of this file for why a keyword-based "does this need a
        diagram" / "which diagram_type" re-classifier was deliberately
        never added anywhere in this module).

    Stage 1 — DOES THIS QUESTION NEED A DIAGRAM? (decide_diagram, below)
        Deterministically classifies every question into exactly one
        of NO_DIAGRAM / BOOK_DIAGRAM / GENERATED_DIAGRAM.

    Stage 2 — BOOK DIAGRAM, INDEPENDENTLY RE-VERIFIED (_verify_book_citation)
        solver.py::_attach_book_diagrams already runs its own strict,
        exact-citation-only match before this module ever sees the
        question (see that function's own docstring for the full
        policy — it is unchanged, reused as-is, never reimplemented
        here). This stage does NOT just trust that a match happened:
        it independently re-extracts the citation from the question's
        own text and re-derives the same figure number, so a future
        regression upstream can never silently reach this module with
        an unlabelled or guessed figure.

    Stage 3 — DIAGRAM TYPE CONSISTENCY (_infer_type_mismatch)
        For a GENERATED_DIAGRAM candidate: first, diagram_type must be
        one of the renderer's actually-registered types at all (closes
        a real gap where diagram_renderer.validate_diagram_spec itself
        silently passes ANY unrecognized diagram_type — see the inline
        comment at that check for why). Then, checks the spec's own
        populated fields against its declared diagram_type. Some
        fields only ever make sense for exactly one type (see
        prompts.py's schema — e.g. "solid" only exists for
        surface_area_volume, "chart_type" only for statistics). This
        is a schema-footprint check, not a decorative-keyword search
        of the question text: it only rejects when the spec's OWN
        structure contradicts its OWN declared type, never based on a
        guess about what kind of diagram the question "should" have.

    Stage 4 — QUESTION vs. DIAGRAMSPEC CROSS-CHECK
        (_is_internally_consistent, _is_textually_grounded)
        Catches Gemini contradicting itself between two fields of the
        SAME response (construction_instruments vs. diagram_type), and
        catches a diagram whose own labelled points/vertex/rays are
        entirely absent from the question's own text — the
        deterministic signature of a hallucinated or mismatched spec.

    Stage 5 — NORMALIZE, REPAIR SAFE MISTAKES, REJECT UNSAFE ONES
        Reuses diagram_renderer.py's own repair pass (_normalize_spec)
        and structural validator (validate_diagram_spec) — never
        reimplemented here — so every check above runs against exactly
        what the renderer itself would see, and any spec that fails
        structural validation is rejected as NO_DIAGRAM before
        rendering is ever attempted, not discovered mid-render.

Everything from Stage 2 onward degrades to NO_DIAGRAM on any failure —
"if uncertain, omit" — never to a best-effort guess.
"""
import re

from utils import logger, extract_figure_reference
from diagram_renderer import validate_diagram_spec, _normalize_spec, _RENDERERS
import diagram_plugin_registry

# ---------------------------------------------------------------------
# The three (and only three) outcomes of the decision engine.
# ---------------------------------------------------------------------
NO_DIAGRAM = "NO_DIAGRAM"
BOOK_DIAGRAM = "BOOK_DIAGRAM"
GENERATED_DIAGRAM = "GENERATED_DIAGRAM"


def _qnum(question: dict) -> str:
    return f"{question.get('question_number')}{question.get('sub_part') or ''}"


def _combined_question_text(question: dict) -> str:
    steps = question.get("steps") or []
    steps_text = " ".join(str(s) for s in steps) if isinstance(steps, list) else str(steps)
    other_fields = " ".join(str(question.get(field) or "") for field in
                             ("question_text", "given", "required", "final_answer"))
    return f"{other_fields} {steps_text}"


# ---------------------------------------------------------------------
# STAGE 2 — book-diagram citation, independently re-verified.
# ---------------------------------------------------------------------

def _verify_book_citation(question: dict) -> tuple[bool, str]:
    """Re-derives the figure citation from the question's OWN text —
    the exact same extraction solver.py::_attach_book_diagrams already
    performed before ever setting book_diagram_base64 — and confirms
    it's still present and (if recorded) still matches the figure
    number that was actually attached. This module never takes
    book_diagram_base64's mere presence as sufficient proof on its own;
    Stage 2 is what makes that proof."""
    text = f"{question.get('question_text', '')} {question.get('given', '')}"
    citation = extract_figure_reference(text)
    if not citation:
        return False, ("book_diagram_base64 is set but no explicit figure citation "
                        "(e.g. 'চিত্ৰ 3.14') was found in the question's own text on re-check")
    recorded_ref = question.get("book_diagram_figure_ref")
    if recorded_ref and str(recorded_ref) != str(citation):
        return False, (f"book_diagram_figure_ref ('{recorded_ref}') does not match the citation "
                        f"re-extracted from the question's own text ('{citation}')")
    return True, citation


# ---------------------------------------------------------------------
# STAGE 3 — diagram_type consistency (schema footprint, not keywords).
# ---------------------------------------------------------------------
# Each check below is restricted to fields that are UNAMBIGUOUS
# signatures of a single type in prompts.py's own schema — "points"
# alone is shared by triangle/circle/quadrilateral and is deliberately
# NOT used here, since a mismatch there would be a guess, not a
# certainty. Every check below IS a certainty: the field literally
# cannot mean anything else per the schema.
def _has_solid(spec):
    return spec.get("solid") is not None


def _has_chart_type(spec):
    return spec.get("chart_type") is not None


def _has_construction_steps(spec):
    return spec.get("construction_steps") is not None


def _has_angle_rays(spec):
    return spec.get("vertex") is not None and spec.get("rays") is not None


def _has_diagonals(spec):
    return spec.get("diagonals") is not None


def _has_trig_dimensions(spec):
    dims = spec.get("dimensions")
    return isinstance(dims, dict) and ("adjacent" in dims or "opposite" in dims or "hypotenuse" in dims)


def _has_xy_points(spec):
    points = spec.get("points")
    return isinstance(points, list) and any(
        isinstance(p, dict) and "x" in p and "y" in p for p in points)


def _has_plain_string_lines(spec):
    # parallel_lines' "lines" is a list of plain line-name strings
    # (["l1", "l2"]); coordinate_plot's "lines" is a list of dicts
    # ({"through": [...], "label": ...}) — the element TYPE, not just
    # the field name, is what disambiguates them.
    lines = spec.get("lines")
    return isinstance(lines, list) and bool(lines) and all(isinstance(l, str) for l in lines)


_SIGNATURE_FIELDS = (
    (_has_solid, "surface_area_volume", "'solid'"),
    (_has_chart_type, "statistics", "'chart_type'"),
    (_has_construction_steps, "construction", "'construction_steps'"),
    (_has_angle_rays, "angle", "'vertex'/'rays'"),
    (_has_diagonals, "quadrilateral", "'diagonals'"),
    (_has_trig_dimensions, "trigonometry", "'dimensions.adjacent/opposite/hypotenuse'"),
    (_has_xy_points, "coordinate_plot", "points with x/y coordinates"),
    (_has_plain_string_lines, "parallel_lines", "'lines' as a plain list of line names"),
)


def _infer_type_mismatch(spec: dict):
    """Returns a reason string if the spec's own fields unambiguously
    contradict its declared diagram_type, else None. High-confidence
    only — deliberately does not attempt to disambiguate the many
    types that legitimately share a plain "points" list (triangle,
    circle, quadrilateral) since that would be a guess, not a fact."""
    declared = spec.get("diagram_type")
    for predicate, only_valid_for, field_desc in _SIGNATURE_FIELDS:
        if predicate(spec) and declared != only_valid_for:
            return f"{field_desc} is only valid for diagram_type '{only_valid_for}', but diagram_type is '{declared}'"
    return None


# ---------------------------------------------------------------------
# STAGE 4 — question vs. DiagramSpec cross-check.
# ---------------------------------------------------------------------

def _is_internally_consistent(question: dict, spec: dict) -> tuple[bool, str]:
    """Catches Gemini contradicting itself WITHIN its own output for
    this SAME question — not a judgement call about the question's
    meaning, just a plain logical inconsistency between two fields it
    set independently in the same response (see prompts.py's
    CONSTRUCTION (অংকন) QUESTIONS rule: construction_instruments and
    diagram_type "construction" are supposed to always travel
    together)."""
    diagram_type = spec.get("diagram_type")
    has_instruments = bool(question.get("construction_instruments"))
    if has_instruments and diagram_type != "construction":
        return False, (f"construction_instruments is set but diagram_type is "
                        f"'{diagram_type}' (expected 'construction')")
    if diagram_type == "construction" and not has_instruments:
        return False, "diagram_type is 'construction' but construction_instruments is null"
    return True, ""


# Geometry point/vertex labels in this project's actual question text
# are almost always concatenated, never isolated with a word boundary
# around each individual letter — "ত্ৰিভুজ ABC", "AB = AC", "OB = OC".
# So grounding is checked as "does this id appear inside some short run
# of uppercase Latin letters", not "does \bID\b match on its own". The
# run length is capped at 4 to keep this from ever matching inside an
# unrelated all-caps English word/abbreviation elsewhere in the text —
# there are no real point labels longer than that in this schema.
_UPPERCASE_RUN_PATTERN = re.compile(r"[A-Z]{1,4}")


def _collect_declared_ids(spec: dict) -> list:
    """Pulls every point/vertex/ray id a spec declares, however that
    type's own schema names the field (see prompts.py) — triangle/
    circle/quadrilateral/trigonometry/coordinate_plot use "points",
    construction uses "initial_points", angle uses "vertex" + "rays".
    "parallel_lines" (line names, not point ids), "statistics" and
    "surface_area_volume" have no letter-labelled points and correctly
    return []. NOTE: this replaces an earlier version of this function
    that only ever looked at a "points"-shaped field and so silently
    never checked "angle" specs at all (angle has no "points" field) —
    found and fixed during this same audit."""
    dtype = spec.get("diagram_type")
    if dtype in ("triangle", "quadrilateral", "circle", "trigonometry", "coordinate_plot"):
        points = spec.get("points") or []
        return [p.get("id") for p in points if isinstance(p, dict) and p.get("id")]
    if dtype == "construction":
        points = spec.get("initial_points") or []
        return [p.get("id") for p in points if isinstance(p, dict) and p.get("id")]
    if dtype == "angle":
        ids = [spec.get("vertex")] if spec.get("vertex") else []
        rays = spec.get("rays") or []
        ids += [r.get("id") for r in rays if isinstance(r, dict) and r.get("id")]
        return [i for i in ids if i]
    return []  # parallel_lines / statistics / surface_area_volume: no letter-labelled points


def _is_textually_grounded(question: dict, spec: dict) -> tuple[bool, str]:
    """Every point/vertex/ray id a diagram_spec declares should appear
    somewhere in the question's own text (question_text / given /
    required / steps / final_answer, all in the SAME Gemini response
    as the diagram_spec itself). A diagram whose labelled points are
    entirely absent from the question it's attached to has no textual
    grounding at all — the deterministic signature of a mismatched or
    hallucinated spec, independent of whether it happens to be
    structurally well-formed."""
    ids = [str(i) for i in _collect_declared_ids(spec) if i]
    if not ids:
        return True, ""  # nothing to ground here (or this type has none by design)

    text = _combined_question_text(question)
    runs = _UPPERCASE_RUN_PATTERN.findall(text)
    grounded = [pid for pid in ids if any(pid in run for run in runs)]
    if not grounded:
        return False, f"none of the diagram's own point ids {ids} appear anywhere in the question's own text"
    return True, ""


# ---------------------------------------------------------------------
# THE DECISION ENGINE ITSELF.
# ---------------------------------------------------------------------

def decide_diagram(question: dict) -> dict:
    """Returns {"decision": NO_DIAGRAM | BOOK_DIAGRAM | GENERATED_DIAGRAM,
    "diagram_spec": dict | None, "reason": str}.

    "diagram_spec" is kept (backward compatible with solver.py's
    existing `q["diagram_spec"] = decide_diagram(q)["diagram_spec"]`
    call): it is None for both NO_DIAGRAM and BOOK_DIAGRAM (the actual
    book image lives entirely in question["book_diagram_base64"],
    untouched by this function) and the normalized, validated spec for
    GENERATED_DIAGRAM — the ONLY diagram_spec that should ever reach
    diagram_renderer.render_diagram. Never mutates the input
    `question` dict."""

    # ---- STAGE 1 + STAGE 2: book diagram, independently re-verified ----
    if question.get("book_diagram_base64"):
        ok, citation_or_reason = _verify_book_citation(question)
        if not ok:
            logger.warning(f"⚠️ Q{_qnum(question)}: {citation_or_reason} — treating as "
                            f"NO_DIAGRAM rather than trusting book_diagram_base64 blindly.")
            return {"decision": NO_DIAGRAM, "diagram_spec": None, "reason": citation_or_reason}
        return {"decision": BOOK_DIAGRAM, "diagram_spec": None,
                "reason": f"verified textbook citation '{citation_or_reason}' takes precedence"}

    # ---- STAGE 1: no diagram_spec at all ----
    raw_spec = question.get("diagram_spec")
    if not raw_spec or not isinstance(raw_spec, dict):
        return {"decision": NO_DIAGRAM, "diagram_spec": None,
                "reason": "no diagram_spec — question doesn't need one"}

    # ---- STAGE 5 (repair half): normalize before any further check, so
    # every check below sees exactly what the renderer itself would see
    # (reuses diagram_renderer.py's own repair pass, never reimplemented) ----
    spec = _normalize_spec(raw_spec)

    # ---- STAGE 3 (part a): diagram_type must be one of the renderer's
    # actually-registered types. NOTE: diagram_renderer.validate_diagram_spec
    # itself does NOT catch this — for any diagram_type it doesn't have a
    # dedicated structural check for (that includes an unrecognized/typo'd
    # string), it falls through to `return True, []` (see its own source:
    # "structural check only implemented for the types dispatched above").
    # That's a safe default for the RENDERER (an unknown type just yields
    # no SVG), but it means "structural validation passed" is not actually
    # a meaningful signal for those types. The decision engine closes that
    # gap here, explicitly and loudly, rather than silently degrading to a
    # blank diagram-box with no audit trail.
    if spec.get("diagram_type") not in _RENDERERS and not diagram_plugin_registry.is_plugin_type(spec.get("diagram_type")):
        reason = f"diagram_type '{spec.get('diagram_type')}' is not a recognized/renderable type"
        logger.warning(f"⚠️ Q{_qnum(question)}: diagram_spec rejected at the decision layer "
                        f"({reason}) — omitting rather than silently producing a blank diagram.")
        return {"decision": NO_DIAGRAM, "diagram_spec": None, "reason": reason}

    # ---- STAGE 3 (part b): diagram_type consistency (schema footprint) ----
    mismatch = _infer_type_mismatch(spec)
    if mismatch:
        logger.warning(f"⚠️ Q{_qnum(question)}: diagram_spec rejected at the decision layer "
                        f"({mismatch}) — omitting rather than publishing a mistyped diagram.")
        return {"decision": NO_DIAGRAM, "diagram_spec": None, "reason": mismatch}

    # ---- STAGE 4: question vs. DiagramSpec cross-check ----
    ok, reason = _is_internally_consistent(question, spec)
    if not ok:
        logger.warning(f"⚠️ Q{_qnum(question)}: diagram_spec rejected at the decision layer "
                        f"({reason}) — omitting rather than publishing a self-contradictory diagram.")
        return {"decision": NO_DIAGRAM, "diagram_spec": None, "reason": reason}

    ok, reason = _is_textually_grounded(question, spec)
    if not ok:
        logger.warning(f"⚠️ Q{_qnum(question)}: diagram_spec rejected at the decision layer "
                        f"({reason}) — omitting rather than publishing an ungrounded diagram.")
        return {"decision": NO_DIAGRAM, "diagram_spec": None, "reason": reason}

    # ---- STAGE 5 (reject-unsafe half): reuse the renderer's own
    # structural validator so an unsafe spec is rejected BEFORE
    # rendering is ever attempted, never reimplemented here ----
    ok, issues = validate_diagram_spec(spec)
    if not ok:
        reason = "; ".join(issues)
        logger.warning(f"⚠️ Q{_qnum(question)}: diagram_spec rejected at the decision layer "
                        f"(structural validation: {reason}) — omitting rather than publishing "
                        f"an invalid diagram.")
        return {"decision": NO_DIAGRAM, "diagram_spec": None, "reason": reason}

    return {"decision": GENERATED_DIAGRAM, "diagram_spec": spec,
            "reason": "passed every decision-engine stage; forwarded to the renderer"}


# ----------------------------------------------------------------------
# WHY THERE IS NO KEYWORD-BASED "does this question need a diagram at
# all" OR "which diagram_type is this" RE-CLASSIFIER IN THIS MODULE
# ----------------------------------------------------------------------
# Both of those are genuine natural-language-understanding judgements —
# exactly the class of decision this project's own architecture
# deliberately reserves for the single Gemini call in prompts.py (see
# that file's MANDATORY RULE). A second, independent, keyword-based
# classifier bolted on here would not make that decision more
# deterministic — Assamese mathematical phrasing is too varied for a
# fixed keyword list to reliably reproduce Gemini's read of the actual
# question, so it would just as often introduce a NEW false rejection
# or a NEW wrong diagram_type as it fixed one, in direct tension with
# "never publish a mathematically misleading diagram" and "if uncertain,
# omit". What this module deterministically checks instead is
# everything that does NOT require re-reading the question from
# scratch: whether the spec's own fields agree with its own declared
# type (Stage 3), whether Gemini's own fields agree with each other
# (Stage 4a), and whether the diagram is even about the same labelled
# points as the question it's attached to (Stage 4b). All three are
# checks a fixed script can make reliably, without needing to
# understand Assamese mathematics itself.
