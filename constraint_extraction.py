"""
constraint_extraction.py — THE CONSTRAINT EXTRACTION ENGINE'S
deterministic boundary (Phase 3).

WHAT THIS MODULE IS, PRECISELY: the "mathematical brain" of this
system — the actual understanding of what a question means — has
always lived in exactly one place, and stays there: ONE Gemini call
per exercise (prompts.py's schema + solver.py::solve_exercise). That
does not move. Moving a working, extensively-tested AI integration
into a new module for the sake of an org chart would be exactly the
kind of unrelated churn this project's own principles rule out.

What WAS missing, and what this module adds, is the explicit,
independently-testable, deterministic CONTRACT at the boundary between
"what Gemini said" and "what every downstream deterministic stage is
allowed to trust" — formalizing something that used to be implicit
and scattered (a bit of coercion here in solver.py, a bit of silent
dropping there in geometry_solver.py) into one auditable place that
runs BEFORE diagram_decision.py ever sees a question, and logs LOUDLY
the moment it finds something Gemini reported that doesn't type-check,
instead of letting it fail silently three modules downstream.

This module NEVER calls Gemini, NEVER decides what to draw, NEVER
computes a coordinate. It only proves — or disproves — that a
diagram_spec's own numeric constraint fields (side_lengths, angles_deg)
are internally well-typed before anything downstream relies on them.

WHY THIS MATTERS NOW SPECIFICALLY: geometry_solver.py (Phase 1) reads
side_lengths/angles_deg and already defensively drops anything
malformed — but it does so SILENTLY, by design (a solver's job is to
solve or decline, not to audit). That silence is correct for the
solver's own narrow contract, but wrong for the SYSTEM's contract: if
Gemini ever reports a length as the string "5 cm" instead of the
number 5, or reports an angle at a vertex that was never even declared
in `points`, that is a genuine extraction defect worth knowing about
at the point it happened — not an event that should vanish into "the
solver just returned None" with no trace of why. This module is where
that gets caught, named, and logged.
"""
import copy

from utils import logger

_NUMERIC_CONSTRAINT_FIELDS = ("side_lengths", "angles_deg")


def _qnum(question: dict) -> str:
    return f"{question.get('question_number')}{question.get('sub_part') or ''}"


def validate_constraints(diagram_spec: dict) -> tuple:
    """Deterministically checks the shape of a single diagram_spec's
    OPTIONAL numeric constraint fields (side_lengths, angles_deg — see
    geometry_solver.py / prompts.py's schema). Returns (ok, issues):
    ok is True whenever there is nothing to complain about (including
    when the spec has neither field at all — that is the normal,
    expected case for the vast majority of questions, and is NOT an
    error). `issues` is a list of human-readable strings describing
    every problem found; never raises.

    Checks performed, each one a concrete, provable fact about the
    spec's own declared data — never a judgement about the
    mathematics itself (that stays entirely out of scope for a
    deterministic module):
      - side_lengths / angles_deg, if present, must be a dict/object.
      - every key must be a string.
      - every value must be coercible to a finite float.
      - every side_lengths key must reference exactly two DISTINCT
        characters (a valid "AB"-style side identifier).
      - every side_lengths / angles_deg key that names a point must
        name a point actually declared in this spec's own `points`
        list — a constraint on an undeclared vertex is a genuine
        extraction inconsistency, not something geometry_solver.py
        should have to silently absorb.
    """
    if not isinstance(diagram_spec, dict):
        return True, []

    declared_ids = {p.get("id") for p in diagram_spec.get("points", []) if isinstance(p, dict)}
    issues = []

    for field in _NUMERIC_CONSTRAINT_FIELDS:
        value = diagram_spec.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            issues.append(f"{field} must be an object mapping id(s) to a number, got "
                          f"{type(value).__name__}: {value!r}")
            continue
        for key, val in value.items():
            if not isinstance(key, str):
                issues.append(f"{field} has a non-string key {key!r}")
                continue
            try:
                num = float(val)
                if num != num or num in (float("inf"), float("-inf")):  # NaN / Infinity
                    issues.append(f"{field}['{key}'] = {val!r} is not a finite number")
            except (TypeError, ValueError):
                issues.append(f"{field}['{key}'] = {val!r} is not numeric")

            if field == "side_lengths":
                if len(key) != 2 or key[0] == key[1]:
                    issues.append(f"side_lengths key '{key}' is not a valid two-point side "
                                  f"identifier (e.g. 'AB')")
                elif declared_ids and not set(key).issubset(declared_ids):
                    issues.append(f"side_lengths key '{key}' references a point not declared "
                                  f"in this diagram's own points list {sorted(declared_ids)}")
            elif field == "angles_deg":
                if declared_ids and key not in declared_ids:
                    issues.append(f"angles_deg key '{key}' references a point not declared "
                                  f"in this diagram's own points list {sorted(declared_ids)}")

    return (len(issues) == 0), issues


def sanitize_constraints(diagram_spec: dict, context_label: str = "") -> dict:
    """Returns a NEW diagram_spec (does not mutate the input) with any
    invalid entry inside side_lengths/angles_deg removed, and logs a
    warning naming exactly what was dropped and why. A spec that
    passes validate_constraints cleanly is returned completely
    unchanged (same object identity avoided regardless, to keep the
    "always returns a fresh dict" contract simple and safe to call
    repeatedly).

    This is deliberately NOT a duplicate of geometry_solver.py's own
    internal filtering — geometry_solver's filtering exists so the
    SOLVER never crashes on bad input even if this function is somehow
    bypassed (defense in depth); THIS function exists so bad input is
    caught, named, and logged at the earliest possible point, before
    diagram_decision.py or geometry_solver.py ever run.
    """
    if not isinstance(diagram_spec, dict):
        return diagram_spec

    ok, issues = validate_constraints(diagram_spec)
    if ok:
        return diagram_spec

    prefix = f"Q{context_label}: " if context_label else ""
    for issue in issues:
        logger.warning(f"⚠️ {prefix}constraint_extraction: dropping invalid geometric "
                        f"constraint — {issue}")

    spec = copy.deepcopy(diagram_spec)
    declared_ids = {p.get("id") for p in spec.get("points", []) if isinstance(p, dict)}

    for field in _NUMERIC_CONSTRAINT_FIELDS:
        value = spec.get(field)
        if not isinstance(value, dict):
            if field in spec and value is not None:
                spec[field] = {}
            continue
        cleaned = {}
        for key, val in value.items():
            if not isinstance(key, str):
                continue
            try:
                num = float(val)
                if num != num or num in (float("inf"), float("-inf")):
                    continue
            except (TypeError, ValueError):
                continue
            if field == "side_lengths":
                if len(key) != 2 or key[0] == key[1]:
                    continue
                if declared_ids and not set(key).issubset(declared_ids):
                    continue
            elif field == "angles_deg":
                if declared_ids and key not in declared_ids:
                    continue
            cleaned[key] = num
        spec[field] = cleaned

    return spec


def sanitize_questions(questions: list) -> None:
    """Convenience entry point: runs sanitize_constraints over every
    question's diagram_spec IN PLACE (matching the mutation style
    solver.py's own per-question loops already use throughout this
    file, for consistency). Call this once, right after a question's
    diagram_spec is finalized and BEFORE diagram_decision.decide_diagram
    runs on it — see solver.py's wiring.

    Never raises: a single malformed question can never abort
    processing of the rest of the exercise.
    """
    for q in questions or []:
        try:
            spec = q.get("diagram_spec")
            if isinstance(spec, dict):
                q["diagram_spec"] = sanitize_constraints(spec, context_label=_qnum(q))
        except Exception as e:
            logger.warning(f"⚠️ constraint_extraction: unexpected error sanitizing Q{_qnum(q)} "
                            f"({e}) — leaving its diagram_spec unchanged.")
