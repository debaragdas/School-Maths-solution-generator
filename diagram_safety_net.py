"""
diagram_safety_net.py — THE MISSED-DIAGRAM SAFETY NET.

Fixes the one real gap left in the diagram pipeline after
diagram_decision.py's audit: that module is entirely correct about
WHAT TO DO once it has Gemini's diagram_spec (or lack of one), but it
has no opinion at all about whether Gemini's silence ("diagram_spec":
null) was actually correct. prompts.py's MANDATORY RULE tells Gemini to
emit a diagram_spec for any question naming/implying a figure — but
that is a single free-form judgement made once, alongside solving every
other question in the exercise in the same combined pass, and it can be
missed on any individual question exactly the way the coordinate-read
audit (solver.py::_verify_coordinate_answers) found the combined pass
can misread a single point.

THIS MODULE'S JOB, PRECISELY (and no more than this):
  1. Deterministically flag questions whose OWN text strongly implies a
     geometric figure/construction/number-line/chart but that ended up
     with decision == NO_DIAGRAM for the "no diagram_spec at all"
     reason (never for a book-diagram or a rejected-spec reason — see
     _is_recoverable below, those already went through their own
     correct, independent rejection logic and must not be
     second-guessed here).
  2. For every flagged question, fire ONE narrow, independent Gemini
     call whose ENTIRE context is that single question's own text (not
     the whole exercise, not the PDF) asking specifically for a
     diagram_spec — or an explicit confirmation that none is needed.
  3. Hand any recovered spec back through diagram_decision.decide_diagram
     exactly as if it had arrived in the main pass — every existing
     Stage 3/4/5 check (type validity, internal consistency, textual
     grounding, structural validation) still applies in full. This
     module has NO authority to accept a spec directly; it only ever
     supplies a second candidate to the SAME decision engine.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:
  - It never infers WHICH diagram_type or WHAT coordinates a missed
    question needs from keywords — that would be exactly the
    "keyword-based re-classifier" diagram_decision.py's own docstring
    explains was deliberately never added, for the same reason: fixed
    keyword lists cannot reliably reproduce genuine Assamese
    mathematical understanding, and guessing the wrong shape is worse
    than the drawing being absent (the user could otherwise learn an
    incorrect construction from it). The keyword check below answers
    only "does this look worth a second AI opinion", never "what does
    it need".
  - It never touches a question that already has a book_diagram or an
    explicit (even if since-rejected) diagram_spec — those already went
    through real judgement once; re-guessing over that judgement is not
    this module's job and would reintroduce exactly the kind of
    unverified guess the rest of this pipeline exists to eliminate.
  - It never raises. A failed/garbled/timed-out recovery call, or a
    recovered spec that still fails decide_diagram's own checks, simply
    leaves the question as NO_DIAGRAM — the same "if it cannot be
    verified, fail safely" policy used everywhere else in this project.
    The only extra thing recorded is an audit flag so the run's logs
    make the gap visible to a human, instead of it silently vanishing.
"""
import re

import config
from utils import logger, retry_with_backoff
from diagram_decision import decide_diagram, NO_DIAGRAM

# ---------------------------------------------------------------------
# STAGE A — deterministic "is this worth a second opinion" flag.
#
# Assamese + English geometry/construction/chart/number-line vocabulary
# that, in this project's own actual question corpus, essentially never
# appears in a question that's purely algebraic/numeric with no figure.
# Deliberately mirrors the same trigger list prompts.py's own MANDATORY
# RULE already gives Gemini — this is not a NEW judgement about what
# counts as "needs a diagram", just a second, independent check for the
# same rule Gemini was already told to follow.
# ---------------------------------------------------------------------
_GEOMETRY_TERMS = (
    "ত্ৰিভুজ", "চতুৰ্ভুজ", "বৃত্ত", "কোণ", "সামান্তৰিক", "আয়তক্ষেত্ৰ",
    "বৰ্গক্ষেত্ৰ", "সমান্তৰাল", "লম্ব", "স্পৰ্শক", "চিত্ৰত", "অংকন কৰা",
    "অংকন কৰক", "সংখ্যা ৰেখা", "বৰ্গমূল সৰ্পিল", "স্থানাংক", "ভুজ", "কোটি",
    "আয়তন", "পৃষ্ঠতল", "ক্ষেত্ৰফল", "পৰিসীমা", "বাৰ গ্ৰাফ", "হিষ্টোগ্ৰাম",
    "ব্যাসাৰ্ধ", "কর্ণ", "সমকোণী", "সমদ্বিখণ্ডক", "মধ্যমা", "উচ্চতা", "চাপ",
    "triangle", "quadrilateral", "circle", "angle", "parallelogram",
    "rectangle", "square", "parallel", "perpendicular", "tangent",
    "in the figure", "construct", "number line", "square root spiral",
    "coordinate", "abscissa", "ordinate", "volume", "surface area",
    "bar graph", "histogram", "radius", "diagonal", "prove that",
    "show that", "চতুর্ভুজ", "bisector", "bisects", "median", "altitude",
    "arc", "chord",
)
_GEOMETRY_PATTERN = re.compile(
    "|".join(re.escape(t) for t in _GEOMETRY_TERMS), re.IGNORECASE
)

# A run of 3+ single, space/comma-separated uppercase letters
# ("ABC", "A, B, C", "PQR") is the standard textbook shorthand for
# naming a figure's vertices — a strong independent signal on its own,
# additive to the vocabulary check above (either one is sufficient).
_VERTEX_LIST_PATTERN = re.compile(r"\b([A-Z])\s*,?\s*([A-Z])\s*,?\s*([A-Z])\b")


def _question_text_blob(question: dict) -> str:
    """PRODUCTION-AUDIT FIX (final pre-launch round): this used to send
    ONLY question_text/given/required to the rebuild call — noticeably
    LESS than what the question's own worked solution actually
    contains. Two concrete failures traced back to this gap:

      1. Weaker diagrams: the "steps" a question was already solved
         with usually name the exact construction the diagram needs
         (e.g. "drop a perpendicular from A to BC, meeting it at E") —
         without seeing them, a from-scratch rebuild has strictly less
         to work with than the main solve pass ever had for the SAME
         question, even though the goal is an equally good diagram.
      2. Diagrams rejected as "not textually grounded": diagram_
         decision._is_textually_grounded() checks a rebuilt spec's
         point ids against question_text + given + required + steps +
         final_answer (see _combined_question_text there) — a point
         label that only appears in "steps" (very common: intermediate
         construction points like a foot-of-perpendicular are rarely
         restated in the original question_text) would ALWAYS fail
         that check for a rebuild that never even saw "steps", making
         "Regenerate Diagram"/"Generate Diagram" silently produce
         nothing far more often than it should.

    Now mirrors diagram_decision._combined_question_text's own field
    list exactly, so the rebuild call sees precisely what will later
    validate it — the standing principle used everywhere else in this
    pipeline (Stage 5 already reuses the renderer's own validator for
    the same reason: never validate against something the generator
    itself never saw).
    """
    steps = question.get("steps") or []
    steps_text = " ".join(str(s) for s in steps) if isinstance(steps, list) else str(steps)
    other_fields = " ".join(str(question.get(f) or "") for f in
                             ("question_text", "given", "required", "final_answer"))
    return f"{other_fields} {steps_text}".strip()


def _looks_like_geometry_question(question: dict) -> bool:
    text = _question_text_blob(question)
    if not text.strip():
        return False
    return bool(_GEOMETRY_PATTERN.search(text) or _VERTEX_LIST_PATTERN.search(text))


def _is_recoverable(question: dict) -> bool:
    """Only ever true for the ONE gap this module exists to catch: a
    question that reached NO_DIAGRAM purely because Gemini's main pass
    never populated diagram_spec at all. A book-diagram question, or a
    question whose diagram_spec was populated but then correctly
    rejected by diagram_decision.py's own Stage 3/4/5 checks, already
    went through real judgement once and is explicitly OUT of scope —
    re-guessing over an already-reasoned rejection is not this
    module's job (see the module docstring)."""
    if question.get("diagram_decision") != NO_DIAGRAM:
        return False
    if question.get("book_diagram_base64"):
        return False
    # Only the specific "no diagram_spec at all" reason qualifies —
    # any other NO_DIAGRAM reason means a real spec existed and was
    # deliberately rejected on its own merits.
    return question.get("diagram_decision_reason") == "no diagram_spec — question doesn't need one"


# ---------------------------------------------------------------------
# STAGE B — the narrow recovery call itself.
# ---------------------------------------------------------------------
_RECOVERY_SCHEMA_HINT = """
Return ONLY one JSON object, no commentary, no code fences:

{"needs_diagram": true|false, "diagram_spec": <DiagramSpec object>|null,
 "construction_instruments": <array of strings>|null}

If needs_diagram is false, diagram_spec AND construction_instruments
MUST both be null. If true, diagram_spec MUST be a complete, valid
object using exactly the schema below (the SAME schema this project
always uses elsewhere) — never a partial or placeholder object. If
(and only if) diagram_spec's own "diagram_type" is "construction",
construction_instruments MUST also be a non-null array describing the
build sequence (ruler/compass steps) — for every other diagram_type,
construction_instruments MUST be null. These two fields must always
travel together exactly like this, since that pairing is independently
re-checked afterward.
"""


def _build_recovery_prompt(question: dict, class_name: int, extra_instruction: str = None) -> str:
    from prompts import DIAGRAM_SPEC_SCHEMA  # local import: avoids any import-cycle risk
    text = _question_text_blob(question)
    intro = (
        f"You are re-checking ONE Class {class_name or '9'} Assamese SEBA/NCERT Maths "
        f"question that may have been missed for a diagram on a first pass."
        if not extra_instruction else
        f"You are REBUILDING, completely from scratch, the diagram for ONE Class "
        f"{class_name or '9'} Assamese SEBA/NCERT Maths question, at a reviewer's "
        f"request. IGNORE any previous diagram entirely — you have not been shown one "
        f"and must not assume anything about it; work ONLY from the question text below "
        f"and the reviewer's guidance."
    )
    guidance = (
        f"\n\nREVIEWER'S GUIDANCE (must be honored — this is why the diagram is being "
        f"rebuilt):\n{extra_instruction}\n"
        if extra_instruction else ""
    )
    return (
        f"{intro} Read ONLY the question below — including its already-worked solution "
        f"(given/required/steps/final answer, included so you have exactly the same "
        f"mathematical context the original solve pass had) — and decide, independently, "
        f"whether it names or implies a geometric figure, chart, 3D solid, number-line "
        f"placement, or square-root-spiral construction.\n\nQUESTION (with its worked "
        f"solution):\n{text}\n{guidance}\n{DIAGRAM_SPEC_SCHEMA}\n{_RECOVERY_SCHEMA_HINT}"
    )


def _is_retryable_api_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in ("429", "resource_exhausted", "quota", "unavailable", "timeout"))


@retry_with_backoff(times=3, base_delay=3.0, max_delay=20.0, retryable_check=_is_retryable_api_error)
def _call_gemini_recovery(prompt: str):
    from google.genai import types
    from solver import get_client, parse_gemini_json_response  # reuse the single shared client + helper
    client = get_client()
    response = client.models.generate_content(
        # Producing a diagram_spec is solving work, not scanning — route
        # to the SOLVE-tier model (fires rarely; correctness matters).
        model=config.GEMINI_SOLVE_MODEL_ID,
        contents=[prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1000),
            response_mime_type="application/json",
        ),
    )
    # PRODUCTION-AUDIT FIX (final pre-launch round): this used to be a
    # bare json.loads with no escape repair at all — the SAME "\frac
    # silently becomes a form-feed byte" risk solver.py's main solve
    # pass and correction_engine.py's correction calls had (see
    # parse_gemini_json_response's own docstring for the full root
    # cause), except here with no fallback whatsoever. construction_
    # instructions strings in particular can carry LaTeX ("draw AB such
    # that \\angle ABC = 60°"), so this needs the same protection.
    return parse_gemini_json_response(response.text or "{}", "Diagram-rebuild response")


def generate_diagram_from_scratch(question: dict, class_name: int, extra_instruction: str = None) -> dict:
    """Public entry point reused by BOTH this module's own automatic
    safety net (apply_diagram_safety_net, below, no extra_instruction)
    and the Human Review app's "Regenerate Diagram" / "Generate
    Diagram" actions (correction_engine.regenerate_diagram_from_scratch,
    extra_instruction = the reviewer's own correction text, e.g.
    "altitude missing" or "use a right triangle, not isosceles").

    ALWAYS builds fresh from the question's own text alone (question_
    text/given/required/steps/final_answer) — the `question` dict's
    existing diagram_spec/construction_instruments, if any, are never
    read or referenced here, by design: this is exactly what stops a
    wrong diagram from biasing its own replacement (rebuilding "from" a
    bad SVG/spec, e.g. by asking an AI to "edit" it, is what tends to
    reproduce the same mistake — see correction_engine.py's docstring
    on why this is a SEPARATE code path from its general single-
    question text/diagram correction call rather than that call with a
    diagram-flavored prompt).

    Returns {"needs_diagram": bool, "diagram_spec": dict|None,
    "construction_instruments": list|None} exactly as the AI returned
    it — UNVALIDATED. Callers MUST still (a) write BOTH diagram_spec
    AND construction_instruments onto the question object BEFORE
    calling diagram_decision.decide_diagram (its Stage 4 consistency
    check requires these two fields to have arrived together in the
    SAME response — see _is_internally_consistent's docstring — so
    leaving a stale construction_instruments from a previous, now-
    discarded diagram in place would wrongly reject a legitimate new
    one, or vice versa), and (b) run the result through decide_diagram
    at all (the same Stage 3/4/5 structural/consistency validation
    every other diagram in this pipeline goes through) before trusting
    or rendering it; this function only ever proposes a candidate.

    Raises ValueError if the AI response isn't a JSON object, or
    propagates the underlying API error after retries are exhausted
    (see _call_gemini_recovery) — callers decide how to fail safely
    (apply_diagram_safety_net below catches and flags rather than
    raising further; correction_engine does the same for the reviewer-
    facing path, per this project's "no diagram beats a wrong diagram,
    but never fail silently" policy).
    """
    prompt = _build_recovery_prompt(question, class_name, extra_instruction=extra_instruction)
    result = _call_gemini_recovery(prompt)
    if not isinstance(result, dict):
        raise ValueError("Diagram generation response wasn't a JSON object.")
    return result


def apply_diagram_safety_net(questions: list, class_name: int, exercise_label: str):
    """Mutates `questions` in place. Never raises — every failure mode
    (API error, bad JSON, a recovered spec that still fails
    decide_diagram's own checks) leaves the question exactly as it was,
    with only an audit flag added so the gap is visible in logs rather
    than silently disappearing.

    IMPORTANT: this must run strictly AFTER the main decide_diagram
    loop in solver.py (needs question["decision"] /
    question["diagram_decision_reason"] already populated), and any
    question this module successfully recovers a diagram for must be
    re-run through decide_diagram so the same full Stage 3/4/5
    validation applies to the recovered spec too — this module never
    writes a final decision itself.
    """
    candidates = [q for q in questions
                  if _is_recoverable(q) and _looks_like_geometry_question(q)]
    if not candidates:
        return

    recovered = 0
    still_missing = 0
    for q in candidates:
        qnum = f"{q.get('question_number')}{q.get('sub_part') or ''}"
        try:
            result = generate_diagram_from_scratch(q, class_name)
        except Exception as e:
            logger.warning(f"⚠️ Exercise {exercise_label} Q{qnum}: diagram safety-net recovery "
                            f"call failed ({e}) — leaving as NO_DIAGRAM.")
            q["diagram_safety_net_flagged"] = True
            still_missing += 1
            continue

        if not isinstance(result, dict) or not result.get("needs_diagram") or not result.get("diagram_spec"):
            # Independent second opinion agrees no diagram is needed —
            # or returned something unusable; either way, fail safely.
            q["diagram_safety_net_flagged"] = not isinstance(result, dict) or bool(result.get("needs_diagram"))
            if q["diagram_safety_net_flagged"]:
                still_missing += 1
            continue

        # Hand the recovered spec through the SAME decision engine every
        # other spec goes through — no special-casing, no bypass.
        # construction_instruments MUST be set alongside diagram_spec
        # (both from the SAME recovery response) BEFORE decide_diagram
        # runs its Stage 4 consistency check, or a stale/absent value
        # left over from this question's prior (rejected) state would
        # wrongly reject a legitimate diagram_type="construction"
        # recovery — see generate_diagram_from_scratch's own docstring.
        q["diagram_spec"] = result["diagram_spec"]
        q["construction_instruments"] = result.get("construction_instruments")
        verdict = decide_diagram(q)
        if verdict["decision"] == NO_DIAGRAM:
            logger.warning(f"⚠️ Exercise {exercise_label} Q{qnum}: safety-net recovered a "
                            f"diagram_spec but it failed decide_diagram's own checks "
                            f"({verdict['reason']}) — leaving as NO_DIAGRAM rather than "
                            f"publishing an unverified drawing.")
            q["diagram_spec"] = None
            q["construction_instruments"] = None
            q["diagram_safety_net_flagged"] = True
            still_missing += 1
            continue

        q["diagram_spec"] = verdict["diagram_spec"]
        q["diagram_decision"] = verdict["decision"]
        q["diagram_decision_reason"] = f"[safety-net recovery] {verdict['reason']}"
        q["diagram_safety_net_flagged"] = False
        recovered += 1

    if recovered:
        logger.info(f"🛟 Exercise {exercise_label}: diagram safety-net recovered {recovered} "
                    f"missed diagram(s) that the main solve pass skipped.")
    if still_missing:
        logger.warning(f"⚠️ Exercise {exercise_label}: {still_missing} question(s) look like they "
                        f"may need a diagram but none could be verified — flagged for human "
                        f"review (question['diagram_safety_net_flagged'] = True), not guessed.")
