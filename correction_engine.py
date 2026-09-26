"""
correction_engine.py — Human Review workflow, capability 1: "Wrong AI
Output Correction".

Applies a reviewer's free-text correction prompt to exactly ONE
question of an already-generated exercise, using a narrow single-
question Gemini call (prompts.build_correction_prompt) — never a
second full-exercise solve call — then runs that one question back
through the SAME deterministic post-processing stages the main solve
pipeline already applies (solver's text normalization,
constraint_extraction, diagram_decision), and re-renders the exercise's
existing HTML/PDF via the UNCHANGED html_renderer.py / pdf_generator.py.
Every other question in the exercise is left byte-for-byte untouched.

This is the whole "regenerate ONLY the affected component" requirement:
the expensive part (an AI call) is scoped to one question; the
deterministic part (re-rendering the PDF, since this pipeline produces
one PDF per exercise) is cheap and was always going to re-run the exact
same template over the exact same (mostly-unchanged) question list.
"""
import os

from google.genai import types

import config
import solver
import review_state
import constraint_extraction
import diagram_safety_net
import figure_database
from prompts import build_correction_prompt
from diagram_decision import decide_diagram, NO_DIAGRAM, BOOK_DIAGRAM
from html_renderer import render_exercise_html
from pdf_generator import render_pdf
from utils import logger, retry_with_backoff


@retry_with_backoff(times=3, base_delay=3.0, max_delay=20.0,
                     retryable_check=solver._is_retryable_api_error)
def _call_gemini_correction(prompt: str):
    client = solver.get_client()
    return client.models.generate_content(
        # A correction re-solves a real question — same rigor bar as the
        # main solve pass, so it routes to the same SOLVE-tier model.
        model=config.GEMINI_SOLVE_MODEL_ID,
        contents=[prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=config.THINKING_BUDGET),
            response_mime_type="application/json",
        ),
    )


def _find_question(questions: list, question_number, sub_part=None):
    for q in questions:
        if str(q.get("question_number")) == str(question_number) and \
                (q.get("sub_part") or None) == (sub_part or None):
            return q
    return None


def edit_question_directly(pdf_output_path: str, question_number, 
                          field_updates: dict, sub_part=None, 
                          reviewer: str = None) -> dict:
    """
    Directly edits a question's text fields without AI regeneration.
    
    This allows human reviewers to manually fix typos, formatting issues,
    or replace entire solution text. The edited content is saved directly
    to the review state and will be rendered in the PDF with the same
    formatting (MathJax, Assamese font) as AI-generated content.
    
    Args:
        pdf_output_path: Path to the exercise PDF
        question_number: The question number to edit
        field_updates: Dict of field names to new values (e.g., 
                      {"given": "new given text", "steps": ["step1", "step2"]})
        sub_part: Optional sub-part (a, b, c, etc.)
        reviewer: Optional reviewer name
    
    Returns:
        The updated state dict
    """
    with review_state.state_lock(pdf_output_path):
        state = review_state.load_state(pdf_output_path)
        if state is None:
            raise ValueError(f"No review state found for {pdf_output_path}")
        
        questions = state.get("solved", {}).get("questions", [])
        question = _find_question(questions, question_number, sub_part)
        
        if question is None:
            raise ValueError(f"Question {question_number}{'(' + sub_part + ')' if sub_part else ''} not found")
        
        # Apply the field updates
        for field, new_value in field_updates.items():
            if field in question:
                question[field] = new_value
                logger.info(f"📝 Edited field '{field}' for Q{question_number}")
            else:
                logger.warning(f"⚠️ Field '{field}' not found in question - skipping")
        
        # Update version and add history entry
        state["version"] += 1
        state["status"] = review_state.STATUS_PENDING
        key = _question_key(question_number, sub_part)
        accepted = state.setdefault("accepted_questions", [])
        if key in accepted:
            accepted.remove(key)
        
        state.setdefault("history", []).append({
            "version": state["version"],
            "action": "direct_edit",
            "question_number": question_number,
            "sub_part": sub_part,
            "correction_instruction": f"Direct edit of fields: {list(field_updates.keys())}",
            "reviewer": reviewer,
            "timestamp": review_state._now() if hasattr(review_state, '_now') else "unknown"
        })
        
        # Re-render the PDF with the edited content
        html = render_exercise_html(
            state["solved"], 
            state.get("class_name"), 
            state.get("chapter"), 
            state.get("chapter_name"), 
            state.get("exercise_label")
        )
        _atomic_render_pdf(html, pdf_output_path)
        
        review_state.save_state(pdf_output_path, state)
        logger.info(f"✅ Direct edit saved for Q{question_number}, PDF re-rendered")
    
    return state


def _question_key(question_number, sub_part=None) -> str:
    """Helper to generate a unique key for a question."""
    return f"{question_number}" + (f"({sub_part})" if sub_part else "")


def _atomic_render_pdf(html: str, final_path: str) -> None:
    """PRODUCTION-AUDIT FIX: unlike main.py's FIRST-EVER render of an
    exercise (where a failed/crashed render just leaves no file, since
    nothing existed there before), a correction re-renders a PDF that
    may already be a human-APPROVED, previously-good file. Calling
    render_pdf() directly against `final_path` would let a crash or
    truncated write during PDF generation (browser crash, disk full,
    a kill mid-write — the same class of event no application code can
    fully prevent, only contain) silently overwrite that good file with
    garbage, permanently losing previously-verified work with no
    recovery. Reproduced directly: mocking render_pdf to write a
    truncated file then raise leaves the truncated file in place at
    `final_path` with the old code path — see
    test_correction_engine.py::test_failed_rerender_never_destroys_the_previous_good_pdf.

    Fixed the same way this codebase already handles every other
    "must not partially overwrite" case (figure_database.py's
    index.json, review_state.py's own state file): render to a
    same-directory temp file, sanity-check its size, and only then
    atomically os.replace() it onto `final_path` — the previous file
    at `final_path` is never touched until a genuinely complete
    replacement is ready.
    """
    tmp_path = final_path + ".correcting.tmp"
    try:
        render_pdf(html, tmp_path)
        # Same "suspiciously small = something went wrong" heuristic
        # utils.already_done() already uses for a freshly generated PDF.
        if not (os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000):
            raise RuntimeError(
                f"Correction re-render produced a missing/suspiciously small PDF at {tmp_path} — "
                f"refusing to overwrite the previous PDF at {final_path}."
            )
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, final_path)


_COMPONENT_PREFIXES = {
    "solution": (
        "SCOPE: only the worked solution (given/required/steps/final_answer) needs to "
        "change. Do NOT touch diagram_spec / construction_instruments unless the "
        "instruction below explicitly asks you to — leave the diagram exactly as it is.\n\n"
    ),
    "diagram": (
        "SCOPE: only the diagram (diagram_spec / construction_instruments) needs to "
        "change. Do NOT touch given/required/steps/final_answer unless the instruction "
        "below explicitly asks you to — leave the worked solution exactly as it is.\n\n"
    ),
}


def regenerate_question(pdf_output_path: str, question_number, correction_instruction: str,
                         sub_part=None, reviewer: str = None, component: str = None) -> dict:
    """Applies `correction_instruction` to question `question_number`
    (+ optional `sub_part`) of the exercise whose draft PDF already
    lives at `pdf_output_path`, re-renders that same PDF in place, and
    returns the updated review state (see review_state.py).

    `component` is an optional UI-level hint — "solution" or "diagram"
    (anything else, including None, is treated as unscoped/"both", the
    original behavior) — used by the Human Review app's separate
    "Regenerate Solution" / "Regenerate Diagram" buttons (as opposed to
    its free-form "Custom Prompt" button, which passes component=None)
    to bias the SAME single-question correction call towards leaving
    the other half of the question untouched, without adding a second
    Gemini call or a second code path. Purely a prompt-scoping hint —
    the reviewer's own `correction_instruction` text is still sent
    verbatim and can still override it (e.g. "fix step 3 AND swap the
    diagram to a right triangle").

    Raises ValueError if there's no persisted review state for this PDF
    (nothing to correct against — e.g. it predates this feature or its
    sidecar was deleted) or the question can't be found. These are real
    usage errors a reviewer should see immediately, unlike the rest of
    this pipeline's fail-open philosophy for optional enhancements.
    """
    # Initial, UNLOCKED read — only to build the correction prompt from
    # the question's current content and grab exercise metadata (class/
    # chapter/label), which are not expected to change between reviewers.
    # This is allowed to be a slightly stale snapshot; the AI call is the
    # slow part and holding a lock across it would only serialize
    # reviewers for no benefit. The actual merge below re-reads fresh.
    state = review_state.load_state(pdf_output_path)
    if state is None:
        raise ValueError(
            f"No review state found for {pdf_output_path} — was it generated by this "
            f"pipeline's main.py, or has its .review.json sidecar been deleted?"
        )
    question_snapshot = _find_question(state["solved"].get("questions", []), question_number, sub_part)
    if question_snapshot is None:
        label = f"{question_number}({sub_part})" if sub_part else str(question_number)
        raise ValueError(f"Question {label} not found in {pdf_output_path}.")

    class_name, chapter = state.get("class_name"), state.get("chapter")
    exercise_label = state.get("exercise_label")

    scoped_instruction = _COMPONENT_PREFIXES.get(component, "") + (correction_instruction or "")
    prompt = build_correction_prompt(question_snapshot, scoped_instruction,
                                      class_name=class_name, chapter=chapter, exercise_label=exercise_label)
    logger.info(f"✏️ Applying reviewer correction ({component or 'both'}) to Question "
                f"{question_number} of Exercise {exercise_label}: {correction_instruction!r}")

    response = _call_gemini_correction(prompt)
    raw_text = response.text or ""
    if not raw_text.strip():
        raise ValueError("Correction call returned empty output.")

    corrected = solver.parse_gemini_json_response(
        raw_text, f"Correction response for Question {question_number}")

    if not isinstance(corrected, dict):
        raise ValueError("Correction response wasn't a JSON object.")

    # Never let the correction call itself change WHICH question this is —
    # only its content.
    corrected["question_number"] = question_snapshot.get("question_number")
    corrected["sub_part"] = question_snapshot.get("sub_part")

    # Reuse the EXACT same text-normalization / answer-kind-classification
    # pass the main solve pipeline applies to every question, rather than
    # re-implementing any of that logic here (see solver.py).
    temp = {"questions": [corrected]}
    solver._normalize_all_text_fields(temp)
    corrected = temp["questions"][0]

    # Same Stage 0 deterministic constraint audit the main pipeline runs,
    # scoped to just this one question.
    try:
        constraint_extraction.sanitize_questions([corrected])
    except Exception as e:
        logger.warning(f"⚠️ Correction for Question {question_number}: constraint sanitation "
                        f"step failed ({e}) — proceeding with the unsanitized diagram_spec.")

    # PRODUCTION-AUDIT FIX: everything from here on (reload, merge,
    # render, save) runs inside ONE lock acquisition against a FRESHLY
    # reloaded state — never the `state`/`question_snapshot` read above,
    # which may now be stale (a concurrent correction to a DIFFERENT
    # question of this same exercise could have already been saved while
    # this one was waiting on the Gemini call). Merging into a stale full
    # copy and saving it wholesale would silently revert that other
    # correction — reproduced directly: two threads correcting different
    # questions of the same exercise, the one that finishes later's stale
    # copy overwrites the other's already-saved fix — see
    # test_correction_engine.py::test_concurrent_corrections_to_different_questions_do_not_lose_either_fix.
    with review_state.state_lock(pdf_output_path):
        fresh_state = review_state.load_state(pdf_output_path)
        if fresh_state is None:
            raise ValueError(f"Review state for {pdf_output_path} disappeared during correction.")
        fresh_solved = fresh_state["solved"]
        fresh_question = _find_question(fresh_solved.get("questions", []), question_number, sub_part)
        if fresh_question is None:
            label = f"{question_number}({sub_part})" if sub_part else str(question_number)
            raise ValueError(f"Question {label} vanished from {pdf_output_path} between correction and save.")

        # Merge the corrected fields into the FRESH question object in
        # place (preserves any bookkeeping field the correction schema
        # doesn't cover, e.g. book_diagram_base64, until the diagram
        # decision stage below re-evaluates it) — every OTHER question in
        # `fresh_solved["questions"]` (including any concurrently-saved
        # fix to a different question) is left completely untouched.
        fresh_question.update(corrected)

        # Same Stage 1 diagram decision the main pipeline runs after every
        # solve — re-derives book-vs-generated precedence for the
        # (possibly now different) diagram_spec, never a second AI call.
        verdict = decide_diagram(fresh_question)
        fresh_question["diagram_spec"] = verdict["diagram_spec"]
        fresh_question["diagram_decision"] = verdict["decision"]
        fresh_question["diagram_decision_reason"] = verdict["reason"]
        if verdict["decision"] != BOOK_DIAGRAM:
            fresh_question["book_diagram_base64"] = None
            fresh_question["book_diagram_mime"] = None
            fresh_question["book_diagram_figure_ref"] = None

        html = render_exercise_html(
            fresh_solved, class_name=fresh_state.get("class_name"), chapter=fresh_state.get("chapter"),
            chapter_name=fresh_state.get("chapter_name"), exercise_label=fresh_state.get("exercise_label"),
        )
        _atomic_render_pdf(html, pdf_output_path)

        try:
            import layout_validation
            layout_validation.validate_and_log(pdf_output_path, exercise_label=fresh_state.get("exercise_label"))
        except Exception as e:
            # Best-effort audit only — must never fail an otherwise successful
            # correction (same policy main.py already applies after a fresh
            # generation — see layout_validation.py's own docstring).
            logger.warning(f"⚠️ Exercise {fresh_state.get('exercise_label')}: layout validation audit "
                            f"failed after correction ({e}) — the PDF was still re-rendered normally.")

        new_state = review_state.finalize_correction(
            fresh_state, pdf_output_path, question_number=question_number,
            correction_instruction=correction_instruction, sub_part=sub_part, reviewer=reviewer,
        )

    logger.info(f"✅ Question {question_number} corrected and Exercise "
                f"{exercise_label} re-rendered → {pdf_output_path} "
                f"(now version {new_state['version']}, status={new_state['status']}).")
    return new_state


def attach_manual_figure_to_question(pdf_output_path: str, question_number, image_bytes: bytes,
                                      figure_number, sub_part=None, subject: str = None,
                                      page_number=None, bbox=None, notes: str = None,
                                      reviewer: str = None, ext: str = None, book_url: str = None) -> dict:
    """Human Review workflow, capability 2: "Missing Book Figure" — the
    figure-attach counterpart to regenerate_question(), used by
    interactive_review.py's per-question loop when a reviewer supplies
    a manually cropped textbook figure right there instead of a text
    correction.

    Permanently stores the crop via figure_database.add_manual_figure()
    (reused unchanged — so it's ALSO picked up automatically by every
    future exercise citing the same figure number, exactly like a
    figure the automatic extractor found), then attaches it to THIS one
    question and re-renders — using the identical locked
    reload-merge-render-save pattern regenerate_question() uses, for
    the same concurrency-safety reasons (see review_state.state_lock()'s
    docstring): two reviewers acting on different questions of the same
    exercise at the same time must never be able to revert each other.
    """
    import base64

    state = review_state.load_state(pdf_output_path)
    if state is None:
        raise ValueError(
            f"No review state found for {pdf_output_path} — was it generated by this "
            f"pipeline's main.py, or has its .review.json sidecar been deleted?"
        )
    if _find_question(state["solved"].get("questions", []), question_number, sub_part) is None:
        label = f"{question_number}({sub_part})" if sub_part else str(question_number)
        raise ValueError(f"Question {label} not found in {pdf_output_path}.")

    resolved_ext = ext or figure_database.detect_image_ext(image_bytes, fallback="png")

    if figure_number:
        # Real textbook citation: store it in the shared, citable
        # figure database (format-validated, reusable by any future
        # question that cites the same figure number) — unchanged
        # from before.
        entry = figure_database.add_manual_figure(
            image_bytes=image_bytes, figure_number=figure_number,
            class_name=state.get("class_name"), chapter=state.get("chapter"),
            subject=subject, page_number=page_number, reviewer=reviewer,
            book_url=book_url or config.BOOK_URL, ext=resolved_ext, bbox=bbox, notes=notes,
        )
        figure_ref = entry["figure_ref"]
        logger.info(f"📌 Storing manually cropped Figure {figure_ref} for Question "
                    f"{question_number} of Exercise {state.get('exercise_label')} (version {entry['version']}).")
    else:
        # No textbook citation — this image doesn't correspond to any
        # numbered book figure (e.g. a construction/number-line
        # question that never had one), so it does NOT belong in the
        # shared, citation-format-validated figure database at all.
        # Attach it directly to just this one question instead. Not
        # reusable by other questions and not subject to the "does
        # this look like a real figure number" validation
        # figure_database.add_manual_figure enforces for citable
        # entries — deliberately, since there's no citation to validate.
        figure_ref = None
        logger.info(f"📌 Attaching a manually uploaded (non-cited) image directly to Question "
                    f"{question_number} of Exercise {state.get('exercise_label')}.")

    # Same locked fresh-reload pattern as regenerate_question — see that
    # function's comment for the exact cross-question race this avoids.
    with review_state.state_lock(pdf_output_path):
        fresh_state = review_state.load_state(pdf_output_path)
        if fresh_state is None:
            raise ValueError(f"Review state for {pdf_output_path} disappeared while attaching a figure.")
        fresh_solved = fresh_state["solved"]
        fresh_question = _find_question(fresh_solved.get("questions", []), question_number, sub_part)
        if fresh_question is None:
            label = f"{question_number}({sub_part})" if sub_part else str(question_number)
            raise ValueError(f"Question {label} vanished from {pdf_output_path} while attaching a figure.")

        fresh_question["book_diagram_base64"] = base64.b64encode(image_bytes).decode("ascii")
        fresh_question["book_diagram_mime"] = f"image/{resolved_ext}"
        fresh_question["book_diagram_figure_ref"] = figure_ref
        fresh_question["has_book_diagram"] = True

        # A reviewer manually uploading an image through this exact
        # action is the highest-trust source this project has — do NOT
        # run it through decide_diagram's automatic citation check
        # (_verify_book_citation), which requires the question's OWN
        # text to explicitly cite a figure number. That check exists to
        # catch the AUTOMATIC extractor attaching the wrong figure; it
        # has no business second-guessing a human who just deliberately
        # uploaded this image for this exact question. Previously, any
        # question with no figure citation in its text at all (common
        # for construction/number-line questions that never had a book
        # figure to begin with) had its manual upload silently
        # discarded here, with no visible error to the reviewer.
        fresh_question["diagram_spec"] = None
        fresh_question["diagram_decision"] = BOOK_DIAGRAM
        fresh_question["diagram_decision_reason"] = "manually attached by reviewer — trusted as-is"

        html = render_exercise_html(
            fresh_solved, class_name=fresh_state.get("class_name"), chapter=fresh_state.get("chapter"),
            chapter_name=fresh_state.get("chapter_name"), exercise_label=fresh_state.get("exercise_label"),
        )
        _atomic_render_pdf(html, pdf_output_path)

        try:
            import layout_validation
            layout_validation.validate_and_log(pdf_output_path, exercise_label=fresh_state.get("exercise_label"))
        except Exception as e:
            logger.warning(f"⚠️ Exercise {fresh_state.get('exercise_label')}: layout validation audit "
                            f"failed after figure attach ({e}) — the PDF was still re-rendered normally.")

        figure_desc = f"textbook Figure {figure_ref}" if figure_ref else "a manually uploaded image (no book citation)"
        new_state = review_state.finalize_correction(
            fresh_state, pdf_output_path, question_number=question_number,
            correction_instruction=f"Manually attached {figure_desc}"
                                    + (f" (reviewer note: {notes})" if notes else ""),
            sub_part=sub_part, reviewer=reviewer, action="figure_added",
        )

    logger.info(f"✅ Question {question_number} of Exercise {fresh_state.get('exercise_label')} "
                f"now uses {figure_desc} (decision=BOOK_DIAGRAM) "
                f"→ {pdf_output_path} (version {new_state['version']}).")
    return new_state


def _locked_diagram_update(pdf_output_path: str, question_number, sub_part, reviewer: str,
                            correction_instruction: str, action: str, mutate_fn) -> dict:
    """Shared plumbing for the three diagram-only actions below (rebuild
    from scratch, remove generated diagram, remove book figure) — the
    exact same locked reload-merge-render-save sequence regenerate_
    question()/attach_manual_figure_to_question() use, for the same
    concurrency-safety reasons (see review_state.state_lock()'s
    docstring). `mutate_fn(fresh_question)` makes the one actual change;
    everything else (decide_diagram re-evaluation, atomic re-render,
    layout-validation audit, state bookkeeping) is identical across all
    three actions, so it lives here once instead of three times.
    """
    with review_state.state_lock(pdf_output_path):
        fresh_state = review_state.load_state(pdf_output_path)
        if fresh_state is None:
            raise ValueError(f"Review state for {pdf_output_path} disappeared during a diagram update.")
        fresh_solved = fresh_state["solved"]
        fresh_question = _find_question(fresh_solved.get("questions", []), question_number, sub_part)
        if fresh_question is None:
            label = f"{question_number}({sub_part})" if sub_part else str(question_number)
            raise ValueError(f"Question {label} vanished from {pdf_output_path} during a diagram update.")

        mutate_fn(fresh_question)

        html = render_exercise_html(
            fresh_solved, class_name=fresh_state.get("class_name"), chapter=fresh_state.get("chapter"),
            chapter_name=fresh_state.get("chapter_name"), exercise_label=fresh_state.get("exercise_label"),
        )
        _atomic_render_pdf(html, pdf_output_path)

        try:
            import layout_validation
            layout_validation.validate_and_log(pdf_output_path, exercise_label=fresh_state.get("exercise_label"))
        except Exception as e:
            logger.warning(f"⚠️ Exercise {fresh_state.get('exercise_label')}: layout validation audit "
                            f"failed after a diagram update ({e}) — the PDF was still re-rendered normally.")

        new_state = review_state.finalize_correction(
            fresh_state, pdf_output_path, question_number=question_number,
            correction_instruction=correction_instruction, sub_part=sub_part,
            reviewer=reviewer, action=action,
        )
    return new_state


def regenerate_diagram_from_scratch(pdf_output_path: str, question_number, sub_part=None,
                                     instruction: str = None, reviewer: str = None) -> dict:
    """Human Review workflow, capability 3: rebuild — or, if none exists
    yet, generate for the very first time — a question's diagram
    completely FROM THE QUESTION, never from any previous diagram.

    This is a deliberately separate code path from regenerate_question()
    (component="diagram"), which asks Gemini to EDIT the question's
    current diagram_spec/steps in place — usable, but exactly the
    failure mode reported in production: an edit call can anchor on the
    previous (wrong) spec, ignore the correction prompt, or drift into
    coordinate geometry for an ordinary construction question. This
    function instead reruns the SAME pipeline a question's diagram goes
    through the very first time it's ever solved:

        question text -> diagram_safety_net.generate_diagram_from_scratch
        (one narrow, independent Gemini call whose entire context is
        this question's own text — optionally plus the reviewer's
        instruction — never the old diagram_spec/SVG)
        -> diagram_decision.decide_diagram (the SAME Stage 3/4/5
        structural/consistency validation every diagram in this
        pipeline must pass — type validity, internal consistency,
        textual grounding, structural repair-or-reject)
        -> diagram_renderer (via html_renderer/pdf_generator's normal
        render path, unchanged)

    This is also what makes "Generate Diagram" for a question with NO
    diagram and NO book figure work at all: `question`'s current
    diagram_spec is irrelevant here (there is none), so the exact same
    call produces one from nothing.

    If the rebuilt candidate fails decide_diagram's validation (or the
    AI call itself fails), the question is left WITHOUT a diagram
    rather than a wrong one — this project's standing "a wrong diagram
    is worse than no diagram" policy (see diagram_safety_net.py) — and
    is flagged `needs_review=True` with a human-readable review_note
    explaining why, so the gap is visible in the review UI instead of
    silently vanishing.

    Never touches book_diagram_* — if a book figure is separately
    attached, decide_diagram's own precedence rules mean it's untouched
    and still shown/preferred exactly as before.
    """
    state = review_state.load_state(pdf_output_path)
    if state is None:
        raise ValueError(
            f"No review state found for {pdf_output_path} — was it generated by this "
            f"pipeline's main.py, or has its .review.json sidecar been deleted?"
        )
    question_snapshot = _find_question(state["solved"].get("questions", []), question_number, sub_part)
    if question_snapshot is None:
        label = f"{question_number}({sub_part})" if sub_part else str(question_number)
        raise ValueError(f"Question {label} not found in {pdf_output_path}.")

    class_name = state.get("class_name")
    exercise_label = state.get("exercise_label")
    logger.info(f"🖼️ Rebuilding diagram from scratch for Question {question_number} of "
                f"Exercise {exercise_label}"
                + (f" — reviewer guidance: {instruction!r}" if instruction else " (no diagram existed yet)"))

    failure_note = None
    result = None
    try:
        result = diagram_safety_net.generate_diagram_from_scratch(
            question_snapshot, class_name, extra_instruction=instruction)
    except Exception as e:
        logger.warning(f"⚠️ Exercise {exercise_label} Q{question_number}: diagram rebuild call "
                        f"failed ({e}) — leaving without a diagram.")
        failure_note = f"Diagram rebuild call failed ({e})."

    def _mutate(fresh_question: dict) -> None:
        notes = fresh_question.setdefault("review_notes", [])
        if result and result.get("needs_diagram") and result.get("diagram_spec"):
            # PRODUCTION-AUDIT FIX (final pre-launch round): diagram_spec
            # and construction_instruments MUST be written together, from
            # THIS SAME rebuild response, before decide_diagram runs — its
            # Stage 4 consistency check requires them paired (see
            # diagram_safety_net.generate_diagram_from_scratch's
            # docstring); leaving fresh_question's OLD construction_
            # instruments in place here is exactly what could make a
            # perfectly good rebuilt diagram get silently rejected (e.g.
            # the old diagram was type "construction" and the new one
            # correctly isn't, or vice versa).
            fresh_question["diagram_spec"] = result["diagram_spec"]
            fresh_question["construction_instruments"] = result.get("construction_instruments")
            verdict = decide_diagram(fresh_question)
            if verdict["decision"] == NO_DIAGRAM:
                fresh_question["diagram_spec"] = None
                fresh_question["construction_instruments"] = None
                fresh_question["diagram_decision"] = NO_DIAGRAM
                fresh_question["diagram_decision_reason"] = (
                    f"[reviewer-requested rebuild] candidate rejected: {verdict['reason']}")
                fresh_question["needs_review"] = True
                notes.append(f"Diagram rebuild produced a candidate that failed validation "
                              f"({verdict['reason']}) — left without a diagram rather than risk "
                              f"an incorrect one. Try Custom Prompt with more specific guidance.")
            else:
                fresh_question["diagram_spec"] = verdict["diagram_spec"]
                fresh_question["diagram_decision"] = verdict["decision"]
                fresh_question["diagram_decision_reason"] = f"[reviewer-requested rebuild] {verdict['reason']}"
                fresh_question["diagram_safety_net_flagged"] = False
                fresh_question["needs_review"] = False
        else:
            # No usable candidate at all — completely discard whatever
            # diagram-side state this question had before (this is the
            # reviewer explicitly asking for a fresh rebuild; a stale
            # construction_instruments left behind here is pure debris
            # that can only cause a FUTURE consistency-check false
            # rejection, never anything useful).
            fresh_question["diagram_spec"] = None
            fresh_question["construction_instruments"] = None
            fresh_question["diagram_decision"] = NO_DIAGRAM
            fresh_question["diagram_decision_reason"] = (
                failure_note or "[reviewer-requested rebuild] independent check found no diagram is needed")
            fresh_question["needs_review"] = True
            notes.append(failure_note or "Diagram rebuild's independent check concluded this question "
                                          "doesn't need a diagram — if you believe it does, use Custom "
                                          "Prompt with more specific guidance instead.")

    correction_instruction = instruction or "Rebuild diagram from scratch (no specific guidance given)"
    new_state = _locked_diagram_update(
        pdf_output_path, question_number, sub_part, reviewer,
        correction_instruction=correction_instruction, action="diagram_rebuilt", mutate_fn=_mutate,
    )
    logger.info(f"✅ Question {question_number} of Exercise {exercise_label} diagram rebuilt "
                f"→ {pdf_output_path} (version {new_state['version']}).")
    return new_state


def remove_generated_diagram(pdf_output_path: str, question_number, sub_part=None, reviewer: str = None) -> dict:
    """Human Review workflow, capability 6: "Remove Generated Diagram" —
    one click clears ONLY the AI-generated diagram_spec; the worked
    solution (given/required/steps/final_answer) is left completely
    untouched. Re-runs decide_diagram afterward so that if a book
    figure is also attached, it's picked up exactly as it would have
    been anyway (this never touches book_diagram_*) — the question
    simply ends up with whatever it would have had if no generated
    diagram had ever been produced.
    """
    def _mutate(fresh_question: dict) -> None:
        fresh_question["diagram_spec"] = None
        fresh_question["construction_instruments"] = None
        verdict = decide_diagram(fresh_question)
        fresh_question["diagram_spec"] = verdict["diagram_spec"]
        fresh_question["diagram_decision"] = verdict["decision"]
        fresh_question["diagram_decision_reason"] = f"[reviewer] generated diagram removed manually; {verdict['reason']}"
        fresh_question["diagram_safety_net_flagged"] = False

    new_state = _locked_diagram_update(
        pdf_output_path, question_number, sub_part, reviewer,
        correction_instruction="Generated diagram removed by reviewer",
        action="diagram_removed", mutate_fn=_mutate,
    )
    logger.info(f"🗑️ Question {question_number}: generated diagram removed → {pdf_output_path} "
                f"(version {new_state['version']}).")
    return new_state


_DIAGRAM_POSITIONS = {"auto", "side", "large"}


def set_diagram_position(pdf_output_path: str, question_number, position: str,
                          sub_part=None, reviewer: str = None) -> dict:
    """Human Review workflow, capability 8: "Diagram Position" — lets a
    reviewer manually override where a question's diagram/book-figure
    is placed in the published PDF, independent of html_renderer.py's
    own auto heuristic (statistics/coordinate-plot diagram TYPES render
    large by default; everything else, including every book-scanned
    figure, defaults to the small fixed-width sidebar box regardless of
    how much fine detail it actually contains).

    position:
      "auto"  — remove any manual override; fall back to the normal
                per-type heuristic (html_renderer._LARGE_DIAGRAM_TYPES).
      "side"  — force the compact 210px sidebar box beside the text.
      "large" — force the wide, centered, stacked-below-text layout
                (html_renderer's "diagram-box--large"/"question-body--
                stacked"), for a book figure whose labels/text are too
                small to read at sidebar size — the exact case a
                reviewer needs this for, since an uploaded book figure
                never gets the large layout automatically today.

    Purely a layout preference: never touches diagram_spec, the SVG, or
    book_diagram_* — decide_diagram() is deliberately NOT re-run here,
    so this can never change *whether* a diagram is shown, only how
    big.
    """
    if position not in _DIAGRAM_POSITIONS:
        raise ValueError(f"Invalid diagram position {position!r} — must be one of {sorted(_DIAGRAM_POSITIONS)}.")

    def _mutate(fresh_question: dict) -> None:
        fresh_question["diagram_position"] = position if position != "auto" else None

    new_state = _locked_diagram_update(
        pdf_output_path, question_number, sub_part, reviewer,
        correction_instruction=f"Diagram position set to '{position}' by reviewer",
        action="diagram_position_set", mutate_fn=_mutate,
    )
    logger.info(f"↔️ Question {question_number}: diagram position set to '{position}' → {pdf_output_path} "
                f"(version {new_state['version']}).")
    return new_state


def remove_book_figure(pdf_output_path: str, question_number, sub_part=None, reviewer: str = None) -> dict:
    """Human Review workflow, capability 7: "Remove Book Figure" — one
    click clears an incorrectly auto-matched (or manually attached)
    textbook figure. The worked solution is left completely untouched.
    Does NOT delete anything from figure_database.py's permanent store
    (a wrong MATCH for this question doesn't mean the crop itself was
    wrong for whatever figure it actually is) — only detaches it from
    this question. Re-runs decide_diagram afterward, so any previously-
    rejected generated diagram_spec is reconsidered fresh rather than
    left stale; if nothing is left, the reviewer can then use
    "Generate Diagram" or upload the correct figure.
    """
    def _mutate(fresh_question: dict) -> None:
        fresh_question["book_diagram_base64"] = None
        fresh_question["book_diagram_mime"] = None
        fresh_question["book_diagram_figure_ref"] = None
        fresh_question["has_book_diagram"] = False
        verdict = decide_diagram(fresh_question)
        fresh_question["diagram_spec"] = verdict["diagram_spec"]
        fresh_question["diagram_decision"] = verdict["decision"]
        fresh_question["diagram_decision_reason"] = f"[reviewer] book figure removed manually; {verdict['reason']}"

    new_state = _locked_diagram_update(
        pdf_output_path, question_number, sub_part, reviewer,
        correction_instruction="Book figure removed by reviewer (was likely mismatched)",
        action="figure_removed", mutate_fn=_mutate,
    )
    logger.info(f"🗑️ Question {question_number}: book figure removed → {pdf_output_path} "
                f"(version {new_state['version']}).")
    return new_state
