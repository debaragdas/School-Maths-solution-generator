"""
interactive_review.py — the interactive, human-in-the-loop review loop
that is main.py's PRIMARY workflow (see main.py's Stage 2): after an
exercise's draft is generated, the pipeline PAUSES on every question
that needs review (config.INTERACTIVE_REVIEW_SCOPE) and will not mark
the exercise approved — and therefore it can never be published, see
review_state.publish_chapter() — until the reviewer has explicitly
accepted every one of its questions, live, one at a time.

There is deliberately NO separate offline/CLI review step in this
workflow: review_cli.py still exists for ad-hoc use (e.g. approving
from a different machine, or bulk-listing status), but main.py's
normal run is now fully self-contained — solve, review, approve,
publish, all in one invocation.

Reuses review_state.py, correction_engine.py, and figure_database.py
completely unchanged — this module is purely the console UI wired on
top of them. `input_fn` / `print_fn` / `open_pdf_fn` are injectable
(default to real input()/print() and a cross-platform PDF opener) so
this loop is unit-testable with a scripted sequence of fake reviewer
responses instead of blocking on real stdin.
"""
import os
import platform
import subprocess

import config
import review_state
import correction_engine
from utils import logger


def open_pdf(path: str) -> None:
    """Best-effort, cross-platform 'open this PDF in whatever the OS
    considers its default viewer' for the [P]review action — never
    raises; a reviewer who can't get the preview to pop up (e.g. no
    GUI on a headless server) can still open the file manually at the
    printed path."""
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(path)  # noqa: only exists on Windows, guarded by the check above
        elif system == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        logger.warning(f"⚠️ Could not auto-open {path} for preview ({e}) — open it manually to review.")


def _question_label(question: dict) -> str:
    qn, sp = question.get("question_number"), question.get("sub_part")
    return f"Q{qn}" + (f"({sp})" if sp else "")


def question_needs_review(question: dict, scope: str) -> bool:
    """scope='all' (the default — see config.py): every question pauses
    for review, matching the brief's primary requirement ("after every
    generated question..."). scope='flagged_or_diagram': only questions
    the pipeline itself flagged (low self-verification confidence) or
    that carry any diagram/figure pause — a lighter-touch mode for
    teams that trust plain-arithmetic answers by default. Any other/
    unrecognized value fails SAFE towards reviewing more, not less.
    """
    if scope == "flagged_or_diagram":
        flagged = bool(question.get("needs_review"))
        has_diagram = bool(question.get("diagram_spec")) or bool(question.get("book_diagram_base64")) \
            or bool(question.get("book_diagram_figure_ref"))
        return flagged or has_diagram
    return True


def _print_question(print_fn, question: dict) -> None:
    print_fn(f"\n{'=' * 72}")
    print_fn(f"{_question_label(question)}: {(question.get('question_text') or '')[:220]}")
    print_fn(f"{'-' * 72}")
    print_fn(f"Final answer : {question.get('final_answer', '')}")
    diagram = question.get("diagram_decision", "NO_DIAGRAM")
    fig_ref = question.get("book_diagram_figure_ref")
    print_fn(f"Diagram      : {diagram}" + (f" (Figure {fig_ref})" if fig_ref else ""))
    if question.get("needs_review"):
        print_fn("⚠️  Flagged for review — self-verification confidence was below threshold.")
    print_fn(f"{'=' * 72}")


def _prompt_for_bbox_crop(print_fn, input_fn, image_bytes: bytes):
    """Returns (possibly-cropped image_bytes, bbox or None). Never
    raises — a bad crop box just leaves the image un-cropped and tells
    the reviewer why, rather than aborting the whole figure-attach."""
    bbox_raw = input_fn("Crop box as 'x1,y1,x2,y2' (leave blank if the image is already cropped): ").strip()
    if not bbox_raw:
        return image_bytes, None
    try:
        from PIL import Image
        import io
        x1, y1, x2, y2 = (int(v) for v in bbox_raw.split(","))
        img = Image.open(io.BytesIO(image_bytes))
        cropped = img.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        cropped.save(buf, format=(img.format or "PNG"))
        return buf.getvalue(), [x1, y1, x2, y2]
    except Exception as e:
        print_fn(f"⚠️ Could not crop with box {bbox_raw!r} ({e}) — using the image uncropped instead.")
        return image_bytes, None


def review_exercise_interactively(pdf_output_path: str, reviewer: str = None, scope: str = None,
                                   input_fn=input, print_fn=print, open_pdf_fn=open_pdf) -> bool:
    """Walks every question of the exercise at `pdf_output_path` that
    needs review, pausing for an Accept / Regenerate / Add-figure /
    Preview / Quit decision on each, looping on that SAME question
    until it's accepted ("Repeat until accepted" — see the brief).
    Marks the whole exercise approved (review_state.approve) only once
    every question has been individually accepted, then returns True.

    Returns False if the reviewer quits early — the exercise stays
    pending_review, but everything accepted (and every correction /
    figure attached) so far is already persisted, so simply re-running
    main.py resumes this exact exercise exactly where they left off,
    without re-asking about anything already handled and without
    wasting another Gemini call on questions that didn't need one.
    """
    scope = scope or getattr(config, "INTERACTIVE_REVIEW_SCOPE", "all")
    state = review_state.load_state(pdf_output_path)
    if state is None:
        logger.warning(f"⚠️ No review state for {pdf_output_path} — cannot run interactive review.")
        return False
    if state["status"] == review_state.STATUS_APPROVED:
        return True  # nothing to do — a previous session already finished this one

    label = state.get("exercise_label")
    print_fn(f"\n📝 Reviewing Exercise {label}  —  {pdf_output_path}")

    # Question IDENTITY (number/sub_part) is fixed once solved; re-fetch
    # each question's CONTENT fresh every loop iteration below, since a
    # correction/figure-attach changes it on disk.
    question_keys = [(q.get("question_number"), q.get("sub_part")) for q in state["solved"].get("questions", [])]

    for qn, sp in question_keys:
        while True:
            state = review_state.load_state(pdf_output_path)
            if state is None:
                logger.warning(f"⚠️ Review state for {pdf_output_path} disappeared mid-review.")
                return False
            fresh_q = correction_engine._find_question(state["solved"].get("questions", []), qn, sp)
            if fresh_q is None:
                logger.warning(f"⚠️ Question {qn} vanished from {pdf_output_path} mid-review — skipping it.")
                break

            if review_state.is_question_accepted(state, qn, sp):
                break

            if not question_needs_review(fresh_q, scope):
                review_state.mark_question_accepted(pdf_output_path, qn, sp, reviewer=reviewer)
                continue

            _print_question(print_fn, fresh_q)
            choice = input_fn(
                "[A]ccept  [R]egenerate w/ correction  [F]igure upload  [P]review PDF  [Q]uit review > "
            ).strip().lower()

            if choice in ("a", "accept"):
                review_state.mark_question_accepted(pdf_output_path, qn, sp, reviewer=reviewer)
                continue

            elif choice in ("r", "regenerate"):
                instruction = input_fn("Correction prompt — what's wrong and what to fix: ").strip()
                if not instruction:
                    print_fn("(empty correction — nothing changed)")
                    continue
                try:
                    correction_engine.regenerate_question(
                        pdf_output_path, qn, instruction, sub_part=sp, reviewer=reviewer,
                    )
                    print_fn("✅ Regenerated — re-showing the updated question.")
                except Exception as e:
                    print_fn(f"❌ Regeneration failed: {e}")
                continue

            elif choice in ("f", "figure"):
                image_path = input_fn("Path to the screenshot (or an already-cropped figure): ").strip()
                if not image_path or not os.path.exists(image_path):
                    print_fn(f"❌ File not found: {image_path!r}")
                    continue
                default_ref = fresh_q.get("book_diagram_figure_ref")
                figure_number = input_fn(
                    f"Figure number as printed in the book{f' [{default_ref}]' if default_ref else ''}: "
                ).strip() or default_ref
                if not figure_number:
                    print_fn("❌ A figure number is required.")
                    continue
                page_raw = input_fn("Page number in the book (optional, press Enter to skip): ").strip()
                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()
                    image_bytes, bbox = _prompt_for_bbox_crop(print_fn, input_fn, image_bytes)
                    correction_engine.attach_manual_figure_to_question(
                        pdf_output_path, qn, image_bytes=image_bytes, figure_number=figure_number,
                        sub_part=sp, page_number=int(page_raw) if page_raw.isdigit() else None,
                        bbox=bbox, reviewer=reviewer,
                    )
                    print_fn("✅ Figure stored and attached — re-showing the updated question.")
                except Exception as e:
                    print_fn(f"❌ Could not attach figure: {e}")
                continue

            elif choice in ("p", "preview"):
                open_pdf_fn(pdf_output_path)
                print_fn(f"(opened {pdf_output_path} for preview — come back here once you've looked at it)")
                continue

            elif choice in ("q", "quit"):
                print_fn(f"⏸️  Pausing review of Exercise {label}. Re-run main.py to resume — "
                         f"every question already accepted (and every correction/figure attached) is saved.")
                return False

            else:
                print_fn("Not a recognized option — try A / R / F / P / Q.")
                continue

    review_state.approve(pdf_output_path, reviewer=reviewer,
                          note="every question individually accepted in interactive review")
    print_fn(f"✅ Exercise {label} fully approved.")
    return True
