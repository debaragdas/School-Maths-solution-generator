"""
review_state.py — Human Review workflow: persistent per-exercise
review state.

Before this module, the pipeline (main.py) wrote only a final PDF to
`output/.../ex_X_Y.pdf` — there was no persisted record of the solved
JSON that produced it and no place to record whether a human has
reviewed it. This adds exactly that, as a sidecar JSON file next to
the PDF (`ex_X_Y.review.json`), WITHOUT changing the PDF's own path or
filename convention at all — anything that just wants the PDF (or the
existing `already_done()` skip-check in main.py/utils.py) is completely
unaffected.

Reuses the same atomic-write pattern figure_database.py already
established (write to a `.tmp` file, then `os.replace()` into place),
so a crash mid-write never leaves a half-written, unparseable state
file behind.

STATUS LIFECYCLE:
    pending_review -> approved   (approve())
                    -> rejected  (reject())
    Any correction (see correction_engine.regenerate_question) resets
    status back to pending_review and bumps `version` — a corrected
    exercise must always be looked at again before it can be approved.
    This is also the enforcement point for the project's "a PDF must
    NEVER be published automatically" rule (see publish_chapter()
    below): publishing a chapter checks every exercise's status here.
"""
import json
import os
import shutil
import time
from contextlib import contextmanager

from utils import logger
from utils import exclusive_file_lock as _exclusive_file_lock
from utils import unique_tmp_path

STATE_SUFFIX = ".review.json"

STATUS_PENDING = "pending_review"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"


@contextmanager
def state_lock(pdf_output_path: str):
    """PRODUCTION-AUDIT FIX: approve()/reject() are each a read-modify-
    write (load_state -> mutate -> save_state) against the SAME file,
    and correction_engine.regenerate_question() needs the identical
    guarantee across its own (longer) reload-merge-render-save sequence
    — see that module for why. Two internal reviewers (or a reviewer
    and an automated script) acting on the same exercise at close to
    the same moment — e.g. one approving while another rejects, or two
    corrections to two DIFFERENT questions of the same exercise landing
    close together — can each load the same starting state, and the
    second write to complete silently discards everything the first
    one did (last-writer-wins on the whole file). Reproduced directly
    both ways: two threads racing approve() vs reject() loses a history
    entry outright, and two concurrent corrections to different
    questions of the same exercise silently reverts whichever one saved
    first — see test_review_state.py::TestConcurrentReviewActions and
    test_correction_engine.py::test_concurrent_corrections_to_different_questions_do_not_lose_either_fix.

    Same advisory-lock pattern figure_database.py already established
    for the equivalent multi-worker race on index.json, scoped to one
    lock file per exercise's state file so unrelated exercises are
    never serialized against each other. Delegates to the shared,
    CROSS-PLATFORM utils.exclusive_file_lock() primitive (POSIX
    fcntl.flock / Windows msvcrt.locking) — so concurrent reviewers are
    now actually serialized on Windows too, where this previously
    degraded to NO locking at all and a racing approve/reject could
    silently lose one action (reproduced by TestConcurrentReviewActions).
    """
    lock_path = state_path(pdf_output_path) + ".lock"
    with _exclusive_file_lock(lock_path, "review_state per-exercise lock"):
        yield


def state_path(pdf_output_path: str) -> str:
    """PRODUCTION-AUDIT FIX (final pre-launch round): the sidecar used
    to sit directly beside the PDF (`ex_7_4.pdf` + `ex_7_4.review.json`
    in the exact same folder) — harmless functionally, but it means the
    one folder meant to be a clean, ready-to-hand-out set of exercise
    PDFs also accumulates a review-workflow artifact per file. This
    moves ONLY that sidecar into a same-level `Review/` subfolder
    (`.../chapter_7/ex_7_4.pdf` + `.../chapter_7/Review/ex_7_4.review.json`)
    — the PDF's own path, filename, and every other module that only
    ever wanted the PDF (main.py's already_done() skip-check, the
    publish gate copying straight from `pdf_path`) are completely
    unaffected; only where the ONE JSON sidecar file lives changes.

    Backward compatible with any exercise whose sidecar was already
    written beside the PDF by an earlier run of this pipeline: if that
    legacy file exists and the new-layout one doesn't, this keeps
    returning the LEGACY path for that exercise (read AND write) rather
    than silently orphaning its review history — never an automatic
    migration/rename, which risks losing history if interrupted
    mid-move; a fresh exercise (no sidecar anywhere yet) always gets
    the new Review/ layout.
    """
    directory = os.path.dirname(pdf_output_path)
    base = os.path.splitext(os.path.basename(pdf_output_path))[0]
    new_path = os.path.join(directory, "Review", base + STATE_SUFFIX)
    legacy_path = os.path.splitext(pdf_output_path)[0] + STATE_SUFFIX
    if not os.path.exists(new_path) and os.path.exists(legacy_path):
        return legacy_path
    return new_path


def _pdf_path_from_state_file(state_file_path: str) -> str:
    """Inverse of state_path() — used by list_states() to recover a
    PDF's path from wherever its sidecar happens to live, supporting
    both the current Review/ subfolder layout and the legacy beside-
    the-PDF layout (see state_path()'s docstring) transparently."""
    directory = os.path.dirname(state_file_path)
    basename = os.path.basename(state_file_path)[: -len(STATE_SUFFIX)] + ".pdf"
    if os.path.basename(directory) == "Review":
        return os.path.join(os.path.dirname(directory), basename)
    return os.path.join(directory, basename)  # legacy layout: sidecar beside the PDF


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = unique_tmp_path(path)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # os.replace is atomic on POSIX. On Windows it can raise WinError 5
        # (Access is denied) transiently — usually Defender/AV briefly
        # scanning the freshly-written file, or a lingering read handle —
        # even though nothing is actually wrong. Retry briefly rather than
        # failing the whole request over what's almost always a race, not
        # a real permissions problem.
        last_err = None
        for attempt in range(6):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.15 * (attempt + 1))
        raise last_err
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def init_state(pdf_output_path: str, solved: dict, class_name, chapter, chapter_name: str,
               exercise_label: str) -> dict:
    """Called once, immediately after an exercise's draft PDF is first
    generated (see main.py's process_exercise). Always starts a FRESH
    pending_review state at version 1 — matching utils.already_done()'s
    own existing semantics (deleting the PDF is how a re-generation is
    forced), so any prior review history for a since-deleted PDF is
    intentionally not resurrected rather than silently carried over."""
    state = {
        "exercise_label": exercise_label,
        "class_name": class_name,
        "chapter": chapter,
        "chapter_name": chapter_name,
        "status": STATUS_PENDING,
        "version": 1,
        "solved": solved,
        "history": [
            {"version": 1, "action": "generated", "timestamp": _now()},
        ],
        "reviewer_notes": [],
        # Interactive Review workflow (see interactive_review.py): keys
        # of every question the reviewer has explicitly accepted so far,
        # e.g. "1" or "2(a)" — see _question_key(). Lets main.py's
        # per-question review loop be RESUMED across separate runs
        # instead of forcing a reviewer to redo already-accepted
        # questions if they quit partway through an exercise.
        "accepted_questions": [],
    }
    _atomic_write(state_path(pdf_output_path), state)
    return state


def load_state(pdf_output_path: str):
    """Returns the parsed state dict, or None if no review state exists
    for this PDF (never raised) — callers decide what that means for
    them (e.g. correction_engine treats it as "nothing to correct")."""
    path = state_path(pdf_output_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ review_state: could not read {path} ({e}).")
        return None


def save_state(pdf_output_path: str, state: dict) -> None:
    _atomic_write(state_path(pdf_output_path), state)


def force_rereview(pdf_output_path: str) -> dict:
    """
    Sets the force_rereview flag to trigger a re-review of the exercise.
    
    When the user edits the review file and sets "force_rereview": true,
    the next run of main.py will reset the exercise to pending_review status
    and allow it to go through the interactive review workflow again.
    
    Args:
        pdf_output_path: Path to the exercise PDF
    
    Returns:
        The updated state dict
    """
    with state_lock(pdf_output_path):
        state = load_state(pdf_output_path)
        if state is None:
            raise ValueError(f"No review state found for {pdf_output_path}")
        
        state["force_rereview"] = True
        state.setdefault("history", []).append({
            "version": state.get("version", 1),
            "action": "force_rereview_requested",
            "question_number": None,
            "sub_part": None,
            "correction_instruction": "Manual rereview requested",
            "reviewer": "manual",
            "timestamp": _now()
        })
        save_state(pdf_output_path, state)
        logger.info(f"🔄 Set force_rereview=True for {pdf_output_path}")
    
    return state


def _question_key(question_number, sub_part=None) -> str:
    return f"{question_number}" + (f"({sub_part})" if sub_part else "")


def is_question_accepted(state: dict, question_number, sub_part=None) -> bool:
    return _question_key(question_number, sub_part) in state.get("accepted_questions", [])


def mark_question_accepted(pdf_output_path: str, question_number, sub_part=None, reviewer: str = None) -> dict:
    """Interactive Review workflow (see interactive_review.py): records
    that the reviewer explicitly accepted this ONE question, right now
    — persisted immediately (not just held in memory) so quitting the
    review session partway through and re-running main.py later resumes
    exactly where the reviewer left off instead of re-asking about
    questions already handled."""
    with state_lock(pdf_output_path):
        state = load_state(pdf_output_path)
        if state is None:
            raise ValueError(f"No review state found for {pdf_output_path}.")
        key = _question_key(question_number, sub_part)
        accepted = state.setdefault("accepted_questions", [])
        if key not in accepted:
            accepted.append(key)
            state.setdefault("history", []).append({
                "version": state["version"], "action": "question_accepted",
                "question_number": question_number, "sub_part": sub_part,
                "reviewer": reviewer, "timestamp": _now(),
            })
            save_state(pdf_output_path, state)
    return state


def finalize_correction(state: dict, pdf_output_path: str, question_number, correction_instruction: str,
                         sub_part=None, reviewer: str = None, action: str = "corrected") -> dict:
    """Bumps version, appends a history entry (action='corrected' for a
    text/diagram fix via correction_engine.regenerate_question(), or
    action='figure_added' for correction_engine.attach_manual_figure_to_question()),
    resets the EXERCISE status to pending_review, and — since the
    question just changed — clears it from `accepted_questions` if it
    was already accepted (a corrected question must always be looked at
    again; see interactive_review.py's loop, which re-shows a question
    after any change rather than assuming it's still fine). Persists
    `state` — used by correction_engine, which must have already:
      1. acquired state_lock(pdf_output_path),
      2. reloaded the FRESHEST on-disk state itself,
      3. merged the corrected fields into THAT fresh copy's target
         question (never a stale copy loaded before the lock/AI call),
      4. rendered + atomically written the PDF from that fresh copy,
    all within the same lock acquisition, before calling this. This
    function does not re-load or lock — it trusts the caller already
    did, since it must run inside the SAME critical section as the
    render/write step it needs to stay consistent with (see
    correction_engine.py and state_lock()'s docstring for the exact
    race this prevents).
    """
    state["version"] += 1
    state["status"] = STATUS_PENDING
    key = _question_key(question_number, sub_part)
    accepted = state.setdefault("accepted_questions", [])
    if key in accepted:
        accepted.remove(key)
    state["history"].append({
        "version": state["version"], "action": action,
        "question_number": question_number, "sub_part": sub_part,
        "correction_instruction": correction_instruction,
        "reviewer": reviewer, "timestamp": _now(),
    })
    save_state(pdf_output_path, state)
    return state


def approve(pdf_output_path: str, reviewer: str = None, note: str = None) -> dict:
    with state_lock(pdf_output_path):
        state = load_state(pdf_output_path)
        if state is None:
            raise ValueError(f"No review state found for {pdf_output_path}.")
        state["status"] = STATUS_APPROVED
        state["history"].append({
            "version": state["version"], "action": "approved",
            "reviewer": reviewer, "note": note, "timestamp": _now(),
        })
        if note:
            state["reviewer_notes"].append({"note": note, "reviewer": reviewer, "timestamp": _now()})
        save_state(pdf_output_path, state)
    logger.info(f"✅ Exercise {state.get('exercise_label')} APPROVED (version {state['version']}).")
    return state


def reject(pdf_output_path: str, reviewer: str = None, note: str = None) -> dict:
    with state_lock(pdf_output_path):
        state = load_state(pdf_output_path)
        if state is None:
            raise ValueError(f"No review state found for {pdf_output_path}.")
        state["status"] = STATUS_REJECTED
        state["history"].append({
            "version": state["version"], "action": "rejected",
            "reviewer": reviewer, "note": note, "timestamp": _now(),
        })
        if note:
            state["reviewer_notes"].append({"note": note, "reviewer": reviewer, "timestamp": _now()})
        save_state(pdf_output_path, state)
    logger.warning(f"❌ Exercise {state.get('exercise_label')} REJECTED (version {state['version']}).")
    return state


def list_states(output_root: str = "output") -> list:
    """Walks `output_root` for every *.review.json sidecar and returns a
    lightweight summary of each — used by review_cli.py `list` and by
    the chapter-level publish gate below."""
    results = []
    if not os.path.isdir(output_root):
        return results
    for dirpath, _dirs, files in os.walk(output_root):
        for fname in sorted(files):
            if not fname.endswith(STATE_SUFFIX):
                continue
            path = os.path.join(dirpath, fname)
            pdf_path = _pdf_path_from_state_file(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ review_state: could not read {path} ({e}) — skipping.")
                continue
            results.append({
                "pdf_path": pdf_path,
                "exercise_label": state.get("exercise_label"),
                "class_name": state.get("class_name"),
                "chapter": state.get("chapter"),
                "status": state.get("status"),
                "version": state.get("version"),
            })
    return results


def chapter_publish_status(class_name, chapter, output_root: str = "output") -> dict:
    """Read-only check for the "a PDF must NEVER be published
    automatically" rule: every exercise for this class/chapter must be
    `approved` before it's considered ready. Doesn't move or copy
    anything itself — see publish_chapter() for the action."""
    all_states = [s for s in list_states(output_root)
                  if str(s["class_name"]) == str(class_name) and str(s["chapter"]) == str(chapter)]
    not_approved = [s for s in all_states if s["status"] != STATUS_APPROVED]
    return {
        "class_name": class_name,
        "chapter": chapter,
        "total_exercises": len(all_states),
        "approved": len(all_states) - len(not_approved),
        "ready_to_publish": len(all_states) > 0 and not not_approved,
        "pending": [s["exercise_label"] for s in not_approved],
    }


def publish_chapter(class_name, chapter, output_root: str = "output",
                     published_root: str = "published") -> dict:
    """Copies every APPROVED exercise PDF for this class/chapter into
    `published_root` — the ONLY path by which a PDF ever reaches
    students. Refuses outright (raises ValueError) if even one
    exercise for this chapter isn't approved yet, rather than silently
    publishing a partial/unreviewed chapter."""
    status = chapter_publish_status(class_name, chapter, output_root)
    if not status["ready_to_publish"]:
        raise ValueError(
            f"Chapter {chapter} (Class {class_name}) is NOT ready to publish — "
            f"{len(status['pending'])} exercise(s) still not approved: {status['pending']}"
        )

    all_states = [s for s in list_states(output_root)
                  if str(s["class_name"]) == str(class_name) and str(s["chapter"]) == str(chapter)]
    dest_dir = os.path.join(published_root, f"class_{class_name}", f"chapter_{chapter}")
    os.makedirs(dest_dir, exist_ok=True)

    published = []
    for s in all_states:
        src = s["pdf_path"]
        if not os.path.exists(src):
            logger.warning(f"⚠️ publish_chapter: {src} is approved but missing on disk — skipping.")
            continue
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        published.append(dst)

    logger.info(f"🚀 Published {len(published)} exercise PDF(s) for Class {class_name} "
                f"Chapter {chapter} → {dest_dir}")
    return {"published_files": published, "dest_dir": dest_dir}
