"""
resync_all_pdfs.py — re-renders EVERY already-solved exercise's PDF
from its already-stored `solved` JSON through the CURRENT rendering
pipeline (html_renderer.py -> math_sanitizer.py -> pdf_generator.py).

WHY THIS EXISTS (the actual root cause behind "I keep fixing the LaTeX
sanitizer and the PDF is still broken"):

main.py deliberately never re-solves an exercise that already has a
review-state sidecar (".review.json") next to its PDF — re-solving
means another Gemini call, and would silently discard any corrections
a reviewer already made. That's the right call for avoiding wasted API
calls. But it has a side effect nobody asked for: review_state.approve()
only flips a status flag, it never re-renders anything — so an
exercise's PDF is permanently frozen at whatever the rendering code
looked like the moment it was first drafted. Every fix made to
math_sanitizer.py / html_renderer.py AFTER that moment never touches
that PDF again, no matter how many times the code is corrected —
because main.py, correctly, never even looks at it again once a
review-state file exists.

This script closes that gap WITHOUT calling Gemini at all: the fully
solved question data (question_text, given, steps, final_answer, every
field) is already sitting in the "solved" key of the .review.json
sidecar. Re-rendering it is pure local computation — the exact same
render_exercise_html() + render_pdf() call main.py itself makes, just
using whatever CURRENT html_renderer.py/math_sanitizer.py you have on
disk right now instead of whatever was live back when the exercise was
first solved.

USAGE:
    python resync_all_pdfs.py                  # resync everything under config.OUTPUT_DIR
    python resync_all_pdfs.py --dry-run         # report what WOULD change, touch nothing
    python resync_all_pdfs.py --path output/class_9/chapter_1/ex_1_5.pdf   # just one exercise

Run this any time you update the rendering/sanitizer code and want
every exercise you've EVER generated — approved, mid-review, or
otherwise — to reflect the fix immediately, at zero Gemini cost.
"""
import argparse
import os
import sys

import config
import review_state
import math_sanitizer
from html_renderer import render_exercise_html
from pdf_generator import render_pdf
from utils import logger


def _text_fields_would_change(solved: dict) -> bool:
    """Cheap pre-check: would re-sanitizing any text field in `solved`
    actually produce different output than what's currently stored?
    Used by --dry-run to report "stale" exercises without spending the
    time to fully re-render every single one."""
    for q in solved.get("questions", []):
        for field in ("question_text", "given", "required", "final_answer"):
            val = q.get(field)
            if isinstance(val, str) and math_sanitizer.sanitize_math_text(val) != val:
                return True
        for step in q.get("steps", []) or []:
            if isinstance(step, str) and math_sanitizer.sanitize_math_text(step) != step:
                return True
    return False


def resync_one(pdf_output_path: str, dry_run: bool = False) -> str:
    state = review_state.load_state(pdf_output_path)
    if state is None:
        return f"no review state — skipped: {pdf_output_path}"

    solved = state["solved"]
    if dry_run:
        stale = _text_fields_would_change(solved)
        return f"{'STALE (would change)' if stale else 'already current'}: {pdf_output_path}"

    html = render_exercise_html(
        solved, state["class_name"], state["chapter"], state["chapter_name"],
        state["exercise_label"],
    )
    render_pdf(html, pdf_output_path)
    logger.info(f"🔄 Resynced (no Gemini call) → {pdf_output_path}")
    return f"resynced: {pdf_output_path}"


def _walk_all_pdfs(output_root: str):
    for dirpath, _dirs, files in os.walk(output_root):
        for fname in sorted(files):
            if fname.endswith(".pdf"):
                yield os.path.join(dirpath, fname)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", help="Resync just one exercise PDF instead of the whole output tree.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Report which exercises would change without touching any files.")
    args = parser.parse_args()

    targets = [args.path] if args.path else list(_walk_all_pdfs(config.OUTPUT_DIR))
    if not targets:
        logger.warning(f"⚠️ No PDFs found under {config.OUTPUT_DIR!r} — nothing to resync.")
        return

    logger.info(f"🔄 Resyncing {len(targets)} exercise PDF(s) "
                f"{'(dry run — no files will be changed)' if args.dry_run else ''} ...")

    results = [resync_one(p, dry_run=args.dry_run) for p in targets]
    for r in results:
        print(r)

    if args.dry_run:
        stale_count = sum(1 for r in results if r.startswith("STALE"))
        logger.info(f"🔍 Dry run complete: {stale_count} of {len(results)} exercise(s) "
                    f"would actually change if resynced for real.")
    else:
        done = sum(1 for r in results if r.startswith("resynced"))
        logger.info(f"✅ Resync complete: {done} of {len(results)} exercise(s) re-rendered "
                    f"with the current code, at zero Gemini cost.")


if __name__ == "__main__":
    sys.exit(main())
