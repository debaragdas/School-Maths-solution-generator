"""
main.py — run the whole pipeline for one chapter:

    python main.py                                    # config.py's CLASS/CHAPTER
    python main.py --class 9 --chapter 7              # whole chapter 7, class 9 book
    python main.py --chapter 7 --exercise 7.4         # only exercise 7.4
    python main.py --chapter 10 --exercise 10.1 10.2  # a few exercises
    python main.py --chapter 10 --construction        # constructions (অংকন) chapter
    python main.py --list-books                       # the class 6-10 book registry

Every flag is OPTIONAL — with no flags this behaves exactly as before,
reading BOOK_URL / CLASS / CHAPTER / IS_CONSTRUCTION_CHAPTER /
SOLVE_SINGLE_EXERCISE / EXERCISE_FILTER from config.py. Any flag given
OVERRIDES config for this run only (config.py itself is never modified).

Two stages:

  STAGE 1 (parallel, unattended): download, detect chapter, split
  exercises, solve each exercise (one Gemini call), verify, render a
  DRAFT PDF, persist review state. An exercise whose review is already
  fully approved is skipped outright; one that's mid-review (or has
  never been reviewed at all) reuses its already-solved content instead
  of wasting another Gemini call — see process_exercise().

  STAGE 2 (sequential, INTERACTIVE — see interactive_review.py): for
  every exercise not yet fully approved, pauses on each question that
  needs review and waits for the reviewer's decision (accept /
  regenerate with a correction / attach a manual textbook figure /
  preview / quit) before moving on. This is intentionally sequential —
  even though Stage 1 solves several exercises in parallel
  (config.PARALLEL_WORKERS), a human can only review one thing at a
  time, and prompting from multiple threads at once would interleave
  garbled terminal output.

A PDF is never considered finished, and a chapter is never published
(review_state.publish_chapter), until every one of its exercises has
been explicitly approved this way — there is no separate offline/CLI
review step in this workflow.
"""
import argparse
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import books
import config
from downloader import download_pdf
from chapter_detector import get_chapter_page_range, get_chapter_title
from exercise_splitter import split_exercises
from solver import solve_exercise
from html_renderer import render_exercise_html
from pdf_generator import render_pdf
from utils import logger, already_done, verify_solution
from vision_ocr import is_text_layer_reliable
import layout_validation
import review_state
import interactive_review
import review_webapp


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="AssamStudyAI Math Solution Factory — solve one chapter "
                    "(or selected exercises) of the SEBA/SCERT Assamese maths "
                    "textbook into reviewed, published solution PDFs. All flags "
                    "are optional; without them config.py's values are used.",
        epilog="Examples:\n"
               "  python main.py --class 9 --chapter 7\n"
               "  python main.py --chapter 7 --exercise 7.4\n"
               "  python main.py --class 10 --chapter 10 --construction\n"
               "  python main.py --list-books",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--class", dest="class_num", type=int, metavar="N",
                        help=f"class number, 6-10 (default: config.CLASS = {config.CLASS})")
    parser.add_argument("--chapter", dest="chapter", type=int, metavar="N",
                        help=f"chapter number (default: config.CHAPTER = {config.CHAPTER})")
    parser.add_argument("--exercise", dest="exercise", nargs="+", metavar="X.Y", default=None,
                        help="solve ONLY these exercise label(s), e.g. --exercise 7.4 or "
                             "--exercise 7.1 7.4 (implies whole-chapter mode off for this run)")
    parser.add_argument("--book-url", dest="book_url", default=None, metavar="URL",
                        help="override the textbook PDF URL directly (wins over the books.py "
                             "registry; cached after first download)")
    construction_group = parser.add_mutually_exclusive_group()
    construction_group.add_argument("--construction", dest="is_construction",
                                        action="store_true", default=None,
                                        help="this is a অংকন/constructions chapter — solutions get "
                                             "step-by-step compass-and-ruler instructions")
    construction_group.add_argument("--no-construction", dest="is_construction",
                                        action="store_false",
                                        help="regular chapter — proofs and solutions, not "
                                             "drawing instructions (config default)")
    parser.add_argument("--review-mode", dest="review_mode", choices=["web", "cli"], default=None,
                        help="Stage 2 review UI: browser app ('web') or console loop ('cli') "
                             f"(default: config.REVIEW_MODE = {getattr(config, 'REVIEW_MODE', 'web')})")
    parser.add_argument("--list-books", dest="list_books", action="store_true",
                        help="print the class 6-10 textbook registry and exit")
    return parser


def apply_cli_overrides(config_module, args):
    """Applies parsed CLI arguments ON TOP OF the config module values
    for THIS RUN ONLY — config.py itself is never written to. Returns
    the effective (class_num, chapter) tuple so main() doesn't have to
    re-read attributes. Only overrides what was actually passed: an
    unflagged run is byte-for-byte behaviorally identical to before."""
    if getattr(args, "list_books", False):
        return None

    if args.class_num is not None:
        config_module.CLASS = args.class_num
    if args.chapter is not None:
        config_module.CHAPTER = args.chapter
    if args.book_url is not None:
        config_module.BOOK_URL = args.book_url
    else:
        # No explicit URL this run — resolve it from the books.py registry
        # by class. If that class has no usable single-file entry (SCERT
        # 6-8), keep config.BOOK_URL and say so loudly rather than failing
        # deep inside the download step with an opaque error.
        try:
            config_module.BOOK_URL = books.book_url_for(config_module.CLASS)
        except books.BookNotRegisteredException as e:
            logger.warning(f"⚠️ {e} Falling back to config.BOOK_URL = "
                            f"{config_module.BOOK_URL}")
    if args.is_construction is not None:
        config_module.IS_CONSTRUCTION_CHAPTER = args.is_construction
    if args.review_mode is not None:
        config_module.REVIEW_MODE = args.review_mode
    if args.exercise:
        config_module.SOLVE_SINGLE_EXERCISE = True
        config_module.EXERCISE_FILTER = list(args.exercise)
    return int(config_module.CLASS), int(config_module.CHAPTER)


def _effective_exercise_filter(config_module):
    """config.SOLVE_SINGLE_EXERCISE is the explicit on/off switch;
    config.EXERCISE_FILTER is only ever consulted when that switch is
    True. Kept as its own tiny function (rather than inlined in main())
    purely so this exact on/off decision has a unit test independent of
    running the whole pipeline."""
    if getattr(config_module, "SOLVE_SINGLE_EXERCISE", False):
        return getattr(config_module, "EXERCISE_FILTER", None)
    return None


def _apply_exercise_filter(exercises: list, exercise_filter, chapter) -> list:
    """Narrows the full list of exercises split_exercises() found down
    to just config.EXERCISE_FILTER, when set — this is what lets a run
    solve one exercise instead of the whole chapter. None/empty means
    "no filter" and returns `exercises` completely unchanged (the
    default, existing behavior). Warns (doesn't raise) about any
    requested label that isn't actually in `exercises`, so a typo
    doesn't silently solve nothing without explanation."""
    if not exercise_filter:
        return exercises
    wanted = {str(label) for label in exercise_filter}
    found = {ex["label"] for ex in exercises if str(ex["label"]) in wanted}
    missing = wanted - found
    if missing:
        logger.warning(f"⚠️ config.EXERCISE_FILTER named {sorted(missing)}, but Chapter "
                        f"{chapter} doesn't contain (or split_exercises couldn't find) "
                        f"{'an exercise' if len(missing) == 1 else 'exercises'} with that label. "
                        f"Available: {sorted(ex['label'] for ex in exercises)}")
    filtered = [ex for ex in exercises if str(ex["label"]) in wanted]
    logger.info(f"🎯 config.EXERCISE_FILTER set — solving only {sorted(found)} this run "
                f"({len(filtered)} of the chapter's exercises).")
    return filtered


def process_exercise(exercise: dict, class_name: int, chapter: int, chapter_name: str) -> str:
    label = exercise["label"]
    output_path = os.path.join(config.OUTPUT_DIR, f"class_{class_name}", f"chapter_{chapter}",
                                f"ex_{label.replace('.', '_')}.pdf")

    # An exercise counts as "done" now only once a human has approved
    # EVERY one of its questions (see interactive_review.py) — not
    # merely once a draft PDF exists on disk. If a draft already exists
    # but review hasn't finished, don't re-solve it (that would waste a
    # Gemini call and discard any corrections/figures/acceptances a
    # reviewer already recorded) — Stage 2 in main() picks it up as-is.
    existing_state = review_state.load_state(output_path)
    if existing_state is not None:
        # Check if force_rereview flag is set in review file
        if existing_state.get("force_rereview", False):
            logger.info(f"🔄 Exercise {label} has force_rereview=True — resetting to pending review.")
            with review_state.state_lock(output_path):
                existing_state["status"] = review_state.STATUS_PENDING
                existing_state["force_rereview"] = False  # Reset the flag
                existing_state["history"].append({
                    "version": existing_state.get("version", 1),
                    "action": "force_rereview",
                    "question_number": None,
                    "sub_part": None,
                    "correction_instruction": "Manual rereview requested via review file edit",
                    "reviewer": "manual",
                    "timestamp": review_state._now() if hasattr(review_state, '_now') else "unknown"
                })
                review_state.save_state(output_path, existing_state)
            # Continue to solving/reviewing
        elif existing_state.get("status") == review_state.STATUS_APPROVED:
            logger.info(f"⏭️ Exercise {label} already reviewed and approved — skipping.")
            return f"skipped: {label}"
        else:
            logger.info(f"⏭️ Exercise {label} already has a draft awaiting review — "
                        f"leaving it for the interactive review stage instead of re-solving.")
            return f"awaiting_review: {label}"
    if already_done(output_path):
        # A draft PDF exists but has no review_state sidecar at all (e.g.
        # generated before this feature existed) — there's no persisted
        # `solved` JSON to review against, so it genuinely can't enter
        # interactive review as-is. Rather than silently treating it as
        # forever-approved (the old behavior) OR promising a review that
        # can't actually happen, say so plainly: delete the PDF to force
        # a fresh solve + draft, which WILL then be reviewable.
        logger.warning(f"⚠️ Exercise {label}: a PDF exists but has no review state (predates this "
                        f"feature) — it cannot go through interactive review as-is. Delete "
                        f"{output_path} and re-run to regenerate it with a reviewable draft.")
        return f"unreviewable_legacy_pdf: {label}"

    attempts_allowed = 1 + config.MAX_RETRIES_PER_EXERCISE
    last_issues = []

    for attempt in range(1, attempts_allowed + 1):
        try:
            solved = solve_exercise(exercise["path"], class_name, chapter, label, 
                                   chapter_title=chapter_name, 
                                   is_construction_chapter=config.IS_CONSTRUCTION_CHAPTER)
            ok, issues = verify_solution(solved)
            if not ok:
                last_issues = issues
                logger.warning(f"⚠️ Exercise {label} failed verification (attempt {attempt}): {issues}")
                if attempt < attempts_allowed:
                    continue
                else:
                    return f"failed: {label} — {issues}"

            html = render_exercise_html(solved, class_name, chapter, chapter_name, label)
            render_pdf(html, output_path)
            try:
                layout_validation.validate_and_log(output_path, exercise_label=label)
            except Exception as e:
                # Best-effort audit only — must never fail an otherwise
                # successful exercise (see layout_validation.py's own docstring).
                logger.warning(f"⚠️ Exercise {label}: layout validation audit itself failed ({e}) — "
                                f"the PDF was still generated and saved normally.")
            try:
                # Human Review workflow (see review_state.py / review_cli.py):
                # persists the solved JSON + a fresh pending_review state next
                # to the PDF. Best-effort only — a failure here must never
                # block an otherwise successful exercise; it just means this
                # one exercise won't be reviewable/correctable until it's
                # regenerated (delete the PDF and re-run to force that).
                review_state.init_state(output_path, solved, class_name, chapter, chapter_name, label)
            except Exception as e:
                logger.warning(f"⚠️ Exercise {label}: could not persist review state ({e}) — "
                                f"the PDF was still generated and saved normally, but it won't be "
                                f"reviewable via review_cli.py until this is fixed.")
            logger.info(f"✅ Exercise {label} → {output_path}")
            return f"done: {label}"

        except Exception as e:
            logger.error(f"❌ Exercise {label} attempt {attempt} raised: {e}\n{traceback.format_exc()}")
            if attempt >= attempts_allowed:
                return f"error: {label} — {e}"

    return f"failed: {label} — {last_issues}"


def _force_utf8_console():
    """Windows consoles default to a legacy codepage (cp1252/cp437) that
    cannot encode a single character of Assamese — without this, even
    --help and --list-books crash with UnicodeEncodeError on the exact
    platform this project runs on. errors='replace' keeps the run alive
    even if some other layer emits something unencodable."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main(argv=None):
    _force_utf8_console()
    args = build_arg_parser().parse_args(argv)

    if args.list_books:
        print("Registered SEBA/SCERT Assamese General Mathematics textbooks:")
        print(books.list_books())
        return

    effective = apply_cli_overrides(config, args)
    if effective is None:
        return
    class_num, chapter = effective

    logger.info("==================================================")
    logger.info(f"🚀 AssamStudyAI Math Solution Factory")
    logger.info(f"📚 Class {class_num} | Chapter {chapter}")
    logger.info(f"🔗 Book: {config.BOOK_URL}")
    logger.info(f"🩺 Construction-chapter mode: {config.IS_CONSTRUCTION_CHAPTER}")
    logger.info("==================================================")

    book_path = download_pdf(config.BOOK_URL, config.TEMP_DIR)

    # Decide ONCE per book whether the PDF's text layer can be trusted.
    # Some SEBA books embed Assamese glyphs with a custom font encoding
    # that renders correctly on screen but returns garbled Unicode from
    # get_text() — in that case both detection stages skip straight to
    # the Gemini Vision fallback instead of failing or asking for manual
    # page numbers.
    text_layer_reliable = is_text_layer_reliable(book_path)

    start_page, end_page = get_chapter_page_range(
        book_path, config.CHAPTER, class_name=config.CLASS,
        text_layer_reliable=text_layer_reliable,
    )
    chapter_name = get_chapter_title(
        book_path, config.CHAPTER, start_page, class_name=config.CLASS,
        text_layer_reliable=text_layer_reliable,
    )

    exercises = split_exercises(
        book_path, start_page, end_page, config.TEMP_DIR,
        class_name=config.CLASS, chapter=config.CHAPTER,
        text_layer_reliable=text_layer_reliable,
    )
    exercises = _apply_exercise_filter(
        exercises, _effective_exercise_filter(config), config.CHAPTER,
    )

    results = []
    results_by_label = {}
    with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as pool:
        futures = {
            pool.submit(process_exercise, ex, config.CLASS, config.CHAPTER, chapter_name): ex["label"]
            for ex in exercises
        }
        for future in as_completed(futures):
            label = futures[future]
            result = future.result()
            results.append(result)
            results_by_label[label] = result

    logger.info("==================================================")
    logger.info("📊 STAGE 1 SUMMARY (generation)")
    for r in sorted(results):
        logger.info(f"   {r}")
    logger.info("==================================================")

    # ------------------------------------------------------------------
    # STAGE 2 — interactive human review, ONE exercise at a time, in this
    # (the main) thread. Deliberately sequential and separate from Stage
    # 1's parallel solving — see main.py's module docstring for why.
    # ------------------------------------------------------------------
    # PRODUCTION-AUDIT NOTE: matched by exact label via results_by_label
    # (built from the futures dict above), NOT by re-parsing the "done: X"
    # log strings with str.startswith() — exercise labels like "7.1" and
    # "7.10" would otherwise falsely prefix-match each other.
    reviewable_labels = sorted(
        label for label, result in results_by_label.items()
        if result.startswith("done:") or result.startswith("awaiting_review:")
    )
    review_mode = getattr(config, "REVIEW_MODE", "web")

    if reviewable_labels and review_mode == "web":
        # Human Review workflow (see review_webapp.py): one local browser
        # app for the WHOLE chapter, instead of interactive_review.py's
        # one-exercise-at-a-time console loop — same underlying
        # review_state.py / correction_engine.py functions either way.
        logger.info("==================================================")
        logger.info(f"📝 STAGE 2 — Human Review app ({len(reviewable_labels)} exercise(s) to review)")
        logger.info("==================================================")
        try:
            completed = review_webapp.run_review_server(config.CLASS, config.CHAPTER, config.OUTPUT_DIR)
        except Exception as e:
            logger.error(f"❌ Human Review app failed: {e}\n{traceback.format_exc()}")
            logger.info("   Falling back to the console review workflow for this run.")
            completed = _run_console_review(reviewable_labels)
        if not completed:
            logger.info("⏸️ Stopping here — re-run main.py to resume review.")

    elif reviewable_labels:
        logger.info("==================================================")
        logger.info(f"📝 STAGE 2 — interactive review ({len(reviewable_labels)} exercise(s))")
        logger.info("==================================================")
        _run_console_review(reviewable_labels)

    # ------------------------------------------------------------------
    # PUBLISH GATE — never automatic unless EVERY exercise in the chapter
    # (not just this run's) is approved (see review_state.publish_chapter).
    # ------------------------------------------------------------------
    try:
        publish_result = review_state.publish_chapter(config.CLASS, config.CHAPTER, config.OUTPUT_DIR)
        logger.info("==================================================")
        logger.info(f"🚀 Chapter {config.CHAPTER} fully approved — published "
                    f"{len(publish_result['published_files'])} PDF(s) to {publish_result['dest_dir']}")
        logger.info("==================================================")
    except ValueError as e:
        logger.info("==================================================")
        logger.info(f"⏸️ Not publishing yet: {e}")
        logger.info("   Re-run main.py to continue reviewing — publishing happens "
                    "automatically once every exercise is approved.")
        logger.info("==================================================")


def _run_console_review(reviewable_labels) -> bool:
    """The original Stage 2 workflow (config.REVIEW_MODE == 'cli', or a
    fallback if the browser review app can't start) — one exercise at a
    time via interactive_review.py. Returns True only if every
    reviewable exercise this run was fully reviewed without the
    reviewer quitting partway through."""
    for label in reviewable_labels:
        output_path = os.path.join(config.OUTPUT_DIR, f"class_{config.CLASS}", f"chapter_{config.CHAPTER}",
                                    f"ex_{label.replace('.', '_')}.pdf")
        try:
            completed = interactive_review.review_exercise_interactively(output_path)
        except Exception as e:
            logger.error(f"❌ Exercise {label}: interactive review raised {e}\n{traceback.format_exc()}")
            completed = False
        if not completed:
            logger.info(f"⏸️ Stopping review here — re-run main.py to resume with Exercise {label} "
                        f"(and any exercise after it in this chapter).")
            return False
    return True


if __name__ == "__main__":
    main()
