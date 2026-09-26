"""
review_cli.py — INTERNAL PRODUCTION TEAM TOOL ONLY. Students never see
this; it is never part of the generated PDF or website. This is the
one entry point the internal team uses for both Human Review
capabilities, before a chapter is published:

  1. Wrong AI Output Correction — `correct`
  2. Missing Book Figure         — `add-figure`

...plus the plumbing every review workflow needs: `list` / `show` to
see what's pending, `approve` / `reject` to record a decision, and
`publish-check` / `publish` as the "never publish automatically unless
everything is approved" gate.

Run `python review_cli.py <command> --help` for a command's full options.
"""
import argparse
import json
import sys

import review_state


def _cmd_list(args):
    states = review_state.list_states(args.output_root)
    if args.status:
        states = [s for s in states if s["status"] == args.status]
    if args.chapter is not None:
        states = [s for s in states if str(s["chapter"]) == str(args.chapter)]
    if not states:
        print("No exercises found.")
        return
    for s in sorted(states, key=lambda s: (str(s["class_name"]), str(s["chapter"]), str(s["exercise_label"]))):
        print(f"[{s['status']:15}] class {s['class_name']} ch.{s['chapter']} "
              f"ex {s['exercise_label']:>8}  v{s['version']}  {s['pdf_path']}")


def _cmd_show(args):
    state = review_state.load_state(args.pdf_path)
    if state is None:
        print(f"No review state found for {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Exercise {state['exercise_label']}  (Class {state['class_name']}, "
          f"Chapter {state['chapter']}) — status={state['status']}, version={state['version']}")
    print("-" * 70)
    for q in state["solved"].get("questions", []):
        label = f"Q{q.get('question_number')}" + (f"({q['sub_part']})" if q.get("sub_part") else "")
        diagram = q.get("diagram_decision", "NO_DIAGRAM")
        needs_review = " ⚠️ NEEDS_REVIEW" if q.get("needs_review") else ""
        print(f"{label}: {q.get('final_answer', '')[:80]}  [diagram={diagram}]{needs_review}")
    if state["history"]:
        print("-" * 70)
        print("History:")
        for h in state["history"]:
            print(f"  v{h['version']} — {h['action']} @ {h['timestamp']}"
                  + (f" — {h.get('correction_instruction')}" if h.get("correction_instruction") else "")
                  + (f" ({h.get('note')})" if h.get("note") else ""))


def _cmd_correct(args):
    import correction_engine
    new_state = correction_engine.regenerate_question(
        args.pdf_path, question_number=args.question, sub_part=args.sub_part,
        correction_instruction=args.instruction, reviewer=args.reviewer,
    )
    print(f"Corrected. Exercise {new_state['exercise_label']} is now version "
          f"{new_state['version']}, status={new_state['status']}. Re-review the PDF at "
          f"{args.pdf_path} before approving.")


def _cmd_approve(args):
    state = review_state.approve(args.pdf_path, reviewer=args.reviewer, note=args.note)
    print(f"Approved. Exercise {state['exercise_label']} version {state['version']}.")


def _cmd_reject(args):
    state = review_state.reject(args.pdf_path, reviewer=args.reviewer, note=args.note)
    print(f"Rejected. Exercise {state['exercise_label']} version {state['version']}. "
          f"Use `correct` to fix a specific question, then re-review.")


def _cmd_add_figure(args):
    import figure_database
    with open(args.image_path, "rb") as f:
        image_bytes = f.read()

    if args.bbox:
        try:
            from PIL import Image
            import io
            x1, y1, x2, y2 = (int(v) for v in args.bbox.split(","))
            img = Image.open(io.BytesIO(image_bytes))
            cropped = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            fmt = (img.format or "PNG")
            cropped.save(buf, format=fmt)
            image_bytes = buf.getvalue()
        except Exception as e:
            print(f"Could not crop image with bbox '{args.bbox}': {e}", file=sys.stderr)
            sys.exit(1)

    ext = figure_database.detect_image_ext(image_bytes, fallback=args.image_path.rsplit(".", 1)[-1].lower()
                                            if "." in args.image_path else "png")
    entry = figure_database.add_manual_figure(
        image_bytes=image_bytes,
        figure_number=args.figure,
        class_name=args.klass,
        chapter=args.chapter,
        subject=args.subject,
        page_number=args.page,
        reviewer=args.reviewer,
        book_id=args.book_id,
        book_url=args.book_url,
        ext=ext,
        bbox=[int(v) for v in args.bbox.split(",")] if args.bbox else None,
        notes=args.notes,
    )
    print(f"Stored figure {entry['figure_ref']} (version {entry['version']}) for future reuse:")
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def _cmd_publish_check(args):
    status = review_state.chapter_publish_status(args.klass, args.chapter, args.output_root)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not status["ready_to_publish"]:
        sys.exit(1)


def _cmd_publish(args):
    result = review_state.publish_chapter(args.klass, args.chapter, args.output_root, args.published_root)
    print(f"Published {len(result['published_files'])} PDF(s) to {result['dest_dir']}:")
    for path in result["published_files"]:
        print(f"  {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal Human Review tool for AssamStudyAI — never exposed to students.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List all exercises and their review status.")
    p_list.add_argument("--output-root", default="output")
    p_list.add_argument("--status", choices=[review_state.STATUS_PENDING, review_state.STATUS_APPROVED,
                                              review_state.STATUS_REJECTED])
    p_list.add_argument("--chapter")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show one exercise's questions, diagrams, and history.")
    p_show.add_argument("pdf_path")
    p_show.set_defaults(func=_cmd_show)

    p_correct = sub.add_parser(
        "correct", help="Apply a correction prompt to ONE question; regenerates only that question.")
    p_correct.add_argument("pdf_path")
    p_correct.add_argument("question", help="question_number, e.g. 5")
    p_correct.add_argument("instruction", help="Free-text correction, e.g. "
                            "'Step 3 calculation is incorrect. Recalculate from Step 2 only. "
                            "Keep everything else unchanged.'")
    p_correct.add_argument("--sub-part", default=None)
    p_correct.add_argument("--reviewer", default=None)
    p_correct.set_defaults(func=_cmd_correct)

    p_approve = sub.add_parser("approve", help="Mark an exercise as approved.")
    p_approve.add_argument("pdf_path")
    p_approve.add_argument("--reviewer", default=None)
    p_approve.add_argument("--note", default=None)
    p_approve.set_defaults(func=_cmd_approve)

    p_reject = sub.add_parser("reject", help="Mark an exercise as rejected (needs a correction).")
    p_reject.add_argument("pdf_path")
    p_reject.add_argument("--reviewer", default=None)
    p_reject.add_argument("--note", default=None)
    p_reject.set_defaults(func=_cmd_reject)

    p_fig = sub.add_parser(
        "add-figure",
        help="Upload/crop a manually-sourced textbook figure into the permanent Figure Database.")
    p_fig.add_argument("image_path", help="Path to the screenshot (full page or already-cropped).")
    p_fig.add_argument("--figure", required=True, help="Figure number as printed in the book, e.g. 4.12")
    p_fig.add_argument("--class", dest="klass", required=True, type=int)
    p_fig.add_argument("--chapter", required=True, type=int)
    p_fig.add_argument("--subject", default=None)
    p_fig.add_argument("--page", type=int, default=None)
    p_fig.add_argument("--reviewer", default=None)
    p_fig.add_argument("--book-id", default=None, help="Overrides the auto-derived book id.")
    p_fig.add_argument("--book-url", default=None, help="config.BOOK_URL is used if omitted.")
    p_fig.add_argument("--bbox", default=None, help="x1,y1,x2,y2 — crop this box out of image_path "
                        "before storing; omit if image_path is already the cropped figure.")
    p_fig.add_argument("--notes", default=None)
    p_fig.set_defaults(func=_cmd_add_figure)

    p_pubcheck = sub.add_parser("publish-check", help="Check whether a chapter is ready to publish.")
    p_pubcheck.add_argument("--class", dest="klass", required=True, type=int)
    p_pubcheck.add_argument("--chapter", required=True, type=int)
    p_pubcheck.add_argument("--output-root", default="output")
    p_pubcheck.set_defaults(func=_cmd_publish_check)

    p_publish = sub.add_parser(
        "publish", help="Copy every approved exercise PDF for a chapter into published/ (fails if any "
                         "exercise isn't approved).")
    p_publish.add_argument("--class", dest="klass", required=True, type=int)
    p_publish.add_argument("--chapter", required=True, type=int)
    p_publish.add_argument("--output-root", default="output")
    p_publish.add_argument("--published-root", default="published")
    p_publish.set_defaults(func=_cmd_publish)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
