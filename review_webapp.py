"""
review_webapp.py — the REAL local Human Review application.

review_cli.py (terminal commands) and interactive_review.py (a
blocking console loop) already implement every Human Review
CAPABILITY this module needs — approve, per-question "regenerate with
a correction" (correction_engine.regenerate_question), and "attach a
manually cropped textbook figure" (correction_engine.attach_manual_
figure_to_question), all against the same review_state.py sidecar
files. This module adds NO new capability and NO new storage; it is
purely a local browser UI wired onto those same, unchanged functions —
exactly how interactive_review.py is "purely the console UI wired on
top of them" (see that module's own docstring).

Why this exists alongside interactive_review.py rather than replacing
it: a terminal loop that blocks on input() for text/diagram/figure
previews is real but slow to actually look at (opening a PDF viewer
per question, per the [P]review action). A local Flask app in the
reviewer's own browser can show the generated solution, the rendered
diagram SVG, and the book figure image side-by-side on one page,
support pasting a screenshot directly (Ctrl+V) for the "Missing Book
Figure" capability, and jump between questions without re-opening a
PDF each time — which is what actually gets a chapter's review down
to the project's 10-15 minute target. Both UIs are kept: config.
REVIEW_MODE picks which one main.py's Stage 2 runs; nothing about
solving, verification, diagram generation, layout validation, or the
publish gate changes either way.

NOT a public web server: binds to 127.0.0.1 by default (config.
REVIEW_HOST) and is only ever started by main.py itself, once, for
the local reviewer running the pipeline — there is no user-facing
hosting story here, no auth, and it is expected to run entirely
offline aside from whatever main.py's OWN earlier stages already
needed the network for (downloading the book, calling Gemini).
"""
import base64
import os
import threading
import time
import webbrowser

from flask import Flask, jsonify, request, Response

import config
import correction_engine
import review_state
from diagram_renderer import render_diagram
from utils import logger

try:
    from werkzeug.serving import make_server
except ImportError:  # pragma: no cover - werkzeug ships with Flask, defensive only
    make_server = None


# ----------------------------------------------------------------------
# Question-state helpers — shared read-only view the API/HTML layer
# renders from. Never mutates anything; every mutation goes through
# review_state.py / correction_engine.py exactly like the CLI does.
# ----------------------------------------------------------------------
def _question_key(q: dict) -> str:
    qn, sp = q.get("question_number"), q.get("sub_part")
    return f"{qn}({sp})" if sp else f"{qn}"


def _question_summary(state: dict, q: dict) -> dict:
    key = _question_key(q)
    return {
        "key": key,
        "question_number": q.get("question_number"),
        "sub_part": q.get("sub_part"),
        "question_text": q.get("question_text") or "",
        "given": q.get("given") or "",
        "required": q.get("required") or "",
        "steps": q.get("steps") or [],
        "final_answer": q.get("final_answer") or "",
        "diagram_decision": q.get("diagram_decision") or "NO_DIAGRAM",
        "diagram_decision_reason": q.get("diagram_decision_reason") or "",
        "diagram_safety_net_flagged": bool(q.get("diagram_safety_net_flagged")),
        "has_generated_diagram": bool(q.get("diagram_spec")),
        "has_book_figure": bool(q.get("book_diagram_base64")),
        "diagram_match_verified": bool(q.get("diagram_match_verified")),
        "diagram_match_low_confidence": bool(q.get("diagram_match_low_confidence")),
        "diagram_position": q.get("diagram_position") or "auto",
        "book_diagram_figure_ref": q.get("book_diagram_figure_ref"),
        "needs_review": bool(q.get("needs_review")),
        "review_notes": q.get("review_notes") or [],
        "accepted": review_state.is_question_accepted(state, q.get("question_number"), q.get("sub_part")),
    }


def _exercise_payload(pdf_output_path: str) -> dict:
    state = review_state.load_state(pdf_output_path)
    if state is None:
        return None
    questions = state["solved"].get("questions", [])
    summaries = [_question_summary(state, q) for q in questions]
    return {
        "pdf_path": pdf_output_path,
        "exercise_label": state.get("exercise_label"),
        "class_name": state.get("class_name"),
        "chapter": state.get("chapter"),
        "status": state.get("status"),
        "version": state.get("version"),
        "total_questions": len(summaries),
        "accepted_count": sum(1 for s in summaries if s["accepted"]),
        "questions": summaries,
    }


def _parse_page_number(raw):
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _pdf_path_for(class_name, chapter, exercise_label: str, output_dir: str) -> str:
    return os.path.join(output_dir, f"class_{class_name}", f"chapter_{chapter}",
                         f"ex_{exercise_label.replace('.', '_')}.pdf")


# ----------------------------------------------------------------------
# Flask app factory
# ----------------------------------------------------------------------
def create_app(class_name, chapter, output_dir: str = None) -> Flask:
    output_dir = output_dir or config.OUTPUT_DIR
    app = Flask(__name__)
    app.config["CLASS_NAME"] = class_name
    app.config["CHAPTER"] = chapter
    app.config["OUTPUT_DIR"] = output_dir
    app.config["STOP_REQUESTED"] = False

    def _exercise_labels():
        states = review_state.list_states(app.config["OUTPUT_DIR"])
        return sorted(
            (s["exercise_label"] for s in states
             if str(s["class_name"]) == str(app.config["CLASS_NAME"])
             and str(s["chapter"]) == str(app.config["CHAPTER"])),
            key=lambda label: [(0, int(p)) if p.isdigit() else (1, p) for p in str(label).split(".")],
        )

    def _path_for(label: str) -> str:
        return _pdf_path_for(app.config["CLASS_NAME"], app.config["CHAPTER"], label, app.config["OUTPUT_DIR"])

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.route("/")
    def dashboard():
        return Response(_DASHBOARD_HTML, mimetype="text/html")

    @app.route("/exercise/<label>")
    def exercise_page(label):
        return Response(_EXERCISE_HTML, mimetype="text/html")

    # ------------------------------------------------------------------
    # Read-only JSON API
    # ------------------------------------------------------------------
    @app.route("/api/exercises")
    def api_exercises():
        out = []
        for label in _exercise_labels():
            payload = _exercise_payload(_path_for(label))
            if payload:
                out.append({
                    "exercise_label": payload["exercise_label"],
                    "status": payload["status"],
                    "version": payload["version"],
                    "total_questions": payload["total_questions"],
                    "accepted_count": payload["accepted_count"],
                })
        chapter_status = review_state.chapter_publish_status(
            app.config["CLASS_NAME"], app.config["CHAPTER"], app.config["OUTPUT_DIR"])
        return jsonify({
            "class_name": app.config["CLASS_NAME"],
            "chapter": app.config["CHAPTER"],
            "exercises": out,
            "chapter_status": chapter_status,
        })

    @app.route("/api/exercise/<label>")
    def api_exercise(label):
        payload = _exercise_payload(_path_for(label))
        if payload is None:
            return jsonify({"error": f"No review state for exercise {label}."}), 404
        return jsonify(payload)

    @app.route("/api/exercise/<label>/diagram/<key>")
    def api_diagram(label, key):
        state = review_state.load_state(_path_for(label))
        if state is None:
            return Response("", status=404)
        q = _find_by_key(state, key)
        if q is None or not q.get("diagram_spec"):
            return Response("", status=404)
        try:
            svg = render_diagram(q["diagram_spec"])
        except Exception as e:
            logger.warning(f"⚠️ review_webapp: could not render diagram preview for {label}/{key}: {e}")
            return Response("", status=500)
        return Response(svg or "", mimetype="image/svg+xml")

    @app.route("/api/exercise/<label>/figure/<key>")
    def api_figure(label, key):
        state = review_state.load_state(_path_for(label))
        if state is None:
            return Response("", status=404)
        q = _find_by_key(state, key)
        b64 = q.get("book_diagram_base64") if q else None
        if not b64:
            return Response("", status=404)
        mime = q.get("book_diagram_mime") or "image/png"
        return Response(base64.b64decode(b64), mimetype=mime)

    # ------------------------------------------------------------------
    # Mutating actions — every one of these delegates to the SAME
    # review_state.py / correction_engine.py functions the CLI/console
    # workflow already uses; this module never touches a review.json or
    # a PDF directly.
    # ------------------------------------------------------------------
    @app.route("/api/exercise/<label>/question/<key>/accept", methods=["POST"])
    def api_accept(label, key):
        qn, sp = _split_key(key)
        reviewer = (request.json or {}).get("reviewer") if request.is_json else None
        try:
            review_state.mark_question_accepted(_path_for(label), qn, sp, reviewer=reviewer)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/question/<key>/regenerate", methods=["POST"])
    def api_regenerate(label, key):
        qn, sp = _split_key(key)
        body = request.json or {}
        instruction = (body.get("instruction") or "").strip()
        component = body.get("component")  # "solution" | "diagram" | None ("custom prompt")

        if component == "diagram":
            # Capability 1 (rebuilt): NEVER edits the previous diagram —
            # always rebuilds from the question itself. Also doubles as
            # "Generate Diagram" when the question has none yet at all
            # (capability 2) — same call either way, see correction_
            # engine.regenerate_diagram_from_scratch's own docstring.
            try:
                correction_engine.regenerate_diagram_from_scratch(
                    _path_for(label), qn, sub_part=sp,
                    instruction=instruction or None, reviewer=body.get("reviewer"),
                )
            except Exception as e:
                logger.error(f"❌ review_webapp: diagram rebuild failed for {label}/{key}: {e}")
                return jsonify({"error": str(e)}), 500
            return _after_mutation(label)

        if not instruction:
            return jsonify({"error": "A correction instruction is required."}), 400
        try:
            correction_engine.regenerate_question(
                _path_for(label), qn, instruction, sub_part=sp,
                reviewer=body.get("reviewer"), component=component,
            )
        except Exception as e:
            logger.error(f"❌ review_webapp: regeneration failed for {label}/{key}: {e}")
            return jsonify({"error": str(e)}), 500
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/question/<key>/diagram", methods=["DELETE"])
    def api_remove_diagram(label, key):
        qn, sp = _split_key(key)
        body = request.json if request.is_json else {}
        try:
            correction_engine.remove_generated_diagram(
                _path_for(label), qn, sub_part=sp, reviewer=(body or {}).get("reviewer"))
        except Exception as e:
            logger.error(f"❌ review_webapp: remove-diagram failed for {label}/{key}: {e}")
            return jsonify({"error": str(e)}), 500
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/question/<key>/figure", methods=["DELETE"])
    def api_remove_figure(label, key):
        qn, sp = _split_key(key)
        body = request.json if request.is_json else {}
        try:
            correction_engine.remove_book_figure(
                _path_for(label), qn, sub_part=sp, reviewer=(body or {}).get("reviewer"))
        except Exception as e:
            logger.error(f"❌ review_webapp: remove-figure failed for {label}/{key}: {e}")
            return jsonify({"error": str(e)}), 500
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/question/<key>/diagram-position", methods=["POST"])
    def api_set_diagram_position(label, key):
        qn, sp = _split_key(key)
        body = request.json or {}
        position = (body.get("position") or "").strip().lower()
        try:
            correction_engine.set_diagram_position(
                _path_for(label), qn, position, sub_part=sp, reviewer=body.get("reviewer"))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/question/<key>/edit", methods=["POST"])
    def api_edit_directly(label, key):
        """Direct edit of question fields without AI regeneration."""
        qn, sp = _split_key(key)
        body = request.json or {}
        field_updates = body.get("field_updates", {})
        
        if not field_updates:
            return jsonify({"error": "No field updates provided."}), 400
        
        try:
            correction_engine.edit_question_directly(
                _path_for(label), qn, field_updates, sub_part=sp,
                reviewer=body.get("reviewer")
            )
        except Exception as e:
            logger.error(f"❌ review_webapp: direct edit failed for {label}/{key}: {e}")
            return jsonify({"error": str(e)}), 500
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/question/<key>/figure", methods=["POST"])
    def api_upload_figure(label, key):
        qn, sp = _split_key(key)
        figure_number = request.form.get("figure_number")
        page_raw = request.form.get("page_number")
        reviewer = request.form.get("reviewer")

        image_bytes = None
        if "file" in request.files and request.files["file"].filename:
            image_bytes = request.files["file"].read()
        elif request.is_json:
            body = request.json or {}
            figure_number = figure_number or body.get("figure_number")
            page_raw = page_raw or body.get("page_number")
            reviewer = reviewer or body.get("reviewer")
            data_url = body.get("image_base64") or ""
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            if data_url:
                try:
                    image_bytes = base64.b64decode(data_url)
                except Exception:
                    image_bytes = None

        if not image_bytes:
            return jsonify({"error": "No image supplied (paste, drop a file, or select one)."}), 400
        # figure_number is optional: leave it set (as typed, or None) —
        # correction_engine.attach_manual_figure_to_question branches on
        # whether it's actually a real citation vs. just "attach this
        # image to this one question, no book citation involved."
        figure_number = figure_number or None

        try:
            correction_engine.attach_manual_figure_to_question(
                _path_for(label), qn, image_bytes=image_bytes, figure_number=figure_number,
                sub_part=sp, page_number=_parse_page_number(page_raw),
                reviewer=reviewer,
            )
        except Exception as e:
            logger.error(f"❌ review_webapp: figure attach failed for {label}/{key}: {e}")
            return jsonify({"error": str(e)}), 500
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/approve", methods=["POST"])
    def api_approve(label):
        payload = _exercise_payload(_path_for(label))
        if payload is None:
            return jsonify({"error": f"No review state for exercise {label}."}), 404
        pending = [s["key"] for s in payload["questions"] if not s["accepted"]]
        if pending:
            return jsonify({"error": f"Cannot approve — question(s) {pending} not yet accepted."}), 400
        body = request.json or {}
        review_state.approve(_path_for(label), reviewer=body.get("reviewer"), note=body.get("note"))
        return _after_mutation(label)

    @app.route("/api/exercise/<label>/approve-all", methods=["POST"])
    def api_approve_all(label):
        """Bulk convenience for a reviewer who has looked over every
        question and is confident none need individual corrections:
        accepts every not-yet-accepted question, THEN approves the
        exercise — all from a single click/request. Does NOT replace
        the per-question Approve button or the single-exercise Approve
        endpoint above (both stay exactly as they are for anyone who
        wants to keep reviewing one question at a time); this is
        purely an additional path to the same end state, for exercises
        where a batch sign-off is appropriate."""
        path = _path_for(label)
        payload = _exercise_payload(path)
        if payload is None:
            return jsonify({"error": f"No review state for exercise {label}."}), 404
        body = request.json or {}
        reviewer = body.get("reviewer")
        for q in payload["questions"]:
            if not q["accepted"]:
                qn, sp = _split_key(q["key"])
                review_state.mark_question_accepted(path, qn, sp, reviewer=reviewer)
        review_state.approve(path, reviewer=reviewer, note=body.get("note"))
        return _after_mutation(label)

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        app.config["STOP_REQUESTED"] = True
        return jsonify({"stopped": True})

    def _after_mutation(label):
        payload = _exercise_payload(_path_for(label))
        return jsonify(payload)

    def _find_by_key(state, key):
        for q in state["solved"].get("questions", []):
            if _question_key(q) == key:
                return q
        return None

    def _split_key(key: str):
        if "(" in key and key.endswith(")"):
            qn, sp = key[:-1].split("(", 1)
            return qn, sp
        return key, None

    return app


# ----------------------------------------------------------------------
# Server lifecycle — auto-launched by main.py's Stage 2 (config.
# REVIEW_MODE == "web", the default) instead of interactive_review.py's
# blocking console loop. Runs until either every exercise in this
# class/chapter is approved (mirrors interactive_review.py returning
# True) or the reviewer clicks "Pause & Exit" in the browser (mirrors
# the console loop's [Q]uit — nothing accepted/corrected/approved so
# far is ever lost either way, since every action is persisted
# immediately by review_state.py exactly like the CLI path).
# ----------------------------------------------------------------------
class _ServerThread(threading.Thread):
    def __init__(self, app: Flask, host: str, port: int):
        super().__init__(daemon=True)
        self._server = make_server(host, port, app)

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()


def run_review_server(class_name, chapter, output_dir: str = None,
                       host: str = None, port: int = None,
                       open_browser: bool = True, poll_interval: float = 0.5) -> bool:
    """Blocking call — starts the review app, opens it in the default
    browser, and waits until this class/chapter is fully approved (or
    the reviewer explicitly pauses/exits), then shuts the server down.

    Returns True if the chapter finished fully approved (main.py should
    proceed to review_state.publish_chapter next — same contract
    interactive_review.review_exercise_interactively already has),
    False if the reviewer paused (main.py should stop for this run;
    re-running it later resumes exactly where the reviewer left off,
    same as the console workflow).
    """
    if make_server is None:
        raise RuntimeError("werkzeug.serving.make_server is unavailable — cannot start the review app.")

    output_dir = output_dir or config.OUTPUT_DIR
    host = host or getattr(config, "REVIEW_HOST", "127.0.0.1")
    port = port or getattr(config, "REVIEW_PORT", 5151)

    app = create_app(class_name, chapter, output_dir)
    server = _ServerThread(app, host, port)
    server.start()

    url = f"http://{host}:{port}/"
    logger.info(f"📝 Human Review app running at {url} — review every exercise, then it will "
                f"close itself automatically once Chapter {chapter} is fully approved.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"⚠️ Could not auto-open a browser ({e}) — open {url} manually.")

    try:
        while True:
            if app.config.get("STOP_REQUESTED"):
                logger.info(f"⏸️ Review paused by reviewer. Re-run main.py to resume Chapter {chapter} — "
                            f"everything accepted/corrected/approved so far is already saved.")
                return False
            status = review_state.chapter_publish_status(class_name, chapter, output_dir)
            if status["ready_to_publish"]:
                logger.info(f"✅ Every exercise in Chapter {chapter} is approved — closing the review app.")
                return True
            time.sleep(poll_interval)
    finally:
        server.shutdown()


# ----------------------------------------------------------------------
# Inline templates — deliberately dependency-free (no new files under
# templates/, which is PDF-rendering territory owned by html_renderer.py
# / pdf_generator.py; this is a completely separate, non-print UI).
# ----------------------------------------------------------------------
_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tiro+Bangla:wght@400&display=swap');
:root { --accent:#1a5c3a; --warn:#b8860b; --bad:#a83232; --bg:#f6f7f5; }
* { box-sizing: border-box; }
body { font-family: "Tiro Bangla", "Noto Sans Bengali", -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); margin:0; color:#222; font-synthesis: none; }
header { background: var(--accent); color:#fff; padding: 14px 20px; display:flex; justify-content:space-between; align-items:center; }
header a { color:#fff; text-decoration:none; font-weight:600; }
main { max-width: 1100px; margin: 20px auto; padding: 0 16px 60px; }
.card { background:#fff; border:1px solid #ddd; border-radius:8px; padding:16px 18px; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; color:#fff; }
.badge.pending_review { background: var(--warn); }
.badge.approved { background: var(--accent); }
.badge.rejected { background: var(--bad); }
table { width:100%; border-collapse: collapse; }
td, th { text-align:left; padding:8px 6px; border-bottom:1px solid #eee; }
.data-table { width:auto; border-collapse: collapse; margin:8px 0; font-size:14px; }
.data-table th, .data-table td { text-align:center; padding:6px 16px; border:1px solid #cfe3d6; }
.data-table th { background:#eaf5ee; color:#1a5c3a; font-weight:700; }
.data-table td { background:#fff; }
a.exlink { text-decoration:none; color: var(--accent); font-weight:600; }
.qcard { border-left: 5px solid #ccc; }
.qcard.accepted { border-left-color: var(--accent); }
.qcard.flagged { border-left-color: var(--warn); }
.row { display:flex; gap:18px; flex-wrap:wrap; }
.col { flex:1 1 320px; min-width:280px; }
.steps { white-space: pre-wrap; font-size: 14px; line-height:1.5; background:#fafafa; padding:8px 10px; border-radius:6px; }
.preview-box { border:1px dashed #bbb; border-radius:6px; min-height:160px; display:flex; align-items:center; justify-content:center; background:#fcfcfc; padding:6px; }
.preview-box img, .preview-box object { max-width:100%; max-height:280px; }
.missing { color:#888; font-style:italic; }
.btnbar { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
button { cursor:pointer; border:none; border-radius:6px; padding:8px 12px; font-size:13px; font-weight:600; }
.btn-accept { background:var(--accent); color:#fff; }
.btn-regen { background:#2a4d8f; color:#fff; }
.btn-fig { background:#6b4fa0; color:#fff; }
.btn-custom { background:#444; color:#fff; }
.btn-edit { background:#d68a00; color:#fff; }
.btn-skip { background:#999; color:#fff; }
.posbar { margin-top:6px; }
.btn-pos { background:#e4e4e4; color:#333; }
.btn-pos.active { background:#1a5c3a; color:#fff; }
.btn-approve { background:#111; color:#fff; padding:10px 18px; }
.btn-stop { background:transparent; color:#fff; border:1px solid #fff; }
textarea, input[type=text], input[type=number] { width:100%; padding:6px; border:1px solid #ccc; border-radius:4px; font-size:13px; }
.inline-form { display:none; margin-top:10px; background:#f2f2f2; padding:10px; border-radius:6px; }
.inline-form.open { display:block; }
.warn { color: var(--warn); font-weight:600; }
.small { font-size:12px; color:#666; }
.info { color: #2a4d8f; font-weight:600; }
.progress { height:8px; background:#eee; border-radius:4px; overflow:hidden; margin-top:6px;}
.edit-field-group { margin-bottom:10px; }
.edit-field-group label { display:block; margin-bottom:4px; font-weight:600; font-size:12px; color:#444; }
.progress > div { height:100%; background: var(--accent); }
.info-badges { display:flex; gap:6px; flex-wrap:wrap; margin:6px 0; }
.ibadge { font-size:11px; font-weight:600; padding:2px 8px; border-radius:10px; border:1px solid; }
.ibadge.ok { color:#1a5c3a; border-color:#1a5c3a; background:#eaf5ee; }
.ibadge.warn { color:#8a6100; border-color:#b8860b; background:#fff8e6; }
.ibadge.bad { color:#a83232; border-color:#a83232; background:#fdeaea; }
.ibadge.neutral { color:#555; border-color:#ccc; background:#f5f5f5; }
.qcard.current { outline: 3px solid #2a4d8f; outline-offset: 2px; }
.kbd-hint { font-size:11px; color:#ccc; margin-top:4px; }
.kbd-hint kbd { background:#333; border-radius:3px; padding:1px 5px; margin:0 2px; color:#fff; }
"""

_DASHBOARD_HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Human Review — AssamStudyAI</title><style>{_BASE_CSS}</style></head>
<body>
<header><div>📚 Human Review — Class/Chapter dashboard</div>
<button class="btn-stop" onclick="stopReview()">Pause &amp; Exit</button></header>
<main>
  <div class="card" id="chapter-summary">Loading…</div>
  <div class="card"><table id="ex-table"><thead><tr>
    <th>Exercise</th><th>Status</th><th>Progress</th><th>Version</th><th></th>
  </tr></thead><tbody></tbody></table></div>
</main>
<script>
async function load() {{
  const r = await fetch('/api/exercises');
  const data = await r.json();
  const cs = data.chapter_status;
  document.getElementById('chapter-summary').innerHTML =
    `<b>Class ${{data.class_name}} · Chapter ${{data.chapter}}</b> — ` +
    `${{cs.approved}}/${{cs.total_exercises}} exercises approved. ` +
    (cs.ready_to_publish ? '<span style="color:#1a5c3a;font-weight:700">✅ Ready to publish — closing automatically…</span>'
                          : '<span class="warn">⏳ Not ready to publish yet</span>');
  const tbody = document.querySelector('#ex-table tbody');
  tbody.innerHTML = '';
  for (const ex of data.exercises) {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><a class="exlink" href="/exercise/${{ex.exercise_label}}">Exercise ${{ex.exercise_label}}</a></td>
      <td><span class="badge ${{ex.status}}">${{ex.status}}</span></td>
      <td>${{ex.accepted_count}}/${{ex.total_questions}}</td>
      <td>v${{ex.version}}</td>
      <td><a class="exlink" href="/exercise/${{ex.exercise_label}}">Review →</a></td>`;
    tbody.appendChild(tr);
  }}
}}
async function stopReview() {{
  await fetch('/api/stop', {{method:'POST'}});
  document.body.innerHTML = '<main><div class="card"><h2>⏸️ Review paused.</h2>' +
    '<p>Everything you accepted, corrected, or approved is saved. Re-run main.py to resume.</p></div></main>';
}}
load();
setInterval(load, 4000);
</script>
</body></html>"""

_EXERCISE_HTML = f"""<!doctype html><html lang="as"><head><meta charset="utf-8">
<title>Review Exercise — AssamStudyAI</title><style>{_BASE_CSS}</style>
<!-- MathJax config for LaTeX rendering in Flask preview -->
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$']]
    }},
    svg: {{ fontCache: 'local' }},
    startup: {{
      ready: () => {{
        MathJax.startup.defaultReady();
      }}
    }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
<header><a href="/">← Dashboard</a><div id="ex-title">Loading…</div>
<button class="btn-stop" onclick="stopReview()">Pause &amp; Exit</button></header>
<main>
  <div class="card">
    <div id="ex-summary"></div>
    <div class="progress"><div id="ex-progress-bar" style="width:0%"></div></div>
    <div class="btnbar">
      <button class="btn-approve" onclick="approveExercise()">✓ Approve Exercise</button>
      <button class="btn-approve" onclick="approveAll()" style="background:#3a3a6b;">✓✓ Approve All &amp; Finish</button>
    </div>
    <div id="approve-error" class="warn small"></div>
    <div class="kbd-hint">
      <kbd>&#8592;</kbd><kbd>&#8594;</kbd> move &nbsp; <kbd>A</kbd> approve &nbsp; <kbd>S</kbd> regen solution &nbsp;
      <kbd>R</kbd> regen/gen diagram &nbsp; <kbd>D</kbd> remove diagram &nbsp; <kbd>B</kbd> remove book figure &nbsp;
      <kbd>U</kbd> upload figure &nbsp; <kbd>C</kbd> custom prompt &nbsp; <kbd>E</kbd> direct edit &nbsp; <kbd>Enter</kbd> save open form
    </div>
  </div>
  <div id="questions"></div>
</main>
<script>
const label = location.pathname.split('/').pop();
let ex = null;  // Global variable to store current exercise data

function escapeHtml(s) {{
  return (s || '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}

async function load() {{
  // BUG FIX: this used to call highlightCurrent() at the end, which
  // ALWAYS scrollIntoView()'d whatever `currentIndex` happened to be —
  // and currentIndex only ever changed via the ArrowLeft/ArrowRight
  // keyboard shortcuts, never via a mouse click. So clicking "Approve"
  // on question 20 called load() -> highlightCurrent() -> scrolled the
  // page straight back to question 0 (question 1) every single time,
  // exactly as reported. Fixed two ways at once: (1) load() now saves
  // and restores the EXACT scroll position across the DOM rebuild
  // instead of jumping anywhere, and (2) every mouse-click action
  // below now updates currentIndex to match whichever question was
  // actually clicked, so keyboard shortcuts immediately afterward
  // still act on the right question too.
  const savedScrollY = window.scrollY;
  const r = await fetch(`/api/exercise/${{label}}`);
  if (!r.ok) {{ document.getElementById('questions').innerHTML = '<div class="card">Not found.</div>'; return; }}
  ex = await r.json();  // Store in global variable for toggleForm to access
  document.getElementById('ex-title').innerText = `Exercise ${{ex.exercise_label}} (Class ${{ex.class_name}}, Ch. ${{ex.chapter}})`;
  document.getElementById('ex-summary').innerHTML =
    `Status: <span class="badge ${{ex.status}}">${{ex.status}}</span> &nbsp; Version v${{ex.version}} &nbsp; ` +
    `${{ex.accepted_count}}/${{ex.total_questions}} questions accepted`;
  const pct = ex.total_questions ? Math.round(100 * ex.accepted_count / ex.total_questions) : 0;
  document.getElementById('ex-progress-bar').style.width = pct + '%';

  const container = document.getElementById('questions');
  container.innerHTML = '';
  ex.questions.forEach((q, i) => container.appendChild(renderQuestion(q, i)));
  totalQuestions = ex.questions.length;
  applyCurrentHighlightClass();  // just the CSS outline — never scrolls
  requestAnimationFrame(() => window.scrollTo(0, savedScrollY));
}}

function badge(text, kind) {{
  return `<span class="ibadge ${{kind}}">${{escapeHtml(text)}}</span>`;
}}

function buildBadges(q) {{
  const b = [];
  if (q.has_book_figure) {{
    if (q.diagram_match_low_confidence) {{
      b.push(badge('⚠ Book Figure (' + (q.book_diagram_figure_ref || '?') + ') — CAPTION MISMATCH, verify crop', 'bad'));
    }} else if (q.diagram_match_verified) {{
      b.push(badge('📷 Book Figure Verified (' + q.book_diagram_figure_ref + ')', 'ok'));
    }} else {{
      b.push(badge(`📷 Book Figure Found${{q.book_diagram_figure_ref ? ' (' + q.book_diagram_figure_ref + ')' : ''}}`, 'ok'));
    }}
  }} else if (q.diagram_decision === 'NO_DIAGRAM' && /citation|figure/i.test(q.diagram_decision_reason || '')) {{
    b.push(badge('📷 Missing Figure', 'warn'));
  }}
  if (q.has_generated_diagram) {{
    b.push(badge('🖼 AI Diagram', 'ok'));
  }}
  if (q.diagram_safety_net_flagged) {{
    b.push(badge('⚠ Diagram Validation Failed', 'bad'));
  }}
  if (q.diagram_decision_reason && /rejected|failed validation|mismatch/i.test(q.diagram_decision_reason)) {{
    b.push(badge('⚠ Geometry/Caption Check Failed', 'bad'));
  }}
  if (q.needs_review) {{
    b.push(badge('⚠ Low Confidence', 'warn'));
  }}
  if (!b.length) {{
    b.push(badge('No issues detected', 'neutral'));
  }}
  return b.join('');
}}

function renderQuestion(q, index) {{
  const div = document.createElement('div');
  div.className = 'card qcard ' + (q.accepted ? 'accepted' : (q.needs_review ? 'flagged' : ''));
  div.dataset.index = index;
  div.dataset.key = q.key;
  // Render solution content as HTML (not escaped) to match PDF output exactly
  // This preserves LaTeX/MathJax delimiters and Assamese text
  const stepsHtml = (q.steps || []).map(s => s).join('\\n');
  const diagramBox = q.has_generated_diagram
    ? `<object data="/api/exercise/${{label}}/diagram/${{encodeURIComponent(q.key)}}" type="image/svg+xml"></object>`
    : '<span class="missing">No diagram</span>';
  const figureBox = q.has_book_figure
    ? `<img src="/api/exercise/${{label}}/figure/${{encodeURIComponent(q.key)}}">`
    : '<span class="missing">Missing Book Figure</span>';
  // Only escape user-generated content (review notes), not solution content
  const warnings = (q.review_notes || []).map(n => `<div class="warn small">⚠️ ${{escapeHtml(n)}}</div>`).join('');
  const diagramBtnLabel = q.has_generated_diagram ? '♻ Regenerate Diagram'
                         : (q.has_book_figure ? '🖼 Generate AI Diagram Too' : '🖼 Generate Diagram');
  const hasAnyDiagram = q.has_generated_diagram || q.has_book_figure;
  const posBtn = (val, label) => `<button class="btn-pos ${{q.diagram_position === val ? 'active' : ''}}"
      onclick="setDiagramPosition('${{q.key}}','${{val}}')">${{label}}</button>`;

  div.innerHTML = `
    <div class="row">
      <div class="col">
        <h3>Q${{q.question_number}}${{q.sub_part ? '(' + q.sub_part + ')' : ''}} ${{q.accepted ? '✅' : ''}}</h3>
        <div class="info-badges">${{buildBadges(q)}}</div>
        <b>Original Question</b>
        <div class="steps">${{q.question_text}}</div>
        <b>Generated Solution</b>
        <div class="steps">Given: ${{q.given}}
Required: ${{q.required}}

${{stepsHtml}}

Final answer: ${{q.final_answer}}</div>
        ${{warnings}}
      </div>
      <div class="col">
        <b>Generated Diagram Preview</b>
        <div class="preview-box">${{diagramBox}}</div>
        <b>Book Figure Preview ${{q.book_diagram_figure_ref ? '(Fig. ' + escapeHtml(String(q.book_diagram_figure_ref)) + ')' : ''}}</b>
        <div class="preview-box">${{figureBox}}</div>
      </div>
    </div>
    <div class="btnbar">
      <button class="btn-accept" onclick="accept('${{q.key}}')">✓ Approve</button>
      <button class="btn-regen" onclick="toggleForm('${{q.key}}','solution')">♻ Regenerate Solution</button>
      <button class="btn-regen" onclick="toggleForm('${{q.key}}','diagram')">${{diagramBtnLabel}}</button>
      ${{q.has_generated_diagram ? `<button class="btn-skip" onclick="removeDiagram('${{q.key}}')">🗑 Remove Diagram</button>` : ''}}
      <button class="btn-fig" onclick="toggleForm('${{q.key}}','figure')">📷 Upload Book Figure</button>
      ${{q.has_book_figure ? `<button class="btn-skip" onclick="removeFigure('${{q.key}}')">🗑 Remove Book Figure</button>` : ''}}
      <button class="btn-custom" onclick="toggleForm('${{q.key}}','custom')">✎ Custom Prompt</button>
      <button class="btn-edit" onclick="toggleForm('${{q.key}}','edit')">✏️ Direct Edit</button>
      <button class="btn-skip" onclick="skip('${{q.key}}')">⏭ Skip</button>
    </div>
    ${{hasAnyDiagram ? `
    <div class="btnbar posbar">
      <span class="small" style="align-self:center;margin-right:4px;">Diagram position:</span>
      ${{posBtn('auto', 'Default')}}
      ${{posBtn('side', '➡ Beside text')}}
      ${{posBtn('large', '⬛ Large, under question')}}
    </div>` : ''}}
    <div class="inline-form" id="form-${{q.key}}"></div>
  `;
  return div;
}}

function toggleForm(key, kind) {{
  setCurrentByKey(key);
  const el = document.getElementById(`form-${{key}}`);
  const isOpen = el.classList.contains('open') && el.dataset.kind === kind;
  el.classList.remove('open');
  el.innerHTML = '';
  if (isOpen) return;
  el.dataset.kind = kind;
  if (kind === 'figure') {{
    el.innerHTML = `
      <label class="small">Figure number as printed in the book — optional; leave blank to just attach this image to the question</label>
      <input type="text" id="fignum-${{key}}" placeholder="e.g. 7.18, or leave blank">
      <label class="small">Page number (optional)</label>
      <input type="number" id="figpage-${{key}}">
      <label class="small">Paste (Ctrl+V) or choose a file</label>
      <input type="file" id="figfile-${{key}}" accept="image/*">
      <div id="figpreview-${{key}}" class="preview-box" style="margin-top:6px;min-height:80px;"></div>
      <div class="btnbar"><button class="btn-fig" onclick="submitFigure('${{key}}')">Save Figure</button></div>
      <div class="small warn" id="err-${{key}}"></div>`;
    const pasteTarget = el;
    let pastedDataUrl = null;
    pasteTarget.tabIndex = 0;
    pasteTarget.addEventListener('paste', (e) => {{
      const items = (e.clipboardData || {{}}).items || [];
      for (const item of items) {{
        if (item.type.indexOf('image') === 0) {{
          const blob = item.getAsFile();
          const reader = new FileReader();
          reader.onload = () => {{
            pastedDataUrl = reader.result;
            document.getElementById(`figpreview-${{key}}`).innerHTML = `<img src="${{pastedDataUrl}}">`;
          }};
          reader.readAsDataURL(blob);
        }}
      }}
    }});
    el._getPastedDataUrl = () => pastedDataUrl;
    pasteTarget.focus();
  }} else if (kind === 'edit') {{
    // Direct edit form - allows editing specific fields without AI regeneration
    el.innerHTML = `
      <label class="small">Edit fields directly (no AI regeneration)</label>
      <div class="edit-field-group">
        <label class="small">Given (প্ৰদত্ত)</label>
        <textarea rows="2" id="edit-given-${{key}}" placeholder="Given information..."></textarea>
      </div>
      <div class="edit-field-group">
        <label class="small">Required (প্ৰয়োজনীয়)</label>
        <textarea rows="2" id="edit-required-${{key}}" placeholder="What to prove/find..."></textarea>
      </div>
      <div class="edit-field-group">
        <label class="small">Steps (সমাধান) - Use \\( \\) for inline math, $$ $$ for display math</label>
        <textarea rows="6" id="edit-steps-${{key}}" placeholder="Step 1...\\nStep 2..."></textarea>
      </div>
      <div class="edit-field-group">
        <label class="small">Final Answer (চূড়ান্ত উত্তৰ)</label>
        <textarea rows="2" id="edit-final-${{key}}" placeholder="Final answer..."></textarea>
      </div>
      <div class="btnbar">
        <button class="btn-edit" onclick="submitDirectEdit('${{key}}')">💾 Save Edits</button>
        <button class="btn-skip" onclick="skip('${{key}}')">Cancel</button>
      </div>
      <div class="small warn" id="err-${{key}}"></div>
      <div class="small info">💡 Supports MathJax: use \\(x^2\\) for inline, $$x^2$$ for display. Assamese text will render with correct font.</div>`;
    
    // Pre-fill with current values
    const currentQ = ex.questions.find(q => q.key === key);
    if (currentQ) {{
      document.getElementById(`edit-given-${{key}}`).value = currentQ.given || '';
      document.getElementById(`edit-required-${{key}}`).value = currentQ.required || '';
      document.getElementById(`edit-steps-${{key}}`).value = (currentQ.steps || []).join('\\n');
      document.getElementById(`edit-final-${{key}}`).value = currentQ.final_answer || '';
    }}
  }} else {{
    const label_ = kind === 'solution' ? "What's wrong with the solution, and what to fix"
                 : kind === 'diagram' ? "Optional guidance for the rebuild (leave blank to just rebuild fresh from the question)"
                 : "Custom instruction (solution and/or diagram)";
    const btnLabel = kind === 'diagram' ? 'Rebuild Diagram' : 'Regenerate';
    el.innerHTML = `
      <label class="small">${{label_}}</label>
      <textarea rows="3" id="instr-${{key}}" placeholder="e.g. Altitude missing / Use textbook method / Diagram labels wrong"></textarea>
      <div class="btnbar"><button class="btn-regen" onclick="submitRegenerate('${{key}}','${{kind === 'custom' ? '' : kind}}')">${{btnLabel}}</button></div>
      <div class="small warn" id="err-${{key}}"></div>`;
  }}
  el.classList.add('open');
}}

async function accept(key) {{
  setCurrentByKey(key);
  const r = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/accept`, {{method:'POST'}});
  if (r.ok) load();
}}

function skip(key) {{
  const el = document.getElementById(`form-${{key}}`);
  if (el) el.classList.remove('open');
}}

async function submitRegenerate(key, component) {{
  const instruction = document.getElementById(`instr-${{key}}`).value.trim();
  const errEl = document.getElementById(`err-${{key}}`);
  if (!instruction && component !== 'diagram') {{ errEl.innerText = 'Please describe what to fix.'; return; }}
  errEl.innerText = component === 'diagram' ? 'Rebuilding from scratch…' : 'Regenerating…';
  const r = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/regenerate`, {{
    method: 'POST', headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{instruction, component: component || null}})
  }});
  const data = await r.json();
  if (!r.ok) {{ errEl.innerText = data.error || 'Regeneration failed.'; return; }}
  load();
}}

async function removeDiagram(key) {{
  setCurrentByKey(key);
  const r = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/diagram`, {{method:'DELETE'}});
  if (r.ok) load();
}}

async function removeFigure(key) {{
  setCurrentByKey(key);
  const r = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/figure`, {{method:'DELETE'}});
  if (r.ok) load();
}}

async function setDiagramPosition(key, position) {{
  setCurrentByKey(key);
  const r = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/diagram-position`, {{
    method: 'POST', headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{position}})
  }});
  if (r.ok) load();
}}

async function submitFigure(key) {{
  const el = document.getElementById(`form-${{key}}`);
  const figure_number = document.getElementById(`fignum-${{key}}`).value.trim();
  const page_number = document.getElementById(`figpage-${{key}}`).value;
  const fileInput = document.getElementById(`figfile-${{key}}`);
  const errEl = document.getElementById(`err-${{key}}`);
  errEl.innerText = 'Uploading…';

  let resp;
  if (fileInput.files.length) {{
    const fd = new FormData();
    fd.append('figure_number', figure_number);
    if (page_number) fd.append('page_number', page_number);
    fd.append('file', fileInput.files[0]);
    resp = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/figure`, {{method:'POST', body: fd}});
  }} else {{
    const pastedDataUrl = el._getPastedDataUrl ? el._getPastedDataUrl() : null;
    if (!pastedDataUrl) {{ errEl.innerText = 'Paste an image (Ctrl+V) or choose a file.'; return; }}
    resp = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/figure`, {{
      method: 'POST', headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{figure_number, page_number: page_number || null, image_base64: pastedDataUrl}})
    }});
  }}
  const data = await resp.json();
  if (!resp.ok) {{ errEl.innerText = data.error || 'Upload failed.'; return; }}
  load();
}}

async function submitDirectEdit(key) {{
  const errEl = document.getElementById(`err-${{key}}`);
  errEl.innerText = 'Saving edits…';
  
  const field_updates = {{}};
  const given = document.getElementById(`edit-given-${{key}}`).value.trim();
  const required = document.getElementById(`edit-required-${{key}}`).value.trim();
  const stepsText = document.getElementById(`edit-steps-${{key}}`).value.trim();
  const final = document.getElementById(`edit-final-${{key}}`).value.trim();
  
  if (given) field_updates.given = given;
  if (required) field_updates.required = required;
  if (stepsText) field_updates.steps = stepsText.split('\\n').filter(s => s.trim());
  if (final) field_updates.final_answer = final;
  
  if (Object.keys(field_updates).length === 0) {{
    errEl.innerText = 'No changes to save.';
    return;
  }}
  
  const r = await fetch(`/api/exercise/${{label}}/question/${{encodeURIComponent(key)}}/edit`, {{
    method: 'POST', headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify({{field_updates}})
  }});
  const data = await r.json();
  if (!r.ok) {{ errEl.innerText = data.error || 'Edit failed.'; return; }}
  load();
}}

async function approveExercise() {{
  const errEl = document.getElementById('approve-error');
  const r = await fetch(`/api/exercise/${{label}}/approve`, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: '{{}}'}});
  const data = await r.json();
  if (!r.ok) {{ errEl.innerText = data.error || 'Could not approve.'; return; }}
  errEl.innerText = '';
  load();
}}

async function approveAll() {{
  const errEl = document.getElementById('approve-error');
  const remaining = document.querySelectorAll('.qcard:not(.accepted)').length;
  const msg = remaining > 0
    ? `Approve all ${{remaining}} remaining question(s) and finish this exercise?`
    : 'Finish and approve this exercise?';
  if (!window.confirm(msg)) return;
  errEl.innerText = 'Approving all…';
  const r = await fetch(`/api/exercise/${{label}}/approve-all`, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: '{{}}'}});
  const data = await r.json();
  if (!r.ok) {{ errEl.innerText = data.error || 'Could not approve all.'; return; }}
  errEl.innerText = '';
  load();
}}

async function stopReview() {{
  await fetch('/api/stop', {{method:'POST'}});
  document.body.innerHTML = '<main><div class="card"><h2>⏸️ Review paused.</h2>' +
    '<p>Everything you accepted, corrected, or approved is saved. Re-run main.py to resume.</p></div></main>';
}}

// ------------------------------------------------------------------
// Keyboard shortcuts (Human Review speed target: reviewer should
// rarely need the mouse). Arrow keys move a "current question"
// highlight; the action keys act on whichever question is current.
// Disabled whenever focus is inside a text input/textarea (e.g. while
// typing a correction instruction) so normal typing is never hijacked.
// ------------------------------------------------------------------
let currentIndex = 0;
let totalQuestions = 0;

function applyCurrentHighlightClass() {{
  document.querySelectorAll('.qcard').forEach(el => {{
    el.classList.toggle('current', Number(el.dataset.index) === currentIndex);
  }});
}}

function highlightCurrent() {{
  // Used ONLY by explicit ArrowLeft/ArrowRight keyboard navigation —
  // this is the one case where actually scrolling the page is exactly
  // what the reviewer asked for. Never call this from load().
  applyCurrentHighlightClass();
  const el = document.querySelector(`.qcard[data-index="${{currentIndex}}"]`);
  if (el) el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
}}

function setCurrentByKey(key) {{
  const el = document.querySelector(`.qcard[data-key="${{key}}"]`);
  if (el) {{
    currentIndex = Number(el.dataset.index);
    applyCurrentHighlightClass();
  }}
}}

function currentKey() {{
  const el = document.querySelector(`.qcard[data-index="${{currentIndex}}"]`);
  return el ? el.dataset.key : null;
}}

document.addEventListener('keydown', (e) => {{
  const tag = (document.activeElement && document.activeElement.tagName) || '';
  if (tag === 'TEXTAREA' || tag === 'INPUT') {{
    if (e.key === 'Enter' && e.target.id && e.target.id.startsWith('instr-')) {{
      // Enter inside a correction textarea saves that form, same as clicking its button.
      e.preventDefault();
      const key = e.target.id.replace('instr-', '');
      const formEl = document.getElementById(`form-${{key}}`);
      const component = formEl ? formEl.dataset.kind : null;
      submitRegenerate(key, component === 'custom' ? '' : component);
    }}
    return;
  }}
  const key = currentKey();
  switch (e.key) {{
    case 'ArrowRight':
      if (currentIndex < totalQuestions - 1) {{ currentIndex++; highlightCurrent(); }}
      break;
    case 'ArrowLeft':
      if (currentIndex > 0) {{ currentIndex--; highlightCurrent(); }}
      break;
    case 'a': case 'A':
      if (key) accept(key);
      break;
    case 's': case 'S':
      if (key) toggleForm(key, 'solution');
      break;
    case 'r': case 'R':
      if (key) toggleForm(key, 'diagram');
      break;
    case 'd': case 'D':
      if (key) removeDiagram(key);
      break;
    case 'b': case 'B':
      if (key) removeFigure(key);
      break;
    case 'u': case 'U':
      if (key) toggleForm(key, 'figure');
      break;
    case 'e': case 'E':
      if (key) toggleForm(key, 'edit');
      break;
    case 'c': case 'C':
      if (key) toggleForm(key, 'custom');
      break;
    case 'Enter': {{
      if (!key) break;
      const formEl = document.getElementById(`form-${{key}}`);
      if (formEl && formEl.classList.contains('open')) {{
        if (formEl.dataset.kind === 'figure') submitFigure(key);
        else if (formEl.dataset.kind === 'edit') submitDirectEdit(key);
        else submitRegenerate(key, formEl.dataset.kind === 'custom' ? '' : formEl.dataset.kind);
      }}
      break;
    }}
  }}
}});

load();
</script>
</body></html>"""
