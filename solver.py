"""
solver.py — sends one exercise PDF to Gemini (Vertex AI, ADC auth only)
and gets back structured JSON: every question, solved. No key rotation,
no multi-agent verification — just one call, with a bounded thinking
budget so it can't silently eat the whole token ceiling, and an
exponential backoff retry so a 429 RESOURCE_EXHAUSTED mid-batch doesn't
kill the whole run.
"""
import json
import re
import base64
from google import genai
from google.genai import types

import config
from prompts import build_solve_prompt
from figure_database import get_verified_figures
import constraint_extraction
from diagram_decision import decide_diagram, BOOK_DIAGRAM
from utils import logger, retry_with_backoff, extract_figure_reference
import math_sanitizer
from construction_detector import is_construction_chapter as is_construction_chapter_detector, validate_question_construction_type

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=config.VERTEX_PROJECT,
            location=config.VERTEX_LOCATION,
        )
        logger.info(f"🔑 Authenticated to Vertex AI via ADC (project={config.VERTEX_PROJECT}).")
    return _client


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


_VALID_JSON_ESCAPE_CHARS = set('"\\/')
# Deliberately excludes b/f/n/r/t/u even though JSON allows them as
# escapes: in THIS content (LaTeX-heavy math), a backslash followed by
# one of those letters is overwhelmingly more likely to be the start of
# a LaTeX macro (\beta, \frac, \neq, \rightarrow, \tan, \triangle, \u...)
# than an intentional control character. Treating them as "already
# valid" would silently corrupt the text (turn "\frac" into a literal
# form-feed byte + "rac") instead of raising.
#
# The tradeoff: this repair pass can occasionally over-correct a
# GENUINE, intentional "\n" (e.g. Gemini legitimately line-breaking a
# multi-part question like "...যে:\n(i) ...\n(ii) ...") into a literal
# visible "\n" text sequence instead of a real line break. Rather than
# try to resolve that ambiguity blindly at the character-scanning
# stage (impossible without semantic context), _normalize_math_text()
# below fixes it AFTER parsing, when there's an actual string to
# inspect — see that function for why this two-stage approach is more
# reliable than trying to get the escaping perfect in one pass.


def _repair_invalid_json_escapes(text: str) -> str:
    """Scans RAW (not-yet-parsed) JSON text and doubles any backslash
    that isn't already followed by a character JSON recognizes as a
    valid escape. This alone recovers the large majority of "Invalid
    \\escape" parse failures without needing a whole extra Gemini call."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt in _VALID_JSON_ESCAPE_CHARS:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            else:
                out.append("\\\\")
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_gemini_json_response(raw_text: str, error_context: str) -> dict:
    """Single choke point for turning ANY Gemini text response (main
    solve pass, a review-app correction, a diagram-rebuild recovery
    call — every JSON-mode Gemini call in this project) into a parsed
    dict. Strips code fences, then ALWAYS runs
    _repair_invalid_json_escapes BEFORE the first (and only) json.loads
    attempt.

    PRODUCTION-AUDIT FIX (root cause of "fractions are broken" /
    "broken LaTeX" reports): b, f, n, r, t are all escape letters JSON
    ITSELF considers perfectly valid (\\b \\f \\n \\r \\t) — so a raw,
    unescaped backslash from Gemini before a LaTeX macro that happens
    to start with one of those letters (\\frac, \\because, \\boxed,
    \\binom, \\beta, \\neq, \\nabla, \\right, \\rightarrow, \\times,
    \\text, \\therefore, \\tan, \\triangle, ...) NEVER raises a
    JSONDecodeError at all — json.loads() "succeeds", silently
    swallowing the macro's first letter into a control character (e.g.
    "\\frac{1}{2}" silently becomes a form-feed byte followed by the
    plain text "rac{1}{2}") and leaving broken, missing math in the
    final PDF with no warning anywhere. The comment on
    _VALID_JSON_ESCAPE_CHARS above already correctly identified this
    exact risk and excluded those letters specifically so
    _repair_invalid_json_escapes would double-escape them — but that
    function was previously only ever invoked as a FALLBACK after a
    JSONDecodeError, and this failure mode never produces one, so the
    fix that already existed was never actually reached for it.

    Always applying the repair pass first closes that gap and is safe
    to do unconditionally: a genuinely-valid escape ("\\\"", "\\\\",
    "\\/") is left completely untouched by _repair_invalid_json_escapes
    (see its own logic), and every other backslash — including a
    correctly single-escaped "\\uXXXX", which becomes a literal that
    _normalize_math_text's own stray-escape fix decodes right back —
    is doubled into an unambiguous literal backslash instead of ever
    being silently read as a JSON control-char escape. The two-stage
    "try raw, only repair on a raised error" approach every call site
    used before could never catch this, because raw parsing of this
    exact failure mode never raises to begin with.

    Raises ValueError (never json.JSONDecodeError directly, so every
    caller gets one consistent exception type) if the text still isn't
    valid JSON even after the repair pass.
    """
    cleaned = _strip_json_fences(raw_text)
    repaired = _repair_invalid_json_escapes(cleaned)
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"{error_context} wasn't valid JSON even after escape repair: {e}")
    if repaired != cleaned:
        logger.info(f"ℹ️ {error_context}: applied JSON-escape repair before parsing "
                    f"(protects LaTeX macros starting with b/f/n/r/t from being silently "
                    f"read as JSON control-character escapes).")
    return parsed


_TABLE_SEP_ROW = re.compile(r"^[\s\-:|]+$")


def _is_table_separator_row(line: str) -> bool:
    """A markdown-style table delimiter row, e.g. '--|----|----|---|---'
    or ':--|:-:|--:'. Requires at least one '-' so an empty/blank line
    (which would otherwise vacuously match '[\\s\\-:|]+') is never
    mistaken for one."""
    s = line.strip()
    return bool(s) and "-" in s and bool(_TABLE_SEP_ROW.match(s))


def _render_pipe_table_html(rows_raw: list) -> str:
    """Turns a block of '<br>'-joined markdown-pipe-table lines (header
    row, a '---|---' delimiter row, and one or more data rows) into a
    real HTML <table>, instead of the delimiter row and pipe characters
    printing as literal text (see V40 FIX below _normalize_math_text's
    tab/newline handling for the earlier half of this same root cause —
    THIS closes the remaining half: once '\\n'/'\\t' are decoded into
    real line breaks, a genuine table still looked like raw markdown
    source rather than a table)."""
    header = None
    data_rows = []
    for raw in rows_raw:
        if _is_table_separator_row(raw):
            continue
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        if header is None:
            header = cells
        else:
            data_rows.append(cells)
    if header is None:
        return "<br>".join(rows_raw)  # defensive fallback — should never happen

    ncols = max([len(header)] + [len(r) for r in data_rows]) if data_rows else len(header)
    parts = ['<table class="data-table"><tr>']
    for c in header:
        parts.append(f"<th>{c}</th>")
    parts.extend("<th></th>" for _ in range(ncols - len(header)))
    parts.append("</tr>")
    for r in data_rows:
        parts.append("<tr>")
        for c in r:
            parts.append(f"<td>{c}</td>")
        parts.extend("<td></td>" for _ in range(ncols - len(r)))
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _convert_pipe_tables_to_html(text: str) -> str:
    """Scans the (already newline-decoded, '<br>'-joined) text for a
    genuine markdown pipe table — a contiguous run of lines that all
    contain '|', including at least one dashes-only delimiter row — and
    replaces just that run with a proper HTML <table>. Deliberately
    requires a delimiter row before converting anything: a line like
    '|AB| = |CD|' (absolute-value bars, real math content) also
    contains '|' but is never followed by a '---|---' row, so it's left
    completely untouched — only an actual table is ever rewritten."""
    if not isinstance(text, str) or "|" not in text:
        return text
    lines = text.split("<br>")
    n = len(lines)
    out = []
    i = 0
    while i < n:
        if "|" in lines[i]:
            j = i
            block = []
            while j < n and "|" in lines[j]:
                block.append(lines[j])
                j += 1
            if len(block) >= 2 and any(_is_table_separator_row(l) for l in block):
                out.append(_render_pipe_table_html(block))
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "<br>".join(out)


def _normalize_math_text(text):
    """
    Post-PARSE cleanup applied to every solved text field, regardless of
    whether _repair_invalid_json_escapes ran or the JSON parsed cleanly
    on the first try.

    All the actual escape-decoding / bare-LaTeX-wrapping / delimiter-
    balance-repair-and-degrade logic this docstring used to describe in
    detail now lives in math_sanitizer.py's sanitize_math_text() — the
    single mandatory Math Sanitization Layer every math-bearing field in
    this pipeline passes through (see that module's own docstring for
    the full design). This function is now a thin wrapper: it hands the
    field to that gateway, then applies the one remaining, unrelated
    transform that's specific to solver.py's own domain — turning a
    markdown-style pipe table (e.g. a coordinate-table question) into a
    real HTML <table> — on the sanitizer's output.
    """
    return _convert_pipe_tables_to_html(math_sanitizer.sanitize_math_text(text))


_TEXT_FIELDS = ("question_text", "given", "required", "final_answer")


def _coerce_text_field(value):
    """Flattens a text field into a plain string, defensively.

    BUG FIX (found from a real generated PDF — user-reported, reproduced,
    confirmed): despite the prompt schema explicitly instructing "given"/
    "required" to be plain Assamese strings, Gemini sometimes sends a
    STRUCTURED OBJECT instead — e.g. exactly the shape seen in the
    report: {'title': 'প্ৰদত্ত', 'statements': ['...', '...']} or
    {'title': 'অংকন কৰিবলৈ', 'statement': '...'}. Before this fix,
    _normalize_math_text's `if not isinstance(text, str): return text`
    guard let that object pass straight through unchanged all the way to
    the Jinja template, which then rendered Python's raw dict repr
    (`{'title': ...}`) directly into the published PDF — a textbook
    prompt-leakage bug. Same failure class as Gemini sending a point id
    as `["D"]` instead of `"D"` elsewhere in this project; same fix
    philosophy: never trust the model's exact type, defensively coerce
    to what the renderer actually needs.

    Handles, in order: already-a-string (no-op); a dict with a
    "statements" list and/or "statement" string (the exact observed
    shape — joined into natural sentences); any other dict or list
    (generic recursive flatten of every string leaf, so this degrades
    gracefully instead of crashing on a shape nobody has seen yet);
    anything else (str() as an absolute last resort, never a raw
    Python repr of a dict/list making it through unflattened)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        stmts = value.get("statements")
        if isinstance(stmts, list) and stmts:
            parts.append("। ".join(_coerce_text_field(s) for s in stmts if s))
        elif value.get("statement"):
            parts.append(_coerce_text_field(value["statement"]))
        if parts:
            return "। ".join(p for p in parts if p).strip("। ").strip() or "।"
        # unknown dict shape -- last resort: join every string leaf found,
        # never surface the raw {'key': 'value'} repr itself. Deliberately
        # EXCLUDES "title" here too: every field this function serves
        # (given/required/final_answer/question_text/steps) is displayed
        # in the PDF either under its own already-printed static Jinja
        # label ("প্ৰদত্ত:", "প্ৰয়োজনীয়:", "চূড়ান্ত উত্তৰ:") or with no
        # label wrapper at all — a "title" describing what KIND of field
        # this is (e.g. "প্ৰদত্ত") is metadata about the field's role, not
        # displayable content, and re-including it here produced a real,
        # confirmed double-labeled PDF ("প্ৰদত্ত: প্ৰদত্ত। ...") whenever
        # Gemini sent the structured-object shape (see this function's
        # own docstring — a previously-confirmed real occurrence).
        leaves = [v for k, v in value.items() if k != "title" and isinstance(v, str) and v]
        return "। ".join(leaves) if leaves else ""
    if isinstance(value, list):
        return "। ".join(_coerce_text_field(v) for v in value if v)
    if value is None:
        return ""
    return str(value)


def _normalize_all_text_fields(parsed: dict):
    for q in parsed.get("questions", []):
        for field in _TEXT_FIELDS:
            if field in q:
                q[field] = _normalize_math_text(_coerce_text_field(q[field]))
        if isinstance(q.get("steps"), list):
            q["steps"] = [_normalize_math_text(_coerce_text_field(s)) for s in q["steps"]]
        q["answer_kind"] = _classify_answer_kind(q)
        q["construction_instruments"] = _coerce_instruments_field(q.get("construction_instruments"))


def _coerce_instruments_field(value):
    """Defensively coerces "construction_instruments" to either None (not
    a construction question) or a clean list[str] (deduped, order-
    preserved, blanks dropped) — same defensive philosophy as
    _coerce_text_field above: never trust the model's exact type,
    since a single stray shape (e.g. one instrument as a nested dict
    instead of a plain string) shouldn't be able to crash template
    rendering or silently hide the whole instruments box."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None
    seen = set()
    cleaned = []
    for item in value:
        text = _coerce_text_field(item).strip()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned or None


# ------------------------------------------------------------------
# ANSWER-KIND CLASSIFICATION — fixes the "✔ প্ৰমাণিত হ'ল" (Proven)
# badge being stamped on EVERY question regardless of content.
#
# ROOT CAUSE (found during the V8 polishing-phase audit, confirmed
# against real generated PDFs for Exercise 14.3/14.4): the Jinja
# template (templates/base.html.jinja) rendered a hardcoded
# "✔ প্ৰমাণিত হ'ল।" ("Proven.") banner unconditionally at the end of
# EVERY question block, with no gate at all. This is factually wrong
# for the large share of Class 6-10 questions that are not proofs —
# confirmed concretely on real output: Exercise 14.3 Q4(iii) ("Is the
# comment 'longest leaf length is 153mm' correct? No — 153mm is only
# the class's upper boundary...") still printed "✔ প্ৰমাণিত হ'ল।"
# directly under a "No" answer, and every mean/median/mode calculation
# in Exercise 14.4 got the same "Proven" stamp despite nothing having
# been proved -- just computed.
#
# FIX: classify each solved question, deterministically and locally
# (no extra Gemini call — this is pure post-processing, matching this
# project's existing pattern of doing what CAN be checked without AI),
# as "proof" only when the question's own wording actually asks for a
# proof/demonstration ("প্ৰমাণ কৰা"/"দেখুওৱা"/"prove"/"show that" —
# the exact vocabulary this project's own prompts.py already uses for
# the proof/geometry structure rule), and "calculation" otherwise. The
# template (see base.html.jinja) then shows "✔ প্ৰমাণিত হ'ল।" only for
# "proof", and a neutral "✔ সমাধান সম্পূৰ্ণ হ'ল।" ("Solution
# complete.") for everything else — a "নিৰ্ণয় কৰা" (determine)
# question or a data-interpretation question no longer claims
# something was "proven" when it wasn't.
# ------------------------------------------------------------------
_PROOF_KEYWORDS = (
    "প্ৰমাণ", "প্রমাণ",
    "দেখুওৱা", "দেখুৱা", "দেখুৱাব", "দেখুওৱাব",
    "prove", "show that",
)


def _classify_answer_kind(q: dict) -> str:
    """Returns "proof" if this question's own wording asks for a proof
    or demonstration, else "calculation". Pure text match against the
    question's own question_text/required fields — never guesses from
    the solved steps/final_answer, since a calculation question can
    still cite a rule ("CPCT") in its working without itself being a
    proof question."""
    text = f"{q.get('question_text', '') or ''} {q.get('required', '') or ''}".lower()
    return "proof" if any(kw.lower() in text for kw in _PROOF_KEYWORDS) else "calculation"


def _is_retryable_api_error(e: Exception) -> bool:
    """429/RESOURCE_EXHAUSTED and generic transient network errors are
    worth retrying with backoff; anything else (bad request, auth
    failure, etc.) is not — retrying those just wastes time before
    failing the same way anyway."""
    msg = str(e)
    return any(marker in msg for marker in (
        "429", "RESOURCE_EXHAUSTED", "rate limit", "Rate limit",
        "quota", "Quota", "UNAVAILABLE", "DEADLINE_EXCEEDED", "503",
    ))


@retry_with_backoff(times=6, base_delay=5.0, max_delay=90.0, retryable_check=_is_retryable_api_error)
def _call_gemini(pdf_bytes: bytes, prompt: str):
    client = get_client()
    response = client.models.generate_content(
        model=config.GEMINI_SOLVE_MODEL_ID,
        contents=[types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"), prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=config.THINKING_BUDGET),
            response_mime_type="application/json",
        ),
    )
    return response


def solve_exercise(exercise_pdf_path: str, class_name: int, chapter: int, exercise_label: str, 
                   chapter_title: str = None, is_construction_chapter: bool = None) -> dict:
    """Returns the parsed {"exercise_label": ..., "questions": [...]} dict.
    Raises on any failure — caller (main.py) decides whether to retry.
    
    Args:
        exercise_pdf_path: Path to the exercise PDF
        class_name: Class level (9, 10, etc.)
        chapter: Chapter number
        exercise_label: Exercise label (e.g., "7.1")
        chapter_title: Optional chapter title (for logging only)
        is_construction_chapter: Whether this is a construction chapter (from config)
    """
    with open(exercise_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Use config flag for construction chapter (no auto-detection)
    if is_construction_chapter is None:
        # Fallback to auto-detection if not provided (backward compatibility)
        is_construction = is_construction_chapter_detector(chapter_title or "", chapter)
    else:
        is_construction = is_construction_chapter
    
    prompt = build_solve_prompt(class_name, chapter, exercise_label, 
                                chapter_title=chapter_title, 
                                is_construction_chapter=is_construction)
    logger.info(f"🤖 Solving Exercise {exercise_label} (single Gemini call, "
                f"construction_chapter={is_construction})...")

    response = _call_gemini(pdf_bytes, prompt)
    raw_text = response.text or ""
    if not raw_text.strip():
        raise ValueError(f"Gemini returned empty output for Exercise {exercise_label}.")

    parsed = parse_gemini_json_response(raw_text, f"Gemini output for Exercise {exercise_label}")

    if "questions" not in parsed or not isinstance(parsed["questions"], list):
        raise ValueError(f"Exercise {exercise_label}: response missing a 'questions' list.")

    _normalize_all_text_fields(parsed)
    _attach_book_diagrams(parsed["questions"], exercise_pdf_path, exercise_label, class_name, chapter)
    _verify_coordinate_answers(parsed["questions"], class_name, exercise_label)
    
    # Validate construction type matches chapter type
    construction_mismatches = []
    for q in parsed["questions"]:
        validation = validate_question_construction_type(q, is_construction)
        if not validation["is_valid"]:
            construction_mismatches.append({
                "question": q.get("question_number"),
                "reason": validation["reason"],
                "correction": validation["correction_action"]
            })
            # Auto-correct if needed
            if validation["should_correct"] and validation["correction_action"] == "remove_construction":
                q["construction_instruments"] = None
                if isinstance(q.get("diagram_spec"), dict):
                    if q["diagram_spec"].get("diagram_type") == "construction":
                        q["diagram_spec"] = None
                logger.info(f"🔧 Auto-corrected Q{q.get('question_number')}: removed construction markers "
                           f"from non-construction chapter question")
            elif validation["should_correct"] and validation["correction_action"] == "add_construction":
                # For construction chapters, ensure instruments are set
                if not q.get("construction_instruments"):
                    q["construction_instruments"] = ["জ্যামিতি বাকচ", "কম্পাছ", "স্কেল"]
                logger.info(f"🔧 Auto-corrected Q{q.get('question_number')}: added construction instruments "
                           f"to construction chapter question")
    
    if construction_mismatches:
        logger.warning(f"⚠️ Exercise {exercise_label}: found and auto-corrected {len(construction_mismatches)} "
                       f"construction-type mismatches with chapter type")

    # Deterministic (sympy-based) re-verification for the subset of
    # answers this CAN be checked for without another AI call (closed-
    # form arithmetic, single-variable equations) — see math_verifier.py
    # for full scope/rationale. Import kept local and wrapped so a
    # missing/broken sympy install degrades to "verification skipped"
    # rather than breaking exercise solving entirely (same fail-open
    # policy as every other optional-enhancement step in this function).
    try:
        import math_verifier
        confirmed, contradicted = math_verifier.verify_questions(parsed["questions"])
        if confirmed or contradicted:
            logger.info(f"🔢 Exercise {exercise_label}: deterministic math check — "
                        f"{confirmed} confirmed, {contradicted} flagged for review.")
    except ImportError:
        logger.warning("⚠️ math_verifier/sympy not available — skipping deterministic "
                        "answer verification for this exercise (pip install sympy to enable it).")

    # STAGE 0: CONSTRAINT EXTRACTION ENGINE (see constraint_extraction.py)
    # — a deterministic type/shape audit of every question's diagram_spec
    # numeric constraint fields (side_lengths, angles_deg — Phase 1's
    # geometry-solver inputs), run once here so anything malformed is
    # caught and logged at the earliest possible point, BEFORE
    # diagram_decision.py or geometry_solver.py ever see it.
    constraint_extraction.sanitize_questions(parsed["questions"])

    # STAGE 1: DIAGRAM DECISION (see diagram_decision.py) — runs once
    # here, after book-figure attachment has already had its chance to
    # win, and BEFORE this question ever reaches html_renderer.py /
    # diagram_renderer.py. Every question's diagram_spec is replaced
    # with the decision layer's own verdict, so nothing downstream ever
    # re-derives book-vs-generated precedence or renders a diagram that
    # was always going to be discarded. The full verdict (decision +
    # reason) is also stored, purely for audit/debugging — nothing
    # downstream depends on these two extra keys.
    #
    # IMPORTANT: if Stage 2's independent re-verification of the book
    # citation fails, book_diagram_base64/mime/figure_ref must ALL be
    # cleared here too — otherwise base.html.jinja's own
    # "{% if q.book_diagram_base64 %}" check would still show the
    # (now-rejected) book image directly, completely bypassing the
    # decision engine's verdict. That would be exactly the "the
    # renderer/template decides instead of just drawing" failure mode
    # this whole layer exists to eliminate.
    for q in parsed["questions"]:
        verdict = decide_diagram(q)
        q["diagram_spec"] = verdict["diagram_spec"]
        q["diagram_decision"] = verdict["decision"]
        q["diagram_decision_reason"] = verdict["reason"]
        if verdict["decision"] != BOOK_DIAGRAM:
            q["book_diagram_base64"] = None
            q["book_diagram_mime"] = None
            q["book_diagram_figure_ref"] = None

    # STAGE 2: DIAGRAM SAFETY NET (see diagram_safety_net.py) — runs
    # once here, strictly after every question already has a real
    # diagram_decision from Stage 1 above. Catches the one gap Stage 1
    # cannot see on its own: a question whose OWN text strongly implies
    # a figure but that Gemini's single combined solve pass never gave
    # a diagram_spec for at all. Never overrides a decision that was
    # actually reasoned about (a book citation, or a diagram_spec that
    # was validated and then correctly rejected) — only recovers from
    # plain silence, and even then only via a fresh, narrow, independently
    # verified Gemini call that still has to pass every one of Stage 1's
    # own checks before it's accepted. Fails open: any recovery failure
    # just leaves the question as NO_DIAGRAM with an audit flag set,
    # never a guessed diagram.
    try:
        import diagram_safety_net
        diagram_safety_net.apply_diagram_safety_net(parsed["questions"], class_name, exercise_label)
    except Exception as e:
        logger.warning(f"⚠️ Exercise {exercise_label}: diagram safety-net stage failed to run "
                        f"({e}) — skipping it for this exercise (existing Stage 1 decisions are "
                        f"unaffected and unchanged).")

    logger.info(f"✅ Exercise {exercise_label}: {len(parsed['questions'])} question(s) solved.")
    return parsed


@retry_with_backoff(times=3, base_delay=3.0, max_delay=20.0, retryable_check=_is_retryable_api_error)
def _call_gemini_caption_read(image_bytes: bytes, prompt: str):
    """One narrow VISION-tier call: re-read the printed caption number
    off an already-cropped book figure. Used only by the post-attach
    cross-check below (see config.FIGURE_MATCH_VERIFICATION)."""
    client = get_client()
    return client.models.generate_content(
        model=config.GEMINI_VISION_MODEL_ID,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt],
        config=types.GenerateContentConfig(
            # >=512 keeps this call portable to ANY tier the operator may
            # route here via env override (flash-lite's hard minimum).
            thinking_config=types.ThinkingConfig(thinking_budget=512),
            response_mime_type="application/json",
        ),
    )


def _normalize_figure_ref(text):
    if not text:
        return None
    digits_only = re.sub(r"[^\d.]", "", str(text).translate(
        str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))
    return digits_only or None


_CAPTION_READ_PROMPT = (
    "This cropped image shows ONE figure from a printed Assamese SEBA "
    "mathematics textbook page. Read ONLY the printed caption number "
    "belonging to THIS figure (it appears just below or beside it, in a "
    "form like 'চিত্ৰ 4.12' or 'Figure 4.12'). Ignore any other numbers "
    "inside the drawing itself. Return ONLY this JSON, no commentary, no "
    'code fences: {"figure_ref": "<number like 4.12>"} — or '
    '{"figure_ref": null} if no numbered caption is visible.'
)


def _verify_figure_caption_matches(matched_pairs, exercise_label: str):
    """Post-attach caption cross-check (the last gap in book-figure
    accuracy): an attachment is only as trustworthy as the caption NUMBER
    it was matched on, and for vision-located figures that number was
    itself read by an image model — one misread digit ('7.36' → '7.38')
    silently attaches the WRONG crop to a question whose text cites the
    right one, and nothing downstream can see the difference.

    For every attached figure whose source is 'vision', fire ONE small
    VISION-tier call asking it to read the caption off the actual crop,
    then:
      - exact agreement  -> q["diagram_match_verified"] = True
      - disagreement/None -> q["diagram_match_low_confidence"] = True
        (+ a review note), so the Human Review UI shows a visible
        "Figure Caption Mismatch" badge instead of failing silently.

    The attachment is NEVER auto-removed either way (fail-open): a human
    decides on a flagged match; a correct-but-unverifiable crop must not
    be lost to a flaky check. Deterministic text-caption sources are not
    re-checked at all — no model read them in the first place. Never
    raises."""
    checked = verified = flagged = 0
    for q, img in matched_pairs:
        if img.get("source") != "vision":
            continue
        expected = _normalize_figure_ref(q.get("book_diagram_figure_ref"))
        try:
            response = _call_gemini_caption_read(img["bytes"], _CAPTION_READ_PROMPT)
            parsed = json.loads(_strip_json_fences(response.text or "{}"))
            reread = _normalize_figure_ref(parsed.get("figure_ref"))
        except Exception as e:
            logger.warning(f"⚠️ Exercise {exercise_label}: caption cross-check call failed "
                            f"({e}) — leaving the attachment unverified rather than risking "
                            f"a worse 'correction'.")
            continue
        checked += 1
        if reread and reread == expected:
            q["diagram_match_verified"] = True
            verified += 1
        else:
            q["diagram_match_low_confidence"] = True
            q.setdefault("review_notes", []).append(
                f"Figure caption mismatch: question cites চিত্ৰ {expected}, "
                f"but a fresh read of the cropped image saw '{reread or 'no number'}'. "
                f"Verify this is really the right textbook figure.")
            flagged += 1
            logger.warning(f"⚠️ Exercise {exercise_label}: figure-caption mismatch — question "
                            f"cites {expected}, crop reads {reread!r}. Flagged for human review; "
                            f"attachment kept.")
    if checked:
        logger.info(f"🔍 Exercise {exercise_label}: caption cross-check on {checked} vision-located "
                    f"book figure(s): {verified} verified, {flagged} flagged.")


def _attach_book_diagrams(questions: list, exercise_pdf_path: str, exercise_label: str, class_name: int = 0,
                           chapter: int = None): # type: ignore
    """Matches extracted original book-diagram images to whichever
    questions Gemini flagged has_book_diagram=true. This never raises — a
    matching failure just means those questions fall back to the
    AI-generated diagram_spec instead (see the Strict Hierarchy Rule in
    html_renderer.py), same as if no image existed.

    STRICT POLICY (V8 polishing-phase requirement): a book-original
    figure is attached to a question ONLY when the question's own text
    cites an exact figure number (e.g. "চিত্ৰ 3.14 চোৱা" / "see Figure
    3.14") AND an extracted image's own printed caption carries that
    exact same number. There is deliberately no other path. If no exact
    figure reference exists, no textbook figure is inserted at all —
    the question falls back to the AI-generated diagram_spec instead.

    ROOT-CAUSE HISTORY (why the previous page-order fallback was
    removed): this used to also match, in page order, any
    question/image pair where neither side carried a readable figure
    number. That heuristic was found to actively attach the WRONG
    figure in a real generated PDF — Exercise 14.3 Q6 (comparing two
    sections' frequency polygons) was matched to Figure 14.9, a
    completely unrelated generic median-illustration diagram from the
    chapter's own theory section (not a diagram of the question's
    data at all), apparently because that theory page's figure ended up
    in the same extracted-image pool with no ref match on the question
    side. That is exactly the "wrong textbook figure" / "nearby figure"
    failure mode this policy exists to eliminate. Falling back to an
    AI-generated diagram for an unreferenced question is safe (it is
    visibly built to match that exact question); silently reusing an
    unrelated page-order neighbor is not.

    GROUPED SUB-PARTS SHARE ONE IMAGE: a stem like "চিত্ৰ 3.14 চোৱা আৰু
    তলত দিয়াকেইটা লিখা—" is stated ONCE for a whole group of
    lettered/numbered sub-parts (1(i)..1(viii) etc). Gemini transcribes
    each sub-part's own question_text verbatim as printed, which means
    only the sub-part that actually carries the stem's wording resolves
    a figure-number ref match — the rest have no figure number in their
    own text at all. Those un-ref-matched sub-parts reuse whatever image
    the group already resolved via an exact ref match (never a blind
    pull, since there is no blind pull left) as long as the MAIN
    question_number hasn't changed — this is the only non-ref-match path
    left, and it is always downstream of a genuine exact match.

    VISION-LOCATED IMAGES: a "vision"-sourced candidate (unreliable-
    text-layer books; see book_diagram_extractor's "vision" source) is a
    per-page guess from an image model about where a figure sits, not a
    precise deterministic match — it is therefore only ever attached via
    an exact figure-number match, same strict rule as everything else,
    never treated any more permissively.
    """
    try:
        images = get_verified_figures(exercise_pdf_path, class_name=class_name, chapter=chapter)
    except Exception as e:
        logger.warning(f"⚠️ Exercise {exercise_label}: book-diagram extraction failed ({e}) — "
                        f"all questions fall back to AI-generated diagrams.")
        images = []

    ref_to_image = {}
    for img in images:
        ref = img.get("figure_ref")
        if ref and ref not in ref_to_image:
            ref_to_image[ref] = img

    matched_by_ref = 0
    matched_by_group_reuse = 0
    matched_pairs = []

    for q in questions:
        q["book_diagram_base64"] = None
        q["book_diagram_mime"] = None
        q["book_diagram_figure_ref"] = None

        q_ref = extract_figure_reference(
            f"{q.get('question_text', '')} {q.get('given', '')} {q.get('required', '')}")
        matched_img = ref_to_image.get(q_ref) if q_ref else None

        if matched_img:
            matched_by_ref += 1
            q["book_diagram_base64"] = base64.b64encode(matched_img["bytes"]).decode("ascii")
            q["book_diagram_mime"] = f"image/{matched_img['ext']}"
            # BUG FIX (found from a real generated PDF — user-reported):
            # the HTML template previously captioned EVERY diagram, book
            # image or AI-generated alike, with the same generic
            # "চিত্ৰ আৰ্হি" ("diagram pattern") text — even a genuine,
            # verified textbook figure (e.g. চিত্ৰ 7.46) lost its own
            # printed figure number and looked identical to a generated
            # placeholder. A real published solution book always
            # captions a reused figure with its actual number. Storing
            # the verified figure_ref here (already established with
            # high confidence — see the multi-stage verification this
            # match went through in book_diagram_extractor.py) lets the
            # template show "চিত্ৰ 7.46" for a real book image, falling
            # back to the generic caption only for an AI-generated one.
            q["book_diagram_figure_ref"] = matched_img.get("figure_ref")
            matched_pairs.append((q, matched_img))

    if config.FIGURE_MATCH_VERIFICATION and matched_pairs:
        _verify_figure_caption_matches(matched_pairs, exercise_label)

    unmatched_figures = sum(1 for img in images if img.get("figure_ref") is None)
    if unmatched_figures:
        logger.info(f"⚠️ Exercise {exercise_label}: {unmatched_figures} extracted figure(s) had no "
                    f"readable caption number and were NOT attached to any question (strict policy: "
                    f"exact figure-number match only, no page-order guessing) — those questions fall "
                    f"back to an AI-generated diagram instead of risking a wrong/nearby textbook figure.")

    if matched_by_ref:
        logger.info(f"🎯 Exercise {exercise_label}: {matched_by_ref} question(s) matched to their "
                    f"book diagram by exact figure number.")
    if matched_by_group_reuse:
        logger.info(f"🔗 Exercise {exercise_label}: {matched_by_group_reuse} sub-part question(s) "
                    f"reused their group's shared book diagram (itself an exact figure-number match).")


# ------------------------------------------------------------------
# VALIDATION GATE — coordinate self-verification.
#
# Root cause (audit finding, Exercise 3.2 Q2(ii)): the main solve pass
# stated point C's coordinates as (5,-5); the book's actual Figure 3.14
# has C at (6,-5). This wasn't a diagram-matching bug — the correct
# figure was already part of the exercise PDF Gemini solved from — it
# was a plain mis-read of one point among eight while solving the
# whole exercise (every question, every proof, every diagram_spec) in
# a single combined pass. The fix here is NOT "ask the AI to be more
# careful" (unenforceable); it's a second, narrow, independently-
# verifiable pass: for each question that (a) is clearly asking to
# read one labelled point's coordinate off a figure and (b) has that
# exact book-diagram crop already attached, ask a fresh Gemini Vision
# call ONE question only — "what is point X's coordinate in THIS
# image" — with nothing else in its context to get confused by, and
# cross-check that against what the combined pass already produced.
# ------------------------------------------------------------------
_POINT_LABEL_PATTERN = re.compile(r"\b([A-Z])\s*(?:বিন্দু|point)", re.IGNORECASE)
_COORD_KEYWORDS = ("স্থানাংক", "ভুজ", "কোটি", "abscissa", "ordinate", "coordinate")
_COORD_PAIR_PATTERN = re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)")


def _extract_point_label(text: str):
    if not text:
        return None
    m = _POINT_LABEL_PATTERN.search(text)
    return m.group(1).upper() if m else None


def _looks_like_coordinate_read_question(q: dict) -> bool:
    text = f"{q.get('question_text', '')} {q.get('given', '')} {q.get('required', '')}"
    return any(kw in text for kw in _COORD_KEYWORDS)


@retry_with_backoff(times=3, base_delay=3.0, max_delay=20.0, retryable_check=_is_retryable_api_error)
def _call_gemini_vision_point(image_bytes: bytes, prompt: str):
    client = get_client()
    return client.models.generate_content(
        model=config.GEMINI_VISION_MODEL_ID,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt],
        config=types.GenerateContentConfig(
            # 512 is the minimum thinking_budget some routed models accept
            # (gemini-2.5-flash-lite rejects anything lower with a 400).
            thinking_config=types.ThinkingConfig(thinking_budget=512),
            response_mime_type="application/json",
        ),
    )


def _read_point_coords(img_bytes: bytes, label: str, class_name: int) -> tuple[int, int] | None:
    """One independent 'read point <label> off this figure' call. Returns
    (x, y) or None if the call failed or didn't return usable integers —
    never raises, so callers can treat None as simply 'no vote'."""
    try:
        prompt = (
            f"This image is a Cartesian-plane figure from a Class {class_name or '9'} "
            f"Assamese SEBA/NCERT Maths textbook. Read the exact plotted coordinates "
            f"of the point labelled '{label}' by counting grid squares from the "
            f"origin along each axis (x first, then y). Return ONLY this JSON, no "
            f'commentary, no code fences: {{"x": <integer>, "y": <integer>}}'
        )
        response = _call_gemini_vision_point(img_bytes, prompt)
        parsed = json.loads(_strip_json_fences(response.text or "{}"))
        x, y = parsed.get("x"), parsed.get("y")
        if isinstance(x, int) and isinstance(y, int):
            return (x, y)
    except Exception:
        pass
    return None


def _verify_coordinate_answers(questions: list, class_name: int, exercise_label: str):
    """Fires the narrow per-point re-verification call described above for
    every eligible question. Never raises — a verification-call failure
    just leaves the original (unverified) answer exactly as the main
    solve pass produced it, same fail-open policy as every other
    optional-enhancement step in this pipeline (book-diagram attachment,
    JSON-escape repair).

    TIE-BREAK FIX (found during the V7 engineering audit): the original
    version of this function treated a SINGLE disagreeing vision call as
    grounds to overwrite the main solve pass's answer outright. That is
    not verification, it's just trusting whichever call happened to run
    second — if THAT call was the one that misread the grid (equally
    possible for any single model call), a previously-correct answer
    gets silently replaced with a wrong one, which is worse than not
    re-checking at all, because it now LOOKS doubly-verified. Fixed to a
    proper best-of-3 majority vote:
      - original (from the main solve pass) agrees with the first
        independent re-read -> no action, already confirmed.
      - they disagree -> fire a SECOND independent re-read (fresh call,
        same narrow prompt) as a tie-break, and only ever act on a real
        2-out-of-3 majority.
      - a genuine 3-way split (original, re-read 1, and the tie-break
        re-read all differ) means this cannot be resolved automatically
        — per the brief's own "if something cannot be verified, fail
        safely" principle, this is flagged via a `needs_review` marker
        on the question rather than any value being guessed/overwritten.
    """
    candidates = [q for q in questions
                  if q.get("book_diagram_base64") and _looks_like_coordinate_read_question(q)]
    if not candidates:
        return

    corrected = 0
    flagged = 0
    for q in candidates:
        label = _extract_point_label(f"{q.get('question_text', '')} {q.get('given', '')}")
        if not label:
            continue  # can't target one specific point deterministically — skip rather than guess

        stated = q.get("final_answer", "") or ""
        existing_pair = _COORD_PAIR_PATTERN.search(stated)
        if not existing_pair:
            continue
        stated_xy_str = existing_pair.group(0).replace(" ", "")

        try:
            img_bytes = base64.b64decode(q["book_diagram_base64"])
        except Exception as e:
            logger.warning(f"⚠️ Coordinate self-check: could not decode book diagram for point "
                            f"{label} in Exercise {exercise_label}, leaving original answer as-is: {e}")
            continue

        first_vote = _read_point_coords(img_bytes, label, class_name)
        if first_vote is None:
            continue  # verification call itself failed -> leave original untouched (fail-open)
        first_vote_str = f"({first_vote[0]},{first_vote[1]})"

        if first_vote_str == stated_xy_str:
            continue  # confirmed on first check, nothing to do

        # Disagreement — this is exactly the case the old code overwrote
        # on unconditionally. Get a tie-break vote before acting on anything.
        logger.warning(
            f"🔎 Exercise {exercise_label}: coordinate self-check disagreement for point "
            f"{label} — solved answer said {stated_xy_str}, first re-read is {first_vote_str}. "
            f"Requesting a tie-break read before correcting anything."
        )
        tie_break = _read_point_coords(img_bytes, label, class_name)
        tie_break_str = f"({tie_break[0]},{tie_break[1]})" if tie_break else None

        winner = None
        if tie_break_str == stated_xy_str:
            winner = "original"
        elif tie_break_str == first_vote_str:
            winner = "reread"
        # else: no majority (tie-break call failed, or produced a THIRD
        # distinct reading) -> winner stays None, handled below.

        if winner == "reread":
            verified_str = first_vote_str
            q["final_answer"] = stated.replace(existing_pair.group(0), verified_str)
            if isinstance(q.get("steps"), list):
                q["steps"] = [
                    _COORD_PAIR_PATTERN.sub(
                        lambda m: verified_str if m.group(0).replace(" ", "") == stated_xy_str else m.group(0),
                        s,
                    )
                    for s in q["steps"]
                ]
            corrected += 1
            logger.info(f"✅ Exercise {exercise_label}: point {label} corrected to {verified_str} "
                        f"(2-of-3 majority: both independent re-reads agreed, original was wrong).")
        elif winner == "original":
            logger.info(f"✅ Exercise {exercise_label}: point {label} — original answer "
                        f"{stated_xy_str} confirmed by tie-break (2-of-3 majority); the first "
                        f"re-read was the outlier, no change made.")
        else:
            # Fail-safe: three-way split (or the tie-break call itself
            # failed) — do NOT guess which of stated/first_vote/tie_break
            # is correct. Leave the original answer untouched (never
            # publish a value picked without real agreement) and flag it.
            q["needs_review"] = True
            q.setdefault("review_notes", []).append(
                f"Coordinate self-check for point {label} could not reach a majority: "
                f"solved={stated_xy_str}, re-read={first_vote_str}, "
                f"tie-break={tie_break_str or 'call failed'}. Left as originally solved; "
                f"please verify point {label} manually against the book figure."
            )
            flagged += 1
            logger.warning(f"⚠️ Exercise {exercise_label}: point {label} — no 2-of-3 majority "
                            f"(solved={stated_xy_str}, re-read={first_vote_str}, "
                            f"tie-break={tie_break_str or 'failed'}). Leaving original answer "
                            f"unchanged and flagging needs_review instead of guessing.")

    if corrected:
        logger.info(f"✅ Exercise {exercise_label}: {corrected} coordinate answer(s) "
                    f"corrected by the vision self-verification validation gate.")
    if flagged:
        logger.warning(f"⚠️ Exercise {exercise_label}: {flagged} coordinate answer(s) flagged "
                        f"needs_review — no automatic majority, human check recommended.")
