"""
html_renderer.py — fills the fixed Jinja2 template with a solved
exercise's content. This is the only place the template and the solved
JSON meet; the LLM never sees or touches HTML.

Also inlines heavy static assets directly into the page so rendering
has zero filesystem-path or network dependency at PDF-generation time
(all read from disk once and cached in memory, not re-read per
exercise):
  - the vendored MathJax bundle (templates/vendor/mathjax/)
  - the premium Assamese font, base64-embedded into the CSS
    (templates/fonts/) — this guarantees correct glyph rendering
    regardless of what fonts happen to be installed on the machine
    running the pipeline, which is far more portable than relying on a
    system font lookup.
  - the brand logo (templates/images/logo.png), used as the page
    watermark instead of the old text-based one.
"""
import io
import os
import base64
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from diagram_renderer import render_diagram
from diagram_final_check import final_pre_pdf_check
import book_intelligence
import math_sanitizer
from utils import logger

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

_mathjax_js_cache = None
_css_cache = None

FONTS_DIR = os.path.join(TEMPLATE_DIR, "fonts")
IMAGES_DIR = os.path.join(TEMPLATE_DIR, "images")

_FONT_FILES = {
    "Tiro Bangla Regular": ("TiroBangla-Regular.ttf", "@@TIRO_BANGLA_REGULAR_BASE64@@"),
}
LOGO_FILE = "logo.png"

# --------------------------------------------------------------------
# WATERMARK TILE — the logo must be tiled once per printed page WITHOUT
# ever stretching it off its own aspect ratio. CSS's background-size
# can only tile at the exact pixel box you give it, so we pre-render a
# transparent PNG "tile" that is already exactly one page tall, with
# the logo drawn undistorted (its own aspect ratio) and centered inside
# that tile. Tiling THAT image (background-repeat: repeat-y) then lands
# one correctly-proportioned logo per page — no per-page CSS stretch.
# --------------------------------------------------------------------
WATERMARK_DISPLAY_WIDTH_PX = 240  # logical CSS px width of the undistorted logo
_MM_TO_CSS_PX = 96 / 25.4
# Must match pdf_generator.py's page margins: A4 (297mm) minus the 18mm
# top / 16mm bottom print margins = 263mm of printable height per page.
_PAGE_PRINTABLE_HEIGHT_MM = 297 - 18 - 16
WATERMARK_TILE_HEIGHT_PX = round(_PAGE_PRINTABLE_HEIGHT_MM * _MM_TO_CSS_PX)
_WATERMARK_OVERSAMPLE = 3  # raster the tile at 3x, downscaled via CSS, for crisp print output

_WATERMARK_TILE_TOKEN = "@@WATERMARK_TILE_BASE64@@"
_WATERMARK_TILE_W_TOKEN = "@@WATERMARK_TILE_WIDTH_PX@@"
_WATERMARK_TILE_H_TOKEN = "@@WATERMARK_TILE_HEIGHT_PX@@"


def _build_watermark_tile(logo_path: str) -> bytes:
    """Returns PNG bytes for one page-height-tall watermark tile: the logo
    drawn at its own aspect ratio (never stretched), centered inside a
    transparent canvas sized to exactly one printed page's height."""
    logo = Image.open(logo_path).convert("RGBA")

    display_w = WATERMARK_DISPLAY_WIDTH_PX * _WATERMARK_OVERSAMPLE
    display_h = round(display_w * logo.height / logo.width)  # aspect ratio preserved
    logo_resized = logo.resize((display_w, display_h), Image.LANCZOS)

    tile_w = WATERMARK_DISPLAY_WIDTH_PX * _WATERMARK_OVERSAMPLE
    tile_h = WATERMARK_TILE_HEIGHT_PX * _WATERMARK_OVERSAMPLE
    tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))

    paste_x = (tile_w - display_w) // 2
    paste_y = (tile_h - display_h) // 2  # vertically centered within the page band
    tile.paste(logo_resized, (paste_x, paste_y), logo_resized)

    buf = io.BytesIO()
    tile.save(buf, format="PNG")
    return buf.getvalue()


def _get_mathjax_js() -> str:
    global _mathjax_js_cache
    if _mathjax_js_cache is None:
        mathjax_path = os.path.join(TEMPLATE_DIR, "vendor", "mathjax", "tex-svg.js")
        if not os.path.exists(mathjax_path):
            raise FileNotFoundError(
                f"MathJax bundle missing at {mathjax_path}. Run "
                f"'npm pack mathjax@3' and copy package/es5/tex-svg.js there "
                f"(SVG output — no external font/network dependency, unlike "
                f"the CHTML bundle)."
            )
        with open(mathjax_path, "r", encoding="utf-8") as f:
            _mathjax_js_cache = f.read()
    return _mathjax_js_cache


def _get_css() -> str:
    """Reads style.css once and substitutes the font's and logo's base64
    data in for their placeholder tokens. Plain str.replace(), not
    str.format() — CSS is nothing but curly braces, so .format() would
    crash exactly the same way prompts.py's original bug did.

    IMPORTANT: fonts embedded here must be STATIC files, never variable
    fonts. Chromium's headless print pipeline has a real, reproducible
    complex-script shaping bug with variable Bengali fonts — conjuncts
    silently split or drop a letter. If you swap in a different font,
    instance it to a fixed weight first (fontTools' varLib.instancer)
    rather than embedding the variable file directly.
    """
    global _css_cache
    if _css_cache is None:
        with open(os.path.join(TEMPLATE_DIR, "style.css"), "r", encoding="utf-8") as f:
            css = f.read()

        for label, (filename, token) in _FONT_FILES.items():
            path = os.path.join(FONTS_DIR, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Font missing: {label} expected at {path}. Place the file "
                    f"there (or update _FONT_FILES in html_renderer.py to point "
                    f"at your own font's filename)."
                )
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            css = css.replace(token, f"url(data:font/ttf;base64,{b64})")

        logo_path = os.path.join(IMAGES_DIR, LOGO_FILE)
        if not os.path.exists(logo_path):
            raise FileNotFoundError(
                f"Logo missing at {logo_path}. Place your watermark image there "
                f"(or update LOGO_FILE in html_renderer.py)."
            )
        tile_bytes = _build_watermark_tile(logo_path)
        tile_b64 = base64.b64encode(tile_bytes).decode("ascii")
        css = css.replace(_WATERMARK_TILE_TOKEN, tile_b64)
        css = css.replace(_WATERMARK_TILE_W_TOKEN, str(WATERMARK_DISPLAY_WIDTH_PX))
        css = css.replace(_WATERMARK_TILE_H_TOKEN, str(WATERMARK_TILE_HEIGHT_PX))

        _css_cache = css
    return _css_cache


_ASCII_TO_ASSAMESE_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

# Diagram types whose renderer uses a wider/data-driven canvas (a full
# coordinate plane with grid + axes, or a chart with several category
# labels/a legend) rather than a compact fixed-size figure. These need
# more room than the standard 210px sidebar box — squeezing them into
# it shrinks their text proportionally and reproduces exactly the
# "overlapping/illegible axis or category labels" failure mode, even
# though the renderer itself already lays the labels out correctly at
# its own native canvas size.
_LARGE_DIAGRAM_TYPES = {"statistics", "coordinate_plot"}


def _resolve_diagram_is_large(q: dict, final: dict, diagram_spec) -> bool:
    """Whether THIS question's diagram (book figure OR AI-generated SVG,
    whichever `final` ended up publishing) uses the wide, centered,
    stacked-below-text layout instead of the fixed 210px sidebar box.

    Two independent inputs, reviewer override always wins:
      1. `q["diagram_position"]` — a manual choice made in the Human
         Review UI (correction_engine.set_diagram_position): "large"
         forces it, "side" forces the compact box, "auto"/None/missing
         (the default for every question, including every book-scanned
         figure) falls through to #2. This is the ONLY way a book
         figure (final["book_diagram_base64"]) can ever get the large
         layout — unlike an AI-generated diagram, a scanned book figure
         has no `diagram_type` to auto-detect "large" from, so without
         an explicit reviewer choice it would otherwise always render
         at the small fixed size regardless of how much fine detail
         (axis labels, a multi-row data table, several category ticks)
         it actually contains — exactly the "diagram unreadable on a
         phone screen" report this exists to fix.
      2. The original auto heuristic: an AI-generated SVG (never a book
         figure) whose diagram_spec.diagram_type is one of
         _LARGE_DIAGRAM_TYPES (statistics chart / coordinate plane).
    """
    override = q.get("diagram_position")
    if override == "large":
        return True
    if override == "side":
        return False
    return bool(final["diagram_svg"]) and isinstance(diagram_spec, dict) \
        and diagram_spec.get("diagram_type") in _LARGE_DIAGRAM_TYPES


def _to_assamese_numeral(n: int) -> str:
    return str(n).translate(_ASCII_TO_ASSAMESE_DIGITS)


def render_exercise_html(solved: dict, class_name: int, chapter: int,
                          chapter_name: str, exercise_label: str) -> str:
    questions = []
    for q in solved.get("questions", []):
        # MATH SANITIZATION LAYER — the mandatory final gateway (see
        # math_sanitizer.py's module docstring). solver.py already runs
        # every field through this at solve-time, but re-running it
        # here, right before the HTML template ever sees the data,
        # makes it a guarantee independent of upstream source: a Human
        # Review correction, a field set directly by correction_engine,
        # or any future code path that builds a question dict without
        # going through solver.py's own normalization still gets the
        # same protection. Sanitizing already-clean text is a safe,
        # cheap no-op (see test_math_sanitizer.TestIdempotence).
        for _field in ("question_text", "given", "required", "final_answer"):
            if _field in q:
                q[_field] = math_sanitizer.sanitize_math_text(q[_field])
        if isinstance(q.get("steps"), list):
            q["steps"] = [math_sanitizer.sanitize_math_text(s) for s in q["steps"]]

        diagram_spec = q.get("diagram_spec")
        diagram_svg = render_diagram(diagram_spec)

        # FINAL VALIDATION (diagram_final_check.py) — runs AFTER
        # rendering, right before this question's HTML is built. Never
        # trust diagram_svg / book_diagram_base64 directly past this
        # point: `final` below is the reconciled, provably-checked
        # version of both, which may have cleared either one if a
        # genuine problem was found (missing render, corrupt/blank book
        # crop, overlapping labels, clipped content, or a decision the
        # actual output contradicts).
        final = final_pre_pdf_check(q, diagram_svg)

        # BOOK INTELLIGENCE LAYER (book_intelligence.py) — self_verify()
        # runs strictly AFTER the real decision above and NEVER changes
        # it; this is a confidence-scoring/QA-reporting pass only (see
        # book_intelligence.py's own module docstring for the full scope
        # boundary). Wrapped defensively: a bug in this reporting layer
        # must never be able to break page rendering for anyone.
        try:
            verify_report = book_intelligence.self_verify(q, final["final_decision"], final["diagram_svg"])
            if verify_report["confidence"]["needs_review"]:
                logger.info(f"ℹ️ Q{q.get('question_number')}{q.get('sub_part', '')}: "
                            f"self-verification confidence {verify_report['confidence']['score']:.0f}/100 "
                            f"— flagged for review ({'; '.join(verify_report['conflicts']) or 'below threshold'}).")
        except Exception as e:
            logger.warning(f"⚠️ Q{q.get('question_number')}: self_verify itself failed ({e}) — "
                            f"this question's diagram is still published normally; only the "
                            f"confidence report was skipped.")

        questions.append({
            **q,
            "diagram_svg": final["diagram_svg"],
            "book_diagram_base64": final["book_diagram_base64"],
            "book_diagram_mime": final["book_diagram_mime"],
            "book_diagram_figure_ref": final["book_diagram_figure_ref"],
            # Overwrite (not just pass through) the audit-trail fields with
            # the FINAL check's own verdict — if final_pre_pdf_check had to
            # downgrade a diagram, the exposed diagram_decision/reason must
            # reflect that true outcome, not the pre-final-check verdict
            # decide_diagram() originally produced. Otherwise inspecting
            # this question's own audit fields after the fact would show a
            # decision that no longer matches what was actually published.
            "diagram_decision": final["final_decision"],
            "diagram_decision_reason": final["final_reason"],
            "diagram_is_large": _resolve_diagram_is_large(q, final, diagram_spec),
        })

    template = _env.get_template("base.html.jinja")
    return template.render(
        css=_get_css(),
        mathjax_js=_get_mathjax_js(),
        class_name=class_name,
        chapter=chapter,
        chapter_assamese=_to_assamese_numeral(chapter),
        chapter_name=chapter_name,
        exercise_label=exercise_label,
        questions=questions,
    )
