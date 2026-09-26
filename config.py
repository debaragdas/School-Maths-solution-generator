"""
config.py — everything you change per run lives here.

Nothing else in the project needs editing to process a different
book/class/chapter. Secrets (GCP project etc.) come from environment
variables, never hardcoded.
"""
import os

# ------------------------------------------------------------------
# WHAT TO PROCESS — the only 3 things you normally touch
# ------------------------------------------------------------------
BOOK_URL = "https://site.sebaonline.org/textbooks/class-9/Maths_Ganit_Assamese_IX.pdf"
CLASS = 9
CHAPTER = 10

# The exercise(s) to solve when SOLVE_SINGLE_EXERCISE (below) is True.
# One exercise: EXERCISE_FILTER = ["7.4"]
# A few:        EXERCISE_FILTER = ["7.1", "7.4"]
# Ignored completely when SOLVE_SINGLE_EXERCISE is False — safe to
# leave a value here even when not in use.
EXERCISE_FILTER = ["7.5"]

# True  -> solve ONLY the exercise(s) named in EXERCISE_FILTER above.
# False -> ignore EXERCISE_FILTER and solve the WHOLE chapter (default).
SOLVE_SINGLE_EXERCISE = False

# Book's own language for chapter/exercise headings, used by
# chapter_detector.py / exercise_splitter.py regexes.
# "assamese" or "english" — SEBA books are usually Assamese-medium
# with English "Exercise" headings mixed in, so both patterns are
# always tried regardless of this setting; this only picks which
# ordinal-word table to check for "Chapter <ordinal>" style headings.
BOOK_LANGUAGE = "assamese"

# Set to True if this chapter is a "Constructions" (অংকন) chapter that requires
# step-by-step drawing instructions with ruler/compass. Set to False for
# regular geometry chapters (Triangles, Circles, etc.) that need mathematical
# solutions and proofs, not construction instructions.
IS_CONSTRUCTION_CHAPTER = False

# Only set this True if a specific book URL's own SSL certificate is
# genuinely misconfigured (self-signed / expired) and you've confirmed
# that yourself — leave False otherwise. This is a real security
# downgrade (no protection against a tampered/substituted download), so
# it's an explicit opt-in per run rather than a silent default.
ALLOW_INSECURE_SSL = True

# ------------------------------------------------------------------
# GOOGLE CLOUD / VERTEX AI — ADC only, no API keys
# ------------------------------------------------------------------
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "assamstudyai")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")

# MODEL ROUTING — one model per CALL-SITE cost/accuracy profile, not one
# model for everything. The main solve call is where mathematical
# correctness is decided; the auxiliary vision/scan calls are simple,
# high-volume, and 3-12x cheaper on cheaper models:
#   SOLVE  — the big per-exercise solve + question corrections + diagram
#            recovery (reasoning quality decides answer correctness).
#   VISION — image-understanding calls that need real accuracy: book-
#            figure bounding boxes, point-coordinate cross-checks.
#   SCAN   — cheap whole-book page scans (chapter/exercise headings,
#            chapter title): simple visual search, highest volume.
# Setting the legacy GEMINI_MODEL_ID env var pins ALL THREE at once;
# a role-specific env var (e.g. GEMINI_SOLVE_MODEL_ID) wins for its role.
# LIVE-VERIFIED against this project (project=assamstudyai,
# us-central1) on 2026-08-26 by probe_models.py / verify_models_live.py:
# ONLY the 2.5 family is currently published to this project/region —
# gemini-3-* returns 404 here. When Google publishes 3.x to this
# project, override via env vars WITHOUT touching this file, e.g.
#   set GEMINI_SOLVE_MODEL_ID=gemini-3-flash
# The split still cuts cost meaningfully: the two CHEAP tiers carry the
# highest-volume calls (whole-book page scans, figure bounding boxes),
# while every correctness-critical call stays on Pro — the same model
# that produced every chapter published so far.
_MODEL_ENV_BASE = os.environ.get("GEMINI_MODEL_ID", "")
GEMINI_SOLVE_MODEL_ID = os.environ.get("GEMINI_SOLVE_MODEL_ID") or _MODEL_ENV_BASE or "gemini-2.5-pro"
GEMINI_VISION_MODEL_ID = os.environ.get("GEMINI_VISION_MODEL_ID") or _MODEL_ENV_BASE or "gemini-2.5-flash"
GEMINI_SCAN_MODEL_ID = os.environ.get("GEMINI_SCAN_MODEL_ID") or _MODEL_ENV_BASE or "gemini-2.5-flash-lite"
# Back-compat alias (anything still reading the single knob gets the
# solve model, which is what GEMINI_MODEL_ID used to drive):
GEMINI_MODEL_ID = GEMINI_SOLVE_MODEL_ID

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
TEMP_DIR = "./temp"
OUTPUT_DIR = "./output"

# ------------------------------------------------------------------
# HUMAN REVIEW — interactive per-question review (see interactive_review.py)
# ------------------------------------------------------------------
# "all"                : every question pauses for review (the default,
#                         and the brief's primary requirement).
# "flagged_or_diagram"  : only questions the pipeline itself flagged
#                         (low self-verification confidence) or that
#                         carry any diagram/figure pause — a lighter
#                         mode for teams that trust plain-arithmetic
#                         answers by default. Any other value is
#                         treated the same as "all" (fails safe towards
#                         reviewing MORE, not less).
INTERACTIVE_REVIEW_SCOPE = "all"

# "web" (default): main.py auto-launches the local browser Human Review
#                   app (review_webapp.py) for Stage 2 — one page per
#                   exercise, side-by-side solution/diagram/figure
#                   previews, no terminal prompts.
# "cli"           : the original console workflow (interactive_review.py)
#                   — kept for headless machines / no-GUI servers where
#                   auto-opening a browser isn't possible.
REVIEW_MODE = "web"
REVIEW_HOST = "127.0.0.1"   # local-only — never bind 0.0.0.0 for this
REVIEW_PORT = 5151

# ------------------------------------------------------------------
# TUNING
# ------------------------------------------------------------------
# Figure-match caption cross-check: after a question is matched to its
# original book diagram by EXACT figure number, figures whose caption
# number was read by Gemini Vision (unreliable-text-layer books) get ONE
# cheap follow-up Vision call asking it to re-read that caption off the
# actual crop. Agreement marks the attachment VERIFIED; any disagreement
# keeps the attachment but flags the question for the human reviewer
# ("Figure Caption Mismatch") instead of silently shipping a possibly
# wrong crop. Deterministic text-layer caption reads are never re-checked
# (no model involved there in the first place). Set env
# FIGURE_MATCH_VERIFICATION=0 to disable (e.g. quota-constrained runs).
FIGURE_MATCH_VERIFICATION = os.environ.get("FIGURE_MATCH_VERIFICATION", "1").strip().lower() not in ("0", "false", "no")

MAX_RETRIES_PER_EXERCISE = 1        # per the spec: retry once, no more
GEMINI_TIMEOUT_SECONDS = 180
PARALLEL_WORKERS = 3                # concurrent exercises in flight
THINKING_BUDGET = 4000              # bounded — avoids MAX_TOKENS truncation

# ROOT-CAUSE FIX for "429 RESOURCE_EXHAUSTED" bursts: PARALLEL_WORKERS
# exercises run concurrently, and EACH exercise's own pipeline can fire
# off several independent Vertex AI calls back-to-back (the main solve
# call, a vision figure-location batch, per-question diagram/answer
# recovery calls) — every one of those 5 call sites (solver.py,
# vision_ocr.py, correction_engine.py, diagram_safety_net.py) retries
# its OWN 429s individually, but nothing previously coordinated how
# many calls were in flight ACROSS the whole process at once. With 2-3
# exercises running in parallel, each cascading into several calls of
# its own, it's easy to burst well past Vertex AI's per-minute quota
# even seconds after a full daily-quota reset — a QPM (queries-per-
# minute) limit, not a total-quota one, so waiting longer between runs
# never helps. utils.retry_with_backoff now enforces this single
# process-wide cap before every one of those 5 call sites' attempts,
# so raising PARALLEL_WORKERS can never re-introduce this burst.
VERTEX_AI_MAX_CALLS_PER_MINUTE = 8

# ------------------------------------------------------------------
# BRANDING (must stay identical across every generated PDF)
# ------------------------------------------------------------------
BRAND_TITLE = "ASSAM STUDY AI | EXECUTIVE EDITION"
WATERMARK_TEXT = "AssamStudyAI"
FOOTER_TEXT = "© 2026 AssamStudyAI. All rights reserved."
