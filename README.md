# AssamStudyAI — Math Solution Factory

Generates professional Assamese-language Mathematics exercise-solution
PDFs for one full chapter, automatically — properly typeset math
(MathJax), enforced প্ৰদত্ত/প্ৰয়োজনীয়/সমাধান proof structure, and
auto-generated geometry diagrams.

## Quick start (2026-08 upgrade)

```bash
# Whole chapter, class from config.py:
python main.py --class 9 --chapter 7

# Only some exercises:
python main.py --chapter 7 --exercise 7.4          # one
python main.py --chapter 10 --exercise 10.1 10.2   # several

# A constructions (অংকন) chapter:
python main.py --class 10 --chapter 11 --construction

# Custom book PDF (cached after first download):
python main.py --chapter 3 --book-url https://example.com/book.pdf

# See every registered class 6-10 textbook:
python main.py --list-books
```

All flags are optional; with none, config.py's values are used exactly
as before. Textbook URLs live in `books.py` (classes 9 & 10 verified
live; SCERT classes 6-8 publish chapter-wise only, so pass `--book-url`
once for those until a stable full-book URL exists).

### Model routing (cost)

One model per call-site role (`config.py` MODEL ROUTING), **live-verified
against this project on 2026-08-26** (`verify_models_live.py`):

| Role | Default | Price/1M (in/out) |
|---|---|---|
| Solve + corrections + diagram recovery | `gemini-2.5-pro` | $1.25 / $10.00 |
| Figure bboxes + point-coordinate checks | `gemini-2.5-flash` | $0.30 / $2.50 |
| Heading/title page scans | `gemini-2.5-flash-lite` | $0.10 / $0.40 |

Every correctness-critical call (math solving, Assamese prose,
diagram specs) stays on Pro — the model that produced every chapter
published so far. The cheap tiers carry only high-volume structural
work (page scans, bounding boxes). Override any tier via env vars
`GEMINI_SOLVE_MODEL_ID`, `GEMINI_VISION_MODEL_ID`,
`GEMINI_SCAN_MODEL_ID` (e.g. when Google publishes gemini-3.x to your
region); setting legacy `GEMINI_MODEL_ID` pins all three. Note:
thinking budgets below 512 are rejected by flash-lite — keep routed
calls at >=512.

### Book-figure caption cross-check

Every book figure whose caption number was read by Gemini Vision gets
one cheap follow-up Vision call re-reading that caption off the actual
crop. Agreement marks the attachment **verified**; disagreement keeps
the attachment but flags it in Human Review ("Figure Caption Mismatch")
so a wrong crop can never ship silently. Disable with env
`FIGURE_MATCH_VERIFICATION=0`.

### Concurrency on Windows

File locking is cross-platform (`utils.exclusive_file_lock`: fcntl on
POSIX, msvcrt byte-range locks on Windows), so `PARALLEL_WORKERS > 1`
is safe on Windows too — previously the locks silently vanished there
and racing workers could lose figures or review actions.

## Setup (one-time)

```bash
pip install -r requirements.txt
playwright install chromium

# ADC auth — no API keys anywhere in this project
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
```

MathJax is already vendored at `templates/vendor/mathjax/tex-svg.js`
(SVG output — draws math as vector paths, so there's no external font
or network dependency at PDF-generation time). Nothing to download.

**Font:** `templates/fonts/TiroBangla-Regular.ttf` is bundled and gets
base64-embedded directly into the CSS at render time
(`html_renderer.py`) — used for the ENTIRE document (body, headers,
steps). Only a Regular weight was provided, so bold text (headers,
labels) relies on the browser's synthesized bold rather than a true
bold face — if you get a real bold file, add it to `_FONT_FILES` in
`html_renderer.py` and reference it in `style.css`'s `@font-face` with
`font-weight: 700`. To swap fonts entirely, replace the `.ttf` (keep
the filename, or update `_FONT_FILES`) — **use static weight files,
not variable fonts**: Chromium's print pipeline has a real,
reproducible complex-script shaping bug with variable Bengali fonts
(conjuncts silently split or drop a letter). If you only have a
variable font, instance it first with fontTools:
```python
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
font = TTFont("YourFont-Variable.ttf")
instantiateVariableFont(font, {"wght": 400}, inplace=True)
font.save("templates/fonts/YourFont-Regular.ttf")
```

**Watermark:** `templates/images/logo.png` is embedded the same way
and tiled down each page at exactly one printed-page-height interval
(`style.css`'s `.watermark-layer`, sized to match the A4 print area —
297mm minus the top/bottom margins set in `pdf_generator.py`) so one
faded, centered copy lands on every page. To swap logos, replace that
file (keep the filename, or update `LOGO_FILE` in `html_renderer.py`).

## Configure a run

Edit `config.py`:

```python
BOOK_URL = "https://..../class9_math.pdf"
CLASS = 9
CHAPTER = 7
```

## Run

```bash
python main.py
```

## Output

```
output/
  class_9/
    chapter_1/
      ex_1_1.pdf
      ex_1_2.pdf
      ex_1_3.pdf
      ex_1_4.pdf
```

Re-running the same command skips any exercise PDF that already exists
— delete a specific file to force it to regenerate.

## Human Review workflow (internal team only — never exposed to students)

**This is now the primary workflow — just run `python main.py`.** There
is no separate review step to remember: the pipeline solves every
exercise as before, then PAUSES and hands control to you, one question
at a time, right there in the terminal. A PDF is never finalized, and a
chapter is never published, until every one of its questions has been
explicitly accepted this way.

```
📝 Reviewing Exercise 7.4  —  output/class_9/chapter_7/ex_7_4.pdf

========================================================================
Q5: চিত্ৰ 4.12 চোৱা। ইয়াত...
------------------------------------------------------------------------
Final answer : ∠AOB = 60°।
Diagram      : BOOK_DIAGRAM (Figure 4.12)
========================================================================
[A]ccept  [R]egenerate w/ correction  [F]igure upload  [P]review PDF  [Q]uit review >
```

At every question, you can:

- **[A]ccept** — move on to the next question.
- **[R]egenerate** — type a correction prompt ("Step 3 calculation is
  incorrect. Recalculate from Step 2 only. Keep everything else
  unchanged."); only THAT question is re-solved and the PDF is
  re-rendered in place, then the same question is shown again for
  another decision — repeat until you accept it.
- **[F]igure upload** — the question cites a textbook figure
  ("চিত্ৰ 4.12") that wasn't auto-matched: point at a screenshot,
  optionally give a crop box (`x1,y1,x2,y2`), and a figure number. It's
  stored permanently — every future question citing Figure 4.12, in any
  exercise, in any future run, reuses this crop automatically, no
  repeated manual work.
- **[P]review** — opens the exercise's current draft PDF in your OS's
  default viewer so you can actually look at it before deciding.
- **[Q]uit** — pause the whole review session. Nothing is lost: every
  question already accepted (and every correction/figure already
  applied) is saved, so simply re-running `python main.py` resumes
  exactly where you left off, without re-asking about anything already
  handled and without wasting another Gemini call.

Once every question in every exercise for the chapter is accepted,
`main.py` automatically publishes — copies the approved PDFs into
`published/` — with no extra command needed.

By default EVERY question pauses for review
(`config.INTERACTIVE_REVIEW_SCOPE = "all"`). Set it to
`"flagged_or_diagram"` for a lighter mode that only pauses on questions
the pipeline itself flagged (low self-verification confidence) or that
carry a diagram/figure, auto-accepting the rest.

### `review_cli.py` — secondary, ad-hoc tool

Everything above runs through the exact same `review_state.py` /
`correction_engine.py` / `figure_database.py` modules `review_cli.py`
already used. It still exists for occasions where the interactive loop
isn't the right fit — checking status from another machine, bulk
listing, or scripting:

```bash
python review_cli.py list
python review_cli.py list --status pending_review --chapter 7
python review_cli.py show output/class_9/chapter_7/ex_7_4.pdf
python review_cli.py correct output/class_9/chapter_7/ex_7_4.pdf 5 "..."
python review_cli.py add-figure screenshot.png --figure 4.12 --class 9 --chapter 7 ...
python review_cli.py approve output/class_9/chapter_7/ex_7_4.pdf --reviewer Priya
python review_cli.py publish-check --class 9 --chapter 7
python review_cli.py publish --class 9 --chapter 7
```

See `review_state.py` (status/version/history/per-question-acceptance
persistence + the publish gate), `correction_engine.py` (targeted
single-question regeneration and figure attachment), and
`interactive_review.py` (the console loop `main.py` runs) for the
implementation.

## How it works

1. **Download** the book PDF (cached — skipped on re-run).
2. **Decide if the text layer can be trusted** (`vision_ocr.is_text_layer_reliable`)
   — some SEBA books embed Assamese glyphs with a custom font encoding
   that *looks* correct on screen but returns garbled Unicode from
   `fitz.get_text()`. This is checked once per book by sampling pages
   and measuring what fraction of extracted characters actually fall
   in the Assamese/Bengali Unicode block.
3. **Locate the chapter**:
   - If the text layer is reliable: PDF bookmarks, then a page-text
     regex scan for "Chapter N" / "অধ্যায় N" — pure Python, no AI.
   - If not (or the chapter isn't found that way): falls back to a
     **cached, one-time Gemini Vision scan** — page images are shown
     to Gemini, which visually locates chapter-heading pages. Cached
     to disk keyed by the book's content hash, so this cost is paid
     once per book, ever. **No manual page-number input, ever.**
   - The chapter's display title is looked up the same way (bookmark
     text, or the vision scan's own `chapter_title` field) — if
     neither is available it's simply omitted from the PDF header
     rather than showing an awkward placeholder.
4. **Split into exercises** — same reliable-text-first / vision-fallback
   pattern, for "Exercise X.Y" / "অনুশীলনী X.Y" headings.
5. **Solve** — exactly **one** Gemini call per exercise PDF
   (`prompts.py`). Gemini reads every question, solves all of them, and
   returns one JSON object — it never writes HTML, CSS, or SVG. The
   prompt strictly enforces:
   - MathJax delimiters only (`\( ... \)` inline, `$$ ... $$` display —
     never bare `$...$`), and `\frac{}{}` for every fraction.
   - The প্ৰদত্ত / প্ৰয়োজনীয় / সমাধান / চূড়ান্ত উত্তৰ structure for any
     proof or geometry question, with each proof step citing its rule
     (CPCT, SSS, etc.), matching standard SEBA/NCERT textbook style.
   - A non-null `diagram_spec` for any question that names or implies a
     geometric figure OR a number-line/spiral construction — not left
     to Gemini's discretion.
   - Correct JSON escaping for LaTeX backslashes (`\\frac`, not
     `\frac`) — and even so, `solver.py` runs a repair pass
     (`_repair_invalid_json_escapes`) on any response that fails to
     parse, since models occasionally still emit a single backslash
     despite the instruction. This alone recovers the large majority of
     `Invalid \escape` failures without spending a second Gemini call.
6. **Verify** — cheap deterministic checks (`utils.verify_solution`):
   no missing questions, no numbering gaps, no empty answers, no
   unbalanced LaTeX. On failure, the *whole exercise* is retried once
   (`MAX_RETRIES_PER_EXERCISE` in `config.py`).
7. **Diagrams — Strict Hierarchy Rule.** Every question gets AT MOST
   one diagram, chosen in this order, never both:
   1. **Original book diagram**, if one exists for that question.
       `book_diagram_extractor.py` extracts embedded images from the
       exercise PDF (via PyMuPDF), filtering out anything too small to
       be a real figure or that repeats across most pages (running
       logos). Gemini flags which questions have one via
       `"has_book_diagram": true` in its JSON; `solver.py`'s
       `_attach_book_diagrams()` attaches an image ONLY on an EXACT
       figure-number match between the question's own citation
       ("চিত্ৰ 3.14") and the image's printed caption — there is no
       page-order guessing path (it was removed after it was proven to
       attach wrong figures in real runs), and vision-read captions get
       a post-attach cross-check (see Quick start above) so a misread
       digit can never silently ship the wrong crop.
2. **Else, the AI-generated SVG** — Gemini only supplies *which*
   shape and *which* relationships (equal sides, right angles, or —
   for Chapter 1-style "Real Numbers" content — a number-line
   placement or square-root spiral construction); `diagram_renderer.py`
   computes every coordinate and draws the actual SVG. 28 types total:
   `triangle`, `circle`, `angle`, `parallel_lines`,
   `coordinate_plot`, `number_line`, `square_root_spiral` (built-in)
   plus plugins: `quadrilateral`, `trigonometry`, `statistics`,
   `surface_area_volume`, `construction`, `polynomial_long_division`,
   `transformation`, `function_graph`, `probability`, `venn_diagram`,
   `elevation_depression`, `bearing`, `unit_circle`,
   `circle_line_intersection`, `solid_net`, `cross_section`,
   `circle_sector`, `composite_shaded_region`,
   `successive_magnification`, and `rectilinear_composite`
   (Class 6-8 compound rectilinear mensuration figures — exact Σw×h
   area, traced union outline, shoelace-verified). Any renderer
   exception is caught and logged rather than propagating —
   a diagram problem degrades to "no diagram for this one question,"
   never to losing the whole exercise's already-solved output.
   3. **Else, nothing** — no empty diagram box, no stray caption.

   The hierarchy itself lives in `templates/base.html.jinja`
   (`{% if q.book_diagram_base64 %}...{% elif q.diagram_svg %}...{% endif %}`).
8. **Render** — the solved JSON fills a single fixed Jinja2 template
   (`templates/base.html.jinja` + `templates/style.css`) — centered blue
   header (`AssamStudy AI - Class N Maths` / `Exercise X.Y` split top
   bar, centered chapter title in Assamese numerals, centered exercise
   subtitle, "➢"-bulleted solution steps, left accent-bar question
   cards). MathJax (vendored, offline) typesets every equation;
   Playwright's headless Chromium **waits for MathJax to actually
   finish** before printing (`pdf_generator.py`), then prints with a
   fixed footer/page-number template. Text flows naturally across page
   boundaries (no `page-break-inside: avoid` on question blocks — that
   was the cause of large blank gaps on long proofs); only the small
   `proved-box` is kept from splitting. Layout is code, not AI, so
   every PDF matches.
9. **Save** — `output/class_{N}/chapter_{N}/ex_{exercise}.pdf`; existing
   files are skipped on future runs.

Exercises within a chapter run in parallel
(`PARALLEL_WORKERS` in `config.py`, default 3) since the dominant cost
is I/O-bound waiting on the Gemini API call, not local CPU.

## Extending to a new book / class / subject

Just change `BOOK_URL`, `CLASS`, `CHAPTER` in `config.py` — nothing
else needs to change unless the new book uses different heading
wording, in which case add a pattern to `_CHAPTER_PATTERNS` /
`_EXERCISE_PATTERNS` in `chapter_detector.py` / `exercise_splitter.py`
(the vision fallback in `vision_ocr.py` needs no changes — it reads
headings visually, not by wording pattern).

New diagram shapes (circles, angles, etc. beyond what's already
handled) go in `diagram_renderer.py` as a new `_render_xxx` function
registered in `_RENDERERS`.

## Module map

```
main.py                — orchestrates the whole pipeline (+ optional CLI flags:
                          --class/--chapter/--exercise/--construction/--book-url)
config.py               — the only file you normally edit per run
books.py                 — class 6-10 SEBA/SCERT Assamese maths textbook registry
downloader.py            — download + cache + verify the book PDF
chapter_detector.py       — chapter page-range + title (text-first, vision-fallback,
                             validates any title before accepting it — see _looks_like_real_title)
exercise_splitter.py       — exercise page-range + PDF splitting (same fallback pattern)
vision_ocr.py               — Gemini Vision fallback + text-layer reliability check
book_diagram_extractor.py    — extracts original book diagram images from the exercise PDF
                                 (vector-cluster + text-caption match when the PDF's text layer
                                 is reliable; Gemini Vision figure-location fallback otherwise)
solver.py                     — one Gemini call per exercise -> structured JSON;
                                 matches book diagrams to questions; repairs bad JSON escapes;
                                 runs a narrow coordinate self-verification pass afterward
prompts.py                     — the solve prompt (MathJax rules, proof structure,
                                  diagram rules, has_book_diagram flag)
diagram_renderer.py             — diagram_spec -> deterministic SVG (never crashes —
                                   any renderer error degrades to no-diagram, not a lost exercise)
html_renderer.py                  — fills the Jinja2 template, inlines MathJax + fonts
pdf_generator.py                   — Playwright print-to-PDF, waits for MathJax
utils.py                            — logging, retry, verification checks
review_state.py                      — Human Review: persistent per-exercise status/version/
                                        history/per-question-acceptance sidecar (.review.json)
                                        + the chapter publish gate
correction_engine.py                  — Human Review: applies a reviewer's correction prompt to
                                         ONE question, or attaches a manually-cropped figure to
                                         ONE question, and re-renders that exercise's PDF in place
interactive_review.py                  — Human Review: the interactive per-question console loop
                                          main.py runs on every exercise — THE primary workflow
review_cli.py                          — Human Review: secondary/ad-hoc command-line tool
                                          (list/show/correct/approve/reject/add-figure/publish)
templates/base.html.jinja            — fixed page shell, strict diagram hierarchy
templates/style.css                   — fixed styling, premium typography
templates/vendor/mathjax/tex-svg.js    — vendored MathJax (offline SVG output)
templates/fonts/TiroBangla-Regular.ttf  — the ONE font for the entire document
templates/images/logo.png                — watermark, tiled once per printed page
```

## Notes

- `config.ALLOW_INSECURE_SSL` (default `False`) skips SSL certificate
  verification on direct (non-Google-Drive) book downloads if set to
  `True`. Only turn this on if you've confirmed a specific book URL's
  own certificate is genuinely misconfigured — it's a real security
  downgrade, so it's an explicit opt-in rather than a default.
- **Escaping**: Gemini's JSON occasionally over-escapes LaTeX (turning
  `\frac` into a literal `\\frac` that shows as visible text) or
  mis-escapes an intended line break (`\n` in a multi-part question
  becoming a literal visible "\n" instead of a real break). Both are
  fixed by `solver.py`'s `_normalize_math_text()`, applied to every
  text field after parsing regardless of which code path produced it:
  runs of 2+ backslashes collapse to one, and a leftover literal `\n`
  becomes a real `<br>` unless it's immediately followed by a lowercase
  letter (which means it's actually the start of a macro like `\neq`
  or `\nabla`, not a newline).
- **Book diagram extraction** renders each detected figure's region via
  `page.get_pixmap(clip=rect)` rather than pulling raw embedded image
  bytes — the latter (`doc.extract_image()`) ignores the page's own
  transformation matrix and can come out mirrored or rotated relative
  to how the page actually displays it.
- **Figure captions on vector-text books**: some SEBA books (confirmed
  on the Class 9 book used for Chapters 3 and 7) render body text
  itself as vector-outlined glyphs rather than real embedded font
  text — `fitz.get_text()`/`get_textbox()` return nothing usable
  anywhere on the page, not just for the chapter/exercise headings
  `vision_ocr.is_text_layer_reliable` already guards. On those books,
  `book_diagram_extractor._vector_candidates`' caption search (which
  depends on `get_textbox()`) can never succeed, so it's skipped
  entirely in favor of `vision_ocr.locate_figures_via_vision` — a
  page-image-based Gemini Vision call that reports each figure's
  bounding box and printed caption number directly, same category of
  fallback the project already uses for chapter/exercise detection.
  The actual crop pixels still always come from the deterministic
  `page.get_pixmap` call — vision only ever answers "where is it /
  what number is printed under it", never draws or approximates
  anything. Gated by `book_diagram_extractor._text_layer_reliable`
  (a local, import-cycle-safe echo of `vision_ocr.is_text_layer_reliable`)
  and cached the same way as the normal path, via `book_figure_index.py`.
- **Coordinate self-verification**: for any solved question that reads
  a labelled point's coordinate off a figure (detected by keywords
  like স্থানাংক/ভুজ/কোটি/coordinate) AND has its exact book-diagram
  image attached, `solver._verify_coordinate_answers` fires one small,
  narrowly-scoped follow-up Gemini Vision call — "what is point X's
  coordinate in THIS cropped image, nothing else" — and cross-checks
  it against what the main combined solve pass produced, correcting
  `final_answer`/`steps` on a confirmed mismatch. This exists because
  the combined single-pass solve can still misread one point among
  several while solving an entire exercise at once, even when the
  correct figure was already part of its input — a narrow, isolated
  re-check catches that class of error that diagram-matching alone
  cannot. Fails open: any verification-call error leaves the original
  answer untouched rather than risking a worse "correction."
- **coordinate_plot diagrams** (the Case B fallback, used only when no
  matching book figure exists) now plot each point at its literal,
  to-scale `(x, y)` position — `diagram_renderer._render_coordinate_plot`
  previously ignored the real coordinates entirely and placed dots at
  arbitrary index-based offsets; `prompts.py`'s `diagram_spec` schema
  now requires an `"x"`/`"y"` pair per point for this diagram type so
  there's actually something real for the renderer to plot.
- **Chapter titles** are looked up with a dedicated, cheap 1-2 page
  Gemini Vision call scoped to just the chapter's own opening page
  (`vision_ocr.get_chapter_title_via_vision`) — this runs regardless of
  whether the whole-book boundary-detection scan ever ran, so a book
  with an otherwise-reliable text layer (which skips vision for page
  detection) still gets a title looked up.
- **Rate limits**: `solver.py` and `vision_ocr.py` wrap every Gemini
  call in `utils.retry_with_backoff` — up to 6 attempts with
  exponential delay (5s, 10s, 20s, ...) specifically for 429
  RESOURCE_EXHAUSTED / quota / transient errors, while a genuinely
  non-retryable error (bad request, auth failure) still fails
  immediately instead of wasting time backing off pointlessly.
- **Diagram safety net** (`diagram_safety_net.py`, new): closes the one
  remaining gap in the diagram pipeline — a question whose own text
  clearly implies a figure (geometry/construction/number-line/chart
  vocabulary, or a 3-letter vertex list like "ABC") but that Gemini's
  single combined solve pass never gave a `diagram_spec` for at all.
  Runs once per exercise, strictly after Stage 1 (`diagram_decision.py`)
  has already made its real decision for every question. It NEVER
  touches a question that already has a book diagram or a `diagram_spec`
  that was validated and then correctly rejected — only plain silence.
  For each flagged question it fires one narrow, independent Gemini call
  scoped to just that question's own text, and any recovered spec still
  has to pass every one of Stage 1's existing checks (type validity,
  internal consistency, textual grounding, structural validation) before
  it's accepted — this module has no authority to publish a diagram
  directly. Fails open: any recovery failure just leaves the question as
  `NO_DIAGRAM` with `question["diagram_safety_net_flagged"] = True` set
  for a human to review, never a guessed diagram. See
  `test_diagram_safety_net.py` for the full behavior spec.
- **Bug fix — triangle spec validation** (`diagram_renderer.py`,
  `validate_diagram_spec`): found while building the safety net above.
  `diagram_type: "triangle"` previously had NO minimum-points check at
  all — every existing check only cross-references altitude/equal_marks/
  right_angle_at ids against whatever points *were* declared, so a spec
  with an empty `points: []` (or none at all) sailed through structural
  validation with zero issues, exactly the kind of "renders something
  not tied to the real question" failure this validation layer exists to
  prevent. Fixed to require at least 3 uniquely-identified points, the
  same way `_validate_quadrilateral_spec` already requires exactly 4.
  This is a real, general-purpose fix — not specific to the safety net —
  so it also protects the main solve pass's own triangle specs.
