# Production Audit — 2026-08-14

## Scope and honest limitations

This sandbox has **no network route to Google Cloud / Vertex AI** and no
GCP credentials, so the actual Gemini solving step (`solver.py`) could
not be exercised end-to-end here. Everything below covers the
**deterministic half of the pipeline**: LaTeX/math sanitization, the
diagram engine (built-in + all 15 plugin types), HTML rendering, and
PDF generation — i.e. exactly the parts responsible for "broken
frac/LaTeX at render time" and "diagrams not rendering correctly."
Nothing here is a substitute for running real chapters through Gemini
and grading the output; that would need real Vertex AI access.

## Baseline

Full test suite before any changes: **1171/1171 passing**. This is a
mature, well-engineered codebase — most of this audit was spent trying
to *break* it, not fixing things that were obviously wrong.

## What was found and fixed

### 1. Undefined LaTeX macros could silently render as broken red text (real bug, fixed)

`math_sanitizer.py`'s structural validator intentionally let any
unrecognized multi-letter macro (e.g. a Gemini typo like `\gibberish`,
`\sinx`, `\vecc`) through, on the assumption that MathJax would either
render it fine or throw a normal parse error some later stage would
catch.

I proved empirically — by rendering test cases through headless
Chromium against this project's **actual vendored MathJax bundle**,
not a guess — that this assumption is false. An undefined macro does
**not** raise a `<merror>` node. MathJax silently draws the raw
command text (backslash included) as ordinary glyphs colored
`fill="red"`, with no `data-mjx-error` marker anywhere in the DOM. That
means it would have been invisible to a naive "check for MathJax
errors" validator too (I had to catch my own first draft of one
missing it).

**Fix, defense in depth:**
- `math_sanitizer.py`: unknown macros are now rejected and degraded to
  safe plain text instead of passing through. Expanded
  `KNOWN_LATEX_MACROS` with ~70 legitimate standard LaTeX
  arrows/relations/operators first, so this tightening doesn't wrongly
  degrade real content (caught by the existing test suite, which had
  a test asserting `\uparrow`-family macros are never touched).
- `pdf_generator.py`: added a real post-typeset DOM scan, after
  `window.__mathjaxReady__`, for **both** failure signatures —
  `[data-mjx-error]` (normal parse errors) and
  `mjx-container[jax="SVG"] g[fill="red"]` (the silent undefined-macro
  fallback). If either is found, `render_pdf` raises
  `RenderValidationError` **before writing any PDF file**, with a
  diagnostic naming the offending question. `main.py`'s existing
  per-exercise retry loop already treats any exception from this
  function as a normal retryable failure — zero other plumbing needed.

**Verification:** hand-crafted bypass case blocked correctly with a
clear diagnostic; clean content produces zero false positives; full
suite still 1171/1171 after the change; added regression tests
(`test_math_sanitizer.py::TestUnknownMacroDegradation`,
`e2e_render_validation_check.py`).

### 2. Diagram engine fuzz testing — no bugs found, but now verified

Built a fuzz harness using this project's own real example specs (from
its test files) and mutated them 2,700 times across **all 27 diagram
type/subtype combinations** (12 built-in + 15 plugin), injecting
zero/negative/huge/tiny/NaN values into every numeric field:

- **0 crashes** escaping `render_diagram()` (it already fails safe by
  design — confirmed under real fuzzing, not just by reading the code)
- **0 NaN/Infinity values leaked** into generated SVG
- **0 malformed SVG** (missing `<svg>` tag) returned
- Bad specs correctly degrade to "no diagram" with a logged reason
  rather than shipping something wrong

No changes were needed here — the existing "never take the exercise
down, log and omit" architecture already holds up under real fuzzing.

### 3. Visual spot-check

Rendered the project's own `demo_regenerate.py` end-to-end (real
Playwright/Chromium, only the Gemini call stubbed) and visually
inspected the output PDF page-by-page: correct Assamese typesetting,
no broken fraction boxes, correct layout, watermark, and footer.

## What I did NOT audit (be aware)

- `layout_validation.py` / `diagram_final_check.py` were read but not
  independently fuzzed — they're explicitly non-blocking audit layers
  by design (see their own docstrings), consistent with the rest of
  the pipeline.
- No real Gemini/Vertex AI calls were made or could be made from this
  environment.
- I did not attempt to estimate a numeric "95% correct" score — that
  number can only come from running real chapters through the real
  solver and grading the output; anyone claiming otherwise without
  doing that isn't measuring anything.

## Files changed

- `math_sanitizer.py` — tightened unknown-macro handling, expanded
  `KNOWN_LATEX_MACROS`
- `pdf_generator.py` — added post-typeset `RenderValidationError` gate
- `test_math_sanitizer.py` — new regression tests
- `e2e_render_validation_check.py` — new e2e check (run manually, real
  browser, follows this project's existing `e2e_*.py` convention)

Full test suite after all changes: **1173/1173 passing**.
