r"""
pdf_generator.py — renders the filled HTML to a PDF via headless
Chromium (Playwright). Margins, page numbers, and the footer are fixed
here so every generated PDF is byte-for-byte consistent in layout.

IMPORTANT: MathJax typesets asynchronously after the page loads. Without
waiting for it, Playwright would snapshot the page mid-render — either
raw, un-rendered "\(...\)" text, or a half-drawn equation. render_pdf()
now waits for the `window.__mathjaxReady__` flag (set by base.html.jinja
once MathJax's startup.promise resolves) before printing.

FINAL RENDER-TIME GATE (added in this audit pass): math_sanitizer.py /
latex_grammar.py catch almost everything BEFORE this point, but they
are static text analysis — they can never be a perfect substitute for
asking the real MathJax engine "did this actually typeset cleanly?"
That question can only be answered here, after real typesetting has
happened in a real browser. Two independent, empirically-verified
failure signatures are checked for after `__mathjaxReady__`:

  1. `[data-mjx-error]` — a genuine MathJax <merror> node (e.g.
     "Missing argument for \frac"). This is the classic "red error
     box" case.
  2. `mjx-container[jax="SVG"] g[fill="red"]` — a SECOND, easy-to-miss
     failure mode confirmed by directly rendering test cases against
     this project's own vendored MathJax bundle: an UNDEFINED control
     sequence (a hallucinated/typo\u2019d macro name that survived static
     validation) does NOT raise a <merror> at all — MathJax silently
     falls back to drawing the literal command text (backslash
     included) as ordinary glyphs colored fill="red", with no
     data-mjx-error marker anywhere in the DOM. A check that only
     looked for (1) would have shipped that PDF with visibly broken
     red text on the page. Scoped strictly to elements inside
     mjx-container so it can never fire on an unrelated diagram SVG
     (diagram_svg is rendered as its own separate <svg>, never inside
     an mjx-container).

If either signature is found, render_pdf raises RenderValidationError
BEFORE writing any PDF file — main.py's existing per-exercise retry
loop (MAX_RETRIES_PER_EXERCISE) already treats any exception from this
function as a normal retryable failure, so this fails safe into the
pipeline's existing recovery path rather than requiring new plumbing.
"""
import os
from playwright.sync_api import sync_playwright

import config

_FOOTER_TEMPLATE = f"""
<div style="font-size:9px; width:100%; padding:0 40px; display:flex;
            justify-content:space-between; color:#888;
            font-family: Arial, sans-serif;">
  <span>{config.FOOTER_TEXT}</span>
  <span>Page <span class="pageNumber"></span></span>
</div>
"""

MATHJAX_RENDER_TIMEOUT_MS = 20_000

# JS run in-page after __mathjaxReady__ to detect both known failure
# signatures. Returns a list of short diagnostic strings (empty list =
# clean). Kept intentionally small/cheap — this runs on every PDF.
_RENDER_ERROR_CHECK_JS = r"""
() => {
  const out = [];
  const seen = new Set();
  const nearestQuestion = (el) => {
    const block = el.closest('.question-block');
    if (!block) return 'unknown question';
    const num = block.querySelector('.question-number');
    return num ? num.textContent.trim() : 'unknown question';
  };
  document.querySelectorAll('[data-mjx-error]').forEach(el => {
    const key = 'merror:' + el.getAttribute('data-mjx-error');
    if (seen.has(key)) return;
    seen.add(key);
    out.push(`MathJax parse error ("${el.getAttribute('data-mjx-error')}") in ${nearestQuestion(el)}`);
  });
  document.querySelectorAll('mjx-container[jax="SVG"] g[fill="red"]').forEach(el => {
    const container = el.closest('mjx-container');
    // No reliable attribute holds the original TeX source at this
    // point, so fall back to nearby visible text for a diagnosable
    // (not necessarily exact) pointer to the offending span.
    const context = container && container.parentElement
      ? container.parentElement.textContent.trim().slice(0, 80)
      : '';
    const key = 'undefined:' + context;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(`Undefined/unrenderable LaTeX command near "${context}" in ${nearestQuestion(el)}`);
  });
  return out;
}
"""


class RenderValidationError(Exception):
    """Raised when real MathJax typesetting produced a detectable
    error (a <merror> node, or the silent undefined-macro red-glyph
    fallback) anywhere on the page. Never write a PDF in this state —
    see module docstring."""


def render_pdf(html: str, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.set_content(html, wait_until="load")

            # Block until MathJax has actually finished typesetting every
            # equation on the page — this is what makes the MathJax script
            # in base.html.jinja actually take effect in the printed PDF,
            # rather than being a no-op.
            page.wait_for_function(
                "window.__mathjaxReady__ === true",
                timeout=MATHJAX_RENDER_TIMEOUT_MS,
            )

            render_errors = page.evaluate(_RENDER_ERROR_CHECK_JS)
            if render_errors:
                raise RenderValidationError(
                    f"{len(render_errors)} MathJax render error(s) detected "
                    f"before PDF write, refusing to publish {output_path}: "
                    + "; ".join(render_errors)
                )

            page.pdf(
                path=output_path,
                format="A4",
                print_background=True,
                display_header_footer=True,
                header_template="<span></span>",
                footer_template=_FOOTER_TEMPLATE,
                margin={"top": "18mm", "bottom": "16mm", "left": "0mm", "right": "0mm"},
            )
        finally:
            browser.close()
