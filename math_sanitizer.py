"""
math_sanitizer.py — the single mandatory Math Sanitization Layer every
text field in this pipeline passes through before it can reach
HTML/PDF rendering.

WHY THIS EXISTS
----------------
Every earlier fix in this project (bare-LaTeX wrapping, "\\n"/"\\t"
leakage decoding, stray "\\uXXXX" decoding, ...) was a correct but
NARROW patch for one specific way Gemini's free-form math text can
come out malformed. Each new malformed shape Gemini produces needs its
own new patch, forever, and until it's found and patched it renders as
a raw, broken, red MathJax parse-error box in a published PDF — with
no possibility of a human catching it before publish, because nothing
in the pipeline ever actually checks whether the text it's about to
hand to MathJax is well-formed.

This module replaces that "patch one shape at a time" approach with a
single generic gateway, `sanitize_math_text()`, built around a real
model of what "well-formed \\( \\) math" MUST structurally look like
(balanced braces/brackets, balanced \\left/\\right, matching
\\begin{...}/\\end{...}, no delimiter nested inside another, no
non-math script mixed in) rather than any one command's syntax. It
works the same way regardless of whether the broken content is a
\\frac, a \\sqrt, a matrix, a vector, an angle, or a LaTeX command that
doesn't even exist yet in this codebase's test suite.

PIPELINE (see sanitize_math_text's own docstring for the exact order):
  1. decode JSON-escape leakage ("\\n" "\\t" "\\r" "\\b" "\\f" literally
     surviving as backslash-letter text instead of real control chars)
  2. collapse over-escaping and decode stray "\\uXXXX" unicode escapes
  3. wrap bare (undelimited) LaTeX in "\\( \\)"
  4. repair delimiter STRUCTURE: fix nested/orphan "\\(" "\\)", balance
     braces/brackets inside every span, fix mismatched \\left/\\right,
     fix mismatched \\begin/\\end environments, split out any
     Assamese/Bengali script that ended up inside a math span
  5. FINAL RENDER VALIDATION: re-check every span against the same
     structural model; anything still invalid is degraded to plain,
     readable text (never a raw "\\(" left visible, never handed to
     MathJax broken) instead of ever showing a red error box.

Every step here is a GENERIC structural transformation — none of it is
"if the text looks like a \\frac, do X". A brand-new command this
domain has never used before gets exactly the same treatment as
\\frac/\\sqrt/matrices/vectors/angles: masked while searching for bare
LaTeX, balance-checked, and degraded if (and only if) it's still
broken after repair.
"""
import re

import latex_grammar


# ----------------------------------------------------------------------
# 0. Known LaTeX macro vocabulary — one master list, used for (a) the
#    JSON-escape-leakage-vs-genuine-macro disambiguation (step 1) and
#    (b) the final "does every command in this span look real"
#    render-validation check (step 5). A single shared list rather than
#    a separate hand-picked table per letter/per check — this is the
#    "generic, not regex patches for individual commands" requirement:
#    add a macro here once and every check benefits.
# ----------------------------------------------------------------------
KNOWN_LATEX_MACROS = frozenset({
    # fractions / roots / big operators
    "frac", "dfrac", "tfrac", "sqrt", "root", "sum", "prod", "int", "oint",
    "lim", "limsup", "liminf", "log", "ln", "exp", "sin", "cos", "tan",
    "cot", "sec", "csc", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan",
    "min", "max", "gcd", "det",
    # operators / relations
    "times", "div", "pm", "mp", "cdot", "ast", "circ", "bullet", "star",
    "leq", "geq", "le", "ge", "neq", "ne", "approx", "equiv", "cong", "sim",
    "simeq", "propto", "ll", "gg",
    # symbols
    "infty", "angle", "triangle", "square", "perp", "parallel", "nparallel",
    "degree", "prime", "backslash", "partial", "nabla", "hbar", "ell",
    "wp", "Re", "Im", "aleph", "top", "bot",
    # arrows / logic
    "rightarrow", "leftarrow", "Rightarrow", "Leftarrow",
    "leftrightarrow", "Leftrightarrow", "longrightarrow", "longleftarrow",
    "mapsto", "to", "gets", "therefore", "because", "implies", "iff",
    # set theory
    "in", "notin", "ni", "subset", "subseteq", "supset", "supseteq",
    "cup", "cap", "setminus", "emptyset", "varnothing", "forall", "exists",
    "nexists", "mathbb",
    # greek
    "alpha", "beta", "gamma", "Gamma", "delta", "Delta", "epsilon",
    "varepsilon", "zeta", "eta", "theta", "vartheta", "Theta", "iota",
    "kappa", "lambda", "Lambda", "mu", "nu", "xi", "Xi", "pi", "Pi", "rho",
    "varrho", "sigma", "Sigma", "tau", "upsilon", "Upsilon", "phi",
    "varphi", "Phi", "chi", "psi", "Psi", "omega", "Omega",
    # accents / decorations
    "vec", "overrightarrow", "overleftarrow", "overline", "underline",
    "hat", "widehat", "tilde", "widetilde", "dot", "ddot", "bar",
    "overbrace", "underbrace", "check", "breve", "acute", "grave",
    # structures
    "binom", "dbinom", "choose", "left", "right", "big", "Big", "bigg",
    "Bigg", "bigl", "bigr", "Bigl", "Bigr", "begin", "end", "matrix",
    "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "smallmatrix", "array",
    "cases",
    # text / formatting
    "text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt", "textbf",
    "textit", "boldsymbol", "operatorname", "boxed", "fbox", "underline",
    "overline", "hspace", "vspace", "quad", "qquad",
    # spacing / punctuation escapes
    "mod", "bmod", "pmod", "colon", "cdots", "ldots", "vdots", "ddots",
    "langle", "rangle", "lvert", "rvert", "lVert", "rVert", "mid",
    "nmid", "hline", "cline", "not",
    # AUDIT ADDITION — arrows/relations/operators that are standard,
    # legitimate LaTeX (not Gemini typos) and plausible in class 6-10
    # geometry/algebra/set-theory content, added after tightening the
    # "unknown multi-letter macro" check in _looks_structurally_valid
    # from "always accept" to "must be in this list" (see that
    # function's comment for why the change was made). Keeping this
    # list generously wide keeps that tightening from ever degrading
    # real, correctly-used LaTeX just because this allowlist happened
    # not to enumerate it yet.
    "uparrow", "downarrow", "updownarrow", "Uparrow", "Downarrow",
    "Updownarrow", "hookrightarrow", "hookleftarrow", "rightharpoonup",
    "leftharpoonup", "rightharpoondown", "leftharpoondown",
    "rightleftharpoons", "xrightarrow", "xleftarrow", "longmapsto",
    "nearrow", "searrow", "swarrow", "nwarrow", "dashrightarrow",
    "dashleftarrow", "multimap", "leadsto",
    "wedge", "vee", "oplus", "otimes", "ominus", "odot", "oslash",
    "uplus", "sqcup", "sqcap", "wr", "amalg",
    "prec", "succ", "preceq", "succeq", "vdash", "Vdash", "vDash",
    "nvDash", "models", "asymp", "doteq", "bumpeq", "smile", "frown",
    "sqsubset", "sqsupset", "sqsubseteq", "sqsupseteq",
    "triangleleft", "triangleright", "trianglelefteq",
    "trianglerighteq", "lhd", "rhd", "unlhd", "unrhd",
    "complement", "measuredangle", "sphericalangle",
    "mathcal", "mathfrak", "stackrel", "overset", "underset",
    "substack", "nleq", "ngeq", "nsubset", "nsupset", "notni",
})

# Single-non-letter-character escapes that are always valid LaTeX
# regardless of context (a backslash immediately followed by a symbol
# rather than a letter) — used by the "does every command look real"
# check in step 5 so these are never mistaken for a malformed command.
_VALID_SYMBOL_ESCAPES = set("{}%$&#_ ,;:!\\")

# The letters JSON can legitimately escape onto a backslash — a
# genuinely leaked control-character escape only ever starts with one
# of these.
_ESCAPE_LEAK_LETTERS = "ntrbf"

_MACRO_CONTINUATIONS_BY_LETTER = {
    letter: tuple(sorted((name[1:] for name in KNOWN_LATEX_MACROS if name[0] == letter),
                         key=len, reverse=True))
    for letter in _ESCAPE_LEAK_LETTERS
}

_LEAK_REPLACEMENT = {
    "n": "<br>",
    "t": "&nbsp;&nbsp;&nbsp;&nbsp;",
    "r": "",
    "b": "",
    "f": "",
}


def _is_macro_continuation(letter: str, rest: str) -> bool:
    """True if `rest` (whatever immediately follows "\\<letter>") completes
    a real macro name for that letter, at a hard word boundary — a
    shared prefix alone doesn't count (so a stray "\\nx" table cell
    isn't mistaken for "\\nexists" just because "x" isn't whitelisted),
    but the match also can't be fooled by a LONGER unrelated macro that
    merely starts the same way (checking longest-option-first, so e.g.
    "\\therefore" is matched whole rather than stopping early at some
    shorter unrelated prefix)."""
    for option in _MACRO_CONTINUATIONS_BY_LETTER.get(letter, ()):
        if rest.startswith(option):
            tail_pos = len(option)
            if tail_pos == len(rest) or not rest[tail_pos].isalpha():
                return True
    return False


def decode_json_escape_leakage(text: str) -> str:
    """Step 1: a literal two-character "\\n" / "\\t" / "\\r" / "\\b" /
    "\\f" surviving in the text (instead of the real control character
    JSON's own escape rules mean it should have decoded to, or instead
    of being a genuine LaTeX macro like "\\tan"/"\\nabla"/"\\beta") is
    NEVER valid output — it's leakage from an over-escaped JSON string
    upstream. "\\n" becomes a real line break; "\\t" becomes visible
    spacing (almost always a column separator in tabular content); the
    control characters with no visual meaning in HTML ("\\r" "\\b"
    "\\f") are simply dropped. A genuine macro continuation for that
    letter is always left completely untouched.
    """
    if not isinstance(text, str) or "\\" not in text:
        return text

    def _replace(m):
        letter = m.group(1)
        rest = text[m.end():]
        if _is_macro_continuation(letter, rest):
            return m.group(0)
        return _LEAK_REPLACEMENT[letter]

    return re.sub(r"\\([ntrbf])", _replace, text)


_STRAY_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _decode_stray_unicode_escapes(text: str) -> str:
    def _decode(m):
        try:
            return chr(int(m.group(1), 16))
        except ValueError:
            return m.group(0)
    return _STRAY_UNICODE_ESCAPE.sub(_decode, text)


_STRAY_UNICODE_LOOKAHEAD = re.compile(r"u[0-9a-fA-F]{4}")


def _collapse_backslash_run(m: re.Match) -> str:
    run = m.group(0)
    n = len(run)
    # A 2-backslash run immediately followed by "uXXXX" is virtually
    # certainly a DOUBLE-escaped unicode escape (Gemini writes
    # "\\u221a", json.loads decodes that per JSON's own rules into this
    # literal 2-backslash text) rather than a genuine matrix row
    # separator — collapse it to 1 so _decode_stray_unicode_escapes can
    # find and decode it afterward (its regex only matches a SINGLE
    # backslash immediately before "uXXXX"; leaving 2 backslashes here
    # would make that regex match starting at the second backslash
    # instead, silently leaving the first one behind as stray literal
    # text — exactly the "\\√ x" instead of "√ x" corruption this
    # lookahead exists to prevent).
    if n == 2 and _STRAY_UNICODE_LOOKAHEAD.match(m.string, m.end()):
        return "\\"
    return "\\\\" if n % 2 == 0 else "\\"


def collapse_and_decode_escapes(text: str) -> str:
    """Step 2: collapse over-escaped runs of backslashes down to what
    was actually intended, then decode any remaining lone "\\uXXXX" run
    into its real character (the double-escaping failure mode: Gemini
    double-escapes "\\u221a" into "\\\\u221a", which json.loads then
    correctly decodes, per JSON's own rules, into the literal
    6-character text "\\u221a" rather than "√" — safe to decode here
    specifically because the collapse just above already resolved any
    ambiguity with a genuine double-backslash-then-u sequence).

    The collapse is PARITY-aware, not a blanket "any run -> one
    backslash": almost every LaTeX command wants exactly ONE leading
    backslash, but the matrix/cases row-separator "\\\\" is a real,
    meaningful TWO-backslash command in its own right — collapsing it
    down to one would silently break every matrix. Each level of
    Gemini's over-escaping doubles the backslash count, so an even-
    length run collapses to 2 (the row-separator, however many extra
    escaping levels it picked up) and an odd-length run collapses to 1
    (an ordinary over-escaped single command); a run of exactly 2 is
    therefore left untouched, exactly as a correct row-separator should
    be — UNLESS it's immediately followed by "uXXXX" (see
    _collapse_backslash_run above), the one case where "exactly 2
    backslashes" has a different, more likely explanation.
    """
    if not isinstance(text, str):
        return text
    collapsed = re.sub(r"\\{2,}", _collapse_backslash_run, text)
    return _decode_stray_unicode_escapes(collapsed)


# ----------------------------------------------------------------------
# Step 3: wrap bare (undelimited) LaTeX in \( \).
# ----------------------------------------------------------------------

# An already-correctly-delimited inline-math span, masked off while
# scanning for bare LaTeX elsewhere — this must never double-wrap
# something that already has its \( \) pair. Matches the FIRST \) after
# a \( (non-greedy) — deliberately naive about nesting here; genuine
# nested/orphan delimiters are a separate, harder problem solved by
# repair_delimiter_structure() (step 4) on the whole text afterward, not
# by this mask.
_DELIMITED_MATH_SPAN = re.compile(r"\\\(.*?\\\)", re.DOTALL)

_BARE_LATEX_CHAR = re.compile(r"[0-9A-Za-z^_{}()=+\-*/.,]")
_BARE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_MAX_UNCLOSED_SPACE_CROSSINGS = 4
_PROSE_WORD_AHEAD = re.compile(r"[A-Za-z]{3,}")


def _extract_bare_latex_runs(text: str):
    """Yields (start, end) spans of maximal bare-LaTeX runs: normal
    LaTeX-shaped characters (letters, digits, ^, _, braces, parens, and
    a handful of operators), or a backslash-command, chained together —
    crucially, a SPACE no longer ends a run while EITHER brace nesting
    OR paren nesting is still unbalanced (depth > 0). Both grouping
    characters must be tracked, not just braces: a perfectly ordinary
    bare expression like "(7 \\times 8)^{\\frac{1}{2}}" has its space
    inside an open "(", not a "{", and without watching parens too it
    gets cut right there — "(7" is left behind as a dangling,
    never-wrapped fragment (no backslash of its own to trigger
    wrapping), "\\times" gets wrapped alone, and "8)^{\\frac{1}{2}}"
    ends up wrongly wrapped as its own group starting with a bare,
    unmatched ")" — the exact "(7\\(\\times\\)\\(8)^{...}\\)" corruption
    this depth tracking exists to prevent (mirroring the identical fix
    already applied for "\\frac{-32 \\times 5}{9}"'s curly braces).
    Depth resets are self-correcting (a run's own closing braces/parens
    decrement it back down), so a genuinely separate expression a few
    words later never gets swallowed into the same run.
    """
    n = len(text)
    i = 0
    spans = []
    while i < n:
        ch = text[i]
        cmd_here = ch == "\\" and _BARE_LATEX_CMD.match(text, i)
        if cmd_here or _BARE_LATEX_CHAR.match(ch):
            start = i
            depth = 0
            # ROOT-CAUSE FIX: "keep crossing spaces while depth > 0"
            # is only safe for the SHORT, genuinely-nested case this
            # docstring describes ("(7 \times 8)^{...}") where depth
            # returns to 0 after a word or two. If a brace/paren is
            # actually just unclosed (a real corruption, not nesting),
            # depth never returns to 0 and this used to keep crossing
            # spaces all the way to the end of the string — silently
            # swallowing whatever unrelated prose came after it into
            # what a later repair step would then treat as part of the
            # same expression (e.g. "\frac{1}{2 and more text" turning
            # an entire trailing sentence into \frac's 2nd argument
            # once the missing "}" got auto-appended at the very end).
            # A small cap on how many space-crossings are allowed
            # while still unclosed keeps the legitimate short-nesting
            # case working while making a genuinely unclosed brace
            # stop the run at a sane boundary instead of consuming the
            # rest of the document.
            unclosed_space_crossings = 0
            while i < n:
                ch = text[i]
                if ch == "\\":
                    m = _BARE_LATEX_CMD.match(text, i)
                    if m:
                        i = m.end()
                        continue
                if ch in "{(":
                    depth += 1
                    i += 1
                    continue
                if ch in "})":
                    depth = max(0, depth - 1)
                    i += 1
                    if depth == 0:
                        unclosed_space_crossings = 0
                    continue
                if _BARE_LATEX_CHAR.match(ch):
                    i += 1
                    continue
                if ch.isspace() and depth > 0:
                    # A genuine math sub-expression's next token, after
                    # crossing this space, is short and symbolic (a
                    # variable, a number, an operator, a \command) —
                    # not a whole English word. If what follows the
                    # space is a run of 3+ plain letters ("and", "more",
                    # "text", ...), that's prose, not part of this
                    # expression, no matter how the depth-counter looks;
                    # stop the run right here instead of swallowing it.
                    if _PROSE_WORD_AHEAD.match(text, i + 1):
                        break
                    if unclosed_space_crossings >= _MAX_UNCLOSED_SPACE_CROSSINGS:
                        break
                    unclosed_space_crossings += 1
                    i += 1
                    continue
                break
            spans.append((start, i))
        else:
            i += 1
    return spans


def wrap_bare_latex(text: str) -> str:
    """Wraps a bare (undelimited) LaTeX run in \\( \\) — e.g. a
    "32^{\\frac{2}{5}}" that Gemini forgot to delimit, which MathJax
    would otherwise never touch at all, showing the raw LaTeX source
    verbatim. Conservative: only wraps a run that actually contains a
    backslash-command or ^/_ ; a bare number or plain "AB = 5" is left
    exactly as-is. Already-delimited spans are masked off first so this
    never double-wraps anything.
    """
    if not isinstance(text, str):
        return text
    if "\\" not in text and "^" not in text and "_" not in text:
        return text

    protected = []

    def _mask(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    masked_text = _DELIMITED_MATH_SPAN.sub(_mask, text)
    spans = _extract_bare_latex_runs(masked_text)
    if spans:
        out = []
        last = 0
        for start, end in spans:
            out.append(masked_text[last:start])
            run = masked_text[start:end]
            if "\x00" in run:
                out.append(run)  # a placeholder token itself — never re-wrap
            else:
                trailing = ""
                while run and run[-1] in ".,":
                    trailing = run[-1] + trailing
                    run = run[:-1]
                if "\\" not in run and "^" not in run and "_" not in run:
                    out.append(masked_text[start:end])  # e.g. bare "32" — leave alone
                else:
                    out.append(f"\\({run}\\){trailing}")
            last = end
        out.append(masked_text[last:])
        masked_text = "".join(out)

    def _unmask(m):
        return protected[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", _unmask, masked_text)


# ----------------------------------------------------------------------
# Step 4/5: delimiter-structure repair + final render validation.
# ----------------------------------------------------------------------

_ASSAMESE_BENGALI_BLOCK = re.compile(r"[\u0980-\u09FF]")
_COMMAND_TOKEN = re.compile(r"\\([A-Za-z]+|.)")


def _split_into_delimiter_tokens(text: str):
    """Splits `text` into a flat list of ('open'|'close'|'text', chunk)
    tokens on every literal "\\(" / "\\)" occurrence, in order. This is
    the generic building block both the nested/orphan-delimiter repair
    and the final structural walk are built on — it has no idea what a
    "frac" or a "matrix" is, only where the math-mode boundaries are."""
    tokens = []
    i = 0
    n = len(text)
    last = 0
    while i < n:
        if text.startswith("\\(", i):
            if i > last:
                tokens.append(("text", text[last:i]))
            tokens.append(("open", "\\("))
            i += 2
            last = i
        elif text.startswith("\\)", i):
            if i > last:
                tokens.append(("text", text[last:i]))
            tokens.append(("close", "\\)"))
            i += 2
            last = i
        else:
            i += 1
    if last < n:
        tokens.append(("text", text[last:]))
    return tokens


def _fix_nested_and_orphan_delimiters(text: str) -> str:
    """Repairs \\( \\) NESTING/ORPHAN problems generically: a stray
    "\\)" with nothing open is dropped; a "\\(" encountered while
    already inside a span auto-closes the PREVIOUS span right there
    (splitting one malformed nested attempt into two well-formed
    adjacent spans) instead of corrupting everything downstream of it;
    a span still open at the very end of the text is auto-closed there.
    After this, the text is guaranteed to contain only well-formed,
    non-nested, alternating \\( \\) pairs — the precondition every
    later structural check in this module relies on.
    """
    tokens = _split_into_delimiter_tokens(text)
    out = []
    inside = False
    for kind, chunk in tokens:
        if kind == "text":
            out.append(chunk)
        elif kind == "open":
            if inside:
                out.append("\\)")  # auto-close the previous, still-open span
            out.append("\\(")
            inside = True
        else:  # close
            if not inside:
                continue  # orphan close — drop it, nothing to close
            out.append("\\)")
            inside = False
    if inside:
        out.append("\\)")  # a span left open all the way to the end of the text
    return "".join(out)


def _balance_bracket_pairs(content: str, open_ch: str, close_ch: str) -> str:
    """Generic brace/bracket balancer: removes any ORPHAN closer (one
    encountered with nothing open to match), and appends whatever
    closers are still owed at the end for any opener left unmatched.
    Works identically for {} and [] — and, being purely structural,
    for absolutely any command that uses them (\\frac, \\sqrt[n]{},
    matrices' row/column groups, a vector's \\binom, ...), not just the
    ones this module's tests happen to name.
    """
    depth = 0
    out = []
    for ch in content:
        if ch == open_ch:
            depth += 1
            out.append(ch)
        elif ch == close_ch:
            if depth > 0:
                depth -= 1
                out.append(ch)
            # else: orphan closer — drop it
        else:
            out.append(ch)
    out.append(close_ch * depth)
    return "".join(out)


_LEFT_RIGHT_TOKEN = re.compile(r"\\(left|right)\b")


def _fix_left_right_pairing(content: str) -> str:
    """\\left/\\right must alternate as a properly nested stack (every
    \\left eventually closed by a \\right, never the reverse). If they
    don't — a common source of "unmatched math delimiters" reports,
    since a missing \\right leaves every following \\big/\\Big/plain
    delimiter mis-sized or the whole expression unclosed — the safe,
    generic repair is to strip EVERY \\left/\\right token in this span:
    "\\left(" degrades to a plain "(" and "\\right)" to a plain ")",
    which always renders (just without MathJax's automatic delimiter
    auto-sizing) instead of ever being invalid.
    """
    depth = 0
    balanced = True
    for m in _LEFT_RIGHT_TOKEN.finditer(content):
        if m.group(1) == "left":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                balanced = False
                break
    if depth != 0:
        balanced = False
    if balanced:
        return content
    return _LEFT_RIGHT_TOKEN.sub("", content)


_BEGIN_END_TOKEN = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")


def _begin_end_environments_balanced(content: str) -> bool:
    """Whether every \\begin{env}/\\end{env} (matrix, pmatrix, cases,
    array, ...) in this span nests and names correctly — a matrix whose
    \\end names a different environment than its \\begin, or that's
    missing one side entirely, can't be safely auto-repaired
    structurally (unlike a brace or a \\left/\\right, there's no
    single generic "just strip it" fix that keeps a matrix a matrix),
    so this is a pure detector: a `False` here routes the whole span to
    plain-text degradation instead."""
    stack = []
    for m in _BEGIN_END_TOKEN.finditer(content):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack.pop() != env:
                return False
    return not stack


_LEFT_RIGHT_DELIM_CHAR = re.compile(r"\\(?:left|right)\s*([\[\]().|])")


def _looks_structurally_valid(content: str) -> bool:
    """The final render-validation check (step 5): a deterministic,
    generic stand-in for 'would MathJax actually render this span
    without a parse error'. True only if EVERY structural precondition
    holds at once: braces/brackets balanced, \\left/\\right balanced,
    \\begin/\\end environments balanced and correctly named, no stray
    Assamese/Bengali script mixed into what's supposed to be pure math,
    and every backslash-escape in the span is either a recognized
    macro name or one of the handful of always-valid symbol escapes
    (never a lone backslash followed by, say, a digit or punctuation
    that isn't one of those — the unambiguous signature of leaked/
    malformed escaping the earlier steps didn't already catch).
    """
    if content.count("{") != content.count("}"):
        return False
    # "[" / "]" right after \left or \right are auto-SIZING delimiter
    # characters, not a macro-argument bracket pair — \left( paired
    # with \right] is completely valid LaTeX (e.g. a half-open
    # interval "\left(0, 1\right]"), so those don't need a same-type
    # partner and must be excluded before checking [ ] balance;
    # \left/\right's OWN pairing is already validated separately below.
    bracket_check_text = _LEFT_RIGHT_DELIM_CHAR.sub("", content)
    if bracket_check_text.count("[") != bracket_check_text.count("]"):
        return False
    depth = 0
    for m in _LEFT_RIGHT_TOKEN.finditer(content):
        depth += 1 if m.group(1) == "left" else -1
        if depth < 0:
            return False
    if depth != 0:
        return False
    if not _begin_end_environments_balanced(content):
        return False
    if _ASSAMESE_BENGALI_BLOCK.search(content):
        return False
    if content.rstrip().endswith("\\"):
        return False
    for m in _COMMAND_TOKEN.finditer(content):
        token = m.group(1)
        if len(token) == 1:
            if token not in _VALID_SYMBOL_ESCAPES and not token.isalpha():
                return False
        elif token not in KNOWN_LATEX_MACROS:
            # AUDIT FIX (see math_sanitizer_audit_notes below): a
            # multi-letter token NOT in KNOWN_LATEX_MACROS used to be
            # accepted outright on the theory that an unrecognized-but-
            # well-formed command "is not the failure mode this check
            # exists to catch". Empirically (headless-Chromium render
            # test against the vendored MathJax bundle) that's false —
            # an undefined control sequence like \gibberish does NOT
            # raise a MathJax parse error / <merror> node the way a
            # malformed \frac does; MathJax's TeX input silently falls
            # back to drawing the raw command name as literal glyphs
            # colored fill="red", with no data-mjx-error marker at all.
            # That's exactly the "broken LaTeX at render time" defect
            # this whole module exists to prevent, and it was
            # completely invisible to every check above (braces/left-
            # right/environments are all still perfectly balanced) and
            # to latex_grammar's grammar check (which validates argument
            # counts for known command shapes, not whether the command
            # itself exists). KNOWN_LATEX_MACROS is a deliberately wide,
            # curated allowlist for this class 6-10 math domain, so an
            # unknown token here is far more likely to be a Gemini
            # hallucination/typo (\sinx, \vecc, \gibberish) than a
            # legitimate macro this list is missing — and degrading it
            # to plain text (this span's fallback path) is a strictly
            # safer failure mode than shipping red glyph-soup in a
            # published PDF.
            return False

    # GRAMMAR CHECK (latex_grammar.py) — everything above is purely
    # STRUCTURAL (balance of braces/left-right/environments). It says
    # nothing about whether a command actually got the number of
    # arguments it requires (\frac{1} — one argument, needs two) or
    # whether every ^ / _ has a real argument (a dangling "x^" at the
    # end, or "x^^2" — a superscript whose "argument" is itself another
    # superscript). Those are GRAMMAR errors, not balance errors, and
    # every one is still a red MathJax parse-error box even though
    # every check above says "fine". A real recursive-descent parser
    # is the only generic way to catch this whole class at once,
    # rather than one more regex per shape.
    ok, _reason = latex_grammar.check_latex_grammar(content)
    if not ok:
        return False
    return True


# A conservative \frac{A}{B} -> "(A/B)"-style plain-text converter used
# only once a span has already failed _looks_structurally_valid() and
# is being degraded — never applied to valid, still-rendered math.
_SYMBOL_TO_PLAIN_TEXT = {
    "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·",
    "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "cong": "≅", "sim": "~", "propto": "∝",
    "infty": "∞", "angle": "∠", "triangle": "△", "perp": "⊥",
    "parallel": "∥", "degree": "°", "prime": "′",
    "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "Leftarrow": "⇐", "leftrightarrow": "↔", "to": "→",
    "therefore": "∴", "because": "∵",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆",
    "supset": "⊃", "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π",
    "sigma": "σ", "phi": "φ", "psi": "ψ", "omega": "ω",
    "sqrt": "√", "cdots": "⋯", "ldots": "…", "mid": "|", "circ": "∘",
    "left": "", "right": "", "big": "", "Big": "", "bigg": "", "Bigg": "",
    "quad": " ", "qquad": "  ", "text": "", "mathrm": "", "boxed": "",
}

_BRACE_GROUP = re.compile(r"\{([^{}]*)\}")
_FRAC_CMD = re.compile(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}")
_SIMPLE_CMD_WITH_ARG = re.compile(r"\\([A-Za-z]+)\{([^{}]*)\}")
_BARE_CMD = re.compile(r"\\([A-Za-z]+)")

_SUPERSCRIPT_MAP = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                     "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
                     "+": "⁺", "-": "⁻"}
_SUBSCRIPT_MAP = {"0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
                   "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
                   "+": "₊", "-": "₋"}
_SUP_GROUP = re.compile(r"\^\{([^{}]*)\}")
_SUB_GROUP = re.compile(r"_\{([^{}]*)\}")
_SUP_BARE = re.compile(r"\^([0-9+\-])")
_SUB_BARE = re.compile(r"_([0-9+\-])")


def _to_unicode_script(inner: str, mapping: dict) -> str:
    """Converts `inner` to Unicode super/subscript characters if every
    character in it is mappable (digits/sign only); otherwise the
    argument isn't representable as a single Unicode script glyph run
    (e.g. a letter, or already-plain-text content), so just return it
    unmarked — still readable, never a leaked '^'/'_'/'{'/'}'."""
    if inner and all(c in mapping for c in inner):
        return "".join(mapping[c] for c in inner)
    return inner


def degrade_to_plain_text(content: str) -> str:
    """Converts a span that failed final render-validation into plain,
    always-readable text — the "if rendering still fails, automatically
    degrade to readable plain text instead of a broken red MathJax
    error" requirement. Never returns anything containing a bare
    backslash or an unmatched brace: repeatedly resolves \\frac{a}{b}
    into "(a/b)" (innermost braces first, so this also copes with a
    frac nested inside another argument), maps every other recognized
    symbol command to its plain Unicode character, strips \\left/
    \\right/\\begin/\\end/environment names entirely, downgrades any
    remaining unrecognized "\\command" to just its bare word (still
    legible, never a dangling backslash), and finally removes any
    leftover structural braces/brackets, which have no meaning once
    everything has been flattened to plain text.
    """
    text = content
    # Resolve \frac{a}{b} repeatedly (innermost first, since the
    # regex only matches brace groups with no nested braces of their
    # own — running it to a fixed point handles nested fracs).
    prev = None
    while prev != text:
        prev = text
        text = _FRAC_CMD.sub(lambda m: f"({m.group(1)}/{m.group(2)})", text)

    # \sqrt{a} -> √(a) ; \sqrt[n]{a} -> ⁿ√(a) handled generically enough
    # for this domain (n is almost always a small literal digit/word).
    text = re.sub(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}", lambda m: f"{m.group(1)}√({m.group(2)})", text)
    text = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: f"√({m.group(1)})", text)

    # \binom{n}{k} -> C(n,k)
    text = re.sub(r"\\d?binom\{([^{}]*)\}\{([^{}]*)\}", lambda m: f"C({m.group(1)},{m.group(2)})", text)

    text = _BEGIN_END_TOKEN.sub("", text)
    text = text.replace("\\\\", "; ").replace("&", ", ")

    # ^ / _ that survive to here belong to a span that failed
    # validation for some OTHER reason (a missing \frac argument, a
    # mismatched environment, ...) — the script operator itself may
    # still be perfectly fine and deserves a readable plain-text
    # rendering rather than a leaked bare "^"/"_" character. A
    # digit/sign run (the overwhelmingly common case: exponents,
    # point-label subscripts) becomes real Unicode super/subscript
    # characters; anything else just has the marker character(s)
    # dropped so the base and its intended argument still read
    # together instead of leaving a stray caret/underscore in the
    # output.
    text = _SUP_GROUP.sub(lambda m: _to_unicode_script(m.group(1), _SUPERSCRIPT_MAP), text)
    text = _SUB_GROUP.sub(lambda m: _to_unicode_script(m.group(1), _SUBSCRIPT_MAP), text)
    text = _SUP_BARE.sub(lambda m: _to_unicode_script(m.group(1), _SUPERSCRIPT_MAP), text)
    text = _SUB_BARE.sub(lambda m: _to_unicode_script(m.group(1), _SUBSCRIPT_MAP), text)
    text = text.replace("^", "").replace("_", "")

    # \vec{X}, \overline{AB}, \hat{X}, etc. -> keep the inner text,
    # drop the decoration (a plain-text rendering has no accent marks).
    # Must run AFTER the \begin/\end strip above — that regex and this
    # one both match a bare "\word{...}" shape, and \begin{matrix} /
    # \end{pmatrix} would otherwise be mistaken for a decoration
    # command here, leaking the environment's own name ("matrix",
    # "pmatrix") into the output as if it were content.
    text = _SIMPLE_CMD_WITH_ARG.sub(lambda m: m.group(2), text)

    # Remaining known bare symbol commands (longest names first so
    # e.g. "leq" isn't cut short by a shorter unrelated match).
    for name in sorted(_SYMBOL_TO_PLAIN_TEXT, key=len, reverse=True):
        text = re.sub(r"\\" + name + r"\b", _SYMBOL_TO_PLAIN_TEXT[name], text)

    # Any command still left (unrecognized, but well-formed) — keep
    # the word, drop the backslash, so it stays legible instead of
    # vanishing or leaving a dangling "\" character.
    text = _BARE_CMD.sub(lambda m: m.group(1), text)

    # Whatever structural punctuation is left over has no meaning once
    # everything is flattened to plain text.
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    return text


_UNICODE_MATH_SYMBOL_TO_LATEX = {
    "×": " \\times ", "÷": " \\div ", "±": " \\pm ", "∓": " \\mp ",
    "≤": " \\leq ", "≥": " \\geq ", "≠": " \\neq ", "≈": " \\approx ",
    "≡": " \\equiv ", "∼": " \\sim ", "∝": " \\propto ",
    "∞": " \\infty ", "∠": " \\angle ", "△": " \\triangle ",
    "⊥": " \\perp ", "∥": " \\parallel ",
    "∴": " \\therefore ", "∵": " \\because ",
    "→": " \\rightarrow ", "⇒": " \\Rightarrow ", "↔": " \\leftrightarrow ",
    "∈": " \\in ", "∉": " \\notin ", "⊂": " \\subset ", "⊆": " \\subseteq ",
    "∪": " \\cup ", "∩": " \\cap ", "∀": " \\forall ", "∃": " \\exists ",
    "·": " \\cdot ", "⋯": " \\cdots ", "…": " \\ldots ",
    "π": " \\pi ", "θ": " \\theta ", "α": " \\alpha ", "β": " \\beta ",
    "λ": " \\lambda ", "μ": " \\mu ", "σ": " \\sigma ", "φ": " \\phi ",
    "ω": " \\omega ", "Δ": " \\Delta ", "Σ": " \\Sigma ",
}

_UNICODE_SQRT_PAREN_ARG = re.compile(r"√\(([^()]*)\)")
_UNICODE_SQRT_BRACE_ARG = re.compile(r"√\{([^{}]*)\}")
_UNICODE_SQRT_BARE_ARG = re.compile(r"√([0-9]+(?:\.[0-9]+)?|[A-Za-z])")
_UNICODE_DEGREE = re.compile(r"(?<!\^)°")


def normalize_unicode_math_symbols(content: str) -> str:
    """Converts bare Unicode math characters (√, ×, ÷, π, ...) that
    ended up sitting directly inside a \\( \\) span into their real
    LaTeX command equivalents (\\sqrt{...}, \\times, \\div, \\pi, ...).

    PRODUCTION-AUDIT FIX (root cause of a THIRD distinct class of "red
    MathJax error box", found analyzing a real irrational-numbers
    exercise (surds/radicals) where every previous check passed:
    braces balanced, brackets balanced, \\left/\\right balanced,
    \\begin/\\end balanced — content like "\\(2√5 × 2\\)" or
    "\\((√7)² - (√6)²\\)" is completely well-STRUCTURED, so
    _looks_structurally_valid correctly said "fine" and nothing
    repaired it. But MathJax's TeX input processor expects actual LaTeX
    commands inside math mode, not the bare Unicode glyphs a person
    (or an LLM writing loosely) would type in ordinary prose — a
    literal "√" or "×" character sitting directly in \\( \\) content is
    not guaranteed valid TeX input and can throw exactly this "raw
    text shown in red" parse error, regardless of how well-balanced
    the surrounding brackets are. Structure was never the problem here;
    the CONTENT was using the wrong alphabet for a LaTeX span.

    Deliberately conservative and mechanical, same spirit as every
    other repair in this module: "√" immediately followed by a
    parenthesized or braced group becomes \\sqrt{that group}; "√"
    immediately followed by a single bare number or letter (the
    overwhelmingly common case — √5, √2, √10) becomes \\sqrt{that
    token}; every other recognized Unicode operator/Greek-letter
    symbol is replaced by its plain LaTeX command via one shared
    lookup table. This runs BEFORE the structural checks below, so the
    result is validated exactly like any other span — if something
    about the surrounding content is still broken after this, the
    normal repair-or-degrade path still applies.
    """
    content = _UNICODE_SQRT_PAREN_ARG.sub(lambda m: f"\\sqrt{{{m.group(1)}}}", content)
    content = _UNICODE_SQRT_BRACE_ARG.sub(lambda m: f"\\sqrt{{{m.group(1)}}}", content)
    content = _UNICODE_SQRT_BARE_ARG.sub(lambda m: f"\\sqrt{{{m.group(1)}}}", content)
    content = _UNICODE_DEGREE.sub("^{\\\\circ}", content)
    for sym, repl in _UNICODE_MATH_SYMBOL_TO_LATEX.items():
        if sym in content:
            content = content.replace(sym, repl)
    return content


#   - a signed run ("-1", "+3", any length) — a bare "^-" is already a
#     grammatically valid (single-token) argument on its own, so
#     without this the sign silently detaches from its digits, e.g.
#     "x^-1" renders as x-superscript-minus followed by a normal "1".
#   - an unsigned run of 2+ digits, or any run containing a decimal
#     point — the multi-character case ^ and _ can't bind to as one
#     token in real LaTeX to begin with.
# A single unsigned digit ("x^2") is deliberately left untouched: it
# is already both correct and already-idiomatic in this codebase's
# output, so rewriting it would only create pointless diffs.
_BARE_SCRIPT_RUN = re.compile(
    r"([\^_])(-[0-9]+(?:\.[0-9]+)?|\+[0-9]+(?:\.[0-9]+)?|[0-9]*\.[0-9]+|[0-9]{2,}(?:\.[0-9]+)?)(?!\})"
)


def normalize_sup_sub_grouping(content: str) -> str:
    """Wraps a multi-character NUMERIC exponent/subscript run that
    wasn't already grouped in braces — "x^23" -> "x^{23}", "a_-1" ->
    "a_{-1}" — the unambiguous repair for a very common, silent
    (never a MathJax error, so nothing above ever caught it) rendering
    bug: LaTeX's ^ and _ each bind to exactly ONE following token, so
    unbraced "x^23" actually renders as x-superscript-2 followed by a
    normal-size "3", not the two-digit exponent the author meant —
    visually indistinguishable from "corrupted/broken" formatting to
    anyone reading the output.

    Deliberately restricted to a DIGIT run (optionally signed): a
    single following ALPHABETIC character after ^/_ (e.g. "x^2y") is
    standard, correct, and semantically different notation — TeX's
    tight-binding behavior there is exactly what the author intended
    (y is a separate factor, not part of the exponent), so widening
    that case would silently change the expression's meaning instead
    of fixing a bug. A numeric run has no such ambiguity: nobody means
    "x to the 2nd power, immediately followed by a bare digit 3" when
    they type "x^23".
    """
    return _BARE_SCRIPT_RUN.sub(lambda m: f"{m.group(1)}{{{m.group(2)}}}", content)


def validate_and_finalize_spans(text: str) -> str:
    """Step 5: walks every already well-formed \\( \\) span (see
    _fix_nested_and_orphan_delimiters, which MUST have already run —
    this function assumes every "\\(" has exactly one matching "\\)")
    and, independently for each:
      - balances {} within it (unlike {}, "[" "]" are NOT auto-
        repaired the same way — this domain uses literal square
        brackets for real math content, e.g. a half-open interval
        "\\left(0, 1\\right]" or \\sqrt[n]{...}'s optional argument,
        so a naive "drop the orphan closer" rule would delete a
        perfectly legitimate "]"; an unbalanced bracket count is still
        caught by _looks_structurally_valid below, just routed to a
        full, safe degrade instead of a surgical (and here, wrong)
        deletion),
      - fixes mismatched \\left/\\right,
      - checks \\begin/\\end environments,
      - splits out any Assamese/Bengali script that ended up inside a
        math span (that script was never meant to be in math mode —
        the surrounding text just accidentally ran into the
        delimiter), by simply closing math before it and reopening
        after,
      - and, if the span is STILL not structurally valid after all of
        that, degrades it to plain text instead of ever publishing a
        broken "\\(" the reader would see literally or a red MathJax
        error box.
    """
    tokens = _split_into_delimiter_tokens(text)

    out = []
    i = 0
    n = len(tokens)
    while i < n:
        kind, chunk = tokens[i]
        if kind != "open":
            out.append(chunk)
            i += 1
            continue
        # tokens[i] is "open"; by construction of
        # _fix_nested_and_orphan_delimiters, tokens[i+1] is always the
        # matching "text" (possibly empty) and tokens[i+2] is "close".
        content = tokens[i + 1][1] if i + 1 < n and tokens[i + 1][0] == "text" else ""
        j = i + 2 if (i + 1 < n and tokens[i + 1][0] == "text") else i + 1

        content = normalize_unicode_math_symbols(content)
        content = normalize_sup_sub_grouping(content)
        content = _balance_bracket_pairs(content, "{", "}")
        content = _fix_left_right_pairing(content)

        assamese_match = _ASSAMESE_BENGALI_BLOCK.search(content)
        if assamese_match and _begin_end_environments_balanced(content):
            # Split the span at the non-math script boundary rather
            # than degrading a whole otherwise-fine expression just
            # because some surrounding prose leaked in: close math
            # right before it, reopen right after.
            start, end = assamese_match.start(), assamese_match.end()
            # Extend to the full contiguous non-math run, not just the
            # one matched character, so "৯টা মিটাৰ" doesn't get split
            # into several pointless open/close pairs.
            while end < len(content) and (_ASSAMESE_BENGALI_BLOCK.match(content[end]) or content[end].isspace()):
                end += 1
            while start > 0 and content[start - 1].isspace():
                start -= 1
            before, script, after = content[:start], content[start:end], content[end:]
            if _looks_structurally_valid(before) and _looks_structurally_valid(after):
                out.append("\\(" + before + "\\)" + script + "\\(" + after + "\\)")
                i = j + 1
                continue

        if _looks_structurally_valid(content):
            out.append("\\(" + content + "\\)")
        else:
            out.append(degrade_to_plain_text(content))
        i = j + 1

    return "".join(out)


def sanitize_math_text(text):
    """THE single mandatory gateway every math-bearing text field in
    this pipeline must pass through before HTML/PDF rendering —
    question/given/required/steps/final_answer/notes/hints/captions,
    regardless of which stage produced the text (the original solve,
    a Human Review correction, a diagram caption, anything).

    Order matters and mirrors the module docstring:
      1. decode_json_escape_leakage        — real "\\n\\t\\r\\b\\f", not
         literal backslash-letter text
      2. collapse_and_decode_escapes       — over-escaping + stray "\\uXXXX"
      3. _fix_nested_and_orphan_delimiters — normalize any \\( \\) that
         already exist into well-formed, non-nested, alternating pairs
         BEFORE step 4 — wrap_bare_latex's own "don't touch already-
         delimited spans" mask only works correctly on well-formed
         pairs; running it on a text with, say, an unclosed "\\(" left
         over to the end of the string would otherwise let the bare-
         LaTeX scanner treat that stray "\\(" as an ordinary character
         and swallow it into an unrelated wrap, corrupting output that
         wasn't even part of the original bug.
      4. wrap_bare_latex                   — delimit anything Gemini forgot to
      5. validate_and_finalize_spans       — fix/validate/degrade every span

    Non-string input is returned unchanged (mirrors every existing
    normalizer in this pipeline, which all tolerate a field Gemini sent
    as the wrong JSON type without crashing the whole render).
    """
    if not isinstance(text, str):
        return text
    if not text:
        return text
    result = text
    # FIXED-POINT ITERATION, not a single fixed sequence of steps:
    # wrap_bare_latex inserts NEW "\(" "\)" delimiters into the middle
    # of the text, and those can end up directly adjacent to backslash
    # characters that were already there — e.g. an original literal
    # "\\\\*" run up against a freshly-inserted "\(" becomes a NEW
    # 3-backslash run that no single step ever analyzed as one unit
    # (splicing two independently-fine strings together can create a
    # shape neither one had). One more pass catches and normalizes
    # exactly that residue. Capped and change-detected so the common
    # case (nothing left to do) costs one extra cheap no-op comparison,
    # never an unbounded loop.
    for _ in range(4):
        next_result = _sanitize_math_text_one_pass(result)
        if next_result == result:
            return next_result
        result = next_result
    return result


def _sanitize_math_text_one_pass(text: str) -> str:
    text = decode_json_escape_leakage(text)
    text = collapse_and_decode_escapes(text)
    text = _fix_nested_and_orphan_delimiters(text)
    text = wrap_bare_latex(text)
    text = validate_and_finalize_spans(text)
    return text


# ----------------------------------------------------------------------
# SVG DIAGRAM LABELS — a second, independent entry point into this same
# gateway.
#
# ROOT-CAUSE NOTE: diagram_renderer.py builds every point/side/angle
# label as a raw SVG <text>...</text> string, computed and positioned
# deterministically in Python — Gemini never writes markup there, only
# short label strings (a point name, a measured value, an angle). BUT
# if Gemini ever puts LaTeX-flavoured text into one of those strings
# (a stray "\circ" for a degree symbol, "x^2", "\frac{1}{2}" as a side
# length, ...) — which the prompts ask it not to do, but this pipeline
# never trusts a prompt instruction as its only line of defense
# anywhere else — it lands DIRECTLY in the SVG as literal text with no
# sanitization at all, and unlike question/given/required/steps/
# final_answer it never passes through sanitize_math_text(). SVG
# <text> content is also never MathJax-typeset the way \\( \\) spans in
# the surrounding HTML are, so wrapping it in \\( \\) would not even
# fix it — this needs its own, always-plain-text-output entry point.
# ----------------------------------------------------------------------
_TEXT_ELEMENT_PATTERN = re.compile(r"(<text\b[^>]*>)(.*?)(</text>)", re.DOTALL)
_DEGREE_CIRC_PATTERN = re.compile(r"\^\s*\{?\s*\\circ\s*\}?")


def _sanitize_one_svg_label(text: str) -> str:
    """Guarantees a single SVG <text> element's content can never carry
    a leaked backslash, brace, or LaTeX command through to the printed
    page — always degrades to plain, readable text (SVG text nodes are
    never MathJax-typeset, so there is no valid-math path to take
    here, only a safe one)."""
    if not text:
        return text
    cleaned = decode_json_escape_leakage(text)
    cleaned = collapse_and_decode_escapes(cleaned)
    if "\\" not in cleaned and cleaned.count("{") == cleaned.count("}"):
        # No LaTeX-shaped content and nothing structurally broken —
        # ordinary label text (point names, plain numbers, Assamese/
        # English prose) is left exactly as-is.
        return cleaned
    # The very common "60^\circ" (or "60^{\circ}") degree-angle idiom
    # reads correctly as plain text once turned into "60°" — handled
    # before the generic degrade below so it doesn't fall back to the
    # less legible generic \circ -> "∘" substitution.
    cleaned = _DEGREE_CIRC_PATTERN.sub("°", cleaned)
    cleaned = normalize_sup_sub_grouping(cleaned)
    cleaned = _balance_bracket_pairs(cleaned, "{", "}")
    return degrade_to_plain_text(cleaned)


def sanitize_svg_labels(svg: str) -> str:
    """THE mandatory gateway for diagram SVG text — the second half of
    this module's "single generic gateway" guarantee. Every renderer
    in diagram_renderer.py funnels through render_diagram()'s one
    return path, and that is where this is called: no individual
    renderer or plugin needs to remember to sanitize its own labels.
    Non-string input, or a string with no <text> element at all, is
    returned completely untouched (cheap no-op — the overwhelmingly
    common case, since virtually every label is already plain text).
    """
    if not isinstance(svg, str) or "<text" not in svg:
        return svg
    return _TEXT_ELEMENT_PATTERN.sub(
        lambda m: m.group(1) + _sanitize_one_svg_label(m.group(2)) + m.group(3),
        svg,
    )


