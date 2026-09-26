"""
latex_grammar.py — a real LaTeX-math GRAMMAR checker: tokenizer +
recursive-descent parser, not a text-replacement/regex heuristic.

WHY THIS EXISTS
----------------
math_sanitizer.py's structural checks (brace/bracket balance,
\\left/\\right balance, \\begin/\\end name matching) catch the "leaked
delimiter" family of corruption, but they say nothing about whether a
span is actually GRAMMATICAL LaTeX. A string like

    \\frac{1}          (missing the 2nd required argument)
    x^                 (dangling superscript operator, no argument)
    x^^2               (a superscript given ANOTHER superscript as its
                         argument instead of a real one)

is perfectly brace-balanced (there may be no braces at all) and yet is
a MathJax parse error ("Missing argument for \\frac", "Missing
superscript", ...) every single time. Those are not one-off shapes to
regex away one at a time — they are all instances of the same general
fact: some LaTeX commands (\\frac, \\sqrt, ^, _, \\vec, \\text, ...)
require a fixed number of arguments, and an argument is either a
{...} group or a single atomic token. That is a GRAMMAR, so it needs a
parser, not a pattern list.

This module tokenizes a math span into a flat stream of LaTeX tokens,
then recursively descends through it the same way a real TeX/MathJax
parser would: consuming exactly the arguments each command's arity
requires, recursing into every {...} group and \\begin{...}...\\end{...}
environment, and validating every ^ / _ has exactly one real argument.
It has NO knowledge of any specific "known bad shape" — a brand-new
command this domain has never used before gets the exact same
treatment as \\frac: if it's not in the (small, honest) arity table,
it is simply treated as taking zero arguments, exactly like every
other unknown symbol command already accepted elsewhere in this
pipeline.

This module is intentionally side-effect-free and dependency-free
(pure functions, no imports from math_sanitizer) so it can be tested,
reasoned about, and reused (e.g. from diagram_renderer.py) in
isolation from the rest of the sanitization pipeline.
"""
from __future__ import annotations


class GrammarError(Exception):
    """Raised internally when a span fails to parse as grammatical
    LaTeX. Never escapes check_latex_grammar()."""


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------
# Token kinds: 'cmd' (a \letters command name, value = name, no
# leading backslash), 'symcmd' (\<single-non-letter>, e.g. \{ \% \&),
# 'row' (the literal two-backslash "\\" row separator), 'lbrace',
# 'rbrace', 'lbracket', 'rbracket', 'sup' (^), 'sub' (_), 'amp' (&),
# 'char' (anything else: letters, digits, operators, spaces, ...).

def tokenize(text: str):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            if i + 1 < n and text[i + 1] == "\\":
                tokens.append(("row", "\\\\", i))
                i += 2
                continue
            if i + 1 < n and text[i + 1].isalpha():
                j = i + 1
                while j < n and text[j].isalpha():
                    j += 1
                tokens.append(("cmd", text[i + 1:j], i))
                i = j
                continue
            if i + 1 < n:
                tokens.append(("symcmd", text[i + 1], i))
                i += 2
                continue
            # Dangling backslash at end-of-string.
            tokens.append(("cmd", "", i))
            i += 1
            continue
        if c == "{":
            tokens.append(("lbrace", c, i)); i += 1; continue
        if c == "}":
            tokens.append(("rbrace", c, i)); i += 1; continue
        if c == "[":
            tokens.append(("lbracket", c, i)); i += 1; continue
        if c == "]":
            tokens.append(("rbracket", c, i)); i += 1; continue
        if c == "^":
            tokens.append(("sup", c, i)); i += 1; continue
        if c == "_":
            tokens.append(("sub", c, i)); i += 1; continue
        if c == "&":
            tokens.append(("amp", c, i)); i += 1; continue
        tokens.append(("char", c, i))
        i += 1
    return tokens


# ----------------------------------------------------------------------
# Arity table — the ONLY place command arity is declared. A command
# absent from every set here is treated as arity 0 (a bare symbol),
# which is always safe: it just means this checker won't notice if
# THAT specific unknown command is itself missing an argument, exactly
# matching this codebase's existing "unrecognized-but-well-formed is
# not a failure mode we claim to catch" stance for macros in general.
# ----------------------------------------------------------------------
ARITY_2 = frozenset({"frac", "dfrac", "tfrac", "binom", "dbinom"})

ARITY_1 = frozenset({
    "vec", "overrightarrow", "overleftarrow", "overline", "underline",
    "hat", "widehat", "tilde", "widetilde", "dot", "ddot", "bar",
    "overbrace", "underbrace", "check", "breve", "acute", "grave",
    "text", "mathrm", "mathbf", "mathit", "mathsf", "mathtt",
    "textbf", "textit", "boldsymbol", "operatorname", "boxed", "fbox",
    "mathbb", "pmod",
})

# \sqrt is special: an OPTIONAL [n] followed by exactly one required
# argument.
SQRT_COMMANDS = frozenset({"sqrt"})

# \left / \right consume exactly one delimiter token (a char or
# symcmd) and are not otherwise argument-bearing.
_LEFT_RIGHT = frozenset({"left", "right"})


def _parse_argument(tokens, i, n):
    """Consumes exactly ONE argument for ^ / _ / a 1-arg command:
    either a {...} group (recursively validated) or a single atomic
    token (a char, a symcmd, or a command together with ITS OWN
    arguments). Returns the index just past the argument, or raises
    GrammarError if there is no valid argument to consume."""
    if i >= n:
        raise GrammarError("missing argument at end of expression")
    kind, val, pos = tokens[i]
    if kind == "lbrace":
        return _parse_group(tokens, i, n)
    if kind == "cmd":
        if val == "":
            raise GrammarError("dangling backslash")
        return _parse_command(tokens, i, n)
    if kind == "symcmd":
        return i + 1
    if kind == "char":
        return i + 1
    # rbrace / rbracket / sup / sub / amp / row can never themselves
    # serve as an argument.
    raise GrammarError(f"missing argument (found {kind!r} instead)")


def _consume_trailing_scripts(tokens, i, n):
    """After any atom, ^ and/or _ may follow (in either order, at most
    once each in a row) — each one consumes exactly one argument via
    _parse_argument. A ^ or _ appearing where an argument was expected
    (e.g. immediately after another ^) is caught by _parse_argument's
    own kind check above ('missing argument (found sup ...)'), which
    is exactly the generic form of the classic 'double superscript'
    MathJax error."""
    while i < n and tokens[i][0] in ("sup", "sub"):
        i += 1
        i = _parse_argument(tokens, i, n)
    return i


def _parse_command(tokens, i, n):
    kind, name, pos = tokens[i]
    i += 1
    if name == "":
        raise GrammarError("dangling backslash")
    if name in SQRT_COMMANDS:
        if i < n and tokens[i][0] == "lbracket":
            i += 1
            while i < n and tokens[i][0] != "rbracket":
                if tokens[i][0] == "lbrace":
                    i = _parse_group(tokens, i, n)
                else:
                    i += 1
            if i >= n:
                raise GrammarError(r"unbalanced [ ] in \sqrt")
            i += 1  # consume ']'
        return _parse_argument(tokens, i, n)
    if name in ARITY_2:
        i = _parse_argument(tokens, i, n)
        i = _parse_argument(tokens, i, n)
        return i
    if name in ARITY_1:
        return _parse_argument(tokens, i, n)
    if name in _LEFT_RIGHT:
        # The delimiter after \left / \right is a single token: an
        # ordinary char ('(' ')' '|' '.' ...), a symcmd (\{ \} \| ...),
        # or a literal '[' ']' (their own dedicated token kinds).
        if i >= n or tokens[i][0] not in ("char", "symcmd", "lbracket", "rbracket"):
            raise GrammarError(fr"\{name} without a delimiter")
        return i + 1
    # Unknown or genuinely zero-arity command (\pi, \times, \alpha, an
    # unrecognized future macro, ...) — consumes no arguments.
    return i


def _parse_environment(tokens, i, n):
    """tokens[i] is the 'cmd' token for 'begin'."""
    i += 1
    i, env_name = _parse_env_name(tokens, i, n, "begin")
    i = _parse_sequence(tokens, i, n, stop_on_rbrace=False, stop_on_end=True)
    if i >= n or not (tokens[i][0] == "cmd" and tokens[i][1] == "end"):
        raise GrammarError(fr"\begin{{{env_name}}} without matching \end")
    i += 1
    i, end_name = _parse_env_name(tokens, i, n, "end")
    if end_name != env_name:
        raise GrammarError(fr"\begin{{{env_name}}} / \end{{{end_name}}} mismatch")
    return i


def _parse_env_name(tokens, i, n, which):
    if i >= n or tokens[i][0] != "lbrace":
        raise GrammarError(fr"\{which} without {{name}}")
    i += 1
    chars = []
    while i < n and tokens[i][0] == "char":
        chars.append(tokens[i][1])
        i += 1
    if i >= n or tokens[i][0] != "rbrace":
        raise GrammarError(fr"malformed \{which}{{...}}")
    return i + 1, "".join(chars)


def _parse_group(tokens, i, n):
    """tokens[i] is 'lbrace'."""
    i = _parse_sequence(tokens, i + 1, n, stop_on_rbrace=True, stop_on_end=False)
    if i >= n or tokens[i][0] != "rbrace":
        raise GrammarError("unbalanced { }")
    return i + 1


def _parse_sequence(tokens, i, n, stop_on_rbrace, stop_on_end):
    while i < n:
        kind, val, pos = tokens[i]
        if kind == "rbrace":
            if stop_on_rbrace:
                return i
            raise GrammarError("unmatched }")
        if kind == "cmd" and val == "end":
            if stop_on_end:
                return i
            raise GrammarError(r"orphan \end")
        if kind == "cmd" and val == "begin":
            i = _parse_environment(tokens, i, n)
            i = _consume_trailing_scripts(tokens, i, n)
            continue
        if kind == "lbrace":
            i = _parse_group(tokens, i, n)
            i = _consume_trailing_scripts(tokens, i, n)
            continue
        if kind == "cmd":
            i = _parse_command(tokens, i, n)
            i = _consume_trailing_scripts(tokens, i, n)
            continue
        if kind in ("sup", "sub"):
            # A script with no preceding base (rare, but not a hard
            # TeX error — e.g. a bare "^2" fragment) still needs a
            # real argument.
            i += 1
            i = _parse_argument(tokens, i, n)
            continue
        # symcmd / char / rbracket / lbracket / amp / row: ordinary
        # atoms that don't themselves open anything.
        i += 1
        i = _consume_trailing_scripts(tokens, i, n)
    return i


def check_latex_grammar(content: str):
    """Returns (True, '') if `content` (the text INSIDE a \\( \\) span,
    no delimiters) parses as grammatical LaTeX, else (False, reason).
    Never raises — any internal problem is reported as a failed check
    rather than propagated, so a bug in this checker can only ever
    make sanitization MORE conservative (degrade to plain text), never
    crash the pipeline.
    """
    try:
        tokens = tokenize(content)
        end = _parse_sequence(tokens, 0, len(tokens), stop_on_rbrace=False, stop_on_end=False)
        if end != len(tokens):
            kind = tokens[end][0] if end < len(tokens) else "?"
            return False, f"trailing unparsed token ({kind})"
        return True, ""
    except GrammarError as e:
        return False, str(e)
    except Exception as e:  # pragma: no cover - defensive, see docstring
        return False, f"grammar checker internal error: {e}"
