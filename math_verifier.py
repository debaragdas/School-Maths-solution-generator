"""
math_verifier.py — deterministic, symbolic (sympy-based) re-verification
of solved answers, for the subset of Class 6-10 questions where this is
genuinely tractable without another AI call.

WHY THIS EXISTS (from the V7 engineering audit, item E1): before this
module, the ONLY answer-correctness re-check anywhere in the pipeline
was solver.py's coordinate-reading self-verification — every other
answer type (arithmetic simplification, linear equations, algebra,
proofs, geometry, trig) was trusted from a single Gemini pass with only
STRUCTURAL sanity checks (utils.verify_solution: non-empty steps,
balanced $ delimiters, sequential numbering) — none of which can catch
a fluently-written, correctly-formatted, but numerically WRONG answer.

SCOPE (deliberately narrow, by design, not an oversight): this module
only attempts verification for the two sub-classes of question that are
genuinely deterministic and unambiguous to re-derive from text alone:

  1. Closed-form ARITHMETIC simplification/evaluation — "Simplify:
     3/4 + 1/2", "Evaluate: 2^3 + 5*(4-1)" — where the question states
     a self-contained numeric expression and the final answer is a
     single number or simple fraction.

  2. Single-variable LINEAR EQUATIONS — "Solve for x: 2x + 3 = 11" —
     where the final answer states "x = <value>"; verified by
     substituting the value back into the ORIGINAL equation and
     checking both sides agree (residual == 0), which is exact and
     unambiguous regardless of how the question was solved.

Proofs, multi-step geometry, word problems, and anything requiring the
free-form step narrative to be understood are explicitly OUT OF SCOPE
here — attempting to symbolically re-derive those from unstructured
text would be unreliable, and a false "contradicted" flag on a
correct answer is its own kind of harm (erodes trust in every other
flag this pipeline raises). Every extraction step below is
deliberately conservative: if a question's given/answer text doesn't
match one of the two well-defined patterns above with high confidence,
verify_question() returns "not_applicable" rather than guessing at a
looser match.

FAIL-SAFE POLICY (matches solver.py's coordinate tie-break, and the
project's overall "if it cannot be verified, fail safely" principle):
a CONTRADICTION found here never auto-rewrites final_answer or steps —
sympy's own reading of a free-form question could itself be a
mis-extraction, so silently "fixing" the answer would carry the same
risk the coordinate tie-break fix was built to eliminate. Instead a
contradiction sets needs_review=True with a review note showing the
independently recomputed value alongside the original, for a human to
resolve — never guessed, never silently published either way.
"""
import re

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

_ASSAMESE_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# Deliberately conservative: only digits, one decimal point, the four
# basic operators, ^ (power), parentheses, and whitespace. Anything
# else in a candidate expression (letters other than a lone variable,
# unrecognized symbols, multiple statements) means "don't guess" —
# extraction returns None and verification is skipped for that question.
_SAFE_ARITHMETIC_CHARS = re.compile(r"^[0-9\.\+\-\*/\^\(\)\s]+$")

_SIMPLIFY_CUE = re.compile(
    r"(?:simplify|evaluate|calculate|find the value of|সমাধান কৰা|নিৰ্ণয় কৰা)\s*[:\-]?\s*",
    re.IGNORECASE,
)

# "Solve for x: 2x + 3 = 11" / "Solve: 2x+3=11" style — captures the
# equation text after the cue. Deliberately requires the word "solve"
# (or its Assamese equivalent) rather than firing on any bare
# "<expr> = <expr>" in the text, since plenty of non-equation-solving
# questions legitimately contain an "=" sign (e.g. definitions).
_SOLVE_EQUATION_CUE = re.compile(
    r"(?:solve(?:\s+for\s+[a-zA-Z])?|সমাধান কৰা)\s*[:\-]?\s*"
    r"([0-9a-zA-Z\.\+\-\*/\^\(\)\s]+=[0-9a-zA-Z\.\+\-\*/\^\(\)\s]+)",
    re.IGNORECASE,
)

_FINAL_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")

_VAR_VALUE_PATTERN = re.compile(r"\b([a-zA-Z])\s*=\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)")


def _normalize_ascii(text: str) -> str:
    return (text or "").translate(_ASSAMESE_DIGITS).replace("×", "*").replace("÷", "/").replace("−", "-")


def _safe_parse(expr_text: str):
    """Parses a short arithmetic/algebraic expression with sympy, but
    ONLY if it consists exclusively of characters this module already
    knows are safe (see _SAFE_ARITHMETIC_CHARS) — never hands raw
    free-form text to sympy's parser. Returns None (never raises) on
    anything that doesn't parse cleanly.

    IMPORTANT: '^' is translated to '**' before parsing. sympy's
    parser treats a bare '^' as Python's bitwise XOR operator, NOT
    exponentiation (confirmed: parsing "2^3" with sympy's default
    parser silently evaluates to 1, not 8) — while every question in
    this project's own notation (and everyday informal math notation)
    uses '^' to mean "to the power of". Without this translation this
    module would silently misjudge any power-containing expression as
    contradicted, exactly the kind of false alarm the module's own
    docstring warns against — or worse, silently confirm a wrong
    answer that happens to match the XOR misreading.
    """
    expr_text = expr_text.strip().replace("^", "**")
    if not expr_text:
        return None
    try:
        return parse_expr(expr_text, transformations=_TRANSFORMATIONS)
    except Exception:
        return None


def extract_arithmetic_expression(text: str) -> str | None:
    """Finds a 'Simplify/Evaluate: <expr>' style cue and returns the
    expression text if — and only if — everything after the cue up to
    the end of that sentence is composed exclusively of safe arithmetic
    characters (digits, + - * / ^ ( ) . and whitespace). Returns None
    for anything else, including expressions containing a variable
    (those aren't "evaluate", they're "solve" — handled separately) or
    any character this module doesn't have high confidence about."""
    text = _normalize_ascii(text)
    m = _SIMPLIFY_CUE.search(text)
    if not m:
        return None
    rest = text[m.end():]
    # take up to the first sentence-ending punctuation or line break,
    # since the source text often continues with more question prose.
    # IMPORTANT: only treat '.' as a sentence boundary when it is NOT
    # immediately followed by a digit — otherwise a decimal point
    # inside the expression itself (e.g. "5.5 - 2.25") gets mistaken
    # for the end of the sentence and the expression is truncated to
    # just "5", producing a nonsense verification.
    candidate = re.split(r"।|\n|\.(?!\d)", rest, maxsplit=1)[0].strip()
    if not candidate or not _SAFE_ARITHMETIC_CHARS.match(candidate):
        return None
    return candidate


def extract_linear_equation(text: str) -> tuple[str, str] | None:
    """Finds a 'Solve [for x]: <lhs> = <rhs>' cue and returns
    (lhs_text, rhs_text) if the equation contains EXACTLY ONE distinct
    variable letter and both sides are otherwise composed of safe
    arithmetic characters plus that one letter. Returns None for
    anything with zero or more-than-one variable (simultaneous
    equations / multi-variable systems are out of scope here — a wrong
    guess at which variable to solve for is worse than not checking)."""
    text = _normalize_ascii(text)
    m = _SOLVE_EQUATION_CUE.search(text)
    if not m:
        return None
    equation_text = m.group(1)
    if "=" not in equation_text:
        return None
    lhs, rhs = equation_text.split("=", 1)
    lhs, rhs = lhs.strip(), rhs.strip()

    letters = set(re.findall(r"[a-zA-Z]", lhs + rhs))
    if len(letters) != 1:
        return None  # zero variables (not an equation to solve) or 2+ (system) -> skip

    var = next(iter(letters))
    safe_with_var = re.compile(rf"^[0-9\.\+\-\*/\^\(\)\s{re.escape(var)}]+$")
    if not (safe_with_var.match(lhs) and safe_with_var.match(rhs)):
        return None
    return lhs, rhs


def extract_single_final_number(final_answer_text: str) -> sympy.Rational | None:
    """Extracts a single number or simple fraction (p/q) from a
    final_answer string, e.g. 'The answer is 5/4.' -> Rational(5, 4).
    Returns None if zero or more than one plausible numeric token is
    found (ambiguous -> don't guess which one is the actual answer)."""
    text = _normalize_ascii(final_answer_text)
    matches = _FINAL_NUMBER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    try:
        return sympy.Rational(_safe_parse(matches[0].replace(" ", "")))
    except Exception:
        return None


def extract_variable_value(final_answer_text: str, variable: str) -> sympy.Rational | None:
    """Extracts 'x = <value>' (for the given variable letter) from a
    final_answer string. Returns None if that exact 'var = number'
    pattern isn't found unambiguously."""
    text = _normalize_ascii(final_answer_text)
    matches = [v for letter, v in _VAR_VALUE_PATTERN.findall(text) if letter == variable]
    if len(matches) != 1:
        return None
    try:
        return sympy.Rational(_safe_parse(matches[0].replace(" ", "")))
    except Exception:
        return None


def verify_arithmetic(question_text: str, given: str, final_answer: str) -> dict:
    """Returns {'status': 'confirmed'|'contradicted'|'not_applicable', 'detail': str}."""
    expr_text = extract_arithmetic_expression(question_text) or extract_arithmetic_expression(given)
    if expr_text is None:
        return {"status": "not_applicable", "detail": ""}

    expr = _safe_parse(expr_text)
    if expr is None:
        return {"status": "not_applicable", "detail": ""}

    stated = extract_single_final_number(final_answer)
    if stated is None:
        return {"status": "not_applicable", "detail": ""}

    try:
        computed = sympy.nsimplify(expr.evalf(30), rational=True)
        computed_simplified = sympy.simplify(computed)
        stated_simplified = sympy.simplify(stated)
    except Exception:
        return {"status": "not_applicable", "detail": ""}

    if computed_simplified == stated_simplified:
        return {"status": "confirmed",
                "detail": f"'{expr_text}' independently recomputed to {computed_simplified}, matching the stated answer."}
    return {"status": "contradicted",
            "detail": f"'{expr_text}' independently recomputes to {computed_simplified}, "
                      f"but the stated final answer is {stated_simplified}."}


def verify_linear_equation(question_text: str, given: str, final_answer: str) -> dict:
    equation = extract_linear_equation(question_text) or extract_linear_equation(given)
    if equation is None:
        return {"status": "not_applicable", "detail": ""}
    lhs_text, rhs_text = equation

    letters = set(re.findall(r"[a-zA-Z]", lhs_text + rhs_text))
    if len(letters) != 1:
        return {"status": "not_applicable", "detail": ""}
    variable_letter = next(iter(letters))

    lhs_expr, rhs_expr = _safe_parse(lhs_text), _safe_parse(rhs_text)
    if lhs_expr is None or rhs_expr is None:
        return {"status": "not_applicable", "detail": ""}

    stated_value = extract_variable_value(final_answer, variable_letter)
    if stated_value is None:
        return {"status": "not_applicable", "detail": ""}

    try:
        sym = sympy.Symbol(variable_letter)
        residual = sympy.simplify(lhs_expr.subs(sym, stated_value) - rhs_expr.subs(sym, stated_value))
    except Exception:
        return {"status": "not_applicable", "detail": ""}

    if residual == 0:
        return {"status": "confirmed",
                "detail": f"Substituting {variable_letter}={stated_value} into "
                          f"'{lhs_text} = {rhs_text}' balances exactly."}

    try:
        solved = sympy.solve(sympy.Eq(lhs_expr, rhs_expr), sym)
        correct_value = solved[0] if solved else None
    except Exception:
        correct_value = None

    detail = (f"Substituting the stated {variable_letter}={stated_value} into "
              f"'{lhs_text} = {rhs_text}' does NOT balance (difference = {residual}).")
    if correct_value is not None:
        detail += f" Independently solving the equation gives {variable_letter}={correct_value}."
    return {"status": "contradicted", "detail": detail}


def verify_question(question: dict) -> dict:
    """Dispatches a solved question dict (with question_text/given/
    final_answer keys, matching solver.py's schema) to whichever
    deterministic check (if any) applies. Never raises — any internal
    failure degrades to 'not_applicable' rather than blocking the
    pipeline, consistent with every other optional-enhancement step
    here (book-diagram attachment, coordinate self-check)."""
    question_text = question.get("question_text", "") or ""
    given = question.get("given", "") or ""
    final_answer = question.get("final_answer", "") or ""

    try:
        result = verify_linear_equation(question_text, given, final_answer)
        if result["status"] != "not_applicable":
            result["check"] = "linear_equation"
            return result

        result = verify_arithmetic(question_text, given, final_answer)
        if result["status"] != "not_applicable":
            result["check"] = "arithmetic"
            return result
    except Exception:
        pass

    return {"status": "not_applicable", "detail": "", "check": None}


def verify_questions(questions: list) -> tuple[int, int]:
    """Runs verify_question() over every question in place, setting
    needs_review/review_notes on any contradiction found (same
    fail-safe pattern as solver.py's coordinate tie-break — a
    contradiction is FLAGGED, never used to silently rewrite
    final_answer/steps). Returns (confirmed_count, contradicted_count)
    for the caller to log."""
    confirmed = 0
    contradicted = 0
    for q in questions:
        result = verify_question(q)
        if result["status"] == "confirmed":
            confirmed += 1
        elif result["status"] == "contradicted":
            contradicted += 1
            q["needs_review"] = True
            q.setdefault("review_notes", []).append(
                f"Deterministic {result['check']} check disagrees with the solved answer: {result['detail']}"
            )
    return confirmed, contradicted
