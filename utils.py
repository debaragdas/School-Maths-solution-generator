"""
utils.py — small shared helpers. Logging setup is ported near-unchanged
from the RAG uploader (same colored-output pattern), plus a couple of
generic helpers used by more than one module.
"""
import logging
import os
import re
import time
import functools
import threading
import collections
from contextlib import contextmanager

import config


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[94m", "SUCCESS": "\033[92m",
        "WARNING": "\033[93m", "ERROR": "\033[91m", "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.msg = f"{color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)


def get_logger(name: str = "SolutionFactory") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


logger = get_logger()


def retry(times: int, delay_seconds: float = 2.0, exceptions=(Exception,)):
    """Simple retry decorator — no key rotation, no cooldown cycles.
    Just: try, log, wait, try again, up to `times` total attempts."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    logger.warning(f"⚠️ {fn.__name__} failed (attempt {attempt}/{times}): {e}")
                    if attempt < times:
                        time.sleep(delay_seconds * attempt)
            raise last_err
        return wrapper
    return decorator


class _GlobalRateLimiter:
    """A single, process-wide, thread-safe sliding-window rate limiter.

    ROOT-CAUSE FIX for 429 RESOURCE_EXHAUSTED bursts (see config.py's
    VERTEX_AI_MAX_CALLS_PER_MINUTE for the full story): every Vertex AI
    call site in this project already retries its OWN 429s individually
    via retry_with_backoff below, but with config.PARALLEL_WORKERS
    exercises running concurrently — and each exercise's pipeline
    capable of firing off several independent calls in a row (main
    solve, vision figure-location, per-question recovery) — nothing
    previously stopped the WHOLE PROCESS from bursting past Vertex AI's
    per-minute quota, no matter how well each individual call retried
    afterward. One shared instance, acquired right before every attempt
    in retry_with_backoff, caps the total outbound call rate across
    every thread and every call site at once.
    """

    def __init__(self, max_calls_per_minute: int):
        self._max_calls = max(1, max_calls_per_minute)
        self._period = 60.0
        self._lock = threading.Lock()
        self._timestamps = collections.deque()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                sleep_for = self._period - (now - self._timestamps[0])
            # Sleep OUTSIDE the lock so other threads can still check in
            # (and find their own slot has freed up) while this one waits.
            time.sleep(max(sleep_for, 0.05))


_vertex_ai_rate_limiter = _GlobalRateLimiter(
    getattr(config, "VERTEX_AI_MAX_CALLS_PER_MINUTE", 8))

# ROOT-CAUSE FIX #2 for "limit reached" bursts: the rate limiter above
# caps the TOTAL call rate per minute, but it does NOT stop several
# threads from all being granted a slot at once and firing their
# requests off simultaneously — with config.PARALLEL_WORKERS exercises
# in flight, 2-3 Vertex AI calls really could go out in the same
# instant (each one legitimately "within quota" for the minute), and
# many Vertex AI quota tiers also cap CONCURRENT in-flight requests
# separately from requests-per-minute. A plain semaphore of size 1
# held only around the actual network call (never around the
# surrounding retry/backoff sleep, and never around any of the local,
# non-API work in the same pipeline — diagram SVG rendering, math
# sanitization, PDF/HTML generation) guarantees literally one Vertex
# AI request in flight process-wide at any instant, for every call
# site, without serializing anything that isn't itself talking to
# Vertex AI. Since the per-minute budget was already the real
# throughput ceiling, this costs effectively nothing in wall-clock
# time — it just stops the simultaneous-burst shape of the failure
# that was forcing costly 5-90s backoff-and-retry cycles instead.
_vertex_ai_concurrency_limiter = threading.Semaphore(1)


def retry_with_backoff(times: int = 6, base_delay: float = 5.0, max_delay: float = 90.0,
                        retryable_check=None):
    """Exponential backoff retry — specifically for API rate limits
    (429 RESOURCE_EXHAUSTED), which need real waiting time (5s, 10s,
    20s, 40s, ...) rather than the fixed short delay `retry()` uses.

    Every attempt (not just retries — the first attempt too) first goes
    through the single shared _vertex_ai_rate_limiter above, so no
    matter how many exercises/threads are calling this concurrently,
    the actual outbound rate to Vertex AI across the whole process
    never exceeds config.VERTEX_AI_MAX_CALLS_PER_MINUTE. This is what
    actually fixes 429 bursts at the root — per-call exponential
    backoff alone only ever reacted to a quota already being blown;
    this prevents blowing it in the first place.

    `retryable_check(exception) -> bool` lets the caller distinguish
    "worth waiting and trying again" (rate limits, transient network
    errors) from "will fail the same way every time" (bad request,
    auth failure) — the latter is re-raised immediately instead of
    burning through all `times` attempts uselessly. If not provided,
    every exception is treated as retryable.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, times + 1):
                _vertex_ai_rate_limiter.acquire()
                # Held ONLY around the actual outbound call — see
                # _vertex_ai_concurrency_limiter's own comment above
                # for why this is a second, independent fix from the
                # rate limiter, not a duplicate of it.
                with _vertex_ai_concurrency_limiter:
                    try:
                        return fn(*args, **kwargs)
                    except Exception as e:
                        last_err = e
                        is_retryable = retryable_check(e) if retryable_check else True
                if not is_retryable:
                    raise last_err
                if attempt >= times:
                    break
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"⏳ {fn.__name__} hit a retryable error (attempt {attempt}/{times}): "
                    f"{last_err} — backing off {delay:.0f}s before retrying..."
                )
                time.sleep(delay)
            raise last_err
        return wrapper
    return decorator


_ASSAMESE_TO_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


@contextmanager
def exclusive_file_lock(lock_path: str, owner_label: str = "file"):
    """Cross-platform inter-process AND inter-thread exclusive advisory
    lock — the ONE locking primitive every read-modify-write file in
    this project must go through.

    WHY THIS EXISTS: figure_database.py and review_state.py originally
    locked with fcntl.flock, which is POSIX-only. On Windows that
    degraded to NO locking at all while config.PARALLEL_WORKERS still
    runs 3 exercises concurrently — so two workers ingesting figures
    into the same chapter, or two reviewers acting on the same
    exercise's state file, raced on a plain load-mutate-save and the
    second writer silently discarded the first writer's work (reproduced
    by test_figure_database.py::
    test_concurrent_ingestion_from_different_exercises_never_loses_a_figure
    and test_review_state.py::TestConcurrentReviewActions). This
    primitive makes the same guarantee real on BOTH platforms:

    - POSIX (Linux/macOS): fcntl.flock(LOCK_EX) — identical semantics to
      what those modules already relied on.
    - Windows: msvcrt.locking() byte-range lock on handle-opened-by-us.
      Windows locks are per file HANDLE, not per process, so two handles
      opened by two threads of this same process conflict exactly like
      two separate processes do — which is precisely what both callers
      need for their ThreadPoolExecutor races.

    The lock file itself is never deleted while held and is safe to
    reuse forever; contention waits bounded (default 30s) then raises,
    because silently proceeding WITHOUT the lock is exactly the failure
    mode that lost data before.
    """
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            deadline = time.time() + 30.0
            delay = 0.01
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.time() >= deadline:
                        raise TimeoutError(
                            f"{owner_label}: could not acquire lock {lock_path} "
                            f"within 30s — another worker/reviewer appears stuck.")
                    time.sleep(delay)
                    delay = min(delay * 2, 0.2)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        acquired = True
        yield
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass  # unlocking must never mask an in-flight exception
        os.close(fd)


def unique_tmp_path(path: str) -> str:
    """A per-process/per-thread unique temp filename beside `path`, for
    atomic write-then-os.replace() sequences. Two writers that somehow
    bypass the file lock can therefore never collide on ONE shared
    '.tmp' name (the Windows 'WinError 32: process cannot access the
    file' seen on index.json.tmp under parallel workers) — each writer
    gets its own temp file, and only the os.replace() winner matters."""
    return f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"

# Matches "Figure 3.14", "Fig. 7.39", "চিত্ৰ ৩.১৪", "চিত্ৰ 7.39" — a figure
# *label* word immediately followed by a number (int or dotted, ASCII or
# Assamese digits). Deliberately requires a digit: a bare caption like
# "চিত্ৰ আৰ্হি" ("diagram template" — the caption used on an AI-generated
# fallback diagram) has no number and must NOT match here, since it isn't a
# reference to a specific numbered textbook figure.
#
# The decimal separator between the chapter/figure parts (e.g. "7" and
# "18" in "Figure 7.18") is matched loosely — '.', '-', en dash, or em
# dash, each optionally padded with whitespace — and always NORMALIZED
# back to '.' below. This is purely an OCR-robustness fix (a scanned
# textbook caption's '.' is one of the single most commonly misread
# characters, and book_diagram_extractor.py's OCR pass can legitimately
# emit "Fig 7-18" or "Fig 7 . 18" for the exact same printed figure);
# it does not change what counts as a figure reference, only which
# literal characters between two digit groups are accepted as the SAME
# separator that '.' already was.
_FIGURE_REF_PATTERN = re.compile(
    r"(?:Figure|Fig\.?|চিত্ৰ|চিত্র)\s*[:.\-]?\s*"
    r"([০-৯0-9]+)(?:\s*[.\-–—]\s*([০-৯0-9]+))?",
    re.IGNORECASE,
)


def extract_figure_reference(text: str):
    """Finds a numbered figure reference (e.g. 'Figure 3.14' / 'চিত্ৰ ৩.১৪',
    or the OCR-noisy 'Fig 7-18' / 'Fig 7 . 18') in `text` and returns it
    normalized to plain ASCII digits with '.' as the decimal separator
    (e.g. '3.14'), or None if no numbered figure reference is present.
    Pure regex, no AI — used to match a question to its ORIGINAL
    textbook figure by the number printed in the question itself,
    rather than relying only on page order.
    """
    if not text:
        return None
    match = _FIGURE_REF_PATTERN.search(text)
    if not match:
        return None
    whole, part = match.group(1), match.group(2)
    number = f"{whole}.{part}" if part else whole
    return number.translate(_ASSAMESE_TO_ASCII_DIGITS)


def already_done(output_path) -> bool:
    """Skip-completed check — a PDF that already exists is considered done.
    Delete the file manually to force a regeneration; no separate state
    file needed for a project this size."""
    import os
    return os.path.exists(output_path) and os.path.getsize(output_path) > 1000


def verify_solution(result: dict) -> tuple[bool, list[str]]:
    """Cheap, deterministic checks on a solved exercise's JSON before it's
    allowed anywhere near rendering. No AI involved — just the checks the
    spec asked for: no missing question, numbering complete, answers
    present, no obviously broken maths, no empty output."""
    issues = []
    questions = result.get("questions", [])

    if not questions:
        issues.append("No questions were returned.")
        return False, issues

    seen_numbers = []
    for q in questions:
        qn = q.get("question_number")
        if not qn:
            issues.append("A question is missing its question_number.")
            continue
        seen_numbers.append(qn)

        if not (q.get("final_answer") or "").strip():
            issues.append(f"Question {qn}{q.get('sub_part') or ''}: empty final_answer.")
        if not q.get("steps"):
            issues.append(f"Question {qn}{q.get('sub_part') or ''}: no solution steps.")

        for text in (q.get("given"), q.get("required"), q.get("final_answer"), *(q.get("steps") or [])):
            if not isinstance(text, str):
                continue
            if text.count("$$") % 2 != 0:
                issues.append(f"Question {qn}: unbalanced $$ block math.")
            stripped = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
            if stripped.count("$") % 2 != 0:
                issues.append(f"Question {qn}: unbalanced inline $ math.")

    # Numbering completeness: unique question numbers should form 1..N with no gaps.
    unique_numbers = sorted({int(n) for n in seen_numbers if str(n).isdigit()})
    if unique_numbers and unique_numbers != list(range(unique_numbers[0], unique_numbers[-1] + 1)):
        issues.append(f"Question numbering has a gap: found {unique_numbers}.")

    return (len(issues) == 0), issues
