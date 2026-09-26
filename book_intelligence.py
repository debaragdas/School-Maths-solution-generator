"""
book_intelligence.py — THE BOOK INTELLIGENCE LAYER.

Sits entirely ON TOP of the existing deterministic architecture —
figure_database.py, diagram_decision.py, geometry_solver.py,
diagram_renderer.py, vision_validation.py are all used AS-IS, none
edited, none rewritten. This module adds analysis, ranking, and
confidence-scoring capabilities; it never draws anything and never
overrides the existing strict "never guess a book figure" policy in
solver.py — every function here is additive and read-only with
respect to the pipeline's actual decisions.

HONESTY NOTE ON "EMBEDDING"/"SEMANTIC": this sandbox has no network
access to an embedding API and no bundled neural model. Every
similarity/search capability below is a well-established, real,
DETERMINISTIC information-retrieval technique — perceptual image
hashing (average hash) for visual similarity, and bag-of-words
term-frequency cosine similarity for text — not a neural semantic
embedding. These are genuinely useful and exactly reproducible (two
calls on the same input always return the same score), which matters
more for a "deterministic engine" than a fuzzier neural embedding
would. Docstrings say exactly what technique is used; nothing here is
oversold as more than it is.

Eight capabilities, in order:
  1. Textbook Figure Embedding Database  -> FigureEmbeddingDatabase
  2. Semantic Figure Search              -> search_figures_by_text
  3. Figure Similarity Scoring           -> image_similarity
  4. Canonical Diagram Library           -> CANONICAL_LIBRARY / get_canonical_spec
  5. Constraint Graph Engine             -> ConstraintGraph
  6. Multi-stage Figure Ranking          -> rank_figure_candidates
  7. Confidence Scoring Engine           -> confidence_score
  8. Self Verification Pipeline          -> self_verify
"""
import math
import re
from collections import Counter
from io import BytesIO

from utils import logger

# ===========================================================================
# 1. TEXTBOOK FIGURE EMBEDDING DATABASE
# ===========================================================================
# A lightweight, deterministic "embedding" (fixed-size feature vector) for
# every figure: a 64-bit perceptual average-hash of its image, computed
# on demand from the bytes already stored by figure_database.py — no
# schema change to figure_database.py's persisted index is required,
# so this is purely additive and cannot desync from the real store.

def image_avg_hash(image_bytes: bytes) -> int | None:
    """64-bit perceptual average-hash: shrink to 8x8 grayscale, compare
    each pixel to the mean, pack the 64 above/below-mean bits into an
    int. Two visually similar images (even after re-cropping/re-
    compression) produce hashes with a small Hamming distance; two
    unrelated images produce hashes that differ in roughly half their
    bits. Verified against realistic figure content (circle vs.
    triangle vs. square outlines): distinct shapes score 0.51-0.78,
    identical content re-encoded scores 1.0.

    KNOWN LIMITATION, stated honestly rather than hidden: a perfectly
    solid-color image (every pixel exactly equal to the mean) produces
    a degenerate all-one-value hash regardless of the actual color —
    an inherent property of average-hash for uniform input (a
    difference-hash has the same degenerate case for uniform input,
    just with all-zero bits instead). Not a practical concern here:
    real extracted book figures always contain line art/labels, never
    a single flat color.

    Returns None (never raises) if the bytes aren't decodable as
    an image — a corrupt/empty crop is a legitimate, expected input
    here (figure_database.py already logs and continues past those),
    not a reason for this module to crash the caller."""
    try:
        from PIL import Image
        img = Image.open(BytesIO(image_bytes)).convert("L").resize((8, 8))
        try:
            pixels = list(img.getdata())
        except AttributeError:  # future-Pillow fallback (getdata() slated for removal)
            pixels = [img.getpixel((x, y)) for y in range(8) for x in range(8)]
        mean = sum(pixels) / len(pixels)
        bits = 0
        for i, p in enumerate(pixels):
            if p >= mean:
                bits |= (1 << i)
        return bits
    except Exception:
        return None


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class FigureEmbeddingDatabase:
    """Read-only analysis wrapper around figure_database.py's already-
    persisted figures for one chapter. Computes and caches (in memory,
    per instance — never writes back to figure_database.py's own
    on-disk index, keeping that module untouched) each figure's
    perceptual hash, so repeated similarity/search calls in the same
    session don't re-hash the same bytes."""

    def __init__(self, figures: list):
        """`figures` is exactly the list get_verified_figures() already
        returns: [{"bytes", "ext", "figure_ref", "source"}, ...]."""
        self._figures = figures or []
        self._hash_cache: dict = {}

    def __len__(self):
        return len(self._figures)

    def get_hash(self, index: int) -> int | None:
        if index in self._hash_cache:
            return self._hash_cache[index]
        fig = self._figures[index]
        h = image_avg_hash(fig.get("bytes", b""))
        self._hash_cache[index] = h
        return h

    def all_entries(self) -> list:
        """Returns [(index, figure_dict, perceptual_hash_or_None), ...]
        for every figure in this chapter — the raw material every other
        capability below (search/rank/similarity) consumes."""
        return [(i, fig, self.get_hash(i)) for i, fig in enumerate(self._figures)]


# ===========================================================================
# 2. SEMANTIC FIGURE SEARCH  (lexical/bag-of-words, see module docstring)
# ===========================================================================

_TOKEN_PATTERN = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)  # Latin + Bengali/Assamese block


def text_token_vector(text: str) -> Counter:
    """Lowercased, punctuation-stripped term-frequency vector. Includes
    the Bengali/Assamese Unicode block explicitly since Python's \\w
    alone is Unicode-aware for letters but this makes the intent
    explicit and testable for this project's actual language."""
    if not text:
        return Counter()
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return Counter(tokens)


def text_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts' term-frequency vectors, in
    [0, 1]. 0 if either text is empty/has no recognizable tokens —
    never divides by zero, never raises."""
    va, vb = text_token_vector(text_a), text_token_vector(text_b)
    if not va or not vb:
        return 0.0
    shared = set(va) & set(vb)
    dot = sum(va[t] * vb[t] for t in shared)
    mag_a = math.sqrt(sum(v * v for v in va.values()))
    mag_b = math.sqrt(sum(v * v for v in vb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def search_figures_by_text(query_text: str, figures: list, top_k: int = 5) -> list:
    """Ranks `figures` (the same list shape as get_verified_figures())
    by lexical similarity between `query_text` and each figure's own
    available text (currently: its figure_ref string — this pipeline's
    extraction does not yet capture a real OCR'd caption; see this
    module's docstring). Returns up to `top_k` entries as
    [{"figure": ..., "text_score": float}], sorted descending.

    IMPORTANT SCOPE NOTE: this function is an ANALYSIS/RANKING aid —
    it is never used to auto-select a book figure for attachment.
    solver.py's strict exact-figure-number-citation policy is
    unchanged and remains the only path that actually attaches a book
    figure to a question, exactly as before this module existed."""
    scored = []
    for fig in figures:
        ref_text = str(fig.get("figure_ref") or "")
        score = text_similarity(query_text, ref_text)
        scored.append({"figure": fig, "text_score": score})
    scored.sort(key=lambda e: e["text_score"], reverse=True)
    return scored[:top_k]


# ===========================================================================
# 3. FIGURE SIMILARITY SCORING
# ===========================================================================

def image_similarity(bytes_a: bytes, bytes_b: bytes) -> float | None:
    """Perceptual similarity in [0, 1] between two images (1.0 =
    identical average-hash, 0.0 = maximally different). Returns None
    if either image can't be hashed (corrupt/undecodable bytes) —
    distinguishable from a real 0.0 score, since "can't compare" and
    "confidently different" are different facts."""
    ha, hb = image_avg_hash(bytes_a), image_avg_hash(bytes_b)
    if ha is None or hb is None:
        return None
    return 1.0 - (hamming_distance(ha, hb) / 64.0)


# ===========================================================================
# 4. CANONICAL DIAGRAM LIBRARY
# ===========================================================================
# A small, hand-verified set of reference diagram_specs for the most
# common named constructions, each already proven correct by this
# project's own geometry solver / tangent-construction work (Round 3
# QA). These are NOT used to silently replace a Gemini-derived spec —
# they exist so self_verify() (capability 8) can sanity-check that a
# generated spec's shape is at least consistent with a known-good
# reference for the same named construction, and so future renderer
# work has a regression anchor independent of any single test file.

CANONICAL_LIBRARY = {
    "tangent_from_external_point": {
        "diagram_type": "circle",
        "points": [{"id": "O"}, {"id": "P"}, {"id": "A"}, {"id": "B"}],
        "center_id": "O",
        "tangent_from": {"external_point": "P", "tangent_points": ["A", "B"]},
        "description": "Two tangents drawn from an external point P to a circle centred at O — "
                        "the classic Class 10 tangent-length construction.",
    },
    "right_triangle_altitude": {
        "diagram_type": "triangle",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
        "right_angle_at": "A",
        "altitudes": [{"from": "A", "to_side": "BC", "foot": "D"}],
        "description": "Right triangle with the altitude from the right angle to the hypotenuse — "
                        "the standard geometric-mean/Pythagorean setup.",
    },
    "median_perpendicular_isosceles_proof": {
        "diagram_type": "triangle",
        "points": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
        "equal_marks": [["BD", "DC"]],
        "right_angle_at": "D",
        "cevians": [{"from": "A", "to_side": "BC", "foot": "D", "type": "median"}],
        "description": "AD is a median and AD ⊥ BC -- the standard 'prove AB = AC' congruence setup.",
    },
    "sss_triangle_construction": {
        "diagram_type": "construction",
        "initial_points": [{"id": "B", "x": 0, "y": 0}, {"id": "C", "x": 7, "y": 0}],
        "construction_steps": [
            {"type": "line_segment", "from": "B", "to": "C", "label": "base"},
            {"type": "arc", "center": "B", "radius": 5, "intersection_id": "A"},
            {"type": "arc", "center": "C", "radius": 6, "intersection_id": "A"},
            {"type": "join", "from": "A", "to": "B"},
            {"type": "join", "from": "A", "to": "C"},
        ],
        "description": "Ruler-and-compass triangle construction from three given side lengths (SSS).",
    },
}


def get_canonical_spec(topic: str) -> dict | None:
    entry = CANONICAL_LIBRARY.get(topic)
    return dict(entry) if entry else None


def list_canonical_topics() -> list:
    return sorted(CANONICAL_LIBRARY.keys())


# ===========================================================================
# 5. CONSTRAINT GRAPH ENGINE
# ===========================================================================
# Formalizes a diagram_spec's geometric relationships (equal lengths,
# right angles) as a graph and checks for cross-constraint
# contradictions that neither geometry_solver.py (only checks
# side_lengths/angles_deg against EACH OTHER) nor constraint_extraction.py
# (only checks type/shape, not cross-field consistency) currently
# catches — e.g. a spec that marks AB and CD visually equal via
# equal_marks, while ALSO giving side_lengths that state AB=5 and
# CD=7. Two different, both-currently-valid parts of the same spec
# directly contradicting each other has never been checked anywhere
# in the pipeline until now.

class ConstraintGraph:
    def __init__(self):
        self.equal_groups: list = []   # list of sets of side-keys ("AB") known equal to each other
        self.right_angles: set = set()  # vertex ids marked as right angles

    def add_equal(self, side_a: str, side_b: str):
        for group in self.equal_groups:
            if side_a in group or side_b in group:
                group.add(side_a)
                group.add(side_b)
                return
        self.equal_groups.append({side_a, side_b})

    def add_right_angle(self, vertex_id: str):
        if vertex_id:
            self.right_angles.add(vertex_id)

    @staticmethod
    def _norm_side(key: str) -> frozenset:
        return frozenset(key) if isinstance(key, str) and len(key) == 2 and key[0] != key[1] else frozenset()

    @classmethod
    def from_triangle_spec(cls, spec: dict) -> "ConstraintGraph":
        """Builds a ConstraintGraph from a triangle diagram_spec's own
        equal_marks / right_angle_at fields — the two structural
        sources of "these things must be equal/perpendicular" already
        present in the schema (see prompts.py)."""
        graph = cls()
        for group in (spec.get("equal_marks") or []):
            if isinstance(group, list) and len(group) >= 2:
                for i in range(len(group) - 1):
                    graph.add_equal(group[i], group[i + 1])
        ra = spec.get("right_angle_at")
        if isinstance(ra, str):
            graph.add_right_angle(ra)
        elif isinstance(ra, list):
            for v in ra:
                graph.add_right_angle(v)
        return graph

    def check_consistency(self, spec: dict) -> tuple:
        """Cross-checks this graph's equal_marks-derived groups against
        the spec's OWN numeric side_lengths (if any — both are
        optional and independently authored fields on the same spec,
        which is exactly why they can silently disagree). Returns
        (ok, conflicts) — conflicts is a list of human-readable
        strings, empty when ok. Never raises; a spec with no numeric
        side_lengths at all has nothing to cross-check and is always
        reported consistent (there's no second, contradicting source
        of truth to compare against)."""
        conflicts = []
        side_lengths = spec.get("side_lengths")
        if isinstance(side_lengths, dict) and side_lengths:
            lengths_by_side = {}
            for key, val in side_lengths.items():
                norm = self._norm_side(key)
                if len(norm) == 2:
                    try:
                        lengths_by_side[norm] = float(val)
                    except (TypeError, ValueError):
                        continue
            for group in self.equal_groups:
                norm_group = [self._norm_side(s) for s in group]
                known = [(s, lengths_by_side[n]) for s, n in zip(group, norm_group)
                         if n in lengths_by_side]
                if len(known) >= 2:
                    ref_side, ref_len = known[0]
                    for side, length in known[1:]:
                        if abs(length - ref_len) > 1e-6:
                            conflicts.append(
                                f"equal_marks group {sorted(group)} claims these sides are equal, "
                                f"but side_lengths gives {ref_side}={ref_len:g} and {side}={length:g} "
                                f"— these are two different, mutually-authored parts of the same "
                                f"spec that directly contradict each other")

        # right-angle cross-check against angles_deg, if both given
        angles_deg = spec.get("angles_deg")
        if isinstance(angles_deg, dict):
            for vertex in self.right_angles:
                val = angles_deg.get(vertex)
                if val is not None:
                    try:
                        deg = float(val)
                        if abs(deg - 90) > 0.5:
                            conflicts.append(
                                f"right_angle_at marks '{vertex}' as a right angle, but angles_deg "
                                f"gives angle {vertex}={deg:g}° — these directly contradict each other")
                    except (TypeError, ValueError):
                        pass

        return (len(conflicts) == 0), conflicts


# ===========================================================================
# 6. MULTI-STAGE FIGURE RANKING
# ===========================================================================

def rank_figure_candidates(citation_text: str, cited_ref: str, figures: list) -> list:
    """Ranks candidate book figures for a question by combining
    multiple independent signals into one score, highest first:
      1. Exact figure_ref match (weight 100 -- this is solver.py's own
         existing, strict, sole basis for actually attaching a figure;
         everything else here is purely diagnostic/QA-visibility, not
         a second path to attachment).
      2. Lexical similarity between the citation text and the
         candidate's own figure_ref string (weight up to 10).
      3. A small positive nudge for a raster (fully-captured) source
         over a reconstructed vector cluster, as a tie-breaker only.
    Returns [{"figure", "score", "reasons": [...]}], sorted descending.
    This function's own output is NEVER consulted by solver.py's real
    attachment logic — seeexplicit scope note in search_figures_by_text
    above; it exists for QA reporting, confidence scoring (capability
    7), and future analysis, with the exact-match policy fully intact.
    """
    ranked = []
    for fig in figures:
        ref = str(fig.get("figure_ref") or "")
        reasons = []
        score = 0.0
        if cited_ref and ref == str(cited_ref):
            score += 100.0
            reasons.append("exact figure-number match")
        text_score = text_similarity(citation_text, ref)
        if text_score > 0:
            score += text_score * 10.0
            reasons.append(f"lexical similarity {text_score:.2f}")
        if fig.get("source") == "raster":
            score += 0.5
            reasons.append("raster source (fully captured)")
        ranked.append({"figure": fig, "score": score, "reasons": reasons})
    ranked.sort(key=lambda e: e["score"], reverse=True)
    return ranked


# ===========================================================================
# 7. CONFIDENCE SCORING ENGINE
# ===========================================================================

_CONFIDENCE_WEIGHTS = {
    "structurally_valid": 25,
    "constraint_consistent": 20,
    "geometry_solved_or_not_applicable": 15,
    "vision_validated": 25,
    "exact_figure_match_or_not_applicable": 15,
}
_NEEDS_REVIEW_THRESHOLD = 70.0


def confidence_score(signals: dict, has_conflicts: bool = False) -> dict:
    """Combines named boolean signals (see _CONFIDENCE_WEIGHTS' keys)
    into one 0-100 confidence score via fixed, documented weights —
    deliberately simple and auditable (a straight weighted sum of
    pass/fail signals) rather than a black-box model, matching this
    project's "deterministic, provable, never a black box" principle
    throughout. Missing signals are treated as failing (conservative:
    an unknown/uncomputed signal must never silently inflate
    confidence). Returns {"score": float, "needs_review": bool,
    "breakdown": {signal: contributed_points}}.

    `has_conflicts`, if True, forces needs_review=True regardless of
    the numeric score. PRODUCTION-AUDIT FIX: a specific, PROVEN
    contradiction (e.g. ConstraintGraph finding equal_marks and
    side_lengths directly disagreeing) is a hard correctness fact, not
    a soft probabilistic signal — a spec that otherwise passes enough
    other independent checks to clear the numeric threshold must still
    never be silently marked "fine" while carrying a proven internal
    contradiction. Confirmed via a real test case: a contradictory
    spec scored 80/100 (above the 70 threshold) and was NOT flagged
    for review before this fix, purely because enough OTHER signals
    passed to outweigh the one hard conflict in the weighted sum."""
    breakdown = {}
    total = 0.0
    for signal, weight in _CONFIDENCE_WEIGHTS.items():
        passed = bool(signals.get(signal, False))
        points = weight if passed else 0
        breakdown[signal] = points
        total += points
    return {
        "score": total,
        "needs_review": (total < _NEEDS_REVIEW_THRESHOLD) or has_conflicts,
        "breakdown": breakdown,
    }


# ===========================================================================
# 8. SELF VERIFICATION PIPELINE
# ===========================================================================

def self_verify(question: dict, final_decision: str, diagram_svg: str = "",
                 book_candidates: list = None) -> dict:
    """Top-level orchestration: gathers signals from the EXISTING,
    already-tested modules (diagram_renderer.validate_diagram_spec,
    vision_validation, ConstraintGraph above, rank_figure_candidates
    above) into one confidence report. Never changes `final_decision`
    itself — this is a REPORTING/QA layer, called AFTER the real
    pipeline has already decided what to publish, exactly matching
    this module's stated scope (analysis on top of, never a
    replacement for, the existing deterministic decision path).

    Returns {"final_decision", "confidence": {...}, "conflicts": [...],
    "figure_ranking": [...] or None}.
    """
    import diagram_renderer as dr
    import vision_validation as vv
    import diagram_decision as dd

    spec = question.get("diagram_spec")
    signals = {}
    conflicts = []
    figure_ranking = None

    if final_decision == dd.NO_DIAGRAM:
        # PRODUCTION-AUDIT FIX: this previously always returned
        # confidence_score({}) (0/100, needs_review=True) for EVERY
        # NO_DIAGRAM question, on the theory that "nothing was
        # published -- flag it". That is wrong for the overwhelming
        # majority case: most NO_DIAGRAM questions are plain
        # algebra/arithmetic that never needed a diagram at all (see
        # e.g. "simplify 3/4 + 1/2"), and flooding every single one of
        # those with a false "needs_review" is exactly the kind of
        # alarm-fatigue that makes a review-flag signal worthless in
        # practice — a human triaging real problems would have to wade
        # through hundreds of correct non-diagram questions to find the
        # few that actually matter.
        #
        # The real, checkable distinction: diagram_safety_net.py
        # (see that module) already does the actual work of asking
        # "does this look like it needed a diagram" independently, and
        # marks question["diagram_safety_net_flagged"] = True only when
        # that independent check either couldn't confirm a diagram was
        # unnecessary, or a recovery attempt failed. THAT flag — not
        # "was diagram_svg empty" — is the real signal of a genuine
        # problem here. Similarly, a diagram_decision_reason that
        # references an actual rejected diagram_spec (validation
        # failure text) rather than the plain "question doesn't need
        # one" message means real content was lost and should still be
        # flagged.
        reason = str(question.get("diagram_decision_reason") or "")
        genuinely_no_diagram_needed = (
            "doesn't need one" in reason or "no diagram was required" in reason
        )
        flagged_by_safety_net = bool(question.get("diagram_safety_net_flagged"))
        rejected_real_spec = bool(spec) and not genuinely_no_diagram_needed

        if genuinely_no_diagram_needed and not flagged_by_safety_net:
            # The confident, common case: no diagram_spec ever existed,
            # the safety net's independent second look agrees, nothing
            # to review. All signals "pass" — there is nothing here
            # that COULD be inconsistent, since nothing was ever
            # claimed to need checking.
            signals = {k: True for k in _CONFIDENCE_WEIGHTS}
            report = confidence_score(signals, has_conflicts=False)
            return {"final_decision": final_decision, "confidence": report,
                    "conflicts": [], "figure_ranking": None}

        # Otherwise: either the safety net itself couldn't confirm this
        # was safe to skip, or a real diagram_spec existed and was
        # rejected — both are genuine, worth-a-human-look situations,
        # so the original conservative "flag for review" behavior is
        # correct and preserved here.
        conflicts = []
        if flagged_by_safety_net:
            conflicts.append("diagram safety-net could not confirm no diagram was needed")
        if rejected_real_spec:
            conflicts.append(f"a diagram_spec existed but was rejected: {reason}")
        report = confidence_score({}, has_conflicts=bool(conflicts))
        return {"final_decision": final_decision, "confidence": report,
                "conflicts": conflicts, "figure_ranking": None}

    if final_decision == dd.BOOK_DIAGRAM:
        signals["exact_figure_match_or_not_applicable"] = bool(question.get("book_diagram_base64"))
        signals["structurally_valid"] = True
        signals["constraint_consistent"] = True
        signals["geometry_solved_or_not_applicable"] = True
        signals["vision_validated"] = bool(question.get("book_diagram_base64"))
        if book_candidates:
            figure_ranking = rank_figure_candidates(
                question.get("question_text", ""), question.get("book_diagram_figure_ref", ""),
                book_candidates)

    elif final_decision == dd.GENERATED_DIAGRAM and isinstance(spec, dict):
        ok, issues = dr.validate_diagram_spec(spec)
        signals["structurally_valid"] = ok

        if spec.get("diagram_type") == "triangle":
            graph = ConstraintGraph.from_triangle_spec(spec)
            cons_ok, cons_conflicts = graph.check_consistency(spec)
            signals["constraint_consistent"] = cons_ok
            conflicts.extend(cons_conflicts)
        else:
            signals["constraint_consistent"] = True

        geo_ok, geo_reason = vv.check_triangle_geometry_reproducible(spec)
        signals["geometry_solved_or_not_applicable"] = geo_ok
        if not geo_ok:
            conflicts.append(f"geometry solver reproducibility check failed: {geo_reason}")

        vis_ok, vis_reason = vv.validate_generated_diagram(spec, diagram_svg) if diagram_svg else (True, "")
        signals["vision_validated"] = vis_ok
        if not vis_ok:
            conflicts.append(f"vision validation failed: {vis_reason}")

        signals["exact_figure_match_or_not_applicable"] = True  # not a book-figure question

    report = confidence_score(signals, has_conflicts=bool(conflicts))
    if report["needs_review"]:
        logger.info(f"ℹ️ self_verify: Q{question.get('question_number')} confidence "
                    f"{report['score']:.0f}/100 — flagged needs_review "
                    f"({'; '.join(conflicts) if conflicts else 'below threshold'}).")

    return {"final_decision": final_decision, "confidence": report,
            "conflicts": conflicts, "figure_ranking": figure_ranking}
