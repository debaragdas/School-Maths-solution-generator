"""
figure_database.py — THE PERSISTENT FIGURE DATABASE (Phase 2).

Replaces book_figure_index.py's per-run cache (keyed by
sha256(exercise_pdf_bytes), evaporates the moment a different exercise
split of the same chapter is processed) with a PERMANENT store keyed
by (book_id, class, chapter, figure_number) — the actual identity of a
textbook figure, not an artifact of how the PDF happened to get sliced
this run.

WHY book_figure_index.py WAS REPLACED, NOT PATCHED (see the
architecture audit + design doc): a cache keyed by the CURRENT
exercise PDF's own byte-hash cannot, by definition, be "searchable"
across a book, cannot survive re-splitting the same chapter
differently, and gives no place to attach durable metadata (has a
human verified this crop is correct?) that should persist forever once
recorded. That is not a bug in the implementation; it is a mismatch
between what a per-run cache IS and what a "Figure Database" NEEDS to
be. book_diagram_extractor.py's actual extraction logic (raster +
captioned-vector-cluster detection) is entirely UNCHANGED and fully
reused here as the ingest step — only the storage/retrieval layer is
new.

STORAGE LAYOUT (see diagram_engine_architecture_design.md Section 6):

    figure_db/
      <book_id>/
        <chapter>/
          index.json         — manifest: entries + extractor_version +
                                which exercise-PDF hashes have already
                                been ingested into this chapter
          figures/
            7.14.png
            7.39.png

MERGE SEMANTICS: because a chapter is processed exercise-by-exercise
(each a separate, smaller PDF covering only that exercise's own
pages), this module ACCUMULATES figures into one chapter-level index
across multiple calls rather than overwriting it — the first exercise
of a chapter to cite Figure 7.14 ingests it; every later exercise in
the same chapter (even ones whose own page range never contained that
figure) can still retrieve it. This is a genuine capability the old
per-run cache never had.

BACKWARD COMPATIBILITY: get_verified_figures() returns the EXACT same
shape book_figure_index.get_figure_index() always has —
list[{"bytes", "ext", "figure_ref", "source"}] — so solver.py's
_attach_book_diagrams() needed only a one-line call-site swap (see
solver.py) with zero changes to its own matching logic.
"""
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager

import utils
from book_diagram_extractor import extract_diagram_images
from utils import logger, extract_figure_reference, unique_tmp_path

DB_ROOT = "figure_db"
EXTRACTOR_VERSION = "v1"  # bump this if book_diagram_extractor.py's
                          # extraction logic changes in a way that
                          # should invalidate previously-ingested
                          # figures for re-extraction.


@contextmanager
def _chapter_lock(chapter_dir: str):
    """PRODUCTION-AUDIT FIX (Phase 7): main.py processes multiple
    exercises CONCURRENTLY via a ThreadPoolExecutor (config.
    PARALLEL_WORKERS, default 3). Two exercises from the SAME chapter
    can therefore call get_verified_figures() at the same moment —
    without this lock, both could read index.json before either
    writes, both extract+merge independently, and the second
    os.replace() would silently discard the first thread's newly-
    ingested figures (last-writer-wins on the WHOLE file, not a
    per-key merge). That is a genuine data-loss race condition, not a
    theoretical one, given this project's own concurrency model.

    Delegates to utils.exclusive_file_lock() — the shared cross-
    platform primitive (POSIX fcntl.flock / Windows msvcrt.locking) —
    so the SAME guarantee now holds on every platform instead of
    silently degrading to NO locking wherever fcntl is missing (which,
    before this change, was exactly what happened on Windows while
    PARALLEL_WORKERS was still 3). Held only across the read-extract-
    merge-write critical section in get_verified_figures, so it
    serializes writers for ONE chapter without blocking unrelated
    chapters or read-only callers of other modules.
    """
    os.makedirs(chapter_dir, exist_ok=True)
    lock_path = os.path.join(chapter_dir, ".lock")
    with utils.exclusive_file_lock(lock_path, "figure_database chapter lock"):
        yield


def _book_id_from_url(url: str) -> str:
    """Deterministic, filesystem-safe identifier for a book, derived
    from its source URL so the same book always maps to the same
    database directory across runs without requiring any new
    config.py field. Not intended to be human-curated — just stable
    and collision-resistant (a short content hash suffix guards
    against two different URLs slugifying to the same text)."""
    if not url:
        return "unknown_book"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.rsplit("/", 1)[-1]).strip("_").lower()
    slug = slug[:60] if slug else "book"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}"


def _fingerprint(pdf_path: str) -> str:
    """Real content hash (same technique book_figure_index.py already
    established and this project's own tests already hold to a
    standard of: path/mtime proxies are NOT acceptable, only actual
    byte content may determine identity)."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _chapter_dir(book_id: str, chapter) -> str:
    return os.path.join(DB_ROOT, str(book_id), str(chapter))


def _load_index(chapter_dir: str) -> dict:
    index_path = os.path.join(chapter_dir, "index.json")
    if not os.path.exists(index_path):
        return {"extractor_version": None, "ingested_pdf_hashes": [], "figures": []}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("ingested_pdf_hashes", [])
        data.setdefault("figures", [])
        return data
    except Exception as e:
        logger.warning(f"⚠️ figure_database: index.json at {index_path} was unreadable ({e}) — "
                        f"treating this chapter's persistent store as empty and rebuilding it.")
        return {"extractor_version": None, "ingested_pdf_hashes": [], "figures": []}


def _save_index(chapter_dir: str, index: dict) -> None:
    os.makedirs(chapter_dir, exist_ok=True)
    index_path = os.path.join(chapter_dir, "index.json")
    tmp_path = unique_tmp_path(index_path)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        # atomic rename — never leaves a half-written index.json. The tmp
        # name is per-process/per-thread (see unique_tmp_path), so even a
        # writer that somehow bypassed the chapter lock can never collide
        # with another writer's in-flight temp file (the Windows
        # "WinError 32" seen on the old shared index.json.tmp).
        last_err = None
        for attempt in range(6):
            try:
                os.replace(tmp_path, index_path)
                return
            except PermissionError as e:
                # Windows AV/indexer can briefly hold the fresh file open;
                # retry rather than failing the whole ingestion.
                last_err = e
                time.sleep(0.15 * (attempt + 1))
        raise last_err
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _ext_for(img: dict) -> str:
    ext = (img.get("ext") or "png").lstrip(".")
    return ext if ext.isalnum() else "png"


def _load_persisted_images(chapter_dir: str, index: dict) -> list:
    """Reads every persisted figure's bytes back off disk into the
    same in-memory shape extraction would have produced."""
    result = []
    for entry in index.get("figures", []):
        path = os.path.join(chapter_dir, entry["file"])
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            logger.warning(f"⚠️ figure_database: could not read persisted figure "
                            f"{entry.get('figure_ref')} at {path} ({e}) — skipping it "
                            f"for this run (it stays in the manifest for a future retry).")
            continue
        result.append({"bytes": data, "ext": entry.get("ext", "png"),
                        "figure_ref": entry.get("figure_ref"), "source": entry.get("source", "raster")})
    return result


def _ingest_new_images(chapter_dir: str, index: dict, images: list) -> list:
    """Persists every figure-ref-bearing image that isn't already in
    the manifest (first-seen wins on a duplicate ref — a figure's
    identity and correct crop shouldn't churn run to run just because
    a later exercise split re-extracted a very slightly different
    bounding box of the same printed figure). Captionless
    (figure_ref is None) images are deliberately NOT persisted — they
    have no stable identity to key on; they remain a per-call,
    in-memory-only, best-effort artifact exactly as they always have
    been (used only for a diagnostic "N unmatched figures" log line,
    never for citation-based retrieval)."""
    existing_refs = {e["figure_ref"] for e in index["figures"]}
    os.makedirs(os.path.join(chapter_dir, "figures"), exist_ok=True)
    newly_added = []
    for img in images:
        ref = img.get("figure_ref")
        if not ref or ref in existing_refs:
            continue
        ext = _ext_for(img)
        # figure_ref may contain characters not safe in a bare filename
        # (rare, but defensive) — sanitize without losing readability.
        safe_ref = re.sub(r"[^a-zA-Z0-9._-]", "_", str(ref))
        filename = f"figures/{safe_ref}.{ext}"
        full_path = os.path.join(chapter_dir, filename)
        try:
            with open(full_path, "wb") as f:
                f.write(img["bytes"])
        except Exception as e:
            logger.warning(f"⚠️ figure_database: failed to persist figure {ref} ({e}) — "
                            f"it will still be available for THIS run from memory, but will "
                            f"need re-extraction next time.")
            continue
        entry = {
            "figure_ref": ref, "file": filename, "ext": ext,
            "source": img.get("source", "raster"),
            "verified_by_human": False,
            "first_seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        index["figures"].append(entry)
        existing_refs.add(ref)
        newly_added.append(ref)
    return newly_added


def get_verified_figures(pdf_path: str, class_name, chapter, book_id: str = None,
                          book_url: str = None) -> list:
    """Drop-in replacement for book_figure_index.get_figure_index(),
    same return shape, persistent instead of per-run. `chapter` is the
    chapter number this exercise belongs to (required to place figures
    in the right permanent bucket) — if not supplied (e.g. an older
    call site that hasn't been updated, or a direct/legacy caller),
    falls back to on-demand extraction with NO persistence, exactly
    matching book_figure_index's old per-run behavior, so this
    function is safe to call from anywhere book_figure_index used to
    be called from.

    Never raises — any failure degrades to "extract fresh, don't
    persist" rather than blocking the question from getting its book
    diagram at all.
    """
    if chapter is None:
        logger.info("ℹ️ figure_database: no chapter given — extracting on-demand without "
                    "persisting (equivalent to the old per-run cache behavior).")
        try:
            return extract_diagram_images(pdf_path, class_name=class_name)
        except Exception as e:
            logger.warning(f"⚠️ figure_database: extraction failed ({e}).")
            return []

    try:
        resolved_book_id = book_id
        if not resolved_book_id:
            url = book_url
            if url is None:
                import config
                url = getattr(config, "BOOK_URL", None)
            resolved_book_id = _book_id_from_url(url)

        chapter_dir = _chapter_dir(resolved_book_id, chapter)
        with _chapter_lock(chapter_dir):
            index = _load_index(chapter_dir)
            pdf_hash = _fingerprint(pdf_path)

            version_matches = index.get("extractor_version") == EXTRACTOR_VERSION
            already_ingested = pdf_hash in index.get("ingested_pdf_hashes", [])

            if not version_matches and index.get("figures"):
                logger.info(f"ℹ️ figure_database: extractor_version changed for "
                            f"{resolved_book_id}/chapter {chapter} — invalidating and "
                            f"re-ingesting this chapter's persisted figures.")
                index = {"extractor_version": EXTRACTOR_VERSION, "ingested_pdf_hashes": [], "figures": []}

            if already_ingested and version_matches:
                # Cache hit: this exact exercise PDF has already contributed
                # its figures to this chapter's permanent store — no need to
                # re-run extraction. Return the FULL accumulated chapter set
                # (a strict superset of what this one PDF alone would give),
                # which is strictly more useful to the caller's exact-ref
                # matching than the old per-run scope ever was.
                return _load_persisted_images(chapter_dir, index)

            # Cache miss (first time this exact PDF is seen for this
            # chapter, or the extractor version was bumped): run the real
            # extraction (book_diagram_extractor.py, entirely unchanged),
            # merge new figures into the permanent store, remember this
            # PDF's hash so future exercises in the same chapter skip
            # straight to the cache-hit path above.
            images = extract_diagram_images(pdf_path, class_name=class_name)
            index["extractor_version"] = EXTRACTOR_VERSION
            newly_added = _ingest_new_images(chapter_dir, index, images)
            if pdf_hash not in index["ingested_pdf_hashes"]:
                index["ingested_pdf_hashes"].append(pdf_hash)
            _save_index(chapter_dir, index)

            if newly_added:
                logger.info(f"📥 figure_database: ingested {len(newly_added)} new figure(s) into "
                            f"{resolved_book_id}/chapter {chapter}: {newly_added}")

            persisted = _load_persisted_images(chapter_dir, index)
            # Include this run's captionless (un-refable) images too, purely
            # for the existing "N unmatched figures" diagnostic log line in
            # solver.py — they were never persisted (see _ingest_new_images).
            unmatched_this_run = [img for img in images if not img.get("figure_ref")]
            return persisted + unmatched_this_run

    except Exception as e:
        logger.warning(f"⚠️ figure_database: persistent lookup failed ({e}) — falling back to "
                        f"a one-off, non-persisted extraction for this exercise only.")
        try:
            return extract_diagram_images(pdf_path, class_name=class_name)
        except Exception as e2:
            logger.warning(f"⚠️ figure_database: fallback extraction also failed ({e2}).")
            return []


def detect_image_ext(image_bytes: bytes, fallback: str) -> str:
    """PRODUCTION-AUDIT FIX (originally review_cli.py's _detect_image_ext,
    moved here so correction_engine.py's attach_manual_figure_to_question()
    can reuse it too instead of duplicating the same logic): the stored
    extension becomes the mime type solver.py embeds downstream verbatim
    (`f"image/{ext}"`), so trusting an uploaded/pasted screenshot's
    FILENAME extension is a real risk — a JPEG saved with a '.png' name
    (common from screenshot tools and browser downloads) would silently
    embed the WRONG mime type in the published PDF.

    Detects the REAL format from the image bytes themselves via PIL
    (already a project dependency) and only falls back to the caller-
    supplied `fallback` if PIL genuinely can't identify the format —
    never raises.
    """
    try:
        from PIL import Image
        import io
        fmt = Image.open(io.BytesIO(image_bytes)).format
        if fmt:
            return {"JPEG": "jpg"}.get(fmt, fmt.lower())
    except Exception:
        pass
    return fallback


def add_manual_figure(image_bytes: bytes, figure_number, class_name, chapter,
                       subject: str = None, page_number=None, reviewer: str = None,
                       book_id: str = None, book_url: str = None, ext: str = "png",
                       bbox=None, notes: str = None) -> dict:
    """HUMAN REVIEW WORKFLOW — "Missing Book Figure" capability.

    Permanently stores a manually cropped/uploaded textbook figure into
    THE SAME persistent store get_verified_figures() already reads from
    — no new storage layer, no new lookup path. It is keyed the exact
    same way an automatically-extracted figure is: figure_ref normalized
    to plain ASCII digits (e.g. "4.12") via utils.extract_figure_reference
    (reused unchanged, by wrapping the reviewer's raw figure number as
    "Figure <n>" text) — so a future question that cites "চিত্ৰ 4.12" /
    "Figure 4.12" resolves to this entry through solver.py's existing,
    UNCHANGED exact-ref matching in _attach_book_diagrams(). Nothing
    downstream needs to know this figure came from a human instead of
    the automatic extractor.

    A manual upload always WINS over (overwrites) any prior entry for
    the same figure_ref — a human reviewer is the highest-trust source
    this project has, and re-uploading a corrected crop for a figure
    number that was previously wrong is exactly how that gets fixed.
    The same filename is kept, so every existing citation path keeps
    working with zero other code changes. A `version` counter tracks
    how many times a given figure_ref has been (re-)supplied.

    Raises ValueError for genuinely unusable input (no bytes, no usable
    figure number) — unlike get_verified_figures()'s fail-open policy,
    this is a deliberate, low-volume, human-driven action where an
    immediate, clear error is more useful to the reviewer than a silent
    no-op.
    """
    if not image_bytes:
        raise ValueError("add_manual_figure: no image bytes supplied.")

    figure_ref = extract_figure_reference(f"Figure {figure_number}")
    if not figure_ref:
        raise ValueError(f"add_manual_figure: '{figure_number}' is not a usable figure number "
                          f"(expected something like '4.12' or '7').")

    resolved_book_id = book_id or _book_id_from_url(book_url)
    chapter_dir = _chapter_dir(resolved_book_id, chapter)
    ext = (ext or "png").lstrip(".").lower()
    if not ext.isalnum():
        ext = "png"
    sha256 = hashlib.sha256(image_bytes).hexdigest()

    with _chapter_lock(chapter_dir):
        index = _load_index(chapter_dir)
        # A manual upload must survive the NEXT get_verified_figures() call
        # for this chapter — without this, an extractor_version mismatch
        # (e.g. this chapter has never been auto-extracted yet, so
        # extractor_version is still None) would make that call think the
        # whole chapter's store is stale and wipe it, silently discarding
        # the figure a reviewer just manually verified.
        index["extractor_version"] = EXTRACTOR_VERSION

        dup = next((e for e in index["figures"]
                    if e.get("sha256") == sha256 and e["figure_ref"] != figure_ref), None)
        if dup:
            logger.warning(f"⚠️ figure_database: uploaded image for figure {figure_ref} is "
                            f"byte-identical to already-stored figure {dup['figure_ref']} in "
                            f"{resolved_book_id}/chapter {chapter} — please double-check the "
                            f"figure numbers before continuing (both entries are kept as-is).")

        safe_ref = re.sub(r"[^a-zA-Z0-9._-]", "_", figure_ref)
        filename = f"figures/{safe_ref}.{ext}"
        os.makedirs(os.path.join(chapter_dir, "figures"), exist_ok=True)
        full_path = os.path.join(chapter_dir, filename)
        with open(full_path, "wb") as f:
            f.write(image_bytes)

        existing = next((e for e in index["figures"] if e["figure_ref"] == figure_ref), None)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry = {
            "figure_ref": figure_ref,
            "file": filename,
            "ext": ext,
            "source": "manual",
            "verified_by_human": True,
            "class_name": class_name,
            "subject": subject,
            "chapter": chapter,
            "page_number": page_number,
            "bbox": bbox,
            "reviewer": reviewer,
            "notes": notes,
            "sha256": sha256,
            "version": (existing.get("version", 1) + 1) if existing else 1,
            "first_seen": existing.get("first_seen", now) if existing else now,
            "uploaded_at": now,
        }
        if existing:
            index["figures"] = [entry if e is existing else e for e in index["figures"]]
            logger.info(f"🔁 figure_database: figure {figure_ref} REPLACED with a new manually "
                        f"verified crop (version {entry['version']}) for {resolved_book_id}/"
                        f"chapter {chapter}.")
        else:
            index["figures"].append(entry)
            logger.info(f"📌 figure_database: figure {figure_ref} manually added and permanently "
                        f"verified for {resolved_book_id}/chapter {chapter} — future questions "
                        f"citing it will reuse this crop automatically, no repeated manual work.")

        _save_index(chapter_dir, index)

    return entry
