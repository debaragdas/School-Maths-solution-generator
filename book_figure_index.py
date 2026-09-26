"""
book_figure_index.py — makes book_diagram_extractor's scan happen ONCE
per exercise PDF, not on every solution-generation run.

Design, matching the existing architecture (temp/ is already the
project's scratch area for per-run intermediate files — see
exercise_splitter.py):

    exercise_X.Y.pdf
        │
        ▼  (first time only — fingerprint miss)
    extract_diagram_images()          [book_diagram_extractor.py,
        │                               raster + caption-gated vector]
        ▼
    temp/figure_index/<fingerprint>/
        manifest.json     — [{"figure_ref", "file", "page_hint"}, ...]
        0000.png, 0001.png, ...
        │
        ▼  (every run after that — fingerprint hit)
    load manifest + PNG bytes straight off disk, no PDF scan at all

The "fingerprint" is a SHA-256 hash of the exercise PDF's own bytes —
the same content-hashing approach vision_ocr.py already uses for its
own cache, and deliberately NOT the weaker path+size+mtime proxy this
module used before the V7 audit. That proxy could theoretically
collide across two different runs: exercise_splitter.py names its
temp exercise PDFs only by label (e.g. "exercise_3.2.pdf", not
namespaced by class/chapter — see exercise_splitter.py's own docstring
for the matching fix there), so re-running the pipeline against a
DIFFERENT chapter/class after a config change, without wiping temp/
first, could produce a same-path file whose size and mtime happened to
coincide with a stale cache entry — silently attaching the wrong
figures. A content hash makes that impossible: two different exercise
PDFs can never collide on it regardless of path, size, or timestamp.

solver.py's _attach_book_diagrams calls get_figure_index() instead of
extract_diagram_images() directly; nothing about the matching logic
downstream changes, only where the image bytes come from.
"""
import hashlib
import json
import os

from book_diagram_extractor import extract_diagram_images
from utils import logger

INDEX_ROOT = os.path.join("temp", "figure_index")
MANIFEST_NAME = "manifest.json"


def _fingerprint(pdf_path: str) -> str:
    """Content hash of the PDF's own bytes — see the module docstring
    above for why this replaced the earlier path+size+mtime proxy."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


def _index_dir(pdf_path: str) -> str:
    return os.path.join(INDEX_ROOT, _fingerprint(pdf_path))


def _load_from_disk(index_dir: str) -> list[dict] | None:
    manifest_path = os.path.join(index_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        images = []
        for entry in entries:
            img_path = os.path.join(index_dir, entry["file"])
            with open(img_path, "rb") as f:
                png_bytes = f.read()
            images.append({"bytes": png_bytes, "ext": "png", "figure_ref": entry.get("figure_ref"),
                           "source": entry.get("source", "raster")})
        return images
    except Exception as e:
        logger.warning(f"⚠️ Figure index at {index_dir} looked present but failed to load ({e}) "
                        f"— rebuilding from the PDF instead.")
        return None


def _save_to_disk(index_dir: str, images: list[dict]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    entries = []
    for i, img in enumerate(images):
        filename = f"{i:04d}.png"
        with open(os.path.join(index_dir, filename), "wb") as f:
            f.write(img["bytes"])
        entries.append({"file": filename, "figure_ref": img.get("figure_ref"), "source": img.get("source", "raster")})
    with open(os.path.join(index_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def get_figure_index(pdf_path: str, class_name: int = 0) -> list[dict]:
    """Returns the same [{"bytes", "ext", "figure_ref"}, ...] shape as
    book_diagram_extractor.extract_diagram_images(), but only actually
    scans/renders the PDF the first time it's asked about this exact
    file content (keyed by a SHA-256 content fingerprint, NOT by
    path+size+mtime — see _fingerprint() below and its own docstring
    for why path/size/mtime was deliberately rejected as a cache key).
    Every subsequent call — including from a different run of the
    pipeline, as long as temp/ wasn't wiped — loads the cached crops
    straight off disk.

    class_name is passed through to extract_diagram_images() for its
    (rare, unreliable-text-layer-only) Gemini Vision fallback path; it
    has no effect on the fast cache-hit path or the normal text-layer-
    reliable extraction path."""
    index_dir = _index_dir(pdf_path)

    cached = _load_from_disk(index_dir)
    if cached is not None:
        logger.info(f"📇 Figure index hit for {os.path.basename(pdf_path)} "
                    f"({len(cached)} figure(s), loaded from {index_dir}, no PDF re-scan).")
        return cached

    logger.info(f"📇 Figure index miss for {os.path.basename(pdf_path)} — scanning once and caching.")
    images = extract_diagram_images(pdf_path, class_name=class_name)
    try:
        _save_to_disk(index_dir, images)
    except Exception as e:
        # Caching is a speed optimization, not a correctness requirement —
        # a failure to write to disk should never block using the images
        # that were already successfully extracted this run.
        logger.warning(f"⚠️ Could not persist figure index to {index_dir} ({e}) — "
                        f"continuing with in-memory results; will re-scan next run.")
    return images
