"""
exercise_splitter.py — within a chapter's page range, find every
"Exercise X.Y" / "অনুশীলনী X.Y" heading and split the chapter PDF into
one small PDF per exercise. Pure Python, no AI.
"""
import os
import re
import fitz  # PyMuPDF
from utils import logger

_ASSAMESE_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_EXERCISE_PATTERNS = [
    re.compile(r"exercise\s*[:\-]?\s*(\d+\.\d+)", re.IGNORECASE),
    re.compile(r"অনুশীলনী\s*[:\-]?\s*([০-৯\d]+\.[০-৯\d]+)"),
]


def _normalize(text: str) -> str:
    return text.translate(_ASSAMESE_DIGITS)


def _find_exercise_label(text: str) -> str | None:
    normalized = _normalize(text)
    for pattern in _EXERCISE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            return m.group(1)
    return None


def find_exercise_headings(pdf_path: str, start_page: int, end_page: int) -> list[tuple[int, str]]:
    """Returns [(page_number, "7.1"), (page_number, "7.2"), ...] within the chapter's range."""
    doc = fitz.open(pdf_path)
    headings = []
    for page_num in range(start_page, end_page + 1):
        text = doc[page_num - 1].get_text()  # fitz pages are 0-indexed
        if not text.strip():
            continue
        top_slice = text[: max(200, len(text) // 3)]
        label = _find_exercise_label(top_slice)
        if label:
            headings.append((page_num, label))
    doc.close()

    # De-duplicate consecutive same-label detections (a heading repeated
    # on a running header shouldn't count as a new exercise boundary).
    deduped = []
    for page, label in headings:
        if not deduped or deduped[-1][1] != label:
            deduped.append((page, label))
    return deduped


def split_exercises(pdf_path: str, start_page: int, end_page: int, temp_dir: str,
                     class_name: int, chapter: int, text_layer_reliable: bool = True) -> list[dict]:
    """
    Fully automatic — never prompts for manual page numbers. If the text
    layer is known-unreliable (see vision_ocr.is_text_layer_reliable) or
    the deterministic regex scan simply finds nothing in this chapter's
    page range, falls back to a cached Gemini Vision scan of the page
    images instead of stopping the pipeline.
    """
    headings: list[tuple[int, str]] = []
    if text_layer_reliable:
        headings = find_exercise_headings(pdf_path, start_page, end_page)

    if not headings:
        if not text_layer_reliable:
            logger.warning(
                "⚠️ Text layer for this PDF is unreliable — skipping regex-based exercise "
                "detection entirely and using Gemini Vision."
            )
        else:
            logger.warning(
                f"⚠️ No exercise headings found via text scan in pages {start_page}-{end_page} "
                f"— falling back to Gemini Vision rather than asking for manual page numbers."
            )
        from vision_ocr import get_exercise_index_for_range
        vision_index = get_exercise_index_for_range(pdf_path, start_page, end_page, class_name, chapter)
        headings = [(h["page"], h["label"]) for h in vision_index]

    if not headings:
        raise ValueError(
            f"No exercise headings found between pages {start_page}-{end_page}, even via "
            f"Gemini Vision. This chapter's page range may be wrong, or the exercises in "
            f"this book use wording the vision prompt doesn't recognize."
        )

    os.makedirs(temp_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    exercises = []

    for i, (heading_page, label) in enumerate(headings):
        next_page = headings[i + 1][0] if i + 1 < len(headings) else end_page + 1
        ex_start = heading_page
        ex_end = next_page - 1

        # Namespaced by class/chapter (fix from the V7 engineering audit):
        # the old filename was just f"exercise_{label}.pdf", so
        # "Exercise 3.2" from two DIFFERENT chapters/classes collided on
        # the exact same temp path. Combined with book_figure_index.py's
        # cache (now content-hashed — see that module — so this no
        # longer risks a wrong-figure cache hit either way), this keeps
        # every chapter/class's exercise files fully separate on disk.
        out_path = os.path.join(temp_dir, f"class{class_name}_ch{chapter}_exercise_{label}.pdf")
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=ex_start - 1, to_page=ex_end - 1)
        new_doc.save(out_path)
        new_doc.close()

        exercises.append({
            "label": label, "path": out_path,
            "start_page": ex_start, "end_page": ex_end,
        })
        logger.info(f"✂️ Exercise {label}: pages {ex_start}-{ex_end} → {out_path}")

    doc.close()
    return exercises
