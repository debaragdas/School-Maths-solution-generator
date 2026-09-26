"""
books.py — SEBA/SCERT Assamese General Mathematics textbook registry.

The one place that maps a class number (6-10) to its official textbook
PDF URL, so a run never needs config.py edits just to switch books:

    python main.py --class 10 --chapter 7

Classes 9 and 10 are published by SEBA/ASSEB Division-I as ONE pdf per
subject — both URLs below were verified live (HTTP 200) on 2026-08-26.

Classes 6-8 are published by SCERT Assam. SCERT's own site lists these
as chapter-wise downloads behind its textbook page rather than stable
single-file PDFs, so no fixed URL can be honestly hard-coded here yet:
the entries carry the official listing page instead, and book_url_for()
raises a clear, actionable error telling you exactly what to do (pass
--book-url once; downloader.py caches by content hash, so you set this
once per book, ever).

NOTE (2026-27 session): ASSEB's March 2026 notification revises the
Class IX and X Mathematics textbooks for the 2026-27 session. When the
board uploads the new editions, update the URLs below — nothing else in
this project needs to change.
"""

SEBA_MATHS_BOOKS = {
    6: {
        "board": "SCERT",
        "medium": "Assamese",
        "title": "গণিত (ষষ্ঠ শ্ৰেণী) / Mathematics Part 1 & 2",
        "url": None,
        "source_page": ("https://scert.assam.gov.in/portlet-sub-innerpage/"
                        "textbook-from-ka-shreni-to-class-viii-2025-26"),
    },
    7: {
        "board": "SCERT",
        "medium": "Assamese",
        "title": "সপ্তম শ্ৰেণীৰ গণিত পাঠ্যপুথি",
        "url": None,
        "source_page": ("https://scert.assam.gov.in/portlet-sub-innerpage/"
                        "textbook-from-ka-shreni-to-class-viii-2025-26"),
    },
    8: {
        "board": "SCERT",
        "medium": "Assamese",
        "title": "অষ্টম শ্ৰেণীৰ গণিত পাঠ্যপুথি (Part 1 & 2)",
        "url": None,
        "source_page": ("https://scert.assam.gov.in/portlet-sub-innerpage/"
                        "textbook-from-ka-shreni-to-class-viii-2025-26"),
    },
    9: {
        "board": "SEBA/ASSEB Div-I",
        "medium": "Assamese",
        "title": "Ganit (Class IX)",
        "url": "https://site.sebaonline.org/textbooks/class-9/Maths_Ganit_Assamese_IX.pdf",
        "source_page": "https://site.sebaonline.org/textbook",
    },
    10: {
        "board": "SEBA/ASSEB Div-I",
        "medium": "Assamese",
        "title": "Ganit Assamese (Class X)",
        "url": "https://www.sebaonline.org/textbooks/class-10/Ganit_assamese_Class%20X.pdf",
        "source_page": "https://site.sebaonline.org/textbook",
    },
}


class BookNotRegisteredException(ValueError):
    """Raised when a class has no usable single-file book URL yet."""


def book_url_for(class_num) -> str:
    """Returns the registered textbook PDF URL for `class_num`, or raises
    BookNotRegisteredException with an actionable message for classes
    whose board does not publish a stable single-file PDF (SCERT 6-8).
    The message is written for the person running main.py — it names the
    exact flag that fixes the run."""
    entry = SEBA_MATHS_BOOKS.get(int(class_num))
    if entry is None:
        raise BookNotRegisteredException(
            f"No textbook is registered for Class {class_num}. Registered "
            f"classes: {sorted(SEBA_MATHS_BOOKS)}. Pass --book-url <URL> "
            f"to use any other book.")
    url = entry.get("url")
    if not url:
        raise BookNotRegisteredException(
            f"Class {class_num} ({entry['board']}) has no single-file PDF URL "
            f"in the registry yet — {entry['board']}'s site publishes it as "
            f"chapter-wise downloads. Get the full-book PDF from the official "
            f"listing ({entry['source_page']}) and pass its URL via "
            f"--book-url <URL> (cached after the first download).")
    return url


def list_books() -> str:
    """Human-readable registry table for --list-books."""
    lines = []
    for class_num in sorted(SEBA_MATHS_BOOKS):
        e = SEBA_MATHS_BOOKS[class_num]
        status = e["url"] if e["url"] else (
            f"(no single-file URL yet — see {e['source_page']})")
        lines.append(f"  Class {class_num:>2} | {e['board']:<15} | "
                      f"{e['title']} | {status}")
    return "\n".join(lines)
