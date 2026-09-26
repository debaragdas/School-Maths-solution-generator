"""
downloader.py — download + cache + verify the source book PDF.
Ported near-unchanged from the RAG uploader's download_pdf().

CACHE-INTEGRITY FIX (found during the V7 engineering audit): the
original cache-hit check here was purely "does a file named
source_book.pdf already exist and is it > 500KB?" — with NO check
against the URL that produced it. Since the filename is fixed
regardless of BOOK_URL, changing config.BOOK_URL (e.g. switching which
class/chapter's book you're generating) without manually clearing
temp/ would silently keep using the PREVIOUS book's PDF under the
NEW config's class/chapter labels — a completely silent wrong-book
failure mode, not just a stale-but-harmless cache hit.

Fixed by writing a small sidecar metadata file (source_book.meta.json)
alongside the PDF recording the exact URL and a SHA-256 content hash
it was downloaded from. A cache "hit" now additionally requires: (a)
the requested URL matches the URL on record, AND (b) the file on disk
still hashes to what's on record (catches a partially-overwritten or
externally-modified file too, not just a URL change). Any mismatch is
treated as a stale cache and triggers a clean redownload — this is
the "fail safely instead of publishing wrong content" principle
applied to the book-acquisition stage, before anything downstream
ever sees the file.
"""
import hashlib
import json
import os
import requests
from utils import logger
import config

try:
    import gdown
except ImportError:
    gdown = None


def _meta_path(output_path: str) -> str:
    return output_path + ".meta.json"


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_meta(meta_path: str) -> dict | None:
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None  # corrupted metadata is treated the same as "no metadata" -> redownload


def _write_meta(meta_path: str, link: str, file_hash: str):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"source_url": link, "sha256": file_hash}, f)


def _cache_is_valid(output_path: str, link: str) -> bool:
    """A cache hit requires the file to exist, be a plausible size, have
    matching recorded metadata for THIS url, and still hash to what
    that metadata recorded — any single mismatch means "not valid",
    which safely falls through to a fresh download rather than ever
    silently serving the wrong book."""
    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 500_000):
        return False
    meta = _read_meta(_meta_path(output_path))
    if not meta or meta.get("source_url") != link:
        return False
    return meta.get("sha256") == _sha256_of_file(output_path)


def download_pdf(link: str, dest_dir: str, filename: str = "source_book.pdf") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    output_path = os.path.join(dest_dir, filename)
    meta_path = _meta_path(output_path)

    if _cache_is_valid(output_path, link):
        logger.info(f"📄 {filename} already downloaded from this exact URL — skipping.")
        return output_path

    if os.path.exists(output_path):
        # Either too small/corrupted, OR a valid PDF but for a DIFFERENT
        # source_url / hash than what's requested now — in every one of
        # those cases the existing file must not be reused as-is.
        logger.info(f"📄 Cached {filename} doesn't match the requested URL/hash — redownloading.")
        os.remove(output_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    if "drive.google.com" in link:
        if gdown is None:
            raise RuntimeError("gdown not installed — pip install gdown to fetch Google Drive links.")
        logger.info("📥 Downloading from Google Drive...")
        gdown.download(url=link, output=output_path, quiet=False, fuzzy=True)
    else:
        logger.info("📥 Downloading from direct URL...")
        headers = {"User-Agent": "Mozilla/5.0"}
        if config.ALLOW_INSECURE_SSL:
            logger.warning(
                "⚠️ ALLOW_INSECURE_SSL is True — skipping SSL certificate verification "
                "for this download. Only use this if you've confirmed the specific book "
                "URL's own certificate is misconfigured (a genuinely broken/self-signed "
                "cert), not as a default — it removes protection against a tampered or "
                "substituted download."
            )
        try:
            _stream_download(link, headers, output_path, verify=not config.ALLOW_INSECURE_SSL)
        except requests.exceptions.SSLError:
            # Very common on Windows Python installs: "unable to get local
            # issuer certificate" even though the site's certificate is
            # perfectly valid, because requests/urllib3 didn't pick up a
            # complete trust chain from the OS. Retrying once with
            # certifi's own bundled CA file explicitly (instead of
            # whatever verify=True resolved to) fixes this in the
            # overwhelming majority of real cases without weakening
            # security the way ALLOW_INSECURE_SSL does.
            if config.ALLOW_INSECURE_SSL:
                raise  # already using the least-strict setting available — nothing left to retry
            try:
                import certifi
                logger.warning(
                    "⚠️ SSL verification failed with the default trust store — retrying once "
                    "with certifi's own CA bundle (a common fix for Windows Python installs)."
                )
                _stream_download(link, headers, output_path, verify=certifi.where())
            except requests.exceptions.SSLError:
                # BUG FIX (V8 polishing-phase audit): this branch used to fall
                # back to `verify=False` (an UNVERIFIED download) unconditionally,
                # regardless of config.ALLOW_INSECURE_SSL. That directly
                # contradicted this module's own documented security model —
                # ALLOW_INSECURE_SSL is supposed to be the ONLY way to disable
                # certificate verification, opt-in only, precisely because it
                # "removes protection against a tampered or substituted
                # download" (see the warning above). In practice this meant any
                # book URL with a merely misconfigured cert chain — not
                # necessarily an attack — would silently download without
                # verification even with the safe default (ALLOW_INSECURE_SSL
                # = False) left untouched. Now: refuse and surface a clear,
                # actionable error instead of silently downgrading security.
                raise requests.exceptions.SSLError(
                    "SSL certificate verification failed for this book URL, even after "
                    "retrying with certifi's own CA bundle. Refusing to fall back to an "
                    "unverified download because config.ALLOW_INSECURE_SSL is False. If "
                    "you have confirmed this URL's certificate is genuinely misconfigured "
                    "(not a sign of tampering), set ALLOW_INSECURE_SSL = True explicitly "
                    "in config.py and re-run."
                )

    _verify_pdf(output_path, link)
    _write_meta(meta_path, link, _sha256_of_file(output_path))
    return output_path


def _stream_download(link: str, headers: dict, output_path: str, verify):
    with requests.get(link, headers=headers, stream=True, timeout=120, allow_redirects=True,
                       verify=verify) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)


def _verify_pdf(path: str, link: str):
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        raise RuntimeError(f"Download failed — {path} missing or too small. Check BOOK_URL.")
    with open(path, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        os.remove(path)
        raise RuntimeError(
            f"Downloaded file isn't a real PDF (got {header!r}). The URL probably points to an "
            f"HTML viewer page, not the raw PDF — find the direct-download link and use that."
        )
    logger.info(f"✅ PDF verified: {path}")
