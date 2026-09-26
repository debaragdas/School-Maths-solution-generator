"""
Regression tests for downloader.py's cache-integrity fix (see the
module docstring in downloader.py for the root cause this closes:
changing config.BOOK_URL used to silently reuse the previous book's
cached PDF).

No real network access is used or required — requests.get is
monkeypatched to write deterministic fake "PDF" bytes locally, which
is enough to exercise the caching logic (the part that was actually
buggy) without depending on internet access being available in CI or
this sandbox.

Run: python3 -m unittest test_downloader -v
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import downloader


def _fake_pdf_bytes(tag: str) -> bytes:
    # Real %PDF- header (required by downloader._verify_pdf) followed by
    # tag-specific padding so two different "books" hash differently and
    # are each comfortably over the module's real 500_000-byte
    # cache-hit size floor (downloader.py only trusts a cache entry
    # that's > 500_000 bytes, matching a real book PDF).
    return b"%PDF-1.4\n" + (tag.encode() * 600_000)


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestDownloaderCacheIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.book_a_url = "https://example.com/class9_book.pdf"
        self.book_b_url = "https://example.com/class10_book.pdf"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mock_get(self, url_to_bytes):
        def fake_get(link, **kwargs):
            return _FakeResponse(url_to_bytes[link])
        return fake_get

    def test_fresh_download_writes_metadata(self):
        with mock.patch("downloader.requests.get", side_effect=self._mock_get({self.book_a_url: _fake_pdf_bytes("A")})):
            path = downloader.download_pdf(self.book_a_url, self.tmpdir)
        self.assertTrue(os.path.exists(path))
        meta_path = path + ".meta.json"
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta["source_url"], self.book_a_url)
        self.assertEqual(len(meta["sha256"]), 64)

    def test_same_url_second_call_is_a_real_cache_hit_no_network_call(self):
        get_mock = mock.Mock(side_effect=self._mock_get({self.book_a_url: _fake_pdf_bytes("A")}))
        with mock.patch("downloader.requests.get", get_mock):
            downloader.download_pdf(self.book_a_url, self.tmpdir)
            self.assertEqual(get_mock.call_count, 1)
            downloader.download_pdf(self.book_a_url, self.tmpdir)
            # cache hit -> requests.get must NOT be called a second time
            self.assertEqual(get_mock.call_count, 1)

    def test_changed_url_triggers_redownload_not_silent_reuse(self):
        """This is the exact bug scenario from the audit: BOOK_URL
        changes (e.g. class 9 book -> class 10 book) without the caller
        clearing temp/ themselves. The fixed cache must detect this and
        redownload rather than silently handing back the old book."""
        url_to_bytes = {self.book_a_url: _fake_pdf_bytes("A"), self.book_b_url: _fake_pdf_bytes("B")}
        with mock.patch("downloader.requests.get", side_effect=self._mock_get(url_to_bytes)):
            path_a = downloader.download_pdf(self.book_a_url, self.tmpdir)
            with open(path_a, "rb") as f:
                content_after_a = f.read()
            self.assertTrue(content_after_a.startswith(b"%PDF-1.4\nAAAA"))

            path_b = downloader.download_pdf(self.book_b_url, self.tmpdir)
            with open(path_b, "rb") as f:
                content_after_b = f.read()
            # same fixed filename/path, but content MUST now be book B's,
            # not silently still book A's.
            self.assertTrue(content_after_b.startswith(b"%PDF-1.4\nBBBB"))
            self.assertNotEqual(content_after_a, content_after_b)

            with open(path_b + ".meta.json") as f:
                meta = json.load(f)
            self.assertEqual(meta["source_url"], self.book_b_url)

    def test_externally_corrupted_file_with_stale_metadata_is_redownloaded(self):
        """If the cached file on disk no longer matches its OWN recorded
        hash (e.g. truncated by a crashed previous run, or edited
        externally), the cache must not trust the stale metadata's URL
        match alone — content hash must also still agree."""
        with mock.patch("downloader.requests.get", side_effect=self._mock_get({self.book_a_url: _fake_pdf_bytes("A")})):
            path = downloader.download_pdf(self.book_a_url, self.tmpdir)

        # tamper with the file without touching the metadata
        with open(path, "ab") as f:
            f.write(b"\x00" * 10)

        get_mock = mock.Mock(side_effect=self._mock_get({self.book_a_url: _fake_pdf_bytes("A")}))
        with mock.patch("downloader.requests.get", get_mock):
            downloader.download_pdf(self.book_a_url, self.tmpdir)
            self.assertEqual(get_mock.call_count, 1, "corrupted-but-metadata-matching file must trigger a redownload")

    def test_missing_metadata_with_existing_large_file_is_redownloaded(self):
        """Simulates upgrading from the OLD downloader (no metadata file
        at all) — an old cached file with no sidecar must not be trusted
        blindly; it should be treated as a cache miss and redownloaded,
        which also backfills correct metadata going forward."""
        output_path = os.path.join(self.tmpdir, "source_book.pdf")
        with open(output_path, "wb") as f:
            f.write(_fake_pdf_bytes("OLD"))  # no .meta.json written — simulates pre-fix state

        get_mock = mock.Mock(side_effect=self._mock_get({self.book_a_url: _fake_pdf_bytes("A")}))
        with mock.patch("downloader.requests.get", get_mock):
            path = downloader.download_pdf(self.book_a_url, self.tmpdir)
        self.assertEqual(get_mock.call_count, 1)
        self.assertTrue(os.path.exists(path + ".meta.json"))

    def test_corrupted_metadata_json_is_treated_as_cache_miss(self):
        with mock.patch("downloader.requests.get", side_effect=self._mock_get({self.book_a_url: _fake_pdf_bytes("A")})):
            path = downloader.download_pdf(self.book_a_url, self.tmpdir)
        with open(path + ".meta.json", "w") as f:
            f.write("{not valid json")

        get_mock = mock.Mock(side_effect=self._mock_get({self.book_a_url: _fake_pdf_bytes("A")}))
        with mock.patch("downloader.requests.get", get_mock):
            downloader.download_pdf(self.book_a_url, self.tmpdir)
        self.assertEqual(get_mock.call_count, 1)


class TestSSLFallbackRespectsAllowInsecureSSL(unittest.TestCase):
    """Regression tests for the SSL-verification-bypass fix.

    ROOT CAUSE: after two SSL failures (default trust store, then
    certifi's bundle), the old code fell back to `verify=False` (an
    UNVERIFIED download) unconditionally — even when
    config.ALLOW_INSECURE_SSL was left at its safe default (False).
    That contradicted the module's own documented security model: the
    flag is supposed to be the ONLY way to disable certificate
    verification. Fixed so a persistent SSL failure now raises instead
    of silently downgrading security, unless the user has explicitly
    opted in via ALLOW_INSECURE_SSL = True."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.url = "https://example.com/class9_book.pdf"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        import config
        config.ALLOW_INSECURE_SSL = False  # restore safe default for other tests

    def test_persistent_ssl_failure_raises_instead_of_falling_back_when_disallowed(self):
        import config
        import requests
        config.ALLOW_INSECURE_SSL = False

        def always_ssl_error(link, **kwargs):
            raise requests.exceptions.SSLError("cert verify failed")

        with mock.patch("downloader.requests.get", side_effect=always_ssl_error):
            with self.assertRaises(requests.exceptions.SSLError):
                downloader.download_pdf(self.url, self.tmpdir)

        # Confirm no partial/unverified file was written to disk.
        self.assertEqual(os.listdir(self.tmpdir), [])

    def test_persistent_ssl_failure_falls_back_when_explicitly_allowed(self):
        import config
        import requests
        config.ALLOW_INSECURE_SSL = True

        calls = {"n": 0}

        def ssl_error_then_ok(link, **kwargs):
            calls["n"] += 1
            if kwargs.get("verify") is not False:
                raise requests.exceptions.SSLError("cert verify failed")
            return _FakeResponse(_fake_pdf_bytes("INSECURE"))

        with mock.patch("downloader.requests.get", side_effect=ssl_error_then_ok):
            path = downloader.download_pdf(self.url, self.tmpdir)

        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
