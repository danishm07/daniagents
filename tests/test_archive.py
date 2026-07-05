"""Archive download/cache/load."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx
import respx

from examples.archive import download_archive, load_archive, read_jsonl_gz
from examples.client import Client
from examples.config import Config
from examples.schemas import ArchiveManifest
from tests.conftest import SAMPLE_ARCHIVE, TEST_BASE_URL


def _gz_bytes(records: list[dict]) -> bytes:
    import io

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as fh:
        for r in records:
            fh.write((json.dumps(r) + "\n").encode("utf-8"))
    return buf.getvalue()


def test_read_jsonl_gz_sample() -> None:
    records = list(read_jsonl_gz(SAMPLE_ARCHIVE))
    assert len(records) == 5
    assert records[0]["focal_assets"][0]["identifier_value"] == "AAPL"


def test_load_archive_adds_provenance_from_filename() -> None:
    df = load_archive(SAMPLE_ARCHIVE)
    assert len(df) == 5
    assert set(df["quarter"]) == {"2025Q3"}
    assert set(df["event_type"]) == {"EARNINGS_RELEASE"}


def test_load_archive_empty_dir(tmp_path: Path) -> None:
    assert load_archive(tmp_path).empty


@respx.mock
def test_download_archive_writes_and_then_caches(tmp_path: Path, config: Config) -> None:
    payload = _gz_bytes([{"event_id": "x", "event_type": "EARNINGS_RELEASE"}])
    route = respx.get("https://cdn.test/signed").mock(
        return_value=httpx.Response(200, content=payload)
    )
    manifest = ArchiveManifest.model_validate(
        {
            "files": [
                {
                    "event_type": "EARNINGS_RELEASE",
                    "quarter": "2025Q3",
                    "key": "k",
                    "url": "https://cdn.test/signed",
                    "bytes": len(payload),
                }
            ]
        }
    )

    paths = download_archive(manifest, tmp_path, on_progress=None)
    assert paths[0].read_bytes() == payload
    assert route.call_count == 1

    # Second run: file is cached at the expected size, so no new download.
    download_archive(manifest, tmp_path, on_progress=None)
    assert route.call_count == 1


@respx.mock
def test_download_archive_refreshes_expired_url(tmp_path: Path, config: Config) -> None:
    payload = _gz_bytes([{"event_id": "y"}])
    respx.get("https://cdn.test/expired").mock(return_value=httpx.Response(403))
    respx.get(f"{TEST_BASE_URL}/archive/EARNINGS_RELEASE/2025Q3").mock(
        return_value=httpx.Response(
            200,
            json={
                "event_type": "EARNINGS_RELEASE",
                "quarter": "2025Q3",
                "key": "k",
                "url": "https://cdn.test/fresh",
            },
        )
    )
    fresh = respx.get("https://cdn.test/fresh").mock(
        return_value=httpx.Response(200, content=payload)
    )
    manifest = ArchiveManifest.model_validate(
        {
            "files": [
                {
                    "event_type": "EARNINGS_RELEASE",
                    "quarter": "2025Q3",
                    "key": "k",
                    "url": "https://cdn.test/expired",
                    "bytes": len(payload),
                }
            ]
        }
    )

    with Client(config) as client:
        paths = download_archive(manifest, tmp_path, client=client, on_progress=None)

    assert fresh.called
    assert paths[0].read_bytes() == payload
