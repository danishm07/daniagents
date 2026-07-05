"""Download, cache, and load the historical event archive.

The archive (``GET /archive``) is how you get historical events for offline
backtesting: one gzip-compressed JSONL file per ``event_type`` by ``quarter``,
each line a past event. Download URLs are short-lived and signed, so this module
caches files locally (default: ``data/archive/``) and can refresh an expired URL
via ``GET /archive/{event_type}/{quarter}``.

Typical use::

    with Client.from_env() as client:
        manifest = client.archive_manifest()
        paths = download_archive(manifest, "data/archive", client=client)
    df = load_archive("data/archive")
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import httpx
import pandas as pd

from examples.client import Client
from examples.schemas import ArchiveFile, ArchiveManifest

ProgressFn = Callable[[str], None]
FileFilter = Callable[[ArchiveFile], bool]

_CHUNK = 1 << 16


def download_archive(
    manifest: ArchiveManifest,
    dest_dir: str | Path,
    *,
    only: FileFilter | None = None,
    client: Client | None = None,
    on_progress: ProgressFn | None = print,
) -> list[Path]:
    """Download the manifest's files into ``dest_dir``, skipping up-to-date ones.

    A file is skipped when it already exists locally at the size the manifest
    reports (``bytes``). Pass ``only=lambda f: f.quarter >= "2026Q1"`` to fetch a
    subset. If a signed URL has expired and ``client`` is provided, the URL is
    refreshed once and the download retried. Returns the local paths, in manifest
    order.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in manifest.files if only is None or only(f)]

    paths: list[Path] = []
    with httpx.Client(follow_redirects=True, timeout=120.0) as downloader:
        for f in files:
            target = dest / f.filename
            if _is_cached(target, f.bytes):
                _log(
                    on_progress, f"skip     {f.filename} (cached, {target.stat().st_size:,} bytes)"
                )
                paths.append(target)
                continue
            _log(on_progress, f"download {f.filename} ({(f.bytes or 0):,} bytes) …")
            url = f.url
            try:
                _stream_to_file(downloader, url, target)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (403, 404) and client is not None:
                    _log(on_progress, f"  URL expired — refreshing {f.event_type}/{f.quarter}")
                    url = client.archive_file(f.event_type, f.quarter).url
                    _stream_to_file(downloader, url, target)
                else:
                    raise
            paths.append(target)
    return paths


def read_jsonl_gz(path: str | Path) -> Iterator[dict]:
    """Yield each record from a gzip-JSONL file, skipping blank lines."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_archive(source: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Load one or more ``*.jsonl.gz`` archive files into a single DataFrame.

    ``source`` may be a directory (all ``*.jsonl.gz`` within it), a single file,
    or an iterable of paths. Provenance columns ``event_type`` and ``quarter``
    are added from each filename — unless a record already carries that field, in
    which case the record's own value is kept.
    """
    frames = [_load_one(p) for p in _resolve_paths(source)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# -- internals -----------------------------------------------------------


def _load_one(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl_gz(path))
    event_type, quarter = _parse_filename(path)
    # Add provenance from the filename, but never clobber a field the record
    # already carries. Insert at the front so it reads left-to-right.
    if "quarter" not in df.columns:
        df.insert(0, "quarter", quarter)
    if "event_type" not in df.columns:
        df.insert(0, "event_type", event_type)
    return df


def _parse_filename(path: Path) -> tuple[str, str]:
    """``EARNINGS_RELEASE_2025Q3.jsonl.gz`` → ``("EARNINGS_RELEASE", "2025Q3")``."""
    stem = path.name.removesuffix(".jsonl.gz")
    event_type, _, quarter = stem.rpartition("_")
    return event_type, quarter


def _resolve_paths(source: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            return sorted(p.glob("*.jsonl.gz"))
        return [p]
    return [Path(p) for p in source]


def _is_cached(target: Path, expected_bytes: int | None) -> bool:
    if not target.exists():
        return False
    # If the manifest reports a size, require an exact match; otherwise any
    # existing non-empty file counts as cached.
    if expected_bytes is None:
        return target.stat().st_size > 0
    return target.stat().st_size == expected_bytes


def _stream_to_file(downloader: httpx.Client, url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    with downloader.stream("GET", url) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(_CHUNK):
                fh.write(chunk)
    tmp.replace(target)  # atomic: a partial download never masquerades as complete


def _log(on_progress: ProgressFn | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)
