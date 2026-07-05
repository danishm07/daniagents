#!/usr/bin/env python
"""Download and cache the historical event archive, headlessly.

The command-line counterpart to ``notebooks/01_historical_archive.ipynb`` — handy
for pulling data on a server or in a cron job. Needs ``EM_API_KEY`` in the
environment or a local ``.env`` file.

Examples:
    # Everything, into the default data/archive/ cache
    uv run python scripts/download_archive.py

    # Just earnings, 2026Q1 onward, into a custom directory
    uv run python scripts/download_archive.py --event-type EARNINGS_RELEASE \\
        --since 2026Q1 --dest /data/em-archive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from examples import Client
from examples.archive import download_archive
from examples.schemas import ArchiveFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/archive"),
        help="Directory to cache files into (default: data/archive).",
    )
    parser.add_argument(
        "--event-type",
        action="append",
        metavar="TYPE",
        help="Only this event type; repeatable. Default: all types.",
    )
    parser.add_argument(
        "--since",
        metavar="QUARTER",
        help="Only quarters >= this (lexical compare, e.g. 2026Q1). Default: all.",
    )
    return parser.parse_args()


def make_filter(event_types: list[str] | None, since: str | None):
    types = set(event_types) if event_types else None

    def keep(f: ArchiveFile) -> bool:
        if types is not None and f.event_type not in types:
            return False
        return not (since is not None and f.quarter < since)

    return keep


def main() -> None:
    args = parse_args()
    keep = make_filter(args.event_type, args.since)

    try:
        client = Client.from_env()
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    with client:
        manifest = client.archive_manifest()
        paths = download_archive(manifest, args.dest, only=keep, client=client)

    total = sum(p.stat().st_size for p in paths)
    print(f"\nDone: {len(paths)} file(s), {total / 1_000_000:.1f} MB in {args.dest}")


if __name__ == "__main__":
    main()
