"""Turn API responses into tidy, display-friendly pandas objects.

Kept separate from :mod:`examples.client` so the client stays a thin transport
layer and the "make it readable" logic lives in one place the notebooks share.
"""

from __future__ import annotations

import pandas as pd

from examples.schemas import ArchiveManifest, CalendarEvent, SubmissionHealth

# Columns, in reading order, for the events table.
_EVENT_COLUMNS = [
    "event_datetime",
    "knowledge_cutoff",
    "event_type",
    "timing_category",
    "tickers",
    "n_assets",
    "event_id",
]


def events_frame(events: list[CalendarEvent]) -> pd.DataFrame:
    """A sorted, human-readable table of calendar events.

    Parses ``event_datetime`` to a UTC timestamp, flattens ``focal_assets`` into
    a comma-joined ``tickers`` string plus an ``n_assets`` count, and sorts
    chronologically.
    """
    rows = [
        {
            "event_datetime": ev.event_datetime,
            "knowledge_cutoff": ev.knowledge_cutoff,
            "event_type": ev.event_type,
            "timing_category": ev.timing_category,
            "tickers": ", ".join(a.identifier_value for a in ev.focal_assets),
            "n_assets": len(ev.focal_assets),
            "event_id": ev.event_id,
        }
        for ev in events
    ]
    df = pd.DataFrame(rows, columns=_EVENT_COLUMNS)
    if not df.empty:
        df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True, format="ISO8601")
        df["knowledge_cutoff"] = pd.to_datetime(df["knowledge_cutoff"], utc=True, format="ISO8601")
        df = df.sort_values("event_datetime").reset_index(drop=True)
    return df


def manifest_frame(manifest: ArchiveManifest) -> pd.DataFrame:
    """A readable overview of the archive manifest, one row per file.

    Shows what each file covers (event type, quarter, event count, size, date
    span, sealed flag) — but not the long signed URLs.
    """
    rows = [
        {
            "event_type": f.event_type,
            "quarter": f.quarter,
            "events": f.events,
            "size_mb": round(f.bytes / 1_000_000, 2) if f.bytes is not None else None,
            "sealed": f.sealed,
            "covers_from": f.event_datetime_min,
            "covers_to": f.event_datetime_max,
        }
        for f in manifest.files
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "event_type",
            "quarter",
            "events",
            "size_mb",
            "sealed",
            "covers_from",
            "covers_to",
        ],
    )
    if not df.empty:
        df = df.sort_values(["event_type", "quarter"]).reset_index(drop=True)
    return df


def health_frame(health: SubmissionHealth) -> pd.DataFrame:
    """A tidy ``group / metric / value`` view of the health counters.

    Only populated fields are shown, split into the outbound-webhook group and
    the inbound-submission group.
    """
    data = health.model_dump()
    rows = [
        {
            "group": "webhook" if metric.startswith("webhook_") else "submission",
            "metric": metric,
            "value": value,
        }
        for metric, value in data.items()
        if value is not None and metric != "submission_id"
    ]
    return pd.DataFrame(rows, columns=["group", "metric", "value"])
