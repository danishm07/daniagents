"""Typed models for the public API responses we exercise in the notebooks.

These mirror the competition's OpenAPI schemas (``/docs/api``). They are
deliberately **lenient** — ``extra="allow"`` — so unknown or newly added fields
pass through untouched rather than raising. Event types in particular are an
open string set — currently only ``EARNINGS_RELEASE`` is emitted — so we keep
them as plain ``str`` and never validate against a closed enum; new types can
appear without breaking parsing.

Only the read/exercise endpoints are modeled here. Prediction *submission*
(``POST /predictions``) is intentionally omitted — that belongs to a live agent
running in a deployed webhook handler, not to these examples.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class AssetIdentifier(_Model):
    """A focal asset, e.g. ``{"identifier_type": "TICKER", "identifier_value": "AAPL"}``."""

    identifier_type: str
    identifier_value: str


class CalendarEvent(_Model):
    """One entry from ``GET /events`` — a scheduled or realized event."""

    event_id: str
    event_type: str
    timing_category: str  # "SCHEDULED" | "UNSCHEDULED"
    event_datetime: str
    # The event's knowledge cutoff: an agent must not use information from
    # after this instant (ISO 8601, UTC). Optional so older cached/sample
    # payloads from before the field existed still parse.
    knowledge_cutoff: str | None = None
    focal_assets: list[AssetIdentifier] = []


class DisclosureItem(_Model):
    """One piece of content attached to a realized event.

    For earnings calls the item to look for is ``kind="facts"`` with
    ``source="earnings_call"``: its ``content`` is the ten-sentence summary whose
    provenance is documented in :mod:`examples.summary`. Both fields are open
    string sets, like ``event_type`` — new kinds and sources can appear without
    breaking parsing, so match on them rather than assuming a single item.

    Larger or binary artifacts are referenced rather than inlined, via
    ``media_type`` / ``url`` / ``bytes`` / ``sha256``.
    """

    id: str | None = None
    kind: str
    source: str | None = None
    content: list[str] = []
    media_type: str | None = None
    url: str | None = None
    bytes: int | None = None
    sha256: str | None = None


class Disclosure(_Model):
    """The ``disclosure`` block on a realized event — a bundle of content items."""

    schema_version: str | None = None
    generated_at: str | None = None
    items: list[DisclosureItem] = []


class ArchiveFile(_Model):
    """One quarter-by-event-type file in the historical archive manifest.

    ``url`` is a short-lived CloudFront-signed link to a gzip-JSONL file. Refresh
    it with ``GET /archive/{event_type}/{quarter}`` if it expires.
    """

    event_type: str
    quarter: str
    key: str
    url: str
    events: int | None = None
    bytes: int | None = None
    sealed: bool | None = None
    url_expires_at: str | None = None
    event_datetime_min: str | None = None
    event_datetime_max: str | None = None

    @property
    def filename(self) -> str:
        """Stable local filename for this file, e.g. ``EARNINGS_RELEASE_2025Q3.jsonl.gz``."""
        return f"{self.event_type}_{self.quarter}.jsonl.gz"


class ArchiveManifest(_Model):
    """Response from ``GET /archive`` — the list of downloadable archive files."""

    files: list[ArchiveFile] = []
    schema_version: str | None = None
    generated_at: str | None = None
    prefix: str | None = None


class SubmissionHealth(_Model):
    """Response from ``GET /health`` — rolling 24h delivery + submission counters."""

    submission_id: str | None = None
    webhook_n_2xx: int | None = None
    webhook_n_4xx: int | None = None
    webhook_n_5xx: int | None = None
    webhook_n_timeout: int | None = None
    webhook_consecutive_failures: int | None = None
    webhook_last_status: str | None = None
    webhook_last_delivery_at: str | None = None
    submission_n_total: int | None = None
    submission_n_duplicate: int | None = None
    submission_n_late: int | None = None
    submission_n_pre_broadcast: int | None = None
    submission_n_invalid_event: int | None = None
    submission_last_at: str | None = None
