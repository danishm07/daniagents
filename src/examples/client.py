"""A small, typed client for the Explaining Markets public API.

Covers the endpoints you *read from* or *trigger* while exploring the
competition:

    client.events()            GET  /events                  calendar
    client.health()            GET  /health                  your counters
    client.archive_manifest()  GET  /archive                 history manifest
    client.archive_file(...)   GET  /archive/{type}/{quarter} fresh signed URL
    client.send_test_event()   POST /webhook/test            synthetic delivery

Prediction *submission* (``POST /predictions``) is deliberately not here: it
places a real, scored entry and belongs to a deployed agent, not an example.

Authentication is the ``X-API-Key`` header. Every call raises
:class:`ApiError` with a readable message on a non-2xx response.
"""

from __future__ import annotations

from datetime import datetime
from types import TracebackType
from typing import Any

import httpx

from examples.config import Config, load_config
from examples.schemas import ArchiveFile, ArchiveManifest, CalendarEvent, SubmissionHealth


class ApiError(RuntimeError):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _isoformat(value: str | datetime | None) -> str | None:
    """Normalize a datetime (or already-ISO string) to an ISO-8601 UTC string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return value


class Client:
    """Thin wrapper over ``httpx`` for the competition API.

    Use as a context manager so the underlying connection pool is closed::

        with Client.from_env() as client:
            events = client.events()
    """

    def __init__(self, config: Config, *, timeout: float = 30.0) -> None:
        self._config = config
        api_key = config.require_api_key()
        self._http = httpx.Client(
            base_url=config.api_base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    @classmethod
    def from_env(cls, *, timeout: float = 30.0) -> Client:
        """Build a client from environment / ``.env`` configuration."""
        return cls(load_config(), timeout=timeout)

    # -- context manager -------------------------------------------------

    def __enter__(self) -> Client:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # -- endpoints -------------------------------------------------------

    def events(
        self,
        *,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> list[CalendarEvent]:
        """``GET /events`` — the events calendar, optionally within a date window.

        Omit both bounds for the full calendar. Bounds may be ``datetime``
        objects or ISO-8601 strings.
        """
        params = {
            k: v
            for k, v in (
                ("start_date", _isoformat(start_date)),
                ("end_date", _isoformat(end_date)),
            )
            if v is not None
        }
        data = self._get("/events", params=params)
        return [CalendarEvent.model_validate(item) for item in data]

    def health(self) -> SubmissionHealth:
        """``GET /health`` — rolling 24h webhook + submission counters."""
        return SubmissionHealth.model_validate(self._get("/health"))

    def archive_manifest(self) -> ArchiveManifest:
        """``GET /archive`` — the historical archive manifest (signed URLs)."""
        return ArchiveManifest.model_validate(self._get("/archive"))

    def archive_file(self, event_type: str, quarter: str) -> ArchiveFile:
        """``GET /archive/{event_type}/{quarter}`` — a fresh signed URL for one file.

        Use this to refresh a single download link that has expired, without
        re-fetching the whole manifest.
        """
        return ArchiveFile.model_validate(self._get(f"/archive/{event_type}/{quarter}"))

    def send_test_event(self) -> int:
        """``POST /webhook/test`` — fire a synthetic ``TEST`` delivery at your webhook.

        Returns the HTTP status code (``202`` when the delivery is enqueued).
        Requires a webhook URL and credentials already configured in the portal.
        """
        resp = self._http.post("/webhook/test")
        self._raise_for_status(resp)
        return resp.status_code

    # -- internals -------------------------------------------------------

    def _get(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        resp = self._http.get(path, params=params)
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        if resp.is_success:
            return
        hint = {
            401: " (check EM_API_KEY — it may be missing, wrong, or for another stage)",
            403: " (submission disabled or key not recognized)",
            404: " (not found)",
        }.get(resp.status_code, "")
        raise ApiError(
            resp.status_code,
            f"{resp.request.method} {resp.request.url.path} → "
            f"{resp.status_code}{hint}: {resp.text[:300]}",
        )
