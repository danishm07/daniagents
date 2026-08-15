"""Authoritative ``knowledge_cutoff`` lookup, cached, with a conservative fallback.

Rules §04 forbids using any information available after an event's
``knowledge_cutoff`` — in data collection, features, prompts, retrieval, search,
model selection or revision — and §07/§10 let the University audit retrieval
records to check. A violation voids prize eligibility **even after final results
are published**.

The webhook body does not carry the cutoff. It carries ``id``, ``event_id``,
``event_type``, ``timing_category``, ``event_datetime``, ``focal_assets``,
``information_url`` and ``prediction_deadline`` — and that is all. So any feature
built on external data has to look the cutoff up, and ``GET /v1/events`` is the
only authoritative source.

**Deriving the cutoff from ``event_datetime`` is not an option.** A derived
cutoff that lands even slightly late is a rules violation, and the failure is
silent: the prediction scores normally and the eligibility problem surfaces, if
ever, at audit. So the derived value is used only as a deliberately *early*
bound when the lookup fails, and which path produced the answer is logged and
returned to the caller, which may reasonably decide to skip an external feature
rather than run it on a guess.

    cutoff, path = knowledge_cutoff(event)
    if path != "api":
        ...  # degrade: skip the external feature rather than risk the window
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import httpx

from explaining_markets.config import Config

#: Measured against the live calendar on 2026-08-15, all 840 scheduled events:
#: the cutoff is **always earlier** than ``event_datetime`` — median 17h, mean
#: 20.5h, max 96h — and always lands at 20:00Z, the US market close. The
#: announcement itself is therefore *later* than the cutoff, essentially always.
#:
#: The first version of this file assumed the opposite and used
#: ``event_datetime - 1h`` as its "conservative" bound. That would have been
#: **later than the true cutoff on 710 of 840 events (85%)** — a Rules §04
#: violation on the large majority of predictions, silent at scoring time and
#: fatal at audit. Even ``(event date - 1 day) at 00:00Z`` overshoots on 7%,
#: because holiday weekends push the lag out to four days.
#:
#: Conclusion: **no function of event_datetime is a safe cutoff.** The fallback
#: is to refuse, and for the caller to skip the external feature.
FALLBACK_SPAN = timedelta(days=5)

EVENTS_TIMEOUT_SECONDS = 10.0

_cache: dict[str, datetime] = {}
_lock = threading.Lock()
_shape_logged = False


def _parse(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _rows(payload: object) -> list[dict]:
    """The event list out of whatever ``/events`` returns.

    Written defensively on purpose: the last time this codebase assumed a
    documented response shape, every event silently took a degraded path for two
    days. An unrecognised shape logs a structural sketch instead of raising.
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("events", "data", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def refresh(*, config: Config | None = None, timeout: float = EVENTS_TIMEOUT_SECONDS) -> int:
    """Pull the events calendar and cache every cutoff it carries.

    One call serves every event in the response, so a dense earnings block costs
    one request rather than one per event. Returns how many cutoffs were cached.
    """
    global _shape_logged
    cfg = config or Config.from_env()
    resp = httpx.get(
        f"{cfg.api_base_url}/events",
        headers={"X-API-Key": cfg.api_key},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()

    rows = _rows(payload)
    if not rows and not _shape_logged:
        keys = sorted(payload)[:12] if isinstance(payload, dict) else type(payload).__name__
        print(f"[CUTOFF] unrecognised /events shape, no rows found. Top level: {keys}")
        _shape_logged = True

    found = 0
    with _lock:
        for row in rows:
            event_id = row.get("event_id") or row.get("id")
            stamp = _parse(row.get("knowledge_cutoff"))
            if isinstance(event_id, str) and stamp is not None:
                _cache[event_id] = stamp
                found += 1
    return found


def knowledge_cutoff(
    event: dict, *, config: Config | None = None, allow_fallback: bool = False
) -> tuple[datetime | None, str]:
    """``(cutoff, path)`` for one event. ``path`` is how it was obtained.

    ``cache``        already known, no request made
    ``api``          fetched from ``GET /v1/events``
    ``unknown``      lookup failed — **fetch nothing external for this event**
    ``conservative`` only with ``allow_fallback=True``: five days before the
                     announcement day, which clears the widest lag observed
                     (96h) but is a guess, and is logged as one

    ``allow_fallback`` defaults to **False** deliberately. There is no safe
    derivation — see :data:`FALLBACK_SPAN`. Skipping an external feature costs
    a little signal on one event; a cutoff that runs late costs prize
    eligibility, retroactively, and no offline check would ever surface it.
    Degrade instead of guessing.
    """
    event_id = event.get("event_id") or event.get("id")

    with _lock:
        if isinstance(event_id, str) and event_id in _cache:
            return _cache[event_id], "cache"

    try:
        refresh(config=config)
        with _lock:
            if isinstance(event_id, str) and event_id in _cache:
                return _cache[event_id], "api"
        print(f"[CUTOFF] {event_id} not present in /events response")
    except Exception as exc:  # network, auth, shape — all degrade the same way
        print(f"[CUTOFF] /events lookup failed for {event_id}: {type(exc).__name__}: {exc}")

    announced = _parse(event.get("event_datetime"))
    if not allow_fallback or announced is None:
        print(
            f"[CUTOFF] {event_id}: no authoritative cutoff — external features must "
            f"be skipped for this event (predicting from the facts alone is fine)"
        )
        return None, "unknown"

    # Midnight UTC on the announcement day, minus five days. Clears the widest
    # lag in the calendar (96h) with a day to spare. Still a guess: it is not
    # derived from anything the rules define, only from what has been observed.
    fallback = announced.replace(hour=0, minute=0, second=0, microsecond=0) - FALLBACK_SPAN
    print(
        f"[CUTOFF] {event_id}: falling back to {fallback.isoformat()} "
        f"({FALLBACK_SPAN.days}d before the announcement day). This is a guess, not the "
        f"rules' cutoff — prefer skipping the feature."
    )
    return fallback, "conservative"


def cached() -> dict[str, datetime]:
    """Everything looked up so far — for logging alongside a retrieval record."""
    with _lock:
        return dict(_cache)
