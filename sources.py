"""Cutoff-aware external data access, with an audit log fit to hand to the organisers.

Rules §04: no information available after an event's ``knowledge_cutoff`` may be
used — directly or indirectly — in data collection, model inputs, features,
prompts, retrieval, external search, model selection, or prediction revision.
Post-announcement market data is named explicitly. Rules §07 and §10: the
University may require logs, timestamps, data-source descriptions, API records
and retrieval records to verify compliance, and may audit. **A violation voids
prize eligibility even after final results are published.**

So external data does not get fetched ad hoc. Everything goes through
:func:`fetch`, which

  * refuses a window that extends past the event's cutoff — a
    :class:`CutoffViolation`, not a warning, because a silent one-day overhang
    is indistinguishable from a good result;
  * records source, endpoint, request time, the cutoff it was checked against,
    the window returned, and the row count, to ``data/audit/fetches.jsonl``;
  * marks anything that is not point-in-time as ``live_safe: false``.

That last one is the trap worth naming. yfinance returns *current* values for
market cap, shares outstanding, sector and short interest — today's sector for a
company that may have been reclassified, today's share count after a buyback.
Those are usable in research as a flagged approximation and are **not**
shippable. Anything derived from them must be rebuilt point-in-time before it
goes near a live prediction.

    from sources import fetch, audit_report

    prices = fetch(
        source="yfinance", endpoint=f"history/{ticker}",
        event_id=event["event_id"], knowledge_cutoff=event["knowledge_cutoff"],
        window=(start, end), loader=lambda: yf.Ticker(ticker).history(...),
    )
    audit_report()        # what the organisers would be handed
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

AUDIT_LOG = Path(__file__).parent / "data" / "audit" / "fetches.jsonl"

_lock = threading.Lock()


class CutoffViolation(RuntimeError):
    """A fetch asked for data at or after the event's knowledge cutoff."""


@dataclass(frozen=True)
class FetchRecord:
    """One retrieval, in the form an auditor would want to read it."""

    requested_at: str
    source: str
    endpoint: str
    event_id: str | None
    knowledge_cutoff: str | None
    window_start: str | None
    window_end: str | None
    n_rows: int | None
    live_safe: bool
    point_in_time: bool
    notes: str


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.isoformat()


def check_window(
    knowledge_cutoff: Any, window_end: Any, *, event_id: str | None = None
) -> None:
    """Raise unless ``window_end`` is strictly before the cutoff.

    Strictly: a window ending exactly at the cutoff instant can include the
    announcement itself, and the announcement is the thing being predicted.
    """
    if knowledge_cutoff is None or window_end is None:
        return
    cutoff = pd.Timestamp(knowledge_cutoff)
    end = pd.Timestamp(window_end)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff
    end = end.tz_localize("UTC") if end.tzinfo is None else end
    if end >= cutoff:
        raise CutoffViolation(
            f"{event_id or 'event'}: window ends {end.isoformat()}, "
            f"at or after knowledge_cutoff {cutoff.isoformat()} — Rules §04"
        )


def fetch(
    *,
    source: str,
    endpoint: str,
    loader: Callable[[], Any],
    event_id: str | None = None,
    knowledge_cutoff: Any = None,
    window: tuple[Any, Any] | None = None,
    point_in_time: bool = True,
    notes: str = "",
) -> Any:
    """Run ``loader``, enforce the cutoff, and record the retrieval.

    ``point_in_time=False`` is how you declare that the source returns current
    values rather than values as of the cutoff. It does not block the call — it
    marks the record ``live_safe: false`` so the audit report can list exactly
    what would have to be rebuilt before shipping.
    """
    window_start, window_end = window if window else (None, None)
    check_window(knowledge_cutoff, window_end, event_id=event_id)

    data = loader()
    try:
        n_rows = len(data)
    except TypeError:
        n_rows = None

    record = FetchRecord(
        requested_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        endpoint=endpoint,
        event_id=event_id,
        knowledge_cutoff=_iso(knowledge_cutoff),
        window_start=_iso(window_start),
        window_end=_iso(window_end),
        n_rows=n_rows,
        live_safe=point_in_time,
        point_in_time=point_in_time,
        notes=notes,
    )
    _append(record)
    return data


def _append(record: FetchRecord) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, AUDIT_LOG.open("a") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")


def audit_log() -> pd.DataFrame:
    if not AUDIT_LOG.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in AUDIT_LOG.open() if l.strip()])


def audit_report() -> pd.DataFrame:
    """Per source: how many retrievals, and how many are not shippable.

    Print this before any deployability audit. A source with
    ``not_live_safe > 0`` has features that cannot ship as measured.
    """
    frame = audit_log()
    if frame.empty:
        print("no external fetches recorded")
        return frame
    summary = (
        frame.groupby("source")
        .agg(
            fetches=("endpoint", "size"),
            events=("event_id", "nunique"),
            not_live_safe=("live_safe", lambda s: int((~s.astype(bool)).sum())),
            first=("requested_at", "min"),
            last=("requested_at", "max"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    unsafe = frame[~frame.live_safe.astype(bool)]
    if len(unsafe):
        print(
            f"\n!! {len(unsafe)} retrievals are not point-in-time and cannot ship as measured:"
        )
        print(unsafe.groupby(["source", "endpoint"]).size().to_string())
    return summary


if __name__ == "__main__":
    # Real checks against the archive's own cutoffs — no invented timestamps.
    # The audit log is redirected to a scratch file first: it is meant to be
    # handable to the organisers as a record of actual research retrievals, and
    # a smoke test's rows are not that.
    import tempfile

    import harness

    AUDIT_LOG = Path(tempfile.mkdtemp()) / "selftest_fetches.jsonl"

    event = harness.events_for("2026Q2")[0]
    cutoff = event["knowledge_cutoff"]
    print(f"event {event['event_id']}  cutoff {cutoff}")

    ok = fetch(
        source="selftest",
        endpoint="window-before-cutoff",
        loader=lambda: [1, 2, 3],
        event_id=event["event_id"],
        knowledge_cutoff=cutoff,
        window=(cutoff - pd.Timedelta(days=30), cutoff - pd.Timedelta(seconds=1)),
        notes="self-test",
    )
    print("fetch ending before the cutoff:", ok)

    for label, end in [
        ("exactly at the cutoff", cutoff),
        ("one second after", cutoff + pd.Timedelta(seconds=1)),
        ("one day after", cutoff + pd.Timedelta(days=1)),
    ]:
        try:
            fetch(
                source="selftest",
                endpoint="window-past-cutoff",
                loader=lambda: [1],
                event_id=event["event_id"],
                knowledge_cutoff=cutoff,
                window=(cutoff - pd.Timedelta(days=30), end),
            )
            raise AssertionError(f"{label}: should have raised CutoffViolation")
        except CutoffViolation as exc:
            print(f"blocked ({label}): {exc}")

    fetch(
        source="selftest",
        endpoint="current-value-field",
        loader=lambda: {"sector": "Technology"},
        event_id=event["event_id"],
        knowledge_cutoff=cutoff,
        point_in_time=False,
        notes="self-test: stands in for a yfinance current-value field",
    )
    print()
    audit_report()
