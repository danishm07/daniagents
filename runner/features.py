"""A shared per-event feature cache, so two arms using a feature compute it once.

Keyed on ``(event_id, feature_name)`` and persisted as one JSONL per feature.
The cache is not the interesting part — the **declarations** are.

Every feature states, as data:

``live_seconds``   how long computing it costs inside the 5-minute window
``live_fetches``   how many external requests that takes
``cutoff_safe``    whether everything it reads predates the event's
                   ``knowledge_cutoff``

:meth:`runner.registry.Arm.live_check` sums these across an arm's features and
refuses arms that cannot ship. This is requirement 2 from ``BUILD_LOOP.md``, and
it is a declaration rather than a measurement on purpose: the point is to fail
*before* paying for an evaluation, not after.

On cutoff safety, the measured constraint that kills the obvious shortcut:

    Across 840 events ``knowledge_cutoff`` is always 20:00Z, always *earlier*
    than ``event_datetime``, median 17h, max 96h. **No function of
    ``event_datetime`` is safe** — a "conservative" bound of announcement−1h
    would have violated §04 on 85% of events.

So a feature that needs the cutoff takes it from the event dict (archive) or
``GET /v1/events`` (live), and anything reaching outside the archive goes
through ``sources.py`` with its audit log. A feature that cannot honour that
declares ``cutoff_safe=False`` and is refused rather than quietly measured.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "features"


@dataclass(frozen=True)
class FeatureSpec:
    """One per-event quantity, with its deployability declared."""

    name: str
    #: ``(events, quarter) -> one JSON-serialisable value per event, in order``.
    #: Takes a batch because most features are cheaper vectorised, and takes the
    #: quarter because anything fitted must ask ``harness.training_data`` for
    #: strictly-prior quarters rather than seeing its own.
    fn: Callable[[Sequence[dict], str], Sequence[Any]]
    live_seconds: float = 0.0
    live_fetches: int = 0
    cutoff_safe: bool = True
    description: str = ""


SPECS: dict[str, FeatureSpec] = {}


def register(
    name: str,
    *,
    live_seconds: float = 0.0,
    live_fetches: int = 0,
    cutoff_safe: bool = True,
    description: str = "",
) -> Callable:
    def wrap(fn):
        if name in SPECS:
            raise ValueError(f"feature {name!r} already registered")
        SPECS[name] = FeatureSpec(
            name=name,
            fn=fn,
            live_seconds=live_seconds,
            live_fetches=live_fetches,
            cutoff_safe=cutoff_safe,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
        )
        return fn

    return wrap


def _path(name: str) -> Path:
    return CACHE / f"{name}.jsonl"


def _load(name: str) -> dict[str, Any]:
    path = _path(name)
    if not path.exists():
        return {}
    out = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            out[row["event_id"]] = row["value"]
    return out


def _append(name: str, rows: list[tuple[str, Any]]) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for event_id, value in rows:
            fh.write(json.dumps({"event_id": event_id, "value": value}) + "\n")


def get(name: str, events: Sequence[dict], quarter: str) -> list[Any]:
    """Values for ``events``, in order, computing and caching only what is missing."""
    spec = SPECS.get(name)
    if spec is None:
        raise KeyError(f"no feature {name!r}; registered: {sorted(SPECS)}")
    cached = _load(name)
    missing = [e for e in events if e["event_id"] not in cached]
    if missing:
        values = list(spec.fn(missing, quarter))
        if len(values) != len(missing):
            raise ValueError(
                f"feature {name!r} returned {len(values)} values for {len(missing)} events"
            )
        fresh = [(e["event_id"], v) for e, v in zip(missing, values, strict=True)]
        _append(name, fresh)
        cached.update(dict(fresh))
    return [cached[e["event_id"]] for e in events]


def clear(name: str) -> None:
    """Drop a feature's cache. Needed when its definition changes — a stale
    cache under a changed definition is the same class of error as a stale
    champion column, and that one has already cost this project four times."""
    _path(name).unlink(missing_ok=True)


def table() -> list[dict]:
    return [
        {
            "feature": s.name,
            "live_seconds": s.live_seconds,
            "live_fetches": s.live_fetches,
            "cutoff_safe": s.cutoff_safe,
            "cached": len(_load(s.name)),
            "description": s.description,
        }
        for s in SPECS.values()
    ]


# --------------------------------------------------------------------------
# Features that need nothing but the event itself
# --------------------------------------------------------------------------


@register(
    "n_facts",
    description="How many facts the extractor produced. Free, live, and a control.",
)
def _n_facts(events, quarter):
    return [len(e["facts"]) for e in events]


@register(
    "facts_chars",
    description="Total characters across the facts — length as a crude effort proxy.",
)
def _facts_chars(events, quarter):
    return [sum(len(f) for f in e["facts"]) for e in events]
