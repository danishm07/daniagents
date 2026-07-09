"""Exact port of the Explaining Markets competition scoring transform.

Scoring happens within a **period** (a calendar quarter). For each realized
``(event, focal asset)`` outcome the competition converts the one-day cumulative
abnormal return (``car1``) into a percentile rank ``y`` in ``[0, 1]`` across the
period's cross-section, and likewise percentile-ranks the earnings surprise. A
submission's period fit is the two-regressor OLS

    y = alpha + beta1 * predicted_percentile + beta2 * surprise_percentile

so its R^2 measures how much of the realized abnormal-return *ranking* the
submission's predicted percentiles explain, over and above the naive
earnings-surprise benchmark.

:func:`percentile_ranks` is a verbatim port of the production scorer, tie handling and
all, so offline numbers match what the leaderboard computes. The rest assembles
the per-``(event, asset)`` analysis frame from archive records (as produced by
:func:`examples.archive.load_archive` / :func:`examples.archive.read_jsonl_gz`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import pandas as pd

# The reference LLM baselines woven into each archive line, mapped to display
# names. Keys match the archive's ``baseline_predictions`` blocks.
BASELINE_LABELS: dict[str, str] = {
    "gemini/ea-explain-contemp-summary": "Gemini 2.5 Flash-Lite",
    "openai/ea-explain-contemp-summary": "GPT-5 nano",
}


def percentile_ranks(values: Sequence[float]) -> list[float]:
    """Map each value to its percentile rank in ``[0, 1]`` (min -> 0, max -> 1).

    Ties share the average rank. With a single value the rank is ``0.5`` (no
    spread to rank against); an empty input returns ``[]``. This is a verbatim
    port of the competition scorer, so ranks match the leaderboard exactly.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    denom = n - 1
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (0-indexed) tie; average position is (i + j) / 2.
        rank = ((i + j) / 2.0) / denom
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def event_asset_rows(record: dict) -> Iterator[dict]:
    """Yield one analysis row per focal asset of an archive event that has a ``car1``.

    Mirrors the production scorer's outcome construction: the event's earnings
    surprise is attached to each focal asset when its ``surprise_status == "ok"``
    (else ``None``), and each reference baseline's predicted percentile is pulled
    in under its display label. Assets whose return leg carries no ``car1`` are
    skipped — they cannot be scored.
    """
    es = (record.get("metrics") or {}).get("earnings_surprise") or {}
    surprise = es.get("surprise") if es.get("surprise_status") == "ok" else None
    returns = record.get("event_returns") or {}
    baselines = record.get("baseline_predictions") or {}
    for asset in record.get("focal_assets") or []:
        ticker = asset.get("identifier_value")
        leg = returns.get(ticker) or {}
        car1 = leg.get("car1")
        if car1 is None:
            continue
        row: dict = {
            "event_id": record.get("event_id"),
            "identifier_value": ticker,
            "car1": float(car1),
            "surprise": float(surprise) if surprise is not None else None,
        }
        for key, label in BASELINE_LABELS.items():
            preds = baselines.get(key) or {}
            if ticker in preds:
                row[label] = float(preds[ticker])
        yield row


def outcomes_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Assemble the per-``(event, asset)`` analysis frame from archive records.

    Pass the records for **one period** (quarter): percentiles are later ranked
    within whatever you pass, matching the competition's per-period scoring. Each
    row carries ``car1``, ``surprise`` (may be ``NaN``), and one column per
    reference baseline (its predicted percentile).
    """
    rows = [row for record in records for row in event_asset_rows(record)]
    return pd.DataFrame(rows).reset_index(drop=True)


def add_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the competition's scoring targets to an :func:`outcomes_frame`.

    Adds two columns:

    - ``y`` — percentile rank of ``car1`` across the whole frame. This is the
      dependent variable: the realized abnormal-return percentile (``r_abn``).
    - ``surprise_pct`` — percentile rank of ``surprise``, ranked over only the
      rows that carry a numeric surprise (others left ``NaN``), exactly as the
      scorer treats the surprise regressor.
    """
    out = frame.copy()
    out["y"] = percentile_ranks(out["car1"].tolist())

    surprise = out["surprise"].tolist()
    have = [i for i, v in enumerate(surprise) if pd.notna(v)]
    ranked = percentile_ranks([surprise[i] for i in have])
    col = [float("nan")] * len(out)
    for i, r in zip(have, ranked, strict=True):
        col[i] = r
    out["surprise_pct"] = col
    return out
