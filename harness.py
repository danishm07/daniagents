"""Offline ΔR² harness for Explaining Markets.

The minimum thing that lets you ask "is this idea better than the baseline?" and
get a trustworthy answer. Everything here is thin glue over
``examples.scoring``, which is the competition's own scorer — no metric is
reimplemented.

Three ideas hold the design together:

1. **Score per quarter, never pooled.** The scorer ranks within a period, and
   the quarter-to-quarter swing is large (the unchanged Gemini baseline moves
   0.048 → 0.076). A pooled number hides exactly the variation you need to see.

2. **Inputs and outcomes are separated by construction.** :func:`events_for`
   hands your model only what a live webhook would carry — facts, ticker,
   timing. Outcomes live behind :func:`training_data`, which you must ask for
   explicitly and which refuses to hand you the quarter you are testing on.
   Leakage becomes something you have to opt into rather than something you can
   do by accident.

3. **One fact adapter.** Archive records and live webhook payloads carry the
   ten facts at different paths. :func:`facts_from_record` and
   :func:`facts_from_summary` normalise both, so what you backtest is what
   production sees.

Typical use::

    import harness

    def my_model(events):
        return [0.5 + 0.1 * ("raised" in " ".join(e["facts"])) for e in events]

    harness.backtest(my_model, "keyword toy")
    harness.compare()          # the official baselines, for reference
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from examples.scoring import add_percentiles, outcomes_frame, score_submission

ARCHIVE_DIR = Path(__file__).parent / "data" / "archive"

#: Chronological. Order matters — the temporal split slices this list.
QUARTERS = ["2025Q4", "2026Q1", "2026Q2", "2026Q3"]

GEMINI = "Gemini 2.5 Flash-Lite"
GPT = "GPT-5 nano"

#: Columns a model must never see for the quarter it is predicting.
OUTCOME_COLUMNS = ["car1", "y"]


# --------------------------------------------------------------------------
# The fact adapter — the one place the two input shapes are reconciled
# --------------------------------------------------------------------------


def facts_from_record(record: dict) -> list[str]:
    """The ten facts out of an *archive* record — or a live payload.

    Archive path is ``disclosure.items[] where kind == "facts"`` → ``content``.
    The live payload is that same object **unwrapped**: ``items`` sits at the top
    level next to ``schema_version`` / ``event_id`` / ``generated_at``, with no
    ``disclosure`` key. Both are handled here, so the two shapes converge before
    anything downstream sees them.

    Matches on ``kind`` rather than assuming a single item, because both ``kind``
    and ``source`` are open string sets upstream.
    """
    items = (record.get("disclosure") or {}).get("items") or record.get("items") or []
    for item in items:
        if item.get("kind") == "facts" and isinstance(item.get("content"), list):
            return [str(c) for c in item["content"]]
    return []


def facts_from_summary(summary: dict) -> list[str]:
    """The ten facts out of a *live* ``information_url`` payload.

    Confirmed against production on 2026-08-14 (`[SHAPE]` log line): the body is

    ``{schema_version, event_id, generated_at, items: [{kind: "facts", content: [...10]}]}``

    — one disclosure item, ``url``/``bytes``/``sha256`` all null. **Ten facts is
    the entire live input**, which is what makes the archive a faithful proxy for
    offline work.

    The documented sample's ``response.facts`` shape is still accepted first, and
    the wrapped archive shape last, so one call site handles all three. Note that
    ``agent/predict.py`` shipped reading ``summary["summary"]`` — a key in none of
    them — and silently dumped the whole JSON blob into the prompt for two days
    before it was caught.
    """
    if not isinstance(summary, dict):
        return []
    facts = (summary.get("response") or {}).get("facts")
    if isinstance(facts, list):
        return [str(f) for f in facts]
    if isinstance(summary.get("facts"), list):
        return [str(f) for f in summary["facts"]]
    return facts_from_record(summary)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load(quarter: str) -> pd.DataFrame:
    """One quarter as a scored analysis frame, cached.

    Columns: ``event_id``, ``identifier_value``, ``car1``, ``surprise``, both
    baselines' predictions, ``y`` (the target — percentile rank of ``car1``),
    ``surprise_pct``, plus ``facts`` / ``n_facts`` / ``event_datetime`` /
    ``quarter``.

    Rows without both ``y`` and ``surprise_pct`` are dropped: those are exactly
    the rows the scorer excludes from the common sample, so keeping them would
    make offline numbers disagree with the leaderboard.
    """
    path = ARCHIVE_DIR / f"EARNINGS_RELEASE_{quarter}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run python scripts/download_archive.py`"
        )
    records = [json.loads(line) for line in gzip.open(path, "rt") if line.strip()]

    frame = add_percentiles(outcomes_frame(records))
    meta = {
        r["event_id"]: (facts_from_record(r), r.get("event_datetime"))
        for r in records
    }
    frame["facts"] = [meta[e][0] for e in frame.event_id]
    frame["n_facts"] = frame.facts.map(len)
    frame["event_datetime"] = pd.to_datetime([meta[e][1] for e in frame.event_id])
    frame["quarter"] = quarter

    return frame.dropna(subset=["y", "surprise_pct"]).reset_index(drop=True)


def load_all(quarters: Sequence[str] = QUARTERS) -> pd.DataFrame:
    return pd.concat([load(q) for q in quarters], ignore_index=True)


# --------------------------------------------------------------------------
# The leakage boundary
# --------------------------------------------------------------------------


def events_for(quarter: str) -> list[dict]:
    """Model inputs for one quarter — **outcomes stripped**.

    Each dict carries only what a live webhook would: ``event_id``, ``ticker``,
    ``facts``, ``event_datetime``. Deliberately excludes ``surprise``: it is
    computed from the market's own reaction window and is the benchmark's
    regressor, not yours. If you want it as a feature, take it from
    :func:`training_data` knowing what you are doing.
    """
    f = load(quarter)
    return [
        {
            "event_id": r.event_id,
            "ticker": r.identifier_value,
            "facts": r.facts,
            "event_datetime": r.event_datetime,
        }
        for r in f.itertuples()
    ]


def training_data(before: str) -> pd.DataFrame:
    """Every quarter strictly *before* ``before``, with outcomes attached.

    This is the only sanctioned way to see ``y``. Fit here, predict on
    :func:`events_for`. Returns an empty frame for the first quarter, which is
    why backtests report it as untestable rather than scoring it.
    """
    idx = QUARTERS.index(before)
    if idx == 0:
        return load(QUARTERS[0]).iloc[0:0]
    return pd.concat([load(q) for q in QUARTERS[:idx]], ignore_index=True)


def residualize(frame: pd.DataFrame, target: str = "y") -> pd.Series:
    """``target`` with ``surprise_pct`` projected out, per quarter.

    This is what the contest actually pays for — the part of the outcome the
    surprise benchmark cannot already explain. Training against it rather than
    raw ``y`` measurably helps, and costs nothing.
    """
    out = pd.Series(index=frame.index, dtype=float)
    for _, g in frame.groupby("quarter"):
        slope, intercept = np.polyfit(g.surprise_pct, g[target], 1)
        out.loc[g.index] = g[target] - (slope * g.surprise_pct + intercept)
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def evaluate(frame: pd.DataFrame, column: str) -> dict:
    """Full score dict for a prediction column (competition scorer verbatim)."""
    return score_submission(frame.dropna(subset=[column]), column)


def delta_r2(frame: pd.DataFrame, column: str) -> float:
    return evaluate(frame, column)["delta_r_squared"]


def backtest(
    predict_fn: Callable[[list[dict]], Sequence[float]],
    name: str = "model",
    quarters: Sequence[str] = QUARTERS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score ``predict_fn`` on every quarter and print the table.

    ``predict_fn`` takes the list from :func:`events_for` and returns one float
    per event, in order. It is called once per quarter, so a model that needs to
    fit on prior quarters can do so inside — call :func:`training_data` with the
    quarter it is predicting.

    Returns a frame with one row per quarter: ``delta_r2``, ``r2_full``,
    ``r2_surprise``, ``n``, and the two baselines for reference.
    """
    rows = []
    for q in quarters:
        frame = load(q).copy()
        preds = list(predict_fn(events_for(q)))
        if len(preds) != len(frame):
            raise ValueError(
                f"{name} returned {len(preds)} predictions for {q}, expected {len(frame)}"
            )
        frame["_pred"] = preds
        s = evaluate(frame, "_pred")
        rows.append(
            {
                "quarter": q,
                "n": s["n_obs"],
                "r2_surprise": s["r_squared_surprise"],
                "r2_full": s["r_squared"],
                "delta_r2": s["delta_r_squared"],
                "vs_gemini": s["delta_r_squared"] - delta_r2(frame, GEMINI),
            }
        )
    out = pd.DataFrame(rows)
    if verbose:
        _print_table(name, out)
    return out


def compare(quarters: Sequence[str] = QUARTERS) -> pd.DataFrame:
    """The reference points: both official baselines, per quarter.

    Run this first. If these numbers don't match the ones in the strategy note,
    the harness is wrong, not the idea you're testing.
    """
    rows = []
    for q in quarters:
        f = load(q)
        row = {"quarter": q, "n": len(f)}
        for col, label in [(GPT, "gpt5_nano"), (GEMINI, "gemini_flash_lite")]:
            row[label] = delta_r2(f, col)
        row["r2_surprise"] = evaluate(f, GEMINI)["r_squared_surprise"]
        rows.append(row)
    out = pd.DataFrame(rows)
    print("\nOFFICIAL BASELINES (delta R^2 per quarter)")
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nmean gpt5_nano={out.gpt5_nano.mean():+.4f}  "
          f"mean gemini={out.gemini_flash_lite.mean():+.4f}")
    return out


def _print_table(name: str, out: pd.DataFrame) -> None:
    print(f"\n{name}")
    print("-" * 68)
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    wins = (out.vs_gemini > 0).sum()
    print(
        f"mean delta_r2 = {out.delta_r2.mean():+.4f}   "
        f"vs gemini = {out.vs_gemini.mean():+.4f}   "
        f"beats gemini on {wins}/{len(out)} quarters"
    )
    if wins == len(out):
        print("=> sign-consistent across all quarters. Worth taking seriously.")
    elif out.vs_gemini.mean() > 0.02:
        print("=> mean gain > 2 SE. Worth taking seriously.")
    else:
        print("=> NOT sign-consistent and gain < 2 SE (0.02). Treat as noise.")


if __name__ == "__main__":
    # Reproducing the published baselines is the harness's own correctness test.
    compare()

    def constant(events):
        return [0.5] * len(events)

    backtest(constant, "constant 0.5 (true null — must be exactly 0.0000)")

    def facts_length(events):
        return [len(" ".join(e["facts"])) for e in events]

    backtest(facts_length, "toy: total length of the facts")
