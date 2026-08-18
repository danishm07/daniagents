"""Analyst estimates — Agent D. Retrieval, feature construction, scoring.

## The survey result, stated first

The prior survey in this project concluded that no free point-in-time analyst
source exists, and that everything free serves a *latest restated snapshot*.
That is **correct for consensus EPS estimates** and **wrong for analyst
actions**. The distinction is the whole finding:

* **Restated snapshot, impossible to backtest.** yfinance `earnings_estimate`,
  `revenue_estimate`, `eps_trend`, `eps_revisions`, `growth_estimates`,
  `recommendations`, `analyst_price_targets` are all served *as of the request*.
  `eps_trend` carries `7daysAgo/30daysAgo/60daysAgo/90daysAgo` columns and
  `eps_revisions` carries `upLast7days/upLast30days`, which look like history but
  are windows anchored on **today**, and the `period` index (`0q/+1q/0y/+1y`) is
  relative to **today's** fiscal calendar. For an event in 2026-05 fetched in
  2026-08 they do not describe a different vintage of the same quantity — they
  describe a different quantity. Not contaminated so much as *unrelated*. There
  is no query parameter that moves the anchor.

* **A dated event log, genuinely backtestable.** `Ticker.upgrades_downgrades`
  (Yahoo `quoteSummary` module `upgradeDowngradeHistory`) returns one row per
  analyst action with an **epoch-second `GradeDate`**, `Firm`, `FromGrade`,
  `ToGrade`, `Action`, and — this is the useful part — `priorPriceTarget` and
  `currentPriceTarget`. History runs to 2012 on large caps. Each row is a
  published, timestamped event, so filtering `GradeDate < knowledge_cutoff`
  reconstructs what was on the tape at the cutoff without any as-of query.

  yfinance builds the index as ``pd.to_datetime(epochGradeDate, unit='s')`` —
  naive **UTC**. Verified against the source at
  `yfinance/scrapers/quote.py:577`. So the cutoff comparison is unambiguous.

  Its residual point-in-time risk is *not* restatement of a row's date; it is
  **completeness**: we cannot prove Yahoo's log contained row X on date X
  without a contemporaneous snapshot. Every feature here is therefore flagged
  ``point_in_time=False`` in the audit and argued, not asserted, in the report.

One HTTP request per ticker returns the whole history, so a full pass over the
2,518 distinct tickers is ~2,500 requests, not 8,020. Free, no key.

## What is directional and what is not

Rule 3 of the dispatch: magnitude without direction is dead. Price-target
*dispersion*, *count of revising analysts* and *days since last action* are
magnitudes. They are computed (they make useful normalisers) and registered
``directional=False`` so `signal_columns()` excludes them.

    uv run python sources_estimates.py fetch    # ~2,500 tickers, cached
    uv run python sources_estimates.py build    # -> data/blocks/estimates.parquet
    uv run python sources_estimates.py score
    uv run python sources_estimates.py snapshot # the going-forward nightly job
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
import sources  # noqa: E402
from runner import blocks as B  # noqa: E402

RAW_DIR = ROOT / "data" / "estimates"
GRADES_PATH = RAW_DIR / "upgrades_downgrades.parquet"
SNAPSHOT_DIR = RAW_DIR / "snapshots"

# --------------------------------------------------------------------------
# Grade vocabulary
# --------------------------------------------------------------------------

#: Yahoo's ToGrade/FromGrade strings, mapped to a 1..5 ladder (5 = most
#: bullish). Brokers use ~40 house labels for the same five rungs. Anything
#: unmapped is NaN, never a midpoint — a guessed rung is a fabricated revision.
GRADE_LADDER: dict[str, float] = {
    # 5 — top of the house scale
    "strong buy": 5.0, "conviction buy": 5.0, "top pick": 5.0, "focus list": 5.0,
    "action list buy": 5.0, "best idea": 5.0, "positive": 5.0,
    # 4 — buy / outperform
    "buy": 4.0, "outperform": 4.0, "overweight": 4.0, "outperformer": 4.0,
    "accumulate": 4.0, "add": 4.0, "market outperform": 4.0, "sector outperform": 4.0,
    "long-term buy": 4.0, "speculative buy": 4.0, "buy (spec)": 4.0,
    "outperform-speculative": 4.0, "trading buy": 4.0, "moderate buy": 4.0,
    "above average": 4.0, "attractive": 4.0,
    # 3 — hold / neutral
    "hold": 3.0, "neutral": 3.0, "market perform": 3.0, "sector perform": 3.0,
    "equal-weight": 3.0, "equal weight": 3.0, "in-line": 3.0, "in line": 3.0,
    "peer perform": 3.0, "perform": 3.0, "sector weight": 3.0, "average": 3.0,
    "market weight": 3.0, "mixed": 3.0, "fair value": 3.0, "equalweight": 3.0,
    # 2 — underperform
    "underperform": 2.0, "underweight": 2.0, "reduce": 2.0, "market underperform": 2.0,
    "sector underperform": 2.0, "below average": 2.0, "weak hold": 2.0,
    "underperformer": 2.0, "unattractive": 2.0,
    # 1 — sell
    "sell": 1.0, "strong sell": 1.0, "conviction sell": 1.0, "negative": 1.0,
}

#: `Action` values Yahoo emits. up/down are the explicit rung changes; `main`
#: and `reit` are reiterations (the price-target move is the information);
#: `init` is initiation of coverage.
ACTION_SIGN = {"up": 1.0, "down": -1.0}


def grade_value(text) -> float:
    if not isinstance(text, str):
        return np.nan
    return GRADE_LADDER.get(text.strip().lower(), np.nan)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def tickers() -> list[str]:
    frame = pd.concat([harness.load(q) for q in harness.QUARTERS], ignore_index=True)
    return sorted(frame.identifier_value.dropna().unique())


def fetch_grades(limit: int | None = None, pause: float = 0.0) -> pd.DataFrame:
    """One `upgradeDowngradeHistory` pull per ticker, cached to parquet.

    Recorded through `sources.fetch` with ``point_in_time=False``: the rows are
    individually dated and the per-event filter below is what enforces the
    cutoff, but we cannot prove the *set* of rows is the set that existed at the
    cutoff, and the audit log should say so rather than imply a guarantee.
    """
    import yfinance as yf

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    have: set[str] = set()
    frames: list[pd.DataFrame] = []
    if GRADES_PATH.exists():
        cached = pd.read_parquet(GRADES_PATH)
        frames.append(cached)
        have = set(cached.ticker.unique())

    todo = [t for t in tickers() if t not in have]
    if limit:
        todo = todo[:limit]
    print(f"{len(have)} tickers cached, fetching {len(todo)}")

    def _one(tk: str):
        """Yahoo rate-limits an unauthenticated caller at a few hundred requests.
        Back off and retry — a partial universe is a coverage artefact, not a
        finding, and it would silently bias the sample toward alphabetical A–M."""
        for attempt in range(6):
            try:
                return sources.fetch(
                    source="yfinance",
                    endpoint=f"quoteSummary/upgradeDowngradeHistory/{tk}",
                    loader=lambda: yf.Ticker(tk).upgrades_downgrades,
                    point_in_time=False,
                    notes=(
                        "dated analyst-action log (epoch GradeDate, UTC); rows are "
                        "filtered to GradeDate < knowledge_cutoff per event. Flagged "
                        "not-PIT because log completeness at the cutoff is unverifiable."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                if "Rate limit" not in str(exc) and "Too Many" not in str(exc):
                    return None  # 404 on a delisted name is a real coverage fact
                time.sleep(min(120.0, 15.0 * 2**attempt))
        return None

    batch: list[pd.DataFrame] = []
    for i, tk in enumerate(todo, 1):
        table = _one(tk)
        if table is not None and len(table):
            table = table.reset_index().rename(columns={"index": "GradeDate"})
            table["ticker"] = tk
            batch.append(table)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}  ({sum(len(b) for b in batch)} new rows)", flush=True)
            # Checkpoint: a 2,500-request pass that dies at 2,400 should not
            # throw away 2,400 requests' worth of retrievals.
            pd.concat(frames + batch, ignore_index=True).to_parquet(GRADES_PATH, index=False)
        if pause:
            time.sleep(pause)

    frames.extend(batch)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["GradeDate"] = pd.to_datetime(out["GradeDate"])  # naive UTC
        out = out.drop_duplicates(subset=["ticker", "GradeDate", "Firm", "ToGrade"])
        out.to_parquet(GRADES_PATH, index=False)
    print(f"{len(out)} rows, {out.ticker.nunique() if len(out) else 0} tickers -> {GRADES_PATH}")
    return out


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

WINDOWS = (7, 14, 21, 30, 90, 180)


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.copy()
    g["GradeDate"] = pd.to_datetime(g["GradeDate"], utc=True)
    g["to_v"] = g.ToGrade.map(grade_value)
    g["from_v"] = g.FromGrade.map(grade_value)
    g["rung_change"] = g.to_v - g.from_v
    g["act_sign"] = g.Action.str.strip().str.lower().map(ACTION_SIGN)
    for col in ("currentPriceTarget", "priorPriceTarget"):
        g[col] = pd.to_numeric(g.get(col), errors="coerce").replace(0.0, np.nan)
    # A price-target revision only exists where *both* legs are present.
    g["pt_rev"] = g.currentPriceTarget / g.priorPriceTarget - 1.0
    g.loc[~np.isfinite(g.pt_rev), "pt_rev"] = np.nan
    g["pt_rev"] = g.pt_rev.clip(-0.75, 0.75)  # a 10x "target" is a data error
    # Yahoo labels the target move directly. Covers actions where priorPriceTarget
    # is absent, so it reaches further than `pt_rev` for the same information.
    g["pt_act_sign"] = (
        g.priceTargetAction.str.strip().str.lower().map({"raises": 1.0, "lowers": -1.0})
    )
    return g.sort_values(["ticker", "GradeDate"])


def build(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per event, features from actions strictly before `knowledge_cutoff`."""
    if raw is None:
        raw = pd.read_parquet(GRADES_PATH)
    g = _prepare(raw)
    by_ticker = {tk: sub for tk, sub in g.groupby("ticker")}

    events = pd.concat([harness.load(q) for q in harness.QUARTERS], ignore_index=True)
    rows = []
    for ev in events.itertuples():
        cutoff = pd.Timestamp(ev.knowledge_cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        row: dict[str, object] = {"event_id": ev.event_id}
        sub = by_ticker.get(ev.identifier_value)
        if sub is not None and len(sub):
            past = sub[sub.GradeDate < cutoff]
            if len(past):
                # Enforcement, not decoration: the last row used must predate
                # the cutoff or sources.check_window raises.
                sources.check_window(cutoff, past.GradeDate.max(), event_id=ev.event_id)
                row.update(_features(past, cutoff))
        rows.append(row)

    frame = pd.DataFrame(rows)
    for col in frame.columns:
        if col != "event_id":
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _features(past: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {}
    age_days = (cutoff - past.GradeDate).dt.total_seconds() / 86400.0

    # ---- magnitude / normaliser (directional=False) ----
    out["days_since_action"] = float(age_days.min())

    for w in WINDOWS:
        win = past[age_days <= w]
        n = len(win)
        out[f"n_actions_{w}d"] = float(n)
        out[f"n_firms_{w}d"] = float(win.Firm.nunique()) if n else 0.0
        if not n:
            continue

        # ---- directional: rating flow ----
        signs = win.act_sign.dropna()
        ups = float((signs > 0).sum())
        downs = float((signs < 0).sum())
        if ups + downs > 0:
            out[f"rating_net_{w}d"] = (ups - downs) / (ups + downs)
        out[f"rating_flow_{w}d"] = (ups - downs) / n

        rung = win.rung_change.dropna()
        if len(rung):
            out[f"rung_change_{w}d"] = float(rung.mean())

        # ---- directional: price-target action tape ----
        pa = win.pt_act_sign.dropna()
        if len(pa):
            out[f"pt_act_net_{w}d"] = float(pa.mean())
            out[f"pt_act_flow_{w}d"] = float(pa.sum()) / n

        # ---- directional: price-target revision ----
        pt = win.pt_rev.dropna()
        if len(pt):
            out[f"pt_rev_mean_{w}d"] = float(pt.mean())
            out[f"pt_rev_net_{w}d"] = float(np.sign(pt).mean())
            out[f"pt_rev_sum_{w}d"] = float(pt.sum())
        # ---- magnitude: dispersion of live targets in the window ----
        tgt = win.currentPriceTarget.dropna()
        if len(tgt) >= 3 and tgt.mean() > 0:
            out[f"pt_dispersion_{w}d"] = float(tgt.std() / tgt.mean())

    # ---- directional: consensus rating level, latest per firm ----
    latest = past.dropna(subset=["to_v"]).groupby("Firm").tail(1)
    recent = latest[(cutoff - latest.GradeDate).dt.days <= 365]
    if len(recent) >= 2:
        out["rating_level"] = float(recent.to_v.mean())
        out["n_firms_1y"] = float(len(recent))

    # ---- directional: consensus target drift, firm-matched ----
    # Mean live target now vs the mean of the same firms' *prior* targets. This
    # is a consensus-level revision, not a per-action one, so it survives a
    # single loud analyst.
    live = past.dropna(subset=["currentPriceTarget"]).groupby("Firm").tail(1)
    live = live[(cutoff - live.GradeDate).dt.days <= 180]
    if len(live) >= 3:
        cur, pri = live.currentPriceTarget, live.priorPriceTarget
        both = cur.notna() & pri.notna()
        if both.sum() >= 3 and pri[both].mean() > 0:
            out["pt_consensus_drift"] = float(cur[both].mean() / pri[both].mean() - 1.0)
    return out


# --------------------------------------------------------------------------
# Block registration
# --------------------------------------------------------------------------

_DIRECTIONAL = [
    ("rating_net_{w}d", "(#up - #down)/(#up + #down) among rung changes in the window"),
    ("rating_flow_{w}d", "(#up - #down) / all actions in the window"),
    ("rung_change_{w}d", "mean ToGrade - FromGrade on the 1..5 house ladder"),
    ("pt_rev_mean_{w}d", "mean current/prior price-target return per action"),
    ("pt_rev_net_{w}d", "mean sign of the price-target revisions"),
    ("pt_rev_sum_{w}d", "summed price-target revision — flow, not average"),
    ("pt_act_net_{w}d", "mean of Yahoo's Raises/Lowers label, +1/-1"),
    ("pt_act_flow_{w}d", "(#Raises - #Lowers) / all actions in the window"),
]
_MAGNITUDE = [
    ("n_actions_{w}d", "count of analyst actions"),
    ("n_firms_{w}d", "distinct broking firms acting"),
    ("pt_dispersion_{w}d", "cross-firm sd/mean of live price targets"),
]

_features_list = [
    B.Feature("days_since_action", point_in_time=False, directional=False,
              description="days from the last analyst action to the cutoff"),
    B.Feature("rating_level", point_in_time=False, directional=True,
              description="mean latest rung across firms acting in the last year"),
    B.Feature("n_firms_1y", point_in_time=False, directional=False,
              description="firms with an action in the last year"),
    B.Feature("pt_consensus_drift", point_in_time=False, directional=True,
              description="firm-matched mean current target / mean prior target - 1"),
]
for _w in WINDOWS:
    for _n, _d in _DIRECTIONAL:
        _features_list.append(B.Feature(_n.format(w=_w), point_in_time=False,
                                        directional=True, description=_d))
    for _n, _d in _MAGNITUDE:
        _features_list.append(B.Feature(_n.format(w=_w), point_in_time=False,
                                        directional=False, description=_d))

BLOCK = B.register(
    B.Block(
        name="estimates",
        owner="Agent D — analyst estimates",
        features=_features_list,
        notes=(
            "Source: Yahoo quoteSummary/upgradeDowngradeHistory via yfinance. Free, "
            "no key, ~2,500 requests for a full pass. Every row carries a UTC epoch "
            "GradeDate and is filtered strictly below knowledge_cutoff. Flagged "
            "point_in_time=False throughout: row dates are trustworthy, log "
            "completeness at the cutoff is not verifiable without a contemporaneous "
            "snapshot. Consensus EPS estimates (eps_trend / eps_revisions / "
            "earnings_estimate) are today-anchored snapshots and are NOT in this "
            "block — see snapshot() for the going-forward job."
        ),
    )
)


# --------------------------------------------------------------------------
# The going-forward nightly snapshot
# --------------------------------------------------------------------------

SNAPSHOT_ENDPOINTS = (
    "earnings_estimate", "revenue_estimate", "eps_trend", "eps_revisions",
    "growth_estimates", "recommendations",
)


def snapshot(symbols: list[str] | None = None) -> Path:
    """Record today's consensus so that *tomorrow* it is point-in-time history.

    The only fix for the today-anchored endpoints. Nothing it writes can be
    backtested; everything it writes is clean for a live prediction made after
    the write. History starts the day this first runs.
    """
    import yfinance as yf

    symbols = symbols or tickers()
    stamp = pd.Timestamp.utcnow()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tk in symbols:
        t = yf.Ticker(tk)
        for endpoint in SNAPSHOT_ENDPOINTS:
            try:
                table = sources.fetch(
                    source="yfinance", endpoint=f"{endpoint}/{tk}",
                    loader=lambda t=t, e=endpoint: getattr(t, e),
                    point_in_time=False,
                    notes="today-anchored consensus snapshot; PIT only for events after observed_at",
                )
            except Exception:  # noqa: BLE001
                continue
            if table is None or not len(table):
                continue
            table = table.reset_index()
            table.insert(0, "endpoint", endpoint)
            table.insert(0, "ticker", tk)
            table.insert(0, "observed_at", stamp)
            rows.append(table.astype({c: str for c in table.columns if table[c].dtype == object}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    path = SNAPSHOT_DIR / f"consensus_{stamp:%Y%m%d}.parquet"
    out.to_parquet(path, index=False)
    print(f"{len(out)} rows, {len(symbols)} tickers -> {path}")
    return path


# --------------------------------------------------------------------------


def score(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if frame is None:
        frame = BLOCK.load()
    table = B.score_features(frame, columns=None)
    signal = set(BLOCK.signal_columns())
    table["directional"] = table.feature.isin(signal)
    return table


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "fetch":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        fetch_grades(limit=n)
    elif cmd == "build":
        frame = build()
        B.write("estimates", frame)
        print(frame.describe().T.to_string())
    elif cmd == "snapshot":
        snapshot()
    else:
        print(score().to_string(index=False))
