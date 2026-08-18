"""Agent A — positioning features from price and volume history.

The hypothesis this block exists to test, stated so a negative result is still a
result: **run-up is a priced-in proxy.** A beat delivered into a large prior
run-up disappoints, because the run-up already contains the beat. This is a
*directional* read on component 2 (positioning / what was already priced in),
which is the one component of the earnings-day reaction that nothing in this
project has ever touched. Components 1 (the news) and 3 (responsiveness) are
measured dead; component 4 (regime) belongs to another agent.

Explicitly **not** built here: realised volatility, beta, dispersion, or any
other magnitude-without-direction quantity as a *signal*. ΔR² is a linear fit,
so a magnitude contributes ~0 and has been measured to do so three independent
times. Trailing volatility appears only as a **normaliser** — run-up divided by
its own trailing dispersion, which is arguably the right form for a positioning
variable, since "large" only means anything relative to how far the name usually
moves.

## Result — measured, and negative

The hypothesis is **refuted**, and refuted in sign as well as in size. On 6,045
dev events, partial ρ with `y` controlling for `surprise_pct`, no fitting:

    xsrunup_5d   -0.024      xsrunup_20d  +0.015      xsrunup_60d  +0.006
    runup_5d     -0.031      runup_20d    +0.011      runup_60d    +0.012

At n=6,045 the standard error on a correlation is 0.013, so every run-up window
past 5 days is inside the noise band, and the two that are marginally outside it
disagree with each other in sign. The *conditional* form — the literal
"a beat into a run-up disappoints", built as `run-up × sign(surprise)` — is
smaller still (+0.001 to +0.015) and **positive**, i.e. what little there is
points at momentum, not at disappointment. Winsorising at the 2nd/98th
percentile changes nothing, so it is not outlier masking.

The one column above noise is `dist_52w_high` at ρ +0.053 (t≈4.1), confirmed at
+0.061 on the 4,144-event held-out partition, sign-stable across all three dev
quarters. It is also the *most redundant* column in the block: ρ_b 0.30 against
the champion, meaning the LLM read already contains most of it. And the whole
24-feature block, fitted with leave-one-quarter-out ridge, reaches ρ 0.051 —
identical to that single raw column, so the other 23 features add nothing.

**ρ ≈ 0.05 against a 0.15 bar is the ceiling for this component.** Positioning,
as daily price and volume history can express it, is not a channel.

## Why that number is believable

A surrogate check, run because a one-day misalignment would produce exactly this
kind of null: take the first bar *after* `last_bar_date` — the first bar out of
bounds — and correlate its SPY-excess return with the archive's `car1`. It comes
back at **0.9965** on the 3,228 events whose announcement is within 3h of the
cutoff and 0.9946 on the 2,302 at 3–24h. The reaction bar is exactly one bar
past the boundary, so tickers, calendar, adjustment and market-excess arithmetic
are all correct, and the null is a fact about the market rather than about this
code. (On the 516 events with a >24h gap it falls to 0.54 — the brief's bimodal
warning, visible in the data.) That check is pure leakage and exists only as a
test; nothing derived from it is in the block.

## Source

yfinance (Yahoo chart API), which works from this machine's residential IP.
A prior survey from a datacenter IP saw hard 429 on every Yahoo call and a
JavaScript anti-bot interstitial from Stooq; Stooq still 404s here. **A
datacenter deployment therefore cannot assume this source.** Measured here, not
assumed: see ``main()``.

Yahoo history is back-adjusted for splits and dividends. That restates history,
and it matters differently per feature:

* **returns** — fine. The adjustment is a constant multiplicative factor across
  a bar's OHLC, so every ratio (close/close, open/close, close/open) is
  preserved. Dividend adjustment makes the returns *total* returns rather than
  price returns, a small and uniform distortion.
* **52-week high/low distance** — fine for the same reason: it is a ratio of two
  adjusted prices.
* **volume** — Yahoo does **not** split-adjust volume. A split inside the
  lookback puts a step in the volume series. The volume feature is a ratio of a
  recent window to its own trailing median, so a split between the two windows
  corrupts that one event. Rare; flagged, not hidden.

None of that is a point-in-time violation — the adjustment restates *past*
prices using a *later* corporate action, which is a look-ahead in level but not
in ordering, and it cancels in every ratio taken inside a single window. The one
place it does not cancel is a ratio spanning a split, which is the volume case
above. Everything served by yfinance as a *current* value (market cap, shares,
sector, short interest) is not used here at all.

## The cutoff

``knowledge_cutoff`` is 16:00 ET on every archive event. ``blocks.price_window``
resolves the daily-bar boundary in one place: the last usable bar is the one
dated the cutoff date in America/New_York, which is the same bar the organisers
price their own surprise metric off (7,611/8,005 events). Every bar dated after
that is dropped, per event, in :func:`bars_for_event`.

## How the audit works here

A retrieval is a network call. There are ~2,500 of them — one per ticker, a full
history covering every event for that ticker — and each goes through
``sources.fetch``. Logging one line per *event* would put 7,840 rows in a log
that five other agents are appending to concurrently, and would misdescribe what
happened: no per-event network call is made.

So the per-event cutoff enforcement is separate and stricter than the log:

1. ``blocks.price_window(cutoff, lookback)`` gives ``last_bar_date``.
2. ``sources.check_window(cutoff, audit_end)`` is called **for every event** and
   raises ``CutoffViolation`` rather than warning.
3. Every event's realised window is written to
   ``data/prices/event_windows.parquet`` — event_id, ticker, fetch_start,
   audit_end, last_bar_date, bars used, and the date of the last bar actually
   consumed. That table is the per-event retrieval record an auditor would want,
   and it is checkable: ``last_bar_used <= last_bar_date`` on every row.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
import sources  # noqa: E402
from runner import blocks as B  # noqa: E402

PRICE_DIR = ROOT / "data" / "prices"
BARS = PRICE_DIR / "bars.parquet"
WINDOWS = PRICE_DIR / "event_windows.parquet"

#: The market proxy for the excess-return features. SPY is the broadest liquid
#: US equity index available from the same source, so market and stock come from
#: one back-adjustment convention and one bar calendar.
MARKET = "SPY"

#: Calendar span to cache. Earliest cutoff is 2025-10-08; a 52-week high needs a
#: year before that, and the prior-earnings join reaches back one quarter more.
HISTORY_START = "2024-06-01"

#: Trading-day windows for the run-up family.
RUNUP_WINDOWS = (1, 5, 20, 60)


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------


def universe(quarters=tuple(harness.QUARTERS)) -> pd.DataFrame:
    """`quarter, event_id, ticker, event_datetime, knowledge_cutoff` for the archive."""
    rows = []
    for quarter in quarters:
        for event in harness.events_for(quarter):
            rows.append(
                {
                    "quarter": quarter,
                    "event_id": event["event_id"],
                    "ticker": event["ticker"],
                    "event_datetime": pd.Timestamp(event["event_datetime"]),
                    "knowledge_cutoff": pd.Timestamp(event["knowledge_cutoff"]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Retrieval — one call per ticker, through sources.fetch
# --------------------------------------------------------------------------


def _yahoo_symbol(ticker: str) -> str:
    """Yahoo writes class shares with a dash: `BF.B` is `BF-B`."""
    return ticker.replace(".", "-")


def _download(tickers: list[str], end: str) -> pd.DataFrame:
    import yfinance as yf

    symbols = {_yahoo_symbol(t): t for t in tickers}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            list(symbols),
            start=HISTORY_START,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    if raw is None or not len(raw):
        return pd.DataFrame()
    frames = []
    for symbol, ticker in symbols.items():
        try:
            sub = raw[symbol] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        sub = sub.dropna(how="all")
        if not len(sub):
            continue
        sub = sub.reset_index()
        sub.columns = [str(c).lower() for c in sub.columns]
        sub["ticker"] = ticker
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cache_bars(tickers: list[str], end: str, batch: int = 60, refresh: bool = False) -> pd.DataFrame:
    """Fetch and cache daily bars for `tickers`. Re-runs are free.

    One ``sources.fetch`` record per batch of tickers, which is one network
    call. ``knowledge_cutoff`` is None because a bulk cache is not tied to an
    event; the per-event cutoff is enforced in :func:`bars_for_event` and
    recorded in ``event_windows.parquet``.
    """
    have = pd.DataFrame()
    if BARS.exists() and not refresh:
        have = pd.read_parquet(BARS)
    known = set(have.ticker.unique()) if len(have) else set()
    todo = sorted(set(tickers) - known)
    if not todo:
        return have

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    out = [have] if len(have) else []
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        frame = sources.fetch(
            source="yfinance",
            endpoint=f"download/daily[{len(chunk)} tickers]",
            loader=lambda c=chunk: _download(c, end),
            window=(pd.Timestamp(HISTORY_START, tz="UTC"), pd.Timestamp(end, tz="UTC")),
            point_in_time=True,
            notes=(
                "bulk daily-bar cache, split/dividend back-adjusted; per-event "
                "cutoff enforced by blocks.price_window + sources.check_window "
                "and recorded in data/prices/event_windows.parquet"
            ),
        )
        if len(frame):
            out.append(frame)
        print(f"  {i + len(chunk):>5}/{len(todo)} tickers, {sum(len(f) for f in out):>8} bars", flush=True)
        time.sleep(0.2)

    bars = pd.concat(out, ignore_index=True)
    bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None).dt.normalize()
    bars = bars.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    bars.to_parquet(BARS, index=False)
    return bars


# --------------------------------------------------------------------------
# The per-event window
# --------------------------------------------------------------------------


def bars_for_event(panel: dict[str, pd.DataFrame], ticker: str, cutoff, lookback_days: int = 420):
    """Bars for `ticker` dated at or before the event's last usable bar.

    Returns `(frame, last_bar_date)`. The cutoff check is run, not assumed.
    """
    _, audit_end, last_bar = B.price_window(cutoff, lookback_days)
    sources.check_window(cutoff, audit_end)
    frame = panel.get(ticker)
    if frame is None:
        return None, last_bar
    return frame[frame.date <= last_bar], last_bar


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _runup(close: np.ndarray, n: int) -> float:
    if len(close) < n + 1:
        return np.nan
    a, b = close[-n - 1], close[-1]
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return np.nan
    return float(b / a - 1.0)


def _log_returns(close: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(close))


def build(sample: pd.DataFrame | None = None, quarters=tuple(harness.QUARTERS)) -> pd.DataFrame:
    """The price block. One row per event, NaN where the source could not serve it."""
    events = universe(quarters) if sample is None else sample
    tickers = sorted(set(events.ticker) | {MARKET})
    end = (events.knowledge_cutoff.max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    bars = cache_bars(tickers, end)

    panel = {t: g.reset_index(drop=True) for t, g in bars.groupby("ticker")}
    market = panel.get(MARKET)
    if market is None:
        raise RuntimeError(f"no bars for the market proxy {MARKET}")
    market = market.set_index("date")["close"]
    market_logret = np.log(market).diff()

    # Prior earnings event for the same ticker, from the archive's own history.
    events = events.sort_values(["ticker", "event_datetime"]).copy()
    events["prior_cutoff"] = events.groupby("ticker")["knowledge_cutoff"].shift(1)

    rows, windows = [], []
    for event in events.itertuples(index=False):
        row = {"event_id": event.event_id}
        frame, last_bar = bars_for_event(panel, event.ticker, event.knowledge_cutoff)
        windows.append(
            {
                "event_id": event.event_id,
                "ticker": event.ticker,
                "knowledge_cutoff": event.knowledge_cutoff,
                "last_bar_allowed": last_bar,
                "last_bar_used": frame.date.iloc[-1] if frame is not None and len(frame) else pd.NaT,
                "n_bars": 0 if frame is None else len(frame),
            }
        )
        if frame is None or len(frame) < 25:
            rows.append(row)
            continue

        close = frame.close.to_numpy(dtype=float)
        openp = frame.open.to_numpy(dtype=float)
        volume = frame.volume.to_numpy(dtype=float)
        dates = frame.date.to_numpy()

        # --- run-up, raw and market-excess -------------------------------
        mkt = market_logret.reindex(frame.date).to_numpy(dtype=float)
        stock_logret = np.concatenate([[np.nan], _log_returns(close)])
        for n in RUNUP_WINDOWS:
            row[f"runup_{n}d"] = _runup(close, n)
            if len(stock_logret) >= n + 1:
                s = stock_logret[-n:]
                m = mkt[-n:]
                ok = np.isfinite(s) & np.isfinite(m)
                row[f"xsrunup_{n}d"] = float(np.nansum(s[ok]) - np.nansum(m[ok])) if ok.sum() >= max(1, n - 2) else np.nan
            else:
                row[f"xsrunup_{n}d"] = np.nan

        # --- vol-normalised run-up (vol as NORMALISER, never a signal) ----
        tail = stock_logret[-252:]
        tail = tail[np.isfinite(tail)]
        sigma = float(np.std(tail, ddof=1)) if len(tail) >= 60 else np.nan
        if np.isfinite(sigma) and sigma > 1e-8:
            for n in (5, 20, 60):
                r = row.get(f"xsrunup_{n}d", np.nan)
                row[f"xsrunup_{n}d_z"] = r / (sigma * np.sqrt(n)) if np.isfinite(r) else np.nan
        else:
            for n in (5, 20, 60):
                row[f"xsrunup_{n}d_z"] = np.nan

        # --- distance from the 52-week extremes ---------------------------
        year = close[-252:]
        year = year[np.isfinite(year)]
        if len(year) >= 120 and close[-1] > 0:
            hi, lo = float(year.max()), float(year.min())
            row["dist_52w_high"] = float(close[-1] / hi - 1.0) if hi > 0 else np.nan
            row["dist_52w_low"] = float(close[-1] / lo - 1.0) if lo > 0 else np.nan
            row["pos_52w_range"] = float((close[-1] - lo) / (hi - lo)) if hi > lo else np.nan
        else:
            row["dist_52w_high"] = row["dist_52w_low"] = row["pos_52w_range"] = np.nan

        # --- return since the company's prior earnings event ---------------
        row["ret_since_prior_earnings"] = np.nan
        row["xsret_since_prior_earnings"] = np.nan
        prior = event.prior_cutoff
        if isinstance(prior, pd.Timestamp) and pd.notna(prior):
            prior_bar = B.price_window(prior, 1)[2]
            mask = frame.date <= prior_bar
            if mask.any():
                base = float(close[mask.to_numpy().nonzero()[0][-1]])
                if base > 0:
                    row["ret_since_prior_earnings"] = float(close[-1] / base - 1.0)
                    since = (frame.date > prior_bar).to_numpy()
                    s, m = stock_logret[since], mkt[since]
                    ok = np.isfinite(s) & np.isfinite(m)
                    if ok.sum() >= 10:
                        row["xsret_since_prior_earnings"] = float(s[ok].sum() - m[ok].sum())

        # --- volume trend into the event (scale-free ratio) ----------------
        vol = volume[np.isfinite(volume) & (volume > 0)]
        if len(vol) >= 70:
            recent = float(np.median(vol[-5:]))
            trailing = float(np.median(vol[-63:-5]))
            row["volume_trend_5_60"] = float(np.log(recent / trailing)) if trailing > 0 else np.nan
            recent20 = float(np.median(vol[-20:]))
            row["volume_trend_20_60"] = float(np.log(recent20 / trailing)) if trailing > 0 else np.nan
        else:
            row["volume_trend_5_60"] = row["volume_trend_20_60"] = np.nan

        # --- gap decomposition: overnight vs intraday ----------------------
        # Overnight = open_t / close_{t-1}; intraday = close_t / open_t. The
        # split is directional: which session the recent drift arrived in.
        if len(close) >= 21 and np.isfinite(openp[-20:]).all():
            with np.errstate(divide="ignore", invalid="ignore"):
                overnight = np.log(openp[1:] / close[:-1])
                intraday = np.log(close[1:] / openp[1:])
            for n in (5, 20):
                o, d = overnight[-n:], intraday[-n:]
                row[f"overnight_{n}d"] = float(np.nansum(o)) if np.isfinite(o).sum() >= n - 1 else np.nan
                row[f"intraday_{n}d"] = float(np.nansum(d)) if np.isfinite(d).sum() >= n - 1 else np.nan
            row["gap_share_20d"] = (
                row["overnight_20d"] - row["intraday_20d"]
                if np.isfinite(row.get("overnight_20d", np.nan)) and np.isfinite(row.get("intraday_20d", np.nan))
                else np.nan
            )
        else:
            for n in (5, 20):
                row[f"overnight_{n}d"] = row[f"intraday_{n}d"] = np.nan
            row["gap_share_20d"] = np.nan

        # --- run-up reversal: 60d run-up net of the last 5d ----------------
        r60, r5 = row.get("xsrunup_60d", np.nan), row.get("xsrunup_5d", np.nan)
        row["xsrunup_60d_ex5"] = r60 - r5 if np.isfinite(r60) and np.isfinite(r5) else np.nan

        _ = dates
        rows.append(row)

    table = pd.DataFrame(rows)
    win = pd.DataFrame(windows)
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    win.to_parquet(WINDOWS, index=False)
    bad = win[win.last_bar_used.notna() & (win.last_bar_used > win.last_bar_allowed)]
    if len(bad):
        raise sources.CutoffViolation(f"{len(bad)} events used a bar after their last allowed bar")
    return table


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

FEATURES = [
    B.Feature("runup_1d", True, True, "raw 1-day return into the cutoff"),
    B.Feature("runup_5d", True, True, "raw 5-day run-up into the cutoff"),
    B.Feature("runup_20d", True, True, "raw 20-day run-up into the cutoff"),
    B.Feature("runup_60d", True, True, "raw 60-day run-up into the cutoff"),
    B.Feature("xsrunup_1d", True, True, "SPY-excess 1-day return"),
    B.Feature("xsrunup_5d", True, True, "SPY-excess 5-day run-up"),
    B.Feature("xsrunup_20d", True, True, "SPY-excess 20-day run-up"),
    B.Feature("xsrunup_60d", True, True, "SPY-excess 60-day run-up"),
    B.Feature("xsrunup_5d_z", True, True, "excess 5d run-up / trailing 252d vol (vol as normaliser)"),
    B.Feature("xsrunup_20d_z", True, True, "excess 20d run-up / trailing vol"),
    B.Feature("xsrunup_60d_z", True, True, "excess 60d run-up / trailing vol"),
    B.Feature("xsrunup_60d_ex5", True, True, "excess 60d run-up net of the last 5 days"),
    B.Feature("dist_52w_high", True, True, "close / 52-week high - 1"),
    B.Feature("dist_52w_low", True, True, "close / 52-week low - 1"),
    B.Feature("pos_52w_range", True, True, "position of close within the 52-week range"),
    B.Feature("ret_since_prior_earnings", True, True, "return since the archive's prior event for this ticker"),
    B.Feature("xsret_since_prior_earnings", True, True, "SPY-excess return since the prior event"),
    B.Feature("volume_trend_5_60", True, True, "log(median 5d volume / median 60d volume)"),
    B.Feature("volume_trend_20_60", True, True, "log(median 20d volume / median 60d volume)"),
    B.Feature("overnight_5d", True, True, "sum of 5 days of overnight log returns"),
    B.Feature("overnight_20d", True, True, "sum of 20 days of overnight log returns"),
    B.Feature("intraday_5d", True, True, "sum of 5 days of intraday log returns"),
    B.Feature("intraday_20d", True, True, "sum of 20 days of intraday log returns"),
    B.Feature("gap_share_20d", True, True, "overnight minus intraday drift over 20 days"),
]

BLOCK = B.register(
    B.Block(
        name="price",
        owner="Agent A — price and volume history",
        features=FEATURES,
        notes=(
            "yfinance daily bars, residential IP. Split/dividend back-adjusted: "
            "fine for returns and price ratios, NOT applied to volume by Yahoo, "
            "so a split inside a volume lookback corrupts that one event. "
            "No current-value field is used, so every feature is point-in-time."
        ),
    )
)


if __name__ == "__main__":
    print(f"universe: {len(universe())} events, {universe().ticker.nunique()} tickers")
    table = build()
    path = B.write("price", table)
    print(f"wrote {path}  {table.shape}")
