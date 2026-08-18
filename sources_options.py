"""Options skew — the directional half of an options read, and a source survey.

Agent E. The brief rules out most of this source before any code is written:
implied move, ATM IV and IV *level* are magnitude-without-direction, which has
been measured dead here three independent ways. What is left is **directional**:

  * put IV minus call IV at comparable moneyness (skew), and its *change*
    against a trailing level;
  * 25-delta risk reversal;
  * put vs call open-interest imbalance;
  * put/call *volume* ratio.

All four need an option chain **as of the event's cutoff date**. That is the
whole question, and it is a sourcing question rather than a modelling one, so
this module is mostly instrumentation for answering it:

  * :func:`option_universe_ceiling` — of the archive's tickers, how many have a
    listed option chain at all. A hard ceiling on coverage no source can beat.
  * :func:`probe_alphavantage_historical` — does ``HISTORICAL_OPTIONS`` serve a
    dated chain on the free key, and what does it contain.
  * :func:`probe_dolthub` — the free public Dolt options archive, by SQL.
  * :func:`probe_cboe_delayed` — Cboe's free delayed chain, current-only.
  * :func:`skew_from_chain` — the feature maths, so that the moment a
    point-in-time chain exists the block is one loop away.

Every retrieval goes through :func:`sources.fetch`, which enforces the cutoff
and writes the audit record (Rules §04/§07/§10). Live chains are fetched with
``point_in_time=False``: a chain pulled today is *today's* chain, and for a past
event that is lookahead of the most direct kind.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sources import fetch  # noqa: E402

CACHE = ROOT / "data" / "options"

DOLTHUB_SQL = "https://www.dolthub.com/api/v1alpha1/{owner}/{repo}/{branch}"
CBOE_DELAYED = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
ALPHAVANTAGE = "https://www.alphavantage.co/query"


# --------------------------------------------------------------------------
# The feature maths — kept separate from sourcing on purpose
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Skew:
    """The directional readings from one chain snapshot.

    Every field is a *difference between puts and calls*. Nothing here is a
    level, because a level is magnitude-without-direction and scores zero.
    """

    iv_skew: float          # put IV − call IV, matched |moneyness|
    risk_reversal_25d: float  # 25Δ put IV − 25Δ call IV
    oi_imbalance: float     # (put OI − call OI) / (put OI + call OI)
    volume_ratio: float     # log(put volume / call volume)
    n_pairs: int


def _nearest_expiry(chain: pd.DataFrame, cutoff_date, min_days: int = 5) -> object:
    """The front expiry at least ``min_days`` out.

    The event is tomorrow; the *front* expiry is the one that prices it. But an
    expiry inside a few days is dominated by pin and gamma effects rather than
    by any directional view, so the window starts a few days out.
    """
    days = (pd.to_datetime(chain.expiration) - pd.Timestamp(cutoff_date)).dt.days
    eligible = chain[days >= min_days]
    if eligible.empty:
        return None
    return eligible.assign(_d=days[days >= min_days]).sort_values("_d").expiration.iloc[0]


def skew_from_chain(
    chain: pd.DataFrame,
    spot: float,
    cutoff_date,
    *,
    band: tuple[float, float] = (0.03, 0.15),
) -> Skew:
    """Directional skew readings from a single dated chain.

    ``chain`` needs columns ``expiration, strike, call_put, iv`` and optionally
    ``open_interest, volume``. ``call_put`` is ``"C"``/``"P"``.

    The IV skew pairs each out-of-the-money put with the call at the mirrored
    moneyness rather than differencing raw IVs, so a steep-but-symmetric smile
    reads as zero and only the *tilt* survives. The band excludes near-ATM
    strikes (where put and call IV are mechanically equal via parity, so the
    difference is noise) and the far tail (where quotes are stale).
    """
    nan = float("nan")
    expiry = _nearest_expiry(chain, cutoff_date)
    if expiry is None or not np.isfinite(spot) or spot <= 0:
        return Skew(nan, nan, nan, nan, 0)
    front = chain[chain.expiration == expiry].copy()
    front["m"] = np.log(front.strike.astype(float) / spot)

    puts = front[(front.call_put == "P") & (front.m < 0)]
    calls = front[(front.call_put == "C") & (front.m > 0)]
    lo, hi = band
    puts = puts[puts.m.abs().between(lo, hi)]
    calls = calls[calls.m.abs().between(lo, hi)]

    pairs = []
    for _, put in puts.iterrows():
        if calls.empty:
            break
        j = (calls.m - abs(put.m)).abs().idxmin()
        call = calls.loc[j]
        if abs(abs(call.m) - abs(put.m)) > 0.02:
            continue
        if np.isfinite(put.iv) and np.isfinite(call.iv):
            pairs.append(float(put.iv) - float(call.iv))

    iv_skew = float(np.mean(pairs)) if pairs else nan

    rr = nan
    if "delta" in front.columns:
        p = front[(front.call_put == "P") & front.delta.notna()]
        c = front[(front.call_put == "C") & front.delta.notna()]
        if len(p) and len(c):
            pi = p.loc[(p.delta.astype(float) + 0.25).abs().idxmin()]
            ci = c.loc[(c.delta.astype(float) - 0.25).abs().idxmin()]
            if np.isfinite(pi.iv) and np.isfinite(ci.iv):
                rr = float(pi.iv) - float(ci.iv)

    oi = nan
    if "open_interest" in front.columns:
        po = front.loc[front.call_put == "P", "open_interest"].sum()
        co = front.loc[front.call_put == "C", "open_interest"].sum()
        if po + co > 0:
            oi = float((po - co) / (po + co))

    vr = nan
    if "volume" in front.columns:
        pv = front.loc[front.call_put == "P", "volume"].sum()
        cv = front.loc[front.call_put == "C", "volume"].sum()
        if pv > 0 and cv > 0:
            vr = float(np.log(pv / cv))

    return Skew(iv_skew, rr, oi, vr, len(pairs))


# --------------------------------------------------------------------------
# Source probes
# --------------------------------------------------------------------------


def option_universe_ceiling(tickers, limit: int | None = None) -> pd.DataFrame:
    """How many archive tickers have a listed option chain **at all**.

    This is a *current* check — `point_in_time=False` — and it is deliberately
    an upper bound: a name with options today may not have had them last
    October, and a name that has since delisted will read as no-options. Its
    only job is to cap the coverage any options source could ever deliver.
    """
    import yfinance as yf

    out = []
    seen = list(dict.fromkeys(tickers))[: limit or None]
    for ticker in seen:
        try:
            expiries = fetch(
                source="yfinance",
                endpoint=f"options/{ticker}",
                loader=lambda t=ticker: list(yf.Ticker(t).options),
                point_in_time=False,
                notes="current expiry list; coverage-ceiling probe only",
            )
        except Exception as exc:  # noqa: BLE001
            out.append({"ticker": ticker, "n_expiries": 0, "error": str(exc)[:80]})
            continue
        out.append({"ticker": ticker, "n_expiries": len(expiries), "error": ""})
    return pd.DataFrame(out)


def yfinance_chain_now(ticker: str) -> tuple[pd.DataFrame, float]:
    """Today's chain for ``ticker``, normalised to the :func:`skew_from_chain` schema.

    **Not point-in-time.** Yahoo exposes no date parameter: this is the chain as
    it stands right now, so for any archive event it is lookahead and it is
    logged ``point_in_time=False``. Its use here is twofold — measuring how
    often a skew is *computable* at all on the archive's kind of name, and
    standing as the going-forward live path, where "right now" is exactly right.
    """
    import yfinance as yf

    def load():
        t = yf.Ticker(ticker)
        expiries = list(t.options)
        frames = []
        for expiry in expiries[:4]:
            opt = t.option_chain(expiry)
            for side, table in (("C", opt.calls), ("P", opt.puts)):
                part = table[["strike", "impliedVolatility", "openInterest", "volume"]].copy()
                part.columns = ["strike", "iv", "open_interest", "volume"]
                part["call_put"] = side
                part["expiration"] = pd.Timestamp(expiry)
                frames.append(part)
        chain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["strike", "iv", "open_interest", "volume", "call_put", "expiration"]
        )
        hist = t.history(period="5d")
        spot = float(hist.Close.iloc[-1]) if len(hist) else float("nan")
        return chain, spot

    return fetch(
        source="yfinance",
        endpoint=f"option_chain/{ticker}",
        loader=load,
        point_in_time=False,
        notes="CURRENT chain only — Yahoo has no date parameter; lookahead for any past event",
    )


def probe_alphavantage_historical(symbol: str = "IBM", date: str = "2026-02-02", key: str | None = None):
    """Does ``HISTORICAL_OPTIONS`` serve a dated chain, and on what key."""
    import requests

    key = key or os.environ.get("ALPHAVANTAGE_API_KEY") or "demo"
    params = {"function": "HISTORICAL_OPTIONS", "symbol": symbol, "date": date, "apikey": key}
    url = f"{ALPHAVANTAGE}?{urllib.parse.urlencode(params)}"
    return fetch(
        source="alphavantage",
        endpoint=f"HISTORICAL_OPTIONS/{symbol}/{date}",
        loader=lambda: requests.get(url, timeout=30).json(),
        point_in_time=True,
        notes=f"dated historical chain probe, key={'demo' if key == 'demo' else 'private'}",
    )


def probe_dolthub(query: str, owner: str = "post-no-preference", repo: str = "options", branch: str = "master"):
    """Run one SQL query against the free public Dolt options archive."""
    import requests

    base = DOLTHUB_SQL.format(owner=owner, repo=repo, branch=branch)
    url = f"{base}?q={urllib.parse.quote(query)}"
    return fetch(
        source="dolthub",
        endpoint=f"{owner}/{repo}: {query[:90]}",
        loader=lambda: requests.get(url, timeout=90).json(),
        point_in_time=True,
        notes="free public SQL API; dated daily option chains",
    )


# --------------------------------------------------------------------------
# The build — delta-space skew from the free Dolt archive
# --------------------------------------------------------------------------

#: Only quotes in this |delta| band are pulled. Deep ITM implied vols in this
#: source are unreliable (a 0.995-delta call quotes IV 0.84 against an ATM 0.22),
#: and the far tail is stale. Everything directional lives in between.
DELTA_BAND = (0.05, 0.45)

CHAIN_SQL = (
    "SELECT expiration, strike, call_put, vol, delta FROM option_chain "
    "WHERE act_symbol='{symbol}' AND `date`='{date}' "
    "AND ABS(delta) BETWEEN {lo} AND {hi}"
)


def dolt_chain(symbol: str, date, *, event_id=None, knowledge_cutoff=None) -> pd.DataFrame:
    """One symbol's chain on one date, delta-banded, from the free Dolt archive.

    A single-date equality query hits the `(date, act_symbol, ...)` primary key
    and returns in well under a second. A *range* over dates does not — it
    forces a scan and the API's server-side deadline kills it at ~55s with
    partial rows, which is why this is one query per (symbol, date) rather than
    one per symbol.
    """
    sql = CHAIN_SQL.format(symbol=symbol, date=date, lo=DELTA_BAND[0], hi=DELTA_BAND[1])
    payload = probe_dolthub(sql)
    if payload.get("query_execution_status") != "Success":
        raise RuntimeError(payload.get("query_execution_message", "dolt query failed"))
    rows = payload.get("rows", [])
    if not rows:
        return pd.DataFrame(columns=["expiration", "strike", "call_put", "vol", "delta"])
    frame = pd.DataFrame(rows)
    for column in ("strike", "vol", "delta"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["expiration"] = pd.to_datetime(frame.expiration)
    return frame


def delta_skew(chain: pd.DataFrame, asof, *, min_days: int = 5) -> dict[str, float]:
    """Directional skew in **delta space** — no spot price required.

    Working in delta rather than strike is what makes this source usable at all:
    the archive carries no underlying price, and delta is a monotone, already-
    normalised moneyness. It also makes the readings comparable across names
    without dividing by a volatility, which would reintroduce a magnitude term.

    All three readings are *differences between puts and calls*. None is a level.

      ``rr_25d`` / ``rr_10d``
        Risk reversal: the 25- (10-) delta put's IV minus the equally-far-OTM
        call's IV. Positive means puts are bid relative to calls — a bearish
        tilt in the options market's pricing of the announcement.
      ``skew_slope``
        The tilt of the whole smile, fitted rather than sampled at two points,
        so it survives a sparse strike grid that has no contract near 25 delta.
        Puts and calls are placed on one axis by the put-delta convention
        (a call of delta d sits at d−1), giving x ∈ [−1, 0] running from far-OTM
        put to far-OTM call. A positive slope is a put-skewed, bearish smile.
    """
    nan = float("nan")
    out = {"rr_25d": nan, "rr_10d": nan, "skew_slope": nan, "n_quotes": 0.0}
    if chain.empty:
        return out
    days = (chain.expiration - pd.Timestamp(asof)).dt.days
    eligible = chain[days >= min_days]
    if eligible.empty:
        return out
    expiry = eligible.assign(_d=days[days >= min_days]).sort_values("_d").expiration.iloc[0]
    front = eligible[eligible.expiration == expiry].dropna(subset=["vol", "delta"])
    front = front[front.vol > 0.01]
    puts = front[front.call_put.str.startswith("P")]
    calls = front[front.call_put.str.startswith("C")]
    out["n_quotes"] = float(len(front))
    if len(front) < 4:
        return out

    for label, target in (("rr_25d", 0.25), ("rr_10d", 0.10)):
        if puts.empty or calls.empty:
            continue
        p = puts.loc[(puts.delta.abs() - target).abs().idxmin()]
        c = calls.loc[(calls.delta.abs() - target).abs().idxmin()]
        # Refuse a "25-delta" reading taken off a 45-delta contract, and refuse
        # a pair whose two legs sit at different distances from the money —
        # that difference would be smile curvature, not tilt.
        if abs(abs(p.delta) - target) > 0.12 or abs(abs(c.delta) - target) > 0.12:
            continue
        if abs(abs(p.delta) - abs(c.delta)) > 0.10:
            continue
        out[label] = float(p.vol - c.vol)

    if len(puts) >= 2 and len(calls) >= 2:
        x = np.concatenate([puts.delta.to_numpy(float), calls.delta.to_numpy(float) - 1.0])
        y = np.concatenate([puts.vol.to_numpy(float), calls.vol.to_numpy(float)])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 4 and np.ptp(x[ok]) > 0.15:
            out["skew_slope"] = float(np.polyfit(x[ok], y[ok], 1)[0])
    return out


def probe_cboe_delayed(symbol: str = "AAPL"):
    """Cboe's free delayed chain. Current-only — no date parameter exists."""
    import requests

    url = CBOE_DELAYED.format(symbol=symbol)
    return fetch(
        source="cboe",
        endpoint=f"delayed_quotes/options/{symbol}",
        loader=lambda: requests.get(url, timeout=30, headers={"User-Agent": "research"}).json(),
        point_in_time=False,
        notes="current delayed chain; no historical/date parameter",
    )
