"""Ownership- and flow-side positioning data. Agent F.

The question this block asks is **what was already priced in from the ownership
side** — how the short base and the daily short-sale flow moved *into* the
print, and whether insiders were buying or selling ahead of it. Agent A asks the
same question from the price side; the number that decides whether both are
worth carrying is the ρ_b between them, not either standalone ρ.

Three sources, and their point-in-time status is not the same:

``finra_short_interest``
    FINRA's consolidated short interest, twice a month, free and unauthenticated
    at ``api.finra.org``. Carries ``settlementDate`` — the *position* date — and
    **not** a publication date. FINRA disseminates roughly eight business days
    after settlement, so the settlement date alone is a lookahead trap: on
    2026-06-30 the 2026-06-30 file does not exist yet. :func:`publication_date`
    applies the lag and every feature filters on the *published* vintage.

``finra_short_volume``
    FINRA's daily consolidated short-sale volume (``CNMSshvol<YYYYMMDD>.txt``) —
    a different product from short interest: aggregate short *volume* per symbol
    per trade date across the FINRA-reported venues. Published after the close of
    the trade date, so the file dated the cutoff date is **not** safe even though
    the price bar dated the cutoff date is. The last file this module will use
    for an event is the last trading day strictly before the cutoff date.

``edgar_form4``
    Insider transactions. Agent B owns the EDGAR parse; this module reads
    ``data/edgar/form4*`` if it is there and builds features on top of it.

Everything external goes through :func:`sources.fetch`, which writes the audit
log the organisers may demand under Rules §07/§10.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sources import fetch  # noqa: E402

CACHE = ROOT / "data" / "flow"
UA = "explaining-markets research (danishtaher7@gmail.com)"

SHORT_VOLUME_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
SHORT_INTEREST_API = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)

#: FINRA disseminates consolidated short interest on the eighth business day
#: after the settlement date (FINRA Rule 4560 reporting schedule). Nothing in
#: the API payload carries the dissemination date, so it is derived — and
#: derived *long*, because being a day early is a Rules §04 violation and being
#: a day late costs a stale vintage on a handful of events.
SI_PUBLICATION_LAG_BDAYS = 9


# --------------------------------------------------------------------------
# The universe we actually need
# --------------------------------------------------------------------------


def universe() -> pd.DataFrame:
    """`event_id, ticker, cutoff_et` for every archive record, all four quarters.

    Read from the raw archive rather than `harness.load`, which drops the 180
    records missing `y` or `surprise_pct`. Those are unscorable and so invisible
    to `score_features`, but they are real events a live model would be asked to
    predict, so the block covers them: 8,020, not 7,840.
    """
    import gzip
    import json

    import harness

    rows = []
    for quarter in harness.QUARTERS:
        path = harness.ARCHIVE_DIR / f"EARNINGS_RELEASE_{quarter}.jsonl.gz"
        for line in gzip.open(path, "rt"):
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                (
                    record["event_id"],
                    next(iter(record.get("event_returns") or {}), None),
                    record.get("knowledge_cutoff"),
                )
            )
    out = pd.DataFrame(rows, columns=["event_id", "ticker", "knowledge_cutoff"])
    out = out.drop_duplicates("event_id")
    cutoff = pd.to_datetime(out.knowledge_cutoff, utc=True)
    out["cutoff_et"] = cutoff.dt.tz_convert("America/New_York")
    out["cutoff_date"] = out.cutoff_et.dt.normalize().dt.tz_localize(None)
    return out.drop(columns=["knowledge_cutoff"])


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    return session


# --------------------------------------------------------------------------
# FINRA daily consolidated short-sale volume
# --------------------------------------------------------------------------


def fetch_short_volume(start: str, end: str, *, refresh: bool = False) -> pd.DataFrame:
    """One row per (trade date, symbol) of the universe. Cached to parquet.

    The raw file is every US symbol (~11k rows/day); only the ~2.5k tickers that
    appear in the archive are kept, which turns 140 MB of text into a few MB.
    """
    path = CACHE / "short_volume.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    CACHE.mkdir(parents=True, exist_ok=True)
    tickers = set(universe().ticker)
    session = _session()
    days = pd.bdate_range(start, end)
    frames, missing = [], 0

    for day in days:
        stamp = day.strftime("%Y%m%d")
        url = SHORT_VOLUME_URL.format(date=stamp)

        def loader(url=url):
            response = session.get(url, timeout=45)
            if response.status_code in (403, 404):
                return None
            response.raise_for_status()
            return response.text

        text = fetch(
            source="finra_short_volume",
            endpoint=f"regsho/daily/CNMSshvol{stamp}",
            loader=loader,
            window=(day, day),
            point_in_time=True,
            notes=(
                "daily consolidated short-sale volume for trade date "
                f"{day.date()}; published after that day's close, so it is only "
                "usable for events whose cutoff date is strictly later"
            ),
        )
        if not text:
            missing += 1
            continue
        frame = pd.read_csv(io.StringIO(text), sep="|")
        frame = frame[frame.Symbol.isin(tickers)]
        frames.append(frame)
        time.sleep(0.05)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out.Date, format="%Y%m%d")
    out = out.rename(
        columns={
            "Symbol": "ticker",
            "ShortVolume": "short_volume",
            "ShortExemptVolume": "short_exempt",
            "TotalVolume": "total_volume",
        }
    )[["date", "ticker", "short_volume", "short_exempt", "total_volume"]]
    out.to_parquet(path, index=False)
    print(f"short volume: {len(out):,} rows, {out.date.nunique()} days, {missing} absent")
    return out


# --------------------------------------------------------------------------
# FINRA consolidated short interest (bi-monthly)
# --------------------------------------------------------------------------


def publication_date(settlement: pd.Series | pd.Timestamp):
    """The date a settlement-date vintage first became public. **Derived.**

    FINRA's own schedule publishes settlement/dissemination pairs; the API does
    not return the dissemination date, so it is reconstructed as the ninth
    business day after settlement — one business day longer than FINRA's stated
    eighth-business-day dissemination, deliberately. An overestimate loses a
    stale vintage on a few events; an underestimate is a Rules §04 violation.
    """
    stamp = pd.to_datetime(settlement)
    return stamp + pd.tseries.offsets.BDay(SI_PUBLICATION_LAG_BDAYS)


def fetch_short_interest(start: str, *, refresh: bool = False) -> pd.DataFrame:
    """Every consolidated short interest row with settlement date >= `start`."""
    path = CACHE / "short_interest.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    CACHE.mkdir(parents=True, exist_ok=True)
    tickers = set(universe().ticker)
    session = _session()
    rows, offset, limit = [], 0, 5000

    while True:
        body = {
            "limit": limit,
            "offset": offset,
            "compareFilters": [
                {"fieldName": "settlementDate", "fieldValue": start, "compareType": "gte"}
            ],
        }

        def loader(body=body):
            response = session.post(SHORT_INTEREST_API, json=body, timeout=90)
            response.raise_for_status()
            return response.text

        text = fetch(
            source="finra_short_interest",
            endpoint=f"consolidatedShortInterest?offset={offset}",
            loader=loader,
            point_in_time=True,
            notes=(
                "bi-monthly consolidated short interest; settlementDate is the "
                "position date, dissemination is ~8 business days later and is "
                "not carried in the payload — see publication_date()"
            ),
        )
        chunk = pd.read_csv(io.StringIO(text))
        if chunk.empty:
            break
        rows.append(chunk[chunk.symbolCode.isin(tickers)])
        offset += limit
        if len(chunk) < limit:
            break
        time.sleep(0.1)

    out = pd.concat(rows, ignore_index=True)
    out["settlement_date"] = pd.to_datetime(out.settlementDate)
    out["publication_date"] = publication_date(out.settlement_date)
    out = out.rename(
        columns={
            "symbolCode": "ticker",
            "currentShortPositionQuantity": "shares_short",
            "previousShortPositionQuantity": "shares_short_prev",
            "averageDailyVolumeQuantity": "adv",
            "daysToCoverQuantity": "days_to_cover",
            "changePercent": "change_pct",
        }
    )[
        [
            "settlement_date",
            "publication_date",
            "ticker",
            "shares_short",
            "shares_short_prev",
            "adv",
            "days_to_cover",
            "change_pct",
        ]
    ]
    out = out.sort_values(["ticker", "settlement_date"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    print(
        f"short interest: {len(out):,} rows, "
        f"{out.settlement_date.nunique()} settlement dates, "
        f"{out.ticker.nunique():,} tickers"
    )
    return out


# --------------------------------------------------------------------------
# Features — short interest
# --------------------------------------------------------------------------


def _asof_vintage(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Attach, per event, the index of the newest vintage *published* before it.

    `allow_exact_matches=False` is the whole point: a vintage published on the
    cutoff date is published at some hour of that date, and the cutoff is 16:00
    ET. Rather than guess the hour, the vintage is refused. That costs at most
    one vintage on the 27 publication dates that coincide with a cutoff date.
    """
    panel = panel.sort_values(["ticker", "publication_date"]).copy()
    panel["seq"] = panel.groupby("ticker").cumcount()
    keys = panel[["ticker", "publication_date", "seq"]].sort_values("publication_date")
    left = events.sort_values("cutoff_date")
    joined = pd.merge_asof(
        left,
        keys,
        left_on="cutoff_date",
        right_on="publication_date",
        by="ticker",
        direction="backward",
        allow_exact_matches=False,
    )
    return joined, panel


def short_interest_features() -> pd.DataFrame:
    """Positioning from the short base: the *change*, not the level.

    Every column is a function of vintages whose derived publication date is
    strictly before the event's cutoff date. The level (`si_days_to_cover`) is
    carried too, because the short-interest literature gives it a sign — high
    short interest predicting low subsequent returns — so it is a directional
    level rather than a magnitude-without-direction characteristic.
    """
    import numpy as np

    events = universe()
    panel = pd.read_parquet(CACHE / "short_interest.parquet")
    joined, panel = _asof_vintage(events, panel)

    indexed = panel.set_index(["ticker", "seq"])
    shares = indexed.shares_short.astype(float)
    adv = indexed.adv.astype(float)
    dtc = indexed.days_to_cover.astype(float)

    def at(offset: int, series: pd.Series) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([joined.ticker, joined.seq - offset])
        return series.reindex(idx).to_numpy(dtype=float)

    now, back1, back3, back6 = (at(k, shares) for k in (0, 1, 3, 6))
    adv_now = at(0, adv)
    adv_now = np.where(adv_now > 0, adv_now, np.nan)

    out = pd.DataFrame({"event_id": joined.event_id})
    out["si_chg_1"] = np.log(now / back1)
    out["si_chg_3"] = np.log(now / back3)
    out["si_chg_6"] = np.log(now / back6)
    out["si_chg_1_over_adv"] = (now - back1) / adv_now
    out["si_chg_3_over_adv"] = (now - back3) / adv_now
    out["si_days_to_cover"] = at(0, dtc)

    # A vintage carried over from a prior *quarter* is not the same object as a
    # two-week-old one; the diagnostic is kept out of the block but reported.
    settle = at(0, indexed.settlement_date.astype("int64").astype(float))
    out["_stale_days"] = (
        joined.cutoff_date.astype("int64").to_numpy(dtype=float) - settle
    ) / 86_400_000_000_000
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# Features — daily consolidated short-sale volume
# --------------------------------------------------------------------------


def short_volume_features() -> pd.DataFrame:
    """Flow, not stock: how the daily short-sale share moved *into* the print.

    The level of the short-volume ratio is a venue-mix characteristic as much as
    a positioning fact — a symbol's share of off-exchange market-maker flow is
    persistent and has nothing to do with this quarter's earnings. So the
    features are all **differences against the symbol's own recent baseline**,
    which differences that mix out.

    Cutoff: the file for trade date T is published after T's close, and the
    cutoff is 16:00 ET on the event date, so the last file used is the last one
    dated strictly before the cutoff date.
    """
    import numpy as np

    events = universe()
    panel = pd.read_parquet(CACHE / "short_volume.parquet")
    panel = panel.sort_values(["ticker", "date"])
    panel["ratio"] = panel.short_volume / panel.total_volume.replace(0, np.nan)
    panel.loc[~np.isfinite(panel.ratio) | (panel.ratio > 1.0), "ratio"] = np.nan

    grouped = panel.groupby("ticker", sort=False).ratio
    for window in (5, 10, 21, 63):
        panel[f"m{window}"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=max(3, w // 2)).mean()
        )
    panel["sd63"] = grouped.transform(
        lambda s: s.rolling(63, min_periods=30).std()
    )
    # Dollar-weighted short flow relative to the symbol's own trading: how many
    # days of average volume were sold short in excess of the usual rate.
    panel["excess_short_shares"] = (panel.ratio - panel.m63) * panel.total_volume
    panel["excess_5d"] = panel.groupby("ticker", sort=False).excess_short_shares.transform(
        lambda s: s.rolling(5, min_periods=3).sum()
    )
    panel["advol_63"] = panel.groupby("ticker", sort=False).total_volume.transform(
        lambda s: s.rolling(63, min_periods=30).mean()
    )

    keys = panel[
        ["ticker", "date", "m5", "m10", "m21", "m63", "sd63", "excess_5d", "advol_63"]
    ].sort_values("date")
    left = events.sort_values("cutoff_date")
    joined = pd.merge_asof(
        left,
        keys,
        left_on="cutoff_date",
        right_on="date",
        by="ticker",
        direction="backward",
        allow_exact_matches=False,
    )

    out = pd.DataFrame({"event_id": joined.event_id})
    out["sv_5d_vs_63d"] = joined.m5 - joined.m63
    out["sv_10d_vs_63d"] = joined.m10 - joined.m63
    out["sv_21d_vs_63d"] = joined.m21 - joined.m63
    out["sv_5d_z"] = (joined.m5 - joined.m63) / joined.sd63.replace(0, np.nan)
    out["sv_excess_5d_over_adv"] = joined.excess_5d / joined.advol_63.replace(0, np.nan)
    out["_sv_lag_days"] = (joined.cutoff_date - joined.date).dt.days
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# Form 4 — insider transactions
# --------------------------------------------------------------------------
#
# Agent B owns the EDGAR parse and had not published a `data/edgar/form4*` table
# when this ran, so this is the "minimal version, and say so" branch. It does
# reuse B's `data/edgar/submissions/` cache read-only, which is where the
# `acceptanceDateTime` per accession comes from — the only timestamp that may be
# used for cutoff filtering, since `filingDate` rolls to the next business day
# for anything accepted after 17:30 ET.
#
# Only transaction codes **P** (open-market purchase) and **S** (open-market
# sale) are kept. Code A (grant/award) is booked at price 0, is a compensation
# event rather than a view, and dominates by count.

FORM4_DOC = "https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{acc}.txt"
SEC_UA = "explaining-markets research danishtaher7@gmail.com"

_TXN = None


def _ticker_cik() -> dict[str, str]:
    import gzip
    import json

    path = ROOT / "data" / "edgar" / "ref" / "company_tickers.json.gz"
    raw = json.load(gzip.open(path))
    records = list(raw.values()) if isinstance(raw, dict) else raw
    return {r["ticker"]: str(r["cik_str"]).zfill(10) for r in records}


def form4_index(window_days: int = 90) -> pd.DataFrame:
    """`event_id, cik, accession, acceptance_et` for every Form 4 in-window.

    In-window means ``acceptanceDateTime`` strictly before the event's cutoff and
    within `window_days` of it. The cutoff test is on acceptance, never on
    ``filingDate`` or on the transaction date inside the document.
    """
    import gzip
    import json

    events = universe()
    events["cik"] = events.ticker.map(_ticker_cik())
    subs = ROOT / "data" / "edgar" / "submissions"
    events = events[events.cik.notna()]
    events = events[[(subs / f"CIK{c}.json.gz").exists() for c in events.cik]]

    rows = []
    for cik, group in events.groupby("cik"):
        recent = json.load(gzip.open(subs / f"CIK{cik}.json.gz"))["filings"]["recent"]
        frame = pd.DataFrame(
            {
                "form": recent["form"],
                "accession": recent["accessionNumber"],
                "raw": recent["acceptanceDateTime"],
            }
        )
        frame = frame[frame.form == "4"]
        if frame.empty:
            continue
        # EDGAR serves acceptanceDateTime in ET with a spurious trailing Z.
        stamp = pd.to_datetime(frame.raw).dt.tz_localize(None)
        frame = frame.assign(
            acceptance_et=stamp.dt.tz_localize(
                "America/New_York", ambiguous=True, nonexistent="shift_forward"
            )
        )
        for event_id, cutoff in zip(group.event_id, group.cutoff_et):
            window = frame[
                (frame.acceptance_et < cutoff)
                & (frame.acceptance_et >= cutoff - pd.Timedelta(days=window_days))
            ]
            for accession, acceptance in zip(window.accession, window.acceptance_et):
                rows.append((event_id, cik, accession, acceptance))
    return pd.DataFrame(rows, columns=["event_id", "cik", "accession", "acceptance_et"])


def _parse_form4(text: str) -> list[dict]:
    """Non-derivative P and S transactions out of a Form 4 submission text file."""
    import re

    global _TXN
    if _TXN is None:
        _TXN = re.compile(
            r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", re.S
        )

    def one(block: str, tag: str) -> str | None:
        m = re.search(rf"<{tag}>\s*<value>(.*?)</value>", block, re.S)
        return m.group(1).strip() if m else None

    out = []
    for block in _TXN.findall(text):
        code = re.search(r"<transactionCode>(.*?)</transactionCode>", block)
        code = code.group(1).strip() if code else None
        if code not in ("P", "S"):
            continue
        shares = one(block, "transactionShares")
        price = one(block, "transactionPricePerShare")
        side = one(block, "transactionAcquiredDisposedCode")
        try:
            shares = float(shares)
        except (TypeError, ValueError):
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = float("nan")
        sign = 1.0 if (side == "A" or code == "P") else -1.0
        out.append({"code": code, "shares": sign * shares, "price": price})
    return out


def fetch_form4(accessions: pd.DataFrame, *, workers: int = 6) -> pd.DataFrame:
    """Download and parse a set of Form 4s. `accessions` needs `cik`, `accession`."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    path = CACHE / "form4_txn.parquet"
    done = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["accession"])
    todo = accessions[~accessions.accession.isin(set(done.accession))]
    todo = todo.drop_duplicates("accession")
    print(f"form 4: {len(todo):,} to fetch, {done.accession.nunique():,} cached")
    if todo.empty:
        return done

    session = requests.Session()
    session.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})
    gate, last = threading.Lock(), [0.0]

    def throttle():
        with gate:
            wait = 0.11 - (time.monotonic() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()

    def one(row):
        nodash = row.accession.replace("-", "")
        url = FORM4_DOC.format(cik=int(row.cik), nodash=nodash, acc=row.accession)
        throttle()
        try:
            response = session.get(url, timeout=30)
            if response.status_code != 200:
                return []
            txns = _parse_form4(response.text)
        except Exception:
            return []
        return [
            {"accession": row.accession, "cik": row.cik, **t} for t in txns
        ] or [{"accession": row.accession, "cik": row.cik, "code": None,
               "shares": float("nan"), "price": float("nan")}]

    results = []
    with ThreadPoolExecutor(workers) as pool:
        for i, batch in enumerate(pool.map(one, todo.itertuples())):
            results.extend(batch)
            if i and i % 5000 == 0:
                print(f"  {i:,}/{len(todo):,}", flush=True)

    fetch(
        source="edgar_form4",
        endpoint="Archives/edgar/data/*/*.txt",
        loader=lambda: results,
        point_in_time=True,
        notes=(
            f"{len(todo):,} Form 4 submission text files; each accession was "
            "selected only if its acceptanceDateTime is strictly before the "
            "event's knowledge_cutoff. Codes P and S retained, A/M/F/G dropped."
        ),
    )
    out = pd.concat([done, pd.DataFrame(results)], ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def form4_features(window_days: int = 90) -> pd.DataFrame:
    """Net insider open-market buying in the window before the cutoff.

    Scaled three ways, because none of them is obviously right and the scaling is
    a normaliser, not the signal: by gross traded shares (a −1..1 net-buy ratio),
    by count of transactions, and raw dollar net.
    """
    import numpy as np

    index = form4_index(window_days)
    txn = pd.read_parquet(CACHE / "form4_txn.parquet")
    txn = txn[txn.code.notna()]
    merged = index.merge(txn, on="accession", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["event_id"])
    merged["dollars"] = merged.shares * merged.price

    grouped = merged.groupby("event_id")
    out = pd.DataFrame(
        {
            "f4_net_share_ratio": grouped.shares.sum() / grouped.shares.apply(lambda s: s.abs().sum()),
            "f4_net_dollars": grouped.dollars.sum(),
            "f4_net_count": grouped.code.apply(lambda s: (s == "P").sum() - (s == "S").sum()),
            "f4_n_txn": grouped.code.size(),
        }
    ).reset_index()
    out["f4_net_dollar_ratio"] = out.f4_net_dollars / grouped.dollars.apply(
        lambda s: s.abs().sum()
    ).reindex(out.event_id).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------
# The block
# --------------------------------------------------------------------------


def _register():
    """Declare the block. Lives here rather than in `runner/blocks.py` because
    that file is shared across six agents and is not mine to edit."""
    from runner.blocks import BLOCKS, Block, Feature, register

    if "flow" in BLOCKS:
        return BLOCKS["flow"]
    return register(
        Block(
            name="flow",
            owner="Agent F — ownership and flow",
            notes=(
                "FINRA consolidated short interest (bi-monthly, filtered on a "
                "*derived* publication date, settlement + 9 business days) and "
                "FINRA daily consolidated short-sale volume (last file strictly "
                "before the cutoff date). No Form 4 and no 13F: see "
                "data/blocks/flow.report.md."
            ),
            features=[
                Feature("si_chg_1", description="log change in shares short, one vintage (~15d)"),
                Feature("si_chg_3", description="log change in shares short, three vintages (~45d)"),
                Feature("si_chg_6", description="log change in shares short, six vintages (~90d)"),
                Feature("si_chg_1_over_adv", description="one-vintage change in shares short, in days of ADV"),
                Feature("si_chg_3_over_adv", description="three-vintage change in shares short, in days of ADV"),
                Feature(
                    "si_days_to_cover",
                    description=(
                        "short interest level in days of ADV. A level, but a "
                        "signed one — the short-interest literature predicts "
                        "high SI -> low subsequent returns — so not a "
                        "magnitude-without-direction characteristic."
                    ),
                ),
                Feature("sv_5d_vs_63d", description="5d mean short-volume ratio minus own 63d mean"),
                Feature("sv_10d_vs_63d", description="10d mean short-volume ratio minus own 63d mean"),
                Feature("sv_21d_vs_63d", description="21d mean short-volume ratio minus own 63d mean"),
                Feature("sv_5d_z", description="(5d - 63d) short-volume ratio over own 63d sd"),
                Feature("sv_excess_5d_over_adv", description="5d shares sold short above the 63d rate, in days of ADV"),
                Feature(
                    "flow_combined",
                    description=(
                        "equal-weight mean of the within-quarter percentile "
                        "ranks of si_chg_3, si_chg_6, sv_excess_5d_over_adv, "
                        "sv_21d_vs_63d, sv_10d_vs_63d, negated. Unfitted, but "
                        "the five were chosen after seeing dev rho — believe "
                        "the confirmation-partition number, not the dev one. "
                        "The within-quarter rank is a cross-sectional transform "
                        "that is not reproducible live; it is monotone, so it "
                        "changes nothing for a single feature, but for this "
                        "average it is an approximation."
                    ),
                    point_in_time=False,
                ),
            ],
        )
    )


def build_block() -> pd.DataFrame:
    """Assemble and persist `data/blocks/flow.parquet`.

    `flow_combined` is an equal-weight average of the within-quarter percentile
    ranks of the five features whose sign agreed on the dev quarters, negated so
    that "more short pressure building" reads low. It is **not fitted** — no
    weights are estimated from `y` — but the choice of which five to average was
    made after seeing dev ρ, so its dev number is optimistic and its
    confirmation-partition number is the one to believe.
    """
    import numpy as np

    import harness

    frame = short_interest_features().drop(columns=["_stale_days"])
    frame = frame.merge(
        short_volume_features().drop(columns=["_sv_lag_days"]), on="event_id"
    )

    parts = [
        "si_chg_3",
        "si_chg_6",
        "sv_excess_5d_over_adv",
        "sv_21d_vs_63d",
        "sv_10d_vs_63d",
    ]
    base = pd.concat([harness.load(q) for q in harness.QUARTERS], ignore_index=True)
    ranked = base[["event_id", "quarter"]].merge(frame, on="event_id", how="left")
    for column in parts:
        ranked[column] = ranked.groupby("quarter")[column].rank(pct=True) - 0.5
    ranked["flow_combined"] = -ranked[parts].mean(axis=1)
    frame = frame.merge(ranked[["event_id", "flow_combined"]], on="event_id", how="left")

    from runner import blocks as blocks_module

    path = blocks_module.write("flow", frame.replace([np.inf, -np.inf], np.nan))
    print(f"wrote {path}  {len(frame):,} events, {len(frame.columns) - 1} features")
    return frame


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what == "form4":
        # A stratified sample of dev events, not all 7,840: the full 90-day
        # index is 102k accessions and the SEC's 10 req/s ceiling makes that
        # ~2.8 hours. n=2,500 puts the standard error near 0.020, which settles
        # "is this 0.15?" without settling "is this 0.04 or 0.00?".
        import harness

        dev = pd.concat([harness.load(q) for q in harness.DEV_QUARTERS])
        sample = set(
            dev.groupby("quarter", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 834), random_state=0))
            .event_id
        )
        index = form4_index(90)
        fetch_form4(index[index.event_id.isin(sample)])
        raise SystemExit

    if what == "block":
        build_block()
        block = _register()
        print(f"registered block {block.name!r}: {len(block.features)} features, "
              f"{len(block.signal_columns())} directional")
        raise SystemExit
    if what in ("si", "both"):
        fetch_short_interest("2025-06-30")
    if what in ("sv", "both"):
        fetch_short_volume("2025-08-01", "2026-08-07")
