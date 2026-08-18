"""Agent C — archive outcomes: sector peers (C1) and the response regime (C2).

Zero external calls. Everything here is computed from settled outcomes already
in ``data/archive/``, so the whole block costs nothing and can be recomputed in
seconds.

Two things this module fixes relative to the existing market-wide ``peer_view``
arm in ``runner/arms_builtin.py`` (ρ 0.047):

1. **The settlement instant is read, not guessed.** ``peer_view`` approximates
   the close of a peer's return window as ``knowledge_cutoff + 1 day``. Measured
   over 8,020 archive events, the real ``event_returns[*].window_end_date`` is
   cutoff+1 calendar day on only 7,382 of them; 616 settle 2–7 days later
   (weekends, holidays) and 12 settle same-day. The +1d approximation therefore
   admits peers whose window had **not** closed on 616 events — a small live
   leak. Here the peer's settlement instant is 16:00 America/New_York on its own
   ``window_end_date``, and the filter is ``peer_settled <= our cutoff``.

2. **Nothing aggregates ``y``.** ``y`` is a percentile rank *within the full
   quarter* and does not exist until the quarter closes. Peer aggregates use
   ``car1`` standardised by the trailing pool's own dispersion; the regime
   regression is fit on ranks computed **within the trailing settled window**,
   never on the quarter-wide ``y``/``surprise_pct`` pair.

The looser filter "the peer reported before we did" is deliberately not used:
it agrees with the correct filter on 99.3% of peer-pairs, which is exactly why
it survives review and is still wrong.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARCHIVE_DIR = ROOT / "data" / "archive"
REFERENCE_DIR = ROOT / "data" / "reference"

QUARTERS = ["2025Q4", "2026Q1", "2026Q2", "2026Q3"]


# --------------------------------------------------------------------------
# The settled-outcome pool
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def base() -> pd.DataFrame:
    """All 8,020 archive events with the facts needed for peer/regime work.

    Columns: ``event_id, ticker, quarter, cutoff, settled, car1, surprise,
    event_datetime``.

    ``settled`` is the instant the event's own return window closed: 16:00
    America/New_York on ``event_returns[ticker].window_end_date``, in UTC. That
    is the only instant at which the event may be used as an input to another
    event's feature.
    """
    rows = []
    for quarter in QUARTERS:
        path = ARCHIVE_DIR / f"EARNINGS_RELEASE_{quarter}.jsonl.gz"
        for line in gzip.open(path, "rt"):
            if not line.strip():
                continue
            record = json.loads(line)
            ticker = record["focal_assets"][0]["identifier_value"]
            returns = (record.get("event_returns") or {}).get(ticker) or {}
            metrics = (record.get("metrics") or {}).get("earnings_surprise") or {}
            rows.append(
                {
                    "event_id": record["event_id"],
                    "ticker": ticker,
                    "quarter": quarter,
                    "cutoff": record.get("knowledge_cutoff"),
                    "window_end_date": returns.get("window_end_date"),
                    "car1": returns.get("car1"),
                    "surprise": metrics.get("surprise"),
                    "event_datetime": record.get("event_datetime"),
                }
            )
    frame = pd.DataFrame(rows)
    frame["cutoff"] = pd.to_datetime(frame.cutoff, utc=True)
    frame["event_datetime"] = pd.to_datetime(frame.event_datetime, utc=True)
    # 16:00 America/New_York on the window's last date, expressed in UTC. Doing
    # this through the tz database rather than a fixed -4h keeps the two DST
    # boundaries inside the archive (Nov 2025, Mar 2026) correct.
    end = pd.to_datetime(frame.window_end_date)
    settled = (end + pd.Timedelta(hours=16)).dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    )
    frame["settled"] = settled.dt.tz_convert("UTC")
    frame["car1"] = pd.to_numeric(frame.car1, errors="coerce")
    frame["surprise"] = pd.to_numeric(frame.surprise, errors="coerce")
    return frame.sort_values("cutoff").reset_index(drop=True)


@lru_cache(maxsize=1)
def pool() -> pd.DataFrame:
    """Usable-as-an-input events: finite ``car1`` and a known settlement instant.

    Sorted by ``settled`` so a trailing window is two ``searchsorted`` calls.
    """
    frame = base()
    ok = frame.car1.notna() & frame.settled.notna()
    out = frame[ok].sort_values("settled").reset_index(drop=True)
    out["settled_ns"] = _ns(out.settled)
    return out


def _ns(series: pd.Series) -> np.ndarray:
    """Integer **nanoseconds** since epoch.

    Not ``.astype("int64")``: pandas 2 keeps these columns at microsecond
    resolution, so the naive cast silently returns microseconds and every
    ``searchsorted`` window becomes 1000× too wide. That bug produced a first
    regime table where the "21-day" pool was the entire archive.
    """
    return series.dt.as_unit("ns").astype("int64").to_numpy()


def _rank01(values: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1), ties averaged — the scorer's own transform."""
    order = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    return order / (len(values) + 1.0)


# --------------------------------------------------------------------------
# C2 — the response regime
# --------------------------------------------------------------------------

#: Trailing window for the "current" response function, in days.
SHORT_DAYS = 21
#: Trailing window for the reference response function it is compared against.
LONG_DAYS = 90
#: Below this many settled events a slope estimate is noise, and the feature is
#: a hole rather than a number.
MIN_FIT = 60


def regime_table(
    short_days: int = SHORT_DAYS,
    long_days: int = LONG_DAYS,
    min_fit: int = MIN_FIT,
) -> pd.DataFrame:
    """Per-event features describing *the current mapping from surprise to rank*.

    For each focal event with cutoff ``C`` the trailing pool is every settled
    event with ``settled <= C``. Within that pool, ``car1`` and ``surprise`` are
    ranked **locally** — this is the point-in-time stand-in for the quarter-wide
    ``y`` / ``surprise_pct`` pair, which does not exist until the quarter closes
    and must never be aggregated. The local rank pair is regressed to get the
    slope of the response function now, and the same is done over a longer
    window to get the reference the short slope is compared to.
    """
    p = pool()
    settled_ns = p.settled_ns.to_numpy()
    car1 = p.car1.to_numpy(dtype=float)
    surprise = p.surprise.to_numpy(dtype=float)

    events = base()
    cut_ns = _ns(events.cutoff)
    own_surprise = events.surprise.to_numpy(dtype=float)
    day = np.int64(86_400_000_000_000)

    quarter_start = events.groupby("quarter").cutoff.transform("min")
    season = ((events.cutoff - quarter_start).dt.total_seconds() / 86_400.0).to_numpy()

    out = {k: np.full(len(events), np.nan) for k in (
        "regime_slope", "regime_slope_delta", "regime_level", "regime_resid_sd",
        "regime_season_day", "regime_x_slope_delta", "regime_x_slope",
        "regime_n_fit",
    )}

    # Cache fits by (lo, hi) index pair: cutoffs repeat heavily (16:00 ET on a
    # few hundred distinct dates), so this collapses ~8k regressions to ~600.
    cache: dict[tuple[int, int, int], tuple] = {}

    def fit(lo: int, hi: int) -> tuple[float, float, float, int]:
        n = hi - lo
        if n < min_fit:
            return (np.nan, np.nan, np.nan, n)
        y_r = _rank01(car1[lo:hi])
        s = surprise[lo:hi]
        ok = np.isfinite(s)
        if ok.sum() < min_fit:
            return (np.nan, np.nan, np.nan, int(ok.sum()))
        s_r = np.full(ok.sum(), np.nan)
        s_r = _rank01(s[ok])
        yy = y_r[ok]
        slope, intercept = np.polyfit(s_r, yy, 1)
        resid = yy - (slope * s_r + intercept)
        return (float(slope), float(intercept), float(resid.std(ddof=1)), int(ok.sum()))

    # 12 of 8,020 events have `settled <= cutoff` — their own return window
    # closes at the same 16:00 ET instant as their cutoff. Without this they
    # would sit in their own trailing pool, which is the purest form of the leak
    # this module exists to avoid, so they are dropped to NaN rather than fudged.
    position = pd.Series(np.arange(len(p)), index=p.event_id.to_numpy())
    self_pos = events.event_id.map(position).to_numpy(dtype=float)
    self_settled = events.settled.notna().to_numpy() & (
        events.settled.fillna(events.cutoff) <= events.cutoff
    ).to_numpy()

    for i in range(len(events)):
        c = cut_ns[i]
        hi = int(np.searchsorted(settled_ns, c, side="right"))
        lo_s = int(np.searchsorted(settled_ns, c - short_days * day, side="left"))
        lo_l = int(np.searchsorted(settled_ns, c - long_days * day, side="left"))
        if self_settled[i] and np.isfinite(self_pos[i]) and lo_l <= self_pos[i] < hi:
            continue

        key = (lo_s, hi, short_days)
        if key not in cache:
            cache[key] = fit(lo_s, hi)
        slope_s, _int_s, sd_s, n_s = cache[key]

        key_l = (lo_l, hi, long_days)
        if key_l not in cache:
            cache[key_l] = fit(lo_l, hi)
        slope_l, _int_l, _sd_l, _n_l = cache[key_l]

        out["regime_slope"][i] = slope_s
        out["regime_slope_delta"][i] = slope_s - slope_l
        out["regime_resid_sd"][i] = sd_s
        out["regime_n_fit"][i] = n_s
        out["regime_season_day"][i] = season[i]

        if hi - lo_s >= min_fit:
            w = car1[lo_s:hi]
            sd = w.std(ddof=1)
            out["regime_level"][i] = float(w.mean() / sd) if sd else np.nan

        # Where does *this* surprise sit in the distribution the recent regime
        # was fit on? Centred, so the interaction is signed the way the
        # hypothesis says: a steeper-than-usual response pushes big beats up and
        # big misses down, and leaves the middle alone.
        if np.isfinite(own_surprise[i]) and hi - lo_l >= min_fit:
            ref = surprise[lo_l:hi]
            ref = ref[np.isfinite(ref)]
            if len(ref) >= min_fit:
                q = float((ref < own_surprise[i]).mean()) - 0.5
                out["regime_x_slope_delta"][i] = q * (slope_s - slope_l)
                out["regime_x_slope"][i] = q * slope_s

    table = pd.DataFrame(out)
    table.insert(0, "event_id", events.event_id.to_numpy())
    return table


# --------------------------------------------------------------------------
# C1 — sector-relative peers
# --------------------------------------------------------------------------


def sic_map() -> pd.DataFrame | None:
    """Agent B's ``data/reference/sic.parquet``, or ``None`` if not delivered."""
    path = REFERENCE_DIR / "sic.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def peer_table(
    codes: pd.Series | None = None,
    windows: tuple[int, ...] = (10, 30, 90),
    widths: tuple[int, ...] = (4, 3, 2),
    min_peers: int = 3,
    label: str = "sic",
) -> pd.DataFrame:
    """Sector-relative peer reaction, at several cohort widths and window lengths.

    ``codes`` maps ticker to an industry string; ``widths`` are prefix lengths
    taken off it, so 4/3/2 give major-group narrowing on SIC. Passing a
    non-SIC grouping (a text cluster, say) with ``widths=(9,)`` runs the same
    machinery on that cohort definition — which is how the channel's ceiling was
    probed before Agent B delivered.

    Every aggregate is over ``car1`` divided by a **single trailing market-wide
    dispersion**, not by the cohort's own sd. A four-name cohort's sd is itself
    an estimate with about 40% error, and dividing by it turns a thin cohort's
    noise into a large number exactly when the cohort is least trustworthy.
    Nothing here aggregates ``y``.
    """
    if codes is None:
        sic = sic_map()
        if sic is None:
            raise FileNotFoundError(f"{REFERENCE_DIR / 'sic.parquet'} not delivered yet")
        codes = sic.drop_duplicates("ticker").set_index("ticker").sic.astype("string")

    events = base().copy()
    events["code"] = events.ticker.str.upper().map(codes).astype("string")
    p = pool().copy()
    p["code"] = p.ticker.str.upper().map(codes).astype("string")

    day = np.int64(86_400_000_000_000)
    cut_ns = _ns(events.cutoff)
    settled_all = p.settled_ns.to_numpy()
    car_all = p.car1.to_numpy(dtype=float)
    ids_all = p.event_id.to_numpy()
    out: dict[str, np.ndarray] = {}

    # One trailing normaliser for everything: the sd of car1 over the last 90
    # days of settled events. Point-in-time, stable, and shared across cohort
    # widths so the widths stay comparable to each other.
    norm = np.full(len(events), np.nan)
    market = {w: np.full(len(events), np.nan) for w in windows}
    for i, c in enumerate(cut_ns):
        hi = int(np.searchsorted(settled_all, c, side="right"))
        lo90 = int(np.searchsorted(settled_all, c - 90 * day, side="left"))
        if hi - lo90 >= 30:
            sd = car_all[lo90:hi].std(ddof=1)
            norm[i] = sd if sd else np.nan
        for w in windows:
            lo = int(np.searchsorted(settled_all, c - w * day, side="left"))
            if hi - lo >= min_peers and np.isfinite(norm[i]):
                market[w][i] = float(car_all[lo:hi].mean() / norm[i])
    for w in windows:
        out[f"peer_mkt_w{w}"] = market[w]

    for width in widths:
        groups = {
            code: g.sort_values("settled_ns").reset_index(drop=True)
            for code, g in p.assign(_k=p.code.str[:width]).groupby("_k", dropna=True)
        }
        ev_key = events.code.str[:width].to_numpy()
        ev_ids = events.event_id.to_numpy()
        for w in windows:
            mean = np.full(len(events), np.nan)
            npeer = np.full(len(events), np.nan)
            recent = np.full(len(events), np.nan)
            beat = np.full(len(events), np.nan)
            miss = np.full(len(events), np.nan)
            for i, c in enumerate(cut_ns):
                g = groups.get(ev_key[i])
                if g is None or not np.isfinite(norm[i]):
                    continue
                ns = g.settled_ns.to_numpy()
                hi = int(np.searchsorted(ns, c, side="right"))
                lo = int(np.searchsorted(ns, c - w * day, side="left"))
                chunk = g.car1.to_numpy(dtype=float)[lo:hi]
                surp = g.surprise.to_numpy(dtype=float)[lo:hi]
                keep = g.event_id.to_numpy()[lo:hi] != ev_ids[i]
                chunk, surp = chunk[keep], surp[keep]
                npeer[i] = len(chunk)
                if len(chunk) < min_peers:
                    continue
                z = chunk / norm[i]
                mean[i] = float(z.mean())
                recent[i] = float(z[-1])
                up = np.isfinite(surp) & (surp > 0)
                dn = np.isfinite(surp) & (surp <= 0)
                if up.sum() >= min_peers:
                    beat[i] = float(z[up].mean())
                if dn.sum() >= min_peers:
                    miss[i] = float(z[dn].mean())
            tag = f"{label}{width}_w{w}"
            out[f"peer_{tag}_mean"] = mean
            out[f"peer_{tag}_n"] = npeer
            out[f"peer_{tag}_excess"] = mean - market[w]
            out[f"peer_{tag}_recent"] = recent
            out[f"peer_{tag}_beat"] = beat
            out[f"peer_{tag}_miss"] = miss
            out[f"peer_{tag}_spread"] = beat - miss

    table = pd.DataFrame(out)
    table.insert(0, "event_id", events.event_id.to_numpy())
    return table



# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def register_blocks() -> None:
    """Declare both tables to :mod:`runner.blocks`. Idempotent."""
    from runner.blocks import BLOCKS, Block, Feature, register

    if "regime" not in BLOCKS:
        register(
            Block(
                name="regime",
                owner="agent-c",
                features=[
                    Feature("regime_slope", True, True,
                            "slope of local car1-rank on local surprise-rank, 21d trailing"),
                    Feature("regime_slope_delta", True, True,
                            "21d slope minus 90d slope — how the response function has moved"),
                    Feature("regime_level", True, True,
                            "mean car1 / dispersion over the 21d settled pool"),
                    Feature("regime_x_slope", True, True,
                            "own centred surprise rank x current slope"),
                    Feature("regime_x_slope_delta", True, True,
                            "own centred surprise rank x slope change — the feature the "
                            "drift hypothesis actually predicts. Measured 0.009 on the "
                            "4,144-event confirmation partition."),
                    Feature("regime_season_day", True, True,
                            "days from the quarter's first cutoff"),
                    Feature("regime_resid_sd", True, False,
                            "residual dispersion of the local fit — MAGNITUDE, normaliser only"),
                    Feature("regime_n_fit", True, False,
                            "events in the 21d pool — a coverage/confidence quantity"),
                ],
                notes=(
                    "Measured dead. The response function does drift — weekly slope of y "
                    "on surprise_pct has sd 0.156 within a quarter against a mean of 0.21 "
                    "— but the drift does not price. A leave-one-out ORACLE that fits the "
                    "weekly slope using the quarter's own future events scores rho 0.020 "
                    "for the interaction and 0.033 for the level, at n=5,965 where the "
                    "standard error is 0.013. That is the ceiling of the whole channel and "
                    "it sits an order of magnitude under the 0.15 bar, so no better "
                    "estimator can rescue it."
                ),
            )
        )

    if "peers" not in BLOCKS:
        register(
            Block(
                name="peers",
                owner="agent-c",
                features=peer_features(),
                notes=(
                    "Measured. Best lagged column rho 0.066 dev / 0.052 confirmation "
                    "across 70 columns, against a 0.15 bar. The ceiling is known: a "
                    "leave-one-out CONTEMPORANEOUS oracle over same-day SIC-4 peers — "
                    "information no live system can have, since a same-day peer's return "
                    "window closes after our cutoff — scores rho 0.113 at n=2,082, "
                    "decaying to 0.064 same-week and 0.041 same-quarter. Sector-relative "
                    "really does beat market-wide (0.113 vs 0.043) and narrow beats broad "
                    "monotonically in SIC width, so the channel is real; it is just small, "
                    "because car1 is r_i - r_m and the common factor is already "
                    "differenced out of the target."
                ),
            )
        )


def peer_features() -> list:
    """A :class:`Feature` per column of the peer table, derived from the header."""
    from runner.blocks import Feature

    path = BLOCK_DIR_PEERS
    if not path.exists():
        return []
    columns = [c for c in pd.read_parquet(path).columns if c != "event_id"]
    out = []
    for c in columns:
        magnitude = c.endswith("_n")
        # The *cohort definition* for the txt* columns is k-means over all four
        # quarters of disclosure text, so the grouping — not the peer reactions —
        # peeks. Quasi-static and harmless in research, not shippable as is.
        # SIC is a current EDGAR attribute rather than an as-of-date one, so the
        # cohort *identity* is approximate in the mild way a sector label always
        # is. The txt* cohorts (only built when sic.parquet is absent) fit their
        # clustering across all four quarters and are a real, if small, peek.
        approximate = c.startswith("peer_txt")
        out.append(
            Feature(
                c,
                point_in_time=not approximate,
                directional=not magnitude,
                description="cohort size — a confidence quantity, not a signal"
                if magnitude
                else "mean standardised car1 of settled cohort peers",
            )
        )
    return out


BLOCK_DIR_PEERS = ROOT / "data" / "blocks" / "peers.parquet"


# --------------------------------------------------------------------------
# A cohort definition that needs nobody: disclosure-text pseudo-industries
# --------------------------------------------------------------------------


@lru_cache(maxsize=8)
def text_cohorts(k: int = 30) -> pd.Series:
    """``ticker -> cohort code``, from TF-IDF over the ten facts, k-means at ``k``.

    A stand-in for SIC that costs nothing and needs no external call: companies
    in the same industry describe the same things. Cohort *identity* is fitted
    over all four quarters of text, so it is **approximate rather than strictly
    point-in-time** — the grouping is quasi-static (a ticker's industry does not
    move), but a shippable version must fit the clustering on prior quarters
    only. Flagged as such in the block registration.

    Everything downstream of the grouping — which peers are in the window, what
    their reaction was — is strictly point-in-time regardless.
    """
    import harness
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    frame = harness.load_all(QUARTERS)[["identifier_value", "facts"]].copy()
    frame["text"] = frame.facts.map(lambda f: " ".join(f))
    pooled = frame.groupby("identifier_value").text.apply(" ".join)
    matrix = TfidfVectorizer(
        max_features=6000, stop_words="english", min_df=3, sublinear_tf=True
    ).fit_transform(pooled.values)
    labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(matrix)
    return pd.Series([f"{v:03d}" for v in labels], index=pooled.index, dtype="string")


def own_history_table(min_prior: int = 2) -> pd.DataFrame:
    """The ticker's *own* settled earnings reactions before this cutoff.

    Not a peer feature, but it comes free from the same pool and answers the
    obvious neighbouring question: does a company's past earnings-day reaction
    predict the next one? Standardised by the same trailing market dispersion.
    """
    events = base()
    p = pool()
    day = np.int64(86_400_000_000_000)
    cut_ns = _ns(events.cutoff)
    settled_all = p.settled_ns.to_numpy()
    car_all = p.car1.to_numpy(dtype=float)

    by_ticker = {t: g.sort_values("settled_ns") for t, g in p.groupby("ticker")}
    ev_ticker = events.ticker.to_numpy()
    ev_ids = events.event_id.to_numpy()

    mean = np.full(len(events), np.nan)
    last = np.full(len(events), np.nan)
    n = np.full(len(events), np.nan)
    for i, c in enumerate(cut_ns):
        hi_all = int(np.searchsorted(settled_all, c, side="right"))
        lo90 = int(np.searchsorted(settled_all, c - 90 * day, side="left"))
        if hi_all - lo90 < 30:
            continue
        sd = car_all[lo90:hi_all].std(ddof=1)
        if not sd:
            continue
        g = by_ticker.get(ev_ticker[i])
        if g is None:
            continue
        ns = g.settled_ns.to_numpy()
        hi = int(np.searchsorted(ns, c, side="right"))
        chunk = g.car1.to_numpy(dtype=float)[:hi]
        keep = g.event_id.to_numpy()[:hi] != ev_ids[i]
        chunk = chunk[keep]
        n[i] = len(chunk)
        if len(chunk) >= min_prior:
            mean[i] = float(chunk.mean() / sd)
            last[i] = float(chunk[-1] / sd)
    return pd.DataFrame(
        {
            "event_id": ev_ids,
            "peer_own_hist_mean": mean,
            "peer_own_hist_last": last,
            "peer_own_hist_n": n,
        }
    )


def build_peers() -> pd.DataFrame:
    """The full C1 table: SIC cohorts when Agent B has delivered, text cohorts always."""
    parts = [own_history_table()]
    sic = sic_map()
    if sic is not None:
        codes = sic.drop_duplicates("ticker").set_index("ticker").sic.astype("string")
        parts.append(peer_table(codes=codes, widths=(4, 3, 2), windows=(10, 30, 90),
                                label="sic"))
    else:  # pragma: no cover - only until Agent B delivers
        # Market-wide reference columns still have to exist; take them from the
        # text-cohort run, which computes the same market aggregates.
        parts.append(peer_table(codes=text_cohorts(30), widths=(3,),
                                windows=(10, 30, 90), label="txt30_"))
        parts.append(
            peer_table(codes=text_cohorts(80), widths=(3,), windows=(10, 30, 90),
                       label="txt80_").drop(columns=["peer_mkt_w10", "peer_mkt_w30",
                                                     "peer_mkt_w90"])
        )
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="event_id", how="outer")
    return out
