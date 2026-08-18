"""The contract six parallel data-source agents write against, and the scorer that composes them.

A **block** is one source's contribution: a table with `event_id` and one column
per feature, covering as many of the 8,020 archive events as the source allows.
Blocks do not compete. They are columns in a shared matrix, and the quantity that
decides the cycle is the **ρ_b matrix across blocks**, because two blocks at
ρ 0.12 that are uncorrelated beat one block at ρ 0.20 that duplicates the read.

The arithmetic that sets the bar, restated so it is in the code and not only in
a brief. Combining k channels each at correlation ρ with mutual correlation ρ_b
asymptotes to ρ/√ρ_b. Four channels at ρ 0.15 and mutual ρ_b 0.10 combine to
0.263; adding the LLM read at ρ_b 0.05 gives **0.322, which is 9.8% of
obtainable**. Five such channels clear 11%.

**So a channel is interesting at ρ ≥ 0.15.** Our peer arm measured ρ 0.047 at
ρ_b 0.051 — the most decorrelated number ever produced here and still worth
nothing, because decorrelation without signal buys nothing. Every block reports
against 0.15, not against zero.

## What a block author must produce

1. `data/blocks/<name>.parquet` — `event_id` plus float feature columns. Missing
   is `NaN`, never 0 and never a neutral: a hole is a coverage fact and filling
   it launders coverage into a measured result.
2. A registration in :data:`BLOCKS` giving, per feature, whether it is
   point-in-time and whether it is directional.
3. Every external retrieval through `sources.fetch`, which enforces the cutoff
   and writes the audit log the organisers may demand under §07/§10.

## The cutoff rule for daily bars, stated once

`knowledge_cutoff` is 16:00 ET on 8,020 of 8,020 events — the closing bell.
Measured: the organisers' own `metrics.earnings_surprise.price_date` equals the
cutoff date on 7,611/8,005 events, so **the daily bar dated the cutoff date is
in-bounds and they use it themselves.**

That bar's information is complete *at* the cutoff instant, and `check_window`
rejects `window_end >= cutoff` by design — correctly, since a window ending
after the cutoff could contain the announcement. Use :func:`price_window`, which
resolves the ambiguity in one place rather than in six agents' heads: the audit
record ends one second before the cutoff, and the last bar included is the one
dated the cutoff date in **America/New_York**, never a later one.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402
import harness  # noqa: E402

BLOCK_DIR = ROOT / "data" / "blocks"

#: A channel below this is not worth combining. See the module docstring.
RHO_BAR = 0.15

#: Mean 1 − R²_surprise over the dev quarters; converts ρ to the scorer's scale.
OBTAINABLE = 0.9413


@dataclass
class Feature:
    """One column, with the two properties that decide whether it may ship."""

    name: str
    #: False for anything the source serves as a *current* value — yfinance's
    #: market cap, shares outstanding, sector, short interest. Usable in research
    #: as a flagged approximation; **not shippable**, and must be rebuilt
    #: point-in-time before it goes near a live prediction.
    point_in_time: bool = True
    #: False for magnitude-without-direction. Volatility, beta, dispersion,
    #: implied move, |surprise|, disclosure length. ΔR² is a linear fit, so these
    #: contribute ~0 — measured three independent ways. They may appear as
    #: normalisers; they may not be registered as signals.
    directional: bool = True
    description: str = ""


@dataclass
class Block:
    """One source's feature table."""

    name: str
    owner: str
    features: list[Feature] = field(default_factory=list)
    notes: str = ""

    @property
    def path(self) -> Path:
        return BLOCK_DIR / f"{self.name}.parquet"

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["event_id"])
        return pd.read_parquet(self.path)

    def signal_columns(self) -> list[str]:
        """Directional features only — the ones allowed to be scored as signals."""
        return [f.name for f in self.features if f.directional]


BLOCKS: dict[str, Block] = {}


def register(block: Block) -> Block:
    if block.name in BLOCKS:
        raise ValueError(f"block {block.name!r} already registered")
    BLOCKS[block.name] = block
    return block


def write(name: str, frame: pd.DataFrame) -> Path:
    """Persist a block's table. `event_id` must be present and unique."""
    if "event_id" not in frame.columns:
        raise ValueError("a block table needs an event_id column")
    if frame.event_id.duplicated().any():
        raise ValueError("event_id must be unique in a block table")
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = BLOCK_DIR / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------
# The cutoff rule for price-like windows
# --------------------------------------------------------------------------


def price_window(knowledge_cutoff, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """`(fetch_start, audit_end, last_bar_date)` for a daily-bar lookback.

    `audit_end` is one second before the cutoff, which is what goes in the
    `sources.fetch` window so the record is unambiguous. `last_bar_date` is the
    calendar date in America/New_York whose close is the last observable price —
    the same bar the organisers price their own surprise metric off.

    Drop every bar dated after `last_bar_date`. Do not "be conservative" by
    dropping one more day: the measured gap between cutoff and announcement is
    **bimodal** (3,963 events under 3h, 3,481 at 3–24h, 576 over 24h), so any
    fixed offset from the announcement is wrong for one of the two modes. The
    cutoff itself is the only correct boundary.
    """
    cutoff = pd.Timestamp(knowledge_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    last_bar = cutoff.tz_convert("America/New_York").normalize().date()
    return (
        cutoff - pd.Timedelta(days=lookback_days),
        cutoff - pd.Timedelta(seconds=1),
        pd.Timestamp(last_bar),
    )


# --------------------------------------------------------------------------
# Scoring — per feature, per block, and across blocks
# --------------------------------------------------------------------------


def _frame(quarters: Sequence[str]) -> pd.DataFrame:
    return pd.concat([harness.load(q) for q in quarters], ignore_index=True)


def score_features(
    table: pd.DataFrame,
    columns: Sequence[str] | None = None,
    quarters: Sequence[str] = tuple(harness.DEV_QUARTERS),
    event_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-feature ρ against the ≥0.15 bar. **No fitting** — this is the honest number.

    ρ is the partial correlation with `y` controlling for `surprise_pct`,
    computed per quarter and pooled in Fisher-z. A raw feature needs no training
    set, so nothing here can overfit, which is exactly why it is the number to
    look at first. Block-level fitted scores come later and are cross-validated.
    """
    base = _frame(quarters)
    if event_ids is not None:
        base = base[base.event_id.isin(set(event_ids))]
    merged = base.merge(table, on="event_id", how="left")
    columns = list(columns) if columns is not None else [
        c for c in table.columns if c != "event_id" and pd.api.types.is_numeric_dtype(table[c])
    ]

    rows = []
    for column in columns:
        per_quarter = []
        for quarter, group in merged.groupby("quarter"):
            values = group[column].to_numpy(dtype=float)
            if np.isfinite(values).sum() < 100:
                continue
            per_quarter.append(
                {
                    "rho": E.partial_corr(
                        values,
                        group.y.to_numpy(dtype=float),
                        group.surprise_pct.to_numpy(dtype=float),
                    ),
                    "rho_b": _rho_b_champion(values, group),
                    "n": int(np.isfinite(values).sum()),
                }
            )
        if not per_quarter:
            rows.append({"feature": column, "rho": np.nan, "rho_b_champion": np.nan,
                         "n": 0, "coverage": 0.0, "clears_bar": False})
            continue
        t = pd.DataFrame(per_quarter)
        rho = _pool(t.rho)
        rows.append(
            {
                "feature": column,
                "rho": rho,
                "abs_rho": abs(rho),
                "rho_b_champion": _pool(t.rho_b),
                "n": int(t.n.sum()),
                "coverage": float(merged[column].notna().mean()),
                "implied_pct_obtainable": rho**2,
                "clears_bar": bool(abs(rho) >= RHO_BAR),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)


def _rho_b_champion(values: np.ndarray, group: pd.DataFrame) -> float:
    column = E.default_champion(quiet=True)
    if column not in group:
        return float("nan")
    champ = group[column].to_numpy(dtype=float)
    surprise = group.surprise_pct.to_numpy(dtype=float)
    matrix = E._correlation_matrix({"a": values, "c": champ}, surprise)
    return float(matrix.loc["a", "c"])


def _pool(values) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(v):
        return float("nan")
    return float(np.tanh(np.arctanh(np.clip(v, -0.999999, 0.999999)).mean()))


def residual_matrix(
    tables: dict[str, pd.DataFrame],
    quarters: Sequence[str] = tuple(harness.DEV_QUARTERS),
) -> pd.DataFrame:
    """ρ_b between every pair of named columns, surprise projected out, Fisher-z pooled.

    Pass `{f"{block}.{feature}": table}` entries, or whole blocks reduced to one
    column each. This is the matrix the cycle turns on — report it **before**
    fitting anything, per the dispatch's ordering.
    """
    base = _frame(quarters)
    champion = E.default_champion(quiet=True)
    stack, names = [], sorted(tables)
    for quarter, group in base.groupby("quarter"):
        surprise = group.surprise_pct.to_numpy(dtype=float)
        columns = {}
        for name in names:
            merged = group[["event_id"]].merge(tables[name], on="event_id", how="left")
            values = merged.iloc[:, 1].to_numpy(dtype=float)
            ok = np.isfinite(values) & np.isfinite(surprise)
            resid = np.full(len(values), np.nan)
            if ok.sum() >= 30:
                resid[ok] = E._residualize(values[ok], surprise[ok])
            columns[name] = resid
        if champion in group:
            champ = group[champion].to_numpy(dtype=float)
            ok = np.isfinite(champ) & np.isfinite(surprise)
            resid = np.full(len(champ), np.nan)
            if ok.sum() >= 30:
                resid[ok] = E._residualize(champ[ok], surprise[ok])
            columns["champion"] = resid
        stack.append(pd.DataFrame(columns).corr())
    if not stack:
        return pd.DataFrame()
    all_names = sorted(set().union(*[set(s.columns) for s in stack]))
    aligned = [s.reindex(index=all_names, columns=all_names) for s in stack]
    z = np.arctanh(np.clip(np.stack([a.to_numpy(dtype=float) for a in aligned]), -0.999999, 0.999999))
    pooled = np.tanh(np.nanmean(z, axis=0))
    np.fill_diagonal(pooled, 1.0)
    return pd.DataFrame(pooled, index=all_names, columns=all_names)


def coverage_report() -> pd.DataFrame:
    """What each registered block actually delivered, against all 8,020 events."""
    total = sum(len(harness.load(q)) for q in harness.QUARTERS)
    rows = []
    for block in BLOCKS.values():
        table = block.load()
        rows.append(
            {
                "block": block.name,
                "owner": block.owner,
                "features": len(block.features),
                "directional": len(block.signal_columns()),
                "not_point_in_time": sum(1 for f in block.features if not f.point_in_time),
                "events": len(table),
                "coverage": len(table) / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import harness as H

    event = H.events_for("2026Q2")[0]
    start, audit_end, last_bar = price_window(event["knowledge_cutoff"], 60)
    print(f"event {event['event_id']}  cutoff {event['knowledge_cutoff']}")
    print(f"  fetch window  {start.isoformat()} .. {audit_end.isoformat()}")
    print(f"  last usable daily bar (America/New_York): {last_bar.date()}")

    # The boundary has to survive sources.py's own check, or six agents will
    # each discover the CutoffViolation separately and each invent a fudge.
    from sources import check_window

    check_window(event["knowledge_cutoff"], audit_end, event_id=event["event_id"])
    print("  price_window's audit end passes sources.check_window")

    cutoff = pd.Timestamp(event["knowledge_cutoff"])
    et = cutoff.tz_convert("America/New_York")
    print(f"  cutoff in ET: {et}  (expect 16:00)")
    assert (et.hour, et.minute) == (16, 0), et
    print("\nblocks registered:", sorted(BLOCKS) or "(none yet)")
