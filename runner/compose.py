"""Compose the six data blocks: the ρ_b matrix first, then exactly one fitted pass.

The ordering is the dispatch's and it is not arbitrary.

**Step 3 — the cross-block ρ_b matrix, reported before anything is fitted.**
Two blocks at ρ 0.12 that are uncorrelated beat one block at ρ 0.20 that
duplicates the read, and you cannot see that from any single block's report. The
matrix is the number the cycle turns on, so it gets computed and printed before
a single weight is estimated — otherwise the fit becomes the headline and the
structure it was supposed to reveal becomes a footnote.

**Step 4 — one fitted pass over the union.** One, not a sweep. Every extra
configuration raises the K that every promotion floor is scaled by, and this
loop has already measured what happens when fitted combinations are selected and
scored on the same events: six of seven archive arms went *negative* on held-out
data, with mean selection bias +0.0052. So every fitted model here is
cross-validated leave-one-quarter-out **and** scored on the 4,144-event
confirmation partition, and the confirmation number is the one that counts.

On ranking objectives, since the dispatch raises it: **NDCG-based objectives are
excluded, and the reason is structural rather than empirical.** NDCG applies a
positional discount, so it pays almost entirely for getting the top of the list
right. Our target is a *symmetric full-rank percentile* — being wrong at the 5th
percentile costs exactly what being wrong at the 95th costs, and the scorer is a
Pearson correlation over the whole cross-section. A top-heavy objective optimises
a different quantity than the one that pays. The symmetric analogue is included
instead: a rank-transformed ridge, which targets Spearman IC over the full
distribution.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402
import harness  # noqa: E402
from runner import blocks as B  # noqa: E402
from runner import schedule as S  # noqa: E402

#: Ridge strength for the fitted pass. One value, not a sweep — see the module
#: docstring on K.
ALPHA = 1.0

#: A feature present on fewer than this fraction of events is a coverage
#: artefact rather than a channel, and pooling it into a fit lets the fit learn
#: "was this feature available", which correlates with ticker size and liquidity.
MIN_COVERAGE = 0.30


# --------------------------------------------------------------------------
# Loading whatever the agents delivered
# --------------------------------------------------------------------------


def delivered() -> dict[str, pd.DataFrame]:
    """Every block table on disk, keyed by block name.

    Reads the directory rather than the registry: the six agents run in parallel
    and an agent that finished its table but not its registration should still
    have its work counted.
    """
    out = {}
    for path in sorted(B.BLOCK_DIR.glob("*.parquet")):
        table = pd.read_parquet(path)
        if "event_id" in table.columns and len(table):
            out[path.stem] = table
    return out


def feature_table(
    tables: dict[str, pd.DataFrame] | None = None,
    quarters: Sequence[str] = tuple(harness.DEV_QUARTERS),
    event_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One wide frame of every block's features, prefixed by block name."""
    tables = delivered() if tables is None else tables
    base = pd.concat([harness.load(q) for q in quarters], ignore_index=True)
    if event_ids is not None:
        base = base[base.event_id.isin(set(event_ids))].reset_index(drop=True)

    columns: dict[str, list[str]] = {}
    wide = base[["event_id", "quarter", "y", "surprise_pct"]].copy()
    champion = E.default_champion(quiet=True)
    if champion in base:
        wide["champion"] = base[champion].to_numpy(dtype=float)

    for name, table in tables.items():
        numeric = [
            c for c in table.columns
            if c != "event_id" and pd.api.types.is_numeric_dtype(table[c])
        ]
        if not numeric:
            continue
        renamed = table[["event_id", *numeric]].rename(
            columns={c: f"{name}.{c}" for c in numeric}
        )
        wide = wide.merge(renamed, on="event_id", how="left")
        columns[name] = [f"{name}.{c}" for c in numeric]
    return wide, columns


# --------------------------------------------------------------------------
# Step 3 — the matrix, before any fitting
# --------------------------------------------------------------------------


def block_composites(
    wide: pd.DataFrame, columns: dict[str, list[str]]
) -> dict[str, pd.Series]:
    """One column per block, so the cross-block matrix is block × block.

    Each block is reduced by an **equal-weight z-score sum of its directional
    features, signed by each feature's own in-quarter partial correlation**.
    Deliberately not a fitted composite: fitting inside the reduction would put
    a trained model in the middle of the diagnostic that is supposed to tell us
    whether fitting is worth doing, and the sign flip alone is enough to stop
    features cancelling each other out.
    """
    out = {}
    for name, cols in columns.items():
        usable = [c for c in cols if wide[c].notna().mean() >= MIN_COVERAGE]
        if not usable:
            continue
        total = pd.Series(0.0, index=wide.index)
        count = pd.Series(0.0, index=wide.index)
        for column in usable:
            values = wide[column].to_numpy(dtype=float)
            rho = E.partial_corr(
                values,
                wide.y.to_numpy(dtype=float),
                wide.surprise_pct.to_numpy(dtype=float),
            )
            if not np.isfinite(rho) or rho == 0:
                continue
            ok = np.isfinite(values)
            sd = values[ok].std() if ok.sum() > 2 else 0.0
            if not sd:
                continue
            z = np.where(ok, (values - values[ok].mean()) / sd, np.nan)
            signed = np.sign(rho) * z
            total = total.add(pd.Series(signed, index=wide.index), fill_value=0.0)
            count = count.add(pd.Series(np.isfinite(signed).astype(float), index=wide.index))
        composite = np.where(count > 0, total / count.replace(0, np.nan), np.nan)
        out[name] = pd.Series(composite, index=wide.index)
    return out


def matrix_report(quarters: Sequence[str] = tuple(harness.DEV_QUARTERS)) -> pd.DataFrame:
    """Print the per-block ρ table and the cross-block ρ_b matrix. Fits nothing."""
    tables = delivered()
    if not tables:
        print("no blocks delivered yet")
        return pd.DataFrame()

    wide, columns = feature_table(tables, quarters)
    composites = block_composites(wide, columns)

    print(f"\n{'=' * 96}\nBLOCKS DELIVERED ({len(tables)})\n{'=' * 96}")
    rows = []
    for name, table in tables.items():
        cols = columns.get(name, [])
        rows.append(
            {
                "block": name,
                "features": len(cols),
                "events": len(table),
                "mean_coverage": float(np.mean([wide[c].notna().mean() for c in cols])) if cols else 0.0,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\n{'=' * 96}\nPER-FEATURE rho (no fitting) vs the {B.RHO_BAR:.2f} bar\n{'=' * 96}")
    per_feature = []
    for name, table in tables.items():
        scored = B.score_features(table, quarters=quarters)
        scored.insert(0, "block", name)
        per_feature.append(scored)
    all_features = (
        pd.concat(per_feature, ignore_index=True).sort_values("abs_rho", ascending=False)
        if per_feature else pd.DataFrame()
    )
    if len(all_features):
        print(all_features.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        clears = all_features[all_features.clears_bar]
        print(
            f"\n{len(clears)}/{len(all_features)} features clear rho >= {B.RHO_BAR:.2f}"
            + (f": {', '.join(clears.feature)}" if len(clears) else
               "  — no single feature is a channel on its own.")
        )
        # se on a correlation at n is ~1/sqrt(n); print it so a reader cannot
        # mistake a small rho for a small effect.
        n = int(all_features.n.max()) if len(all_features) else 0
        if n:
            print(f"se on a correlation at n={n} is ~{1 / np.sqrt(n):.4f}; "
                  f"|rho| below ~{2 / np.sqrt(n):.4f} is not distinguishable from zero.")

    print(f"\n{'=' * 96}\nCROSS-BLOCK rho_b (surprise projected out, Fisher-z pooled)\n{'=' * 96}")
    named = {name: pd.DataFrame({"event_id": wide.event_id, name: series})
             for name, series in composites.items()}
    matrix = B.residual_matrix(named, quarters) if named else pd.DataFrame()
    if len(matrix):
        print(matrix.to_string(float_format=lambda v: f"{v:+.3f}"))
        off = matrix.where(~np.eye(len(matrix), dtype=bool))
        peers = [c for c in matrix.columns if c != "champion"]
        if len(peers) > 1:
            mutual = off.loc[peers, peers].stack().mean()
            print(f"\nmean mutual rho_b between blocks: {mutual:+.3f}")
            if "champion" in matrix.columns:
                print("rho_b vs champion: " + ", ".join(
                    f"{p} {matrix.loc[p, 'champion']:+.3f}" for p in peers))
            print(
                "\nCeiling arithmetic: k channels at correlation rho with mutual rho_b\n"
                "asymptote to rho/sqrt(rho_b). Read the table above against that, not\n"
                "against zero — a block at rho 0.12 and rho_b 0.05 outranks one at\n"
                "rho 0.20 and rho_b 0.80."
            )
    return all_features


# --------------------------------------------------------------------------
# Step 4 — one fitted pass, cross-validated and confirmed
# --------------------------------------------------------------------------


def _fit_predict(model_name: str, X_tr, y_tr, X_te):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    if model_name == "gbm":
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=3, learning_rate=0.05, random_state=0
        )
        return model.fit(X_tr, y_tr).predict(X_te)
    if model_name == "rank_ridge":
        # Symmetric full-rank objective: rank-transform the target so the fit
        # targets Spearman IC over the whole distribution. NOT NDCG — a
        # positional discount pays for the head of the list, and our percentile
        # target is symmetric, so top-heaviness optimises the wrong quantity.
        ranked = pd.Series(y_tr).rank(pct=True).to_numpy()
        return Ridge(alpha=ALPHA).fit(X_tr, ranked).predict(X_te)
    return Ridge(alpha=ALPHA).fit(X_tr, y_tr).predict(X_te)


def fitted_pass(
    models: Sequence[str] = ("ridge", "rank_ridge", "gbm"),
    quarters: Sequence[str] = tuple(harness.DEV_QUARTERS),
) -> pd.DataFrame:
    """Leave-one-quarter-out fit over the union of blocks, then confirm.

    The selection number and the confirmation number are reported side by side
    and their difference is the measured selection bias. That difference, not a
    best-of-K formula, is this project's answer to "how much of the margin is
    real" — a split needs no assumption about K, independence or normality, and
    our features are nowhere near independent.
    """
    tables = delivered()
    if not tables:
        print("no blocks delivered yet")
        return pd.DataFrame()

    confirm_ids = {e["event_id"] for v in S.confirmation_events().values() for e in v}
    wide, columns = feature_table(tables, quarters)
    usable = [
        c for cols in columns.values() for c in cols
        if wide[c].notna().mean() >= MIN_COVERAGE
    ]
    if not usable:
        print(f"no feature reaches {MIN_COVERAGE:.0%} coverage — nothing to fit")
        return pd.DataFrame()

    wide = wide.copy()
    wide["y_resid"] = harness.residualize(wide, "y")
    # Median-fill within quarter: a missing feature must not carry information
    # about the event, and dropping the row instead would silently restrict the
    # fit to the liquid, well-covered names.
    filled = wide.copy()
    for column in usable:
        filled[column] = filled.groupby("quarter")[column].transform(
            lambda s: s.fillna(s.median())
        )
        filled[column] = filled[column].fillna(0.0)

    is_confirm = filled.event_id.isin(confirm_ids).to_numpy()
    print(f"\n{'=' * 96}\nFITTED PASS — {len(usable)} features, "
          f"{int((~is_confirm).sum())} selection / {int(is_confirm.sum())} confirmation events"
          f"\n{'=' * 96}")

    rows = []
    for model_name in models:
        preds = np.full(len(filled), np.nan)
        for quarter in filled.quarter.unique():
            test = (filled.quarter == quarter).to_numpy()
            train = ~test
            if train.sum() < 200 or test.sum() < 50:
                continue
            preds[test] = _fit_predict(
                model_name,
                filled.loc[train, usable].to_numpy(dtype=float),
                filled.loc[train, "y_resid"].to_numpy(dtype=float),
                filled.loc[test, usable].to_numpy(dtype=float),
            )

        for label, mask in (("selection", ~is_confirm), ("confirmation", is_confirm)):
            sub = filled[mask]
            p = preds[mask]
            ok = np.isfinite(p)
            if ok.sum() < 200:
                continue
            rho = E.partial_corr(
                p[ok], sub.y.to_numpy(dtype=float)[ok], sub.surprise_pct.to_numpy(dtype=float)[ok]
            )
            rho_b = float("nan")
            if "champion" in sub:
                champ = sub.champion.to_numpy(dtype=float)[ok]
                m = E._correlation_matrix(
                    {"a": p[ok], "c": champ}, sub.surprise_pct.to_numpy(dtype=float)[ok]
                )
                rho_b = float(m.loc["a", "c"])
            rows.append(
                {
                    "model": model_name,
                    "partition": label,
                    "n": int(ok.sum()),
                    "rho": rho,
                    "rho_b_champion": rho_b,
                    "implied_pct_obtainable": rho**2,
                    "clears_bar": bool(abs(rho) >= B.RHO_BAR),
                }
            )

    frame = pd.DataFrame(rows)
    if len(frame):
        print(frame.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        pivot = frame.pivot(index="model", columns="partition", values="rho")
        if {"selection", "confirmation"} <= set(pivot.columns):
            pivot["bias"] = pivot["selection"] - pivot["confirmation"]
            print("\nselection minus confirmation = the measured selection bias")
            print(pivot.to_string(float_format=lambda v: f"{v:+.4f}"))
        print(
            f"\nRead the confirmation column only. The bar is rho >= {B.RHO_BAR:.2f}; "
            "a fitted union\nbelow it does not become a channel by being fitted harder."
        )
    return frame


if __name__ == "__main__":
    matrix_report()
    fitted_pass()
