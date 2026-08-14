"""The single entry point every experiment goes through.

``harness.py`` answers "what does the competition scorer say about this column?".
This module answers the question one level up: "should I believe it, and what did
it cost?" — and it writes the answer down whether or not the run succeeded.

Four things it adds over calling :func:`harness.backtest` directly.

1. **2026Q3 is held out.** :data:`DEV_QUARTERS` excludes it and :func:`run`
   refuses to touch it unless you pass ``final=True``. Every such touch is
   logged, so "we only evaluated on the holdout once" is a checkable claim
   rather than a memory.

2. **Partial correlation alongside ΔR², everywhere.** ΔR² is a squared
   quantity, so it throws away the sign and compresses differences near zero.
   The partial correlation of the prediction with ``y`` after projecting out
   ``surprise_pct`` is the same information, signed and on a readable scale —
   and it is what the ``ρ/√ρ_b`` combination arithmetic below is stated in.
   Identity worth knowing: ``ΔR² = pc² · (1 − R²_surprise)``, asserted in
   ``__main__``.

3. **Inter-view correlation is a first-class output, not an afterthought.**
   Combining ``k`` views whose individual correlation with the target is ``ρ``
   and whose mutual correlation is ``ρ_b`` asymptotes to ``ρ/√ρ_b`` — no matter
   how many you add. With a single LLM read at ρ ≈ 0.269, mutual correlation
   above ~0.17 puts leader-level performance out of reach at any ``k``. So a
   view's *decorrelation* from what we already have is as decision-relevant as
   its standalone score, and both get reported together.

4. **Every run is appended to a JSONL log — including failures.** Config hash,
   feature set, hyperparameters, all metrics, runtime, cost, git commit,
   traceback. The reason is honesty about multiple comparisons: a headline
   number means nothing without the count of configurations tried to reach it,
   and that count is only trustworthy if the failures are in the log too.

Typical use::

    import eval

    def tfidf(events, quarter):
        train = eval.training_data(quarter)      # prior quarters only
        ...
        return preds

    eval.run(tfidf, "tfidf residual", config={"alpha": 1.0, "w": 0.25})

    # several views at once: each is scored, and the correlation matrix
    # between them is part of the output
    eval.run({"llm": llm_view, "tfidf": tfidf, "embed": embed_view},
             "three views", combine=eval.zblend)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import subprocess
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import harness
from examples.scoring import no_spread
from harness import GEMINI, GPT, QUARTERS, load, training_data  # noqa: F401  (re-exported)

#: Views the caller derived from other views. Excluded from the combination
#: arithmetic — a blend is correlated with its own inputs by construction, and
#: counting it as an independent view inflates every ceiling in the table.
DERIVED_VIEWS = {"combined"}

#: Never evaluated until the final model is chosen. One shot, at the end.
HOLDOUT = "2026Q3"

#: What every experiment runs on.
DEV_QUARTERS = [q for q in QUARTERS if q != HOLDOUT]

RUN_LOG = Path(__file__).parent / "runs" / "eval_runs.jsonl"

#: Bootstrap sd of ΔR² at n≈1,900. Two of these is the "believe it" bar for a
#: mean improvement; below that, only sign consistency across quarters counts.
SE_DELTA_R2 = 0.010

#: A view function takes the leakage-safe event list plus the quarter it is
#: predicting, and returns one float per event, in order. Single-argument
#: functions are accepted too — see :func:`_call_view`.
ViewFn = Callable[..., Sequence[float]]


# --------------------------------------------------------------------------
# The two statistics the scorer's headline number hides
# --------------------------------------------------------------------------


def _residualize(values: np.ndarray, surprise: np.ndarray) -> np.ndarray:
    """``values`` with ``surprise`` projected out by OLS.

    Everything the contest pays for lives in this residual space: the benchmark
    already owns whatever the surprise explains.
    """
    slope, intercept = np.polyfit(surprise, values, 1)
    return values - (slope * surprise + intercept)


def partial_corr(pred: np.ndarray, y: np.ndarray, surprise: np.ndarray) -> float:
    """Correlation of ``pred`` with ``y``, controlling for ``surprise``.

    Signed, unlike ΔR². A view can be genuinely informative and *inverted*; the
    scorer is indifferent (affine invariance) but you are not, because a sign
    flip between quarters means something different from a consistent edge.
    """
    ok = np.isfinite(pred) & np.isfinite(y) & np.isfinite(surprise)
    if ok.sum() < 3:
        return float("nan")
    if _degenerate(pred[ok]) or _degenerate(y[ok]):
        return 0.0
    pr = _residualize(pred[ok], surprise[ok])
    yr = _residualize(y[ok], surprise[ok])
    if _degenerate(pr) or _degenerate(yr):
        return 0.0
    return float(np.corrcoef(pr, yr)[0, 1])


def _degenerate(values: np.ndarray) -> bool:
    """Zero spread, by the scorer's own definition.

    Reusing :func:`examples.scoring.no_spread` rather than testing ``std == 0``
    matters: residualizing a constant vector leaves floating-point dust with a
    nonzero standard deviation, and correlating that dust against ``y`` produced
    a spurious ±0.011 partial correlation for a constant 0.5 — a prediction the
    scorer correctly calls exactly zero.
    """
    centred = values - values.mean()
    return no_spread(float(centred @ centred), len(values))


def combination_ceiling(rho: float, rho_b: float) -> float:
    """Best correlation reachable by combining infinitely many views.

    ``rho`` is each view's correlation with the target, ``rho_b`` their mutual
    correlation. The limit of an equal-weight average is ``ρ/√ρ_b`` — adding
    views past that point buys nothing, which is why a decorrelated weak view
    beats another strong-but-redundant one.

    Returns ``inf`` when ``rho_b <= 0``: uncorrelated views have no ceiling in
    this arithmetic, and that is a modelling signal, not a number to trust.
    """
    if rho_b <= 0:
        return float("inf")
    return rho / math.sqrt(rho_b)


# --------------------------------------------------------------------------
# Running an experiment
# --------------------------------------------------------------------------


def _call_view(fn: ViewFn, events: list[dict], quarter: str) -> list[float]:
    """Call a view with ``(events, quarter)``, or ``(events)`` if that is all it takes.

    The two-argument form is the one to write: a model that fits on prior
    quarters needs to know which quarter it is predicting so it can ask
    :func:`harness.training_data` for the right slice.
    """
    try:
        n_params = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n_params = 2
    return list(fn(events, quarter) if n_params >= 2 else fn(events))


def zblend(weights: Mapping[str, float] | None = None) -> Callable[[dict], np.ndarray]:
    """Additive blend of z-scored views — the combination that has actually worked.

    Z-scoring first is what makes the weights comparable; the scorer's affine
    invariance means the z-scoring itself is free.
    """

    def combine(views: dict[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(len(next(iter(views.values()))))
        for name, values in views.items():
            w = 1.0 if weights is None else weights.get(name, 0.0)
            sd = np.nanstd(values)
            out = out + w * ((values - np.nanmean(values)) / (sd if sd else 1.0))
        return out

    return combine


@dataclass
class RunResult:
    """Everything one experiment produced, plus the record that was logged."""

    name: str
    run_id: str
    config_hash: str
    per_quarter: pd.DataFrame
    summary: pd.DataFrame
    correlations: dict[str, pd.DataFrame] = field(default_factory=dict)
    ceilings: pd.DataFrame | None = None
    runtime_s: float = 0.0
    cost_usd: float | None = None
    status: str = "ok"

    def report(self) -> None:
        print(f"\n{self.name}   [{self.run_id[:8]}  cfg {self.config_hash[:8]}]")
        print("-" * 78)
        print(self.per_quarter.to_string(index=False, float_format=_fmt))
        print()
        print(self.summary.to_string(index=False, float_format=_fmt))
        for label, matrix in self.correlations.items():
            print(f"\ninter-view correlation ({label}, surprise projected out)")
            print(matrix.to_string(float_format=_fmt))
        if self.ceilings is not None and len(self.ceilings):
            print("\ncombination ceiling  rho / sqrt(rho_b)")
            print(self.ceilings.to_string(index=False, float_format=_fmt))
        print(f"\nruntime {self.runtime_s:.1f}s"
              + (f"   cost ${self.cost_usd:.2f}" if self.cost_usd is not None else ""))


def run(
    views: ViewFn | Mapping[str, ViewFn],
    name: str,
    *,
    combine: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None,
    config: Mapping | None = None,
    quarters: Sequence[str] | None = None,
    baseline: str = GEMINI,
    cost_usd: float | None = None,
    notes: str = "",
    final: bool = False,
    log: bool = True,
) -> RunResult:
    """Score one or more views per quarter, log the run, return the result.

    ``views`` is a single function or a name → function mapping. Each is scored
    on its own; ``combine`` additionally scores a combination of them. Nothing
    is combined implicitly — an unweighted average of views you have not looked
    at is a guess, not a result.

    ``config`` is free-form and is hashed into ``config_hash``: put every
    hyperparameter and feature-set choice in it, because that hash is what makes
    the run log a record of *how many* configurations were tried.

    ``final=True`` is the only way to touch :data:`HOLDOUT`, and it is recorded.
    """
    if quarters is None:
        quarters = QUARTERS if final else DEV_QUARTERS
    quarters = list(quarters)

    touching_holdout = HOLDOUT in quarters
    if touching_holdout and not final:
        raise ValueError(
            f"{HOLDOUT} is the holdout. Pass final=True to spend it — one evaluation, "
            f"after the model is chosen. Default dev quarters are {DEV_QUARTERS}."
        )
    if touching_holdout:
        print(f"!! SPENDING THE HOLDOUT: {HOLDOUT} is being evaluated. This is logged.")

    view_map: dict[str, ViewFn] = (
        dict(views) if isinstance(views, Mapping) else {"model": views}
    )
    config = dict(config or {})
    config_hash = hashlib.sha256(
        json.dumps({"views": sorted(view_map), "config": config}, sort_keys=True, default=str).encode()
    ).hexdigest()
    run_id = uuid.uuid4().hex
    started = time.time()

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "status": "running",
        "config_hash": config_hash,
        "config": config,
        "views": sorted(view_map),
        "quarters": quarters,
        "spent_holdout": touching_holdout,
        "baseline": baseline,
        "cost_usd": cost_usd,
        "notes": notes,
        "git": _git_commit(),
    }

    try:
        rows, matrices = [], {}
        for quarter in quarters:
            frame = load(quarter).copy()
            events = harness.events_for(quarter)
            y = frame["y"].to_numpy(dtype=float)
            surprise = frame["surprise_pct"].to_numpy(dtype=float)

            preds: dict[str, np.ndarray] = {}
            for view_name, fn in view_map.items():
                values = _call_view(fn, events, quarter)
                if len(values) != len(frame):
                    raise ValueError(
                        f"view {view_name!r} returned {len(values)} predictions for "
                        f"{quarter}, expected {len(frame)}"
                    )
                preds[view_name] = np.asarray(values, dtype=float)

            if combine is not None:
                preds["combined"] = np.asarray(combine(dict(preds)), dtype=float)

            baseline_delta = harness.delta_r2(frame, baseline)
            for view_name, values in preds.items():
                frame["_pred"] = values
                scored = harness.evaluate(frame, "_pred")
                rows.append(
                    {
                        "quarter": quarter,
                        "view": view_name,
                        "n": scored["n_obs"],
                        "r2_surprise": scored["r_squared_surprise"],
                        "delta_r2": scored["delta_r_squared"],
                        "delta_r2_imp": scored["delta_r_squared_imputed"],
                        "partial_corr": partial_corr(values, y, surprise),
                        f"vs_{_short(baseline)}": scored["delta_r_squared"] - baseline_delta,
                    }
                )

            if len(preds) > 1:
                matrices[quarter] = _correlation_matrix(preds, surprise)

        per_quarter = pd.DataFrame(rows)
        summary = _summarize(per_quarter, baseline)
        pooled = _pool(matrices)
        result = RunResult(
            name=name,
            run_id=run_id,
            config_hash=config_hash,
            per_quarter=per_quarter,
            summary=summary,
            correlations=({"pooled": pooled} if pooled is not None else {}),
            ceilings=_ceilings(summary, pooled, float(per_quarter.r2_surprise.mean())),
            runtime_s=time.time() - started,
            cost_usd=cost_usd,
        )
        record.update(
            status="ok",
            runtime_s=result.runtime_s,
            per_quarter=per_quarter.to_dict("records"),
            summary=summary.to_dict("records"),
            correlations={q: m.to_dict() for q, m in matrices.items()},
            ceilings=None if result.ceilings is None else result.ceilings.to_dict("records"),
        )
        return result

    except Exception as exc:  # logged, then re-raised — a failed run still counts
        record.update(
            status="error",
            runtime_s=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if log:
            _append(record)


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def _correlation_matrix(preds: Mapping[str, np.ndarray], surprise: np.ndarray) -> pd.DataFrame:
    """Pairwise correlation between views, in residual space.

    Residualized against the surprise first, because two views can look
    correlated purely by both tracking the benchmark — which says nothing about
    whether they add to *each other* on the part that is actually scored.
    """
    residuals = {}
    for view_name, values in preds.items():
        ok = np.isfinite(values) & np.isfinite(surprise)
        column = np.full(len(values), np.nan)
        if ok.sum() >= 3:
            column[ok] = _residualize(values[ok], surprise[ok])
        residuals[view_name] = column
    return pd.DataFrame(residuals).corr()


def _pool(matrices: Mapping[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Mean correlation matrix across quarters — the number to design against."""
    if not matrices:
        return None
    return sum(matrices.values()) / len(matrices)


def _summarize(per_quarter: pd.DataFrame, baseline: str) -> pd.DataFrame:
    vs_column = f"vs_{_short(baseline)}"
    rows = []
    for view_name, group in per_quarter.groupby("view", sort=False):
        wins = int((group[vs_column] > 0).sum())
        rows.append(
            {
                "view": view_name,
                "mean_delta_r2": group.delta_r2.mean(),
                "mean_partial_corr": group.partial_corr.mean(),
                f"mean_{vs_column}": group[vs_column].mean(),
                "quarters_won": f"{wins}/{len(group)}",
                "verdict": _verdict(group[vs_column]),
            }
        )
    return pd.DataFrame(rows)


def _verdict(vs_baseline: pd.Series) -> str:
    """The promotion rule, applied mechanically so it cannot be argued with."""
    if len(vs_baseline) >= 3 and (vs_baseline > 0).all():
        return "sign-consistent"
    if vs_baseline.mean() > 2 * SE_DELTA_R2:
        return "mean > 2 SE"
    return "noise"


def _ceilings(
    summary: pd.DataFrame, pooled: pd.DataFrame | None, r2_surprise: float
) -> pd.DataFrame | None:
    """Per-view: how far combining more views *like this one* could ever get.

    ``rho_b`` is the view's mean correlation with the other **independent**
    views; anything in :data:`DERIVED_VIEWS` is dropped first, since a blend
    correlates with its own inputs by construction and would inflate every
    ceiling in the table.

    ``implied_delta_r2`` converts the ceiling correlation back to the scorer's
    scale via ``ΔR² = pc² · (1 − R²_surprise)``, so it is directly comparable to
    a leaderboard number rather than being an unscaled squared correlation.
    """
    if pooled is None:
        return None
    independent = [c for c in pooled.columns if c not in DERIVED_VIEWS]
    if len(independent) < 2:
        return None
    rows = []
    for view_name in independent:
        others = [c for c in independent if c != view_name]
        rho_b = float(pooled.loc[view_name, others].mean())
        match = summary.loc[summary.view == view_name, "mean_partial_corr"]
        rho = float(match.iloc[0]) if len(match) else float("nan")
        ceiling = combination_ceiling(abs(rho), rho_b)
        rows.append(
            {
                "view": view_name,
                "rho": rho,
                "rho_b": rho_b,
                "corr_ceiling": ceiling,
                "implied_delta_r2": (
                    ceiling**2 * (1 - r2_surprise) if math.isfinite(ceiling) else float("inf")
                ),
            }
        )
    return pd.DataFrame(rows)


def _short(label: str) -> str:
    return label.split()[0].lower()


def _fmt(v: float) -> str:
    return f"{v:+.4f}" if isinstance(v, (int, float)) and math.isfinite(v) else str(v)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def _append(record: Mapping) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def runs(status: str | None = None) -> pd.DataFrame:
    """The run log as a frame — the audit trail for "how many things did we try?"."""
    if not RUN_LOG.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in RUN_LOG.open() if line.strip()]
    frame = pd.DataFrame(records)
    return frame if status is None else frame[frame.status == status]


# --------------------------------------------------------------------------
# Self-test — the module's own correctness gate
# --------------------------------------------------------------------------

if __name__ == "__main__":
    harness.compare()

    null = run(lambda events: [0.5] * len(events), "constant 0.5 (must be exactly 0)", log=False)
    null.report()
    assert abs(null.per_quarter.delta_r2).max() < 1e-12, "constant prediction must score 0"

    # ΔR² = pc² · (1 − R²_surprise). If this identity fails, one of the two
    # numbers this module reports is not what it claims to be.
    for quarter in DEV_QUARTERS:
        frame = load(quarter)
        pred = frame[GEMINI].to_numpy(dtype=float)
        y = frame["y"].to_numpy(dtype=float)
        surprise = frame["surprise_pct"].to_numpy(dtype=float)
        scored = harness.evaluate(frame, GEMINI)
        implied = partial_corr(pred, y, surprise) ** 2 * (1 - scored["r_squared_surprise"])
        assert abs(implied - scored["delta_r_squared"]) < 1e-9, (
            f"{quarter}: partial corr and delta_r2 disagree "
            f"({implied:.6f} vs {scored['delta_r_squared']:.6f})"
        )
    print(f"\nidentity check passed on {DEV_QUARTERS}: delta_r2 == pc^2 * (1 - r2_surprise)")

    # Two views, one of them deliberately redundant, to exercise the matrix and
    # the ceiling arithmetic.
    def gemini_view(events, quarter):
        return load(quarter)[GEMINI].tolist()

    def gpt_view(events, quarter):
        return load(quarter)[GPT].tolist()

    # log=False throughout the self-test: the run log is the audit trail for how
    # many configurations were tried to reach a headline number, and a module
    # smoke test is not one of them.
    both = run(
        {"gemini": gemini_view, "gpt": gpt_view},
        "official baselines as two views",
        combine=zblend(),
        config={"note": "known to correlate ~0.8 — the matrix should show it"},
        log=False,
    )
    both.report()

    try:
        run(lambda events: [0.5] * len(events), "holdout guard", quarters=[HOLDOUT], log=False)
    except ValueError as exc:
        print(f"\nholdout guard works: {exc}")
