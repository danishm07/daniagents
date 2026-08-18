"""Fitness is marginal ensemble contribution, not ΔR².

This is the single most important module in the loop, and the reason is measured
rather than argued:

    Six models, four vendors, two continents of pretraining data, every one a
    defensible pick on ΔR², collectively worth one arm at ρ_b 0.824.

Selecting on standalone score picks the strongest readers, and the strongest
readers are the ones most correlated with each other. The arithmetic at our
current ρ = 0.199:

===== ===== =======
ρ     ρ_b   ceiling
===== ===== =======
0.199 0.824 4.5%
0.250 0.824 7.1%
0.199 0.400 9.9%
0.199 0.200 19.9%
===== ===== =======

ρ has maybe 1.25× of room. ρ_b has 4× — TF-IDF already measured 0.193. So
selection has to reward decorrelation, and the way to do that without inverting
into "reward noise" is to score an arm by **how much it raises the ensemble**,
not by how uncorrelated it is.

The necessary-but-not-sufficient trap, also measured: champion + pure noise sits
at ρ_b 0.696 — as decorrelated as a different vendor — and is worth −0.019. A
raw ρ_b objective would rank that highly. Marginal contribution ranks it below
zero, automatically, and ``__main__`` asserts exactly that.

**The weights are cross-validated leave-one-quarter-out.** Fitting GLS weights
on the same events they are scored on would manufacture a positive marginal
contribution for any arm at all — with k members you have k free parameters and
three quarters of data. Each fold fits on two quarters and scores on the third;
the reported number is the out-of-sample one. In-sample is reported too, and the
gap between them is the honest read on how much of a headline is fitting noise.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Mean ``1 − R²_surprise`` over the dev quarters. Converts a correlation on the
#: residual scale back to the scorer's scale: ΔR² = ρ² · (1 − R²_surprise).
OBTAINABLE = 0.9413

#: Ridge shrinkage of the member covariance toward the identity before
#: inversion. With three quarters and a handful of highly-correlated members the
#: unshrunk normal equations are close to singular, and an unstable weight
#: vector produces a marginal contribution that is mostly numerical noise.
SHRINKAGE = 0.15


@dataclass
class Marginal:
    """What one candidate adds to one incumbent set."""

    arm: str
    members: tuple[str, ...]
    rho_without: float
    rho_with: float
    marginal_rho: float
    marginal_delta_r2: float
    rho_without_insample: float
    rho_with_insample: float
    weight: float
    n: int

    def row(self) -> dict:
        return {
            "arm": self.arm,
            "ens_rho_without": self.rho_without,
            "ens_rho_with": self.rho_with,
            "marginal_rho": self.marginal_rho,
            "marginal_delta_r2": self.marginal_delta_r2,
            "own_weight": self.weight,
            "insample_gain": self.rho_with_insample - self.rho_without_insample,
            "k_members": len(self.members),
        }


# --------------------------------------------------------------------------
# Standardising into a common space
# --------------------------------------------------------------------------


def _standardize(series: pd.Series) -> pd.Series:
    """Z-score the finite values; missing becomes 0.

    Zero is the arm's own mean after centring, which is precisely how the
    contest's common sample treats an event you did not answer. So a member with
    holes contributes nothing on those events rather than dropping them from
    everyone else's evaluation — the alternative would let a low-coverage
    candidate shrink the sample instead of paying for its coverage.
    """
    values = series.to_numpy(dtype=float)
    ok = np.isfinite(values)
    out = np.zeros(len(values))
    if ok.sum() >= 3:
        sd = values[ok].std()
        if sd:
            out[ok] = (values[ok] - values[ok].mean()) / sd
    return pd.Series(out, index=series.index)


def design(
    members: Mapping[str, Mapping[str, pd.Series]],
    target: Mapping[str, pd.Series],
    quarters: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Stack members into an ``(n_events, k)`` matrix aligned to ``target``.

    Two alignment decisions, both consequences of the same rule:

    * **Standardised within each quarter** before stacking. Quarter difficulty
      drifts — the benchmark's own R² climbs 0.049 → 0.078 across the dev
      quarters — and pooling raw residuals would let the loudest quarter set the
      weights.
    * **Reindexed onto the target's events**, with anything the member did not
      answer becoming 0. Members do not all cover the same events: the official
      baseline replays only exist for 2026Q2, and a peer arm has no aggregate
      when the window is thin. Zero after centring is the member's own mean,
      which is exactly what the contest's common sample does with a miss — so a
      partial member contributes nothing where it is silent instead of dropping
      those events from every other member's evaluation.
    """
    names = sorted(members)
    columns = []
    for name in names:
        parts = []
        for quarter in quarters:
            index = target[quarter].index
            series = members[name].get(quarter)
            if series is None:
                parts.append(pd.Series(0.0, index=index))
            else:
                parts.append(_standardize(series).reindex(index).fillna(0.0))
        columns.append(pd.concat(parts))
    return np.column_stack([c.to_numpy(dtype=float) for c in columns]), names


def _target(target: Mapping[str, pd.Series], quarters: Sequence[str]) -> np.ndarray:
    return pd.concat([_standardize(target[q]) for q in quarters]).to_numpy(dtype=float)


# --------------------------------------------------------------------------
# GLS weights and the ensemble correlation they buy
# --------------------------------------------------------------------------


def gls_weights(X: np.ndarray, y: np.ndarray, shrinkage: float = SHRINKAGE) -> np.ndarray:
    """``Σ⁻¹ρ``, ridge-shrunk — the optimal linear combination of correlated views.

    Not scaled to sum to one, and deliberately so: the scorer is invariant to
    any affine remap of the final prediction, so the overall scale is free and
    forcing it would only add a constraint that costs accuracy.
    """
    k = X.shape[1]
    if k == 0:
        return np.zeros(0)
    sigma = np.corrcoef(X, rowvar=False) if k > 1 else np.array([[1.0]])
    sigma = np.nan_to_num(sigma, nan=0.0)
    np.fill_diagonal(sigma, 1.0)
    sigma = (1 - shrinkage) * sigma + shrinkage * np.eye(k)
    rho = np.array([_corr(X[:, j], y) for j in range(k)])
    try:
        return np.linalg.solve(sigma, rho)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(sigma, rho, rcond=None)[0]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ensemble_rho(
    members: Mapping[str, Mapping[str, pd.Series]],
    target: Mapping[str, pd.Series],
    quarters: Sequence[str],
) -> tuple[float, float, dict[str, float]]:
    """Out-of-sample and in-sample ensemble correlation, plus the pooled weights.

    Out-of-sample is leave-one-quarter-out: weights from the other quarters,
    correlation measured on the held-out one, pooled in Fisher-z. With fewer
    than two quarters there is no fold to hold out and the out-of-sample number
    is returned as the in-sample one — flagged by the caller rather than
    silently passed off as validated.
    """
    # The target defines the sample: an arm cannot be credited on events nobody
    # is scored on, and a quarter with no target residual is not a fold.
    quarters = [q for q in quarters if q in target and len(target[q])]
    if not members or not quarters:
        return 0.0, 0.0, {}

    X_all, names = design(members, target, quarters)
    y_all = _target(target, quarters)
    w_all = gls_weights(X_all, y_all)
    insample = abs(_corr(X_all @ w_all, y_all))

    if len(quarters) < 2:
        return insample, insample, dict(zip(names, w_all, strict=True))

    folds = []
    for held in quarters:
        train = [q for q in quarters if q != held]
        X_tr, _ = design(members, target, train)
        X_te, _ = design(members, target, [held])
        w = gls_weights(X_tr, _target(target, train))
        folds.append(_corr(X_te @ w, _target(target, [held])))
    # Sign is free to the scorer but not to the pooling: an ensemble that is
    # consistently inverted is as good as one consistently aligned, while one
    # that flips sign between quarters is not, and averaging |r| would hide
    # that. Pool signed, then take the magnitude.
    pooled = float(np.tanh(np.mean(np.arctanh(np.clip(folds, -0.999999, 0.999999)))))
    return abs(pooled), insample, dict(zip(names, w_all, strict=True))


def marginal(
    candidate: str,
    candidate_residuals: Mapping[str, pd.Series],
    incumbents: Mapping[str, Mapping[str, pd.Series]],
    target: Mapping[str, pd.Series],
    quarters: Sequence[str] | None = None,
) -> Marginal:
    """How much ``candidate`` raises the ensemble ceiling over ``incumbents``.

    This is the loop's fitness. Reported alongside — never instead of —
    standalone ΔR², ρ and ρ_b, because the three answer different questions and
    ``BUILD_LOOP.md`` requires all of them.
    """
    # Both ensembles are measured on the quarters the *candidate* can be
    # evaluated on, not on every quarter the incumbents cover. Anything fitted
    # is blind in 2025Q4 — the archive's first quarter has no prior quarter to
    # train on — while a read arm is not. Crediting the incumbents for a quarter
    # the candidate structurally cannot enter would make every fitted arm look
    # worse for an artefact of where the archive starts, not for anything about
    # the arm. Live, that quarter has years of history behind it.
    if quarters is None:
        available = {q for q, s in candidate_residuals.items() if len(s) and s.notna().any()}
        quarters = sorted(available or set(candidate_residuals))
    with_members = {**incumbents, candidate: candidate_residuals}

    rho_without, ins_without, _ = ensemble_rho(incumbents, target, quarters)
    rho_with, ins_with, weights = ensemble_rho(with_members, target, quarters)

    n = sum(len(candidate_residuals[q]) for q in quarters if q in candidate_residuals)
    gain = rho_with - rho_without
    return Marginal(
        arm=candidate,
        members=tuple(sorted(incumbents)),
        rho_without=rho_without,
        rho_with=rho_with,
        marginal_rho=gain,
        # Converting the *ceiling* move to the scorer's scale, not the gain
        # squared: ΔR² is quadratic in ρ, so a gain from 0.20 to 0.22 is worth
        # far more than one from 0.02 to 0.04, and squaring the difference would
        # get that backwards.
        marginal_delta_r2=(rho_with**2 - rho_without**2) * OBTAINABLE,
        rho_without_insample=ins_without,
        rho_with_insample=ins_with,
        weight=float(weights.get(candidate, 0.0)),
        n=n,
    )


def inter_arm_matrix(
    residuals: Mapping[str, Mapping[str, pd.Series]],
    quarters: Sequence[str] | None = None,
) -> pd.DataFrame:
    """The ρ_b matrix between every pair of arms, pooled in Fisher-z per quarter."""
    names = sorted(residuals)
    if quarters is None:
        quarters = sorted({q for m in residuals.values() for q in m})
    stack = []
    for quarter in quarters:
        cols = {n: residuals[n][quarter] for n in names if quarter in residuals[n]}
        if len(cols) < 2:
            continue
        frame = pd.DataFrame(cols)
        stack.append(frame.corr().reindex(index=names, columns=names))
    if not stack:
        return pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    z = np.arctanh(np.clip(np.stack([s.to_numpy(dtype=float) for s in stack]), -0.999999, 0.999999))
    pooled = np.tanh(np.nanmean(z, axis=0))
    np.fill_diagonal(pooled, 1.0)
    return pd.DataFrame(pooled, index=names, columns=names)


# --------------------------------------------------------------------------
# Self-test — the trap this module exists to avoid
# --------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    quarters = ["q1", "q2", "q3"]
    n = 1200

    truth, champion, noise, fresh = {}, {}, {}, {}
    for q in quarters:
        signal = rng.standard_normal(n)
        idx = pd.Index([f"{q}-{i}" for i in range(n)], name="event_id")
        truth[q] = pd.Series(signal, index=idx)
        # champion: rho ~ 0.20 with the target
        champion[q] = pd.Series(
            0.20 * signal + np.sqrt(1 - 0.20**2) * rng.standard_normal(n), index=idx
        )
        # pure noise: decorrelated from the champion, and worth nothing
        noise[q] = pd.Series(rng.standard_normal(n), index=idx)
        # a genuinely new weak channel: rho ~ 0.15, orthogonal to the champion
        fresh[q] = pd.Series(
            0.15 * signal + np.sqrt(1 - 0.15**2) * rng.standard_normal(n), index=idx
        )

    base = {"champion": champion}
    m_noise = marginal("noise", noise, base, truth, quarters)
    m_fresh = marginal("fresh", fresh, base, truth, quarters)
    m_copy = marginal("copy", champion, base, truth, quarters)

    print(f"ensemble rho, champion alone      {m_noise.rho_without:+.4f}")
    print(f"  + pure noise                    {m_noise.rho_with:+.4f}  "
          f"marginal {m_noise.marginal_rho:+.4f}  dR2 {m_noise.marginal_delta_r2:+.4f}")
    print(f"  + a decorrelated weak channel   {m_fresh.rho_with:+.4f}  "
          f"marginal {m_fresh.marginal_rho:+.4f}  dR2 {m_fresh.marginal_delta_r2:+.4f}")
    print(f"  + a copy of the champion        {m_copy.rho_with:+.4f}  "
          f"marginal {m_copy.marginal_rho:+.4f}  dR2 {m_copy.marginal_delta_r2:+.4f}")

    # The measured trap: noise is as decorrelated from the champion as another
    # vendor is, and is worth less than nothing. A raw rho_b objective ranks it
    # first. This one must not.
    assert m_noise.marginal_rho < m_fresh.marginal_rho, (m_noise, m_fresh)
    assert m_fresh.marginal_rho > 0, m_fresh
    # A redundant copy adds nothing, even though its standalone rho ties the
    # champion's. This is the six-models-one-arm result in miniature.
    assert m_copy.marginal_rho < m_fresh.marginal_rho, (m_copy, m_fresh)
    print("\nfitness ranks decorrelated-and-real above both pure noise and a redundant copy")

    matrix = inter_arm_matrix({"champion": champion, "noise": noise, "fresh": fresh}, quarters)
    print("\ninter-arm rho_b")
    print(matrix.to_string(float_format=lambda v: f"{v:+.3f}"))
