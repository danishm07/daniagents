"""Score an arm on a rung — the competition scorer, restricted to a subset of events.

``eval.run`` scores whole quarters. ASHA needs to score the *same* arm on a
growing subset, so this module does the quarter loop itself and calls the same
underlying object: ``examples.scoring.score_submission``, via ``harness``.
Nothing here reimplements ΔR².

Three things it reports that a bare ΔR² does not, all required by
``BUILD_LOOP.md``:

``rho``          the signed partial correlation. ΔR² is its square times
                 ``1 − R²_surprise``, so this is the same information on a
                 readable scale — and it is the scale the ``ρ/√ρ_b``
                 combination arithmetic is stated in.
``rho_b``        correlation with the champion in residual space. The number
                 that says whether an arm is a new channel or a paraphrase.
``coverage``     and the neutral rate. A 0.5 is sitting out in disguise, and
                 coverage multiplies the contest metric roughly linearly, so an
                 arm that quietly declines on 15% of events is not the arm its
                 own-sample ΔR² describes.

Missing predictions are ``NaN``, never 0.5. Filling a hole with a neutral
converts a coverage failure into a measured result, which is exactly the kind of
laundering the run log exists to prevent.
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
from examples.scoring import score_submission  # noqa: E402

#: Below this many answered rows in a quarter the scorer's regressions are not
#: worth reading, and it returns ``None`` rather than a number. Skip the quarter
#: instead of coercing — a coverage hole is not a null result.
MIN_QUARTER_ROWS = 30


def _f(value) -> float:
    """The scorer's ``None`` is missing, not zero."""
    return float("nan") if value is None else float(value)


@dataclass
class ArmScore:
    """Everything one arm produced on one rung."""

    arm: str
    n: int
    n_asked: int
    delta_r2: float
    delta_r2_imputed: float
    pct_obtainable: float
    rho: float
    rho_b_champion: float
    vs_champion: float
    signs: str
    coverage: float
    neutral_rate: float
    cost_usd: float
    per_quarter: pd.DataFrame
    #: ``{quarter: Series indexed by event_id}`` — predictions with the surprise
    #: projected out. The input to every ρ_b and to the ensemble fitness.
    residuals: dict[str, pd.Series] = field(default_factory=dict)
    #: Raw predictions, kept so a promotion decision can bootstrap without
    #: re-running the arm. For an LLM arm that is the difference between a free
    #: decision and paying for the run twice.
    raw: dict[str, pd.Series] = field(default_factory=dict)
    error: str | None = None

    def row(self) -> dict:
        return {
            "arm": self.arm,
            "n": self.n,
            "delta_r2": self.delta_r2,
            "pct_obtainable": self.pct_obtainable,
            "rho": self.rho,
            "rho_b_champion": self.rho_b_champion,
            "vs_champion": self.vs_champion,
            "signs": self.signs,
            "coverage": self.coverage,
            "neutral_rate": self.neutral_rate,
            "cost_usd": self.cost_usd,
        }


def champion_column() -> str:
    return E.default_champion(quiet=True)


def score_arm(
    arm,
    events_by_quarter: dict[str, list[dict]],
    *,
    champion: str | None = None,
) -> ArmScore:
    """Score one arm over the given per-quarter event subsets.

    ``arm`` is a :class:`runner.registry.Arm`. Its :meth:`predict` must be free
    and deterministic — :meth:`ensure` is the scheduler's job, called before
    this, so that scoring the same arm at three rungs measures three subsets of
    one column rather than three different columns.
    """
    champion = champion_column() if champion is None else champion
    rows, residuals, raw = [], {}, {}
    n_asked = sum(len(v) for v in events_by_quarter.values())

    for quarter, events in events_by_quarter.items():
        if not events:
            continue
        wanted = [e["event_id"] for e in events]
        frame = harness.load(quarter)
        frame = frame[frame.event_id.isin(set(wanted))].copy()
        # Reorder the frame to the caller's event order so predictions line up
        # positionally — the arm contract is "one float per event, in order".
        frame = frame.set_index("event_id").reindex(wanted).reset_index()

        values = np.asarray(arm.predict(events, quarter), dtype=float)
        frame["_pred"] = values

        # An arm that answered almost nothing in a quarter has no measurable
        # score there — the scorer returns ``None`` rather than a number, and
        # coercing that to 0 would read a coverage hole as a null result. The
        # official-baseline replays only cover 2026Q2, so this is not
        # hypothetical.
        answered = frame.dropna(subset=["_pred"])
        if len(answered) < MIN_QUARTER_ROWS:
            continue

        # Own sample: the scorer's own number, on the rows the arm answered.
        scored = score_submission(answered, "_pred")
        # Imputed: holes filled with the arm's own mean, which is how the
        # contest's common sample treats a miss. The gap between the two is the
        # price of not answering.
        imputed = score_submission(frame, "_pred")

        surprise = frame.surprise_pct.to_numpy(dtype=float)
        y = frame.y.to_numpy(dtype=float)
        champ = frame[champion].to_numpy(dtype=float)

        paired = frame[np.isfinite(values) & np.isfinite(champ)]
        vs = float("nan")
        if len(paired) >= MIN_QUARTER_ROWS:
            vs = _f(harness.evaluate(paired, "_pred")["delta_r_squared"]) - _f(
                harness.evaluate(paired, champion)["delta_r_squared"]
            )

        matrix = E._correlation_matrix({"a": values, "c": champ}, surprise)
        ok = np.isfinite(values) & np.isfinite(surprise)
        resid = pd.Series(np.nan, index=frame.event_id)
        if ok.sum() >= 3:
            resid.iloc[np.where(ok)[0]] = E._residualize(values[ok], surprise[ok])

        rows.append(
            {
                "quarter": quarter,
                "n": scored["n_obs"],
                "n_asked": len(frame),
                "r2_surprise": _f(scored["r_squared_surprise"]),
                "delta_r2": _f(scored["delta_r_squared"]),
                "delta_r2_imputed": _f(imputed["delta_r_squared_imputed"]),
                "pct_obtainable": E.as_pct_obtainable(
                    _f(scored["delta_r_squared"]), _f(scored["r_squared_surprise"])
                ),
                "rho": E.partial_corr(values, y, surprise),
                "rho_b_champion": float(matrix.loc["a", "c"]),
                "vs_champion": vs,
                "coverage": float(np.isfinite(values).mean()),
                "neutral_rate": float(np.mean(values[np.isfinite(values)] == 0.5))
                if np.isfinite(values).any()
                else float("nan"),
            }
        )
        residuals[quarter] = resid
        raw[quarter] = pd.Series(values, index=frame.event_id)

    if not rows:
        return ArmScore(
            arm=arm.name, n=0, n_asked=n_asked, delta_r2=float("nan"),
            delta_r2_imputed=float("nan"), pct_obtainable=float("nan"), rho=float("nan"),
            rho_b_champion=float("nan"), vs_champion=float("nan"), signs="0/0",
            coverage=0.0, neutral_rate=float("nan"), cost_usd=0.0,
            per_quarter=pd.DataFrame(), error="no scorable quarters",
        )

    per_quarter = pd.DataFrame(rows)
    wins = int((per_quarter.vs_champion > 0).sum())
    graded = int(per_quarter.vs_champion.notna().sum())
    return ArmScore(
        arm=arm.name,
        n=int(per_quarter.n.sum()),
        n_asked=n_asked,
        delta_r2=float(per_quarter.delta_r2.mean()),
        delta_r2_imputed=float(per_quarter.delta_r2_imputed.mean()),
        pct_obtainable=float(per_quarter.pct_obtainable.mean()),
        # Fisher-z, not the arithmetic mean: at r ~ 0.7 the linear average is
        # biased toward zero by enough to move a ceiling that decides whether a
        # whole family is worth pursuing.
        rho=_pool_r(per_quarter.rho),
        rho_b_champion=_pool_r(per_quarter.rho_b_champion),
        vs_champion=float(per_quarter.vs_champion.mean()),
        signs=f"{wins}/{graded}",
        coverage=float(per_quarter.n.sum() / max(per_quarter.n_asked.sum(), 1)),
        neutral_rate=float(per_quarter.neutral_rate.mean()),
        cost_usd=arm.estimated_cost(int(per_quarter.n.sum())),
        per_quarter=per_quarter,
        residuals=residuals,
        raw=raw,
    )


def _pool_r(values: Sequence[float]) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(v):
        return float("nan")
    return float(np.tanh(np.arctanh(np.clip(v, -0.999999, 0.999999)).mean()))


def champion_score(events_by_quarter: dict[str, list[dict]]) -> ArmScore:
    """The champion, scored on the same rung — the reference every arm is read against."""
    from runner.registry import Arm

    column = champion_column()

    class _Champion(Arm):
        def predict(self, events, quarter):
            frame = harness.load(quarter).set_index("event_id")
            return [float(frame[column].get(e["event_id"], np.nan)) for e in events]

    return score_arm(
        _Champion(name="champion", family="reference", cost_usd_per_event=0.0),
        events_by_quarter,
    )


def target_residuals(events_by_quarter: dict[str, list[dict]]) -> dict[str, pd.Series]:
    """``y`` with the surprise projected out — what the contest actually pays for."""
    out = {}
    for quarter, events in events_by_quarter.items():
        if not events:
            continue
        wanted = [e["event_id"] for e in events]
        frame = harness.load(quarter)
        frame = frame[frame.event_id.isin(set(wanted))].set_index("event_id").reindex(wanted)
        y = frame.y.to_numpy(dtype=float)
        surprise = frame.surprise_pct.to_numpy(dtype=float)
        out[quarter] = pd.Series(E._residualize(y, surprise), index=frame.index)
    return out
