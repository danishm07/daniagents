"""The research loop: propose → evaluate → select → evolve, with our adaptations.

Read ``BUILD_LOOP.md`` first. The three things that make this not a stock
FunSearch/AlphaEvolve port, restated because every module here depends on them:

1. **Fitness is marginal ensemble contribution, not ΔR².** Selecting on
   standalone score picks the strongest readers, and the strongest readers are
   the ones most correlated with each other — measured, six models across four
   vendors, collectively worth one arm at ρ_b 0.824. See :mod:`runner.fitness`.
2. **The evaluation is noisy.** Separating a ρ=0.25 candidate from a ρ=0.15 one
   needs n ≥ 2,000. Rungs below that buy direction only. See
   :mod:`runner.schedule`.
3. **Branches die on mechanism, variants die on score.** The archive is
   quality-diversity (MAP-Elites) so that a decorrelated weak arm survives a
   strong redundant one. See :mod:`runner.archive`.

Module map::

    registry.py   arms register themselves and declare cost, live-computability,
                  cutoff-safety, and which features they need
    features.py   shared (event_id, feature_name) cache; arms sharing a feature
                  compute it once
    score.py      rung-restricted scoring, on the competition scorer
    fitness.py    GLS-weighted marginal ensemble contribution, cross-validated
                  leave-one-quarter-out
    archive.py    MAP-Elites archive keyed on (rho_b band, family, cost tier)
    schedule.py   ASHA over n: 300 -> 1000 -> 2000 -> full
    report.py     the full matrix, including the negatives
    loop.py       one command
"""

from __future__ import annotations

__all__ = ["registry", "features", "score", "fitness", "archive", "schedule", "report"]
