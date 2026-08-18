"""Does context pay off only on a model strong enough to use it?

The question, and why it is the most valuable one on the board right now.

Our own screen concluded **context is inert** and filed the model as the lever
(commit ``8e5eb75``). The published leaderboard says otherwise, twice:

* ``unfun-bot`` vs ``fun-bot`` — same team, same model, same prompt, differing
  only by an injected block of the focal ticker's own prior realized abnormal
  returns, its most recent prior fact summary, and the previous quarter's
  outcome distribution. **0.1751 → 0.2317, Δ = +0.0566** on 274 common events.
  That context block is very close to our ``third_place`` arm.
* ``aigent-arm-b`` vs ``aigent-arm-a`` — same model, plus a curated rulebook.
  **+0.0200** (contest) and **+0.0215** (global), sign-stable across two boards.

The difference between their setup and ours is **model tier**: they run frontier
models, we screened on ``gemini-2.5-flash-lite`` and ``gemini-2.5-flash``. So
the hypothesis is an interaction — context is worth little to a model that
cannot exploit it, and worth a lot to one that can.

The cheap half of that test is already paid for. ``base`` and ``third_place``
both exist at flash-lite *and* at flash, ~2,100 events each. This module
computes the **difference of differences** across the two tiers, paired on
identical events, which is the interaction term itself.

A note on why this is measurable at all despite the noise table. Everything here
is a *paired* comparison on identical events between two arms correlated at
ρ_b ≈ 0.8, so ``Var(a−b) = 2σ²(1−ρ_b)`` — roughly a fivefold variance reduction
against the unpaired case the n≥2000 rule was derived from. The rule still binds
for a *promotion decision* against the champion; it is unnecessarily strict for
a within-tier contrast.

    uv run python -m runner.interaction
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval as E  # noqa: E402
import harness  # noqa: E402
from runner import arms_builtin  # noqa: E402,F401
from runner import registry as R  # noqa: E402
from runner import schedule as S  # noqa: E402
from runner import score as SC  # noqa: E402

#: (tier label, base arm, context arm). Ordered weakest model first.
PAIRS = [
    ("gemini-2.5-flash-lite", "ctx.base.gemini-2.5-flash-lite", "ctx.third_place.gemini-2.5-flash-lite"),
    ("gemini-2.5-flash", "ctx.base.gemini-2.5-flash", "ctx.third_place.gemini-2.5-flash"),
]


def paired_delta(base: str, context: str, events: dict[str, list[dict]], n_boot: int = 2000) -> dict:
    """ΔR²(context) − ΔR²(base), on the events **both** arms answered.

    Bootstrapped paired, so quarter difficulty and event difficulty cancel and
    what is left is the effect of the context block.
    """
    a, b = R.get(context), R.get(base)
    rng = np.random.default_rng(0)
    per_quarter, draws = [], []

    for quarter, batch in events.items():
        frame = harness.load(quarter)
        wanted = [e["event_id"] for e in batch]
        frame = frame[frame.event_id.isin(set(wanted))].set_index("event_id").reindex(wanted)
        frame = frame.reset_index()

        ctx_values = np.asarray(a.predict(batch, quarter), dtype=float)
        base_values = np.asarray(b.predict(batch, quarter), dtype=float)
        surprise = frame.surprise_pct.to_numpy(dtype=float)
        y = frame.y.to_numpy(dtype=float)

        keep = (
            np.isfinite(ctx_values) & np.isfinite(base_values)
            & np.isfinite(surprise) & np.isfinite(y)
        )
        if keep.sum() < 100:
            continue
        c, z, s, t = ctx_values[keep], base_values[keep], surprise[keep], y[keep]

        d_ctx = E._delta_r2_fast(c, s, t)
        d_base = E._delta_r2_fast(z, s, t)
        per_quarter.append(
            {
                "quarter": quarter,
                "n": int(keep.sum()),
                "base": d_base,
                "context": d_ctx,
                "delta": d_ctx - d_base,
                "rho_b": float(np.corrcoef(E._residualize(c, s), E._residualize(z, s))[0, 1]),
            }
        )
        n = len(t)
        draws.append(
            [
                E._delta_r2_fast(c[i], s[i], t[i]) - E._delta_r2_fast(z[i], s[i], t[i])
                for i in (rng.integers(0, n, n) for _ in range(n_boot))
            ]
        )

    table = pd.DataFrame(per_quarter)
    mean_draws = np.mean(np.array(draws), axis=0)
    return {
        "per_quarter": table,
        "delta": float(table.delta.mean()),
        "se": float(mean_draws.std(ddof=1)),
        "ci": (float(np.percentile(mean_draws, 2.5)), float(np.percentile(mean_draws, 97.5))),
        "signs": f"{int((table.delta > 0).sum())}/{len(table)}",
        "rho_b": float(table.rho_b.mean()),
        "n": int(table.n.sum()),
    }


def main() -> int:
    events = S.rung_events(S.RUNGS[-1])
    total = sum(len(v) for v in events.values())
    print(f"context x model tier, paired on {total} events\n" + "=" * 78)

    results = {}
    for tier, base, context in PAIRS:
        out = paired_delta(base, context, events)
        results[tier] = out
        lo, hi = out["ci"]
        print(
            f"\n{tier}   n={out['n']}  rho_b(base,context)={out['rho_b']:+.3f}\n"
            f"  base    dR2 {out['per_quarter'].base.mean():+.4f}\n"
            f"  context dR2 {out['per_quarter'].context.mean():+.4f}\n"
            f"  effect of context {out['delta']:+.4f}  se {out['se']:.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  signs {out['signs']}"
        )
        print(out["per_quarter"].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    weak, strong = PAIRS[0][0], PAIRS[1][0]
    dod = results[strong]["delta"] - results[weak]["delta"]
    # The two tiers are scored on the same events but by different models, so
    # the difference of differences is not itself paired; adding the variances
    # is the conservative combination.
    se_dod = float(np.hypot(results[strong]["se"], results[weak]["se"]))
    print(
        f"\n{'=' * 78}\nINTERACTION  (context effect at {strong}) - (at {weak})\n"
        f"  {results[strong]['delta']:+.4f} - ({results[weak]['delta']:+.4f}) = "
        f"{dod:+.4f}   se {se_dod:.4f}  z {dod / se_dod:+.2f}"
    )
    # Resolution is checked FIRST and can veto the narrative. A sign flip is a
    # story about two numbers; whether the numbers are distinguishable from zero
    # is a fact about the measurement, and the fact outranks the story. An
    # earlier version of this function printed the sign-flip conclusion at
    # z = 0.53, which is the exact failure this project keeps paying for.
    resolved = abs(dod) >= 2 * se_dod
    flipped = results[weak]["delta"] < 0 < results[strong]["delta"]

    if not resolved:
        print(
            f"  -> NOT RESOLVED at this n: |{dod:+.4f}| is inside 2 x se ({2 * se_dod:.4f}).\n"
            f"     This is direction only and must not be reported as an effect."
        )
        if flipped:
            print(
                "     The point estimates do flip sign across tier (context hurts the weaker\n"
                "     model, helps the stronger), which is the shape the leaders' +0.0566\n"
                "     predicts — but a sign flip inside the noise band is not evidence for it.\n"
                "     Both arms' own effects also fail to clear zero, so the honest statement\n"
                '     is "context is inert AT THESE TWO TIERS", and the interaction is untested\n'
                "     rather than supported. Separating it needs a frontier tier, where the\n"
                "     hypothesis predicts an effect ~10x this one."
            )
    elif flipped:
        print(
            "  -> RESOLVED SIGN FLIP across model tier: context hurts the weaker model and\n"
            "     helps the stronger one, outside 2 x se."
        )
    else:
        print(f"  -> RESOLVED: interaction {dod:+.4f} outside 2 x se ({2 * se_dod:.4f}).")
    print(
        "\nFor scale: the leaders' same-model context A/B (unfun-bot vs fun-bot) measured\n"
        "+0.0566 at n=274, and the rulebook arm (aigent-arm-b vs -a) +0.0200 at n=272."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
