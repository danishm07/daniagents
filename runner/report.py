"""The full matrix — including the arms that lost.

``BUILD_LOOP.md`` on reporting:

    Per arm: ΔR², % of obtainable, ρ, ρ_b vs champion, marginal ensemble
    contribution, neutral-rate, n, cost, live-computable, cutoff-compliant, and
    the K it was judged at. [...] Negative results with equal prominence. State
    how many configurations were tried to reach any headline number.

So this module prints one table with every arm in it, sorted by fitness, with no
"top results" section. K is read off the run log rather than counted by hand,
and the promotion floor is computed per candidate from a paired bootstrap —
because the sd of the difference depends on how much the challenger resembles
the champion, and is measured at 0.0008–0.0047 rather than the 0.010 once
assumed.

Every arm scored here is appended to ``eval.RUN_LOG`` under its own config hash,
so the next run's K includes it. That is the ledger, not a brake: it makes "we
tried forty things and the best cleared the floor" a statement someone can
check.
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
from runner import archive as A  # noqa: E402
from runner import fitness as F  # noqa: E402
from runner import registry as R  # noqa: E402
from runner import schedule as S  # noqa: E402
from runner.score import ArmScore  # noqa: E402


def log_run(arm: R.Arm, result: ArmScore, rung: int, marginal: F.Marginal | None) -> str:
    """Append this evaluation to the shared run log and return its config hash.

    Deliberately shares ``eval.RUN_LOG`` with everything else in the project
    rather than keeping a runner-local log. K is a property of how many looks
    the *data* has had, not of which script took them.
    """
    import hashlib
    import json
    import uuid
    from datetime import datetime, timezone

    config = {"arm": arm.name, "family": arm.family, "model": arm.model, **arm.config}
    config_hash = hashlib.sha256(
        json.dumps({"name": arm.name, "config": config}, sort_keys=True, default=str).encode()
    ).hexdigest()
    record = {
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "run",
        "name": f"runner:{arm.name}",
        "status": "ok" if result.error is None else "error",
        "config_hash": config_hash,
        "config": config,
        "views": [arm.name],
        "quarters": sorted(result.residuals),
        "spent_holdout": False,
        "champion": E.default_champion(quiet=True),
        "cost_usd": result.cost_usd,
        "notes": f"rung {rung} (n={result.n})",
        "rung": rung,
        "summary": [result.row()],
        "marginal": marginal.row() if marginal else None,
        "git": E._git_commit(),
    }
    if result.error:
        record["error"] = result.error
    E._append(record)
    return config_hash


def decision(arm: R.Arm, result: ArmScore, rung: int) -> dict:
    """Bootstrapped, K-aware verdict on replacing the champion.

    Refuses below the selection rung. The noise table says a rung-1 comparison
    cannot tell a ρ=0.25 arm from a ρ=0.15 one, so a "decision" there would be a
    coin flip wearing a p-value.
    """
    if not S.can_decide(rung):
        return {
            "arm": arm.name,
            "ship": False,
            "reason": f"rung {rung} (n={result.n}) is below the n>=2000 selection rung",
        }

    champion_column = E.default_champion(quiet=True)
    frames, challenger, incumbent = {}, {}, {}
    for quarter, raw in result.raw.items():
        frame = harness.load(quarter).set_index("event_id").reindex(raw.index)
        frames[quarter] = frame.reset_index()
        challenger[quarter] = raw.to_numpy(dtype=float)
        incumbent[quarter] = frame[champion_column].to_numpy(dtype=float)

    boot = E.bootstrap_diff(challenger, incumbent, frames, n_boot=2000)
    k = E.config_count()
    floor = E.promotion_floor(boot["se_mean"], k)
    gain = result.vs_champion
    return {
        "arm": arm.name,
        "mean_vs_champion": gain,
        "signs": result.signs,
        "se_mean": boot["se_mean"],
        "paired_n": boot["paired_n"],
        "k_configs": k,
        "best_of_k_multiplier": max(E.expected_best_of_k(k), E.Z_SINGLE_TEST),
        "floor": floor,
        "ship": bool(gain > floor),
        "reason": f"gain {gain:+.4f} vs floor {floor:+.4f} at K={k}",
    }


def matrix(
    results: dict[str, ArmScore],
    marginals: dict[str, F.Marginal],
    rungs: dict[str, int],
    decisions: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """One row per arm, every column BUILD_LOOP.md asks for, losers included."""
    decisions = decisions or {}
    rows = []
    for name, result in results.items():
        arm = R.ARMS.get(name)
        check = arm.live_check() if arm else None
        m = marginals.get(name)
        d = decisions.get(name, {})
        rows.append(
            {
                "arm": name,
                "family": arm.family if arm else "-",
                "rung": rungs.get(name, -1),
                "n": result.n,
                "delta_r2": result.delta_r2,
                "pct_obt": result.pct_obtainable,
                "rho": result.rho,
                "rho_b_champ": result.rho_b_champion,
                "band": A.rho_b_band(result.rho_b_champion),
                "vs_champ": result.vs_champion,
                "signs": result.signs,
                "marginal_rho": m.marginal_rho if m else float("nan"),
                "marginal_dr2": m.marginal_delta_r2 if m else float("nan"),
                "insample_gain": (m.rho_with_insample - m.rho_without_insample) if m else float("nan"),
                "coverage": result.coverage,
                "neutral": result.neutral_rate,
                "cost_usd": result.cost_usd,
                "live": bool(check.ok) if check else False,
                "cutoff_safe": arm.cutoff_safe if arm else False,
                "K": d.get("k_configs", float("nan")),
                "floor": d.get("floor", float("nan")),
                "ship": d.get("ship", False),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["marginal_rho", "delta_r2"], ascending=False).reset_index(drop=True)


def render(
    table: pd.DataFrame,
    rho_b: pd.DataFrame | None = None,
    archive=None,
    *,
    title: str = "runner",
) -> str:
    """The report as a string, so it can be printed and written to the notes."""
    fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) and np.isfinite(v) else str(v)
    out = [f"\n{'=' * 100}", f"{title}", "=" * 100]

    if table.empty:
        out.append("no arms scored")
        return "\n".join(out)

    out.append("\nall arms, sorted by marginal ensemble contribution "
               "(negatives are not omitted — they are the result)\n")
    out.append(table.to_string(index=False, float_format=fmt))

    positive = table[table.marginal_rho > 0]
    out.append(
        f"\n{len(positive)}/{len(table)} arms have a positive marginal contribution. "
        f"K to date: {int(E.config_count())} distinct configurations logged."
    )
    over = table[table.insample_gain > 2 * table.marginal_rho.clip(lower=1e-9)]
    if len(over):
        out.append(
            f"in-sample gain more than double the cross-validated gain on "
            f"{len(over)} arms ({', '.join(over.arm)}) — that gap is fitted noise, not signal."
        )

    dead = table[~table.live | ~table.cutoff_safe]
    if len(dead):
        out.append(
            f"\nnot deployable ({len(dead)}): "
            + ", ".join(f"{r.arm} ({'not live' if not r.live else 'cutoff'})" for r in dead.itertuples())
            + "\n  Their numbers are diagnostics, not candidates."
        )

    if rho_b is not None and len(rho_b) > 1:
        out.append("\ninter-arm rho_b (surprise projected out, pooled Fisher-z)\n")
        out.append(rho_b.to_string(float_format=lambda v: f"{v:+.3f}"))

    if archive is not None:
        elites = archive.elites()
        out.append(f"\nMAP-Elites archive: {len(elites)} cells occupied")
        for elite in elites:
            out.append(
                f"  [{'|'.join(elite.cell):<34}] {elite.arm:<22} "
                f"marginal {elite.marginal_rho:+.4f}  dR2 {elite.delta_r2:+.4f}  "
                f"rho_b {elite.rho_b_champion:+.3f}  n={elite.n}"
            )
        if archive.best:
            out.append(f"  best overall: {archive.best.arm} "
                       f"(marginal {archive.best.marginal_rho:+.4f})")

    out.append(
        f"\nmetered spend ${max(S.spent_usd(), S.SPEND_PRIOR):.2f} of ${S.SPEND_CAP:.2f}"
    )
    return "\n".join(out)
