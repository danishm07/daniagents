"""One command: run the queue, report, re-rank.

    uv run python -m runner.loop                     # every registered arm, all rungs
    uv run python -m runner.loop --arms peers,tfidf  # a subset
    uv run python -m runner.loop --max-rung 0        # screen only, spend nothing new
    uv run python -m runner.loop --dry-run           # what it would cost, and why

The loop is rung-major, not arm-major. Every arm eligible for rung *r* is scored
on the *same* events before anything is promoted, because the fitness is a
*marginal* quantity — what an arm adds depends on what the archive already
holds, and comparing a candidate measured against a three-member archive with
one measured against a five-member archive is not a comparison.

Money is spent in exactly one place: :meth:`Arm.ensure`, and only for events the
arm has not already generated. Everything downstream — scoring, ρ_b, the
ensemble fitness, the bootstrap — is arithmetic on cached columns and is free.
That is what makes screening at rung 1 worth doing: it is the *generation* that
costs, and a screened-out arm never generates the other 1,700 events.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner import archive as A  # noqa: E402
from runner import fitness as F  # noqa: E402
from runner import registry as R  # noqa: E402
from runner import report as REP  # noqa: E402
from runner import schedule as S  # noqa: E402
from runner import score as SC  # noqa: E402
from runner import arms_builtin  # noqa: E402,F401  — registration side effect

#: Minimum confirmation-set rows before an arm gets a confirmation number at
#: all. Below this the number describes the arm's cache coverage rather than
#: its skill.
MIN_CONFIRM_N = 500


def eligible(arm: R.Arm, state: S.State) -> tuple[bool, str]:
    """Should this arm be run at all? Deployability first, cost second.

    An arm that cannot ship is not a candidate. Measuring one anyway is how a
    research programme accumulates numbers that do not transfer — which this
    project has already paid for once, when the offline champion was a proxy
    running a different model and a different prompt.
    """
    if arm.killed:
        return False, f"killed: {arm.killed}"
    if arm.name in state.dropped:
        return False, state.dropped[arm.name]
    check = arm.live_check()
    if not check.ok:
        return False, str(check)
    if not arm.cutoff_safe:
        return False, "declared not cutoff-safe"
    return True, ""


def run_rung(
    rung: int,
    arms: list[R.Arm],
    state: S.State,
    archive: A.Archive,
    *,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, SC.ArmScore]:
    """Generate, score and rank every arm eligible for ``rung``."""
    n = S.rung_size(rung)
    events = S.rung_events(n)
    total = sum(len(v) for v in events.values())
    cohort = [a for a in arms if state.rung.get(a.name, -1) == rung - 1]
    if not cohort:
        return {}

    if verbose:
        print(f"\n{'=' * 100}\nrung {rung}  n={total}  "
              f"{'decision-grade' if S.can_decide(rung) else 'direction only'}  "
              f"cohort {len(cohort)}: {', '.join(a.name for a in cohort)}\n{'=' * 100}")

    # -- generation: the only step that costs money -----------------------
    for arm in cohort:
        todo = S.new_events(state.rung.get(arm.name) if state.rung.get(arm.name, -1) >= 0 else None, rung)
        n_new = sum(len(v) for v in todo.values())
        estimate = arm.estimated_cost(n_new)
        if estimate > 0 and verbose:
            print(f"  [{arm.name}] {n_new} new events, estimated ${estimate:.2f}")
        if estimate > S.budget_remaining():
            state.drop(arm.name, f"would cost ${estimate:.2f}, ${S.budget_remaining():.2f} left")
            if verbose:
                print(f"  [{arm.name}] SKIPPED — {state.dropped[arm.name]}")
            continue
        if dry_run or not n_new:
            continue
        for quarter, batch in todo.items():
            if batch:
                arm.ensure(batch, quarter)

    cohort = [a for a in cohort if a.name not in state.dropped]
    if dry_run:
        return {}

    # -- scoring: free ----------------------------------------------------
    results = {a.name: SC.score_arm(a, events) for a in cohort}
    target = SC.target_residuals(events)

    # The incumbent set the marginal contribution is measured against: the
    # champion always, plus whatever currently holds an archive cell, all
    # re-scored on this rung's events so the ensemble arithmetic is on one
    # sample rather than three.
    champion = SC.champion_score(events)
    incumbents = {"champion": champion.residuals}
    for name in archive.members():
        if name in results:
            continue
        arm = R.ARMS.get(name)
        if arm is not None and arm.cost_usd_per_event == 0 or (arm and _fully_cached(arm, events)):
            incumbents[name] = SC.score_arm(arm, events).residuals

    # An arm that answered too little to score anywhere is a coverage failure,
    # not a null result. Letting it into the archive with a marginal of exactly
    # zero would park it in a cell it never earned.
    #
    # But at a small rung "too little" is often an artefact of the rung, not of
    # the arm: a column covering 500 events per quarter has under 30 rows inside
    # a 100-per-quarter rung and scores nothing, then has 500 inside a
    # 667-per-quarter one. Six read columns were being retired permanently on
    # exactly that. Below the selection rung the arm is skipped and advanced;
    # only at decision grade is a coverage failure a verdict.
    for name in [n for n, r in results.items() if r.n == 0]:
        if S.can_decide(rung):
            state.drop(name, f"no scorable rows at n={total} — coverage, not score")
        else:
            state.record(name, rung, float("-inf"), float("nan"))
            print(f"  [{name}] too few rows inside rung {rung}; carried to the next rung")
        results.pop(name)

    marginals, decisions = {}, {}
    for name, result in results.items():
        others = {k: v for k, v in incumbents.items() if k != name}
        marginals[name] = F.marginal(name, result.residuals, others, target)
        arm = R.ARMS[name]
        decisions[name] = REP.decision(arm, result, rung)
        REP.log_run(arm, result, rung, marginals[name])

        state.record(name, rung, marginals[name].marginal_rho, result.delta_r2)
        archive.consider(_elite(arm, result, marginals[name], rung))

    # -- refresh the incumbents ------------------------------------------
    #
    # A MAP-Elites archive without elite re-evaluation is a winner's-curse
    # amplifier: every cell independently keeps the argmax over noisy
    # evaluations, so C occupied cells run C simultaneous selections and the
    # bias grows with archive size — making the grid *finer* makes the estimates
    # *worse*. The uncertain-QD literature benchmarks vanilla MAP-Elites as the
    # worst performer under noise for exactly this reason, and prescribes
    # spending a fixed share of each generation re-evaluating incumbents.
    #
    # Here that share is free: an incumbent's column is already cached, so
    # re-scoring it at the new rung costs nothing. There is a second, unrelated
    # reason it is necessary — marginal contribution is defined against the
    # *current* archive, so every insertion silently stales every stored
    # fitness. An arm admitted early against a weak incumbent set keeps a number
    # it could not earn today.
    stale = [n for n in archive.members() if n not in results and n in R.ARMS]
    for name in stale:
        arm = R.ARMS[name]
        if arm.cost_usd_per_event > 0:
            continue  # re-evaluating a metered arm is not free; skip it
        refreshed = SC.score_arm(arm, events)
        if refreshed.n == 0:
            continue
        others = {k: v for k, v in incumbents.items() if k != name}
        m = F.marginal(name, refreshed.residuals, others, target)
        archive.consider(_elite(arm, refreshed, m, rung))
    if verbose and stale:
        print(f"  re-evaluated {len(stale)} incumbent elites at rung {rung}")

    state.save()
    archive.save()

    # -- ASHA: who survives to the next rung ------------------------------
    if rung + 1 < len(S.RUNGS):
        holders = set(archive.members())
        for name in results:
            ok, why = S.promotable(
                state,
                name,
                rung,
                holds_cell=name in holders,
                costed=R.ARMS[name].cost_usd_per_event > 0,
            )
            if not ok and "need" not in why:
                state.drop(name, f"rung {rung}: {why}")
        state.save()

    if verbose:
        table = REP.matrix(results, marginals, {k: rung for k in results}, decisions)
        rho_b = F.inter_arm_matrix({**{k: v.residuals for k, v in results.items()},
                                    "champion": champion.residuals})
        print(REP.render(table, rho_b, archive, title=f"rung {rung} (n={total})"))
        for name in results:
            if name in state.dropped:
                print(f"  dropped: {name} — {state.dropped[name]}")
    return results


def _elite(arm: R.Arm, result: SC.ArmScore, m: F.Marginal, rung: int) -> A.Elite:
    return A.Elite(
        arm=arm.name,
        family=arm.family,
        tier=arm.tier,
        rung=rung,
        n=result.n,
        delta_r2=result.delta_r2,
        pct_obtainable=result.pct_obtainable,
        rho=result.rho,
        rho_b_champion=result.rho_b_champion,
        vs_champion=result.vs_champion,
        marginal_rho=m.marginal_rho,
        marginal_delta_r2=m.marginal_delta_r2,
        coverage=result.coverage,
        neutral_rate=result.neutral_rate,
        cost_usd=result.cost_usd,
        live_computable=arm.live_check().ok,
        cutoff_safe=arm.cutoff_safe,
        signs=result.signs,
    )


def confirm(archive: A.Archive, verbose: bool = True) -> pd.DataFrame:
    """Score the archive's elites on the partition selection never saw.

    The gap between an arm's selection-set marginal and its confirmation-set
    marginal **is** the selection bias — measured, not estimated from a
    best-of-K formula whose independence assumption our arms violate.

    Run once per generation, after the rungs. Metered arms are skipped: their
    columns do not cover the confirmation events, and generating them there
    would be paying to check a number rather than to find one.
    """
    events = S.confirmation_events()
    total = sum(len(v) for v in events.values())
    target = SC.target_residuals(events)
    champion = SC.champion_score(events)

    # An arm needs real coverage here or its confirmation number measures its
    # cache, not its skill. The cached LLM columns stop at ~2,100 events — the
    # rung set plus a hundred spill — so on this partition they have n≈100 and
    # a marginal that rounds to zero for want of data. Reporting that as
    # "did not survive confirmation" would be the same coverage-as-verdict error
    # the rungs already had to fix once.
    scored, rows, unevaluable = {}, [], []
    for name in archive.members():
        arm = R.ARMS.get(name)
        if arm is None or arm.cost_usd_per_event > 0:
            continue
        result = SC.score_arm(arm, events)
        if result.n >= MIN_CONFIRM_N:
            scored[name] = result
        else:
            unevaluable.append((name, result.n))

    incumbents = {"champion": champion.residuals}
    incumbents.update({k: v.residuals for k, v in scored.items()})
    for name, result in scored.items():
        others = {k: v for k, v in incumbents.items() if k != name}
        m = F.marginal(name, result.residuals, others, target)
        elite = next(e for e in archive.elites() if e.arm == name)
        rows.append(
            {
                "arm": name,
                "n_confirm": result.n,
                "sel_marginal": elite.marginal_rho,
                "cnf_marginal": m.marginal_rho,
                "bias": elite.marginal_rho - m.marginal_rho,
                "sel_delta_r2": elite.delta_r2,
                "cnf_delta_r2": result.delta_r2,
                "sel_rho_b": elite.rho_b_champion,
                "cnf_rho_b": result.rho_b_champion,
            }
        )
    frame = pd.DataFrame(rows).sort_values("cnf_marginal", ascending=False)
    if verbose and len(frame):
        fmt = lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v)
        print(f"\n{'=' * 100}\nCONFIRMATION SET — {total} dev events no rung touched\n{'=' * 100}")
        print(frame.to_string(index=False, float_format=fmt))
        kept = int((frame.cnf_marginal > 0).sum())
        print(
            f"\nmean selection bias {frame.bias.mean():+.4f} over {len(frame)} evaluable elites; "
            f"{kept}/{len(frame)} keep a positive marginal off the selection set."
        )
        if kept == 0:
            print(
                "  Every evaluable arm's marginal contribution is selection bias. The archive's\n"
                "  ordering is a ranking of noise, and the honest value of this generation is\n"
                "  zero — which is a result, not a failure of the run."
            )
        if unevaluable:
            print(
                f"\n  not evaluable here ({len(unevaluable)}) — cached column does not reach this "
                f"partition, so no confirmation number exists at any price but a new run:\n    "
                + ", ".join(f"{n} (n={c})" for n, c in unevaluable)
            )
    return frame


def _fully_cached(arm: R.Arm, events: dict[str, list[dict]]) -> bool:
    """True if scoring this arm spends nothing — every event already on disk."""
    for quarter, batch in events.items():
        values = arm.predict(batch, quarter)
        if any(v != v for v in values):  # NaN == a hole
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="", help="comma-separated subset; default all registered")
    parser.add_argument("--max-rung", type=int, default=len(S.RUNGS) - 1)
    parser.add_argument("--dry-run", action="store_true", help="cost and eligibility only")
    parser.add_argument("--reset", action="store_true", help="forget the schedule, keep the columns")
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    args = parser.parse_args(argv)

    if args.list:
        print(pd.DataFrame([a.describe() for a in R.ARMS.values()]).to_string(index=False))
        print(f"\nfeatures:\n{pd.DataFrame(__import__('runner.features', fromlist=['x']).table()).to_string(index=False)}")
        return 0

    if args.reset:
        S.STATE.unlink(missing_ok=True)
        A.STATE.unlink(missing_ok=True)

    state = S.State.load()
    archive = A.Archive.load()

    wanted = [s.strip() for s in args.arms.split(",") if s.strip()]
    arms = [R.get(w) for w in wanted] if wanted else R.live()

    print(f"{len(arms)} arms registered, "
          f"${max(S.spent_usd(), S.SPEND_PRIOR):.2f} of ${S.SPEND_CAP:.2f} spent")
    for arm in arms:
        ok, why = eligible(arm, state)
        if not ok:
            state.drop(arm.name, why)
            print(f"  ineligible: {arm.name} — {why}")
    arms = [a for a in arms if a.name not in state.dropped]
    for arm in arms:
        state.rung.setdefault(arm.name, -1)

    for rung in range(0, args.max_rung + 1):
        run_rung(rung, arms, state, archive, dry_run=args.dry_run)

    if not args.dry_run:
        confirm(archive)
        print("\n" + "=" * 100)
        print("FINAL ARCHIVE")
        print("=" * 100)
        for elite in archive.elites():
            print(f"  [{'|'.join(elite.cell):<34}] {elite.arm:<24} "
                  f"marginal {elite.marginal_rho:+.4f}  dR2 {elite.delta_r2:+.4f}  "
                  f"rho_b {elite.rho_b_champion:+.3f}  rung {elite.rung}  n={elite.n}")
        print(f"\n{len(state.dropped)} arms dropped:")
        for name, why in state.dropped.items():
            print(f"  {name:<24} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
