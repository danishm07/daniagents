"""H1: does restoring DSPy's step-by-step instruction recover the published baseline?

The sole surviving explanation for our replay of the published repo scoring
**1.05% of obtainable** against the paper's own **5.8%** for the same
architecture. H2 (floating model alias) and H3 (output quantisation) are both
killed; H4 explained the *leaderboard* gap but not this one.

**The mechanism, confirmed at zero cost.** ``dspy/predict/chain_of_thought.py``
in dspy 3.3.0::

    desc = "${reasoning}"
    rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)

That bare placeholder renders as an **empty** field description, so the prompt
carries ``1. `reasoning` (str):`` and nothing after it. Issue #409 shows what
older DSPy put there instead::

    Reasoning: Let's think step by step in order to ${produce the answer}. We ...

Passing an explicit ``rationale_field`` restores it — verified by dumping both
prompts through ``ChatAdapter`` before spending anything:

    DEFAULT  ->  1. `reasoning` (str):
    H1 FIX   ->  1. `reasoning` (str): Let's think step by step in order to ...

So the *prompt* difference is established. What is not established, and is what
this script costs money to find out, is whether it moves the **score**.

Paired against the existing ``data/reads/official__gpt5nano__2026Q2.jsonl``
column: same 700 events, same pinned model (``gpt-5-nano-2025-08-07``), same
program, one field description different.

Routes through the **OpenAI** key, not OpenRouter — ``dspy.LM("openai/...")``
goes direct — so it does not touch the capped OpenRouter key.

    uv run python h1_rationale.py --limit 700
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dspy  # noqa: E402

import baseline_program as BP  # noqa: E402
import eval as E  # noqa: E402
import harness  # noqa: E402
import reads  # noqa: E402
from runner import ledger  # noqa: E402

#: Reconstructed from dspy issue #409, which shows the pre-regression prompt
#: verbatim. The trailing "We ..." is part of it — it cues continuation rather
#: than a one-line answer.
RATIONALE = "Let's think step by step in order to produce the answer. We ..."

OUT = ROOT / "data" / "reads" / "official__gpt5nano__2026Q2__rationale.jsonl"


def run(events: list[dict], threads: int = 8) -> dict[str, float]:
    """The published program with one field description restored."""
    lm = dspy.LM(BP.LM_MODELS["gpt5nano"], timeout=120, cache=False)
    predictor = dspy.ChainOfThought(
        BP.PredictEarningsReturn, rationale_field=dspy.OutputField(desc=RATIONALE)
    )

    done = {}
    if OUT.exists():
        done = {json.loads(l)["event_id"]: json.loads(l)["prediction"]
                for l in OUT.open() if l.strip()}
    todo = [e for e in events if e["event_id"] not in done]
    print(f"cached {len(done)}, to run {len(todo)}")
    if not todo:
        return done

    def one(event):
        if not event["facts"]:
            return event["event_id"], BP.NEUTRAL_PERCENTILE, 0, 0
        try:
            with dspy.context(lm=lm):
                out = predictor(
                    key_facts_discussed_in_earnings_call=BP.format_facts(event["facts"])
                )
            value = BP.normalize_percentile(float(out.predict_percentile))
            usage = {}
            try:
                usage = (lm.history[-1] or {}).get("usage") or {}
            except Exception:
                pass
            return (event["event_id"], value,
                    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        except Exception as exc:
            print(f"  [fail] {event['event_id']}: {type(exc).__name__}: {str(exc)[:90]}",
                  flush=True)
            return event["event_id"], BP.NEUTRAL_PERCENTILE, 0, 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    started, tin, tout = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=threads) as pool, OUT.open("a") as fh:
        for i, (event_id, value, pin, pout) in enumerate(pool.map(one, todo), 1):
            done[event_id] = value
            tin += pin
            tout += pout
            fh.write(json.dumps({"event_id": event_id, "prediction": value,
                                 "prompt_tokens": pin, "completion_tokens": pout}) + "\n")
            if i % 100 == 0:
                rate = i / (time.time() - started)
                print(f"  {i}/{len(todo)}  {rate:.1f}/s  "
                      f"eta {(len(todo) - i) / rate / 60:.1f}m", flush=True)

    # One accounting path, whatever wrote the column.
    if tin or tout:
        usd = ledger.record("openai/gpt-5-nano-2025-08-07", tin, tout,
                            source="h1.rationale_field",
                            note="H1 test: restored step-by-step rationale desc")
        print(f"\nlogged to ledger: {tin} in / {tout} out -> ${usd:.4f}")
    return done


def score(quarter: str = "2026Q2") -> None:
    frame = harness.load(quarter)
    baseline_path = ROOT / "data" / "reads" / "official__gpt5nano__2026Q2.jsonl"
    baseline = {json.loads(l)["event_id"]: json.loads(l)["prediction"]
                for l in baseline_path.open() if l.strip()}
    fixed = {json.loads(l)["event_id"]: json.loads(l)["prediction"]
             for l in OUT.open() if l.strip()} if OUT.exists() else {}

    both = set(baseline) & set(fixed)
    sub = frame[frame.event_id.isin(both)].copy()
    if len(sub) < 100:
        print(f"only {len(sub)} paired events — not scorable yet")
        return
    sub["_base"] = sub.event_id.map(baseline)
    sub["_fix"] = sub.event_id.map(fixed)

    y = sub.y.to_numpy(dtype=float)
    surprise = sub.surprise_pct.to_numpy(dtype=float)
    print(f"\nH1 — paired on {len(sub)} events of {quarter}, same pinned model\n" + "=" * 72)
    for label, col in (("published (blank reasoning)", "_base"),
                       ("H1 fix (rationale restored)", "_fix")):
        scored = harness.evaluate(sub, col)
        pct = E.as_pct_obtainable(scored["delta_r_squared"], scored["r_squared_surprise"])
        rho = E.partial_corr(sub[col].to_numpy(dtype=float), y, surprise)
        print(f"  {label:<30} dR2 {scored['delta_r_squared']:+.4f}  "
              f"= {pct:.2%} of obtainable   rho {rho:+.4f}   "
              f"distinct {sub[col].nunique()}")

    a = sub["_fix"].to_numpy(dtype=float)
    b = sub["_base"].to_numpy(dtype=float)
    gain = E._delta_r2_fast(a, surprise, y) - E._delta_r2_fast(b, surprise, y)
    rng = np.random.default_rng(0)
    n = len(y)
    draws = [
        E._delta_r2_fast(a[i], surprise[i], y[i]) - E._delta_r2_fast(b[i], surprise[i], y[i])
        for i in (rng.integers(0, n, n) for _ in range(2000))
    ]
    se = float(np.std(draws, ddof=1))
    matrix = E._correlation_matrix({"a": a, "b": b}, surprise)
    print(f"\n  paired gain {gain:+.4f}  se {se:.4f}  z {gain / se:+.2f}")
    print(f"  rho_b between the two columns {matrix.loc['a', 'b']:+.4f}")
    print(f"  target: the paper's own figure for this architecture is 5.8% of obtainable")
    # Resolution before narrative — the standing rule.
    if abs(gain) < 2 * se:
        print("\n  => NOT RESOLVED: the gain is inside 2 x se. Direction only.")
    else:
        print(f"\n  => RESOLVED: {'restoring the rationale helps' if gain > 0 else 'it does not help'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=700)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()

    if not args.score_only:
        screen = {e["event_id"] for e in reads.screen_events(700)}
        events = [e for e in harness.events_for("2026Q2") if e["event_id"] in screen][: args.limit]
        print(f"H1: {len(events)} events, pinned {BP.LM_MODELS['gpt5nano']}, "
              f"rationale = {RATIONALE!r}\n")
        run(events, args.threads)
    score()
