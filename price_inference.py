"""Block 1: what does an ACE-style inference call actually cost and how long does it take?

Measured, not read off a rate card. The decision this settles: only the
Opus-at-inference path reaches the 11-15% band (Table 8's transfer numbers all
sit *below* un-ACE'd Opus), so if Opus is unaffordable or too slow we are
building toward ~7% and that should be a decision rather than a discovery.

Three things get measured per model:

``tokens``    actual prompt and completion tokens on a real ten-fact payload
              plus a realistic rulebook, not an estimate
``dollars``   those tokens at the published price, then multiplied by the
              ~300 remaining contest events
``latency``   the distribution over repeated calls, not one sample. Production
              budgets 270s worst case inside a 5-minute deadline
              (15 + 120x2 + 15), so the number that matters is the tail.

    uv run python price_inference.py --calls 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
import reads  # noqa: E402

#: Contest events still to come. Coverage projections put final contest
#: coverage at 90-94% against a 2026-10-09 freeze.
REMAINING_EVENTS = 300

#: Production's own worst case: 15s summary + 2x120s LLM + 15s = 270s inside the
#: 5-minute window. A model whose p95 exceeds ~120s cannot be retried once and
#: still land.
SINGLE_CALL_BUDGET_S = 120.0

MODELS = {
    "anthropic/claude-opus-4.5": {"reasoning": True},
    "anthropic/claude-sonnet-4.5": {"reasoning": False},
    "anthropic/claude-haiku-4.5": {"reasoning": False},
    "google/gemini-2.5-flash-lite": {"reasoning": False},
}


def synthetic_rulebook(n_rules: int = 80) -> str:
    """A rulebook of the size and shape ACE would actually produce.

    Rules follow Koijen & Levy's Figure 3 form — condition(s) -> percentile band
    with amplifiers — because token count depends on shape, and a rulebook of
    terse one-liners would understate the context an real one carries.
    """
    seeds = [
        ("pre-profit company reports revenue growth >30% YoY BUT operating losses remain flat "
         "or worsen (EBITDA not narrowing proportionally)", "0.10-0.20",
         "forward guidance suggests growth momentum peaking (flat sequential); hot theme "
         "sector; customer conversion below 15%"),
        ("management uses cautionary forward language ('aggressive consensus', 'gradual "
         "recovery', guiding to low end)", "0.35-0.50",
         "forward quarter guidance represents >50% deceleration from recent organic growth"),
        ("record or near-record margins include acknowledged non-recurring benefits (metals "
         "revaluation, inventory, one-time mix) AND a concrete operational execution failure "
         "in the same quarter AND a major geographic segment declining >15%", "0.10-0.20",
         "multi-sector demand narratives present but do not offset"),
        ("headline EPS beat is driven by a lower share count from buybacks while absolute net "
         "income is flat or down", "0.30-0.45", "no guidance raise accompanies it"),
        ("gross margin expands sequentially AND operating cash flow conversion improves AND "
         "guidance is raised above the prior range", "0.75-0.90",
         "the raise is attributed to demand rather than to cost"),
        ("revenue beat but the beat is entirely price/mix with volumes declining", "0.25-0.40",
         "management describes elasticity or trade-down behaviour"),
        ("segment previously described as a growth driver decelerates below corporate average",
         "0.20-0.35", "management stops disclosing the segment's growth rate"),
        ("net debt reduced materially ahead of a stated target AND leverage falls toward the "
         "long-term goal", "0.55-0.70", "no dilution and dividend maintained"),
    ]
    lines = []
    for i in range(n_rules):
        condition, band, amplifier = seeds[i % len(seeds)]
        lines.append(
            f"[rul-{i:05d}] helpful={i % 7} harmful={i % 3} :: When {condition}, apply {band} "
            f"range. Amplified when: {amplifier}."
        )
    return "## RULEBOOK\n" + "\n".join(lines)


SYSTEM = (
    "You are an equity analyst predicting how the market will react to an earnings "
    "announcement. Apply the rulebook. Return JSON with keys reasoning, bullet_ids, "
    "predicted_percentile (a float in [0,1])."
)


def measure(model: str, rulebook: str, facts: list[str], ticker: str,
            calls: int, reasoning: bool) -> dict:
    client = reads._client()
    user = (
        f"{rulebook}\n\nEVENT — {ticker}\nFacts from the earnings call:\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\n\nReturn JSON: {\"reasoning\": str, \"bullet_ids\": [str], "
          "\"predicted_percentile\": float}"
    )
    latencies, prompt_tokens, completion_tokens, failures = [], [], [], 0

    for _ in range(calls):
        kwargs = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}],
            "max_tokens": 1200,
        }
        if reasoning:
            # High reasoning effort is the configuration the paper's headline
            # uses, and it is the one that threatens the deadline.
            kwargs["extra_body"] = {"reasoning": {"max_tokens": 2000}}
        started = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            failures += 1
            print(f"    [fail] {type(exc).__name__}: {str(exc)[:120]}")
            continue
        latencies.append(time.time() - started)
        if resp.usage:
            prompt_tokens.append(resp.usage.prompt_tokens)
            completion_tokens.append(resp.usage.completion_tokens)

    if not latencies:
        return {"model": model, "ok": False, "failures": failures}

    import frontier

    price_in, price_out = frontier.PRICES.get(model, (1.0, 5.0))
    mean_in = statistics.mean(prompt_tokens) if prompt_tokens else float("nan")
    mean_out = statistics.mean(completion_tokens) if completion_tokens else float("nan")
    per_event = mean_in * price_in / 1e6 + mean_out * price_out / 1e6
    ordered = sorted(latencies)
    p95 = ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]
    return {
        "model": model,
        "ok": True,
        "calls": len(latencies),
        "failures": failures,
        "prompt_tokens": round(mean_in),
        "completion_tokens": round(mean_out),
        "usd_per_event": per_event,
        "usd_300_events": per_event * REMAINING_EVENTS,
        "latency_mean_s": statistics.mean(latencies),
        "latency_median_s": statistics.median(latencies),
        "latency_max_s": max(latencies),
        "latency_p95_s": p95,
        "fits_with_one_retry": p95 * 2 + 30 < 300,
        "fits_single_call": p95 < SINGLE_CALL_BUDGET_S,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--rules", type=int, default=80)
    parser.add_argument("--models", default="")
    args = parser.parse_args()

    rulebook = synthetic_rulebook(args.rules)
    event = harness.events_for("2026Q2")[0]
    approx_tokens = len(rulebook) // 4
    print(f"rulebook: {args.rules} rules, {len(rulebook)} chars, ~{approx_tokens} tokens")
    print(f"event: {event['ticker']} with {len(event['facts'])} facts")
    print(f"pricing over {REMAINING_EVENTS} remaining contest events, "
          f"{args.calls} calls per model\n")

    wanted = [m.strip() for m in args.models.split(",") if m.strip()] or list(MODELS)
    rows = []
    for model in wanted:
        config = MODELS.get(model, {"reasoning": False})
        label = model + (" [reasoning]" if config["reasoning"] else "")
        print(f"  measuring {label} ...")
        result = measure(model, rulebook, event["facts"], event["ticker"],
                         args.calls, config["reasoning"])
        result["reasoning"] = config["reasoning"]
        rows.append(result)
        if result.get("ok"):
            print(f"    in {result['prompt_tokens']} out {result['completion_tokens']} "
                  f"| ${result['usd_per_event']:.4f}/event "
                  f"| ${result['usd_300_events']:.2f} for {REMAINING_EVENTS} "
                  f"| p95 {result['latency_p95_s']:.1f}s")

    out = ROOT / "data" / "ace" / "inference_pricing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwritten to {out}")

    import pandas as pd

    frame = pd.DataFrame([r for r in rows if r.get("ok")])
    if len(frame):
        cols = ["model", "reasoning", "prompt_tokens", "completion_tokens", "usd_per_event",
                "usd_300_events", "latency_median_s", "latency_p95_s", "latency_max_s",
                "fits_single_call", "fits_with_one_retry"]
        print()
        print(frame[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
