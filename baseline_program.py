"""The official baseline's DSPy program, copied verbatim from the public repo.

Source: ``explaining-markets/baseline-earnings-summary``,
``src/em_baseline/predictor.py`` @ 9751a4b. Its own docstring states the program
is "a direct port of the research pipeline's DSPy program (signature, prompt
text, and percentile normalization are verbatim)", and the repo contains **no
compiled artifact** — no ``.load()``, no saved demos, no teleprompter. It is
zero-shot, so copying the signature copies the whole program.

Why this file exists rather than an optimiser run: the read sweep showed our own
prompt scores 2.5–3.5% of obtainable on *every* model tested, while this program
scores 4.1–5.0% on two — non-overlapping ranges, same +1.66pp swing on two
independent vendors. The deficit is the prompt, and the prompt was public the
whole time.

Three things here that our prompt does not have, and the reason to adopt
verbatim rather than cherry-pick:

* **base rates** to calibrate against (25 / 50 / 25);
* **class↔percentile consistency constraints**, which the source says are the
  calibration mechanism — that is why ``predict_class`` and ``rationale`` are
  requested at all, so dropping them to keep a single float would remove the
  scaffolding, not just the logging;
* an **anti-pattern instruction** (never cite fact numbers).

⚠️ The bands quantize the output into three ranges, which is the likely source
of the baselines' 24–62 distinct values over ~2,000 events. Pearson pays for
spacing, so that coarseness has a direct ΔR² cost. Adopt first, then widen
granularity while keeping the calibration — that is a hypothesis with a
mechanism, and it is cheaper to test than a general optimisation run.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import dspy

AGENT = Path(__file__).parent.parent / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "src")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT / ".env")

import eval as E  # noqa: E402
import harness  # noqa: E402

#: Verbatim from the source. ``gemini-flash-lite-latest`` is a floating alias,
#: so only the GPT column is exactly reproducible; the Gemini archive column was
#: generated against whatever that alias resolved to at the time.
LM_MODELS = {
    "gpt5nano": "openai/gpt-5-nano-2025-08-07",
    "gemini": "gemini/gemini-flash-lite-latest",
}

NEUTRAL_PERCENTILE = 0.5


class PredictEarningsReturn(dspy.Signature):
    """Predict the unexpected stock return following an earnings call.

    You are given key facts from a company's earnings call transcript.
    Predict the stock's unexpected return as a class and percentile.

    Base rates — calibrate your predictions to these proportions:
      - ~25% of stocks go UP (price increases 5%+ after the call)
      - ~50% of stocks are NEUTRAL (price moves less than 5%)
      - ~25% of stocks go DOWN (price decreases 5%+ after the call)

    Consistency constraints between class and percentile:
      - "down"    → percentile in [0.00, 0.25]
      - "neutral" → percentile in [0.25, 0.75]
      - "up"      → percentile in [0.75, 1.00]

    Your rationale must reference substantive evidence directly
    (e.g., "Revenue grew 18% year-over-year…"). Never reference fact
    numbers (e.g., never say "fact 3 shows…" or "according to fact 7").
    """

    key_facts_discussed_in_earnings_call: str = dspy.InputField(
        desc="Bullet-point summary of key facts from the earnings call"
    )

    predict_class: Literal["up", "neutral", "down"] = dspy.OutputField(
        desc='Exactly one of: "up" (5%+ increase), "neutral" (<5% move), "down" (5%+ decrease)'
    )
    predict_percentile: float = dspy.OutputField(
        desc="Percentile rank of unexpected return: 0.0 (worst) to 1.0 (best)",
    )
    rationale: str = dspy.OutputField(
        desc="2-3 sentence explanation justifying the prediction using substantive evidence"
    )


def format_facts(facts: list[str]) -> str:
    """``"- <fact>"`` lines — the source's formatting, which ours already matched."""
    return "\n".join(f"- {fact}" for fact in facts)


def normalize_percentile(val: float) -> float:
    """Verbatim: models occasionally answer on a 0–100 scale."""
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def program() -> dspy.Module:
    return dspy.ChainOfThought(PredictEarningsReturn)


def run(events: list[dict], model: str = "gpt5nano", threads: int = 8) -> dict[str, float]:
    """Predictions keyed by ``event_id``, neutral on any failure."""
    lm = dspy.LM(LM_MODELS[model], timeout=120, cache=False)
    predictor = program()

    def one(event: dict) -> tuple[str, float]:
        if not event["facts"]:
            return event["event_id"], NEUTRAL_PERCENTILE
        try:
            with dspy.context(lm=lm):
                out = predictor(
                    key_facts_discussed_in_earnings_call=format_facts(event["facts"])
                )
            return event["event_id"], normalize_percentile(float(out.predict_percentile))
        except Exception as exc:
            print(f"  [fail] {event['event_id']}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            return event["event_id"], NEUTRAL_PERCENTILE

    started = time.time()
    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, (event_id, value) in enumerate(pool.map(one, events), 1):
            out[event_id] = value
            if i % 200 == 0:
                rate = i / (time.time() - started)
                print(f"  {i}/{len(events)}  {rate:.1f}/s  eta "
                      f"{(len(events)-i)/rate/60:.1f}m", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    import json

    import reads

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt5nano", choices=list(LM_MODELS))
    p.add_argument("--quarter", default="2026Q2")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    screen = {e["event_id"] for e in reads.screen_events(700)}
    events = [e for e in harness.events_for(args.quarter) if e["event_id"] in screen]
    print(f"replaying the official program ({LM_MODELS[args.model]}) on "
          f"{len(events)} {args.quarter} screen events")

    preds = run(events, args.model, args.threads)
    cache = Path(__file__).parent / "data" / "reads" / f"official__{args.model}__{args.quarter}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        for event_id, value in preds.items():
            fh.write(json.dumps({"event_id": event_id, "prediction": value}) + "\n")

    frame = harness.load(args.quarter)
    frame = frame[frame.event_id.isin(preds)].copy()
    frame["_p"] = frame.event_id.map(preds)
    scored = harness.evaluate(frame, "_p")
    pct = E.as_pct_obtainable(scored["delta_r_squared"], scored["r_squared_surprise"])
    archive = harness.evaluate(frame, harness.GPT if args.model == "gpt5nano" else harness.GEMINI)
    print(
        f"\nour replay of the official program: {scored['delta_r_squared']:+.4f} = {pct:.2%}\n"
        f"the archive's own column:            {archive['delta_r_squared']:+.4f} = "
        f"{E.as_pct_obtainable(archive['delta_r_squared'], archive['r_squared_surprise']):.2%}\n"
        f"our champion on the same events:     "
        f"{harness.evaluate(frame, harness.CHAMPION_COLUMN)['delta_r_squared']:+.4f}\n"
        f"distinct values: {len(set(preds.values()))} over {len(preds)} events"
    )
