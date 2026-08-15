"""Sweep the read itself: reasoning scaffold × model.

The champion scores 4.2% of obtainable against the archive's 2025 ``gpt-5-nano``
baseline at 4.6% and Gemini Flash-Lite at 5.8%. A newer model running our own
prompt is losing to a year-old one, and there are two candidate explanations
that the archive cannot separate:

* **scaffold** — the baselines are a ``dspy.ChainOfThought``; our prompt asks
  for the structured answer directly, with no reasoning step;
* **model** — ``gpt-5.4-nano`` may simply be worse at this than ``gpt-5-nano``.

So the sweep is a 2×2: {direct, cot} × {gpt-5.4-nano, gpt-5-nano}, plus a
larger model to price the size lever. Same events, same prompt text, one thing
changing at a time.

**This is a measurement fix before it is an improvement.** The champion is the
reference for every ρ_b, every floor and every ship/reject decision on the
board. A degraded reference reads as *decorrelated from everything*, because
noise is uncorrelated with everything — so TF-IDF's ρ_b = 0.193 is provisional
until the reference stops underperforming a 2025 baseline. Any read improvement
means re-measuring ρ_b for every open channel.

Screening runs on a deterministic stratified sample; the winner is then
replayed over the full dev quarters at full statistical power.

    uv run python reads.py --screen          # the 2x2 plus a size probe
    uv run python reads.py --full <spec>     # promote one spec to all events
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

AGENT = Path(__file__).parent.parent / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "src")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT / ".env")

import predict  # noqa: E402

import champion  # noqa: E402
import eval as E  # noqa: E402
import harness  # noqa: E402

READS = Path(__file__).parent / "data" / "reads"

#: Events per quarter in the screening pass. Deterministic and stratified, so
#: every spec is scored on exactly the same events and the comparison stays
#: paired. Smaller than the full archive by design — screening buys direction,
#: not a promotion decision.
SCREEN_PER_QUARTER = 700


class Direct(BaseModel):
    """Production's schema, byte-identical: one number, no reasoning field."""

    predicted_percentile: float = Field(ge=0.0, le=1.0)


class WithReasoning(BaseModel):
    """Reasoning first, then the number.

    Field order is the mechanism, not decoration: structured outputs are
    generated in declaration order, so the model must produce its reasoning
    before it commits to a percentile. Declaring the number first would let it
    answer and then rationalise, which is the opposite of a chain of thought.
    """

    reasoning: str = Field(description="Brief analysis of the facts before answering.")
    predicted_percentile: float = Field(ge=0.0, le=1.0)


SCHEMAS = {"direct": Direct, "cot": WithReasoning}


def user_prompt(summary: dict, ticker: str, event_type: str) -> str:
    """The deployed user prompt, reproduced exactly.

    Asserted equal to what ``predict._ask_llm`` actually sends — see
    :func:`check_prompt_fidelity`. A sweep that silently changed the prompt
    while claiming to change only the model would answer the wrong question.
    """
    return (
        f"Event type: {event_type}\n"
        f"Ticker: {ticker}\n\n"
        f"Facts extracted from the earnings call:\n{predict._facts_text(summary)}\n\n"
        "Weigh, in roughly this order:\n"
        "  1. Quantitative surprise vs expectations — revenue, EPS, segment metrics.\n"
        "  2. Guidance / outlook — raises, holds, cuts vs the prior trajectory.\n"
        "  3. Strategic shifts — product launches, M&A, capital allocation, leadership.\n"
        "  4. Tone and confidence in management commentary (small weight).\n"
        "  5. Risks called out — regulatory, supply chain, demand, competition.\n\n"
        f"Predict the next-day unexpected-return percentile for {ticker}."
    )


def check_prompt_fidelity() -> None:
    """Capture what production would send and compare, without spending a call."""
    event = harness.events_for("2026Q2")[0]
    payload = champion.live_payload(event["facts"])
    captured: dict = {}

    class _Stop(Exception):
        pass

    class _Fake:
        class chat:
            class completions:
                @staticmethod
                def parse(**kwargs):
                    captured.update(kwargs)
                    raise _Stop

    saved = predict._openai
    predict._openai = _Fake
    try:
        predict._ask_llm(summary=payload, ticker=event["ticker"], event_type=champion.EVENT_TYPE)
    except _Stop:
        pass
    finally:
        predict._openai = saved

    theirs = [m for m in captured["messages"] if m["role"] == "user"][0]["content"]
    mine = user_prompt(payload, event["ticker"], champion.EVENT_TYPE)
    assert theirs == mine, "sweep prompt has drifted from the deployed prompt"
    system = [m for m in captured["messages"] if m["role"] == "system"][0]["content"]
    assert system == predict.SYSTEM_PROMPT
    print("prompt fidelity: sweep sends exactly what production sends")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def screen_events(per_quarter: int = SCREEN_PER_QUARTER) -> list[dict]:
    """A deterministic stratified sample — same events for every spec.

    Evenly spaced over each quarter's events sorted by ``event_id`` rather than
    randomly drawn: reproducible without carrying a seed, and it cannot
    accidentally concentrate in one part of the quarter.
    """
    out = []
    for quarter in harness.DEV_QUARTERS:
        events = sorted(harness.events_for(quarter), key=lambda e: e["event_id"])
        step = max(1, len(events) // per_quarter)
        out.extend(events[::step][:per_quarter])
    return out


def _path(model: str, variant: str) -> Path:
    return READS / f"{variant}__{model.replace('/', '_')}.jsonl"


def generate(model: str, variant: str, events: list[dict], workers: int = 10) -> Path:
    """One read column for one (model, variant), cached and resumable."""
    from openai import OpenAI

    path = _path(model, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {json.loads(l)["event_id"] for l in path.open() if l.strip()}
    todo = [e for e in events if e["event_id"] not in done]
    print(f"{variant}/{model}: cached {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return path

    client = OpenAI(timeout=predict.LLM_TIMEOUT_SECONDS, max_retries=champion.BACKFILL_MAX_RETRIES)
    schema = SCHEMAS[variant]
    lock = threading.Lock()
    started, failures, completed = time.time(), 0, 0

    def one(event: dict) -> dict:
        payload = champion.live_payload(event["facts"])
        if not predict._extract_facts(payload):
            return {"event_id": event["event_id"], "prediction": 0.5, "no_facts": True}
        # cot emits reasoning tokens, so its per-call cost is roughly double a
        # direct call's; the throttle needs the estimate up front and the truth
        # afterwards, or the token budget is breached before anyone notices.
        champion._throttle.wait(champion._throttle.estimate(1600 if variant == "cot" else 900))
        resp = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": predict.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(payload, event["ticker"], champion.EVENT_TYPE)},
            ],
            response_format=schema,
        )
        parsed = resp.choices[0].message.parsed
        if resp.usage:
            champion._throttle.record(resp.usage.total_tokens)
        return {
            "event_id": event["event_id"],
            "prediction": 0.5 if parsed is None else parsed.predicted_percentile,
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futures = {pool.submit(one, e): e for e in todo}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                failures += 1
                print(f"  [fail] {futures[future]['event_id']}: {type(exc).__name__}: {exc}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            completed += 1
            if completed % 250 == 0:
                rate = completed / (time.time() - started)
                print(f"  {completed}/{len(todo)} {rate:.1f}/s eta {(len(todo)-completed)/rate/60:.1f}m", flush=True)

    print(f"{variant}/{model}: {completed} done, {failures} failures, "
          f"{(time.time()-started)/60:.1f} min", flush=True)
    return path


def column(model: str, variant: str) -> dict[str, float]:
    path = _path(model, variant)
    if not path.exists():
        return {}
    return {json.loads(l)["event_id"]: json.loads(l)["prediction"] for l in path.open() if l.strip()}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(model: str, variant: str, events: list[dict]) -> dict:
    """Paired score against the champion, on exactly the sampled events.

    Reports ΔR² and its share of the 0.9413 obtainable ceiling, ρ, the paired
    difference against the champion, and ρ_b — the last of which is the reason
    the sweep matters beyond its own score.
    """
    preds = column(model, variant)
    wanted = {e["event_id"] for e in events}
    rows, rho_bs = [], []
    for quarter in harness.DEV_QUARTERS:
        frame = harness.load(quarter)
        frame = frame[frame.event_id.isin(wanted) & frame.event_id.isin(preds)].copy()
        if frame.empty:
            continue
        frame["_read"] = frame.event_id.map(preds)
        scored = harness.evaluate(frame, "_read")
        champ = harness.evaluate(frame, harness.CHAMPION_COLUMN)
        surprise = frame.surprise_pct.to_numpy(dtype=float)
        rows.append(
            {
                "quarter": quarter,
                "n": scored["n_obs"],
                "delta_r2": scored["delta_r_squared"],
                "pct_obtainable": E.as_pct_obtainable(
                    scored["delta_r_squared"], scored["r_squared_surprise"]
                ),
                "rho": E.partial_corr(
                    frame["_read"].to_numpy(dtype=float), frame.y.to_numpy(dtype=float), surprise
                ),
                "vs_champion": scored["delta_r_squared"] - champ["delta_r_squared"],
            }
        )
        matrix = E._correlation_matrix(
            {
                "read": frame["_read"].to_numpy(dtype=float),
                "champion": frame[harness.CHAMPION_COLUMN].to_numpy(dtype=float),
            },
            surprise,
        )
        rho_bs.append(float(matrix.loc["read", "champion"]))

    frame = pd.DataFrame(rows)
    return {
        "spec": f"{variant}/{model}",
        "n": int(frame.n.sum()),
        "delta_r2": float(frame.delta_r2.mean()),
        "pct_obtainable": float(frame.pct_obtainable.mean()),
        "rho": float(frame.rho.mean()),
        "vs_champion": float(frame.vs_champion.mean()),
        "signs": f"{int((frame.vs_champion > 0).sum())}/{len(frame)}",
        "rho_b_champion": float(np.tanh(np.arctanh(np.clip(rho_bs, -0.999, 0.999)).mean())),
        "per_quarter": frame.to_dict("records"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--full", help="model:variant to run over all dev quarters")
    parser.add_argument("--per-quarter", type=int, default=SCREEN_PER_QUARTER)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rpm", type=int, default=150)
    parser.add_argument("--tpm", type=int, default=champion.DEFAULT_TPM)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    champion._throttle = champion._Throttle(args.rpm, args.tpm)

    check_prompt_fidelity()

    #: One thing changes at a time. cot/gpt-5.4-nano isolates the scaffold,
    #: direct/gpt-5-nano isolates the model version against the same prompt,
    #: and the mini pair prices the size lever on both scaffolds.
    SPECS = [
        ("gpt-5.4-nano", "cot"),
        ("gpt-5-nano", "direct"),
        ("gpt-5-nano", "cot"),
        ("gpt-5.4-mini", "direct"),
    ]

    if args.full:
        model, variant = args.full.split(":")
        generate(model, variant, harness.events_for("2025Q4") + harness.events_for("2026Q1")
                 + harness.events_for("2026Q2"), workers=args.workers)
        raise SystemExit(0)

    events = screen_events(args.per_quarter)
    print(f"screening on {len(events)} events, identical for every spec\n")
    if not args.score_only:
        for model, variant in SPECS:
            generate(model, variant, events, workers=args.workers)

    results = [score(m, v, events) for m, v in SPECS if column(m, v)]
    champ = [
        harness.evaluate(harness.load(q), harness.CHAMPION_COLUMN) for q in harness.DEV_QUARTERS
    ]
    print("\nchampion (full archive, for reference): "
          f"{np.mean([s['delta_r_squared'] for s in champ]):+.4f} = "
          f"{np.mean([E.as_pct_obtainable(s['delta_r_squared'], s['r_squared_surprise']) for s in champ]):.1%}")
    if results:
        table = pd.DataFrame(results).drop(columns=["per_quarter"])
        print()
        print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
