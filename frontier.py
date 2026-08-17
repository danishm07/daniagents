"""Frontier sweep: does model tier explain the gap, and do lineages disagree?

The hypothesis is model tier — the leaders' cards say "a single **frontier** LLM
call" and everything we have tested is small-tier (flash-lite, flash, nano, mini,
llama, deepseek). Prior is low, and the literature is why: context-aided
forecasting work puts the capability threshold around 14B parameters, which
flash-class already clears, and documents an "Execution Gap" where models explain
correctly and still fail to forecast better — persisting at frontier scale.
QuantSightBench finds no frontier model meets its calibration target on numerical
forecasting. A multiple-x gap from tier alone would contradict all of that.

So the second output matters as much as the first. **Span lineages, not sizes.**
Our five small models sat at ρ_b ≈ 0.81 with each other — one opinion sampled
five times. If frontier models trained on different corpora genuinely disagree,
the family ceiling reopens *regardless* of whether any single one scores better,
because ρ/√ρ_b improves when ρ_b falls. Chinese vs Western pretraining is a
different axis from parameter count and it is the one that could move ρ_b.

Design constraints, each with a reason:

* **flash-lite runs in the same sweep** as a paired reference, so "frontier is
  better" cannot be confused with "this sample was easier".
* **n=300** — hunting a multiple-x effect, not making a selection decision. The
  n>=2000 rule applies to ships, not to direction.
* **Output capped hard.** We need one float. Reasoning models burned 2,472 tokens
  against Gemini's 874, and output is priced 4-10x input.
* **Spend guard between models**, computed from actual usage, so the cap binds
  even if a model is more expensive per call than its list price implied.
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

AGENT = Path(__file__).parent.parent / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "src")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT / ".env")

import predict  # noqa: E402

import arms  # noqa: E402
import champion  # noqa: E402
import eval as E  # noqa: E402
import harness  # noqa: E402
import reads  # noqa: E402

OUT = Path(__file__).parent / "data" / "frontier"

#: id -> (input $/M, output $/M), read from OpenRouter on 2026-08-16.
PRICES = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "deepseek/deepseek-chat": (0.2574, 1.0287),
    "moonshotai/kimi-k2": (0.57, 2.30),
    "qwen/qwen3-max": (0.78, 3.90),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "openai/gpt-5": (1.25, 10.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "anthropic/claude-opus-4.5": (5.00, 25.00),
}

#: Western and Chinese pretraining, three vendor families at frontier tier, plus
#: the small-tier reference. Opus is excluded on cost, Sonnet stands in for
#: Anthropic.
DEFAULT_MODELS = [
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-chat",
    "moonshotai/kimi-k2",
    "qwen/qwen3-max",
    "google/gemini-2.5-pro",
    "openai/gpt-5",
    "anthropic/claude-sonnet-4.5",
]


#: Models that reason by default. They need effort turned down or they never
#: reach the answer inside a sane output cap.
REASONING = {"openai/gpt-5": True, "google/gemini-2.5-pro": True}


def spend(model: str) -> float:
    path = OUT / f"{model.replace('/', '_')}.jsonl"
    if not path.exists():
        return 0.0
    pin, pout = PRICES.get(model, (1.0, 5.0))
    tin = tout = 0
    for line in path.open():
        r = json.loads(line)
        tin += r.get("prompt_tokens") or 0
        tout += r.get("completion_tokens") or 0
    return tin * pin / 1e6 + tout * pout / 1e6


def total_spend(models) -> float:
    return sum(spend(m) for m in models)


def run_model(model: str, events: list[dict], workers: int, max_tokens: int) -> None:
    path = OUT / f"{model.replace('/', '_')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["event_id"] for l in path.open()} if path.exists() else set()
    todo = [e for e in events if e["event_id"] not in done]
    if not todo:
        print(f"[{model}] cached", flush=True)
        return

    client = reads._client()
    lock = threading.Lock()
    started, failures, completed = time.time(), 0, 0

    def one(event):
        payload = champion.live_payload(event["facts"])
        champion._throttle.wait(champion._throttle.estimate(1100))
        resp = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": predict.SYSTEM_PROMPT},
                {"role": "user", "content": reads.user_prompt(payload, event["ticker"], champion.EVENT_TYPE)},
            ],
            response_format=reads.Direct,
            max_completion_tokens=max_tokens,
            # Reasoning models spend the whole cap thinking and never emit the
            # answer — gemini-2.5-pro and gpt-5 both failed 12/12 at 256 tokens.
            # Minimal effort keeps them in the sweep without paying for chains of
            # thought at $10/M output, which would blow the cap on one model.
            **({"reasoning_effort": "minimal"} if REASONING.get(model) else {}),
        )
        if resp.usage:
            champion._throttle.record(resp.usage.total_tokens)
        parsed = resp.choices[0].message.parsed
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
                if failures <= 2:
                    print(f"  [fail] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            completed += 1
    print(f"[{model}] {completed} done, {failures} failures, "
          f"{(time.time()-started)/60:.1f} min, ${spend(model):.2f}", flush=True)


def score(models, events) -> tuple[pd.DataFrame, pd.DataFrame]:
    wanted = {e["event_id"] for e in events}
    rows, resid = [], {}
    for model in models:
        path = OUT / f"{model.replace('/', '_')}.jsonl"
        if not path.exists():
            continue
        col = {json.loads(l)["event_id"]: json.loads(l)["prediction"] for l in path.open()}
        per, parts = [], []
        for q in harness.DEV_QUARTERS:
            f = harness.load(q)
            f = f[f.event_id.isin(wanted) & f.event_id.isin(col)].copy()
            if len(f) < 5:
                continue
            f["_p"] = f.event_id.map(col)
            s = harness.evaluate(f, "_p")
            surprise = f.surprise_pct.to_numpy(float)
            v = f["_p"].to_numpy(float)
            m = E._correlation_matrix({"a": v, "c": f[harness.CHAMPION_COLUMN].to_numpy(float)}, surprise)
            per.append({
                "n": s["n_obs"],
                "pct": E.as_pct_obtainable(s["delta_r_squared"], s["r_squared_surprise"]),
                "rho": E.partial_corr(v, f.y.to_numpy(float), surprise),
                "rho_b": float(m.loc["a", "c"]),
                "neutral": float((v == 0.5).mean()),
            })
            parts.append(pd.Series(E._residualize(v, surprise), index=f.event_id))
        if not per:
            continue
        t = pd.DataFrame(per)
        rows.append({
            "model": model.split("/")[-1],
            "n": int(t.n.sum()),
            "pct_obtainable": t.pct.mean(),
            "rho": t.rho.mean(),
            "rho_b_champion": t.rho_b.mean(),
            "neutral": t.neutral.mean(),
            "spend_$": round(spend(model), 2),
        })
        resid[model.split("/")[-1]] = pd.concat(parts)
    if not rows:
        return pd.DataFrame(), None
    table = pd.DataFrame(rows).sort_values("pct_obtainable", ascending=False)
    matrix = pd.DataFrame(resid).corr() if len(resid) > 1 else None
    return table, matrix


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--cap", type=float, default=5.0)
    p.add_argument("--rpm", type=int, default=300)
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()

    champion._throttle = champion._Throttle(args.rpm, 2_000_000)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_events = arms.tag_quarters(reads.screen_events(700))
    step = max(1, len(all_events) // args.n)
    events = all_events[::step][: args.n]
    print(f"{len(models)} models x {len(events)} events, cap ${args.cap:.2f}\n")

    if not args.score_only:
        for model in models:
            already = total_spend(models)
            est = PRICES.get(model, (1.0, 5.0))
            projected = already + (est[0] * 850 + est[1] * 150) / 1e6 * len(events)
            if projected > args.cap:
                print(f"[cap] skipping {model}: ${already:.2f} spent, "
                      f"projected ${projected:.2f} > ${args.cap:.2f}")
                continue
            run_model(model, events, args.workers, args.max_tokens)

    table, matrix = score(models, events)
    if table.empty:
        print("no scorable output")
        raise SystemExit(1)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    if matrix is not None:
        print("\ninter-model correlation, surprise projected out "
              "(0.81 was the small-tier baseline — lower reopens the family ceiling):")
        print(matrix.to_string(float_format=lambda v: f"{v:+.3f}"))
        off = matrix.where(~np.eye(len(matrix), dtype=bool))
        print(f"\nmean off-diagonal rho_b: {off.stack().mean():+.3f}")
    print(f"\nTOTAL SPEND: ${total_spend(models):.2f} of ${args.cap:.2f} cap")
