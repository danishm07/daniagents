"""Validation gate: does the rulebook beat no rulebook on events it never saw?

The step ACE does not have. The reference commits every delta unconditionally and
checks only a 100-sample checkpoint; that is survivable when feedback is a
correct/incorrect bit and is not survivable at rho ~ 0.25. D5's ablation is the
warning: its unvalidated proposer scored 4-12%, and the validator was not a
refinement but the thing that made the system work.

**The rulebook is injected as a PREFIX to the deployed prompt**, via
``arms.build_prompt``, which routes through ``reads.user_prompt`` and
``predict._facts_text`` — the same objects the live worker calls. So this is a
context arm in exactly the sense ``ctx.*`` arms already are: directly comparable
to the cached ``base@flash-lite`` column, and deployable by pointing production
at the same context.

The baseline costs nothing: an empty playbook is the base prompt, and those
predictions are already on disk.

    uv run python ace_gate.py --playbook data/ace/rules_v1_core.json --tag v1
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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ace as A  # noqa: E402
import arms as ARMS  # noqa: E402
import champion  # noqa: E402
import eval as E  # noqa: E402
import harness  # noqa: E402
import reads  # noqa: E402
from runner import ledger  # noqa: E402

OUT = ROOT / "data" / "ace"
MODEL = "google/gemini-2.5-flash-lite"
BASE_COLUMN = ROOT / "data" / "arms" / "base__google_gemini-2.5-flash-lite.jsonl"


def _column(tag: str) -> Path:
    return OUT / f"gate__{tag}.jsonl"


def playbook_text(rules_path: Path) -> str:
    """Render a rule file as the context block, without touching the live playbook."""
    playbook = A.Playbook()
    for proposal in json.loads(Path(rules_path).read_text()):
        playbook.add(proposal.get("section", "interactions"), proposal["content"])
    body = playbook.render()
    return (
        "You are given a rulebook of empirical regularities learned from thousands of past "
        "earnings announcements. Apply it.\n\n"
        "Rules are written as: condition(s) -> percentile band. When a rule's condition holds, "
        "its band is strong evidence for where this event should land. Where rules conflict, "
        "prefer the more specific condition.\n\n"
        f"{body}\n"
    )


def generate(events: list[dict], context: str, tag: str, workers: int = 12) -> dict[str, float]:
    """Run flash-lite with the rulebook prefixed to the deployed prompt."""
    path = _column(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if path.exists():
        done = {json.loads(l)["event_id"]: json.loads(l)["prediction"]
                for l in path.open() if l.strip()}
    todo = [e for e in events if e["event_id"] not in done]
    print(f"[{tag}] cached {len(done)}, to run {len(todo)}")
    if not todo:
        return done

    client = reads._client()
    lock = threading.Lock()
    started, cost, fails = time.time(), 0.0, 0

    def one(event):
        prompt = ARMS.build_prompt(event, context)
        if prompt is None:  # no facts: production submits 0.5 without calling
            return {"event_id": event["event_id"], "prediction": 0.5,
                    "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
        champion._throttle.wait(champion._throttle.estimate(2500))
        resp = client.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": __import__("predict").SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=reads.Direct,
        )
        usage = resp.usage
        if usage:
            champion._throttle.record(usage.total_tokens)
        parsed = resp.choices[0].message.parsed
        return {
            "event_id": event["event_id"],
            "prediction": float("nan") if parsed is None else parsed.predicted_percentile,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "cost": float(getattr(usage, "cost", 0.0) or 0.0) if usage else 0.0,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futures = {pool.submit(one, e): e for e in todo}
        tin = tout = 0
        for i, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:
                fails += 1
                if fails <= 3:
                    print(f"  [fail] {type(exc).__name__}: {str(exc)[:100]}")
                continue
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            done[row["event_id"]] = row["prediction"]
            tin += row["prompt_tokens"]
            tout += row["completion_tokens"]
            cost += row["cost"]
            if i % 100 == 0:
                rate = i / (time.time() - started)
                print(f"  {i}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-i)/rate/60:.1f}m",
                      flush=True)

    usd = ledger.record(MODEL, tin, tout, source=f"ace.gate.{tag}",
                        note=f"validation gate, {len(todo)} events", usd=cost or None)
    print(f"[{tag}] done, {fails} failures, ${usd:.4f}")
    return done


def score(tag: str, event_ids: list[str], baseline_tag: str | None = None) -> dict:
    """Paired against the no-rulebook baseline on identical events.

    ``baseline_tag`` names a gate column to use as the baseline instead of the
    cached ``base@flash-lite`` file. Needed on the confirmation partition, which
    the cached column does not reach — the arms.py screening sample stops at the
    rung boundary, so a baseline there has to be generated rather than reused.
    """
    if baseline_tag:
        base = {json.loads(l)["event_id"]: json.loads(l)["prediction"]
                for l in _column(baseline_tag).open() if l.strip()}
    else:
        base = {json.loads(l)["event_id"]: json.loads(l)["prediction"]
                for l in BASE_COLUMN.open() if l.strip()}
    rule = {json.loads(l)["event_id"]: json.loads(l)["prediction"]
            for l in _column(tag).open() if l.strip()}

    wanted = set(event_ids) & set(base) & set(rule)
    frames = []
    for quarter in harness.DEV_QUARTERS:
        f = harness.load(quarter)
        f = f[f.event_id.isin(wanted)].copy()
        if len(f):
            frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    frame["_base"] = frame.event_id.map(base)
    frame["_rule"] = frame.event_id.map(rule)
    frame = frame.dropna(subset=["_base", "_rule"])

    y = frame.y.to_numpy(dtype=float)
    surprise = frame.surprise_pct.to_numpy(dtype=float)
    a = frame["_rule"].to_numpy(dtype=float)
    b = frame["_base"].to_numpy(dtype=float)

    gain = E._delta_r2_fast(a, surprise, y) - E._delta_r2_fast(b, surprise, y)
    rng = np.random.default_rng(0)
    n = len(y)
    draws = [
        E._delta_r2_fast(a[i], surprise[i], y[i]) - E._delta_r2_fast(b[i], surprise[i], y[i])
        for i in (rng.integers(0, n, n) for _ in range(2000))
    ]
    se = float(np.std(draws, ddof=1))

    out = {
        "tag": tag,
        "n": int(n),
        "base_dr2": E._delta_r2_fast(b, surprise, y),
        "rule_dr2": E._delta_r2_fast(a, surprise, y),
        "gain": gain,
        "se": se,
        "z": gain / se if se else float("nan"),
        "base_rho": E.partial_corr(b, y, surprise),
        "rule_rho": E.partial_corr(a, y, surprise),
        "base_distinct": int(frame["_base"].nunique()),
        "rule_distinct": int(frame["_rule"].nunique()),
        "rule_neutral_rate": float((a == 0.5).mean()),
        # The pre-registered bar: beat the baseline by more than 2 x the paired
        # bootstrap se. Paired because both arms are the same model on the same
        # events, so the difference is far less noisy than either level.
        "passes_gate": bool(gain > 2 * se),
    }
    print(f"\n{'=' * 78}\nGATE — {tag}, paired on {n} held-out events\n{'=' * 78}")
    print(f"  no rulebook     dR2 {out['base_dr2']:+.4f}   rho {out['base_rho']:+.4f}   "
          f"distinct {out['base_distinct']}")
    print(f"  with rulebook   dR2 {out['rule_dr2']:+.4f}   rho {out['rule_rho']:+.4f}   "
          f"distinct {out['rule_distinct']}")
    print(f"  gain {gain:+.4f}   se {se:.4f}   z {out['z']:+.2f}   "
          f"neutral rate {out['rule_neutral_rate']:.3f}")
    print(f"  => {'PASS' if out['passes_gate'] else 'FAIL'} (bar: gain > 2 x se = {2*se:+.4f})")
    (OUT / f"gate_result_{tag}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--playbook", type=Path)
    p.add_argument("--tag", required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--baseline-tag", default=None)
    p.add_argument("--empty", action="store_true",
                   help="generate the no-rulebook baseline arm")
    p.add_argument("--rpm", type=int, default=400)
    p.add_argument("--partition", default="validation",
                   choices=["validation", "confirmation"],
                   help="confirmation = the 4,144 events selection never touched")
    args = p.parse_args()

    champion._throttle = champion._Throttle(args.rpm, 4_000_000)
    if args.partition == "confirmation":
        # K3's partition. Never used by training or by the n=400 gate, so a
        # number here is the arm's value rather than a search heuristic.
        from runner import schedule as S
        ids = [e["event_id"] for v in S.confirmation_events().values() for e in v]
    else:
        splits = json.loads((OUT / "splits.json").read_text())
        ids = splits["validation_event_ids"]
    by_id = {e["event_id"]: e for q in harness.DEV_QUARTERS for e in harness.events_for(q)}
    events = ARMS.tag_quarters([by_id[i] for i in ids if i in by_id])

    if not args.score_only:
        context = "" if args.empty else playbook_text(args.playbook)
        print(f"rulebook context: {len(context)} chars (~{len(context)//4} tokens)")
        ledger.guard(estimated_usd=0.60)
        generate(events, context, args.tag, args.workers)
    score(args.tag, ids, args.baseline_tag)
