"""Grow the ACE rulebook: generate -> reflect -> curate, with a validation gate.

    uv run python ace_run.py --dry-run          # what it would cost, spend nothing
    uv run python ace_run.py --train 1000       # build the rulebook
    uv run python ace_run.py --score            # score the current rulebook

**Cost is the binding constraint, so read this before running.** $11.81 of $25 is
already spent. Generation dominates: one Generator call per training event per
epoch, plus one per validation event per gate. Reflector and Curator are a few
dozen calls total and are cheap even on a strong model. :func:`estimate` prints
the arithmetic before anything is spent and :class:`Session` refuses to start a
run whose estimate exceeds the remaining budget.

The division of labour follows Koijen & Levy rather than ACE. ACE deliberately
uses **one model for all three roles** to avoid distillation confounds — a
scientific control, not a recommendation. We have the opposite objective: we
*want* the strong model's structure distilled into a rulebook a cheap model can
execute, because that is the deployment story. So the Reflector and Curator run
on a strong model and the Generator runs on flash-lite, which is also what makes
the run affordable.

Note what the transfer numbers actually say, since it sets expectations: every
Table 8 transfer result is *below* un-ACE'd Opus (Haiku+ACE 11.7%, Sonnet+ACE
12.1%, Opus-low+ACE 13.9%, vs Opus-high-no-ACE 14.1%). Transfer buys ~85% of the
quality at a fraction of the cost. It does not make a small model competitive
with a large one.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ace as A  # noqa: E402
import eval as E  # noqa: E402
import harness  # noqa: E402
import reads  # noqa: E402
from runner import schedule as S  # noqa: E402

OUT = ROOT / "data" / "ace"
PLAYBOOK = OUT / "playbook.json"
TRACE = OUT / "trace.jsonl"

#: Cheap, and joint-best of the six models swept. The Generator runs here both
#: during training and live — that is the whole point of distillation through
#: context.
GENERATOR_MODEL = "google/gemini-2.5-flash-lite"

#: The Reflector and Curator run once per batch, not once per event, so a strong
#: model costs little in total and is where the structure comes from.
REFLECTOR_MODEL = "anthropic/claude-sonnet-4.5"

#: Events per Reflector call. **The single most important adaptation.** At
#: ρ≈0.25 one event's error is ~94% noise, so a rule inferred from one event is
#: a rule inferred from noise; at 50 a batch-level rank comparison has a usable
#: standard error. ACE reflects per-instance; we do not.
BATCH = 50

#: Validate every N curator calls. Each gate costs one Generator pass over the
#: validation slice, so this trades cost against how finely a bad delta block
#: can be caught.
GATE_EVERY = 4
VALIDATION_N = 400


@dataclass
class Spend:
    """Metered tokens, priced per model, accumulated across roles."""

    tokens: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, model: str, prompt: int, completion: int) -> None:
        with self.lock:
            a, b = self.tokens.get(model, (0, 0))
            self.tokens[model] = (a + prompt, b + completion)

    def usd(self) -> float:
        import frontier

        total = 0.0
        for model, (pin, pout) in self.tokens.items():
            price_in, price_out = frontier.PRICES.get(model, (1.0, 5.0))
            total += pin * price_in / 1e6 + pout * price_out / 1e6
        return total


def estimate(n_train: int, epochs: int, batch: int = BATCH,
             gate_every: int = GATE_EVERY, validation_n: int = VALIDATION_N) -> dict:
    """Cost arithmetic, printed before anything is spent.

    Token counts are measured from the existing arm columns (~1,050 prompt
    tokens for a bare read) plus the playbook, which grows. Assumed 2,500
    prompt tokens per Generator call at steady state — stated as an assumption
    because the playbook's final size is not knowable in advance.
    """
    import frontier

    n_batches = (n_train // batch) * epochs
    n_gates = max(n_batches // gate_every, 1)

    gen_calls = n_train * epochs + n_gates * validation_n
    gen_in, gen_out = gen_calls * 2500, gen_calls * 120
    ref_in, ref_out = n_batches * 12000, n_batches * 900
    cur_in, cur_out = n_batches * 6000, n_batches * 600

    def cost(model, pin, pout):
        a, b = frontier.PRICES.get(model, (1.0, 5.0))
        return pin * a / 1e6 + pout * b / 1e6

    generator = cost(GENERATOR_MODEL, gen_in, gen_out)
    reflector = cost(REFLECTOR_MODEL, ref_in + cur_in, ref_out + cur_out)
    return {
        "batches": n_batches,
        "gates": n_gates,
        "generator_calls": gen_calls,
        "generator_usd": generator,
        "reflector_curator_calls": n_batches * 2,
        "reflector_curator_usd": reflector,
        "total_usd": generator + reflector,
    }


def _client():
    return reads._client()


def _ask(model: str, system: str, user: str, spend: Spend, max_tokens: int = 1500) -> dict:
    """One JSON-returning call. Returns ``{}`` rather than raising on a parse failure.

    A malformed Curator response must not abort a run that has already spent
    money; an empty operations list is a valid outcome anyway.
    """
    client = _client()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    try:
        resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except Exception:
        # Not every provider honours json_object through OpenRouter — Anthropic
        # models in particular. Retrying without it and parsing the object out
        # of the text is strictly better than losing the call, since the prompts
        # already specify the schema.
        resp = client.chat.completions.create(**kwargs)
    if resp.usage:
        spend.record(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return _parse_json(resp.choices[0].message.content or "")


def _parse_json(text: str) -> dict:
    """Best-effort JSON object out of a model response.

    Returns ``{}`` rather than raising: a malformed Curator reply must not abort
    a run that has already spent money, and "no operations" is a valid outcome
    anyway.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text[4:] if text.lower().startswith("json") else text
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


class Session:
    """One rulebook-growing run, resumable and cost-capped."""

    def __init__(self, playbook: A.Playbook | None = None, workers: int = 12) -> None:
        self.playbook = playbook or A.Playbook.load(PLAYBOOK)
        self.spend = Spend()
        self.workers = workers
        self.history: list[dict] = []

    # -- generation ---------------------------------------------------

    def generate(self, events: list[dict]) -> pd.DataFrame:
        """Run the Generator over ``events`` with the current playbook."""
        rendered = self.playbook.render()
        rows, lock = [], threading.Lock()

        def one(event):
            user = A.generator_prompt(self.playbook, event["facts"], event["ticker"])
            out = _ask(GENERATOR_MODEL, A.GENERATOR_SYSTEM, user, self.spend, max_tokens=700)
            value = out.get("predicted_percentile")
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
            # Out of range is a missed event live, so treat it as missing here
            # rather than clipping — clipping would hide a broken generator.
            if not (0.0 <= value <= 1.0):
                value = float("nan")
            return {
                "event_id": event["event_id"],
                "ticker": event["ticker"],
                "facts": event["facts"],
                "predicted": value,
                "cited": [str(i) for i in (out.get("bullet_ids") or [])],
            }

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(one, e) for e in events]
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    continue
                with lock:
                    rows.append(row)
        return pd.DataFrame(rows)

    # -- reflection and curation --------------------------------------

    def reflect(self, batch: pd.DataFrame) -> dict:
        user = A.reflector_prompt(self.playbook, batch)
        return _ask(REFLECTOR_MODEL, A.REFLECTOR_SYSTEM, user, self.spend, max_tokens=2000)

    def curate(self, reflection: dict) -> list[dict]:
        user = A.curator_prompt(self.playbook, json.dumps(reflection, indent=2)[:12000])
        out = _ask(REFLECTOR_MODEL, A.CURATOR_SYSTEM, user, self.spend, max_tokens=2000)
        return [op for op in (out.get("operations") or []) if op.get("content")]

    def apply(self, operations: list[dict]) -> list[A.Rule]:
        """Deterministic merge. No LLM ever re-emits the playbook."""
        return [
            self.playbook.add(op.get("section", "interactions"), op["content"])
            for op in operations
        ]

    # -- the bookkeeping ACE leaves undone -----------------------------

    def attribute(self, batch: pd.DataFrame, tags: list[dict]) -> None:
        """Update per-rule statistics from a batch.

        Two channels. The reference's ``helpful``/``harmful`` counters come from
        the Reflector's tags and are kept because the Generator sees them. But
        at our SNR a per-batch tag is still weak evidence, so the number that
        actually gates pruning is ``contribution`` — the mean signed error over
        the events that *cited* the rule, accumulated across batches. A rule
        cited on events the model over-ranked is a rule pushing the wrong way.
        """
        index = self.playbook.by_id()
        for tag in tags or []:
            rule = index.get(str(tag.get("id")))
            if rule is None:
                continue
            if tag.get("tag") == "helpful":
                rule.helpful += 1
            elif tag.get("tag") == "harmful":
                rule.harmful += 1

        for row in batch.itertuples():
            if not np.isfinite(row.err):
                continue
            for rule_id in row.cited:
                rule = index.get(rule_id)
                if rule is None:
                    continue
                # err > 0 means the model ranked it too high, so a rule cited
                # there contributed in the wrong direction.
                rule.contribution = (
                    rule.contribution * rule.citations - row.err
                ) / (rule.citations + 1)
                rule.citations += 1

    # -- the gate ACE does not have ------------------------------------

    def validate(self, events: list[dict], frame: pd.DataFrame) -> dict:
        """Score the current playbook on a held-out slice."""
        preds = self.generate(events)
        merged = frame.merge(preds[["event_id", "predicted"]], on="event_id", how="inner")
        if len(merged) < 100:
            return {"n": len(merged), "delta_r2": float("nan"), "rho": float("nan")}
        y = merged.y.to_numpy(dtype=float)
        surprise = merged.surprise_pct.to_numpy(dtype=float)
        p = merged.predicted.to_numpy(dtype=float)
        return {
            "n": int(np.isfinite(p).sum()),
            "delta_r2": E._delta_r2_fast(p, surprise, y),
            "rho": E.partial_corr(p, y, surprise),
            "band": A.bootstrap_band(p, y, surprise, n_boot=400),
            "coverage": float(np.isfinite(p).mean()),
        }

    def save(self) -> None:
        self.playbook.save(PLAYBOOK)
        TRACE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE.open("a") as fh:
            for record in self.history:
                fh.write(json.dumps(record, default=str) + "\n")
        self.history = []


def train(n_train: int = 1000, epochs: int = 1, workers: int = 12, cap: float | None = None) -> Session:
    plan = estimate(n_train, epochs)
    remaining = S.budget_remaining() if cap is None else cap
    print(json.dumps(plan, indent=2))
    print(f"budget remaining ${remaining:.2f}")
    if plan["total_usd"] > remaining:
        raise SystemExit(
            f"estimated ${plan['total_usd']:.2f} exceeds ${remaining:.2f} remaining — "
            "reduce --train or raise the cap deliberately"
        )

    quarters = harness.DEV_QUARTERS
    frame = pd.concat([harness.load(q) for q in quarters], ignore_index=True)
    ranks = A.residual_ranks(frame)
    frame = frame.assign(residual_rank=ranks)

    confirm_ids = {e["event_id"] for v in S.confirmation_events().values() for e in v}
    pool = frame[~frame.event_id.isin(confirm_ids)]
    train_frame = pool.iloc[:n_train]
    val_frame = pool.iloc[n_train:n_train + VALIDATION_N]

    by_id = {e["event_id"]: e for q in quarters for e in harness.events_for(q)}
    train_events = [by_id[i] for i in train_frame.event_id if i in by_id]
    val_events = [by_id[i] for i in val_frame.event_id if i in by_id]

    session = Session(workers=workers)
    print(f"\ntrain {len(train_events)}  validate {len(val_events)}  "
          f"batch {BATCH}  gate every {GATE_EVERY} curator calls\n")

    best = session.validate(val_events, val_frame)
    best_playbook = json.loads(json.dumps({"next_id": session.playbook.next_id,
                                           "rules": [r.__dict__ for r in session.playbook.rules]}))
    print(f"baseline (empty playbook): dR2 {best['delta_r2']:+.4f}  rho {best['rho']:+.4f}  "
          f"n={best['n']}  coverage {best['coverage']:.3f}")

    step = 0
    for epoch in range(epochs):
        for start in range(0, len(train_events), BATCH):
            chunk = train_events[start:start + BATCH]
            if len(chunk) < 10:
                continue
            step += 1
            preds = session.generate(chunk)
            sub = train_frame[train_frame.event_id.isin(set(preds.event_id))]
            batch = preds.merge(
                sub[["event_id", "residual_rank"]], on="event_id", how="inner"
            )
            batch["err"] = batch.predicted - batch.residual_rank
            batch = batch.dropna(subset=["err"])
            if len(batch) < 10:
                continue

            reflection = session.reflect(batch)
            operations = session.curate(reflection)
            added = session.apply(operations)
            session.attribute(batch, reflection.get("bullet_tags"))

            print(f"[epoch {epoch} step {step}] batch {len(batch)}  "
                  f"+{len(added)} rules  playbook {len(session.playbook.rules)}  "
                  f"spent ${session.spend.usd():.2f}")

            if step % GATE_EVERY == 0:
                scored = session.validate(val_events, val_frame)
                improved = scored["delta_r2"] > best["delta_r2"] + best.get("band", 0.0)
                print(f"  gate: dR2 {scored['delta_r2']:+.4f} vs best {best['delta_r2']:+.4f} "
                      f"(band {best.get('band', 0):.4f}) -> "
                      f"{'ACCEPT' if improved else 'REVERT'}")
                if improved:
                    best = scored
                    best_playbook = json.loads(json.dumps(
                        {"next_id": session.playbook.next_id,
                         "rules": [r.__dict__ for r in session.playbook.rules]}))
                else:
                    session.playbook = A.Playbook(
                        rules=[A.Rule(**r) for r in best_playbook["rules"]],
                        next_id=best_playbook["next_id"],
                    )
                session.history.append({"step": step, "kind": "gate", **scored,
                                        "accepted": improved})
                session.save()

    session.playbook = A.Playbook(
        rules=[A.Rule(**r) for r in best_playbook["rules"]], next_id=best_playbook["next_id"]
    )
    session.save()
    print(f"\nfinal playbook: {len(session.playbook.rules)} rules, "
          f"best validation dR2 {best['delta_r2']:+.4f}, spent ${session.spend.usd():.2f}")
    return session


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    if args.show:
        pb = A.Playbook.load(PLAYBOOK)
        print(pb.render())
        print(json.dumps(pb.stats(), indent=2))
    elif args.dry_run or not args.train:
        for n in (500, 1000, 2000):
            plan = estimate(n, args.epochs)
            print(f"n_train={n:<5} batches={plan['batches']:<4} gates={plan['gates']:<3} "
                  f"gen=${plan['generator_usd']:.2f} ref/cur=${plan['reflector_curator_usd']:.2f} "
                  f"TOTAL=${plan['total_usd']:.2f}")
        print(f"\nbudget remaining ${S.budget_remaining():.2f}")
    else:
        train(args.train, args.epochs, args.workers)
