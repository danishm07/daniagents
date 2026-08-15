"""Generate the champion column: the deployed prompt, the deployed model, over the archive.

Everything measured so far — every ρ_b, every promotion floor, the +0.0121
model-choice result — was measured against the archive's ``GPT-5 nano``
baseline standing in for our submission. It is not our submission. The archive
column is ``gpt-5-nano-2025-08-07`` running the official DSPy program; we deploy
``gpt-5.4-nano`` running our own prompt. A newer model with a different prompt
could already have closed the gap this project is trying to close.

So this script does not re-implement the champion. It imports ``predict`` from
``agent/`` and calls :func:`predict._ask_llm` — the same function the Modal
worker calls on a live webhook — with the facts wrapped in the *live* payload
shape, so ``_facts_text`` takes the same path in the backtest that it takes in
production. If the deployed prompt changes, regenerate; the model name and a
hash of the prompt are recorded in every row so a stale column is detectable
rather than silently wrong.

    uv run python champion.py                 # dev quarters, resumable
    uv run python champion.py --quarters 2026Q3   # the holdout, at the very end

Cached to ``data/champion/predictions.jsonl`` and merged into
:func:`harness.load` automatically. Interrupting is safe: every completed event
is on disk, and a rerun only fills the gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

AGENT = Path(__file__).parent.parent / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "src")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT / ".env")

import predict  # noqa: E402  — the deployed module, imported not copied
from explaining_markets.config import openai_model  # noqa: E402

import harness  # noqa: E402

OUT = Path(__file__).parent / "data" / "champion" / "predictions.jsonl"

#: Every archive file is an EARNINGS_RELEASE, and that is what the live webhook
#: carries in ``event_type``. Passed through so the prompt matches production.
EVENT_TYPE = "EARNINGS_RELEASE"

_write_lock = threading.Lock()
_usage: list[tuple[int, int]] = []

#: Org limits observed on 2026-08-15: 200,000 TPM and 500 RPM for
#: ``gpt-5.4-nano``. At ~840 tokens per event TPM binds first, at ~238 events a
#: minute; 200 leaves headroom for the completion side and for retries. Running
#: unthrottled failed 5,824 of 6,139 events in 1.7 minutes.
DEFAULT_RPM = 200

#: Transport-level retries for the offline replay. Production keeps
#: ``LLM_MAX_RETRIES = 1`` because it is racing a five-minute deadline; a
#: backfill has no deadline and every dropped event is a hole in the column.
#: This changes reliability, not the prompt or the model.
BACKFILL_MAX_RETRIES = 8


#: Tokens per minute the org allows. The request limit is not the binding one:
#: a direct call is ~870 tokens, so 200 rpm sits comfortably under 200k TPM —
#: but a chain-of-thought call emits reasoning, and at ~1,350 tokens the same
#: 200 rpm asks for 270k TPM and the 429s come straight back. Counting requests
#: alone is not throttling; this window counts tokens too.
DEFAULT_TPM = 180_000


class _Throttle:
    """Rolling-60s budget for **both** requests and tokens.

    ``wait(estimate)`` reserves an estimated token cost up front, because the
    real cost is only known after the response arrives — by which point the
    limit has already been breached. :meth:`record` then trues the reservation
    up against actual usage, so the estimate self-corrects within a minute
    instead of being a constant someone has to maintain.
    """

    def __init__(self, rpm: int, tpm: int = DEFAULT_TPM) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.lock = threading.Lock()
        self.window: list[list[float]] = []  # [timestamp, tokens]
        self.observed: list[int] = []

    def estimate(self, default: int) -> int:
        """Mean observed total tokens, once there is anything to average."""
        with self.lock:
            recent = self.observed[-200:]
        return int(sum(recent) / len(recent)) if recent else default

    def wait(self, tokens: int = 900) -> None:
        while True:
            with self.lock:
                now = time.time()
                self.window = [w for w in self.window if now - w[0] < 60.0]
                spent = sum(w[1] for w in self.window)
                if len(self.window) < self.rpm and spent + tokens <= self.tpm:
                    self.window.append([now, float(tokens)])
                    return
                oldest = self.window[0][0] if self.window else now
                sleep_for = 60.0 - (now - oldest) + 0.05
            time.sleep(min(max(sleep_for, 0.05), 5.0))

    def record(self, tokens: int) -> None:
        with self.lock:
            self.observed.append(tokens)
            if self.window:
                self.window[-1][1] = float(tokens)


_throttle = _Throttle(DEFAULT_RPM)


def prompt_fingerprint() -> str:
    """Hash of the system prompt + the deployed model name.

    Rows carrying a different fingerprint were generated by a different
    champion, which is the failure this column exists to prevent.
    """
    material = f"{openai_model()}|{predict.SYSTEM_PROMPT}"
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def live_payload(facts: list[str]) -> dict:
    """The facts in the confirmed live ``information_url`` shape.

    Confirmed from production 2026-08-14: the archive's disclosure object,
    unwrapped. Using it here means ``_facts_text`` resolves the facts by the
    same path offline as it does on a live event — including the bullet-list
    rendering, which is part of the prompt.
    """
    return {
        "schema_version": "1.0",
        "items": [{"kind": "facts", "source": "earnings_call", "content": facts}],
    }


def _instrument() -> None:
    """Record token usage without changing the call path.

    ``predict._ask_llm`` returns a bare float, so usage is captured by wrapping
    the client's ``parse`` rather than by reimplementing the request.
    """
    from openai import OpenAI

    client = OpenAI(
        timeout=predict.LLM_TIMEOUT_SECONDS, max_retries=BACKFILL_MAX_RETRIES
    )
    inner = client.chat.completions.parse

    def parse(**kwargs):
        response = inner(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            _usage.append((usage.prompt_tokens, usage.completion_tokens))
        return response

    client.chat.completions.parse = parse
    predict._openai = client


def already_done() -> dict[str, dict]:
    if not OUT.exists():
        return {}
    rows = {}
    for line in OUT.open():
        if line.strip():
            row = json.loads(line)
            rows[row["event_id"]] = row
    return rows


def one(event: dict, fingerprint: str, model: str) -> dict:
    started = time.time()
    payload = live_payload(event["facts"])
    # Mirror predict()'s no-facts contract exactly. predict() returns 0.5
    # without calling the model when the bundle carries no usable facts, and
    # this file calls _ask_llm directly — so without this branch the champion
    # column would disagree with the champion on precisely the rows where the
    # champion does something special. Five events in the archive, and the
    # point of this column is that it is a faithful replay.
    if not predict._extract_facts(payload):
        value = 0.5
    else:
        _throttle.wait()
        value = predict._ask_llm(
            summary=payload,
            ticker=event["ticker"],
            event_type=EVENT_TYPE,
        )
    return {
        "event_id": event["event_id"],
        "ticker": event["ticker"],
        "prediction": float(value),
        "model": model,
        "prompt_fingerprint": fingerprint,
        "runtime_s": round(time.time() - started, 3),
    }


def generate(quarters: list[str], workers: int = 10, limit: int | None = None) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — champion column would be all 0.5")

    model, fingerprint = openai_model(), prompt_fingerprint()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = already_done()
    stale = {e for e, r in done.items() if r.get("prompt_fingerprint") != fingerprint}
    if stale:
        print(f"!! {len(stale)} cached rows came from a different prompt/model — rerun after deleting {OUT}")

    todo = []
    for quarter in quarters:
        for event in harness.events_for(quarter):
            if event["event_id"] not in done:
                todo.append(event)
    if limit:
        todo = todo[:limit]

    print(f"model {model}  prompt {fingerprint}  cached {len(done)}  to run {len(todo)}")
    if not todo:
        return

    _instrument()
    started, failures, completed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, e, fingerprint, model): e for e in todo}
        with OUT.open("a") as fh:
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:  # one dead event must not kill the run
                    failures += 1
                    print(f"  [fail] {futures[future]['event_id']}: {type(exc).__name__}: {exc}")
                    continue
                with _write_lock:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                completed += 1
                if completed % 250 == 0:
                    rate = completed / (time.time() - started)
                    print(
                        f"  {completed}/{len(todo)}  {rate:.1f}/s  "
                        f"eta {(len(todo) - completed) / rate / 60:.1f}m  failures {failures}",
                        flush=True,
                    )

    prompt_tokens = sum(p for p, _ in _usage)
    completion_tokens = sum(c for _, c in _usage)
    print(
        f"\ndone: {completed} predictions, {failures} failures, "
        f"{(time.time() - started) / 60:.1f} min\n"
        f"tokens: {prompt_tokens:,} prompt + {completion_tokens:,} completion"
    )
    if failures:
        print("rerun to fill the gaps — completed events are cached and will be skipped")


def validate(quarters: list[str] | None = None) -> bool:
    """Is the column complete, self-consistent, and actually varying?

    Checks worth having because each corresponds to a way this file could be
    quietly wrong: gaps (a rate-limited run that was never resumed), duplicates
    (two runs appending), mixed fingerprints (a prompt change mid-run), and a
    degenerate spread (an all-0.5 column from a missing API key, which scores
    exactly zero while looking like a real submission).
    """
    quarters = quarters or harness.DEV_QUARTERS
    rows = already_done()
    if not rows:
        print("no champion column")
        return False

    import collections

    seen = collections.Counter()
    for line in OUT.open():
        if line.strip():
            seen[json.loads(line)["event_id"]] += 1
    duplicates = {e: n for e, n in seen.items() if n > 1}

    wanted = {e["event_id"] for q in quarters for e in harness.events_for(q)}
    missing = wanted - set(rows)
    values = [r["prediction"] for e, r in rows.items() if e in wanted]
    fingerprints = {r["prompt_fingerprint"] for r in rows.values()}
    out_of_range = [v for v in values if not 0.0 <= v <= 1.0]

    import statistics

    ok = not missing and not duplicates and len(fingerprints) == 1 and not out_of_range
    print(
        f"champion column: {len(values)}/{len(wanted)} events over {quarters}\n"
        f"  missing            {len(missing)}\n"
        f"  duplicate rows     {len(duplicates)}\n"
        f"  prompt fingerprint {sorted(fingerprints)}\n"
        f"  out of [0,1]       {len(out_of_range)}\n"
        f"  mean {statistics.mean(values):.4f}  sd {statistics.pstdev(values):.4f}  "
        f"distinct {len(set(values))}\n"
        f"  => {'OK' if ok else 'NOT USABLE'}"
    )
    if len(set(values)) < 10:
        print("  !! fewer than 10 distinct values — check the API key and the model")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="check the column, fetch nothing")
    parser.add_argument("--quarters", nargs="*", default=harness.DEV_QUARTERS)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM,
                        help="requests per minute; TPM is the binding limit")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test a few events first")
    args = parser.parse_args()

    if harness.QUARTERS[-1] in args.quarters:
        print(f"!! {harness.QUARTERS[-1]} is the sealed holdout — generating it is fine, "
              f"scoring on it is not")
    if args.validate:
        raise SystemExit(0 if validate(list(args.quarters)) else 1)

    _throttle = _Throttle(args.rpm)
    generate(args.quarters, workers=args.workers, limit=args.limit)
    validate(list(args.quarters))
