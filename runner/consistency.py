"""Live/offline consistency — the frozen-vector test, for the prediction path.

``BUILD_LOOP.md``, requirement 4:

    Take a recent real event, run both paths, assert identical output. Same role
    the frozen HMAC vectors play for the webhook.

The failure this exists to prevent has already happened once and cost the
project a month of misdirected measurement: the offline champion was a proxy
(``gpt-5-nano`` + the old prompt) while production ran ``gpt-5.4-nano`` + a
different prompt, so every ρ_b and every promotion floor was measured against
the wrong object. Nothing in the numbers themselves showed it.

What is asserted, and what each one catches:

``payload``    the offline payload is the confirmed live shape — the archive's
               ``disclosure`` object *unwrapped*, ``{schema_version, event_id,
               generated_at, items}`` at top level. This is the shape the
               payload bug got wrong; production logged 14 ``no facts found``
               before it was caught.
``prompt``     what an arm sends is byte-identical to what ``predict._ask_llm``
               sends. Catches a sweep that silently reworded the prompt while
               claiming to vary only the model — which is exactly how "+0.0150
               model lever" turned out to be a prompt effect.
``system``     same, for the system prompt.
``model``      the champion column's fingerprint matches the currently deployed
               model + prompt. Catches a stale column.
``determinism``an arm's ``predict`` returns the same values twice. The scheduler
               scores the same arm at three rungs and treats them as three
               subsets of one column; a stochastic ``predict`` would silently
               make them three different columns.

Run it::

    uv run python -m runner.consistency
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENT = ROOT.parent / "agent"
for extra in (str(AGENT), str(AGENT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

#: The live body's top-level keys, read off two real production events on
#: 2026-08-14. Frozen here so a schema drift surfaces as a failing test rather
#: than as a quarter of degraded predictions.
LIVE_PAYLOAD_KEYS = {"schema_version", "event_id", "generated_at", "items"}
LIVE_ITEM_KEYS = {"id", "kind", "source", "media_type", "content", "url", "bytes", "sha256"}


class ConsistencyError(AssertionError):
    pass


def check_payload_shape(event: dict) -> str:
    import champion

    payload = champion.live_payload(event["facts"])
    if set(payload) != LIVE_PAYLOAD_KEYS:
        raise ConsistencyError(
            f"payload keys {sorted(payload)} != confirmed live shape {sorted(LIVE_PAYLOAD_KEYS)}"
        )
    items = payload["items"]
    if len(items) != 1:
        raise ConsistencyError(f"live payload carries exactly one disclosure item, got {len(items)}")
    item = items[0]
    missing = LIVE_ITEM_KEYS - set(item)
    if missing:
        raise ConsistencyError(f"disclosure item is missing {sorted(missing)}")
    if item["kind"] != "facts":
        raise ConsistencyError(f"disclosure kind is {item['kind']!r}, expected 'facts'")
    if list(item["content"]) != list(event["facts"]):
        raise ConsistencyError("payload content does not round-trip the event's facts")
    return f"payload shape matches production ({len(item['content'])} facts, kind={item['kind']})"


def check_prompt(event: dict) -> str:
    """Capture what production would send, without spending a call.

    ``predict._openai`` is swapped for a fake that records the kwargs and
    aborts, so this runs offline and costs nothing — which is what makes it
    cheap enough to be a standing test rather than an occasional audit.
    """
    import champion
    import predict
    import reads

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

    if not captured:
        raise ConsistencyError("predict._ask_llm never reached the client — cannot compare paths")

    theirs = [m for m in captured["messages"] if m["role"] == "user"][0]["content"]
    mine = reads.user_prompt(payload, event["ticker"], champion.EVENT_TYPE)
    if theirs != mine:
        raise ConsistencyError(
            "the research prompt has drifted from the deployed prompt:\n"
            f"  production: {theirs[:160]!r}\n  research:   {mine[:160]!r}"
        )
    system = [m for m in captured["messages"] if m["role"] == "system"][0]["content"]
    if system != predict.SYSTEM_PROMPT:
        raise ConsistencyError("system prompt differs between the two paths")

    # A context arm is the same prompt with a block prepended. Assert the base
    # is preserved verbatim rather than reformatted, or every context result is
    # confounded with a prompt change.
    import arms as A

    with_context = A.build_prompt({**event, "quarter": "2026Q2"}, "CONTEXT BLOCK\n")
    if not with_context.endswith(mine):
        raise ConsistencyError("a context arm rewrites the deployed prompt instead of prefixing it")
    return f"prompt and system prompt identical across paths ({len(mine)} chars); context arms prefix"


def check_champion_fingerprint() -> str:
    import champion

    expected = champion.prompt_fingerprint()
    path = champion.OUT
    if not path.exists():
        raise ConsistencyError("no champion column — every rho_b would be against a proxy")
    import json

    seen = set()
    for line in path.open():
        if line.strip():
            seen.add(json.loads(line).get("fingerprint"))
    stale = {f for f in seen if f and f != expected}
    if stale:
        raise ConsistencyError(
            f"champion column carries fingerprints {sorted(stale)}, deployed is {expected!r} — "
            "regenerate with `uv run python champion.py`"
        )
    return f"champion column matches the deployed model + prompt (fingerprint {expected})"


def check_determinism(arm_names: list[str], quarter: str = "2026Q2", n: int = 40) -> str:
    """Every arm's ``predict`` must return the same thing twice."""
    import harness
    from runner import registry as R

    events = harness.events_for(quarter)[:n]
    drifted = []
    for name in arm_names:
        arm = R.ARMS.get(name)
        if arm is None:
            continue
        first = arm.predict(events, quarter)
        second = arm.predict(events, quarter)
        same = all(
            (a != a and b != b) or a == b for a, b in zip(first, second, strict=True)
        )
        if not same:
            drifted.append(name)
    if drifted:
        raise ConsistencyError(
            f"predict() is not deterministic for {drifted} — the rungs are not subsets of one column"
        )
    return f"predict() is deterministic across {len(arm_names)} arms"


def run(verbose: bool = True) -> list[str]:
    import harness
    from runner import arms_builtin  # noqa: F401  — registration side effect
    from runner import registry as R

    # The most recent dev event: the closest thing offline to "a recent real
    # event", and it exercises the same facts payload production receives.
    events = harness.events_for("2026Q2")
    event = max(events, key=lambda e: e["event_datetime"])

    checks = [
        check_payload_shape(event),
        check_prompt(event),
        check_champion_fingerprint(),
        check_determinism(sorted(R.ARMS)[:12]),
    ]
    if verbose:
        print(f"live/offline consistency on {event['ticker']} @ {event['event_datetime']}")
        for line in checks:
            print(f"  ok  {line}")
    return checks


if __name__ == "__main__":
    try:
        run()
    except ConsistencyError as exc:
        print(f"\nFAILED: {exc}")
        raise SystemExit(1)
    print("\nall consistency checks passed")
