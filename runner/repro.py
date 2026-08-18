"""The reproducibility gap: why does the same public code produce three numbers?

| | ΔR² | % obtainable |
|---|---|---|
| Gemini Flash Lite — Summary, live leaderboard | 0.0917 | **9.7%** |
| its archive column claims | — | 4.17% |
| our replay of the published repo | — | **1.05–1.85%** |
| our live own-sample | 0.0534 | 5.9% |

Closing this is worth ~4pp. The entire six-agent data cycle bought +0.31pp. It
needs no new data source, and steps 1–4 cost nothing.

**The discipline this module enforces is the kill condition.** Each hypothesis
gets a falsification criterion written down *before* its test runs, and
:func:`log` refuses a result whose kill condition was not registered first. That
is not ceremony: this project has repeatedly produced a confident conclusion from
a measurement that could not carry it — a "SIGN FLIP" at z = 0.53, a "+0.0150
model lever" that was a prompt effect, a peer ρ that was one-fifth leak and
seven-tenths small sample. A criterion written after seeing the number is not a
criterion.

Run the free ladder::

    uv run python -m runner.repro --free
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = ROOT / "runs" / "repro_log.jsonl"


@dataclass
class Hypothesis:
    """One explanation, with the test that could kill it stated in advance."""

    key: str
    claim: str
    #: The test that changes what you believe about the OTHER hypotheses, not
    #: just this one. Run these first — that is what makes this a ladder rather
    #: than a list.
    discriminating_test: str
    #: Written before the test. If met, the hypothesis is closed and logged, and
    #: not revisited on a hunch.
    kill_condition: str
    predicts: str = ""
    status: str = "open"
    result: str = ""
    evidence: dict = field(default_factory=dict)


HYPOTHESES = {
    "H1": Hypothesis(
        key="H1",
        claim=(
            "DSPy adapter/version drift. Documented regression stanfordnlp/dspy #6743: "
            "vanilla ChainOfThought emits a BLANK description for the reasoning field, "
            "losing the step-by-step instruction that issue #409 (2024) shows older DSPy "
            "put in the prompt verbatim: 'Reasoning: Let's think step by step in order to "
            "${produce the answer}.'"
        ),
        discriminating_test=(
            "Dump the actual prompt DSPy sends via lm.inspect_history(). Is the reasoning "
            "field's description blank or populated?"
        ),
        predicts=(
            "If version drift is the cause, BOTH model columns should fail to reproduce, "
            "because the prompt degradation is model-independent."
        ),
        kill_condition=(
            "The dumped prompt contains an explicit step-by-step / reasoning instruction "
            "attached to the reasoning field. If the instruction is present, the documented "
            "regression is not firing here and H1 is closed."
        ),
    ),
    "H2": Hypothesis(
        key="H2",
        claim=(
            "Model alias resolution. The baseline names gemini/gemini-flash-lite-latest, a "
            "floating alias; what it resolved to when the archive was generated is not what "
            "it resolves to now. The GPT column is pinned (gpt-5-nano-2025-08-07)."
        ),
        discriminating_test=(
            "Does the PINNED GPT column reproduce? Free, and it separates H1 from H2 "
            "immediately."
        ),
        predicts=(
            "If the alias is the cause, the pinned GPT column MUST reproduce and only the "
            "Gemini column should fail."
        ),
        kill_condition=(
            "The pinned GPT column also fails to reproduce. A pinned model cannot drift, so "
            "if it fails too the alias is not the cause and H2 is closed — H1 and H3 gain "
            "prior."
        ),
    ),
    "H3": Hypothesis(
        key="H3",
        claim=(
            "The archive column was produced by a pipeline that differs from the published "
            "code — different input, a retry/aggregation step, or a different percentile "
            "normalisation."
        ),
        discriminating_test=(
            "Compare the value DISTRIBUTION of the archive columns against ours. Gemini's "
            "archive column reportedly has a tie group of 836 events; ours has 62 distinct "
            "values. A free-form float output does not produce an 836-event tie group."
        ),
        predicts=(
            "If a quantisation or aggregation step exists that we do not have, the archive "
            "column's distribution will show structure ours lacks — heavy ties on round "
            "numbers, or a coarse grid."
        ),
        kill_condition=(
            "The archive columns' value distributions are structurally the same shape as "
            "ours — similar distinct-value counts and no tie group we cannot produce. Then "
            "the difference is upstream of output formatting and this specific form of H3 "
            "is closed."
        ),
    ),
    "H4": Hypothesis(
        key="H4",
        claim=(
            "There is no gap to close: 9.7% and 5.9% are not the same sample. The "
            "leaderboard exposes own-sample, imputed and n per submission."
        ),
        discriminating_test=(
            "Match the leaderboard samples before treating the gap as real. This check has "
            "already dissolved one apparent gap in this project (0.378 -> 16%)."
        ),
        predicts=(
            "If the samples differ, the baseline's 9.7% and our 5.9% are not comparable and "
            "part of the 'gap' is arithmetic rather than capability."
        ),
        kill_condition=(
            "The baseline's 9.7% and our 5.9% are measured on samples of comparable size and "
            "difficulty. Then the gap is real and H4 is closed."
        ),
    ),
}


def register(keys=None) -> None:
    """Write the kill conditions down. Must happen before any test result."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with LOG.open("a") as fh:
        for key in keys or HYPOTHESES:
            record = asdict(HYPOTHESES[key])
            record.update(kind="hypothesis_registered", timestamp=stamp)
            fh.write(json.dumps(record) + "\n")


def registered() -> set[str]:
    if not LOG.exists():
        return set()
    return {
        json.loads(line)["key"]
        for line in LOG.open()
        if line.strip() and json.loads(line).get("kind") == "hypothesis_registered"
    }


def log(key: str, *, result: str, killed: bool, evidence: dict) -> None:
    """Record a test outcome. Refuses if the kill condition was not registered first."""
    if key not in registered():
        raise RuntimeError(
            f"{key} has no registered kill condition. Write it down before you look at the "
            "number — a criterion chosen after seeing the result is not a criterion."
        )
    hypothesis = HYPOTHESES[key]
    hypothesis.status = "closed" if killed else "open"
    hypothesis.result = result
    hypothesis.evidence = evidence
    with LOG.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "hypothesis_result",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "key": key,
                    "kill_condition": hypothesis.kill_condition,
                    "kill_condition_met": killed,
                    "status": hypothesis.status,
                    "result": result,
                    "evidence": evidence,
                },
                default=str,
            )
            + "\n"
        )
    print(f"\n[{key}] {'KILLED — branch closed' if killed else 'SURVIVES'}: {result}")


def summary() -> None:
    print(f"\n{'=' * 92}\nHYPOTHESIS LADDER\n{'=' * 92}")
    for h in HYPOTHESES.values():
        mark = "x" if h.status == "closed" else " "
        print(f"[{mark}] {h.key}  {h.claim[:78]}")
        print(f"      discriminator: {h.discriminating_test[:76]}")
        print(f"      kill if:       {h.kill_condition[:76]}")
        if h.result:
            print(f"      RESULT:        {h.result}")


if __name__ == "__main__":
    register()
    summary()
    print(f"\nkill conditions registered to {LOG}")
