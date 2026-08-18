"""Gate 0 — can the Reflector tell a real residual batch from a shuffled one?

The test none of the source material runs, and the one most likely to stop this
build. LLMs assert patterns in structureless data 72-100% of the time
(arXiv:2510.09709: GPT-4.1 acknowledged randomness in 5.0% of random integer
sequences, o3 in 28.0%). ACE was validated on tasks with verifiable answers,
where that failure mode cannot surface. Ours is ~94% noise per event, where it
is the default outcome.

**Blinded, because the Reflector is a model running in-session.** If the
reflecting model knows which batches had their labels destroyed, the test
measures nothing. So :func:`build` writes six batches under opaque ids, keeps
the assignment in a separate file, and :func:`reveal` unseals it only after the
verdicts are recorded. The reflector reads ``gate0_batches.json`` and never the
key.

Costs nothing: the Generator's predictions come from the cached
``base__google_gemini-2.5-flash-lite`` column, and shuffling is free.

    uv run python gate0.py build       # writes batches + sealed key
    uv run python gate0.py reveal      # after verdicts are recorded
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ace as A  # noqa: E402
import harness  # noqa: E402

OUT = ROOT / "data" / "ace"
BATCHES = OUT / "gate0_batches.json"
KEY = OUT / "gate0_key.json"
VERDICTS = OUT / "gate0_verdicts.json"

BATCH_SIZE = 50
N_REAL = 3
N_SHUFFLED = 3

#: The Generator's cached output. Real predictions from the deployed-shape
#: prompt, so the residuals are the ones the loop would actually reflect on.
GENERATOR_COLUMN = ROOT / "data" / "arms" / "base__google_gemini-2.5-flash-lite.jsonl"


# --------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS — written before any batch was generated
# --------------------------------------------------------------------------

THRESHOLDS = {
    "registered_at": "2026-08-18, before build() was first run",
    "primary": {
        "name": "rule-production ratio",
        "rule": "KILL if mean rules proposed on SHUFFLED batches >= 0.60 x mean "
                "rules proposed on REAL batches",
        "why": (
            "Shuffling destroys the fact-to-ranking relationship by construction, so a "
            "discriminating reflector should collapse toward proposing nothing. If "
            "destroying all signal cuts rule production by less than 40%, then most rules "
            "on real data are not attributable to the label relationship — and since real "
            "data is itself ~94% noise per event, the genuinely signal-attributable "
            "fraction would be far below even that."
        ),
    },
    "secondary": {
        "name": "null-acknowledgement rate",
        "rule": "KILL if the reflector declines to propose rules on fewer than 50% of "
                "SHUFFLED batches",
        "why": (
            "The prompt explicitly offers 'these groups are not distinguishable; propose "
            "no rule' as a first-class outcome. Explicit permission raised randomness "
            "acknowledgement from 12.8% to 75.8% in arXiv:2510.09709, so 50% on batches "
            "with no signal at all is a lenient bar. ACE's own prompt does the opposite — "
            "it demands error_identification and root_cause_analysis as required fields, "
            "structurally forbidding 'nothing here'."
        ),
    },
    "both_must_pass": True,
    "note": (
        "n=3 shuffled batches is a small sample and these thresholds are coarse by "
        "necessity. They are set to catch a gross failure — a reflector that cannot tell "
        "signal from noise at all — not to certify a subtle one. Passing Gate 0 is "
        "necessary, not sufficient; K2 and K3 in runs/ace_gonogo.md remain."
    ),
}


def _generator_column() -> dict[str, float]:
    return {
        json.loads(l)["event_id"]: json.loads(l)["prediction"]
        for l in GENERATOR_COLUMN.open()
        if l.strip()
    }


def build(seed: int = 20260818) -> None:
    """Six batches under opaque ids, plus a sealed key."""
    rng = np.random.default_rng(seed)
    preds = _generator_column()

    frame = pd.concat([harness.load(q) for q in harness.DEV_QUARTERS], ignore_index=True)
    frame = frame.assign(residual_rank=A.residual_ranks(frame))
    frame = frame[frame.event_id.isin(preds)].copy()
    frame["predicted"] = frame.event_id.map(preds)
    frame = frame.dropna(subset=["predicted", "residual_rank"])

    pool = frame.sample(n=BATCH_SIZE * (N_REAL + N_SHUFFLED), random_state=seed)
    chunks = [pool.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] for i in range(N_REAL + N_SHUFFLED)]

    kinds = ["real"] * N_REAL + ["shuffled"] * N_SHUFFLED
    order = rng.permutation(len(chunks))

    batches, key = {}, {}
    for position, index in enumerate(order):
        chunk = chunks[index].copy()
        kind = kinds[index]
        batch_id = hashlib.sha256(f"{seed}-{position}".encode()).hexdigest()[:8]
        if kind == "shuffled":
            # Destroy the fact->ranking relationship, keep both marginals exactly.
            # The batch is statistically identical in every respect except the
            # one thing the reflector is asked to find.
            chunk["residual_rank"] = rng.permutation(chunk["residual_rank"].to_numpy())
        chunk["err"] = chunk["predicted"] - chunk["residual_rank"]

        rows = []
        for r in chunk.sort_values("err").itertuples():
            rows.append(
                {
                    "ticker": r.identifier_value,
                    "predicted": round(float(r.predicted), 3),
                    "actual_residual_rank": round(float(r.residual_rank), 3),
                    "facts": list(r.facts)[:10],
                }
            )
        batches[batch_id] = {"batch_id": batch_id, "n": len(rows), "events": rows}
        key[batch_id] = {"kind": kind, "position": position, "source_chunk": int(index)}

    OUT.mkdir(parents=True, exist_ok=True)
    BATCHES.write_text(json.dumps({"thresholds": THRESHOLDS, "batches": batches}, indent=2))
    KEY.write_text(json.dumps(key, indent=2))
    print(f"wrote {len(batches)} batches of {BATCH_SIZE} to {BATCHES.name}")
    print(f"sealed key to {KEY.name} — DO NOT READ until verdicts are recorded")
    print(f"\nbatch ids (order is randomised, {N_REAL} real / {N_SHUFFLED} shuffled):")
    for batch_id in batches:
        print(f"  {batch_id}")


def record(verdicts: dict) -> None:
    """``{batch_id: {"n_rules": int, "confidence": "high|medium|low|none", "rules": [...]}}``"""
    VERDICTS.write_text(json.dumps(verdicts, indent=2))
    print(f"recorded {len(verdicts)} verdicts to {VERDICTS.name}")


def reveal() -> dict:
    if not VERDICTS.exists():
        raise SystemExit("record verdicts before revealing the key")
    key = json.loads(KEY.read_text())
    verdicts = json.loads(VERDICTS.read_text())

    rows = []
    for batch_id, verdict in verdicts.items():
        rows.append(
            {
                "batch": batch_id,
                "kind": key[batch_id]["kind"],
                "n_rules": verdict["n_rules"],
                "confidence": verdict.get("confidence", ""),
                "declined": verdict["n_rules"] == 0,
            }
        )
    frame = pd.DataFrame(rows).sort_values("kind")
    print(frame.to_string(index=False))

    real = frame[frame.kind == "real"].n_rules.mean()
    shuffled = frame[frame.kind == "shuffled"].n_rules.mean()
    ratio = (shuffled / real) if real else float("inf")
    declined = float((frame[frame.kind == "shuffled"].n_rules == 0).mean())

    print(f"\nmean rules  real {real:.2f}   shuffled {shuffled:.2f}   ratio {ratio:.2f}")
    print(f"null-acknowledgement rate on shuffled batches: {declined:.0%}")

    primary_fail = ratio >= 0.60
    secondary_fail = declined < 0.50
    print(f"\nPRE-REGISTERED THRESHOLDS")
    print(f"  primary   ratio < 0.60 : {ratio:.2f} -> {'FAIL' if primary_fail else 'PASS'}")
    print(f"  secondary declined >= 50% : {declined:.0%} -> "
          f"{'FAIL' if secondary_fail else 'PASS'}")
    verdict = "KILL — build stops" if (primary_fail or secondary_fail) else "PASS — proceed"
    print(f"\n  => {verdict}")

    out = {
        "mean_rules_real": real,
        "mean_rules_shuffled": shuffled,
        "ratio": ratio,
        "null_acknowledgement_rate": declined,
        "primary_pass": not primary_fail,
        "secondary_pass": not secondary_fail,
        "verdict": verdict,
        "per_batch": rows,
    }
    (OUT / "gate0_result.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    if command == "build":
        build()
    elif command == "reveal":
        reveal()
    else:
        raise SystemExit(f"unknown command {command!r}")
