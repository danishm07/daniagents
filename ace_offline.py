"""Offline ACE: batches to disk, reflection in-session, curation deterministic.

The paper's recipe is "a strong model trains the rulebook, a cheap model runs
it". Here the strong model is unmetered — reflection and curation happen
in-session rather than through an API — so the only metered step is the
Generator, and the training phase does not even need that: an *empty* playbook
is exactly the deployed base prompt, and 2,097 of those predictions are already
cached from ``arms.py``.

So the split is:

===================== ============ ==========================================
phase                 cost         what
===================== ============ ==========================================
build batches         **$0**       cached base predictions + residual ranks
reflect               **$0**       in-session, batched, blinded where it matters
curate                **$0**       deterministic ADD-only merge
validation gate       metered      regenerate WITH the playbook, score, accept
                                   or revert
decision-grade score  metered      n>=2,000 held out
===================== ============ ==========================================

Dumping batches to disk rather than holding them in an API call is what makes
the reflection inspectable and rerunnable — a transient call leaves nothing to
audit, and the whole point of the validation gate is that someone can check what
was proposed and why.

    uv run python ace_offline.py build --n 1000
    uv run python ace_offline.py curate --rules rules.json
    uv run python ace_offline.py show
"""

from __future__ import annotations

import argparse
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
from runner import schedule as S  # noqa: E402

OUT = ROOT / "data" / "ace"
TRAIN = OUT / "train"
PLAYBOOK = OUT / "playbook.json"

BATCH = 50
BASE_COLUMN = ROOT / "data" / "arms" / "base__google_gemini-2.5-flash-lite.jsonl"

#: Held back from training so the gate has events the rulebook has never seen.
#: Distinct from the 4,144-event confirmation partition, which is reserved for
#: the decision-grade score and is never touched during training.
VALIDATION_N = 400


def _base_predictions() -> dict[str, float]:
    return {
        json.loads(l)["event_id"]: json.loads(l)["prediction"]
        for l in BASE_COLUMN.open()
        if l.strip()
    }


def build(n_train: int = 1000, seed: int = 20260818) -> None:
    """Batches of 50, sorted by error, written one file per batch."""
    preds = _base_predictions()
    frame = pd.concat([harness.load(q) for q in harness.DEV_QUARTERS], ignore_index=True)
    frame = frame.assign(residual_rank=A.residual_ranks(frame))
    frame = frame[frame.event_id.isin(preds)].copy()
    frame["predicted"] = frame.event_id.map(preds)
    frame = frame.dropna(subset=["predicted", "residual_rank"])

    # The confirmation partition is reserved for the decision-grade score. It is
    # not available to training or to the gate, or the gate would be scoring on
    # events that later have to certify the result.
    confirm = {e["event_id"] for v in S.confirmation_events().values() for e in v}
    pool = frame[~frame.event_id.isin(confirm)].sample(frac=1.0, random_state=seed)

    train = pool.iloc[:n_train]
    validation = pool.iloc[n_train : n_train + VALIDATION_N]

    TRAIN.mkdir(parents=True, exist_ok=True)
    for path in TRAIN.glob("batch_*.json"):
        path.unlink()

    n_batches = len(train) // BATCH
    for i in range(n_batches):
        chunk = train.iloc[i * BATCH : (i + 1) * BATCH].copy()
        chunk["err"] = chunk.predicted - chunk.residual_rank
        rows = [
            {
                "ticker": r.identifier_value,
                "predicted": round(float(r.predicted), 3),
                "actual_residual_rank": round(float(r.residual_rank), 3),
                "facts": list(r.facts)[:10],
            }
            for r in chunk.sort_values("err").itertuples()
        ]
        (TRAIN / f"batch_{i:02d}.json").write_text(
            json.dumps({"batch": i, "n": len(rows), "events": rows}, indent=2)
        )

    (OUT / "splits.json").write_text(
        json.dumps(
            {
                "train_event_ids": train.event_id.tolist(),
                "validation_event_ids": validation.event_id.tolist(),
                "n_batches": n_batches,
                "batch_size": BATCH,
                "seed": seed,
                "confirmation_excluded": len(confirm),
            },
            indent=2,
        )
    )
    print(f"{n_batches} batches of {BATCH} -> {TRAIN}")
    print(f"validation slice: {len(validation)} events (held out of training)")
    print(f"confirmation partition excluded from both: {len(confirm)} events")
    print(f"\ncost so far: $0.00 — every prediction came from the cached base column")


def curate(rules_path: Path) -> None:
    """Apply proposed rules to the playbook. Deterministic ADD-only merge.

    ``rules_path`` is ``[{"section": ..., "content": ...}, ...]``. No LLM
    re-emits the playbook: that is what prevents context collapse, and it is the
    single largest lever in ACE's own ablation (+13.4 average).
    """
    proposals = json.loads(Path(rules_path).read_text())
    playbook = A.Playbook.load(PLAYBOOK)
    before = len(playbook.rules)

    rejected = []
    for proposal in proposals:
        text = (proposal.get("content") or "").strip()
        if not text:
            continue
        # Look-ahead watch: a rule naming a specific company or quarter is not a
        # rule, it is a memorised outcome. The reflecting model may have training
        # exposure to these events.
        leak = _proper_noun_leak(text)
        if leak:
            rejected.append({"content": text, "reason": f"names {leak}"})
            continue
        playbook.add(proposal.get("section", "interactions"), text)

    playbook.save(PLAYBOOK)
    print(f"playbook {before} -> {len(playbook.rules)} rules")
    if rejected:
        print(f"\nREJECTED {len(rejected)} for naming specific entities:")
        for r in rejected:
            print(f"  [{r['reason']}] {r['content'][:90]}")
    print()
    print(playbook.render())


#: Tickers are 1-5 uppercase letters; a rule that names one has memorised an
#: outcome rather than learned a pattern. Allow-list the ordinary uppercase words
#: that legitimately appear in financial prose.
_ALLOWED = {
    # accounting / reporting
    "YOY", "QOQ", "EPS", "EBITDA", "EBIT", "FCF", "GAAP", "IFRS", "FIFO", "LIFO",
    "COGS", "SGA", "SG&A", "R&D", "CAPEX", "OPEX", "DSO", "DPO", "DIO", "PPE",
    # metrics / ratios
    "ROIC", "ROE", "ROA", "ROTE", "NIM", "BPS", "LTV", "CAC", "ARR", "ARPU",
    "MRR", "GMV", "AUM", "NPL", "CET1", "TBV", "EPSX", "IRR", "NPV", "TSR",
    # roles / corporate
    "CEO", "CFO", "COO", "CTO", "CIO", "M&A", "IPO", "SPAC", "JV", "ESOP",
    # business model / channel
    "TAM", "SAM", "SOM", "DTC", "B2B", "B2C", "OEM", "ODM", "SKU", "SaaS",
    "IaaS", "PaaS", "API", "AI", "ML", "EV", "ICE", "P&C", "PC", "QSR",
    # calendar / periods
    "Q1", "Q2", "Q3", "Q4", "FY", "H1", "H2", "LTM", "NTM", "YTD", "TTM",
    # macro / geography
    "US", "EU", "UK", "APAC", "EMEA", "LATAM", "GDP", "CPI", "PPI", "FX",
    "USD", "EUR", "GBP", "FED", "SEC", "IRS",
    # prose connectives that survive the uppercase regex
    "NOT", "AND", "OR", "IF", "BUT", "ALL", "ANY", "IT", "NO", "DO", "BE",
    "CRITICAL", "NEVER", "ALWAYS", "ABOVE", "BELOW", "RANK", "WHEN", "THEN",
}


def _english_words() -> set[str]:
    """The system dictionary, for telling a ticker from an emphasised word.

    An allow-list alone does not work: rules legitimately shout ``GUIDED``,
    ``VOID``, ``NOT`` for emphasis, and the list grows forever while still
    rejecting the next word nobody predicted. The first version of this filter
    rejected the highest-conviction rule in the set because it contained the
    word "GUIDED". A dictionary check is the general form of the same test.
    """
    for path in (Path("/usr/share/dict/words"), Path("/usr/dict/words")):
        if path.exists():
            return {w.strip().lower() for w in path.open() if w.strip()}
    return set()


_WORDS = _english_words()


def _is_english(token: str) -> bool:
    """Is this an ordinary word in caps rather than a ticker?

    The system dictionary stores stems, so ``GUIDED`` is absent while ``guide``
    is present — which is exactly the case that rejected the highest-conviction
    rule in the first curation run. Strip the common inflections before giving
    up on a token.
    """
    word = token.lower()
    if word in _WORDS:
        return True
    for suffix, restore in (("ed", ""), ("ed", "e"), ("s", ""), ("es", ""),
                            ("ing", ""), ("ing", "e"), ("d", ""), ("er", ""),
                            ("ly", ""), ("est", "")):
        if word.endswith(suffix):
            stem = word[: -len(suffix)] + restore
            if len(stem) >= 3 and stem in _WORDS:
                return True
    return False


def _proper_noun_leak(text: str) -> str | None:
    """Flag a rule that names a specific company or year rather than a pattern.

    The reflecting model may have training exposure to these outcomes, so a rule
    naming a ticker is a memorised result wearing a rule's clothes. Flags a
    2-6 character all-caps token only when it is neither a known financial
    abbreviation nor an ordinary English word in caps.
    """
    import re

    # 2-5 characters: the actual range of US equity tickers. A 6+ character
    # all-caps token is emphasis, not a symbol.
    for token in re.findall(r"\b[A-Z][A-Z0-9.]{1,4}\b", text):
        if token in _ALLOWED or _is_english(token):
            continue
        return f"possible ticker {token!r}"
    for year in re.findall(r"\b(?:19|20)[0-9]{2}\b", text):
        return f"specific year {year!r}"
    return None


def show() -> None:
    playbook = A.Playbook.load(PLAYBOOK)
    print(playbook.render())
    print(json.dumps(playbook.stats(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "curate", "show"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--rules", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        build(args.n)
    elif args.command == "curate":
        if not args.rules:
            raise SystemExit("--rules <path to proposals json> required")
        curate(args.rules)
    else:
        show()
