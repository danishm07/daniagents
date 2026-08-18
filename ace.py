"""ACE — a rulebook grown by generate / reflect / curate, adapted for a noisy continuous target.

Zhang et al. 2026 (arXiv:2510.04618), applied to this task by Koijen & Levy
(SSRN 6474601, §3.4.3) who report combined R² 14.1% → 17.1%, +3.0pp, and a
rulebook that transfers from Opus down to Haiku.

**Read this before changing anything: ACE as published is a binary-correctness
algorithm end to end.** Its control flow hinges on
``data_processor.answer_is_correct(pred, target) -> bool``. There is no
continuous loss, no ranking metric and no error magnitude anywhere in the
official implementation. Our target is a percentile whose best known predictor
correlates ~0.25 with the outcome. So every adaptation below is *our design*,
not theirs, and is marked as such — the paper leaves it open rather than
answering it badly.

Four facts from the reference implementation that contradict the summaries, and
that this module is built to rather than around:

1. **Only ``ADD`` exists.** ``UPDATE`` / ``MERGE`` / ``DELETE`` are a
   commented-out ``TODO`` in ``ace-agent/ace``. Koijen & Levy's description of
   rules being "updated based on whether they improved or degraded performance"
   paraphrases ACE's *abstract*, not its code.
2. **The counters are decorative.** Nothing in the reference ever reads
   ``harmful`` to delete, demote or reorder. Their only effect is that the
   Generator sees them.
3. **The Reflector is the smallest lever** — +1.7 of +17.0 on AppWorld, against
   ~+13.4 for incremental delta updates. Generator+Curator alone captures ~75%.
4. **There is no validation gating at the delta level.** Every ADD is committed
   unconditionally; the only check is a 100-sample checkpoint snapshot.

(4) is survivable when feedback is a correct/incorrect bit. It is not survivable
here, and :func:`Session.step` gates every delta block on a held-out slice.

Our adaptations, in order of how much they matter:

``batched reflection``  The Reflector never sees a single event. At ρ≈0.25 one
                        event's error is ~94% noise; a rule inferred from it is
                        a rule inferred from noise. It sees a batch and is asked
                        for what separates the batch's best-ranked residuals
                        from its worst.
``residual target``     Following Koijen & Levy: regress the outcome on the
                        surprise first, reflect on the *residual*. The benchmark
                        already owns whatever the surprise explains.
``anti-level directive``ΔR² is invariant to any affine remap, so a rule that
                        moves every prediction the same way is worth exactly
                        zero. GEPA converged on precisely that ("good is priced
                        in") because de-biasing is the path of least resistance
                        under MSE. Both prompts forbid it explicitly.
``validation gate``     Delta blocks are accepted only if they beat a bootstrap
                        noise band on a held-out slice, else reverted.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "ace"

#: The reference's seven sections are generic ("FORMULAS & CALCULATIONS", "CODE
#: SNIPPETS"). Section names are the caller's choice in the reference too, so
#: these are ours, shaped by what the ten facts actually contain.
SECTIONS = (
    "surprise_quality",       # is the beat clean, managed, or one-off driven
    "guidance_and_outlook",   # the second derivative — what the leaders price
    "margins_and_cashflow",   # harder to manage than headline earnings
    "narrative_and_tone",     # management language, hedging, confidence
    "interactions",           # rules that only fire in combination
    "common_mistakes",        # what the model gets wrong, stated as a warning
)

#: Rules take Koijen & Levy's form — condition(s) → percentile band (Figure 3):
#: "pre-profit, revenue growth >30% YoY BUT operating losses flat → 0.10-0.20".
#: Bands are not calibration: different conditions map to different bands, so
#: they are an *ordering* device and survive affine invariance.
RULE_FORM = "condition(s) -> percentile band, e.g. '... -> 0.10-0.20'"

_LINE = re.compile(r"\[([^\]]+)\]\s*helpful=(-?\d+)\s*harmful=(-?\d+)\s*::\s*(.*)")


# --------------------------------------------------------------------------
# The playbook
# --------------------------------------------------------------------------


@dataclass
class Rule:
    rule_id: str
    section: str
    text: str
    helpful: int = 0
    harmful: int = 0
    #: Ours, not the reference's. The mean signed residual over events whose
    #: Generator cited this rule, and the citation count. Per-event helpful /
    #: harmful tags are noise at our SNR; a mean over n citations is not.
    contribution: float = 0.0
    citations: int = 0

    def render(self) -> str:
        return f"[{self.rule_id}] helpful={self.helpful} harmful={self.harmful} :: {self.text}"


@dataclass
class Playbook:
    """Sectioned, flat within sections, one rule per line. A string at rest.

    The merge is deterministic Python and never asks an LLM to re-emit the
    playbook — that is what prevents context collapse, and in the reference's
    own ablation it is the single largest lever in the method.
    """

    rules: list[Rule] = field(default_factory=list)
    next_id: int = 1

    def add(self, section: str, text: str) -> Rule:
        section = section.lower().replace(" ", "_").replace("&", "and")
        if section not in SECTIONS:
            section = "interactions"
        rule = Rule(rule_id=f"{section[:3]}-{self.next_id:05d}", section=section, text=text.strip())
        self.next_id += 1
        self.rules.append(rule)
        return rule

    def by_id(self) -> dict[str, Rule]:
        return {r.rule_id: r for r in self.rules}

    def render(self) -> str:
        if not self.rules:
            return "(the playbook is empty — you are seeing it before any rules exist)"
        out = []
        for section in SECTIONS:
            members = [r for r in self.rules if r.section == section]
            if not members:
                continue
            out.append(f"## {section.upper().replace('_', ' ')}")
            out.extend(r.render() for r in members)
            out.append("")
        return "\n".join(out)

    def stats(self) -> dict:
        cited = [r for r in self.rules if r.citations]
        return {
            "total_rules": len(self.rules),
            "cited_at_least_once": len(cited),
            "never_cited": len(self.rules) - len(cited),
            "mean_contribution": float(np.mean([r.contribution for r in cited])) if cited else 0.0,
            "by_section": {s: sum(1 for r in self.rules if r.section == s) for s in SECTIONS},
        }

    def prune(self, min_citations: int = 25) -> list[Rule]:
        """Drop rules that are well-cited and non-positive. **Ours, not theirs.**

        The reference has no pruning at all — the ablation that varies it is not
        reproducible from the public repo, because the mechanism does not exist
        there. Monotone growth is variance growth over thousands of events, so a
        bound is necessary; the criterion is a choice and this one is
        deliberately conservative: a rule must have been *used* enough times for
        its mean contribution to mean something before it can be removed.
        """
        keep, dropped = [], []
        for rule in self.rules:
            if rule.citations >= min_citations and rule.contribution <= 0:
                dropped.append(rule)
            else:
                keep.append(rule)
        self.rules = keep
        return dropped

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"next_id": self.next_id, "rules": [r.__dict__ for r in self.rules]}, indent=2
            )
        )

    @classmethod
    def load(cls, path: Path) -> Playbook:
        if not path.exists():
            return cls()
        blob = json.loads(path.read_text())
        return cls(rules=[Rule(**r) for r in blob["rules"]], next_id=blob["next_id"])


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

#: Stated in both the Reflector and Curator prompts. Without it this reproduces
#: the paper's GEPA result: a beautifully de-biased model with no ΔR² gain.
AFFINE_WARNING = """\
CRITICAL — how this is scored, and it rules out most obvious rules:
The score is the squared partial correlation between the prediction and the outcome,
after the earnings surprise is projected out. It is INVARIANT to any affine remap of
the predictions: p -> a*p + b scores identically. Therefore:
  - A rule that shifts ALL predictions in one direction is worth EXACTLY ZERO.
  - "The model is systematically too optimistic" is NOT a usable finding.
  - Rescaling, recentring, or widening/narrowing the whole distribution: worth zero.
Only ORDERING and RELATIVE SPACING are scored. Every rule you propose must
DISCRIMINATE — it must push some events up relative to OTHER events. A rule that
fires on almost every event, or that moves everything the same way, is useless
however true it is.
"""

#: The objective, stated from the measured decomposition rather than in the
#: abstract. Every event contributes e_p * e_y to the score, where e_p is our
#: prediction net of the benchmark and e_y is the outcome net of the benchmark.
#: Measured over 6,144 dev events:
#:
#:     44.0% of events contribute NEGATIVELY, totalling -0.233
#:     56.0% contribute positively,           totalling +0.437
#:     net rho = +0.204
#:
#: So the score is the small residue of two large opposing flows, and the
#: reachable win is shrinking the negative side rather than growing the positive
#: one. A rule that stops us being confidently wrong on 200 events is worth more
#: than a rule that makes us slightly righter on 2,000.
OBJECTIVE = """\
WHAT WE ARE ACTUALLY TRYING TO FIX — read this before proposing anything.

Every event adds (our prediction, net of the benchmark) x (the outcome, net of the
benchmark) to the score. Measured across 6,144 past events:

  44% of events contribute NEGATIVELY, totalling  -0.233
  56% contribute positively,            totalling +0.437
  net                                             +0.204

The score is the small residue of two large opposing flows. **The win is shrinking
the negative flow, not growing the positive one.** A rule that stops the model
being confidently wrong on a few hundred events is worth far more than a rule that
makes it slightly more right on thousands.

THE SPECIFIC FAILURE, measured. The most damaging events are ones the model ranked
far TOO LOW that then ripped:

    predicted 0.18 -> actual 0.90      predicted 0.23 -> actual 0.93
    predicted 0.28 -> actual 0.97      predicted 0.12 -> actual 0.89

These are mostly quarters that LOOK bad — weak or declining revenue, guidance
trimmed, a loss — where the market nonetheless marked the stock up hard. Something
in the facts of a bad-looking quarter is being read as straightforwardly bearish
when the market reads it as a turn.

An independent analysis of one batch found the mechanism: among events with
declining year-over-year revenue, those whose call showed a REALIZED TURN
(sequential revenue or EBITDA growth, book-to-bill above 1x, a rising or record
backlog, a destocking target met, a named drag with a completion date) landed at
0.79 on average, while those without landed at 0.32 — and the model predicted 0.49
and 0.42 respectively. It discriminated on the LEVEL of the decline when it should
have discriminated on the SECOND DERIVATIVE.

Aim there first. Rules that rescue wrongly-pessimistic events are the highest-value
thing you can produce.
"""

GENERATOR_SYSTEM = """\
You are an equity analyst predicting how the market will react to an earnings announcement.

You are given a curated playbook of rules learned from thousands of past announcements, and
the key facts from one earnings call. Apply the playbook.

Instructions:
- Read the playbook carefully and apply the rules that fit this event.
- Rules are written as: condition(s) -> percentile band. When a rule's condition holds,
  its band is strong evidence for where this event should land.
- Where rules conflict, prefer the more specific condition, and say so in your reasoning.
- Note which rules actually informed you: report their bullet ids.
- Your prediction is a percentile in [0,1] against ALL other announcements this quarter.
  0 = most negative reaction, 0.5 = typical, 1 = most positive.
"""

REFLECTOR_SYSTEM = """\
You are diagnosing a prediction model by looking at a BATCH of its predictions against what
actually happened, and proposing what the model should have known.

You are shown many events at once ON PURPOSE. Any single earnings reaction is dominated by
noise — even a perfect model correlates only about 0.25 with the outcome. A pattern you infer
from one event is a pattern you inferred from noise. Only propose something you can see
ACROSS events in this batch.

""" + AFFINE_WARNING + OBJECTIVE + """
Your job:
- Compare the events where the model ranked TOO HIGH against those where it ranked TOO LOW.
- Find what distinguishes them: a fact pattern, a phrasing, a combination of conditions.
- Say which existing playbook rules helped and which misled, by id.
- Propose insights in the form: condition(s) -> percentile band.
"""

CURATOR_SYSTEM = """\
You are the curator of a playbook of rules used to predict earnings-announcement reactions.

You are given the current playbook and a reflection over a batch of recent predictions. Your
job is to identify ONLY the NEW insights that are MISSING from the playbook, and emit them as
additions.

""" + AFFINE_WARNING + OBJECTIVE + """
Instructions:
- Do NOT regenerate the playbook. Emit additions only.
- Avoid redundancy: if similar advice exists, add nothing unless yours is a genuine complement.
- Each rule must be specific and actionable, in the form: condition(s) -> percentile band.
- A rule must DISCRIMINATE between events. Reject your own proposal if it would fire on most
  announcements or would move all predictions the same way.
- Quality over quantity. Emitting zero operations is a valid and often correct answer.
"""


def generator_prompt(playbook: Playbook, facts: list[str], ticker: str) -> str:
    body = "\n".join(f"- {f}" for f in facts)
    return (
        f"PLAYBOOK:\n{playbook.render()}\n\n"
        f"EVENT — {ticker}\nFacts from the earnings call:\n{body}\n\n"
        "Return JSON: {\"reasoning\": str, \"bullet_ids\": [str], "
        "\"predicted_percentile\": float in [0,1]}"
    )


def reflector_prompt(playbook: Playbook, batch: pd.DataFrame) -> str:
    """``batch`` carries ticker, facts, predicted, residual_rank and cited ids."""
    worst = batch.nlargest(min(8, len(batch)), "err")
    best = batch.nsmallest(min(8, len(batch)), "err")

    def block(frame: pd.DataFrame, label: str) -> str:
        rows = []
        for r in frame.itertuples():
            facts = "\n".join(f"    - {f}" for f in list(r.facts)[:10])
            rows.append(
                f"  [{r.ticker}] predicted {r.predicted:.2f}, actual residual rank "
                f"{r.residual_rank:.2f}\n{facts}"
            )
        return f"{label}\n" + "\n\n".join(rows)

    cited = sorted({i for ids in batch.cited for i in ids})
    index = playbook.by_id()
    cited_text = "\n".join(index[i].render() for i in cited if i in index) or "(none cited)"

    return (
        f"BATCH OF {len(batch)} EVENTS.\n\n"
        f"{block(worst, 'MODEL RANKED THESE TOO HIGH (predicted >> actual):')}\n\n"
        f"{block(best, 'MODEL RANKED THESE TOO LOW (predicted << actual):')}\n\n"
        f"PLAYBOOK RULES THE MODEL CITED ON THIS BATCH:\n{cited_text}\n\n"
        "Return JSON: {\"reasoning\": str, \"what_separates_the_two_groups\": str, "
        "\"key_insights\": [str], \"bullet_tags\": [{\"id\": str, "
        "\"tag\": \"helpful\"|\"harmful\"|\"neutral\"}]}"
    )


def curator_prompt(playbook: Playbook, reflection: str) -> str:
    return (
        f"CURRENT PLAYBOOK:\n{playbook.render()}\n\n"
        f"PLAYBOOK STATS: {json.dumps(playbook.stats())}\n\n"
        f"REFLECTION OVER THE MOST RECENT BATCH:\n{reflection}\n\n"
        f"Available sections: {', '.join(SECTIONS)}\n"
        f"Rule form: {RULE_FORM}\n\n"
        "Return JSON: {\"reasoning\": str, \"operations\": "
        "[{\"type\": \"ADD\", \"section\": str, \"content\": str}]}"
    )


# --------------------------------------------------------------------------
# The training signal
# --------------------------------------------------------------------------


def residual_ranks(frame: pd.DataFrame) -> np.ndarray:
    """Percentile rank of ``y`` after the surprise is projected out.

    Koijen & Levy's two-stage construction, and the right target for the same
    reason the contest is: the benchmark already owns whatever the surprise
    explains, so a rulebook trained on raw ``y`` would spend its capacity
    re-deriving the surprise. Ranked so the Reflector sees a bounded, readable
    number rather than a raw residual.
    """
    import harness

    resid = harness.residualize(frame, "y")
    return resid.rank(pct=True).to_numpy(dtype=float)


def bootstrap_band(pred: np.ndarray, y: np.ndarray, surprise: np.ndarray,
                   n_boot: int = 1000, seed: int = 0) -> float:
    """One-sided noise band on ΔR² for the validation gate."""
    import eval as E

    rng = np.random.default_rng(seed)
    ok = np.isfinite(pred) & np.isfinite(y) & np.isfinite(surprise)
    p, t, s = pred[ok], y[ok], surprise[ok]
    n = len(t)
    if n < 100:
        return float("inf")
    draws = [
        E._delta_r2_fast(p[i], s[i], t[i])
        for i in (rng.integers(0, n, n) for _ in range(n_boot))
    ]
    return float(np.std(draws, ddof=1))


if __name__ == "__main__":
    import harness

    pb = Playbook()
    pb.add("surprise_quality", "Beat driven by a one-off tax or asset-sale item, with flat "
                               "operating income -> 0.25-0.40")
    pb.add("guidance_and_outlook", "Beat on results BUT full-year guidance trimmed -> 0.15-0.30")
    pb.add("interactions", "Record margin attributed to non-recurring benefit AND a major "
                           "segment down >15% -> 0.10-0.20 regardless of positive demand talk")
    print(pb.render())
    print("stats:", json.dumps(pb.stats(), indent=2))

    frame = harness.load("2026Q2")
    ranks = residual_ranks(frame)
    print(f"\nresidual ranks on 2026Q2: n={len(ranks)} "
          f"min={ranks.min():.3f} median={np.median(ranks):.3f} max={ranks.max():.3f}")
    # Uniform by construction, and centred — if this drifts, the target is wrong.
    assert abs(np.median(ranks) - 0.5) < 0.02, np.median(ranks)

    import eval as E
    sur = frame.surprise_pct.to_numpy(float)
    y = frame.y.to_numpy(float)
    print(f"partial corr of the residual rank with y (should be high): "
          f"{E.partial_corr(ranks, y, sur):+.3f}")
    print(f"partial corr of the SURPRISE with the residual rank (should be ~0): "
          f"{E.partial_corr(sur, ranks, sur):+.3f}")
    print(f"\nnoise band on dR2 at n={len(frame)}: "
          f"{bootstrap_band(np.random.default_rng(0).random(len(frame)), y, sur, 200):.4f}")
