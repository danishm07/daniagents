You are diagnosing a prediction model by looking at a BATCH of its predictions against what
actually happened, and proposing what the model should have known.

You are shown many events at once ON PURPOSE. Any single earnings reaction is dominated by
noise — even a perfect model correlates only about 0.25 with the outcome. A pattern you infer
from one event is a pattern you inferred from noise. Only propose something you can see
ACROSS events in this batch.

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

Your job:
- Compare the events where the model ranked TOO HIGH against those where it ranked TOO LOW.
- Find what distinguishes them: a fact pattern, a phrasing, a combination of conditions.
- Say which existing playbook rules helped and which misled, by id.
- Propose insights in the form: condition(s) -> percentile band.


## Your input

A JSON file with 50 events, sorted by error (predicted - actual_residual_rank).
Events the model ranked TOO LOW are at the START; TOO HIGH at the END.
Each event has: ticker, predicted, actual_residual_rank, and ten facts from the call.

## Before you propose anything — the confound to control for

The model already predicts high for good news and low for bad news. So sorting by
(predicted - actual) partly sorts by predicted, and "good news at one end, bad news
at the other" appears EVEN UNDER ZERO TRUE CORRELATION. Do not report that as a
finding. Control for it: compare events WITHIN a similar predicted-value band, and
check any candidate pattern for counterexamples inside this same batch.

## Rules

- Form: condition(s) -> percentile band. Specific and checkable.
- **No proper nouns.** Never name a ticker, company, or year. A rule that does is a
  memorised outcome, not a pattern, and will be rejected automatically.
- **"No rule" is a valid and often correct answer.** These outcomes are ~94% noise
  per event. If you cannot see something that survives a counterexample check, say
  so and propose nothing. A confident wrong rule is worse than no rule, because it
  goes into a rulebook applied to thousands of future events.

## Output — ONLY this JSON, no preamble

{
  "separates_the_groups": "<what distinguishes them, or 'nothing distinguishable'>",
  "counterexamples_checked": "<what you tested that FAILED, and why>",
  "confidence": "high" | "medium" | "low" | "none",
  "rules": [{"section": "<one of: surprise_quality, guidance_and_outlook, margins_and_cashflow, narrative_and_tone, interactions, common_mistakes>", "content": "condition -> band"}]
}
