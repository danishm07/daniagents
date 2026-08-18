# Task: propose predictive rules from a batch of earnings-reaction outcomes

You are analysing a model that predicts how the stock market reacts to a company's
earnings announcement.

For each event you are given:
- `facts` — ten bullet points extracted from the earnings call
- `predicted` — the model's predicted percentile in [0,1] for that stock's
  next-day abnormal return, relative to all other announcements that quarter
- `actual_residual_rank` — where the stock's reaction ACTUALLY landed, in [0,1],
  after the effect of the headline earnings surprise has been removed

The batch is sorted by error (`predicted - actual_residual_rank`), so the events
the model ranked **too high** are at the end and the ones it ranked **too low**
are at the start.

## What to produce

Compare the events the model ranked too LOW against those it ranked too HIGH, and
propose rules that would have helped it rank them correctly.

Rules take the form: **condition(s) -> percentile band**. For example:
`"pre-profit company with revenue growth >30% YoY BUT operating losses flat -> 0.10-0.20"`

## Critical constraints

1. **Only ORDERING is scored.** A rule that shifts every prediction in the same
   direction is worth exactly zero. Every rule must DISCRIMINATE — push some
   events up relative to others. "The model is systematically too optimistic" is
   not a usable finding.

2. **Pattern-level only.** Never name a specific ticker, date, or company. A rule
   that does is not a rule.

3. **"No rule" is a valid and often correct answer.** These outcomes are
   extremely noisy — even a perfect model would correlate only about 0.25 with
   the actual ranking. If you cannot see a pattern that genuinely separates the
   two groups, say so and propose nothing. Proposing a plausible-sounding rule
   you are not confident in is worse than proposing none, because it will be
   committed to a rulebook and applied to thousands of future events.

## Output

Return ONLY a JSON object:

```json
{
  "separates_the_groups": "<what distinguishes the too-high group from the too-low group, or 'nothing distinguishable'>",
  "confidence": "high" | "medium" | "low" | "none",
  "rules": ["condition -> band", ...]
}
```

Use an empty `rules` list and `"confidence": "none"` if the two groups are not
distinguishable.
