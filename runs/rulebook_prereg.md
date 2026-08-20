# Third-place rulebook on Haiku 4.5 — pre-registered before the run

Registered 2026-08-19, before any call. Budget $8.42 spendable.

## Design
Two arms, identical 2,000 events from the 4,144-event confirmation partition,
identical model (`anthropic/claude-haiku-4.5`), identical deployed prompt.
  arm A: deployed prompt, no rulebook
  arm B: Rulebook.md (1,838 tokens) prepended to the same prompt
The rulebook is hand-written from published research, frozen before the period,
never fitted to our residuals — so there is no leakage surface and nothing to
validate offline first.

## The bar — same as ACE
gain must exceed 2 x sqrt(bootstrap_se^2 + replicate_se(n)^2).
replicate_se(2000) = 0.00417, so the bar is approximately +0.0081 to +0.0100.

## Target
Paper Table 8: Haiku 4.5 with the ACE rulebook reaches 11.7% combined R2, which
is dR2 0.067 = 7.1% of obtainable against a 5.0% surprise-only benchmark.
Our current live own-sample is 5.7%. Third place's own arm comparison measures a
curated rulebook at +0.0200 at fixed model.

## ⚠️ Caveat stated in advance: a null will be AMBIGUOUS
Measured in the price check: both arms emit ~14 output tokens, i.e. a bare float
with no reasoning. The rulebook has 18 numbered priors and must be applied
entirely inside the forward pass with no scratchpad.

So a null does NOT distinguish:
  (a) the rulebook carries no signal, from
  (b) the model cannot apply 18 rules without room to reason.

We are NOT fixing this. Reasoning tokens would blow the budget (Opus with
reasoning measured $0.0298/event, 685 output tokens), and the CoT scaffold was
already measured at +0.0011 with a CI excluding a meaningful effect, so the
cheap version of (b) has been tested and found empty.

A positive result is unambiguous. A null is one-sided evidence.

## Decision rule
Clears the bar -> deploy Haiku with the rulebook (prompt change in predict.py,
prepend the rulebook, redeploy in a quiet window; nothing structural).
Fails -> stop and write up.
