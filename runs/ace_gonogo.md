# ACE build — go/no-go, and the kill conditions, written before the training run

Registered 2026-08-18, **before any ACE training call is made.** A criterion
chosen after seeing the number is not a criterion.

---

## Verdict: GO, at reduced scope, with three pre-registered kills

Approved scope is items 2–6 of the flash-lite path (~$5). Opus-at-inference
(item 7, $23) stays declined until item 5 produces decision-grade evidence.

---

## Expected gain, as an interval

**Central estimate +0.5pp of obtainable. Interval −0.5pp to +2.0pp. Probability
the gain exceeds +0.5pp on held-out data: roughly 35%.**

Stated as an interval because every input to it is itself an interval, and
because the point estimate is the least reliable part.

**What pushes it up:**

- `aigent-arm-b` vs `aigent-arm-a` — a live, published, same-model A/B isolating
  a *curated rulebook* at **+0.0200 ΔR² = +2.1pp of obtainable**, sign-stable
  across both leaderboards (n=272 contest, n=273 global). This is the single best
  evidence that a frozen rulebook in the prompt is worth anything, and it is
  independent of the paper. **measured**
- The paper's ACE result, +3.0pp of combined R² = **+3.2pp of obtainable**.
  **measured, but see below**
- The mechanism is cheap to apply and already deployed-shaped: a text block in
  the prompt, no new data source, no latency risk (flash-lite p95 3.1s).

**What pushes it down, and there is more of it:**

- **The paper's ACE effect is not separated from zero.** Table 7: no-ACE
  14.1% [0.09, 0.20], with-ACE 17.1% [0.12, 0.24], 99% bootstrap CIs. Each point
  estimate lies inside the other's interval; overlap width 0.08. That is
  *identical* to GEPA's overlap ([0.06,0.16] vs [0.08,0.19]), which the paper
  explicitly caveats — and ACE's, which it does not. No CI on the difference is
  reported anywhere. **measured**
- **We have already tested a hand-written rulebook on flash-lite and it bought
  nothing.** `ctx.rulebook.gemini-2.5-flash-lite`: ΔR² +0.0313, vs_champion
  **−0.0011**, marginal +0.0065, n=1,999. ACE's claim is that a *learned*
  rulebook beats a hand-written one; that is exactly what is untested. **measured**
- **Context × model tier is unresolved at z = 0.53**, and flash-lite is the weak
  end of the two tiers tested. Every Table 8 transfer number sits *below*
  un-ACE'd Opus, so a rulebook on a weak model is a cost play, not a performance
  play. **measured**
- **LLMs assert patterns in structureless data 72–100% of the time**
  (arXiv:2510.09709; GPT-4.1 acknowledged randomness in 5.0% of random sequences,
  o3 in 28.0%). At ρ≈0.25 a 50-event residual batch is mostly noise, and the
  Reflector will return a confident, articulate, plausible rule from it every
  single time. **measured, and it is the central risk**
- **Rule induction under noise damages consistency while leaving accuracy flat**
  (arXiv:2502.16169) — so the failure will not show up in the headline metric.
- **Nobody has published ACE, or anything like it, on a low-SNR continuous
  target.** Koijen & Levy report the result and **not one hyperparameter**.
  **measured**

**Derivation of the interval.** Floor: our own hand-written rulebook on
flash-lite measured −0.001 vs champion, so a learned one doing slightly worse
than nothing is inside the evidence — hence −0.5pp. Ceiling: the strongest
same-model rulebook effect anyone has measured is +2.1pp (`aigent-arm-b`), on a
frontier model, which flash-lite is not — hence +2.0pp as a generous cap.
Centre: the paper's +3.2pp discounted for an unseparated CI, for a weaker model,
and for a task the method has never been validated on.

---

## Kill conditions — all three registered in advance

### K1 — the null-batch control. Runs BEFORE the training run. Cost ≈ $0.

Feed the Reflector a batch of 50 events whose residuals have been **randomly
shuffled**, so any relationship between the facts and the ranking is destroyed by
construction. Run it three times, alongside three real batches.

**KILL if:** the Reflector proposes confident, specific rules on shuffled batches
at a rate indistinguishable from real batches — i.e. it cannot tell signal from
noise, and every downstream rule is a coin flip dressed as a finding.

This is the cheapest and most decisive test in the whole build, and it exists
because of the Idola Tribus result. It is also the one the source material never
runs.

**Mitigation to apply regardless:** the Reflector prompt must offer *"these two
groups are not distinguishable; propose no rule"* as a prominent, first-class
outcome. Explicitly permitting the null raised randomness acknowledgment from
12.8% to 75.8% in that paper. ACE's own prompt does the opposite — it demands
`error_identification` and `root_cause_analysis` as required JSON fields, so the
model is structurally forbidden from returning "nothing here."

### K2 — the admission rate. Runs during training. Cost: already budgeted.

**KILL if:** fewer than 20% of Curator-proposed rules survive the validation
gate, sustained over the run. A proposer whose output is rejected four times in
five is producing noise, and the gate is doing all the work — at which point the
rulebook is a filter artefact rather than learned structure.

### K3 — the decision-grade gate. Runs after training, at item 5. Cost: $1.80.

**KILL if:** the rulebook's ΔR² on the 4,144-event confirmation partition does
not exceed the no-rulebook flash-lite baseline by more than **2 × the paired
bootstrap standard error**.

Paired, because both arms run the same model on the same events, so the
comparison is a difference and the noise is far smaller than either level's.
Confirmation partition, because this loop has already measured six of seven
archive arms going *negative* on held-out data with a mean selection bias of
+0.0052.

**If K3 fails, the ACE branch is closed and Opus-at-inference is not revisited.**
The overlap in Table 7 will have been telling us something.

---

## What gets built, in ablation order

Build order follows ACE's own Table 3, not its narrative:

| component | measured worth | build |
|---|---|---|
| incremental ADD-only deltas | **+13.4 avg** (Table 18) | **first — this is the method** |
| Generator + Curator | +12.7 of +17.0 (**74.7%**) | **first** |
| multi-epoch | +2.6 | second |
| Reflector | **+1.7**, inside the paper's own ~3.6pp run-to-run noise | **last, and treat as possibly null** |

Everything else in ACE's §3 — retrieval, pruning, non-LLM merge, counter-driven
refinement — is unimplemented, off by default, or has no reader in the released
code. Do not build to the paper's prose.

**Reflector and Curator run in-session on the Claude Code plan (unmetered).**
Generator stays on flash-tier via OpenRouter. Two cautions carried from the
brief: reflection stays batched with the gate intact, since a stronger reflector
proposes *more* confident rules and is therefore more dangerous under a loose
filter, not less; and any rule naming a specific ticker or date is a look-ahead
signal, not a rule — pattern-level only.

## What we are inventing, stated plainly

Per Block 3, and none of it is in ACE:

1. Batch size chosen for **statistical power** (n≈50) rather than cost (the field
   uses 1, 3, 4, 35, 64).
2. Reflection over **residual structure** — a rank-sorted contrastive split —
   rather than over individually-labelled failures.
3. **Per-rule** admission gating inside a context-optimisation loop. GEPA and
   MIPROv2 validate whole contexts; ACE tracks per-rule counters and ignores
   them.
4. A feedback format showing **only ordering, never a numeric error**, because
   the metric is affine-invariant and an absolute error is a quantity the scorer
   discards.

The closest precedent for (1)+(2)+(3) together is **D5** (Zhong et al., NeurIPS
2023): a contrastive proposer over 25 samples per side, then held-out validation
by one-sided t-test at p<0.001 with Benjamini–Hochberg at 10% FDR. Its ablation
is the reason K2 and K3 exist — **the unvalidated proposer scored 4–12%.** The
validator is not a refinement there; it is what makes the system work.
