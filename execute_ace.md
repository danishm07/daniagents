# EXECUTE — ACE build, gated

Credits reloaded, key cap raised. **Hard cap $20 total for the project.**
Currently ~$9.55 spent, so ~$10 of headroom. Approved scope is ~$5. Opus at
inference stays closed and does not get revisited until item 4 reports.

---

## FIRST — free, and it protects the whole cycle

**Commit everything.** `MAP.md` reports ~3,000 lines untracked across `runner/`,
six `sources_*.py`, `ace*.py`, and every data block. A `git clean` would delete
the entire cycle. This is the highest-risk item on the board and it costs nothing.
Do it before any run.

---

## GATE 0 — the shuffled-residual test. Run before spending anything.

You proposed this and it is the right first move. It is also the test none of the
source material runs, which is itself notable: LLMs assert patterns in
structureless data 72–100% of the time, and ACE was validated on tasks with
verifiable answers where that failure mode cannot surface.

Feed the Reflector 50 events with **shuffled residuals**. Then, so the result is
interpretable rather than a vibe:

- Run it on shuffled residuals **and** on real residuals, same 50 events, same
  prompt, same temperature.
- Report **rule count**, and some measure of stated confidence, for each.
- **Pre-register the threshold now:** if shuffled-rule-rate is within what
  fraction of real-rule-rate do we stop? Write the number before you look.
- Repeat the shuffle 3–5 times with different seeds. One shuffle is one draw.

**If it fails, the build stops and that is a real finding** — worth writing up
regardless of the competition, since it would be a measured demonstration that
reflective rule induction cannot survive a noisy continuous target.

---

## THEN, in order

| # | item | metered calls | cost |
|---|---|---|---|
| 1 | **H1** — regenerate **both** arms on OpenRouter | 1,400 | $0.30 |
| 2 | **ACE training** — Generator on flash-lite, 1,000 events; Reflector + Curator in-session | 1,000 | $0.90 |
| 3 | **Validation gates** — 5 × 400 | 2,000 | $1.80 |
| 4 | **Decision-grade scoring**, n=2,000 held out | 2,000 | $1.80 |
| 5 | **Deploy** if it clears the pre-registered bar | 300 | $0.27 |

**On H1:** yes, regenerate both arms on OpenRouter. A confounded test is not worth
$0.15 saved, and the existing `official__gpt5nano__2026Q2.jsonl` was generated
against OpenAI direct. Nothing touches the OpenAI key.

**On item 2:** Generator on flash-lite via OpenRouter; Reflector and Curator run
in-session on the Max plan, unmetered. Dump the error batches to disk as JSON so
reflection is inspectable and rerunnable rather than a transient API call. That is
the paper's own "strong model trains, cheap model runs" recipe with the strong
model being free.

---

## What must hold in the build

Carried from the four adaptations, since these are what make ACE survive a noisy
continuous target:

1. **Batched reflection only.** At ρ ≈ 0.25 a single event's error is ~94% noise.
   The Reflector never sees one event.
2. **Residual target.** Verified: residual rank correlates +0.997 with the
   outcome and +0.000 with the surprise. The rulebook cannot spend capacity
   re-deriving the benchmark.
3. **Anti-level directive in both prompts.** ΔR² is affine-invariant, so a rule
   that shifts every prediction equally is worth exactly zero. This is the
   specific trap GEPA fell into — de-biasing is the cheapest win under MSE and
   the paper's own numbers show it bought nothing that survives.
4. **Validation gate per block.** Accept only if it beats a bootstrap noise band
   on a held-out slice; otherwise revert.

**Look-ahead watch.** The reflecting model may have training exposure to these
outcomes. Pattern-level rules only — a rule naming specific tickers, dates, or
events is a leak signal, not a finding. Scan the rulebook for proper nouns before
it ships.

**Build ADD-only.** Confirmed at `playbook_utils.py:127–162`; UPDATE/MERGE are
commented out, DELETE absent, and the maintainers' own issues #26/#29 list them as
future work. Do not build to the paper's prose. And since Generator+Curator is
74.7% of the gain while the Reflector's +1.7 sits inside the paper's own
run-to-run noise band of ~3.6pp, treat the Reflector as an upper bound on a
possibly-null effect — measure with and without it if that is cheap.

---

## One verification I want, and it is free

I reconstructed SLE by hand and the result is worth confirming against real data.

Benchmark slope β = √0.068 = **0.261**, intercept **0.370**. At
`surprise_pct ≈ 0.97` (a +23.51% beat is near the top of the distribution) the
benchmark predicts **0.623**.

We predicted **0.62**.

If that holds, our single worst miss — realised `y` = 0.01, error −0.61 — was us
reproducing the benchmark's own answer to three decimals, contributing
`e_y × e_p ≈ 0` to ΔR². Pull SLE's actual `surprise_pct` and confirm or correct
the arithmetic.

The reason it matters for the rulebook: the contribution of alternative
predictions on that event goes 0.62 → +0.002, 0.50 → +0.075, 0.20 → +0.259.
**Saying a flat 0.50 would have earned 40× more; 0.20 earns 130×.** Not because
0.20 is nearer the truth, but because it *disagrees with the benchmark.*

That is the objective in one sentence: **not "be more accurate" but "disagree with
the benchmark on the events where the benchmark is wrong."** Both prompts should
say so explicitly.

If SLE turns out representative — a cluster of near-0.62 predictions on large
beats that dropped — report the distribution: how many of our events sit within
±0.02 of the benchmark's own fitted value? That number is a direct measure of how
much of our output is redundant, and it would be the cleanest possible statement
of what the rulebook needs to fix.

---

## Reporting

At item 4, report against the pre-registered bar in `runs/ace_gonogo.md`:

- ΔR², % of obtainable, ρ, ρ_b vs champion, on the held-out partition
- selection-set number alongside it, so the selection bias is visible
- neutral-rate and coverage
- the rulebook itself, so I can read the rules
- spend to date against the $20 cap
- K, and the floor at that K

Label every claim **measured / derived / assumed**. Report the negative case with
equal prominence — a rulebook that fails its bar is a result, and given the
paper's ACE interval overlaps its own control by 0.08 with no CI on the
difference reported anywhere, failure is a live outcome rather than an unlikely
one.

Your own go/no-go put the expected gain at −0.5pp to +2.0pp with centre +0.5pp
and ~35% chance of clearing +0.5pp. Hold to that. If the result lands inside the
noise band, we say so and stop.