# BUILD: the research loop

## Session setup

This runs in a **separate worktree**, not the same directory as the production
session:

```bash
cd ~/Documents/explaining-markets
git worktree add ../em-loop -b feature/research-loop
cd ../em-loop
```

Two Claude Code sessions in one working tree will fight over the git index and
the same files. A branch alone is not enough — you need a separate tree.

Nothing here touches `agent/` or deploys. Production is running v9 and stays
untouched.

---

## Why this exists

**Eight days in (Aug 9 → Aug 17). One family tested, thoroughly. Five untouched.**

What's been closed — LLM reads of the ten facts, on four independent axes:

| axis | result |
|---|---|
| prompt content | +1.66pp, replicated across two vendors |
| scaffold (CoT) | +0.0011, CI excludes a meaningful effect |
| model tier | no detectable effect, cheapest model came 2nd of 6 |
| pretraining lineage | ρ_b 0.824 across 4 vendors, Chinese + Western corpora |

Family asymptote ρ/√ρ_b = 0.199/0.908 = 0.219 → **4.5% of obtainable.**
We are at 4.7%. That family is exhausted.

What has never been run: external data (zero fetches, ever), peer/sector
channel, embeddings, extraction, fitted layers, many-shot above N=20.

**The bottleneck is not ideas, budget, or model quality. It is throughput.**
Everything has run serially with a human between each step — one experiment per
session. Ten context arms ran in a single session and produced a full comparison
matrix, which is the proof that parallel works.

---

## The architecture — read these first

The loop you are building is a known architecture. Do not invent it.

**FunSearch → AlphaEvolve.** Core loop: propose → evaluate → select → evolve.
An LLM generates candidates, an automated evaluator scores them, high scorers
are retained and fed back. AlphaEvolve extends this to whole codebases and
multiple simultaneous objectives.

- Paper: https://arxiv.org/abs/2506.13131
- **OpenEvolve** (open-source implementation, and it ships a `CLAUDE.md`):
  https://github.com/algorithmicsuperintelligence/openevolve
- Its `CLAUDE.md`: https://github.com/algorithmicsuperintelligence/openevolve/blob/main/CLAUDE.md
- OpenAlpha_Evolve (simpler, modular agent structure, LiteLLM-based):
  https://github.com/shyamsaktawat/OpenAlpha_Evolve
- CodeEvolve (island-based GA, inspiration-based crossover): https://arxiv.org/html/2510.14150v1
- ShinkaEvolve (Sakana) — bandit-based LLM ensemble + novelty-based rejection
  filtering, aimed specifically at **sample efficiency**, which matters here
  because our evaluations are noisy and metered
- MadEvolve — the same architecture applied to **trading systems**:
  https://arxiv.org/pdf/2605.23007

OpenEvolve's design combines LLM-generated mutations with island-based
evolutionary search and a **MAP-Elites-style quality-diversity archive**. That
QD archive is the piece that matters most for us — see adaptation 3.

**The load-bearing property**, stated across this literature: *the loop's
correctness rests on the evaluator, not on the generator.* Our evaluator is
`lab/eval.py` and it is validated — null scores exactly 0, known baselines
reproduce, fast ΔR² agrees with the scorer to 1e-9, perfect foresight returns
1−R²_bench to six decimals, offline 4.2% predicted live 4.5%. **That precondition
is met.**

**Successive Halving / ASHA** — the budget allocation. Evaluate all candidates at
a small budget, discard the worst fraction, multiply the budget for survivors,
repeat. Parallelises naturally and is robust to noisy evaluation.

- Hyperband paper: https://arxiv.org/abs/1603.06560
- ASHA (asynchronous): https://arxiv.org/abs/1810.05934
- CMU write-up: https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/

**First decision for you:** read OpenEvolve's architecture and decide whether to
adopt it or build a lighter runner informed by its design. Our "programs" are
arm configurations rather than arbitrary code, and our evaluator already exists —
so a custom runner borrowing the MAP-Elites archive, island structure and async
controller may fit better than wholesale adoption. Report the call and the
reasoning.

---

## THREE ADAPTATIONS OUR PROBLEM NEEDS

These are where a naive port fails.

### 1. The fitness function is NOT ΔR²

This is the single most important instruction in this document.

An arm's value is how much it raises the **ensemble ceiling** ρ/√ρ_b, not its
standalone score.

Why: selecting on ΔR² picks the strongest readers, and the strongest readers are
the ones most correlated with each other. We have measured this — six models,
four vendors, two continents of pretraining data, every one a defensible pick on
ΔR², collectively worth one arm at ρ_b 0.824.

The arithmetic, at our current ρ = 0.199:

| ρ | ρ_b | ceiling |
|---|---|---|
| 0.199 | 0.824 | 4.5% |
| 0.250 | 0.824 | 7.1% |
| 0.199 | 0.400 | 9.9% |
| **0.199** | **0.200** | **19.9%** |

ρ has room to rise maybe 1.25×. ρ_b has room to fall 4× — TF-IDF already
measured 0.193. **Selection must reward decorrelation.**

Suggested fitness: marginal gain in the GLS-weighted ensemble's ΔR² when the arm
is added to the current archive. Report standalone ΔR², ρ, ρ_b vs champion, and
this marginal contribution — separately, always.

**Necessary but not sufficient:** champion + pure noise measures ρ_b 0.696 — as
decorrelated as a different vendor — and is worth −0.019. An arm needs low ρ_b
*and* real ρ. The marginal-contribution fitness handles this automatically; a raw
ρ_b objective would not.

### 2. Our fitness is noisy; AlphaEvolve's usually is not

Matrix multiplication is verified deterministically. Ours is not.

**Measured**: to distinguish a ρ=0.25 candidate from a ρ=0.15 one, the gap is
0.074 and the noise is:

| n | noise | usable |
|---|---|---|
| 35 | 0.231 | no |
| 250 | 0.103 | no |
| 1,000 | 0.046 | no |
| 2,000 | 0.033 | **yes** |

So: **successive halving on n.** Rung 1 = 300 (direction only), rung 2 = 1,000,
rung 3 = 2,000+ (selection). Never make a promotion decision below 2,000.

And **K compounds across generations.** Ten generations × twenty candidates is
K=200, and the floor rises with it. This is a ledger, not a brake — log every arm
including failures, compute the floor at the true K, report what clears it. The
failure mode was never testing too much; it was reporting the best of many as if
it were the only one.

### 3. Kill on mechanism, not on score

Killing the LLM-read family was legitimate: ρ_b 0.824 is a *structural*
measurement that caps the entire branch regardless of how many more members you
add.

Killing an arm because it scored 0.002 lower is noise-chasing.

Branch elimination requires a stated reason. Log it. Variant elimination inside a
rung is fine — that is what successive halving is for — but eliminating a whole
*family* needs a mechanism.

This is also why the archive should be **quality-diversity** (MAP-Elites style)
rather than a single best: keep an archive of arms that are *different from each
other*, with ρ_b as a behaviour dimension.

---

## ARMS MUST BE DEPLOYABLE ARTIFACTS

A research loop that does not connect back to `predict.py` produces numbers that
may not transfer. We have already paid for this once: the offline champion was a
proxy (`gpt-5-nano` + old prompt) while production ran `gpt-5.4-nano` + a
different prompt, and every ρ_b and floor was measured against the wrong thing.

Four requirements:

1. **One shared prediction path.** The function an arm defines is the same object
   the live worker calls — not a reimplementation. `champion.py` already does this
   correctly by importing from `agent/` and calling the same `_ask_llm`; make it
   the general pattern.
2. **Live-computability declared per feature.** An arm using a feature requiring
   30 peer price fetches under a 5-minute deadline should fail its own check
   before anyone measures its ΔR².
3. **Promotion is a config change, not a rewrite.** Deploy by pointing production
   at an arm's config.
4. **A live/offline consistency test.** Take a recent real event, run both paths,
   assert identical output. Same role the frozen HMAC vectors play for the
   webhook.

---

## WHAT TO BUILD

```
runner/
  registry.py    arms register themselves; each declares features,
                 model, cost estimate, live-computable, cutoff-safe
  features.py    shared feature cache keyed by (event_id, feature_name).
                 Arms sharing a feature compute it once.
  schedule.py    ASHA: rungs at n=300 / 1000 / 2000, async promotion,
                 resumable, parallel across arms
  archive.py     MAP-Elites-style QD archive. Behaviour dimensions:
                 rho_b vs champion, feature family, cost tier.
                 Keeps diverse arms, not just the best.
  fitness.py     marginal ensemble contribution (GLS-weighted), NOT raw dR2
  report.py      full matrix: dR2, rho, rho_b vs champion, inter-arm rho_b
                 matrix, marginal contribution, n, cost, K, floor, ship
  loop.py        one command: run the queue, report, re-rank
```

Reuse `lab/eval.py` as the evaluator. Do not reimplement the scorer.

---

## THE QUEUE — unranked, incomplete, add to it

**Non-fact channels (the point of this exercise):**
- peer/sector channel — archive-computable, directional, structurally outside the
  read family. Filter `peer.window_end_date ≤ our.knowledge_cutoff` (not
  "reported first" — that filter is right 99.3% of the time, which is why it
  survives review). Aggregate `car1` standardised by the peer's own historical
  dispersion; **never aggregate `y`**, which is a full-quarter rank that does not
  exist at prediction time. Carry `n_peers`.
- embeddings of the facts — TF-IDF measured ρ_b 0.193, the only decorrelation
  ever observed here. Representation-not-read is the live axis.
- extraction — structured fields, written weeks ago, never executed. Extractor
  must never see "percentile" or "return". Audit on 100 events first; cut any
  field `not_stated` >80%.
- many-shot at N ∈ {50, 200, 500} — the literature's claim is that pretrained
  LLMs perform regression in-context rivalling supervised methods at ~500
  examples. We tested 5 and 20, far below where the effect is claimed, and filed
  it as dead. Example order matters.
- retrieved nearest-neighbour examples with outcomes — kNN through the prompt
- analyst revisions, earnings-cycle position, run-up, option skew
- external-data survey: what is actually ingestible, free, point-in-time, and
  cutoff-safe. EDGAR 8-K/10-Q are public, structured, timestamped, and often
  released at the announcement — different material from the ten-fact summary.

**Fitted layers — do not inherit the leaders' choice:**
Both top submissions fit nothing. That is their design decision, not a
measurement. Run fitted arms: stacking, ridge/GLS weights, gradient boosting,
LambdaRankIC. Note NDCG-based ranking objectives are top-heavy and wrong for a
symmetric full-rank target.

**Mechanical diversity generators — zero K cost, because they combine rather than
select:**
- residual targeting: train arm N on the champion's residual → orthogonal by
  construction
- random subspaces: models on random ~70% feature subsets
- algorithmic diversity: tree / linear / kernel on the same features; keep all
  as arms rather than picking a sweep winner
- negative correlation learning: explicit penalty on correlation with the archive

**Already shipped or measured, do not re-litigate:** prompt content (+1.66pp),
`base@flash` (+1.14pp, 3/3 signs, CI excludes zero).

---

## HARD CONSTRAINTS

Measured facts, not preferences.

1. **n ≥ 2,000 for any promotion decision.** Rungs below that are direction only.
2. **Report ρ_b vs champion for every arm**, plus the inter-arm matrix.
3. **Surrogate check** before optimising against any proxy metric: score a
   constant *and a weak-but-real predictor* (ρ ≈ 0.15–0.25). If the constant
   wins, the surrogate is inverted. An oracle-vs-constant check does not catch
   this — it already failed to.
4. **Verify the base component** before building measurement on it. Four failures
   of this shape: champion proxy, unoptimised prompt, optimiser surrogate,
   unread reference implementation.
5. **2026Q3 sealed.** Enforced in code.
6. **Compliance** — nothing postdating an event's `knowledge_cutoff`. Measured
   across 840 events: always 20:00Z, always *earlier* than `event_datetime`,
   median 17h, max 96h. **No function of `event_datetime` is safe** — a
   "conservative" bound of announcement−1h would have violated §04 on 85% of
   events. Fetch the cutoff from `GET /v1/events`; refuse on lookup failure.
   Everything through `sources.py` with its audit log. Archive-derived context
   follows the leaders' pattern: snapshot frozen pre-period, embedded, no fetch
   at prediction time.
7. **Never sit out.** Coverage scales effective correlation by √q — and a neutral
   0.5 is sitting out in disguise. Track neutral-rate per arm.
8. **Log K.** Floors per candidate at se × best-of-K, se measured by paired
   bootstrap (paired σ is 0.0008–0.0047, not the 0.010 once assumed).

## Spend

Metered spend is **OpenRouter/OpenAI only** — the calls that generate
predictions. Claude Code is on a Max plan, so building the runner, reading,
analysis and orchestration are unmetered. **Build freely; economise only on
scoring runs.**

$11.81 of $25 already spent. **Hold at $25 cumulative and ask before exceeding.**
Prefer flash-lite — cheapest and joint-best of the six models swept, and cost is
a multiplier on every future run. Screen at rung 1 before paying for resolution.

---

## Reporting

Per arm: ΔR², % of obtainable (divide by 0.9413), ρ, ρ_b vs champion, marginal
ensemble contribution, neutral-rate, n, cost, live-computable, cutoff-compliant,
and the K it was judged at.

Label every claim **measured** (number + source), **derived** (show the step), or
**assumed** (say what checking would cost).

Negative results with equal prominence. State how many configurations were tried
to reach any headline number.

When a result surprises you, check for **attribution error** before building on
it. The "+0.0150 model lever" drove a sweep, a cost analysis and a procurement,
and was the prompt effect misattributed — identical +1.66pp across two vendors is
a prompt signature, not a model signature.

Never treat a leaderboard number as a target without checking its n. The current
target is a crowded 11–15% band at n≥1500, not the 22–26% scores that all live at
n≈273.

---

## Autonomy

Run without asking. Escalate only for: production deploys, spending the sealed
holdout, metered spend above $25, and compliance questions the rules do not
clearly answer.

**Write findings to the notes as they land, not at session end.** A result that
exists only in a chat transcript will be re-derived later. That has already cost
sessions.

If you find yourself arguing something is not worth testing — test it. That
argument has been wrong every time it has been made in this project, by anyone.