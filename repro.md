# THE REPRODUCIBILITY GAP — investigate, then build the recovery framework

## Why this is the highest-value open item

| | ΔR² | % obtainable |
|---|---|---|
| Gemini Flash Lite — Summary, **live leaderboard** | 0.0917 | **9.7%** |
| its archive column claims | — | 4.17% |
| our replay of the published repo | — | **1.05–1.85%** |
| our live own-sample | 0.0534 | 5.9% |

Same public code. Same ten-fact input. Three different numbers.

The entire six-agent data cycle bought **+0.31pp**. Closing this is worth **~4pp**.
It is more than ten times larger than anything else on the board and it needs no
new data source.

Already eliminated: `temperature` (rejected by the API), `reasoning_effort=high`
(scored 0.91%, no better than default). No compiled artifact exists — `grep` for
`.load(`, `load_state`, `compiled`, `demos`, `BootstrapFewShot`, `MIPRO`,
`teleprompt` returns zero hits, and `_run_program` constructs
`dspy.ChainOfThought(PredictEarningsReturn)` fresh on every call.

---

## HYPOTHESIS 1 — DSPy adapter/version drift. **Test this first.**

There is a **documented DSPy regression that produces exactly this symptom.**

**stanfordnlp/dspy issue #6743** — *"ChainOfThought `reasoning` output field is
blank in the prompt."* Vanilla `dspy.ChainOfThought` emits **no description for
the reasoning field**, which the reporter measures as *degraded performance (no
prompting for "step-by-step")*. Passing an explicit `rationale_type` restores it.

Compare against **issue #409** (2024), which shows what older DSPy actually put in
the prompt:

```
Follow the following format.
Question: ${question}
Reasoning: Let's think step by step in order to ${produce the answer}. We ...
Answer: often between 1 and 5 words
```

So: **older DSPy emitted an explicit step-by-step instruction; newer DSPy emits a
blank reasoning field.** Identical source code, materially different prompt,
measurably worse output. That is precisely "same repo, three different numbers."

Two more version-dependent behaviours worth checking:

- **Adapters.** `ChatAdapter` / `JSONAdapter` / `TwoStepAdapter` determine how a
  signature becomes a prompt. Defaults have changed across releases.
- **`dspy.Reasoning`.** On reasoning-capable models,
  `Reasoning.adapt_to_native_lm_feature()` sets `reasoning_effort` in `lm_kwargs`
  and **removes the field from the signature entirely**, parsing reasoning out of
  the native response instead. Whether that path fires is model- *and*
  version-dependent.

### The diagnostic, and it is cheap

`lm.inspect_history()` prints the **actual prompt DSPy sent**. That is the ground
truth and nobody has looked at it.

1. Run the published baseline once under our current DSPy. Dump the exact system
   and user messages.
2. Read them. **Is there a step-by-step instruction attached to the reasoning
   field, or is it blank?**
3. Check the repo's lockfile / `pyproject.toml` for a pinned `dspy-ai` version.
   Check the archive's generation date. Install that version in an isolated venv
   and dump the prompt again.
4. Diff the two prompts. Any difference is the answer.
5. If the pinned version restores an explicit rationale, re-score at n≥2,000.

If a version pin recovers 4.17%, the whole gap closes in an afternoon.

**Fallback if version pinning is impractical:** pass an explicit `rationale_type`
to `ChainOfThought` reproducing the old wording, and re-score. That tests the
mechanism without the dependency archaeology.

---

## HYPOTHESIS 2 — model alias resolution

The baseline names `gemini/gemini-flash-lite-latest` — a **floating alias**. What
that resolved to when the archive was generated is not what it resolves to now.
The GPT column is pinned (`gpt-5-nano-2025-08-07`); the Gemini column is not.

Test: resolve `-latest` to each specific dated flash-lite release available on
OpenRouter and score each. If an older dated model recovers the number, the alias
is the answer and the fix is to pin.

Note this hypothesis predicts the **GPT column should reproduce** and the Gemini
column should not. **Check that first — it is free and it discriminates between
hypotheses 1 and 2 immediately.** If GPT also fails to reproduce, the alias is not
the cause and H1 gains a lot of prior.

---

## HYPOTHESIS 3 — something not in the repo

If 1 and 2 both fail, the archive column was produced by a pipeline that differs
from the published code. Candidates, in order of testability:

- **Different input.** Does the archive column's generation predate the ten-fact
  summarisation format? Check `lab/src/examples/summary.py` history — if the
  summariser changed, the baseline was reading different text.
- **A retry/aggregation step.** Multiple samples averaged, or invalid outputs
  retried rather than defaulted to 0.5.
- **Different percentile normalisation.** The repo's docstring says normalisation
  is verbatim; verify that claim against the archive column's actual value
  distribution. Ours had 62 distinct values; Gemini's archive column had a tie
  group of 836 events. **That is a large structural difference and it has never
  been explained.**

That last point is worth its own look regardless. A 836-event tie group is not
what a free-form float output produces. It suggests a quantisation step we do not
have.

---

## THE RECOVERY FRAMEWORK — how to keep going after each failure

This investigation will mostly produce nulls. Structure it so nulls are cheap and
informative rather than dead ends.

### Ladder, not a list

Each hypothesis has a **discriminating test** — one that changes what you believe
about the *other* hypotheses, not just its own. Run discriminating tests first.

```
H2-discriminator: does the PINNED GPT column reproduce?
    reproduces  → alias is the cause for Gemini. Pin and move on.
    fails too   → not the alias. H1 and H3 gain prior. Cheap, decisive.

H1-discriminator: dump the actual prompt. Blank reasoning field?
    blank       → documented regression, test the fix directly
    populated   → version drift is not the cause, H3 gains

H3-discriminator: value distribution of the archive column vs ours
    tie groups  → a quantisation step exists that we do not have
    matches     → the difference is upstream of output formatting
```

### Falsification budget

Each hypothesis gets a stated **kill condition** written *before* the test. If the
condition is met, it is closed and logged — not revisited on a hunch.

### When all three fail

Do not stop; widen the frame. Open questions that survive:

- Is the leaderboard's baseline entry running the *published* code at all, or an
  internal version? The organisers publish both a Summary and a Full Transcript
  baseline; the Full Transcript one demonstrably runs on material participants
  never receive. **Assume nothing about the Summary one either.**
- Is 9.7% a *different sample* from our 5.9%? The leaderboard exposes own-sample,
  imputed, and n per submission. Match them before treating the gap as real. This
  check has already dissolved one apparent gap this project (0.378 → 16%).
- Reverse-engineer rather than reproduce: what output *distribution* would produce
  ΔR² 0.0917 on the archive's events? Fit backwards from the target. That
  characterises the answer even without recovering the code.

### Logging

Every test: hypothesis, discriminating question, kill condition stated in advance,
result, and whether it was met. Label **measured / derived / assumed**. Count
toward K. A closed hypothesis is a permanent branch removal and is worth as much
as a hit.

---

## Cost

Approximately zero for H1 and H2 discriminators — prompt dumps and value-
distribution checks are free. Re-scoring at n≥2,000 on flash-lite is ~$0.60 per
full column. $13.19 remains of $25.

## Order

1. **Free:** dump the actual prompt via `inspect_history()`. Read it.
2. **Free:** check whether the pinned GPT column reproduces.
3. **Free:** compare value distributions — ours vs both archive columns.
4. **Free:** match leaderboard samples (own vs imputed vs n) before assuming 9.7%
   is comparable to our 5.9%.
5. Only then spend on re-scoring whichever hypothesis survives.

Steps 1–4 cost nothing and may settle it entirely.