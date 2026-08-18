"""Arms register themselves here, and declare what they cost and whether they ship.

An **arm** is a candidate prediction channel. Not a prompt, not a model, not a
feature — those are ingredients. An arm is the whole path from an event to a
float, and the reason it is the unit of the loop is stated in ``BUILD_LOOP.md``:

    A research loop that does not connect back to ``predict.py`` produces
    numbers that may not transfer. We have already paid for this once.

So an arm carries deployability metadata as *data*, checked before it is
measured rather than argued about after:

``live_computable``  can this run inside the 5-minute prediction window? An arm
                     needing 30 peer price fetches fails its own check before
                     anyone pays for its ΔR².
``cutoff_safe``      does everything it reads predate the event's
                     ``knowledge_cutoff``? Archive-derived context is safe by
                     construction; anything through ``sources.py`` is audited.
``cost_usd_per_event`` metered spend only — OpenRouter/OpenAI. Zero for arms
                     that are arithmetic on cached columns.

Two methods, deliberately split:

``ensure(events, quarter)``   does the metered work and caches it. Called by the
                              scheduler when an arm is promoted to a bigger rung,
                              so promotion pays only for the *new* events.
``predict(events, quarter)``  returns one float per event, in order, for free.
                              Must be deterministic given the cache — the
                              scheduler re-scores at every rung and a stochastic
                              ``predict`` would make the rungs incomparable.

Registration::

    @register(family="context", model=FLASH_LITE, cost_usd_per_event=1.4e-5,
              live_computable=True, cutoff_safe=True,
              rationale="third place supplies ticker history")
    class TickerCar1(Arm):
        ...

or, for the common case of "same prompt, different context block", via
:func:`context_arm`, which wires the caching, throttling and prompt path to the
exact objects production uses.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The 5-minute prediction window, less the headroom production already spends
#: on the LLM call itself (predict.py budgets 270s worst case). What is left for
#: feature computation is what an arm's declared live cost has to fit inside.
LIVE_BUDGET_SECONDS = 25.0

#: Cost tiers, in USD per event. Used as a MAP-Elites behaviour dimension, so
#: the archive cannot fill up with expensive arms that crowd out cheap ones of
#: similar value. Flash-lite lands in ``cheap`` at ~1.4e-5.
COST_TIERS = ((0.0, "free"), (5e-5, "cheap"), (5e-4, "mid"), (float("inf"), "expensive"))


def cost_tier(cost_usd_per_event: float) -> str:
    for bound, name in COST_TIERS:
        if cost_usd_per_event <= bound:
            return name
    return "expensive"


@dataclass
class LiveCheck:
    """Whether an arm could actually run in production, and why not if not."""

    ok: bool
    seconds: float
    fetches: int
    reasons: tuple[str, ...] = ()

    def __str__(self) -> str:
        verdict = "live-computable" if self.ok else "NOT live-computable"
        detail = f"{self.seconds:.1f}s, {self.fetches} fetches"
        return f"{verdict} ({detail})" + (f": {'; '.join(self.reasons)}" if self.reasons else "")


@dataclass
class Arm:
    """One candidate prediction channel, plus everything needed to judge it.

    Subclass and override :meth:`predict` (and :meth:`ensure` if the arm spends
    money), or build one from a function with :func:`from_fn`.
    """

    name: str
    family: str
    #: Free-form, hashed into the config hash that the run log counts K with.
    #: Put every hyperparameter here or K is silently understated.
    config: dict = field(default_factory=dict)
    model: str | None = None
    features: tuple[str, ...] = ()
    cost_usd_per_event: float = 0.0
    live_computable: bool = True
    cutoff_safe: bool = True
    rationale: str = ""
    #: Set when an arm is retired. A *branch* kill needs a mechanism, not a
    #: score — see BUILD_LOOP.md adaptation 3. Recorded so the reason survives
    #: the session that made the call.
    killed: str | None = None

    # -- the two halves -------------------------------------------------

    def ensure(self, events: Sequence[dict], quarter: str) -> None:
        """Do any metered work for ``events`` and cache it. Default: nothing."""

    def predict(self, events: Sequence[dict], quarter: str) -> list[float]:
        raise NotImplementedError

    # -- declared properties --------------------------------------------

    @property
    def tier(self) -> str:
        return cost_tier(self.cost_usd_per_event)

    def live_check(self) -> LiveCheck:
        """Sum the declared live cost of every feature this arm needs.

        Declared, not measured — a feature's own module states its latency and
        fetch count, and the scheduler refuses to spend an evaluation on an arm
        whose declared cost cannot fit the window. That is the cheap half of the
        deployability question; :func:`runner.report.consistency` is the other.
        """
        from runner import features as F

        seconds, fetches, reasons = 0.0, 0, []
        if not self.live_computable:
            reasons.append("declared not live-computable")
        for name in self.features:
            spec = F.SPECS.get(name)
            if spec is None:
                reasons.append(f"feature {name!r} is not registered")
                continue
            seconds += spec.live_seconds
            fetches += spec.live_fetches
            if not spec.cutoff_safe:
                reasons.append(f"feature {name!r} is not cutoff-safe")
        if seconds > LIVE_BUDGET_SECONDS:
            reasons.append(f"{seconds:.0f}s of feature work exceeds the {LIVE_BUDGET_SECONDS:.0f}s budget")
        return LiveCheck(
            ok=self.live_computable and not reasons,
            seconds=seconds,
            fetches=fetches,
            reasons=tuple(reasons),
        )

    def estimated_cost(self, n_events: int) -> float:
        return self.cost_usd_per_event * n_events

    def describe(self) -> dict:
        check = self.live_check()
        return {
            "arm": self.name,
            "family": self.family,
            "model": self.model or "-",
            "tier": self.tier,
            "cost_per_event": self.cost_usd_per_event,
            "live_computable": check.ok,
            "live_seconds": check.seconds,
            "live_fetches": check.fetches,
            "cutoff_safe": self.cutoff_safe,
            "features": ",".join(self.features) or "-",
            "killed": self.killed or "",
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

ARMS: dict[str, Arm] = {}


def add(arm: Arm) -> Arm:
    """Register ``arm``. Duplicate names are an error, not a silent overwrite.

    A silently overwritten arm is the same failure as an unlogged run: two
    different configurations collapse to one entry, and K undercounts.
    """
    if arm.name in ARMS:
        raise ValueError(f"arm {arm.name!r} is already registered")
    ARMS[arm.name] = arm
    return arm


def get(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"no arm {name!r}; registered: {sorted(ARMS)}")
    return ARMS[name]


def live(include_killed: bool = False) -> list[Arm]:
    return [a for a in ARMS.values() if include_killed or a.killed is None]


def by_family(family: str) -> list[Arm]:
    return [a for a in ARMS.values() if a.family == family and a.killed is None]


def kill(name: str, mechanism: str) -> None:
    """Retire an arm, with the mechanism that justifies it.

    Refuses an empty reason on purpose. Killing a *variant* inside a rung is
    what successive halving is for and needs no ceremony; this function is for
    retiring a channel, and BUILD_LOOP.md requires that to be a structural
    statement ("ρ_b 0.824 caps the branch") rather than "it scored 0.002 lower".
    """
    if not mechanism.strip():
        raise ValueError("a kill needs a stated mechanism, not a score")
    get(name).killed = mechanism


# --------------------------------------------------------------------------
# Building arms from plain functions
# --------------------------------------------------------------------------


def from_fn(
    name: str,
    fn: Callable[[Sequence[dict], str], Sequence[float]],
    *,
    family: str,
    ensure_fn: Callable[[Sequence[dict], str], None] | None = None,
    **kwargs,
) -> Arm:
    """Wrap ``fn(events, quarter) -> floats`` as an arm and register it."""

    class _FnArm(Arm):
        def ensure(self, events, quarter):
            if ensure_fn is not None:
                ensure_fn(events, quarter)

        def predict(self, events, quarter):
            values = list(fn(events, quarter))
            if len(values) != len(events):
                raise ValueError(
                    f"arm {name!r} returned {len(values)} predictions for {len(events)} events"
                )
            return [float(v) for v in values]

    return add(_FnArm(name=name, family=family, **kwargs))


# --------------------------------------------------------------------------
# The LLM-read arms, sharing production's prediction path
# --------------------------------------------------------------------------

#: Cheapest and joint-best of the six models swept. Cost is a multiplier on
#: every future run, so this is the default and anything else needs a reason.
FLASH_LITE = "google/gemini-2.5-flash-lite"

#: Measured on the existing arm columns: ~1,050 prompt + ~10 completion tokens
#: for the base arm at flash-lite's (0.10, 0.40) $/M. Context arms cost more and
#: override this; many-shot at N=500 costs two orders more and must.
COST_PER_EVENT = {FLASH_LITE: 1.1e-4}


def llm_cost_per_event(model: str, prompt_tokens: int, completion_tokens: int = 12) -> float:
    """USD per event at the given token counts, from the frontier price table."""
    import frontier

    price_in, price_out = frontier.PRICES.get(model, (1.0, 5.0))
    return prompt_tokens * price_in / 1e6 + completion_tokens * price_out / 1e6


def context_arm(
    name: str,
    builder: Callable[[dict, dict], str],
    *,
    model: str = FLASH_LITE,
    prompt_tokens: int = 1050,
    cache_dir: Path | None = None,
    **kwargs,
) -> Arm:
    """An arm that prepends a context block to the *deployed* prompt.

    The prompt path is ``arms.build_prompt`` -> ``reads.user_prompt`` ->
    ``predict._facts_text``, which is the production module imported rather than
    reimplemented, and ``reads.check_prompt_fidelity`` asserts the two agree.
    That is requirement 1 from BUILD_LOOP.md ("one shared prediction path") and
    it is why a context arm's ΔR² is a number about production rather than about
    a lookalike.

    Caching is by ``(arm, model)`` in ``data/arms/``, keyed on ``event_id``, so
    the columns already generated by ``arms.py`` are picked up as-is and a
    promotion to a larger rung only pays for the events not already on disk.
    """
    import arms as A

    if name not in A.ARMS:
        A.ARMS[name] = builder

    def _ensure(events, quarter):
        todo = list(events)
        if not todo:
            return
        A.run_arm(name, model, [{**e, "quarter": quarter} for e in todo])

    def _predict(events, quarter):
        column = A.column(name, model)
        # An event with no cached read is a *hole*, not a neutral prediction:
        # scoring 0.5 there would quietly convert a coverage failure into a
        # measured result. NaN keeps it out of the paired comparison.
        return [column.get(e["event_id"], float("nan")) for e in events]

    return from_fn(
        name,
        _predict,
        family=kwargs.pop("family", "context"),
        ensure_fn=_ensure,
        model=model,
        cost_usd_per_event=kwargs.pop(
            "cost_usd_per_event", llm_cost_per_event(model, prompt_tokens)
        ),
        **kwargs,
    )


def cached_column(path: Path) -> dict[str, float]:
    """``event_id -> prediction`` from a JSONL column file, or empty."""
    if not path.exists():
        return {}
    out = {}
    for line in path.open():
        if line.strip():
            row = json.loads(line)
            out[row["event_id"]] = float(row["prediction"])
    return out
