"""A MAP-Elites archive: keep arms that are *different from each other*, not the best k.

OpenEvolve's quality-diversity archive is the piece of that design worth taking
wholesale, and the reason is our own measurement rather than the literature's.
A single-best archive, fed by a ΔR² objective, converges on the readers most
correlated with each other — which is the state this project is already in:

    Six models, four vendors, ρ_b 0.824, collectively worth one arm.

So the archive is a grid, and an arm competes only against arms in its own cell.
The behaviour dimensions, per ``BUILD_LOOP.md``:

``rho_b_band``  correlation with the champion, banded. The load-bearing one: an
                arm at ρ_b 0.2 and an arm at ρ_b 0.8 are not competing for the
                same slot even if the second scores higher standalone.
``family``      what kind of channel it is — read, context, embedding,
                extraction, fitted, mechanical, peer. Cheap insurance against a
                single family filling the grid through sheer arm count.
``tier``        cost. Keeps a cheap arm of similar value from being evicted by
                an expensive one, because cost is a multiplier on every future
                run and the archive is what the next generation is built from.

Within a cell the winner is **marginal contribution**, not ΔR². A cell is a
place where redundancy is already controlled for; the question inside it is
still "what does this add?".

Two properties inherited from the reference implementation and worth keeping:

* the absolute best arm is tracked separately from the grid, so a champion
  cannot be lost to a cell reshuffle;
* the archive is persisted after every update, so an interrupted loop resumes
  rather than restarts.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE = ROOT / "runs" / "archive.json"

#: ρ_b bands. The boundaries are where the ceiling arithmetic changes character:
#: below 0.4 a second arm roughly doubles the ceiling, above 0.8 it buys almost
#: nothing. TF-IDF's measured 0.193 sits in ``<0.20``, every LLM read measured so
#: far sits in ``0.60-0.80`` or above.
RHO_B_BANDS = ((0.20, "<0.20"), (0.40, "0.20-0.40"), (0.60, "0.40-0.60"),
               (0.80, "0.60-0.80"), (1.01, ">=0.80"))

#: Mirrors :data:`runner.schedule.SELECTION_RUNG`. Imported by value rather than
#: by module to keep the archive free of a circular import; asserted equal in
#: ``runner.schedule.__main__``.
_SELECTION_RUNG = 2


def rho_b_band(rho_b: float) -> str:
    """Band on |ρ_b|. Sign is not a behaviour: an anti-correlated arm is as
    combinable as a correlated one, and the GLS weights find the sign."""
    if rho_b != rho_b:  # NaN
        return "unknown"
    value = abs(rho_b)
    for bound, name in RHO_B_BANDS:
        if value < bound:
            return name
    return ">=0.80"


@dataclass
class Elite:
    """One arm's record in the archive, at the rung it was last scored on."""

    arm: str
    family: str
    tier: str
    rung: int
    n: int
    delta_r2: float
    pct_obtainable: float
    rho: float
    rho_b_champion: float
    vs_champion: float
    marginal_rho: float
    marginal_delta_r2: float
    coverage: float
    neutral_rate: float
    cost_usd: float
    live_computable: bool
    cutoff_safe: bool
    signs: str = ""
    timestamp: str = ""

    @property
    def cell(self) -> tuple[str, str, str]:
        return (rho_b_band(self.rho_b_champion), self.family, self.tier)


@dataclass
class Archive:
    """The grid, plus the outright best, plus everything ever inserted.

    ``history`` is not a nicety. ``BUILD_LOOP.md``: *log every arm including
    failures, compute the floor at the true K, report what clears it. The
    failure mode was never testing too much; it was reporting the best of many
    as if it were the only one.*
    """

    cells: dict[str, Elite] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    best: Elite | None = None

    # -- insertion -------------------------------------------------------

    def consider(self, elite: Elite) -> tuple[bool, str]:
        """Try to place ``elite``. Returns ``(accepted, why)``.

        Every attempt lands in ``history`` whether or not it wins a cell, so the
        K the promotion floor is scaled by counts the losses too.
        """
        elite.timestamp = datetime.now(timezone.utc).isoformat()
        self.history.append(asdict(elite))

        # An arm's ρ_b is re-measured at every rung, so it can *move* between
        # bands as the estimate sharpens. Without this, the arm's rung-1 record
        # keeps holding its old cell forever while the arm itself moves on —
        # and a rung-1 record is a 666-event estimate against a noise floor of
        # 0.046. Observed: text.tfidf_ridge_a10 held 0.20-0.40 at marginal
        # +0.0467 (rung 1) while its own rung-2 measurement was +0.0096.
        for key in [k for k, v in self.cells.items() if v.arm == elite.arm]:
            del self.cells[key]

        # "Best" is a claim, and a claim needs a measurement that can carry it.
        # Below the selection rung the noise exceeds the gaps being ranked, so a
        # rung-0 winner is a draw from the noise, not a champion.
        if elite.rung >= _SELECTION_RUNG and (
            self.best is None or _score(elite) > _score(self.best)
        ):
            self.best = elite

        key = "|".join(elite.cell)
        incumbent = self.cells.get(key)
        if incumbent is None:
            self.cells[key] = elite
            return True, f"new cell {key}"
        # A larger rung is a better measurement of the same hypothesis, so an
        # arm re-scored deeper replaces its own record even if the number moved
        # down. Refusing that would freeze a lucky rung-1 estimate into the
        # archive, which is precisely the noise-chasing this design avoids.
        if incumbent.arm == elite.arm:
            self.cells[key] = elite
            return True, f"refreshed at rung {elite.rung}"
        if _score(elite) > _score(incumbent):
            self.cells[key] = elite
            return True, f"evicted {incumbent.arm} from {key}"
        return False, f"{incumbent.arm} holds {key}"

    # -- reading ---------------------------------------------------------

    def elites(self) -> list[Elite]:
        return sorted(self.cells.values(), key=_score, reverse=True)

    def members(self) -> list[str]:
        """Arm names currently in the archive — the incumbent set for fitness."""
        return [e.arm for e in self.elites()]

    def occupancy(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for elite in self.cells.values():
            counts[rho_b_band(elite.rho_b_champion)] = (
                counts.get(rho_b_band(elite.rho_b_champion), 0) + 1
            )
        return counts

    def seen(self) -> set[str]:
        return {row["arm"] for row in self.history}

    # -- persistence -----------------------------------------------------

    def save(self, path: Path = STATE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cells": {k: asdict(v) for k, v in self.cells.items()},
                    "best": asdict(self.best) if self.best else None,
                    "history": self.history,
                },
                indent=2,
                default=str,
            )
        )

    @classmethod
    def load(cls, path: Path = STATE) -> Archive:
        if not path.exists():
            return cls()
        blob = json.loads(path.read_text())
        return cls(
            cells={k: Elite(**v) for k, v in blob.get("cells", {}).items()},
            best=Elite(**blob["best"]) if blob.get("best") else None,
            history=blob.get("history", []),
        )


def _score(elite: Elite) -> float:
    """Within-cell ranking: marginal contribution, standalone ΔR² as tiebreak.

    Not ΔR² first. An arm that adds nothing to the ensemble is worth nothing
    however well it scores alone — that is the whole thesis of this loop, and
    putting the tiebreak second is where it becomes operative rather than
    rhetorical.
    """
    marginal = elite.marginal_rho if elite.marginal_rho == elite.marginal_rho else -1e9
    standalone = elite.delta_r2 if elite.delta_r2 == elite.delta_r2 else -1e9
    return marginal + 1e-6 * standalone
