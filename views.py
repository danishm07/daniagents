"""Candidate views. Each one answers a single question: is this a new channel?

A view is a function ``(events, quarter) -> one float per event``. What matters
about a new view is **not** its ΔR². It is ρ_b against the champion — whether it
is reading something the deployed system is not already reading. Combining ``k``
views asymptotes to ``ρ/√ρ_b``, so a weaker view that is decorrelated raises the
ceiling while a stronger view that is redundant does not.

Decorrelation is necessary, not sufficient: noise is perfectly decorrelated and
worth nothing. A view has to clear both bars — real ρ *and* low ρ_b.

Everything here trains on :func:`harness.training_data`, which hands back only
quarters strictly before the one being predicted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

import harness


def _facts_text(facts: list[str]) -> str:
    return "\n".join(facts)


def tfidf_residual(
    alpha: float = 1.0, max_features: int = 20_000, ngram_range: tuple[int, int] = (1, 2)
):
    """Bag-of-words ridge trained to predict ``y`` net of the surprise.

    The one signal that survived the 2026-08-11 study (+0.006 mean, positive on
    3/3 quarters) — but it was judged against the wrong bar, and against Gemini
    rather than the champion. It is re-run here for two reasons, and the second
    one is the real one:

    1. its floor was assumed at 0.010 when the measured floor for a
       high-ρ_b tweak is far lower, so it may have been rejected wrongly;
    2. it is the **cheapest available probe of the text-features-that-are-not-
       an-LLM-read family**. Its ρ_b against the champion is the first evidence
       on whether that family is decorrelated at all — which is exactly the bet
       extraction and embeddings are making. A low ρ_b here is a green light for
       both; a high one says the facts text has a single readable signal and
       everyone who reads it lands in the same place.

    Trained on the residual rather than raw ``y`` because the residual is what
    the contest pays for — the benchmark already owns the surprise.
    """

    def view(events: list[dict], quarter: str) -> list[float]:
        train = harness.training_data(quarter)
        if train.empty:
            # NaN, not 0.5. A neutral is sitting out in disguise: it scores
            # exactly zero, drags the arm's pooled rho toward zero, and shows up
            # in the metrics as a *measured* null rather than as the coverage
            # hole it is. The runner's neutral-rate column caught this reading
            # 0.333 on the first quarter, which has no prior quarter to fit on.
            return [float("nan")] * len(events)

        target = harness.residualize(train, "y").to_numpy(dtype=float)
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=3,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(train.facts.map(_facts_text))
        model = Ridge(alpha=alpha).fit(matrix, target)
        return list(model.predict(vectorizer.transform([_facts_text(e["facts"]) for e in events])))

    return view


def champion_plus_tfidf(weight: float = 0.25, **kwargs):
    """The champion's read with the text residual blended in on z-scores.

    ``p = z(champion) + w·z(tfidf)``. The blend the study proposed; kept because
    the question "does this add to what we run" is different from "does this
    score on its own", and only the first one decides anything.
    """
    base = tfidf_residual(**kwargs)

    def view(events: list[dict], quarter: str) -> list[float]:
        frame = harness.load(quarter)
        champion = frame[harness.CHAMPION_COLUMN].to_numpy(dtype=float)
        text = np.asarray(base(events, quarter), dtype=float)
        return list(_z(champion) + weight * _z(text))

    return view


def _z(values: np.ndarray) -> np.ndarray:
    sd = np.nanstd(values)
    return (values - np.nanmean(values)) / (sd if sd else 1.0)


def facts_length(events: list[dict], quarter: str) -> list[float]:
    """Total characters in the facts.

    Not a candidate — a control. It has no mechanism, so if it clears the
    promotion floor the floor is too low. Scored deliberately, and its result is
    recorded, so the run log shows what a mechanism-free feature does at this
    K rather than leaving it to intuition.
    """
    return [float(len(_facts_text(e["facts"]))) for e in events]
