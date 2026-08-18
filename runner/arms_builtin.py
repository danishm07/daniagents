"""The queue, registered as arms.

Ordered by what ``BUILD_LOOP.md`` says is untouched, not by what looks likely.
Eight days in, one family had been tested thoroughly and five had never been
run at all, so the bar for inclusion here is "has this channel ever produced a
number", not "do I expect it to win".

**Free arms first.** Generation is the only step that costs money; scoring, ρ_b,
the ensemble fitness and the bootstrap are arithmetic on cached columns. So
every arm that reads the archive rather than an LLM runs at every rung for
nothing, and the metered budget is reserved for the channels that genuinely
need a model call.

Families registered here:

``read``        the LLM columns already generated — the champion's own family.
                Included so the archive has an honest incumbent set, not because
                the family has room: ρ_b 0.824 across four vendors caps it at
                4.5% of obtainable, which is where we already are.
``context``     what we put in front of the model. Cached from ``arms.py``.
``text``        TF-IDF and its relatives. **The one measured decorrelation in
                this project** — ρ_b 0.193 — and free.
``peer``        the cross-sectional channel. Archive-computable, structurally
                outside the read family, and never scored as a direct predictor.
``fitted``      stacking, ridge, gradient boosting. Both top submissions fit
                nothing; that is their design decision, not a measurement.
``mechanical``  residual targeting, random subspaces, algorithmic diversity.
                Zero K cost, because they combine rather than select.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import harness  # noqa: E402
import views  # noqa: E402
from runner import registry as R  # noqa: E402

FLASH_LITE = R.FLASH_LITE


# --------------------------------------------------------------------------
# read — the existing LLM columns, at zero marginal cost
# --------------------------------------------------------------------------


def _read_column(path: Path):
    def predict(events, quarter):
        column = R.cached_column(path)
        return [column.get(e["event_id"], float("nan")) for e in events]

    return predict


def _register_cached_reads() -> None:
    """Every read column already on disk, as a zero-cost arm.

    These are sunk cost: the calls are paid for, the columns are cached, and
    re-scoring them at every rung is free. They belong in the archive because
    the *marginal* fitness needs an honest incumbent set — measuring what a new
    channel adds over "the champion alone" would overstate it, since we already
    have six reads and they are worth one arm between them.
    """
    for path in sorted((ROOT / "data" / "reads").glob("*.jsonl")):
        variant, _, model = path.stem.partition("__")
        name = f"read.{variant}.{model}"
        R.from_fn(
            name,
            _read_column(path),
            family="read",
            model=model.replace("_", "/", 1),
            cost_usd_per_event=0.0,  # already generated
            config={"variant": variant, "model": model, "source": "cached"},
            rationale="already generated; incumbent set for the marginal fitness",
        )


def _register_cached_contexts() -> None:
    """The context arms ``arms.py`` generated, likewise free to re-score."""
    for path in sorted((ROOT / "data" / "arms").glob("*.jsonl")):
        arm, _, model = path.stem.partition("__")
        name = f"ctx.{arm}.{model.split('_')[-1]}"
        R.from_fn(
            name,
            _read_column(path),
            family="context",
            model=model.replace("_", "/", 1),
            cost_usd_per_event=0.0,
            config={"arm": arm, "model": model, "source": "cached"},
            rationale="cached context arm; re-scored under the marginal fitness",
        )


_register_cached_reads()
_register_cached_contexts()


# --------------------------------------------------------------------------
# text — the only measured decorrelation, and it is free
# --------------------------------------------------------------------------

R.from_fn(
    "text.tfidf_ridge",
    views.tfidf_residual(alpha=1.0),
    family="text",
    cost_usd_per_event=0.0,
    config={"alpha": 1.0, "max_features": 20000, "ngram": "1-2"},
    features=("facts_chars",),
    rationale="ρ_b 0.193 against the champion — the only decorrelation ever measured here",
)

for alpha in (0.3, 3.0, 10.0):
    R.from_fn(
        f"text.tfidf_ridge_a{alpha:g}",
        views.tfidf_residual(alpha=alpha),
        family="text",
        cost_usd_per_event=0.0,
        config={"alpha": alpha},
        rationale="ridge strength sweep — variants die on score inside a rung, which is what rungs are for",
    )

R.from_fn(
    "text.tfidf_char",
    views.tfidf_residual(alpha=1.0, ngram_range=(3, 5)),
    family="text",
    cost_usd_per_event=0.0,
    config={"ngram": "3-5", "note": "word n-grams at 3-5, a different view of the same text"},
    rationale="representation-not-read: a different tokenisation is a different channel",
)

R.from_fn(
    "control.facts_length",
    views.facts_length,
    family="control",
    cost_usd_per_event=0.0,
    config={},
    features=("facts_chars",),
    rationale="no mechanism. If it clears the floor, the floor is too low. Scored on purpose",
)


# --------------------------------------------------------------------------
# peer — the cross-sectional channel, as a direct predictor
# --------------------------------------------------------------------------
#
# ``arms.ctx_peers`` puts this in front of an LLM. That costs money and buries
# the channel inside a read. Scored directly it costs nothing and answers the
# structural question first: does the peer aggregate carry *any* residual signal?
#
# Two leak fixes, both load-bearing and both measured:
#
# * the filter is ``peer.window_end <= our knowledge_cutoff``, not "peer reported
#   first". The looser filter agrees 99.3% of the time over 4.69M peer-pairs —
#   which is exactly why it would survive review and still be wrong.
# * we aggregate ``car1``, never ``y``. ``y`` is a percentile rank within the
#   full quarter and does not exist until the quarter closes. Aggregating it
#   would embed the entire cross-section in a feature that looks causal.


@lru_cache(maxsize=1)
def _settled_frame() -> pd.DataFrame:
    """Every dev event with the instant its return window **actually** closed.

    ⚠️ **This function used to leak, and the leak was a Rules §04 violation
    rather than a modelling error.** It approximated settlement as
    ``knowledge_cutoff + 1 day``, reasoning from ``n_obs == 2`` on 100% of rows
    and ``window_start_date == knowledge_cutoff`` on 94%. Measured against the
    archive's own ``event_returns[ticker].window_end_date``: cutoff+1 is right
    on 7,382 of 8,020 events, **616 settle two to seven days later**, and 12
    settle same-day. On those 616 the peer arm was admitting peers whose return
    window had not yet closed at our prediction time — using an outcome that did
    not exist yet.

    That is exactly the failure mode the ``peer.window_end_date <= our.cutoff``
    rule exists to prevent, and it slipped in as a *convenience approximation of
    the same rule*. Any recorded ρ for the peer arm predating this fix is
    inflated and must be re-measured, not adjusted.

    Settlement is 16:00 America/New_York on ``window_end_date``, resolved
    through the tz database rather than a fixed −4h offset: the archive spans
    two DST boundaries (Nov 2025, Mar 2026) and a fixed offset is wrong on one
    side of each.
    """
    import gzip
    import json

    rows = []
    for quarter in harness.QUARTERS[:-1]:
        path = harness.ARCHIVE_DIR / f"EARNINGS_RELEASE_{quarter}.jsonl.gz"
        for line in gzip.open(path, "rt"):
            if not line.strip():
                continue
            record = json.loads(line)
            ticker = record["focal_assets"][0]["identifier_value"]
            returns = (record.get("event_returns") or {}).get(ticker) or {}
            rows.append(
                {
                    "event_id": record["event_id"],
                    "car1": returns.get("car1"),
                    "window_end_date": returns.get("window_end_date"),
                }
            )
    frame = pd.DataFrame(rows).dropna(subset=["car1", "window_end_date"])
    frame["car1"] = pd.to_numeric(frame.car1, errors="coerce")
    end = pd.to_datetime(frame.window_end_date) + pd.Timedelta(hours=16)
    frame["window_end"] = end.dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    ).dt.tz_convert("UTC")
    return frame.dropna(subset=["car1"]).reset_index(drop=True)


def peer_view(window_days: int = 10, min_peers: int = 3):
    """Mean standardised abnormal return of peers whose window has closed.

    Not a sector or supply-chain peer set — every announcer in the window. That
    is the weakest possible version of the channel and therefore the right one
    to measure first: if the market-wide "how is earnings news being received
    right now" aggregate carries nothing, a narrower peer set is a much larger
    build resting on an untested premise.
    """

    def view(events, quarter):
        settled = _settled_frame()
        sd = settled.car1.std() or 1.0
        out = []
        for event in events:
            cutoff = event["knowledge_cutoff"]
            window = settled[
                (settled.window_end <= cutoff)
                & (settled.window_end >= cutoff - pd.Timedelta(days=window_days))
                & (settled.event_id != event["event_id"])
            ]
            # A thin window is a hole, not a neutral. NaN keeps it out of the
            # paired comparison instead of laundering it into a measured 0.5.
            out.append(float(window.car1.mean() / sd) if len(window) >= min_peers else float("nan"))
        return out

    return view


for days in (3, 10, 30):
    R.from_fn(
        f"peer.window{days}d",
        peer_view(window_days=days),
        family="peer",
        cost_usd_per_event=0.0,
        config={"window_days": days, "min_peers": 3, "aggregate": "car1_z", "filter": "window_end<=cutoff"},
        rationale="archive-computable, directional, structurally outside the read family",
    )


def peer_dispersion(window_days: int = 10, min_peers: int = 5):
    """Share of recent peers with a positive reaction — a breadth measure.

    Different statistic, same window: the mean is moved by one large mover,
    breadth is not, and the two disagree exactly when the aggregate is thin.
    """

    def view(events, quarter):
        settled = _settled_frame()
        out = []
        for event in events:
            cutoff = event["knowledge_cutoff"]
            window = settled[
                (settled.window_end <= cutoff)
                & (settled.window_end >= cutoff - pd.Timedelta(days=window_days))
                & (settled.event_id != event["event_id"])
            ]
            out.append(float((window.car1 > 0).mean()) if len(window) >= min_peers else float("nan"))
        return out

    return view


R.from_fn(
    "peer.breadth10d",
    peer_dispersion(),
    family="peer",
    cost_usd_per_event=0.0,
    config={"window_days": 10, "statistic": "share_positive"},
    rationale="breadth rather than mean — robust to a single large mover",
)


# --------------------------------------------------------------------------
# fitted — the leaders fit nothing, but that is a design choice not a result
# --------------------------------------------------------------------------


def _text_matrix(train: pd.DataFrame, events, max_features: int, ngram: tuple[int, int]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram, sublinear_tf=True,
        min_df=3, stop_words="english",
    )
    X = vectorizer.fit_transform(train.facts.map(views._facts_text))
    Z = vectorizer.transform([views._facts_text(e["facts"]) for e in events])
    return X, Z


def gbm_text(n_estimators: int = 300, max_depth: int = 3, max_features: int = 2000):
    """Gradient boosting on the same text features the ridge sees.

    Algorithmic diversity as a *channel*, not a sweep: a tree ensemble and a
    linear model on identical features disagree in a structured way, so both are
    kept as arms rather than one being declared the winner.
    """

    def view(events, quarter):
        from sklearn.ensemble import HistGradientBoostingRegressor

        train = harness.training_data(quarter)
        if train.empty:
            return [float("nan")] * len(events)
        target = harness.residualize(train, "y").to_numpy(dtype=float)
        X, Z = _text_matrix(train, events, max_features, (1, 1))
        model = HistGradientBoostingRegressor(
            max_iter=n_estimators, max_depth=max_depth, learning_rate=0.05,
            random_state=0,
        ).fit(np.asarray(X.todense()), target)
        return list(model.predict(np.asarray(Z.todense())))

    return view


R.from_fn(
    "fitted.gbm_text",
    gbm_text(),
    family="fitted",
    cost_usd_per_event=0.0,
    config={"n_estimators": 300, "max_depth": 3, "max_features": 2000},
    rationale="tree vs linear on identical features — kept as an arm, not a sweep winner",
)


# --------------------------------------------------------------------------
# mechanical — diversity generators, zero K cost
# --------------------------------------------------------------------------
#
# These combine rather than select, so they do not consume a look at the data in
# the way a new hypothesis does. Residual targeting in particular is orthogonal
# *by construction*: train on what the champion got wrong and the result cannot
# be a paraphrase of the champion.


def residual_target(alpha: float = 1.0, max_features: int = 20_000):
    """Text ridge trained on the champion's residual, not on ``y``.

    Orthogonality by construction rather than by hope. The champion's residual
    is what is left after the deployed system has read the facts, so a model
    fitted to it is fitted to the part of the text the champion is not using.
    """
    column = harness.CHAMPION_COLUMN

    def view(events, quarter):
        train = harness.training_data(quarter)
        if train.empty or column not in train:
            return [float("nan")] * len(events)
        train = train.dropna(subset=[column])
        if len(train) < 200:
            return [float("nan")] * len(events)
        from sklearn.linear_model import Ridge

        # What the champion missed: y net of the surprise, net of the champion.
        y_resid = harness.residualize(train, "y").to_numpy(dtype=float)
        champ = train[column].to_numpy(dtype=float)
        slope, intercept = np.polyfit(champ, y_resid, 1)
        target = y_resid - (slope * champ + intercept)

        X, Z = _text_matrix(train, events, max_features, (1, 2))
        model = Ridge(alpha=alpha).fit(X, target)
        return list(model.predict(Z))

    return view


R.from_fn(
    "mech.residual_target",
    residual_target(),
    family="mechanical",
    cost_usd_per_event=0.0,
    config={"alpha": 1.0, "target": "y - surprise - champion"},
    rationale="orthogonal to the champion by construction, not by hope",
)


def random_subspace(seed: int, fraction: float = 0.7, alpha: float = 1.0, max_features: int = 20_000):
    """Ridge on a random ~70% of the vocabulary.

    Cheap ensemble diversity with a real mechanism: each member sees a different
    part of the text, so their errors decorrelate without any of them being
    noise. The seed is the whole configuration.
    """

    def view(events, quarter):
        train = harness.training_data(quarter)
        if train.empty:
            return [float("nan")] * len(events)
        from sklearn.linear_model import Ridge

        target = harness.residualize(train, "y").to_numpy(dtype=float)
        X, Z = _text_matrix(train, events, max_features, (1, 2))
        rng = np.random.default_rng(seed)
        keep = rng.random(X.shape[1]) < fraction
        model = Ridge(alpha=alpha).fit(X[:, keep], target)
        return list(model.predict(Z[:, keep]))

    return view


for seed in (1, 2, 3):
    R.from_fn(
        f"mech.subspace{seed}",
        random_subspace(seed),
        family="mechanical",
        cost_usd_per_event=0.0,
        config={"seed": seed, "fraction": 0.7, "alpha": 1.0},
        rationale="random feature subspaces — decorrelated members with a mechanism",
    )
