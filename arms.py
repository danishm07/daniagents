"""Context arms: what we put in front of the model, held everything else fixed.

Three weeks of infrastructure and not one new signal source tested. This is the
file that tests them. Each arm is a *context builder* — a function from an event
to a block of text prepended to the same base prompt, run through the same model,
scored by the same harness. Only the context varies, so a difference between arms
is a difference in what the model was told.

Compliance, which constrains every builder here:

* Context comes from **strictly prior quarters only**, never the target's own
  quarter. The target quarter's `y` is a full-quarter rank that does not exist at
  prediction time, and its `car1` values postdate most cutoffs.
* That mirrors the leaders' stated pattern — a snapshot frozen before the period,
  embedded, no fetch at prediction time — and it is why every builder takes its
  material from :func:`prior_index` rather than from the live frame.
* No external data. Everything here is archive-derived, so nothing goes through
  ``sources.py`` and there is no audit-log obligation.

The arms, and where each came from:

``base``          the deployed prompt, no context. The control.
``ticker_car1``   the focal ticker's own realized abnormal returns at its prior
                  earnings events. Third place supplies this.
``prev_quarter``  the previous completed quarter's distribution of realized
                  abnormal returns. Third place supplies this.
``prior_facts``   the ticker's most recent prior event's fact summary, so the
                  model can judge change-in-trajectory. Third place supplies this.
``third_place``   all three together — a replication of the third-place design.
``rulebook``      curated earnings-reaction priors from published research.
                  Second place supplies this.
``manyshot_N``    N labelled (facts → realized percentile) pairs from prior
                  quarters. **Neither top submission does this** — both supply
                  summary statistics rather than labelled examples, and the ICL
                  literature says ~500 examples can rival supervised methods.

Usage::

    uv run python arms.py --arms base,third_place,manyshot_50 --model google/gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

AGENT = Path(__file__).parent.parent / "agent"
sys.path[:0] = [str(AGENT), str(AGENT / "src")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AGENT / ".env")

import predict  # noqa: E402

import champion  # noqa: E402
import eval as E  # noqa: E402
import harness  # noqa: E402
import reads  # noqa: E402

OUT = Path(__file__).parent / "data" / "arms"

#: Curated from the research dossier. Frozen text, no fitting, no lookahead —
#: the same shape as the second-place submission's "rulebook of earnings-reaction
#: priors compiled from public research before the quarter".
RULEBOOK = """\
Empirical priors from published research on earnings-announcement reactions:

- Revenue surprise carries information largely independent of EPS surprise
  (correlation ~0.145 between them), with an economic effect around 9% of a
  standard deviation. Weight a revenue beat or miss separately from EPS.
- Prefer absolute net income over per-share figures. Buybacks shrink the share
  count and inflate EPS without improving the business.
- Guidance that diverges from reported results is more informative than either
  alone. A beat with lowered guidance is frequently punished.
- Small beats are suspect. Companies manage toward just-above-consensus, so a
  narrow beat carries less good news than its sign suggests and can reverse.
- Gross margin direction and operating cash flow are less easily managed than
  headline earnings and correlate only moderately with EPS surprise.
- One-time items, asset sales and accounting changes should be discounted
  relative to operational drivers.
- Peer firms that reported earlier in the same quarter move prices for firms yet
  to report; transfer is stronger from larger announcers, and incorporation is
  incomplete.
- The market reacts to the surprise relative to expectations, not to the level.
  Strong absolute numbers that were already expected move prices little.
"""


# --------------------------------------------------------------------------
# The prior-quarters index — every arm's material comes from here
# --------------------------------------------------------------------------


def prior_index(target_quarter: str) -> dict:
    """Everything knowable from quarters strictly before ``target_quarter``.

    Built once per target quarter and reused across events, which is both the
    compliant construction and the cheap one. Returns per-ticker history, the
    previous quarter's outcome distribution, and a pool of labelled examples.
    """
    idx = harness.QUARTERS.index(target_quarter)
    if idx == 0:
        return {"by_ticker": {}, "prev": None, "pool": []}

    prior = pd.concat([harness.load(q) for q in harness.QUARTERS[:idx]], ignore_index=True)
    prior = prior.sort_values("event_datetime")

    by_ticker: dict[str, list[dict]] = {}
    for row in prior.itertuples():
        by_ticker.setdefault(row.identifier_value, []).append(
            {
                "quarter": row.quarter,
                "car1": float(row.car1),
                "y": float(row.y),
                "facts": row.facts,
                "when": row.event_datetime,
            }
        )

    last_q = harness.QUARTERS[idx - 1]
    last = harness.load(last_q)
    prev = {
        "quarter": last_q,
        "n": len(last),
        "mean": float(last.car1.mean()),
        "sd": float(last.car1.std()),
        "quantiles": {q: float(last.car1.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
    }

    pool = [
        {"facts": r.facts, "y": float(r.y), "ticker": r.identifier_value}
        for r in prior.itertuples()
        if r.facts
    ]

    # Peers whose return window has CLOSED, across the target quarter itself and
    # the one before it. Same-quarter peers are the point of the channel — their
    # outcomes are public before our cutoff — but only once settled.
    span = harness.QUARTERS[max(0, idx - 1) : idx + 1]
    frames = []
    for q in span:
        f = harness.load(q)[["event_id", "car1", "knowledge_cutoff"]].copy()
        # window_end_date is one trading day after the cutoff; the archive check
        # showed the window closes a median 4h after an event's own cutoff.
        f["window_end"] = f.knowledge_cutoff + pd.Timedelta(days=1)
        frames.append(f)
    settled = pd.concat(frames, ignore_index=True).dropna(subset=["car1"])

    return {"by_ticker": by_ticker, "prev": prev, "pool": pool, "settled": settled}


# --------------------------------------------------------------------------
# Context builders
# --------------------------------------------------------------------------


def ctx_base(event, idx) -> str:
    return ""


def ctx_ticker_car1(event, idx) -> str:
    hist = idx["by_ticker"].get(event["ticker"], [])
    if not hist:
        return (
            f"Historical context: no prior earnings events for {event['ticker']} "
            f"are on record.\n"
        )
    lines = [
        f"- {h['quarter']}: abnormal return {h['car1']:+.2%} "
        f"(percentile {h['y']:.2f} among that quarter's announcements)"
        for h in hist[-8:]
    ]
    moves = [abs(h["car1"]) for h in hist]
    return (
        f"Historical context — how {event['ticker']} has moved on its own past "
        f"earnings:\n" + "\n".join(lines) + "\n"
        f"Typical absolute move: {np.mean(moves):.2%} over {len(hist)} prior events.\n"
    )


def ctx_prev_quarter(event, idx) -> str:
    p = idx["prev"]
    if not p:
        return ""
    q = p["quantiles"]
    return (
        f"Cross-sectional context — distribution of abnormal returns across all "
        f"{p['n']} announcements in {p['quarter']}, the last completed quarter:\n"
        f"  5th pct {q[0.05]:+.2%} | 25th {q[0.25]:+.2%} | median {q[0.5]:+.2%} | "
        f"75th {q[0.75]:+.2%} | 95th {q[0.95]:+.2%}\n"
        f"  mean {p['mean']:+.2%}, sd {p['sd']:.2%}\n"
    )


def ctx_prior_facts(event, idx) -> str:
    hist = idx["by_ticker"].get(event["ticker"], [])
    if not hist or not hist[-1]["facts"]:
        return ""
    last = hist[-1]
    body = "\n".join(f"  - {f}" for f in last["facts"])
    return (
        f"For comparison, the facts reported at {event['ticker']}'s previous "
        f"earnings event ({last['quarter']}, which was followed by a "
        f"{last['car1']:+.2%} abnormal return):\n{body}\n"
    )


def ctx_third_place(event, idx) -> str:
    return (
        ctx_ticker_car1(event, idx)
        + "\n"
        + ctx_prior_facts(event, idx)
        + "\n"
        + ctx_prev_quarter(event, idx)
    )


def ctx_peers(event, idx) -> str:
    """How the market has received earnings reported *just before* this one.

    Both leak fixes are load-bearing here:

    * The filter is ``peer.window_end_date <= our knowledge_cutoff``, not "peer
      reported first". The looser filter agrees 99.3% of the time — measured over
      4.69M peer-pairs, 0.7% of which still have an open return window at our
      prediction time — which is precisely why it would survive review.
    * We aggregate ``car1``, never ``y``. ``y`` is a percentile rank within the
      full quarter and does not exist until the quarter closes; aggregating it
      would embed the entire cross-section into a feature that looks causal and
      would make the backtest look excellent.

    ``car1`` is standardised by the cross-section's own dispersion so a 4% move
    means the same thing for a biotech and a utility. Dispersion is used as a
    normaliser, not as a signal — magnitude without direction pays ~0, measured
    three separate ways. ``n_peers`` is reported so the model can discount a thin
    aggregate rather than treat 3 peers like 40.
    """
    settled = idx.get("settled")
    if settled is None or settled.empty:
        return ""
    cutoff = event["knowledge_cutoff"]
    window = settled[(settled.window_end <= cutoff) & (settled.window_end >= cutoff - pd.Timedelta(days=10))]
    if len(window) < 3:
        return ""
    sd = settled.car1.std() or 1.0
    z = window.car1 / sd
    return (
        f"Recent market context — {len(window)} companies reported earnings in the "
        f"10 days before this one and their one-day abnormal returns have already "
        f"settled:\n"
        f"  mean {window.car1.mean():+.2%} ({z.mean():+.2f} sd), "
        f"median {window.car1.median():+.2%}, "
        f"share positive {(window.car1 > 0).mean():.0%}\n"
        f"  This says how earnings news is being received right now, not how this "
        f"company will do.\n"
    )


def ctx_rulebook(event, idx) -> str:
    return RULEBOOK


def make_manyshot(n: int):
    """N labelled examples, deterministically chosen, most-similar-last.

    Order matters in ICL, and the convention that helps is putting the strongest
    material closest to the question. Examples are drawn from prior quarters
    only, so no target-quarter outcome is ever shown.
    """

    def builder(event, idx) -> str:
        pool = idx["pool"]
        if not pool:
            return ""
        step = max(1, len(pool) // n)
        chosen = pool[::step][:n]
        blocks = []
        for ex in chosen:
            facts = "\n".join(f"  - {f}" for f in ex["facts"][:10])
            blocks.append(f"{facts}\n  => outcome percentile: {ex['y']:.2f}")
        return (
            f"Here are {len(chosen)} previous earnings events, each with the facts "
            f"reported and the percentile its abnormal return actually landed at. "
            f"Use them to calibrate.\n\n" + "\n\n".join(blocks) + "\n"
        )

    return builder


ARMS = {
    "base": ctx_base,
    "ticker_car1": ctx_ticker_car1,
    "prev_quarter": ctx_prev_quarter,
    "prior_facts": ctx_prior_facts,
    "third_place": ctx_third_place,
    "rulebook": ctx_rulebook,
    "peers": ctx_peers,
    **{f"manyshot_{n}": make_manyshot(n) for n in (5, 20, 50, 200, 500)},
}


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def tag_quarters(events: list[dict]) -> list[dict]:
    """Attach ``quarter`` to each event.

    ``harness.events_for`` deliberately hands back only what a live webhook
    carries, which does not include the quarter — so the arm runner adds it from
    the frame rather than deriving it from ``event_datetime``, which would be a
    guess.
    """
    by_id = {}
    for q in harness.DEV_QUARTERS:
        for event_id in harness.load(q).event_id:
            by_id[event_id] = q
    return [{**e, "quarter": by_id[e["event_id"]]} for e in events if e["event_id"] in by_id]


def build_prompt(event, context: str) -> str | None:
    """The deployed prompt with ``context`` prefixed, or ``None`` for no facts.

    ``None`` means *do not call the model, submit 0.5* — which is what
    ``predict.predict`` does live, and what ``reads`` already did on the backfill
    path. Every other offline generator asked anyway, and ``_facts_text`` then
    fell through to dumping the raw JSON blob into the prompt.

    Five of 6,144 dev events (UTZ, FUBO, HOG, FWONK, SRE, all 2025Q4) carry an
    empty ``content`` list. Scoring impact is nil at that count; the reason to
    fix it is that those five rows made offline columns and production disagree,
    so any offline-vs-baseline comparison was measuring two things at once.
    Returning ``None`` rather than a neutral string keeps a forgotten call site
    loud instead of silently re-degrading.
    """
    payload = champion.live_payload(event["facts"])
    if not predict._extract_facts(payload):
        return None
    base = reads.user_prompt(payload, event["ticker"], champion.EVENT_TYPE)
    return f"{context}\n{base}" if context else base


def _path(arm: str, model: str) -> Path:
    return OUT / f"{arm}__{model.replace('/', '_')}.jsonl"


def run_arm(arm: str, model: str, events: list[dict], workers: int = 12) -> Path:
    builder = ARMS[arm]
    path = _path(arm, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["event_id"] for l in path.open()} if path.exists() else set()
    todo = [e for e in events if e["event_id"] not in done]
    print(f"[{arm}/{model}] cached {len(done)}, to run {len(todo)}", flush=True)
    if not todo:
        return path

    indices = {q: prior_index(q) for q in harness.DEV_QUARTERS}
    client = reads._client()
    lock = threading.Lock()
    started, failures, completed = time.time(), 0, 0

    def one(event):
        idx = indices[event["quarter"]]
        prompt = build_prompt(event, builder(event, idx))
        if prompt is None:  # no facts: production submits 0.5 without calling
            return {"event_id": event["event_id"], "prediction": 0.5,
                    "prompt_tokens": 0, "completion_tokens": 0, "no_facts": True}
        champion._throttle.wait(champion._throttle.estimate(1200))
        resp = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": predict.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format=reads.Direct,
        )
        if resp.usage:
            champion._throttle.record(resp.usage.total_tokens)
        parsed = resp.choices[0].message.parsed
        return {
            "event_id": event["event_id"],
            "prediction": 0.5 if parsed is None else parsed.predicted_percentile,
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool, path.open("a") as fh:
        futures = {pool.submit(one, e): e for e in todo}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    print(f"  [fail] {futures[future]['event_id']}: "
                          f"{type(exc).__name__}: {str(exc)[:110]}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            completed += 1
            if completed % 250 == 0:
                rate = completed / (time.time() - started)
                print(f"  [{arm}] {completed}/{len(todo)} {rate:.1f}/s "
                      f"eta {(len(todo)-completed)/rate/60:.1f}m", flush=True)
    print(f"[{arm}/{model}] done {completed}, failures {failures}, "
          f"{(time.time()-started)/60:.1f} min", flush=True)
    return path


def column(arm: str, model: str) -> dict[str, float]:
    path = _path(arm, model)
    if not path.exists():
        return {}
    return {json.loads(l)["event_id"]: json.loads(l)["prediction"] for l in path.open()}


def cost(arm: str, model: str) -> tuple[int, int]:
    path = _path(arm, model)
    if not path.exists():
        return 0, 0
    rows = [json.loads(l) for l in path.open()]
    return (
        sum(r.get("prompt_tokens") or 0 for r in rows),
        sum(r.get("completion_tokens") or 0 for r in rows),
    )


def score_arms(arms: list[str], model: str, events: list[dict]) -> pd.DataFrame:
    """Per-arm metrics plus the inter-arm ρ_b matrix, all on identical events."""
    wanted = {e["event_id"] for e in events}
    cols = {a: column(a, model) for a in arms}
    cols = {a: c for a, c in cols.items() if c}

    rows, resid = [], {}
    for arm, preds in cols.items():
        per, res_parts = [], []
        for q in harness.DEV_QUARTERS:
            f = harness.load(q)
            f = f[f.event_id.isin(wanted) & f.event_id.isin(preds)].copy()
            if f.empty:
                continue
            f["_p"] = f.event_id.map(preds)
            s = harness.evaluate(f, "_p")
            surprise = f.surprise_pct.to_numpy(float)
            values = f["_p"].to_numpy(float)
            champ = f[harness.CHAMPION_COLUMN].to_numpy(float)
            m = E._correlation_matrix({"a": values, "c": champ}, surprise)
            per.append(
                {
                    "n": s["n_obs"],
                    "delta_r2": s["delta_r_squared"],
                    "pct": E.as_pct_obtainable(s["delta_r_squared"], s["r_squared_surprise"]),
                    "rho": E.partial_corr(values, f.y.to_numpy(float), surprise),
                    "rho_b": float(m.loc["a", "c"]),
                    "vs_champ": s["delta_r_squared"]
                    - harness.evaluate(f, harness.CHAMPION_COLUMN)["delta_r_squared"],
                    "neutral_rate": float((values == 0.5).mean()),
                }
            )
            res_parts.append(pd.Series(E._residualize(values, surprise), index=f.event_id))
        if not per:
            continue
        t = pd.DataFrame(per)
        pin, pout = cost(arm, model)
        rows.append(
            {
                "arm": arm,
                "n": int(t.n.sum()),
                "delta_r2": t.delta_r2.mean(),
                "pct_obtainable": t.pct.mean(),
                "rho": t.rho.mean(),
                "rho_b_champion": t.rho_b.mean(),
                "vs_champion": t.vs_champ.mean(),
                "signs": f"{int((t.vs_champ > 0).sum())}/{len(t)}",
                "neutral_rate": t.neutral_rate.mean(),
                "prompt_tok": pin,
                "completion_tok": pout,
            }
        )
        resid[arm] = pd.concat(res_parts)

    table = pd.DataFrame(rows).sort_values("pct_obtainable", ascending=False)
    matrix = pd.DataFrame(resid).corr() if len(resid) > 1 else None
    return table, matrix


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default="base,third_place,rulebook,manyshot_50")
    p.add_argument("--model", default="google/gemini-2.5-flash-lite")
    p.add_argument("--per-quarter", type=int, default=700)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--rpm", type=int, default=400)
    args = p.parse_args()

    champion._throttle = champion._Throttle(args.rpm, 4_000_000)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    events = tag_quarters(reads.screen_events(args.per_quarter))
    print(f"{len(arms)} arms x {len(events)} events on {args.model}\n")

    if not args.score_only:
        for arm in arms:
            run_arm(arm, args.model, events, args.workers)

    table, matrix = score_arms(arms, args.model, events)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    if matrix is not None:
        print("\ninter-arm correlation (surprise projected out):")
        print(matrix.to_string(float_format=lambda v: f"{v:+.3f}"))
