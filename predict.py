"""★ THIS IS THE ONLY FILE YOU NEED TO EDIT. ★

`predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

The default implementation asks an OpenAI model for a calibrated percentile. If
`OPENAI_API_KEY` is not set, it returns a 0.5 baseline so the full deploy →
receive → submit round-trip still works without burning credits. Replace the body
of `predict` with whatever strategy you like — the only contract is the return
shape documented below.
"""

from __future__ import annotations

import json
import os

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field

from explaining_markets.config import openai_model

_openai: OpenAI | None = None  # lazy: importing this file must not require a key
_openai_warned = False         # one-shot warning when no key is configured
_shape_logged = False          # one-shot structural sketch of the live payload

# Timeouts, sized against the 5-minute prediction window that opens when your
# handler ACKs the webhook. Worst case is 15 + (120 x 2) + 15 = 270s, which
# fits with ~30s to spare. Nothing upstream retries a failed prediction — once
# the delivery is ACKed the platform considers it done — so the one retry here
# is the only one you get. Raising either value can push you past the deadline.
SUMMARY_TIMEOUT_SECONDS = 15.0
LLM_TIMEOUT_SECONDS = 120.0
LLM_MAX_RETRIES = 1

# Hard cap on how much of the event payload reaches the prompt. Ten facts run
# well under this; it only bites on the raw-JSON fallback path below.
SUMMARY_CHAR_LIMIT = 8000


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL with the event summary JSON
      event["prediction_deadline"] ISO timestamp; submit before this fires

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — where you predict the asset's
    next-day abnormal (market-adjusted) return will rank across all of the
    quarter's event outcomes: 0 = the quarter's most negative reaction,
    0.50 = median, 1 = its most positive. It's a cross-sectional rank across the
    quarter's events, not a percentile within the asset's own history.
    """
    summary = httpx.get(event["information_url"], timeout=SUMMARY_TIMEOUT_SECONDS)
    summary.raise_for_status()
    summary_json = summary.json()

    # One model call per focal asset, in series — so the LLM budget below is
    # per asset, not per event. Today every event carries a single asset; if
    # that changes and you need several, run them concurrently rather than
    # raising the timeout.
    n_facts = len(_extract_facts(summary_json))

    # One structural sketch per container. `facts=10` proves we FOUND ten
    # sentences; it does not prove ten sentences are ALL the payload carries —
    # and the extractor only located them via the depth-first fallback, so the
    # live shape matches neither the documented sample nor the archive. Whether
    # anything richer (a transcript, statements) sits alongside them decides
    # whether the offline archive is a faithful proxy of the real input.
    global _shape_logged
    if not _shape_logged:
        print(f"[SHAPE] information_url payload: {_describe_shape(summary_json)}")
        _shape_logged = True

    # No facts, no call. The official baselines return 0.5 when the bundle
    # carries no usable facts; we were dumping the raw JSON into the prompt and
    # asking anyway. It is 5 events in the whole 8,020-event archive, so the
    # scoring impact is nil — but it made our system differ from the baselines
    # on those rows, which means any comparison against them was measuring two
    # things at once. Matching the contract costs nothing and removes that.
    predictions = [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": (
                _ask_llm(
                    summary=summary_json,
                    ticker=asset["identifier_value"],
                    event_type=event["event_type"],
                )
                if n_facts
                else 0.5
            ),
        }
        for asset in event["focal_assets"]
    ]

    # One greppable line per prediction. Without this the logs record only
    # failures, so the only way to answer "are my predictions varying at all?"
    # is to re-run the model offline — and a submission stuck on a constant
    # value scores exactly 0 while looking perfectly healthy from the outside.
    # `facts=` doubles as a per-event check that extraction worked.
    for row in predictions:
        print(
            f"[PREDICT] event={event.get('event_id')} "
            f"ticker={row['identifier_value']} "
            f"p={row['predicted_percentile']:.3f} facts={n_facts}"
        )

    return predictions


# ----------------------------------------------------------------------
# Default strategy: a single calibrated LLM call per asset.
# Swap this out, or rewrite `predict` entirely, to enter your own model.
# ----------------------------------------------------------------------


class Prediction(BaseModel):
    """Structured response shape for the LLM call.

    The `Field(ge=0, le=1)` constraint flows through into the JSON schema OpenAI's
    structured-outputs mode enforces during decoding, so the model is guaranteed to
    return a percentile in [0, 1] — no manual clamping or fallback parsing needed.
    """

    predicted_percentile: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """\
You are a senior equity analyst predicting how a stock will react to an event.

Predict a single percentile in [0, 1] for how the focal asset's next-day
abnormal return will rank across all of the quarter's event outcomes:
0 = the quarter's most negative reaction, 0.50 = median, 1 = its most positive.
The relevant return is the *unexpected*, market-adjusted return — a
great-but-fully-priced-in beat is not a top-decile event.

Calibration discipline:
- Long-run base rates: about 25% of events land "up" (>0.75), 50% "neutral"
  (0.25-0.75), 25% "down" (<0.25). Default toward 0.40-0.60 when signals are
  mixed or modest.
- Reserve values above 0.80 or below 0.20 for cases with unambiguous,
  multi-signal evidence. Do not exceed 0.90 or fall below 0.10 without
  overwhelming, lopsided evidence.
- Tone alone (confident vs hedging language) should move you no more than
  ~0.03 absent quantitative confirmation.
"""


#: A list of at least this many strings, averaging at least this many characters,
#: is taken to be the fact list. Tuned against the real article: ten sentences
#: averaging ~180 chars. No other list in any observed payload comes close —
#: tickers, ids and status strings are short, and metadata lists are tiny.
FACTS_MIN_COUNT = 3
FACTS_MIN_MEAN_CHARS = 40


def _as_fact_list(value: object) -> list[str] | None:
    """``value`` if it looks like a list of fact sentences, else ``None``."""
    if not isinstance(value, list):
        return None
    strings = [v.strip() for v in value if isinstance(v, str) and v.strip()]
    if len(strings) < FACTS_MIN_COUNT:
        return None
    if sum(len(s) for s in strings) / len(strings) < FACTS_MIN_MEAN_CHARS:
        return None
    return strings


def _search_for_facts(node: object, depth: int = 0) -> list[str] | None:
    """Depth-first hunt for a fact-shaped list anywhere in the payload.

    The safety net. The exact live shape of ``information_url`` was not known
    when this was written — the documented sample says ``response.facts``, and
    production logs proved otherwise — so rather than guess again, find the
    sentences wherever they are. ``_as_fact_list`` is strict enough that a
    false positive would need three-plus long strings in a list, which nothing
    but the facts has.
    """
    if depth > 6:
        return None
    found = _as_fact_list(node)
    if found:
        return found
    if isinstance(node, dict):
        for value in node.values():
            found = _search_for_facts(value, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _search_for_facts(value, depth + 1)
            if found:
                return found
    return None


def _extract_facts(summary: object) -> list[str]:
    """Pull the event's facts out of an `information_url` payload.

    Tries the two documented shapes first, because an exact match is worth more
    than a heuristic:

      live sample    ``{"response": {"facts": [...]}}``
      archive record ``{"disclosure": {"items": [{"kind": "facts", "content": [...]}]}}``

    Matches on ``kind`` rather than assuming a single disclosure item — ``kind``
    and ``source`` are open string sets upstream. Falls back to
    :func:`_search_for_facts` for anything else, including a bare list of
    sentences. Returns ``[]`` only when the payload holds nothing fact-shaped.
    """
    direct = _as_fact_list(summary)
    if direct:
        return direct

    if isinstance(summary, dict):
        response = summary.get("response")
        if isinstance(response, dict):
            known = _as_fact_list(response.get("facts"))
            if known:
                return known

        known = _as_fact_list(summary.get("facts"))
        if known:
            return known

        # The live shape, confirmed from production on 2026-08-14: the archive's
        # disclosure object, unwrapped. ``items`` sits at the top level beside
        # schema_version / event_id / generated_at, with no ``disclosure`` key.
        # Handled explicitly so the depth-first fallback below stops being
        # load-bearing for every single event — a rescue path that always runs
        # is not a rescue path, and it would keep finding *something* even if
        # the facts moved somewhere they should not be.
        disclosure = summary.get("disclosure")
        containers = [disclosure] if isinstance(disclosure, dict) else []
        containers.append(summary)
        for container in containers:
            for item in container.get("items") or []:
                if isinstance(item, dict) and item.get("kind") == "facts":
                    known = _as_fact_list(item.get("content"))
                    if known:
                        return known

    return _search_for_facts(summary) or []


def _describe_shape(node: object, depth: int = 0) -> str:
    """A compact structural sketch of a payload — keys and types, no values.

    Printed alongside the fallback warning so an unrecognised shape can be
    diagnosed from the logs without echoing the event's content.
    """
    if depth > 3:
        return "..."
    if isinstance(node, dict):
        inner = ", ".join(
            f"{k}: {_describe_shape(v, depth + 1)}" for k, v in list(node.items())[:12]
        )
        return "{" + inner + "}"
    if isinstance(node, list):
        head = _describe_shape(node[0], depth + 1) if node else "empty"
        return f"[{len(node)} x {head}]"
    if isinstance(node, str):
        return f"str({len(node)})"
    return type(node).__name__


def _facts_text(summary: object) -> str:
    """Render the facts as a bullet list for the prompt.

    Worth knowing why this exists: this file used to read ``summary["summary"]``,
    a key present in neither payload shape. Every event therefore fell through
    to ``json.dumps(summary)``, feeding the model the whole raw blob —
    provenance comments, metadata and all — instead of ten clean sentences. The
    official baselines render the facts as a bullet list, so this matches them.

    The raw-JSON dump is kept as a last resort, but now warns: an unrecognised
    payload should still yield a prediction rather than an exception, and a
    silent fallback is exactly the failure this fix exists to surface.
    """
    facts = _extract_facts(summary)
    if facts:
        return "\n".join(f"- {fact}" for fact in facts)[:SUMMARY_CHAR_LIMIT]

    legacy = summary.get("summary") if isinstance(summary, dict) else None
    if isinstance(legacy, str) and legacy.strip():
        return legacy[:SUMMARY_CHAR_LIMIT]

    print(
        "[WARN] no facts found in information_url payload — falling back to raw "
        "JSON, so the model is getting a degraded prompt. Payload shape was: "
        f"{_describe_shape(summary)}"
    )
    return json.dumps(summary)[:SUMMARY_CHAR_LIMIT]


def _ask_llm(*, summary: dict, ticker: str, event_type: str) -> float:
    """Ask the configured model for a calibrated percentile via structured outputs.

    Returns the model's `predicted_percentile`. Falls back to 0.5 if no
    `OPENAI_API_KEY` is configured or the model refuses; the [0, 1] bound is
    enforced by the JSON schema, not by us.
    """
    global _openai, _openai_warned
    if not os.environ.get("OPENAI_API_KEY"):
        if not _openai_warned:
            print(
                "[WARN] OPENAI_API_KEY not set — submitting 0.5 placeholder. "
                "Set the key (or edit predict.py) for real predictions."
            )
            _openai_warned = True
        return 0.5
    if _openai is None:
        # picks up OPENAI_API_KEY from env
        _openai = OpenAI(
            timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
        )

    summary_text = _facts_text(summary)

    user_prompt = (
        f"Event type: {event_type}\n"
        f"Ticker: {ticker}\n\n"
        f"Facts extracted from the earnings call:\n{summary_text}\n\n"
        "Weigh, in roughly this order:\n"
        "  1. Quantitative surprise vs expectations — revenue, EPS, segment metrics.\n"
        "  2. Guidance / outlook — raises, holds, cuts vs the prior trajectory.\n"
        "  3. Strategic shifts — product launches, M&A, capital allocation, leadership.\n"
        "  4. Tone and confidence in management commentary (small weight).\n"
        "  5. Risks called out — regulatory, supply chain, demand, competition.\n\n"
        f"Predict the next-day unexpected-return percentile for {ticker}."
    )

    resp = _openai.chat.completions.parse(
        model=openai_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Prediction,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        return 0.5  # model refused; competition expects a number
    return parsed.predicted_percentile
