"""Exact port of the Explaining Markets earnings-call fact-extraction summarizer.

Every ``EARNINGS_RELEASE`` event you receive carries a ``disclosure`` block whose
``items`` include one with ``kind="facts"`` and ``source="earnings_call"``. Its
``content`` is a list of exactly ten sentences. This module is the call that
produces them, so you can see what the extraction optimizes for rather than
treating the facts as an opaque input.

The production request is deliberately plain: a ``system`` string, one ``user``
message, and extended thinking. There is **no tool use, no JSON schema, no
``response_format``, and no ``temperature``** — the ten-fact JSON structure is
enforced by the prompt text alone and then validated after the fact by
:func:`parse_facts` / :func:`recover_facts`. That is worth knowing: the model is
free-forming its output, and the ``parse_note`` you may see in an artifact
records how cleanly it did so.

:data:`SUMMARY_SYSTEM_PROMPT`, :data:`SUMMARY_USER_TEMPLATE`,
:data:`MODEL_CONFIGS`, :func:`parse_facts` and :func:`recover_facts` are exact
semantic ports of the production summarizer — fence stripping, bracket repair,
trim-to-ten and every failure note included — so a request you build here is
byte-identical to the one the pipeline sends. :func:`format_transcript` is the
same renderer the pipeline feeds it.

**You will not reproduce the competition's facts by running this.** Two reasons,
both unavoidable: the production input is a licensed full-length earnings-call
transcript that is not distributed with this repo, and extended thinking makes
the call nondeterministic even on identical input. Treat a local run as an
illustration of the mechanism, not a way to regenerate or predict a disclosure.

Typical use::

    from examples.summary import build_request, format_transcript

    md = format_transcript(transcript)          # what the model actually sees
    params = build_request(md)                  # the exact Messages API kwargs
    print(params["system"])
    print(params["messages"][0]["content"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# The number of facts the prompt demands and the parsers enforce. Not a
# soft target: a response with nine or eleven facts is a parse failure.
N_FACTS = 10

# The model the live pipeline runs. This tracks the production constant
# (``ANTHROPIC_SUMMARY_MODELS`` in the pipeline's ``app.py``) and is updated when
# production switches models, so a disclosure you received months ago may have
# been produced by an earlier entry in MODEL_CONFIGS. The ``model`` that produced
# a given artifact is recorded on the event's source metadata — trust that over
# this default when you are reasoning about a specific past event.
SUMMARY_MODEL = "claude-opus-4-8"

# Per-model request configuration. The thinking API differs by model generation:
#   - opus-4-5: manual extended thinking (enabled + budget_tokens); adaptive
#     unsupported.
#   - opus-4-6 / opus-4-7 / opus-4-8: adaptive thinking (no fixed budget).
#     opus-4-7+ *requires* it (enabled is rejected) and defaults
#     display="omitted", so display="summarized" is set explicitly to keep
#     capturing the thinking summary. effort="high" guarantees thinking on this
#     extraction task (medium skips it on 4-7). max_tokens is a hard cap on
#     thinking+text, not a charge; it is set generously on adaptive models to
#     avoid truncation.
SUMMARY_THINKING_BUDGET = 1024
SUMMARY_MAX_TOKENS = SUMMARY_THINKING_BUDGET + 2000

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "claude-opus-4-5": {
        "max_tokens": SUMMARY_MAX_TOKENS,
        "thinking": {"type": "enabled", "budget_tokens": SUMMARY_THINKING_BUDGET},
        "output_config": None,
    },
    "claude-opus-4-6": {
        "max_tokens": 16000,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    },
    "claude-opus-4-7": {
        "max_tokens": 16000,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    },
    "claude-opus-4-8": {
        "max_tokens": 16000,
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"},
    },
}

# ----- The prompt (verbatim) -------------------------------------------------

SUMMARY_SYSTEM_PROMPT = (
    "You are a financial analyst. You extract the most investor-relevant facts from "
    "earnings call transcripts to help investors anticipate the stock's reaction to the call."
)

SUMMARY_USER_TEMPLATE = """Analyze the following earnings call transcript and extract the {n_facts} most investor-relevant facts.

Each fact must:
- Be drawn only from the transcript (do not use any outside information).
- Be a single sentence capturing a specific, quantified insight relevant to investors.
- Avoid regurgitating the transcript verbatim; instead, synthesize and distill the information into investor-relevant facts.

Return only a JSON object with this exact structure (no markdown fences, no commentary):
{{"facts": ["<fact 1>", "<fact 2>", ..., "<fact {n_facts}>"]}}

The list must contain exactly {n_facts} facts.

Transcript:
<transcript>
{transcript}
</transcript>"""  # noqa: E501


def build_summary_user_prompt(transcript_md: str) -> str:
    """Render the user message for one transcript.

    ``transcript_md`` is the markdown from :func:`format_transcript`. It is
    interpolated whole — see that function for what does and does not reach the
    model.
    """
    return SUMMARY_USER_TEMPLATE.format(n_facts=N_FACTS, transcript=transcript_md)


# ----- The request -----------------------------------------------------------


def build_request(transcript_md: str, model: str = SUMMARY_MODEL) -> dict[str, Any]:
    """Build the exact Anthropic Messages API kwargs the pipeline sends.

    Raises ``ValueError`` for a model with no entry in :data:`MODEL_CONFIGS`,
    rather than guessing a thinking configuration that generation may reject.

    The returned dict is complete — pass it straight to
    ``anthropic.Anthropic().messages.create(**params)``. Note what is *absent*:
    no ``tools``, no ``response_format``, no ``temperature``, no stop sequences.
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"No config for model {model!r}; known: {list(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[model]
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": cfg["max_tokens"],
        "thinking": cfg["thinking"],
        "system": SUMMARY_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_summary_user_prompt(transcript_md)}],
    }
    if cfg.get("output_config"):
        params["output_config"] = cfg["output_config"]
    return params


def build_batch_request(custom_id: str, transcript_md: str, model: str = SUMMARY_MODEL) -> dict:
    """Wrap :func:`build_request` as one Anthropic Batch API request.

    Production summarizes historical quarters through the Batch API, keying each
    request by the transcript's id so results can be joined back.
    """
    return {"custom_id": str(custom_id), "params": build_request(transcript_md, model)}


# ----- The input the model sees ----------------------------------------------


def format_transcript(transcript: dict, *, markdown: bool = True) -> str:
    """Render a raw transcript record to the string that goes into the prompt.

    Worth reading closely, because it bounds what the summarizer can know. The
    *only* metadata prepended is the transcript's own header title (e.g.
    ``"Adeia Inc., Q1 2026 Earnings Call, May 04, 2026"``). No ticker, no event
    date, no fiscal period, and no market data are injected separately — and
    there is **no truncation, chunking, or token budgeting**: the full call goes
    into a single user message.

    ``markdown=False`` reproduces the plain component join, which production uses
    only for internal diagnostics — never for the prompt.
    """
    if not markdown:
        return "\n\n".join([comp["text"] for comp in transcript["components"]])

    transcript_md = ""
    transcript_md += f"# Transcript of {transcript['header']['title']}\n\n"
    for component in transcript["components"]:
        speaker, speaker_type, text = (
            component["speakerName"],
            component["speakerType"],
            component["text"],
        )
        transcript_md += f"**{speaker} ({speaker_type}):** {text}\n\n"

    return transcript_md


# ----- Parsing ---------------------------------------------------------------


def parse_facts(response_text: str) -> list[str]:
    """Parse a model response into exactly :data:`N_FACTS` non-empty facts.

    Strict: raises ``ValueError`` on any deviation (bad JSON, missing key, wrong
    count, empty fact). Production uses this on its offline path;
    :func:`recover_facts` is what the live pipeline runs.
    """
    text = response_text.strip()
    # Strip markdown code fences if the model added them despite instructions.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[len("json") :]
        text = text.strip()
    # Fall back to extracting the first {...} block if there is leading/trailing prose.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in response: {response_text[:200]!r}")
        text = text[start : end + 1]

    obj = json.loads(text)
    if "facts" not in obj:
        raise ValueError(f"Response JSON missing 'facts' key: {list(obj.keys())}")
    facts = obj["facts"]
    if not isinstance(facts, list):
        raise ValueError(f"'facts' is not a list: {type(facts).__name__}")
    if len(facts) != N_FACTS:
        raise ValueError(f"Expected {N_FACTS} facts, got {len(facts)}")
    facts = [str(f).strip() for f in facts]
    if any(not f for f in facts):
        raise ValueError("One or more facts are empty")
    return facts


def recover_facts(response_text: str) -> tuple[list[str] | None, str]:
    """Lenient recovery of ten facts from a possibly malformed response.

    This is the function the live pipeline runs, and its second return value is
    the ``parse_note`` recorded on the published artifact. It applies ONLY safe,
    non-corrupting repairs and never fabricates or splits a fact:

    - strips markdown fences / surrounding prose
    - if the model omitted the array's closing ``]`` (output ends ``"}`` rather
      than ``"]}``), inserts it
    - if the model returned MORE than ten facts, trims to the first ten (the
      prompt orders them by relevance)

    Returns ``(facts, note)``. On anything it cannot recover safely it returns
    ``(None, reason)`` so the call can be re-run rather than emitting wrong
    output — which is why a published artifact can legitimately carry
    ``"facts": null``.

    Note values: ``"strict"``, ``"repaired_close"``, ``"trimmed_from_{n}"`` (or a
    ``"+"``-join), or a failure reason: ``"unparseable"``, ``"no_facts_list"``,
    ``"empty_fact"``, ``"too_few:{n}"``.
    """
    t = response_text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[len("json") :]
        t = t.strip()
    s = t.find("{")
    if s == -1:
        return None, "unparseable"
    t = t[s:]

    # Candidate 1: first '{' to last '}' (handles trailing prose, the normal case).
    # Candidate 2: normalize a broken tail to canonical '"]}'. Strip any trailing run of
    # structural chars/whitespace (']', '}', spaces) back to the last closing quote of the
    # final fact, then append '"]}'. This is non-corrupting: it only touches structural
    # characters after the last string, and is only attempted when the last fact string is
    # properly closed (body ends with '"'), so truncated-mid-string outputs are rejected.
    candidates = []
    e = t.rfind("}")
    if e != -1:
        candidates.append(("strict", t[: e + 1]))
    body = t.rstrip("}] \t\r\n")
    if body.endswith('"'):
        candidates.append(("repaired_close", body + "]}"))

    note: list[str] = []
    obj = None
    for name, cand in candidates:
        try:
            obj = json.loads(cand)
            if name != "strict":
                note.append(name)
            break
        except json.JSONDecodeError:
            continue
    if obj is None:
        return None, "unparseable"

    facts = obj.get("facts")
    if not isinstance(facts, list):
        return None, "no_facts_list"
    facts = [str(f).strip() for f in facts]
    if any(not f for f in facts):
        return None, "empty_fact"
    if len(facts) < N_FACTS:
        return None, f"too_few:{len(facts)}"
    if len(facts) > N_FACTS:
        note.append(f"trimmed_from_{len(facts)}")
        facts = facts[:N_FACTS]
    return facts, "+".join(note) if note else "strict"


# ----- Running the call ------------------------------------------------------


@dataclass(frozen=True)
class SummaryResult:
    """One summarization call's outcome.

    ``facts`` is ``None`` when :func:`recover_facts` could not safely recover ten
    facts; ``parse_note`` then carries the reason. ``thinking_summary`` is the
    model's summarized extended thinking, present only because the request sets
    ``display="summarized"``. ``output_tokens`` includes thinking tokens.
    """

    facts: list[str] | None
    parse_note: str
    raw_response: str
    thinking_summary: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None

    @property
    def ok(self) -> bool:
        """True when ten facts were recovered."""
        return self.facts is not None


def summarize_transcript(
    transcript_md: str,
    client: Any,
    model: str = SUMMARY_MODEL,
) -> SummaryResult:
    """Run one summarization call and parse the result.

    ``client`` is an ``anthropic.Anthropic`` instance, passed in rather than
    constructed here so it can be reused across calls (see
    :meth:`examples.config.Config.anthropic_client`).

    API errors propagate; a *parse* failure does not — it comes back as a
    :class:`SummaryResult` with ``facts=None`` and an explanatory ``parse_note``,
    matching how production records failures without dropping the response.
    """
    resp = client.messages.create(**build_request(transcript_md, model))

    raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    thinking_summary = (
        "".join(b.thinking for b in resp.content if getattr(b, "type", None) == "thinking") or None
    )
    facts, parse_note = recover_facts(raw)
    return SummaryResult(
        facts=facts,
        parse_note=parse_note,
        raw_response=raw,
        thinking_summary=thinking_summary,
        input_tokens=getattr(resp.usage, "input_tokens", None),
        output_tokens=getattr(resp.usage, "output_tokens", None),
        stop_reason=getattr(resp, "stop_reason", None),
    )


# ----- Reading what the API delivers -----------------------------------------

FACTS_KIND = "facts"
EARNINGS_CALL_SOURCE = "earnings_call"


def facts_from_disclosure(record: dict) -> list[str]:
    """Pull the earnings-call facts out of an event or archive record.

    Looks for the first ``disclosure.items[]`` entry with ``kind="facts"`` and
    ``source="earnings_call"`` and returns its ``content``. Returns ``[]`` when
    the record carries no such item — a scheduled-but-not-yet-occurred event, or
    a call whose summarization failed, legitimately has none.

    ``record`` is a raw archive line (as yielded by
    :func:`examples.archive.read_jsonl_gz`) or an equivalent event payload.
    """
    items = (record.get("disclosure") or {}).get("items") or []
    for item in items:
        if item.get("kind") == FACTS_KIND and item.get("source") == EARNINGS_CALL_SOURCE:
            return list(item.get("content") or [])
    return []
