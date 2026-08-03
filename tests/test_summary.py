"""The summarization port — prompt shape, request construction, and fact parsing.

The parser cases mirror the production test suite one-for-one; if one of these
fails, the port has drifted from the pipeline.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from examples.archive import read_jsonl_gz
from examples.summary import (
    MODEL_CONFIGS,
    N_FACTS,
    SUMMARY_MODEL,
    SUMMARY_SYSTEM_PROMPT,
    build_batch_request,
    build_request,
    build_summary_user_prompt,
    facts_from_disclosure,
    format_transcript,
    parse_facts,
    recover_facts,
    summarize_transcript,
)
from tests.conftest import SAMPLE_ARCHIVE, SAMPLE_SUMMARY, SAMPLE_TRANSCRIPT


def _facts(n: int) -> list[str]:
    return [f"Fact number {i} with revenue of {i} million." for i in range(1, n + 1)]


def _payload(facts: list[str]) -> str:
    return json.dumps({"facts": facts})


# ----- parse_facts (strict) --------------------------------------------------


def test_parse_facts_exactly_ten() -> None:
    facts = _facts(N_FACTS)
    assert parse_facts(_payload(facts)) == facts


def test_parse_facts_strips_markdown_fences() -> None:
    facts = _facts(N_FACTS)
    fenced = "```json\n" + _payload(facts) + "\n```"
    assert parse_facts(fenced) == facts


def test_parse_facts_extracts_object_from_surrounding_prose() -> None:
    facts = _facts(N_FACTS)
    text = "Here you go:\n" + _payload(facts) + "\nHope that helps!"
    assert parse_facts(text) == facts


def test_parse_facts_wrong_count_raises() -> None:
    with pytest.raises(ValueError):
        parse_facts(_payload(_facts(9)))


def test_parse_facts_missing_key_raises() -> None:
    with pytest.raises(ValueError):
        parse_facts(json.dumps({"items": _facts(N_FACTS)}))


def test_parse_facts_empty_fact_raises() -> None:
    facts = _facts(N_FACTS)
    facts[3] = "   "
    with pytest.raises(ValueError):
        parse_facts(_payload(facts))


def test_parse_facts_garbage_raises() -> None:
    with pytest.raises(ValueError):
        parse_facts("not json at all")


# ----- recover_facts (lenient — this is what the live pipeline runs) ---------


def test_recover_facts_strict_passthrough() -> None:
    facts = _facts(N_FACTS)
    got, note = recover_facts(_payload(facts))
    assert got == facts
    assert note == "strict"


def test_recover_facts_trims_when_too_many() -> None:
    facts = _facts(12)
    got, note = recover_facts(_payload(facts))
    assert got == facts[:N_FACTS]
    assert "trimmed_from_12" in note


def test_recover_facts_repairs_missing_closing_bracket() -> None:
    # Model emitted the array without its closing ']' (ends '..."}' not '..."]}').
    facts = _facts(N_FACTS)
    broken = '{"facts": [' + ", ".join(json.dumps(f) for f in facts) + "}"
    got, note = recover_facts(broken)
    assert got == facts
    assert "repaired_close" in note


def test_recover_facts_too_few_is_unrecoverable() -> None:
    got, note = recover_facts(_payload(_facts(8)))
    assert got is None
    assert note == "too_few:8"


def test_recover_facts_empty_fact_is_unrecoverable() -> None:
    facts = _facts(N_FACTS)
    facts[0] = ""
    got, note = recover_facts(_payload(facts))
    assert got is None
    assert note == "empty_fact"


def test_recover_facts_no_json_is_unrecoverable() -> None:
    got, note = recover_facts("the model refused to answer")
    assert got is None
    assert note == "unparseable"


def test_recover_facts_truncated_mid_string_is_rejected() -> None:
    # The bracket repair must not "fix" an output that was cut off inside a fact,
    # which would silently publish a mangled final sentence.
    facts = _facts(N_FACTS)
    truncated = '{"facts": [' + ", ".join(json.dumps(f) for f in facts)[:-8]
    got, note = recover_facts(truncated)
    assert got is None
    assert note == "unparseable"


def test_recover_facts_facts_not_a_list() -> None:
    got, note = recover_facts(json.dumps({"facts": "ten facts"}))
    assert got is None
    assert note == "no_facts_list"


# ----- The prompt ------------------------------------------------------------


def test_system_prompt_is_verbatim() -> None:
    assert SUMMARY_SYSTEM_PROMPT == (
        "You are a financial analyst. You extract the most investor-relevant facts from "
        "earnings call transcripts to help investors anticipate the stock's reaction to the call."
    )


def test_user_prompt_embeds_transcript_in_tags() -> None:
    prompt = build_summary_user_prompt("# Transcript of Example\n\n**A (Executives):** hi\n")
    assert "<transcript>\n# Transcript of Example" in prompt
    assert prompt.rstrip().endswith("</transcript>")


def test_user_prompt_states_the_ten_fact_contract() -> None:
    prompt = build_summary_user_prompt("body")
    assert "extract the 10 most investor-relevant facts" in prompt
    assert "The list must contain exactly 10 facts." in prompt


def test_user_prompt_emits_literal_json_braces() -> None:
    # Guards the str.format escaping: the template's {{...}} must survive as a
    # literal JSON example rather than being interpolated away.
    prompt = build_summary_user_prompt("body")
    assert '{"facts": ["<fact 1>", "<fact 2>", ..., "<fact 10>"]}' in prompt


# ----- The request -----------------------------------------------------------


def test_build_request_default_model_is_adaptive_with_high_effort() -> None:
    params = build_request("body")
    assert params["model"] == SUMMARY_MODEL
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["output_config"] == {"effort": "high"}
    assert params["max_tokens"] == 16000


def test_build_request_opus_4_5_uses_manual_thinking_and_no_output_config() -> None:
    params = build_request("body", "claude-opus-4-5")
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "output_config" not in params


def test_build_request_has_no_structured_output_controls() -> None:
    # Structure is enforced by the prompt text and the parsers, not by the API.
    for model in MODEL_CONFIGS:
        params = build_request("body", model)
        assert set(params) <= {
            "model",
            "max_tokens",
            "thinking",
            "system",
            "messages",
            "output_config",
        }


def test_build_request_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="No config for model"):
        build_request("body", "claude-opus-9-9")


def test_build_batch_request_wraps_params_with_custom_id() -> None:
    batch = build_batch_request("12345", "body")
    assert batch["custom_id"] == "12345"
    assert batch["params"] == build_request("body")


# ----- The transcript renderer ----------------------------------------------

_TRANSCRIPT = {
    "header": {"title": "Example Corp., Q2 2026 Earnings Call, Jul 28, 2026"},
    "components": [
        {"speakerName": "Operator", "speakerType": "Operator", "text": "Welcome."},
        {"speakerName": "Jane Doe", "speakerType": "Executives", "text": "Revenue was $1B."},
    ],
}


def test_format_transcript_markdown_prepends_only_the_header_title() -> None:
    md = format_transcript(_TRANSCRIPT)
    assert md.startswith("# Transcript of Example Corp., Q2 2026 Earnings Call, Jul 28, 2026\n\n")
    assert "**Operator (Operator):** Welcome.\n\n" in md
    assert "**Jane Doe (Executives):** Revenue was $1B.\n\n" in md
    # No ticker/date/fiscal metadata is injected beyond what the title carries.
    assert "EXMP" not in md


def test_format_transcript_plain_joins_component_text_only() -> None:
    assert format_transcript(_TRANSCRIPT, markdown=False) == "Welcome.\n\nRevenue was $1B."


# ----- Running the call (stubbed client) ------------------------------------


class _Block:
    def __init__(self, type_: str, **kw: Any) -> None:
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    input_tokens = 1234
    output_tokens = 567


class _Response:
    stop_reason = "end_turn"

    def __init__(self, text: str, thinking: str | None = None) -> None:
        self.content = [_Block("text", text=text)]
        if thinking is not None:
            self.content.insert(0, _Block("thinking", thinking=thinking))
        self.usage = _Usage()


class _Messages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _Response:
        self.last_kwargs = kwargs
        return self._response


class _StubClient:
    def __init__(self, response: _Response) -> None:
        self.messages = _Messages(response)


def test_summarize_transcript_success() -> None:
    facts = _facts(N_FACTS)
    client = _StubClient(_Response(_payload(facts), thinking="Weighing the guidance raise."))
    result = summarize_transcript("# Transcript of X\n\n", client)

    assert result.ok
    assert result.facts == facts
    assert result.parse_note == "strict"
    assert result.thinking_summary == "Weighing the guidance raise."
    assert (result.input_tokens, result.output_tokens) == (1234, 567)
    assert result.stop_reason == "end_turn"
    # The call went out with exactly the ported request.
    assert client.messages.last_kwargs == build_request("# Transcript of X\n\n")


def test_summarize_transcript_parse_failure_does_not_raise() -> None:
    client = _StubClient(_Response(_payload(_facts(8))))
    result = summarize_transcript("body", client)

    assert not result.ok
    assert result.facts is None
    assert result.parse_note == "too_few:8"
    assert result.raw_response  # the response is kept for inspection


def test_summarize_transcript_without_thinking_blocks() -> None:
    client = _StubClient(_Response(_payload(_facts(N_FACTS))))
    assert summarize_transcript("body", client).thinking_summary is None


# ----- Reading what the API delivers ----------------------------------------


def test_facts_from_disclosure_reads_the_sample_archive() -> None:
    records = list(read_jsonl_gz(SAMPLE_ARCHIVE))
    assert records
    for record in records:
        facts = facts_from_disclosure(record)
        assert facts
        assert all(isinstance(f, str) and f for f in facts)


def test_facts_from_disclosure_ignores_other_kinds_and_sources() -> None:
    record = {
        "disclosure": {
            "items": [
                {"kind": "facts", "source": "press_release", "content": ["nope"]},
                {"kind": "transcript", "source": "earnings_call", "content": ["nope"]},
                {"kind": "facts", "source": "earnings_call", "content": ["yes"]},
            ]
        }
    }
    assert facts_from_disclosure(record) == ["yes"]


def test_facts_from_disclosure_missing_is_empty() -> None:
    assert facts_from_disclosure({}) == []
    assert facts_from_disclosure({"disclosure": None}) == []
    assert facts_from_disclosure({"disclosure": {"items": []}}) == []


# ----- The bundled samples the notebook runs on -----------------------------


def test_sample_transcript_renders_and_is_prompt_ready() -> None:
    transcript = json.loads(SAMPLE_TRANSCRIPT.read_text())
    md = format_transcript(transcript)
    assert md.startswith("# Transcript of Northwind Logistics")
    assert "**Priya Anand (Executives):**" in md
    # The whole rendered call must fit in one user message — no chunking exists.
    assert build_summary_user_prompt(md).count("<transcript>") == 1


def test_sample_summary_matches_the_published_artifact_shape() -> None:
    artifact = json.loads(SAMPLE_SUMMARY.read_text())
    assert artifact["event_id"] == artifact["metadata"]["event_id"]
    assert artifact["response"]["parse_note"] == "strict"
    facts = artifact["response"]["facts"]
    assert len(facts) == N_FACTS
    assert all(isinstance(f, str) and f.strip() for f in facts)
    # A published artifact must survive the same parser the pipeline applies.
    recovered, note = recover_facts(json.dumps({"facts": facts}))
    assert recovered == facts
    assert note == "strict"
