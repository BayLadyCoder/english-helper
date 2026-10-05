"""Tests for prompt building and JSON validation. The model is mocked."""

import json

import pytest

from llm_client import (
    LLMUnavailableError,
    MalformedLLMResponseError,
    _normalize_pos,
    build_prompt,
    call_model,
    lookup,
    parse_lookup,
    adjust_tone,
)

EXAMPLES = [
    "I ride a bicycle.",
    "The bicycle is red.",
    "She locked her bicycle.",
]
NOUN_SENSE = {
    "part_of_speech": "noun",
    "definition": "A two-wheeled vehicle.",
    "synonyms": ["bike", "cycle"],
    "antonyms": [],
    "examples": EXAMPLES,
}
VALID = {
    "senses": [NOUN_SENSE],
    "sentiment": "neutral",
}


def _payload(**overrides: object) -> str:
    """Return VALID as JSON with the given top-level overrides applied."""
    return json.dumps({**VALID, **overrides})


def _sense(**overrides: object) -> dict:
    """Return the noun sense with the given overrides applied."""
    return {**NOUN_SENSE, **overrides}


def test_parse_valid_json() -> None:
    """Accept a payload that has every required key and the right types."""
    parsed = parse_lookup(json.dumps(VALID))
    assert parsed["senses"][0]["definition"] == NOUN_SENSE["definition"]
    assert parsed["senses"][0]["examples"] == EXAMPLES


def test_multiple_parts_of_speech_are_kept() -> None:
    """Keep one sense per part of speech, each with its own examples."""
    verb = _sense(
        part_of_speech="verb",
        definition="To travel by bicycle.",
        examples=[
            "We bicycle to work.",
            "They bicycled home.",
            "He bicycles daily.",
        ],
    )
    parsed = parse_lookup(_payload(senses=[NOUN_SENSE, verb]))
    assert [sense["part_of_speech"] for sense in parsed["senses"]] == ["noun", "verb"]
    assert len(parsed["senses"][1]["examples"]) == 3


def test_duplicate_part_of_speech_is_dropped() -> None:
    """Keep the first sense when the model repeats a part of speech."""
    parsed = parse_lookup(_payload(senses=[NOUN_SENSE, _sense(definition="Another bike.")]))
    assert len(parsed["senses"]) == 1


def test_verb_noun_example_is_rejected() -> None:
    """Reject a verb sense whose example uses the word as a noun."""
    verb = _sense(
        part_of_speech="verb",
        definition="To brighten something.",
        examples=[
            "The light filled the room.",
            "She lit the candle.",
            "He lightened the mood.",
        ],
    )
    with pytest.raises(MalformedLLMResponseError, match="as a noun"):
        parse_lookup(_payload(senses=[verb]), "light")


def test_lenient_parse_keeps_good_verb_examples() -> None:
    """Drop only the noun-use sentence when strict checking is off."""
    verb = _sense(
        part_of_speech="verb",
        definition="To brighten something.",
        examples=[
            "The light filled the room.",
            "She lit the candle.",
            "He lightened the mood.",
        ],
    )
    parsed = parse_lookup(_payload(senses=[verb]), "light", False)
    assert parsed["senses"][0]["examples"] == [
        "She lit the candle.",
        "He lightened the mood.",
    ]


def test_light_requires_noun_verb_adjective() -> None:
    """Reject a light lookup that omits one of the three required senses."""
    verb = _sense(
        part_of_speech="verb",
        definition="To set fire to something.",
        examples=["She lit the candle.", "He lit the stove.", "They lit the lamp."],
    )
    adjective = _sense(
        part_of_speech="adjective",
        definition="Not heavy.",
        examples=["The bag felt light.", "She wore a light coat.", "The box was light."],
    )
    with pytest.raises(MalformedLLMResponseError, match="Missing: adjective"):
        parse_lookup(_payload(senses=[NOUN_SENSE, verb]), "light")
    parsed = parse_lookup(_payload(senses=[NOUN_SENSE, verb, adjective]), "light")
    assert {_normalize_pos(s["part_of_speech"]) for s in parsed["senses"]} == {
        "noun",
        "verb",
        "adjective",
    }


def test_empty_senses_raises() -> None:
    """Reject a payload with no parts of speech."""
    with pytest.raises(MalformedLLMResponseError):
        parse_lookup(_payload(senses=[]))


def test_sense_missing_definition_raises() -> None:
    """Reject a sense that has no definition."""
    broken = dict(NOUN_SENSE)
    del broken["definition"]
    with pytest.raises(MalformedLLMResponseError):
        parse_lookup(_payload(senses=[broken]))


def test_missing_key_raises() -> None:
    """Reject a payload that omits a required key."""
    payload = dict(VALID)
    del payload["sentiment"]
    with pytest.raises(MalformedLLMResponseError):
        parse_lookup(json.dumps(payload))


def test_non_json_raises() -> None:
    """Reject text that is not JSON."""
    with pytest.raises(MalformedLLMResponseError):
        parse_lookup("not json")


def test_string_where_list_required_raises() -> None:
    """Reject a string in a field that must be a list."""
    with pytest.raises(MalformedLLMResponseError):
        parse_lookup(_payload(senses=[_sense(synonyms="bike")]))


def test_two_examples_fail() -> None:
    """Reject fewer than three example sentences."""
    with pytest.raises(MalformedLLMResponseError):
        parse_lookup(_payload(senses=[_sense(examples=EXAMPLES[:2])]))


def test_four_examples_keep_first_three() -> None:
    """Keep only the first three examples when the model returns more."""
    extra = "Four."
    parsed = parse_lookup(_payload(senses=[_sense(examples=[*EXAMPLES, extra])]))
    assert parsed["senses"][0]["examples"] == EXAMPLES


def test_blank_example_fails() -> None:
    """Reject an example list that includes a blank sentence."""
    with pytest.raises(MalformedLLMResponseError):
        parse_lookup(_payload(senses=[_sense(examples=["", EXAMPLES[1], EXAMPLES[2]])]))


def test_prompt_contains_the_lookup() -> None:
    """Put the lookup text in the prompt and nothing about a target language."""
    prompt = build_prompt("bicycle")
    assert "bicycle" in prompt
    assert "Target language" not in prompt


def test_lookup_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call the model a second time after an invalid first payload."""
    calls = {"n": 0}

    def fake_call(prompt: str, *, json_mode: bool = False) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return json.dumps(VALID)

    monkeypatch.setattr("llm_client.call_model", fake_call)
    result = lookup("bicycle")
    assert calls["n"] == 2
    assert result["senses"][0]["definition"] == NOUN_SENSE["definition"]


def test_strips_terminal_color_codes() -> None:
    """Drop ANSI color codes the model sometimes wraps around strings."""
    colored = [
        "\x1b[34mThe bank is closed.\x1b[0m",
        EXAMPLES[1],
        EXAMPLES[2],
    ]
    parsed = parse_lookup(_payload(senses=[_sense(examples=colored)]))
    assert parsed["senses"][0]["examples"][0] == "The bank is closed."


def test_adjust_tone_returns_rewritten_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the model rewrite for a supported tone."""

    def fake_call(prompt: str, *, json_mode: bool = False) -> str:
        assert "professional" in prompt.casefold()
        assert "Thanks for the update" in prompt
        assert json_mode is False
        return " Thank you for the update. "

    monkeypatch.setattr("llm_client.call_model", fake_call)
    assert adjust_tone("Thanks for the update", "Professional") == "Thank you for the update."


def test_adjust_tone_wraps_profane_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mark profane input as text to rewrite and drop tags the model echoes back."""

    def fake_call(prompt: str, *, json_mode: bool = False) -> str:
        assert "<text>What the fuck?</text>" in prompt
        assert "never refuse" in prompt
        return "<text>What is going on?</text>"

    monkeypatch.setattr("llm_client.call_model", fake_call)
    assert adjust_tone("What the fuck?", "Professional") == "What is going on?"


def test_adjust_tone_adds_guidance_only_for_professional(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the American workplace guidance only with the professional tone."""
    prompts = []

    def fake_call(prompt: str, *, json_mode: bool = False) -> str:
        prompts.append(prompt)
        return "ok"

    monkeypatch.setattr("llm_client.call_model", fake_call)
    adjust_tone("hi", "Professional")
    adjust_tone("hi", "Casual")
    assert "American company" in prompts[0]
    assert "American company" not in prompts[1]


def test_adjust_tone_rejects_unknown_tone() -> None:
    """Refuse a tone that is not one of the offered options."""
    with pytest.raises(ValueError):
        adjust_tone("Hello", "sarcastic")
    with pytest.raises(ValueError):
        adjust_tone("Hello", "Formal")


def test_call_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn a closed Ollama port into a friendly unavailable error."""
    monkeypatch.setattr("llm_client.OLLAMA_HOST", "http://127.0.0.1:1")
    with pytest.raises(LLMUnavailableError, match="Is Ollama running?"):
        call_model("hello")


def test_lookup_stops_after_three_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give up after three invalid responses instead of calling the model again."""
    calls = {"n": 0, "prompts": []}

    def fake_call(prompt: str, *, json_mode: bool = False) -> str:
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return "still not json"

    monkeypatch.setattr("llm_client.call_model", fake_call)
    with pytest.raises(MalformedLLMResponseError):
        lookup("Home")
    assert calls["n"] == 3
    assert "non-empty" in calls["prompts"][1]
