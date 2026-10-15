"""Tests for prompt building and JSON validation. The model is mocked."""

import json

import pytest

from llm_client import (
    LLMUnavailableError,
    MalformedLLMResponseError,
    POS_SYSTEM_PROMPT,
    _normalize_pos,
    build_prompt,
    call_model,
    generate_study,
    list_parts_of_speech,
    lookup,
    parse_lookup,
    parse_parts_of_speech,
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


def _fast_senses() -> list[dict]:
    """Return one sense for each of the four parts of speech of 'fast'."""
    return [
        _sense(
            part_of_speech="noun",
            definition="A period of going without food.",
            examples=["They began a three-day fast.", "The fast lasted a week.", "She ended her fast."],
        ),
        _sense(
            part_of_speech="verb",
            definition="To go without food.",
            examples=["They fast during Ramadan.", "She fasted yesterday.", "He fasts every Friday."],
        ),
        _sense(
            part_of_speech="adjective",
            definition="Moving or happening quickly.",
            examples=["She caught a fast train.", "The car was fast.", "He is a fast runner."],
        ),
        _sense(
            part_of_speech="adverb",
            definition="Quickly.",
            examples=["He runs fast.", "She drove too fast.", "The news spread fast."],
        ),
    ]


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


def test_fast_requires_noun_verb_adjective_adverb() -> None:
    """Reject a fast lookup that omits one of the four required senses."""
    senses = _fast_senses()
    with pytest.raises(MalformedLLMResponseError, match="Missing: adverb"):
        parse_lookup(_payload(senses=senses[:3]), "fast")
    parsed = parse_lookup(_payload(senses=senses), "fast")
    assert [sense["part_of_speech"] for sense in parsed["senses"]] == [
        "noun",
        "verb",
        "adjective",
        "adverb",
    ]


def test_normalize_pos_distinguishes_adverb_from_verb() -> None:
    """Keep adverb as adverb; 'verb' is a substring of 'adverb'."""
    assert _normalize_pos("adverb") == "adverb"
    assert _normalize_pos("Adverb") == "adverb"
    assert _normalize_pos("verb") == "verb"


def test_parse_parts_of_speech_orders_and_dedupes() -> None:
    """Keep known labels in noun-verb-adjective-adverb order."""
    raw = json.dumps({"parts_of_speech": ["Adverb", "noun", "verb", "noun", "adjective"]})
    assert parse_parts_of_speech(raw) == ["noun", "verb", "adjective", "adverb"]


def test_parse_parts_of_speech_rejects_empty() -> None:
    """Reject a parts-of-speech payload with no usable labels."""
    with pytest.raises(MalformedLLMResponseError):
        parse_parts_of_speech(json.dumps({"parts_of_speech": []}))
    with pytest.raises(MalformedLLMResponseError):
        parse_parts_of_speech("not json")


def test_list_parts_of_speech_for_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ask the model for parts of speech and keep all four for 'fast'."""
    prompts: list[str] = []

    def fake_call(prompt: str, *, json_mode: bool = False, **kwargs: object) -> str:
        prompts.append(prompt)
        assert json_mode is True
        assert kwargs["system_prompt"] == POS_SYSTEM_PROMPT
        return json.dumps({"parts_of_speech": ["adjective", "adverb", "noun", "verb"]})

    monkeypatch.setattr("llm_client.call_model", fake_call)
    assert list_parts_of_speech("fast") == ["noun", "verb", "adjective", "adverb"]
    assert "fast" in prompts[0]


def test_list_parts_of_speech_fills_in_known_senses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a required sense when the model omits it for a known word."""
    monkeypatch.setattr(
        "llm_client.call_model",
        lambda *_args, **_kwargs: json.dumps({"parts_of_speech": ["noun", "verb"]}),
    )
    assert list_parts_of_speech("light") == ["noun", "verb", "adjective"]
    assert list_parts_of_speech("fast") == ["noun", "verb", "adjective", "adverb"]


def test_list_parts_of_speech_skips_phrases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not ask the model for parts of speech of a multi-word phrase."""

    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("phrases should skip the parts-of-speech call")

    monkeypatch.setattr("llm_client.call_model", fail)
    assert list_parts_of_speech("break the ice") == []


def test_list_parts_of_speech_falls_back_when_the_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the known list, or nothing, when parts-of-speech JSON is unusable."""
    monkeypatch.setattr("llm_client.call_model", lambda *_args, **_kwargs: "not json")
    assert list_parts_of_speech("bicycle") == []
    assert list_parts_of_speech("fast") == ["noun", "verb", "adjective", "adverb"]


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
    assert "parts of speech" not in prompt


def test_prompt_names_required_parts_of_speech() -> None:
    """Tell the details step which parts of speech to cover."""
    prompt = build_prompt("fast", ["noun", "verb", "adjective", "adverb"])
    assert "noun, verb, adjective, adverb" in prompt
    assert "exactly 4 sense" in prompt


def test_lookup_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Call the model a second time after an invalid first payload."""
    calls = {"n": 0}

    def fake_call(prompt: str, *, json_mode: bool = False) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return json.dumps(VALID)

    monkeypatch.setattr("llm_client.list_parts_of_speech", lambda _text: ["noun"])
    monkeypatch.setattr("llm_client.call_model", fake_call)
    result = lookup("bicycle")
    assert calls["n"] == 2
    assert result["senses"][0]["definition"] == NOUN_SENSE["definition"]


def test_lookup_checks_parts_of_speech_before_details(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ask for parts of speech first, then require those senses in the details."""
    calls: list[str] = []
    payload = {**VALID, "senses": _fast_senses()}

    def fake_call(
        prompt: str,
        *,
        json_mode: bool = False,
        system_prompt: str | None = None,
        **_kwargs: object,
    ) -> str:
        calls.append(system_prompt or "")
        if system_prompt == POS_SYSTEM_PROMPT:
            return json.dumps({"parts_of_speech": ["noun", "verb", "adjective", "adverb"]})
        assert "noun, verb, adjective, adverb" in prompt
        return json.dumps(payload)

    monkeypatch.setattr("llm_client.call_model", fake_call)
    result = lookup("fast")
    assert calls[0] == POS_SYSTEM_PROMPT
    assert calls[1] != POS_SYSTEM_PROMPT
    assert [sense["part_of_speech"] for sense in result["senses"]] == [
        "noun",
        "verb",
        "adjective",
        "adverb",
    ]


def test_discovered_parts_are_required() -> None:
    """Reject details that skip a part of speech the first step found."""
    with pytest.raises(MalformedLLMResponseError, match="Missing: verb"):
        parse_lookup(_payload(senses=[NOUN_SENSE]), "run", expected_parts=["noun", "verb"])


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

    monkeypatch.setattr("llm_client.list_parts_of_speech", lambda _text: ["noun"])
    monkeypatch.setattr("llm_client.call_model", fake_call)
    with pytest.raises(MalformedLLMResponseError):
        lookup("Home")
    assert calls["n"] == 3
    assert "non-empty" in calls["prompts"][1]


def test_generate_study_story_uses_every_word(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send each word to the model and return the story in one call."""
    prompts: list[str] = []

    def fake_call(prompt: str, **_: object) -> str:
        prompts.append(prompt)
        return "Mia lit a candle and felt brave."

    monkeypatch.setattr("llm_client.call_model", fake_call)
    assert generate_study(["light", "brave"], "Story") == "Mia lit a candle and felt brave."
    assert len(prompts) == 1
    assert "- light\n- brave" in prompts[0]
    assert "story" in prompts[0]


def test_generate_study_conversation_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ask for speaker lines when a conversation is chosen."""
    prompts: list[str] = []
    reply = "Anna: Did you light the candle?\nBen: Yes, I was brave."

    def fake_call(prompt: str, **_: object) -> str:
        prompts.append(prompt)
        return reply

    monkeypatch.setattr("llm_client.call_model", fake_call)
    assert generate_study(["light", "brave"], "Conversation") == reply
    assert "Anna: ..." in prompts[0]
    assert "in the conversation" in prompts[0]


def test_generate_study_retries_with_skipped_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ask again, naming the words the first answer left out."""
    replies = iter(["Mia felt brave.", "Mia lit a candle and felt brave."])
    prompts: list[str] = []

    def fake_call(prompt: str, **_: object) -> str:
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr("llm_client.call_model", fake_call)
    assert generate_study(["light", "brave"], "Story") == "Mia lit a candle and felt brave."
    assert '"light"' in prompts[1]


def test_generate_study_strips_angle_brackets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the angle brackets the model sometimes puts around the studied words."""
    monkeypatch.setattr(
        "llm_client.call_model",
        lambda *_args, **_kwargs: "Mia found <peace of mind> and felt <brave>.",
    )
    assert generate_study(["peace of mind", "brave"], "Story") == "Mia found peace of mind and felt brave."


@pytest.mark.parametrize("words, style", [(["  "], "Story"), (["light"], "Poem")])
def test_generate_study_rejects_bad_input(words: list[str], style: str) -> None:
    """Refuse to call the model without words or with an unknown format."""
    with pytest.raises(ValueError):
        generate_study(words, style)


def test_generate_study_empty_reply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat a blank model reply as malformed."""
    monkeypatch.setattr("llm_client.call_model", lambda *_args, **_kwargs: "  ")
    with pytest.raises(MalformedLLMResponseError):
        generate_study(["light"], "Conversation")
