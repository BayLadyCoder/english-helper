"""Tests for grammar-check parsing and retries. The model is mocked."""

import json

import pytest

from llm_client import (
    GRAMMAR_SYSTEM_PROMPT,
    SPLIT_SYSTEM_PROMPT,
    MalformedLLMResponseError,
    _is_lumped,
    check_grammar,
    parse_grammar,
)

TEXT = "She go to school yesterday and buy a apple."
CORRECTED = "She went to school yesterday and bought an apple."
CORRECTIONS = [
    {"original": "go", "correction": "went", "explanation": "Use the past tense for a finished action."},
    {"original": "buy", "correction": "bought", "explanation": "Use the past tense for a finished action."},
    {"original": "a apple", "correction": "an apple", "explanation": "Use 'an' before a vowel sound."},
]


def _payload(**overrides: object) -> str:
    """Return a valid grammar response as JSON with the given overrides applied."""
    return json.dumps({"corrected_text": CORRECTED, "corrections": CORRECTIONS, **overrides})


def test_parse_valid_grammar() -> None:
    """Return the corrected text and every listed fix."""
    parsed = parse_grammar(_payload(), TEXT)
    assert parsed["corrected_text"] == CORRECTED
    assert [c["correction"] for c in parsed["corrections"]] == ["went", "bought", "an apple"]
    assert parsed["changed"] is True


def test_correct_text_has_no_fixes() -> None:
    """Report no changes when the model returns the text as it was."""
    parsed = parse_grammar(_payload(corrected_text=CORRECTED, corrections=[]), CORRECTED)
    assert parsed == {"corrected_text": CORRECTED, "corrections": [], "changed": False}


def test_whitespace_only_difference_is_not_a_change() -> None:
    """Ignore extra spaces when deciding whether the text changed."""
    parsed = parse_grammar(_payload(corrected_text="I  am  fine.", corrections=[]), "I am fine.")
    assert parsed["changed"] is False


def test_drops_noop_and_invented_corrections() -> None:
    """Drop fixes that change nothing or quote words missing from the text."""
    corrections = [
        *CORRECTIONS,
        {"original": "school", "correction": "school", "explanation": "Same."},
        {"original": "teacher", "correction": "teachers", "explanation": "Not in the text."},
        {"original": "", "correction": "the", "explanation": "Empty original."},
    ]
    parsed = parse_grammar(_payload(corrections=corrections), TEXT)
    assert len(parsed["corrections"]) == 3


def test_duplicate_corrections_are_dropped() -> None:
    """Keep one copy when the model repeats a fix."""
    parsed = parse_grammar(_payload(corrections=[CORRECTIONS[0], CORRECTIONS[0]]), TEXT)
    assert len(parsed["corrections"]) == 1


def test_listed_fixes_missing_from_corrected_text_are_applied() -> None:
    """Apply fixes the model listed but left out of corrected_text."""
    parsed = parse_grammar(_payload(corrected_text=TEXT), TEXT)
    assert parsed["corrected_text"] == CORRECTED
    removal = {"original": "am agree", "correction": "agree", "explanation": "No 'am'."}
    parsed = parse_grammar(
        _payload(corrected_text="I am agree with you.", corrections=[removal]),
        "I am agree with you.",
    )
    assert parsed["corrected_text"] == "I agree with you."


def test_applying_fixes_respects_word_boundaries() -> None:
    """Replace only the whole wrong word, not the same letters inside another word."""
    fix = {"original": "go", "correction": "went", "explanation": "Past tense."}
    text = "It was a good day and I go home."
    parsed = parse_grammar(_payload(corrected_text=text, corrections=[fix]), text)
    assert parsed["corrected_text"] == "It was a good day and I went home."


def test_unapplied_insertion_is_rejected() -> None:
    """Ask again when an inserted-word fix is listed but the text is unchanged."""
    text = "I bought apple."
    insertion = {"original": "apple", "correction": "an apple", "explanation": "Add 'an'."}
    raw = _payload(corrected_text=text, corrections=[insertion])
    with pytest.raises(MalformedLLMResponseError, match="identical"):
        parse_grammar(raw, text)
    lenient = parse_grammar(raw, text, strict=False)
    assert len(lenient["corrections"]) == 1


def test_strips_echoed_text_tags() -> None:
    """Drop the <text> tags the model sometimes copies into its answer."""
    parsed = parse_grammar(_payload(corrected_text=f"<text>{CORRECTED}</text>"), TEXT)
    assert parsed["corrected_text"] == CORRECTED


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        json.dumps({"corrected_text": CORRECTED}),
        _payload(corrected_text=""),
        _payload(corrections="go -> went"),
        _payload(corrections=[{"original": "go", "correction": "went"}]),
        _payload(corrections=[{"original": "go", "correction": 1, "explanation": ""}]),
    ],
)
def test_malformed_grammar_raises(raw: str) -> None:
    """Reject responses that do not match the grammar JSON shape."""
    with pytest.raises(MalformedLLMResponseError):
        parse_grammar(raw, TEXT)


def test_check_grammar_retries_with_grammar_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send the grammar system prompt and retry after an invalid response."""
    calls: list[dict] = []

    def fake_call(prompt: str, **kwargs: object) -> str:
        calls.append({"prompt": prompt, **kwargs})
        return "not json" if len(calls) == 1 else _payload()

    monkeypatch.setattr("llm_client.call_model", fake_call)
    result = check_grammar(TEXT)
    assert result["corrected_text"] == CORRECTED
    assert calls[0]["prompt"] == f"<text>{TEXT}</text>"
    assert calls[0]["json_mode"] is True
    assert calls[0]["system_prompt"] == GRAMMAR_SYSTEM_PROMPT
    assert "previous response was invalid" in calls[1]["prompt"]
    assert calls[2]["prompt"] == f"<text>{CORRECTED}</text>"
    assert len(calls) == 3


def test_check_grammar_rechecks_and_merges_missed_fixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check corrected text again and add the mistake the first pass skipped."""
    text = "Yesterday I go to the office. The team have spent weeks on it."
    first = "Yesterday I went to the office. The team have spent weeks on it."
    final = "Yesterday I went to the office. The team has spent weeks on it."
    go = {"original": "I go", "correction": "I went", "explanation": "Past."}
    have = {"original": "team have", "correction": "team has", "explanation": "Singular."}
    responses = {
        text: json.dumps({"corrected_text": first, "corrections": [go]}),
        first: json.dumps({"corrected_text": final, "corrections": [have]}),
        final: json.dumps({"corrected_text": final, "corrections": []}),
    }
    prompts: list[str] = []

    def fake_call(prompt: str, **kwargs: object) -> str:
        prompts.append(prompt)
        return responses[prompt.removeprefix("<text>").removesuffix("</text>")]

    monkeypatch.setattr("llm_client.call_model", fake_call)
    result = check_grammar(text)
    assert result["corrected_text"] == final
    assert [c["correction"] for c in result["corrections"]] == ["I went", "team has"]
    assert result["changed"] is True
    assert len(prompts) == 3


def test_check_grammar_clean_text_needs_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop after one pass when the text has no mistakes."""
    calls = {"n": 0}

    def fake_call(prompt: str, **kwargs: object) -> str:
        calls["n"] += 1
        return json.dumps({"corrected_text": CORRECTED, "corrections": []})

    monkeypatch.setattr("llm_client.call_model", fake_call)
    result = check_grammar(CORRECTED)
    assert result == {"corrected_text": CORRECTED, "corrections": [], "changed": False}
    assert calls["n"] == 1


def test_check_grammar_stops_after_three_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap the re-checks when the model keeps changing the text."""
    calls = {"n": 0}

    def fake_call(prompt: str, **kwargs: object) -> str:
        calls["n"] += 1
        return json.dumps({"corrected_text": f"Version {calls['n']}.", "corrections": []})

    monkeypatch.setattr("llm_client.call_model", fake_call)
    assert check_grammar(TEXT)["corrected_text"] == "Version 3."
    assert calls["n"] == 3


def test_failed_recheck_keeps_earlier_fixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the first pass when a later re-check only gets malformed answers."""
    calls = {"n": 0}

    def fake_call(prompt: str, **kwargs: object) -> str:
        calls["n"] += 1
        return _payload() if calls["n"] == 1 else "not json"

    monkeypatch.setattr("llm_client.call_model", fake_call)
    result = check_grammar(TEXT)
    assert result["corrected_text"] == CORRECTED
    assert len(result["corrections"]) == 3


@pytest.mark.parametrize(
    ("original", "correction", "lumped"),
    [
        ("Me and him goes to meeting.", "He and I go to the meeting.", True),
        ("and he have went to store", "and he went to the store", True),
        ("My sister and me was", "My sister and I were", True),
        ("Me and him", "He and I", False),
        ("If I would have known", "If I had known", False),
        ("I bought apple", "I bought an apple", False),
    ],
)
def test_is_lumped(original: str, correction: str, lumped: bool) -> None:
    """Flag long fixes that change several separate places, not one-rule fixes."""
    fix = {"original": original, "correction": correction, "explanation": ""}
    assert _is_lumped(fix) is lumped


SPLIT_TEXT = "Me and him goes to meeting."
SPLIT_CORRECTED = "He and I go to the meeting."
SPLIT_PARTS = [
    {"original": "Me and him", "correction": "He and I", "explanation": "Pronoun case."},
    {"original": "goes", "correction": "go", "explanation": "Plural subject."},
    {"original": "to meeting", "correction": "to the meeting", "explanation": "Add 'the'."},
]


def _split_fake(split_answers: list[object], calls: list[dict]):
    """Return a fake model: lumped grammar answers, then the given split answers."""
    lumped = {"original": SPLIT_TEXT, "correction": SPLIT_CORRECTED, "explanation": "Fixed."}
    answers = iter(split_answers)

    def fake_call(prompt: str, **kwargs: object) -> str:
        calls.append({"prompt": prompt, **kwargs})
        if kwargs.get("system_prompt") == SPLIT_SYSTEM_PROMPT:
            answer = next(answers)
            return answer if isinstance(answer, str) else json.dumps({"corrections": answer})
        if prompt == f"<text>{SPLIT_TEXT}</text>":
            return json.dumps({"corrected_text": SPLIT_CORRECTED, "corrections": [lumped]})
        return json.dumps({"corrected_text": SPLIT_CORRECTED, "corrections": []})

    return fake_call


def test_lumped_fix_is_split_into_separate_fixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace a whole-sentence fix with one fix per mistake, each with its own reason."""
    calls: list[dict] = []
    monkeypatch.setattr("llm_client.call_model", _split_fake([SPLIT_PARTS], calls))
    result = check_grammar(SPLIT_TEXT)
    assert result["corrected_text"] == SPLIT_CORRECTED
    assert result["corrections"] == SPLIT_PARTS
    split_prompt = calls[-1]["prompt"]
    assert f"<before>{SPLIT_TEXT}</before>" in split_prompt
    assert f"<after>{SPLIT_CORRECTED}</after>" in split_prompt
    assert '- added "the"' in split_prompt


def test_split_retries_when_still_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ask once more when the split answer is still a single entry."""
    lumped = [{"original": SPLIT_TEXT, "correction": SPLIT_CORRECTED, "explanation": "x"}]
    calls: list[dict] = []
    monkeypatch.setattr("llm_client.call_model", _split_fake([lumped, SPLIT_PARTS], calls))
    assert len(check_grammar(SPLIT_TEXT)["corrections"]) == 3
    assert "combined several mistakes" in calls[-1]["prompt"]


def test_failed_split_keeps_combined_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the original fix when the split answers are unusable or invented."""
    invented = [
        {"original": "She go", "correction": "She went", "explanation": "Not in the text."},
        {"original": "goes", "correction": "go", "explanation": "Plural subject."},
    ]
    calls: list[dict] = []
    monkeypatch.setattr("llm_client.call_model", _split_fake(["not json", invented], calls))
    corrections = check_grammar(SPLIT_TEXT)["corrections"]
    assert [c["original"] for c in corrections] == [SPLIT_TEXT]


def test_check_grammar_stops_after_three_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give up after three invalid responses."""
    calls = {"n": 0}

    def fake_call(prompt: str, **kwargs: object) -> str:
        calls["n"] += 1
        return "still not json"

    monkeypatch.setattr("llm_client.call_model", fake_call)
    with pytest.raises(MalformedLLMResponseError):
        check_grammar(TEXT)
    assert calls["n"] == 3
