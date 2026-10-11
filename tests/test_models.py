"""Tests for lookup input validation and history entries."""

import pytest

from models import (
    correction_segments,
    entry_from_lookup,
    entry_senses,
    highlight_words,
    missing_words,
    underline_lookup,
    validate_lookup_input,
)


def test_correction_segments_mark_only_changed_words() -> None:
    """Keep shared words plain and mark only the words that changed."""
    assert correction_segments("go to the store", "went to the store") == [
        ("removed", "go"),
        ("added", "went"),
        ("same", "to the store"),
    ]
    assert correction_segments("buy a apple", "bought an apple") == [
        ("removed", "buy a"),
        ("added", "bought an"),
        ("same", "apple"),
    ]


def test_correction_segments_handle_removed_and_inserted_words() -> None:
    """Show a dropped word as removed only and a new word as added only."""
    assert correction_segments("am agree", "agree") == [("removed", "am"), ("same", "agree")]
    assert correction_segments("go to office", "go to the office") == [
        ("same", "go to"),
        ("added", "the"),
        ("same", "office"),
    ]


def test_correction_segments_catch_capitalization_and_punctuation() -> None:
    """Treat a change in case or attached punctuation as a changed word."""
    assert correction_segments("yesterday I", "Yesterday I") == [
        ("removed", "yesterday"),
        ("added", "Yesterday"),
        ("same", "I"),
    ]
    assert correction_segments("However we", "However, we") == [
        ("removed", "However"),
        ("added", "However,"),
        ("same", "we"),
    ]


def test_empty_input_raises() -> None:
    """Reject an empty lookup."""
    with pytest.raises(ValueError):
        validate_lookup_input("")


def test_whitespace_input_raises() -> None:
    """Reject input that is only whitespace."""
    with pytest.raises(ValueError):
        validate_lookup_input("   \n\t")


def test_too_long_input_raises() -> None:
    """Reject input longer than a single sentence."""
    with pytest.raises(ValueError):
        validate_lookup_input("a" * 301)


def test_sentence_is_stripped() -> None:
    """Return the lookup text without surrounding whitespace."""
    assert validate_lookup_input("  break the ice  ") == "break the ice"


def test_entry_from_lookup_adds_word_and_timestamp() -> None:
    """Store the user word, language, and a created_at timestamp with the payload."""
    payload = {
        "senses": [
            {
                "part_of_speech": "noun",
                "definition": "A bike.",
                "synonyms": ["bike"],
                "antonyms": [],
                "examples": ["I ride a bicycle."],
            }
        ],
        "sentiment": "neutral",
    }
    entry = entry_from_lookup("bicycle", payload)
    assert entry["word"] == "bicycle"
    assert entry["sentiment"] == "neutral"
    assert entry["senses"][0]["part_of_speech"] == "noun"
    assert "translation" not in entry
    assert entry["created_at"]


def test_entry_senses_reads_an_older_entry() -> None:
    """Show an entry saved before senses existed as a single sense."""
    old = {
        "word": "bicycle",
        "target_language": "Spanish",
        "part_of_speech": "noun",
        "definition": "A bike.",
        "synonyms": ["bike"],
        "antonyms": [],
        "examples": ["I ride a bicycle."],
        "sentiment": "neutral",
        "translation": "bicicleta",
    }
    senses = entry_senses(old)
    assert len(senses) == 1
    assert senses[0]["part_of_speech"] == "noun"
    assert senses[0]["definition"] == "A bike."


def test_underline_lookup_word_ignores_case() -> None:
    """Underline the lookup word wherever it appears in an example."""
    html_text = underline_lookup("I'm going home after work.", "Home")
    assert "<u>home</u>" in html_text
    assert "after work." in html_text


def test_underline_lookup_phrase() -> None:
    """Underline a whole phrase, not a shorter word inside another word."""
    phrase = underline_lookup("She tried to break the ice.", "break the ice")
    assert "<u>break the ice</u>" in phrase
    word = underline_lookup("Notice the ice.", "ice")
    assert "<u>ice</u>" in word
    assert "<u>Notice</u>" not in word


def test_underline_changed_verb_tense() -> None:
    """Underline an irregular or inflected verb, not a longer unrelated word."""
    lit = underline_lookup("She lit the candle.", "light")
    assert "<u>lit</u>" in lit
    lighted = underline_lookup("He lighted the stove.", "light")
    assert "<u>lighted</u>" in lighted
    phrase = underline_lookup("She broke the ice with a joke.", "break the ice")
    assert "<u>broke the ice</u>" in phrase
    other = underline_lookup("The sunrise lightened the sky.", "light")
    assert "<u>" not in other


def test_highlight_words_underlines_every_word() -> None:
    """Underline each history word, including a changed verb tense."""
    html_text = highlight_words("She lit the lamp and broke the ice.", ["light", "break the ice", "lamp"])
    assert html_text == "She <u>lit</u> the <u>lamp</u> and <u>broke the ice</u>."


def test_missing_words_lists_unused_words() -> None:
    """Report only the words the story never uses."""
    story = "The kids ran home under a light rain."
    assert missing_words(story, ["run", "light", "umbrella"]) == ["umbrella"]
