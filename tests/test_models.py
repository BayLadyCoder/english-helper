"""Tests for lookup input validation and flashcard records."""

import pytest

from models import card_from_lookup, card_senses, underline_lookup, validate_lookup_input


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


def test_card_from_lookup_adds_word_and_timestamp() -> None:
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
    card = card_from_lookup("bicycle", payload)
    assert card["word"] == "bicycle"
    assert card["sentiment"] == "neutral"
    assert card["senses"][0]["part_of_speech"] == "noun"
    assert "translation" not in card
    assert card["created_at"]


def test_card_senses_reads_an_older_card() -> None:
    """Show a card saved before senses existed as a single sense."""
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
    senses = card_senses(old)
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
