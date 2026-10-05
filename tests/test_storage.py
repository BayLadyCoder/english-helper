"""Tests for JSON flashcard load, save, and dedupe."""

import json
from pathlib import Path

from models import card_from_lookup
from storage import delete_flashcard, load_flashcards, save_flashcard

EXAMPLES = [
    "I ride a bicycle.",
    "The bicycle is red.",
    "She locked her bicycle.",
]


def _card(word: str, language: str, definition: str = "A two-wheeled vehicle.") -> dict:
    """Build a card dict for storage tests."""
    payload = {
        "senses": [
            {
                "part_of_speech": "noun",
                "definition": definition,
                "synonyms": ["bike"],
                "antonyms": [],
                "examples": EXAMPLES,
            }
        ],
        "sentiment": "neutral",
    }
    return card_from_lookup(word, payload)


def test_first_save_creates_file(tmp_path: Path) -> None:
    """Create the JSON file on the first save."""
    path = tmp_path / "flashcards.json"
    save_flashcard(_card("bicycle", "Spanish"), path)
    assert path.exists()
    assert len(json.loads(path.read_text())) == 1


def test_dedupe_updates_in_place(tmp_path: Path) -> None:
    """Replace a repeated word and language instead of appending it."""
    path = tmp_path / "flashcards.json"
    save_flashcard(_card("bicycle", "Spanish"), path)
    save_flashcard(_card("run", "Spanish"), path)
    save_flashcard(_card("ephemeral", "French", "Lasting a short time."), path)
    save_flashcard(_card("bicycle", "Spanish", "Updated definition."), path)
    cards = load_flashcards(path)
    assert len(cards) == 3
    bicycle = next(card for card in cards if card["word"] == "bicycle")
    assert bicycle["senses"][0]["definition"] == "Updated definition."
    assert bicycle["updated_at"]
    assert cards[0]["word"] == "bicycle"


def test_same_word_replaces_the_previous_card(tmp_path: Path) -> None:
    """Keep one card per word."""
    path = tmp_path / "flashcards.json"
    save_flashcard(_card("bicycle", "Spanish"), path)
    save_flashcard(_card("bicycle", "French", "Updated."), path)
    cards = load_flashcards(path)
    assert len(cards) == 1
    assert cards[0]["senses"][0]["definition"] == "Updated."


def test_case_only_difference_dedupes(tmp_path: Path) -> None:
    """Treat Bicycle and bicycle as the same lookup."""
    path = tmp_path / "flashcards.json"
    save_flashcard(_card("Bicycle", "Spanish"), path)
    save_flashcard(_card("bicycle", "spanish", "Updated."), path)
    cards = load_flashcards(path)
    assert len(cards) == 1
    assert cards[0]["senses"][0]["definition"] == "Updated."
    assert cards[0]["created_at"]


def test_delete_removes_the_word(tmp_path: Path) -> None:
    """Remove one saved word and leave the others in place."""
    path = tmp_path / "flashcards.json"
    save_flashcard(_card("bicycle", "Spanish"), path)
    save_flashcard(_card("run", "Spanish"), path)
    remaining = delete_flashcard("Bicycle", path)
    assert [card["word"] for card in remaining] == ["run"]
    assert [card["word"] for card in load_flashcards(path)] == ["run"]


def test_delete_missing_word_leaves_the_list(tmp_path: Path) -> None:
    """Leave the file unchanged when the word is not saved."""
    path = tmp_path / "flashcards.json"
    save_flashcard(_card("bicycle", "Spanish"), path)
    assert len(delete_flashcard("home", path)) == 1
    assert load_flashcards(tmp_path / "missing.json") == []
    assert delete_flashcard("home", tmp_path / "missing.json") == []


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    """Return an empty list when the file has not been created yet."""
    assert load_flashcards(tmp_path / "missing.json") == []


def test_corrupt_file_loads_empty_and_leaves_backup(tmp_path: Path) -> None:
    """Start over from an empty list and keep the corrupt file as a backup."""
    path = tmp_path / "flashcards.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_flashcards(path) == []
    assert not path.exists()
    assert (tmp_path / "flashcards.json.bak").exists()
