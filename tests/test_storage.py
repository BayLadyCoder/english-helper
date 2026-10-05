"""Tests for JSON history load, save, and dedupe."""

import json
from pathlib import Path

from models import entry_from_lookup
from storage import delete_entry, find_entry, load_history, save_entry

EXAMPLES = [
    "I ride a bicycle.",
    "The bicycle is red.",
    "She locked her bicycle.",
]


def _entry(word: str, language: str, definition: str = "A two-wheeled vehicle.") -> dict:
    """Build a history entry dict for storage tests."""
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
    return entry_from_lookup(word, payload)


def test_first_save_creates_file(tmp_path: Path) -> None:
    """Create the JSON file on the first save."""
    path = tmp_path / "history.json"
    save_entry(_entry("bicycle", "Spanish"), path)
    assert path.exists()
    assert len(json.loads(path.read_text())) == 1


def test_dedupe_updates_in_place(tmp_path: Path) -> None:
    """Replace a repeated word and language instead of appending it."""
    path = tmp_path / "history.json"
    save_entry(_entry("bicycle", "Spanish"), path)
    save_entry(_entry("run", "Spanish"), path)
    save_entry(_entry("ephemeral", "French", "Lasting a short time."), path)
    save_entry(_entry("bicycle", "Spanish", "Updated definition."), path)
    entries = load_history(path)
    assert len(entries) == 3
    bicycle = next(entry for entry in entries if entry["word"] == "bicycle")
    assert bicycle["senses"][0]["definition"] == "Updated definition."
    assert bicycle["updated_at"]
    assert entries[0]["word"] == "bicycle"


def test_same_word_replaces_the_previous_entry(tmp_path: Path) -> None:
    """Keep one entry per word."""
    path = tmp_path / "history.json"
    save_entry(_entry("bicycle", "Spanish"), path)
    save_entry(_entry("bicycle", "French", "Updated."), path)
    entries = load_history(path)
    assert len(entries) == 1
    assert entries[0]["senses"][0]["definition"] == "Updated."


def test_case_only_difference_dedupes(tmp_path: Path) -> None:
    """Treat Bicycle and bicycle as the same lookup."""
    path = tmp_path / "history.json"
    save_entry(_entry("Bicycle", "Spanish"), path)
    save_entry(_entry("bicycle", "spanish", "Updated."), path)
    entries = load_history(path)
    assert len(entries) == 1
    assert entries[0]["senses"][0]["definition"] == "Updated."
    assert entries[0]["created_at"]


def test_delete_removes_the_word(tmp_path: Path) -> None:
    """Remove one saved word and leave the others in place."""
    path = tmp_path / "history.json"
    save_entry(_entry("bicycle", "Spanish"), path)
    save_entry(_entry("run", "Spanish"), path)
    remaining = delete_entry("Bicycle", path)
    assert [entry["word"] for entry in remaining] == ["run"]
    assert [entry["word"] for entry in load_history(path)] == ["run"]


def test_delete_missing_word_leaves_the_list(tmp_path: Path) -> None:
    """Leave the file unchanged when the word is not saved."""
    path = tmp_path / "history.json"
    save_entry(_entry("bicycle", "Spanish"), path)
    assert len(delete_entry("home", path)) == 1
    assert load_history(tmp_path / "missing.json") == []
    assert delete_entry("home", tmp_path / "missing.json") == []


def test_find_entry_ignores_case_and_whitespace(tmp_path: Path) -> None:
    """Find a saved word regardless of case or surrounding spaces."""
    path = tmp_path / "history.json"
    save_entry(_entry("Bicycle", "Spanish"), path)
    found = find_entry("  bicycle ", path)
    assert found is not None
    assert found["word"] == "Bicycle"


def test_find_entry_returns_none_when_missing(tmp_path: Path) -> None:
    """Return None for unsaved words, blank input, or a missing file."""
    path = tmp_path / "history.json"
    save_entry(_entry("bicycle", "Spanish"), path)
    assert find_entry("run", path) is None
    assert find_entry("   ", path) is None
    assert find_entry("bicycle", tmp_path / "missing.json") is None


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    """Return an empty list when the file has not been created yet."""
    assert load_history(tmp_path / "missing.json") == []


def test_corrupt_file_loads_empty_and_leaves_backup(tmp_path: Path) -> None:
    """Start over from an empty list and keep the corrupt file as a backup."""
    path = tmp_path / "history.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_history(path) == []
    assert not path.exists()
    assert (tmp_path / "history.json.bak").exists()
