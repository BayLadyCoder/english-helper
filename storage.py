"""Load, save, and dedupe flashcards in a local JSON file."""

import json
from pathlib import Path

from config import FLASHCARDS_PATH


def _resolve_path(path: Path | None) -> Path:
    """Use the given path, or the default flashcard file."""
    return path if path is not None else FLASHCARDS_PATH


def _backup_corrupt_file(path: Path) -> None:
    """Rename a corrupt file so the next save does not destroy it."""
    backup = path.parent / f"{path.name}.bak"
    if backup.exists():
        backup.unlink()
    path.rename(backup)


def load_flashcards(path: Path | None = None) -> list[dict]:
    """Return saved flashcards, or an empty list if the file is missing or corrupt."""
    file_path = _resolve_path(path)
    if not file_path.exists():
        return []
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _backup_corrupt_file(file_path)
        return []
    if not isinstance(data, list):
        _backup_corrupt_file(file_path)
        return []
    return data


def save_flashcard(card: dict, path: Path | None = None) -> list[dict]:
    """Insert or replace a card keyed by word, then return the list."""
    if not card.get("word"):
        raise ValueError("A flashcard needs a word.")
    file_path = _resolve_path(path)
    cards = load_flashcards(file_path)
    key = str(card["word"]).casefold()
    replaced = False
    for index, existing in enumerate(cards):
        if str(existing.get("word", "")).casefold() != key:
            continue
        updated = dict(card)
        updated["created_at"] = existing.get("created_at", card.get("created_at"))
        updated["updated_at"] = card.get("created_at")
        cards[index] = updated
        replaced = True
        break
    if not replaced:
        cards.append(card)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(cards, indent=2), encoding="utf-8")
    return cards


def delete_flashcard(word: str, path: Path | None = None) -> list[dict]:
    """Remove the saved card for this word and return the remaining list."""
    if not str(word).strip():
        raise ValueError("A flashcard needs a word.")
    file_path = _resolve_path(path)
    cards = load_flashcards(file_path)
    key = str(word).casefold()
    kept = [card for card in cards if str(card.get("word", "")).casefold() != key]
    if len(kept) == len(cards):
        return cards
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    return kept
