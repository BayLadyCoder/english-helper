"""Load, save, and dedupe lookup history in a local JSON file."""

import json
from pathlib import Path

from config import HISTORY_PATH


def _resolve_path(path: Path | None) -> Path:
    """Use the given path, or the default history file."""
    return path if path is not None else HISTORY_PATH


def _backup_corrupt_file(path: Path) -> None:
    """Rename a corrupt file so the next save does not destroy it."""
    backup = path.parent / f"{path.name}.bak"
    if backup.exists():
        backup.unlink()
    path.rename(backup)


def load_history(path: Path | None = None) -> list[dict]:
    """Return saved lookups, or an empty list if the file is missing or corrupt."""
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


def find_entry(word: str, path: Path | None = None) -> dict | None:
    """Return the saved entry for this word, ignoring case, or None."""
    key = str(word).strip().casefold()
    if not key:
        return None
    for entry in load_history(_resolve_path(path)):
        if str(entry.get("word", "")).casefold() == key:
            return entry
    return None


def save_entry(entry: dict, path: Path | None = None) -> list[dict]:
    """Insert or replace an entry keyed by word, then return the list."""
    if not entry.get("word"):
        raise ValueError("A history entry needs a word.")
    file_path = _resolve_path(path)
    entries = load_history(file_path)
    key = str(entry["word"]).casefold()
    replaced = False
    for index, existing in enumerate(entries):
        if str(existing.get("word", "")).casefold() != key:
            continue
        updated = dict(entry)
        updated["created_at"] = existing.get("created_at", entry.get("created_at"))
        updated["updated_at"] = entry.get("created_at")
        entries[index] = updated
        replaced = True
        break
    if not replaced:
        entries.append(entry)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries


def delete_entry(word: str, path: Path | None = None) -> list[dict]:
    """Remove the saved entry for this word and return the remaining list."""
    if not str(word).strip():
        raise ValueError("A history entry needs a word.")
    file_path = _resolve_path(path)
    entries = load_history(file_path)
    key = str(word).casefold()
    kept = [entry for entry in entries if str(entry.get("word", "")).casefold() != key]
    if len(kept) == len(entries):
        return entries
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
    return kept
