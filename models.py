"""History entry shape and lookup-input validation shared by the app layers."""

import difflib
import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from config import MAX_INPUT_LENGTH


@dataclass
class LookupEntry:
    """One saved lookup, including the fields the model returns."""

    word: str
    senses: list[dict]
    sentiment: str
    created_at: str
    updated_at: str | None = None

    def to_dict(self) -> dict:
        """Return a JSON-ready dictionary, omitting an unset updated_at."""
        data = {
            "word": self.word,
            "senses": self.senses,
            "sentiment": self.sentiment,
            "created_at": self.created_at,
        }
        if self.updated_at is not None:
            data["updated_at"] = self.updated_at
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "LookupEntry":
        """Build an entry from a stored or model-merged dictionary."""
        return cls(
            word=data["word"],
            senses=entry_senses(data),
            sentiment=data["sentiment"],
            created_at=data["created_at"],
            updated_at=data.get("updated_at"),
        )


def entry_senses(entry: dict) -> list[dict]:
    """Return the senses of an entry, adapting entries saved before senses existed."""
    senses = entry.get("senses")
    if isinstance(senses, list) and senses:
        return list(senses)
    if "definition" not in entry:
        return []
    return [
        {
            "part_of_speech": entry.get("part_of_speech", ""),
            "definition": entry.get("definition", ""),
            "synonyms": list(entry.get("synonyms", [])),
            "antonyms": list(entry.get("antonyms", [])),
            "examples": list(entry.get("examples", [])),
        }
    ]


_VOWELS = set("aeiou")
_IRREGULAR_VERBS = {
    "be": {"am", "is", "are", "was", "were", "been", "being"},
    "bear": {"bore", "born", "borne"},
    "beat": {"beaten"},
    "become": {"became"},
    "begin": {"began", "begun"},
    "bite": {"bit", "bitten"},
    "blow": {"blew", "blown"},
    "break": {"broke", "broken"},
    "bring": {"brought"},
    "build": {"built"},
    "buy": {"bought"},
    "catch": {"caught"},
    "choose": {"chose", "chosen"},
    "come": {"came"},
    "cut": set(),
    "deal": {"dealt"},
    "do": {"does", "did", "done"},
    "draw": {"drew", "drawn"},
    "drink": {"drank", "drunk"},
    "drive": {"drove", "driven"},
    "eat": {"ate", "eaten"},
    "fall": {"fell", "fallen"},
    "feel": {"felt"},
    "fight": {"fought"},
    "find": {"found"},
    "fly": {"flew", "flown"},
    "forget": {"forgot", "forgotten"},
    "forgive": {"forgave", "forgiven"},
    "freeze": {"froze", "frozen"},
    "get": {"got", "gotten"},
    "give": {"gave", "given"},
    "go": {"went", "gone", "goes"},
    "grow": {"grew", "grown"},
    "hang": {"hung"},
    "have": {"has", "had"},
    "hear": {"heard"},
    "hide": {"hid", "hidden"},
    "hit": set(),
    "hold": {"held"},
    "hurt": set(),
    "keep": {"kept"},
    "know": {"knew", "known"},
    "lead": {"led"},
    "leave": {"left"},
    "lend": {"lent"},
    "let": set(),
    "lie": {"lay", "lain"},
    "light": {"lit"},
    "lose": {"lost"},
    "make": {"made"},
    "mean": {"meant"},
    "meet": {"met"},
    "pay": {"paid"},
    "put": set(),
    "read": set(),
    "ride": {"rode", "ridden"},
    "ring": {"rang", "rung"},
    "rise": {"rose", "risen"},
    "run": {"ran"},
    "say": {"said"},
    "see": {"saw", "seen"},
    "seek": {"sought"},
    "sell": {"sold"},
    "send": {"sent"},
    "set": set(),
    "shake": {"shook", "shaken"},
    "shine": {"shone"},
    "show": {"shown"},
    "shut": set(),
    "sing": {"sang", "sung"},
    "sink": {"sank", "sunk"},
    "sit": {"sat"},
    "sleep": {"slept"},
    "speak": {"spoke", "spoken"},
    "spend": {"spent"},
    "stand": {"stood"},
    "steal": {"stole", "stolen"},
    "swim": {"swam", "swum"},
    "take": {"took", "taken"},
    "teach": {"taught"},
    "tear": {"tore", "torn"},
    "tell": {"told"},
    "think": {"thought"},
    "throw": {"threw", "thrown"},
    "understand": {"understood"},
    "wake": {"woke", "woken"},
    "wear": {"wore", "worn"},
    "win": {"won"},
    "write": {"wrote", "written"},
}


def _doubles_final_consonant(verb: str) -> bool:
    """Return whether a short verb doubles its last letter before -ed or -ing."""
    if len(verb) < 3 or verb[-1] in _VOWELS or verb[-1] in "wxy":
        return False
    return verb[-2] in _VOWELS and verb[-3] not in _VOWELS


def _verb_forms(verb: str) -> set[str]:
    """Return the base verb plus regular and irregular tense forms."""
    word = verb.casefold()
    forms = {word, word + "s", word + "ed", word + "ing"}
    if word.endswith("e"):
        forms.add(word + "d")
        forms.add(word[:-1] + "ing")
    if word.endswith("y") and len(word) > 1 and word[-2] not in _VOWELS:
        forms.add(word[:-1] + "ies")
        forms.add(word[:-1] + "ied")
    if word.endswith(("s", "x", "z", "ch", "sh", "o")):
        forms.add(word + "es")
    if _doubles_final_consonant(word):
        forms.add(word + word[-1] + "ed")
        forms.add(word + word[-1] + "ing")
    forms.update(_IRREGULAR_VERBS.get(word, set()))
    return forms


def _lookup_forms(lookup: str) -> list[str]:
    """Return match strings, longest first, including a changed verb tense."""
    parts = lookup.split()
    if len(parts) > 1:
        forms = {" ".join([form, *parts[1:]]) for form in _verb_forms(parts[0])}
        forms.add(lookup)
    else:
        forms = _verb_forms(lookup)
        forms.add(lookup.casefold())
    return sorted(forms, key=len, reverse=True)


def _lookup_pattern(lookups: list[str]) -> re.Pattern | None:
    """Match any of the lookups, including a changed verb tense, or None if all are blank."""
    forms = {form for lookup in lookups if (cleaned := lookup.strip()) for form in _lookup_forms(cleaned)}
    if not forms:
        return None
    alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)


def underline_lookup(text: str, lookup: str) -> str:
    """Return HTML with the lookup, including a changed verb tense, underlined."""
    return highlight_words(text, [lookup])


def highlight_words(text: str, lookups: list[str]) -> str:
    """Return HTML with every lookup, including a changed verb tense, underlined."""
    if not text:
        return ""
    pattern = _lookup_pattern(lookups)
    if pattern is None:
        return html.escape(text)
    pieces: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        pieces.append(html.escape(text[last : match.start()]))
        pieces.append(f"<u>{html.escape(match.group(0))}</u>")
        last = match.end()
    pieces.append(html.escape(text[last:]))
    return "".join(pieces)


def missing_words(text: str, lookups: list[str]) -> list[str]:
    """Return the lookups that do not appear in the text in any verb tense."""
    return [lookup for lookup in lookups if (pattern := _lookup_pattern([lookup])) and not pattern.search(text)]


def correction_segments(original: str, correction: str) -> list[tuple[str, str]]:
    """Split a fix into ("same" | "removed" | "added", words) pieces, word by word.

    Words both versions share come back as "same", so only the changed words are
    marked even when the model quotes a longer phrase than the mistake.
    """
    before = original.split()
    after = correction.split()
    segments: list[tuple[str, str]] = []
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, start_a, end_a, start_b, end_b in matcher.get_opcodes():
        if tag == "equal":
            segments.append(("same", " ".join(before[start_a:end_a])))
            continue
        if end_a > start_a:
            segments.append(("removed", " ".join(before[start_a:end_a])))
        if end_b > start_b:
            segments.append(("added", " ".join(after[start_b:end_b])))
    return segments


def validate_lookup_input(
    text: str,
    *,
    max_length: int = MAX_INPUT_LENGTH,
    empty_message: str = "Please enter a word, phrase, or sentence.",
) -> str:
    """Clean and validate text the user typed."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(empty_message)
    if len(cleaned) > max_length:
        raise ValueError("Input is too long. Enter a sentence, not a paragraph.")
    return cleaned


def entry_from_lookup(word: str, payload: dict) -> dict:
    """Merge a validated model payload with the lookup word and a timestamp."""
    created_at = datetime.now(timezone.utc).isoformat()
    entry = LookupEntry(
        word=word,
        senses=list(payload["senses"]),
        sentiment=payload["sentiment"],
        created_at=created_at,
    )
    return entry.to_dict()
