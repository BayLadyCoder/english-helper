"""Headless tests for the Study tab. History and the model are mocked."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
HISTORY = [
    {"word": "brave", "senses": [], "sentiment": "positive", "created_at": "2026-01-01"},
    {"word": "light", "senses": [], "sentiment": "neutral", "created_at": "2026-01-02"},
    {"word": "storm", "senses": [], "sentiment": "negative", "created_at": "2026-01-03"},
]


def _study_tab(at: AppTest):
    """Return the Study tab from a run app."""
    return next(tab for tab in at.tabs if tab.label == "Study")


def _button(at: AppTest, label: str):
    """Return the Study tab button with this label."""
    return next(button for button in _study_tab(at).button if button.label == label)


def _start(monkeypatch: pytest.MonkeyPatch, history: list[dict], calls: list[list[str]]) -> AppTest:
    """Run the app with the given history and a fake story writer."""

    def fake_story(words: list[str]) -> str:
        calls.append(list(words))
        return "In the storm, a brave girl held a light."

    monkeypatch.setattr("storage.load_history", lambda *_: [dict(entry) for entry in history])
    monkeypatch.setattr("llm_client.generate_story", fake_story)
    return AppTest.from_file(APP_PATH).run()


def test_study_tab_selects_every_word_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show every history word, newest first, all selected."""
    at = _start(monkeypatch, HISTORY, [])
    assert at.session_state.story_words == ["storm", "light", "brave"]
    assert _study_tab(at).caption[0].value == "3 of 3 words selected"


def test_study_tab_generates_with_selected_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send only the words the user kept and show the story."""
    calls: list[list[str]] = []
    at = _start(monkeypatch, HISTORY, calls)
    at.session_state.story_words = ["storm", "brave"]
    at.run()
    _button(at, "Generate story").click()
    at.run()
    assert calls == [["storm", "brave"]]
    assert _study_tab(at).subheader[0].value == "Your story"


def test_study_tab_clear_and_select_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every word, refuse to generate, then select all again."""
    calls: list[list[str]] = []
    at = _start(monkeypatch, HISTORY, calls)
    _button(at, "Clear").click()
    at.run()
    assert at.session_state.story_words == []
    _button(at, "Generate story").click()
    at.run()
    assert _study_tab(at).error[0].value == "Choose at least one word for the story."
    assert calls == []
    _button(at, "Select all").click()
    at.run()
    assert at.session_state.story_words == ["storm", "light", "brave"]


def test_study_tab_selects_new_words_and_keeps_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newly looked-up word starts selected; an excluded word stays excluded."""
    history = list(HISTORY)
    at = _start(monkeypatch, history, [])
    at.session_state.story_words = ["storm", "brave"]
    at.run()
    history.append({"word": "calm", "senses": [], "sentiment": "positive", "created_at": "2026-01-04"})
    at.run()
    assert at.session_state.story_words == ["calm", "storm", "brave"]


def test_study_tab_without_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain that words must be looked up first."""
    at = _start(monkeypatch, [], [])
    assert "Look up some words first" in _study_tab(at).info[0].value
