"""Headless tests for the Study tab. History, the model, and Kokoro are mocked."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
HISTORY = [
    {"word": "brave", "senses": [], "sentiment": "positive", "created_at": "2026-01-01"},
    {"word": "light", "senses": [], "sentiment": "neutral", "created_at": "2026-01-02"},
    {"word": "storm", "senses": [], "sentiment": "negative", "created_at": "2026-01-03"},
]
REPLIES = {
    "Story": "In the storm, a brave girl held a light.",
    "Conversation": "Anna: Was the storm scary?\nBen: A little, but I tried to be brave.",
}


def _study_tab(at: AppTest):
    """Return the Study tab from a run app."""
    return next(tab for tab in at.tabs if tab.label == "Study")


def _button(at: AppTest, label: str):
    """Return the Study tab button with this label."""
    return next(button for button in _study_tab(at).button if button.label == label)


def _start(
    monkeypatch: pytest.MonkeyPatch,
    history: list[dict],
    calls: list[tuple[list[str], str]],
    speech_calls: list[tuple[str, str, str]] | None = None,
) -> AppTest:
    """Run the app with the given history, a fake writer, and a fake speech model."""

    def fake_study(words: list[str], style: str) -> str:
        calls.append((list(words), style))
        return REPLIES[style]

    def fake_synthesize(text: str, accent: str, voice: str):
        if speech_calls is not None:
            speech_calls.append((text, accent, voice))
        return np.sin(np.linspace(0, 100, 2400, dtype=np.float32)), 24000

    monkeypatch.setattr("storage.load_history", lambda *_: [dict(entry) for entry in history])
    monkeypatch.setattr("llm_client.generate_study", fake_study)
    monkeypatch.setattr("tts.synthesize", fake_synthesize)
    return AppTest.from_file(APP_PATH).run()


def _generate(at: AppTest, style: str = "Story") -> AppTest:
    """Pick a format, click its generate button, and return the app."""
    at.session_state.study_style = style
    at.run()
    _button(at, f"Generate {style.casefold()}").click()
    return at.run()


def test_study_tab_selects_every_word_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show every history word, newest first, all selected, with a story as the default."""
    at = _start(monkeypatch, HISTORY, [])
    assert at.session_state.study_words == ["storm", "light", "brave"]
    assert at.session_state.study_style == "Story"
    assert _study_tab(at).caption[0].value == "3 of 3 words selected"


def test_study_tab_generates_story_with_selected_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send only the words the user kept and show the story."""
    calls: list[tuple[list[str], str]] = []
    at = _start(monkeypatch, HISTORY, calls)
    at.session_state.study_words = ["storm", "brave"]
    at.run()
    _button(at, "Generate story").click()
    at.run()
    assert calls == [(["storm", "brave"], "Story")]
    assert _study_tab(at).subheader[0].value == "Your story"


def test_study_tab_generates_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch to a conversation and show one line per speaker with bold names."""
    calls: list[tuple[list[str], str]] = []
    at = _start(monkeypatch, HISTORY, calls)
    at.session_state.study_style = "Conversation"
    at.run()
    _button(at, "Generate conversation").click()
    at.run()
    tab = _study_tab(at)
    assert calls == [(["storm", "light", "brave"], "Conversation")]
    assert tab.subheader[0].value == "Your conversation"
    assert tab.markdown[-1].value == (
        "**Anna:** Was the <u>storm</u> scary?\n\n**Ben:** A little, but I tried to be <u>brave</u>."
    )
    assert tab.caption[-1].value == "Not used in this conversation: light"


def test_study_tab_clear_and_select_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every word, refuse to generate, then select all again."""
    calls: list[tuple[list[str], str]] = []
    at = _start(monkeypatch, HISTORY, calls)
    _button(at, "Clear").click()
    at.run()
    assert at.session_state.study_words == []
    _button(at, "Generate story").click()
    at.run()
    assert _study_tab(at).error[0].value == "Choose at least one word."
    assert calls == []
    _button(at, "Select all").click()
    at.run()
    assert at.session_state.study_words == ["storm", "light", "brave"]


def test_study_tab_selects_new_words_and_keeps_exclusions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newly looked-up word starts selected; an excluded word stays excluded."""
    history = list(HISTORY)
    at = _start(monkeypatch, history, [])
    at.session_state.study_words = ["storm", "brave"]
    at.run()
    history.append({"word": "calm", "senses": [], "sentiment": "positive", "created_at": "2026-01-04"})
    at.run()
    assert at.session_state.study_words == ["calm", "storm", "brave"]


def test_study_tab_without_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain that words must be looked up first."""
    at = _start(monkeypatch, [], [])
    assert "Look up some words first" in _study_tab(at).info[0].value


def test_study_tab_prepares_story_audio_and_reads_on_click(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepare the audio right after the story, then play it when Read is clicked."""
    speech_calls: list[tuple[str, str, str]] = []
    at = _generate(_start(monkeypatch, HISTORY, [], speech_calls))
    assert speech_calls == [(REPLIES["Story"], "American", "af_heart")]
    assert not _study_tab(at).get("audio")
    _button(at, "Read story").click()
    at.run()
    assert len(_study_tab(at).get("audio")) == 1
    assert len(speech_calls) == 1


def test_study_tab_reads_conversation_with_two_voices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each speaker their own voice and leave the names out of the audio."""
    speech_calls: list[tuple[str, str, str]] = []
    at = _generate(_start(monkeypatch, HISTORY, [], speech_calls), "Conversation")
    assert speech_calls == [
        ("Was the storm scary?", "American", "af_heart"),
        ("A little, but I tried to be brave.", "American", "am_michael"),
    ]
    assert _button(at, "Read conversation")


def test_study_tab_warns_when_audio_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the story and explain that the audio is missing, without a Read button."""
    at = _start(monkeypatch, HISTORY, [])

    def broken_synthesize(*_: object):
        raise RuntimeError("model missing")

    monkeypatch.setattr("tts.synthesize", broken_synthesize)
    at = _generate(at)
    tab = _study_tab(at)
    assert tab.subheader[0].value == "Your story"
    assert tab.warning[0].value == "The audio could not be prepared."
    assert not [button for button in tab.button if button.label == "Read story"]
