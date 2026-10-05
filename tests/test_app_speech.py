"""Headless tests for the Pronunciation tab. Kokoro is mocked."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _speech_tab(at: AppTest):
    """Return the Pronunciation tab from a run app."""
    return next(tab for tab in at.tabs if tab.label == "Pronunciation")


def _fake_synthesize(calls: list):
    """Record each call and return a short tone."""

    def synthesize(text: str, accent: str, voice: str):
        calls.append((text, accent, voice))
        return np.sin(np.linspace(0, 100, 2400, dtype=np.float32)), 24000

    return synthesize


def test_speech_tab_defaults_to_american_voice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speak with the American Heart voice unless the user picks another."""
    calls: list = []
    monkeypatch.setattr("tts.synthesize", _fake_synthesize(calls))
    at = AppTest.from_file(APP_PATH).run()
    tab = _speech_tab(at)
    tab.text_area[0].input("  Nice to meet you.  ")
    tab.button[0].click()
    at.run()
    assert calls == [("Nice to meet you.", "American", "af_heart")]
    assert not _speech_tab(at).error
    assert at.session_state.speech["sample_rate"] == 24000


def test_speech_tab_uses_chosen_accent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching to British offers British voices."""
    calls: list = []
    monkeypatch.setattr("tts.synthesize", _fake_synthesize(calls))
    at = AppTest.from_file(APP_PATH).run()
    tab = _speech_tab(at)
    tab.text_area[0].input("Schedule")
    tab.button_group[0].set_value("British")
    at.run()
    _speech_tab(at).button[0].click()
    at.run()
    assert calls == [("Schedule", "British", "bf_emma")]


def test_speech_tab_rejects_empty_text() -> None:
    """Ask for text instead of loading the model on an empty box."""
    at = AppTest.from_file(APP_PATH).run()
    _speech_tab(at).button[0].click()
    at.run()
    assert _speech_tab(at).error[0].value == "Please enter the text you want to hear."
