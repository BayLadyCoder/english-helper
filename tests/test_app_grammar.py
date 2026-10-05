"""Headless tests for the Grammar tab. The model is mocked."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _grammar_tab(at: AppTest):
    """Return the Grammar tab from a run app."""
    return next(tab for tab in at.tabs if tab.label == "Grammar")


def _run_check(monkeypatch: pytest.MonkeyPatch, text: str, result: dict) -> AppTest:
    """Type text into the Grammar tab, click the button, and return the app."""
    monkeypatch.setattr("llm_client.check_grammar", lambda _: result)
    at = AppTest.from_file(APP_PATH).run()
    tab = _grammar_tab(at)
    tab.text_area[0].input(text)
    tab.button[0].click()
    return at.run()


def test_grammar_tab_shows_fixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show the corrected text and one bordered card per fix."""
    result = {
        "corrected_text": "She went home.",
        "corrections": [
            {"original": "She go home", "correction": "She went home", "explanation": "Use the past tense."}
        ],
        "changed": True,
    }
    at = _run_check(monkeypatch, "She go home.", result)
    tab = _grammar_tab(at)
    assert tab.code[0].value == "She went home."
    assert tab.subheader[-1].value == "1 fix"
    assert tab.markdown[-1].value == "She :red[~~go~~] :green[**went**] home"
    assert tab.caption[-1].value == "Use the past tense."


def test_grammar_tab_reports_clean_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Show a success message when nothing needs fixing."""
    result = {"corrected_text": "Hello.", "corrections": [], "changed": False}
    at = _run_check(monkeypatch, "Hello.", result)
    assert _grammar_tab(at).success[0].value == "No grammar mistakes found."


def test_grammar_tab_rejects_empty_text() -> None:
    """Ask for text instead of calling the model on an empty box."""
    at = AppTest.from_file(APP_PATH).run()
    _grammar_tab(at).button[0].click()
    at.run()
    assert _grammar_tab(at).error[0].value == "Please enter the text you want to check."
