"""Streamlit UI for looking up words, checking grammar, and reviewing lookup history."""

import re

import streamlit as st

import tts
from config import (
    ACCENTS,
    DEFAULT_ACCENT,
    MAX_GRAMMAR_LENGTH,
    MAX_SPEECH_LENGTH,
    MAX_STORY_WORDS,
    MAX_TONE_LENGTH,
    TONE_OPTIONS,
)
from llm_client import (
    LLMUnavailableError,
    MalformedLLMResponseError,
    adjust_tone,
    check_grammar,
    generate_story,
    lookup,
)
from models import (
    correction_segments,
    entry_from_lookup,
    entry_senses,
    highlight_words,
    missing_words,
    underline_lookup,
    validate_lookup_input,
)
from storage import delete_entry, find_entry, load_history, save_entry

st.set_page_config(page_title="EnglishHelper", page_icon="EH")
st.title("EnglishHelper")
st.write(
    "Look up a word, fix grammar mistakes, rewrite something you wrote in a different tone, "
    "hear how text is pronounced, or study the words you looked up in a short story."
)

if "last_entry" not in st.session_state:
    st.session_state.last_entry = None
if "adjusted_text" not in st.session_state:
    st.session_state.adjusted_text = None
if "grammar_result" not in st.session_state:
    st.session_state.grammar_result = None
if "speech" not in st.session_state:
    st.session_state.speech = None
if "story" not in st.session_state:
    st.session_state.story = None

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]~#|<>$:])")


def _escape_markdown(text: str) -> str:
    """Escape characters so user text renders literally inside Markdown."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _example_text(example: object) -> str:
    """Return the English sentence from a new or previously saved example."""
    if isinstance(example, dict):
        return str(example.get("sentence", ""))
    return str(example)


def _sentiment_color(label: str) -> str:
    """Return a display color that accompanies the sentiment text label."""
    normalized = label.casefold()
    if "pos" in normalized:
        return "#1b7f4e"
    if "neg" in normalized:
        return "#9b2c2c"
    return "#3d4f66"


def render_entry(entry: dict) -> None:
    """Show one looked-up word, including its definitions and examples."""
    st.subheader(entry["word"])
    sentiment = entry["sentiment"]
    color = _sentiment_color(sentiment)
    st.markdown(
        f"**Sentiment:** <span style='color: {color}'>{sentiment}</span>",
        unsafe_allow_html=True,
    )
    lookup_text = str(entry.get("word", ""))
    senses = entry_senses(entry)
    for sense in senses:
        with st.container(border=True):
            part_of_speech = str(sense.get("part_of_speech", "unknown")).capitalize()
            st.subheader(part_of_speech)
            st.write(sense.get("definition", ""))
            synonyms = sense.get("synonyms") or []
            antonyms = sense.get("antonyms") or []
            st.write("**Synonyms:** " + (", ".join(synonyms) if synonyms else "none"))
            st.write("**Antonyms:** " + (", ".join(antonyms) if antonyms else "none"))
            st.markdown("**Examples**")
            for example in sense.get("examples") or []:
                sentence = underline_lookup(_example_text(example), lookup_text)
                st.markdown(f"- {sentence}", unsafe_allow_html=True)


_SEGMENT_MARKDOWN = {
    "same": "{}",
    "removed": ":red[~~{}~~]",
    "added": ":green[**{}**]",
}


def _correction_markdown(original: str, correction: str) -> str:
    """Show a fix inline, striking out only the words that changed."""
    return " ".join(
        _SEGMENT_MARKDOWN[kind].format(_escape_markdown(words))
        for kind, words in correction_segments(original, correction)
    )


def render_grammar(result: dict) -> None:
    """Show the corrected text and explain each grammar fix."""
    corrections = result["corrections"]
    if not corrections and not result["changed"]:
        st.success("No grammar mistakes found.", icon=":material/check_circle:")
        return
    st.subheader("Corrected text")
    st.code(result["corrected_text"], language=None, wrap_lines=True)
    if not corrections:
        return
    count = len(corrections)
    st.subheader(f"{count} fix" if count == 1 else f"{count} fixes")
    for correction in corrections:
        with st.container(border=True):
            st.markdown(_correction_markdown(correction["original"], correction["correction"]))
            if correction["explanation"]:
                st.caption(correction["explanation"])


def _sync_story_words(words: list[str]) -> None:
    """Select every word by default, including words added since the last run.

    Words the user excluded stay excluded, and deleted words are dropped so the
    selection only holds current options.
    """
    if "story_words" not in st.session_state:
        st.session_state.story_words = list(words)
    else:
        selected = set(st.session_state.story_words)
        known = set(st.session_state.get("story_known_words", []))
        st.session_state.story_words = [w for w in words if w in selected or w not in known]
    st.session_state.story_known_words = list(words)


def _set_story_words(words: list[str]) -> None:
    """Replace the story selection, used by the select-all and clear buttons."""
    st.session_state.story_words = list(words)


lookup_tab, grammar_tab, tone_tab, speech_tab, study_tab, history_tab = st.tabs(
    ["Lookup", "Grammar", "Tone", "Pronunciation", "Study", "History"]
)

with lookup_tab:
    text = st.text_input("Word, phrase, or sentence")
    if st.button("Look up"):
        try:
            cleaned = validate_lookup_input(text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            try:
                saved = find_entry(cleaned)
            except OSError:
                saved = None
            if saved is not None:
                st.session_state.last_entry = saved
                st.caption("Showing the saved result from your history.")
            else:
                try:
                    with st.spinner("Looking up..."):
                        payload = lookup(cleaned)
                        entry = entry_from_lookup(cleaned, payload)
                        save_entry(entry)
                    st.session_state.last_entry = entry
                except LLMUnavailableError as exc:
                    st.error(str(exc))
                except MalformedLLMResponseError:
                    st.error("The model returned an unexpected result. Please try again.")
                except OSError:
                    st.error("The history could not be saved. Please try again.")
    if st.session_state.last_entry is not None:
        render_entry(st.session_state.last_entry)

with grammar_tab:
    grammar_text = st.text_area("Text to check", height=140)
    if st.button("Check grammar"):
        try:
            cleaned = validate_lookup_input(
                grammar_text,
                max_length=MAX_GRAMMAR_LENGTH,
                empty_message="Please enter the text you want to check.",
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            try:
                with st.spinner("Checking grammar..."):
                    st.session_state.grammar_result = check_grammar(cleaned)
            except LLMUnavailableError as exc:
                st.error(str(exc))
            except MalformedLLMResponseError:
                st.error("The model returned an unexpected result. Please try again.")
    if st.session_state.grammar_result is not None:
        render_grammar(st.session_state.grammar_result)

with tone_tab:
    tone_text = st.text_area("Text to adjust", height=140)
    tone = st.segmented_control(
        "Tone",
        TONE_OPTIONS,
        default="Professional",
        wrap=True,
    )
    if st.button("Adjust tone"):
        try:
            cleaned = validate_lookup_input(
                tone_text,
                max_length=MAX_TONE_LENGTH,
                empty_message="Please enter the text you want to adjust.",
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            if not tone:
                st.error("Choose a tone.")
            else:
                try:
                    with st.spinner("Adjusting tone..."):
                        st.session_state.adjusted_text = adjust_tone(cleaned, tone)
                except LLMUnavailableError as exc:
                    st.error(str(exc))
                except MalformedLLMResponseError:
                    st.error("The model returned an unexpected result. Please try again.")
    if st.session_state.adjusted_text:
        st.subheader("Adjusted text")
        st.write(st.session_state.adjusted_text)

with speech_tab:
    speech_text = st.text_area("Text to pronounce", height=140)
    with st.container(horizontal=True):
        accent = st.segmented_control(
            "Accent",
            list(ACCENTS),
            default=DEFAULT_ACCENT,
            key="speech_accent",
        )
        voices = ACCENTS[accent or DEFAULT_ACCENT]["voices"]
        voice_name = st.selectbox("Voice", list(voices), key=f"speech_voice_{accent}")
    just_spoken = False
    if st.button("Generate Speech", icon=":material/volume_up:"):
        try:
            cleaned = validate_lookup_input(
                speech_text,
                max_length=MAX_SPEECH_LENGTH,
                empty_message="Please enter the text you want to hear.",
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            if not accent:
                st.error("Choose an accent.")
            else:
                try:
                    with st.spinner("Generating speech..."):
                        samples, sample_rate = tts.synthesize(cleaned, accent, voices[voice_name])
                    st.session_state.speech = {"samples": samples, "sample_rate": sample_rate}
                    just_spoken = True
                except ValueError as exc:
                    st.error(str(exc))
                except Exception:
                    st.error("Speech could not be generated. Please try again.")
    if st.session_state.speech is not None:
        st.audio(
            st.session_state.speech["samples"],
            sample_rate=st.session_state.speech["sample_rate"],
            autoplay=just_spoken,
        )

with study_tab:
    try:
        story_entries = load_history()
    except OSError:
        st.error("The history could not be loaded.")
        story_entries = []
    story_entries = sorted(story_entries, key=lambda entry: entry.get("created_at", ""), reverse=True)
    history_words = list(dict.fromkeys(w for entry in story_entries if (w := str(entry.get("word", "")).strip())))
    if not history_words:
        st.info("Look up some words first, then turn them into a story here.")
    else:
        _sync_story_words(history_words)
        st.pills(
            "Words to use",
            history_words,
            selection_mode="multi",
            key="story_words",
            help="Click a word to exclude it from the story, or click it again to include it.",
        )
        selected_words = list(st.session_state.story_words)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"{len(selected_words)} of {len(history_words)} words selected")
            st.button(
                "Select all",
                icon=":material/select_all:",
                on_click=_set_story_words,
                args=(history_words,),
            )
            st.button(
                "Clear",
                icon=":material/deselect:",
                on_click=_set_story_words,
                args=([],),
            )
        if st.button("Generate story", icon=":material/auto_stories:", type="primary"):
            if not selected_words:
                st.error("Choose at least one word for the story.")
            elif len(selected_words) > MAX_STORY_WORDS:
                st.error(f"Choose at most {MAX_STORY_WORDS} words for one story.")
            else:
                try:
                    with st.spinner("Writing your story..."):
                        story = generate_story(selected_words)
                    st.session_state.story = {"text": story, "words": selected_words}
                except LLMUnavailableError as exc:
                    st.error(str(exc))
                except MalformedLLMResponseError:
                    st.error("The model returned an unexpected result. Please try again.")
    if st.session_state.story is not None:
        story_text = st.session_state.story["text"]
        story_words = st.session_state.story["words"]
        st.subheader("Your story")
        with st.container(border=True):
            st.markdown(highlight_words(story_text, story_words), unsafe_allow_html=True)
        skipped = missing_words(story_text, story_words)
        if skipped:
            st.caption("Not used in this story: " + ", ".join(skipped))

with history_tab:
    try:
        entries = load_history()
    except OSError:
        st.error("The history could not be loaded.")
        entries = []
    if not entries:
        st.info("No lookups yet.")
    else:
        ordered = sorted(entries, key=lambda entry: entry.get("created_at", ""), reverse=True)
        for index, entry in enumerate(ordered):
            title = str(entry.get("word", "Untitled"))
            with st.expander(title):
                render_entry(entry)
                if st.button("Delete", key=f"delete-{index}-{title.casefold()}"):
                    try:
                        delete_entry(title)
                    except OSError:
                        st.error("The history could not be updated.")
                    else:
                        last_entry = st.session_state.last_entry
                        if last_entry and str(last_entry.get("word", "")).casefold() == title.casefold():
                            st.session_state.last_entry = None
                        st.rerun()
