"""Streamlit UI for looking up words, checking grammar, and reviewing lookup history."""

import re

import streamlit as st

import tts
from config import (
    ACCENTS,
    DEFAULT_ACCENT,
    MAX_GRAMMAR_LENGTH,
    MAX_SPEECH_LENGTH,
    MAX_STUDY_WORDS,
    MAX_TONE_LENGTH,
    STUDY_STYLES,
    TONE_OPTIONS,
)
from llm_client import (
    LLMUnavailableError,
    MalformedLLMResponseError,
    adjust_tone,
    check_grammar,
    generate_study,
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
    "hear how text is pronounced, or study the words you looked up in a short story or conversation."
)

if "last_entry" not in st.session_state:
    st.session_state.last_entry = None
if "adjusted_text" not in st.session_state:
    st.session_state.adjusted_text = None
if "grammar_result" not in st.session_state:
    st.session_state.grammar_result = None
if "speech" not in st.session_state:
    st.session_state.speech = None
if "study_result" not in st.session_state:
    st.session_state.study_result = None

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


def _sync_study_words(words: list[str]) -> None:
    """Select every word by default, including words added since the last run.

    Words the user excluded stay excluded, and deleted words are dropped so the
    selection only holds current options.
    """
    if "study_words" not in st.session_state:
        st.session_state.study_words = list(words)
    else:
        selected = set(st.session_state.study_words)
        known = set(st.session_state.get("study_known_words", []))
        st.session_state.study_words = [w for w in words if w in selected or w not in known]
    st.session_state.study_known_words = list(words)


def _set_study_words(words: list[str]) -> None:
    """Replace the word selection, used by the select-all and clear buttons."""
    st.session_state.study_words = list(words)


_SPEAKER_LINE = re.compile(r"^\**(\w[\w .'-]{0,20}?)\**:\**\s*(.*)$")


def _study_lines(text: str, style: str) -> list[tuple[str | None, str]]:
    """Split study text into (speaker, words) lines; speaker is None outside a conversation."""
    lines: list[tuple[str | None, str]] = []
    for line in (line.strip() for line in text.splitlines()):
        if not line:
            continue
        speaker = _SPEAKER_LINE.match(line) if style == "Conversation" else None
        lines.append(speaker.groups() if speaker else (None, line))
    return lines


def _study_markdown(text: str, words: list[str], style: str) -> str:
    """Underline the studied words, one paragraph per line, with bold speaker names."""
    return "\n\n".join(
        f"**{_escape_markdown(name)}:** {highlight_words(line, words)}" if name else highlight_words(line, words)
        for name, line in _study_lines(text, style)
    )


def _study_speech_parts(text: str, style: str) -> list[tuple[str, str]]:
    """Pair each line with a voice: a story uses one voice, and speakers take turns between two."""
    voices = ACCENTS[DEFAULT_ACCENT]["voices"]
    if style != "Conversation":
        return [(text, voices["Female"])]
    speaker_voices: dict[str, str] = {}
    parts: list[tuple[str, str]] = []
    for name, line in _study_lines(text, style):
        voice = voices["Female"]
        if name:
            turn = ("Female", "Male")[len(speaker_voices) % 2]
            voice = speaker_voices.setdefault(name.casefold(), voices[turn])
        parts.append((line, voice))
    return parts


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
        study_entries = load_history()
    except OSError:
        st.error("The history could not be loaded.")
        study_entries = []
    study_entries = sorted(study_entries, key=lambda entry: entry.get("created_at", ""), reverse=True)
    history_words = list(dict.fromkeys(w for entry in study_entries if (w := str(entry.get("word", "")).strip())))
    if not history_words:
        st.info("Look up some words first, then study them in a story or conversation here.")
    else:
        _sync_study_words(history_words)
        st.pills(
            "Words to use",
            history_words,
            selection_mode="multi",
            key="study_words",
            help="Click a word to exclude it, or click it again to include it.",
        )
        selected_words = list(st.session_state.study_words)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"{len(selected_words)} of {len(history_words)} words selected")
            st.button(
                "Select all",
                icon=":material/select_all:",
                on_click=_set_study_words,
                args=(history_words,),
            )
            st.button(
                "Clear",
                icon=":material/deselect:",
                on_click=_set_study_words,
                args=([],),
            )
        style = st.segmented_control(
            "Format",
            STUDY_STYLES,
            default=STUDY_STYLES[0],
            key="study_style",
        )
        kind = (style or STUDY_STYLES[0]).casefold()
        icon = ":material/forum:" if style == "Conversation" else ":material/auto_stories:"
        if st.button(f"Generate {kind}", icon=icon, type="primary"):
            if not style:
                st.error("Choose a story or a conversation.")
            elif not selected_words:
                st.error("Choose at least one word.")
            elif len(selected_words) > MAX_STUDY_WORDS:
                st.error(f"Choose at most {MAX_STUDY_WORDS} words at a time.")
            else:
                try:
                    with st.spinner(f"Writing your {kind}..."):
                        text = generate_study(selected_words, style)
                    st.session_state.study_result = {"text": text, "words": selected_words, "style": style}
                except LLMUnavailableError as exc:
                    st.error(str(exc))
                except MalformedLLMResponseError:
                    st.error("The model returned an unexpected result. Please try again.")
    if st.session_state.study_result is not None:
        result = st.session_state.study_result
        result_kind = result["style"].casefold()
        st.subheader(f"Your {result_kind}")
        with st.container(border=True):
            st.markdown(_study_markdown(result["text"], result["words"], result["style"]), unsafe_allow_html=True)
        skipped = missing_words(result["text"], result["words"])
        if skipped:
            st.caption(f"Not used in this {result_kind}: " + ", ".join(skipped))
        if "audio" not in result:
            try:
                with st.spinner("Preparing audio..."):
                    samples, sample_rate = tts.synthesize_parts(
                        _study_speech_parts(result["text"], result["style"]), DEFAULT_ACCENT
                    )
                result["audio"] = {"samples": samples, "sample_rate": sample_rate}
            except Exception:
                result["audio"] = None
        if result["audio"] is None:
            st.warning("The audio could not be prepared.", icon=":material/volume_off:")
        else:
            read_now = st.button(f"Read {result_kind}", icon=":material/volume_up:", key="study_read")
            if read_now:
                result["show_player"] = True
            if result.get("show_player"):
                st.audio(
                    result["audio"]["samples"],
                    sample_rate=result["audio"]["sample_rate"],
                    autoplay=read_now,
                )

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
