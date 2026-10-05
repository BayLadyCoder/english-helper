"""Streamlit UI for looking up words and reviewing lookup history."""

import streamlit as st

from config import MAX_TONE_LENGTH, TONE_OPTIONS
from llm_client import LLMUnavailableError, MalformedLLMResponseError, adjust_tone, lookup
from models import card_from_lookup, card_senses, underline_lookup, validate_lookup_input
from storage import delete_flashcard, load_flashcards, save_flashcard

st.set_page_config(page_title="EnglishHelper", page_icon="EH")
st.title("EnglishHelper")
st.write("Look up a word, or rewrite something you wrote in a different tone.")

if "last_card" not in st.session_state:
    st.session_state.last_card = None
if "adjusted_text" not in st.session_state:
    st.session_state.adjusted_text = None


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


def render_card(card: dict) -> None:
    """Show one looked-up word, including its definitions and examples."""
    st.subheader(card["word"])
    sentiment = card["sentiment"]
    color = _sentiment_color(sentiment)
    st.markdown(
        f"**Sentiment:** <span style='color: {color}'>{sentiment}</span>",
        unsafe_allow_html=True,
    )
    lookup_text = str(card.get("word", ""))
    senses = card_senses(card)
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



lookup_tab, tone_tab, history_tab = st.tabs(["Lookup", "Tone", "History"])

with lookup_tab:
    text = st.text_input("Word, phrase, or sentence")
    if st.button("Look up"):
        try:
            cleaned = validate_lookup_input(text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            try:
                with st.spinner("Looking up..."):
                    payload = lookup(cleaned)
                    card = card_from_lookup(cleaned, payload)
                    save_flashcard(card)
                st.session_state.last_card = card
            except LLMUnavailableError as exc:
                st.error(str(exc))
            except MalformedLLMResponseError:
                st.error("The model returned an unexpected result. Please try again.")
            except OSError:
                st.error("The history could not be saved. Please try again.")
    if st.session_state.last_card is not None:
        render_card(st.session_state.last_card)

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

with history_tab:
    try:
        cards = load_flashcards()
    except OSError:
        st.error("The history could not be loaded.")
        cards = []
    if not cards:
        st.info("No lookups yet.")
    else:
        ordered = sorted(cards, key=lambda card: card.get("created_at", ""), reverse=True)
        for index, card in enumerate(ordered):
            title = str(card.get("word", "Untitled"))
            with st.expander(title):
                render_card(card)
                if st.button("Delete", key=f"delete-{index}-{title.casefold()}"):
                    try:
                        delete_flashcard(title)
                    except OSError:
                        st.error("The history could not be updated.")
                    else:
                        last_card = st.session_state.last_card
                        if last_card and str(last_card.get("word", "")).casefold() == title.casefold():
                            st.session_state.last_card = None
                        st.rerun()
