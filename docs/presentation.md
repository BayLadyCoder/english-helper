# Presentation notes (5–10 minutes)

Rehearse this sequence twice on a running app before you present. Both passes below were walked with the same modules the UI calls: validate, look up, save, then read the study list.

## 1. Problem (about 1 minute)

Learning one word usually means a dictionary, a translator, and a flashcard app. People quit at the step where they retype the word into the flashcard app. EnglishHelper makes the card as part of the search.

## 2. Live demo (3–4 minutes)

1. Start from an empty Study List, or say the list is whatever you have saved locally.
2. Look up `bicycle` in Spanish. Show definition, three examples, sentiment as a word, and the translation.
3. Look up `break the ice`. Point out that the definition is the idiom, not "damage frozen water."
4. Change the target language and look up another word. Show that the translation changed and that a repeat of the same word plus language updates one card.
5. Open Study List and show the saved cards.
6. Optional: stop Ollama, submit a word, and show "Is Ollama running?" Then start Ollama again.

## 3. Workflow diagram (1–2 minutes)

Open `docs/workflow.md`. The model is used once, at the Ollama step. `format="json"` is not a guarantee, so `parse_flashcard` checks the keys and `lookup` retries one time.

## 4. Tools (about 1 minute)

Ollama and Llama 3.2 run locally. Streamlit is the UI. uv installs dependencies. Flashcards are a JSON file, not a database, because the app does not search a document corpus.

## 5. Challenges (1–2 minutes)

Use `docs/quality_log.md`.

- The model sometimes returns invalid JSON or an empty field. The second attempt usually works. When it does not, the UI asks the user to try again.
- Without an English-language instruction, definitions came back in the target language. The prompt now asks for English on every field except translation.
- Saving the same word twice used to risk a duplicate row. Storage replaces the card with the same word and language and keeps its place in the list.
- If Ollama is down, the client raises `LLMUnavailableError` instead of a stack trace. A closed-port test covers that message.

## 6. Next steps (under 1 minute)

Not in this version: spaced repetition with `last_reviewed` and `interval_days`, an Anki CSV export, and SQLite if the list gets large.

## Rehearsal log

Pass 1 and pass 2 used the demo words through `lookup` and `save_flashcard` (bicycle in Spanish, break the ice in Spanish, then a second language). The Study List read those cards back. A Streamlit check covers empty input, a paragraph over 300 characters, and one successful lookup rendered in the app. Speak the outline out loud twice before the presentation so the timing stays inside 10 minutes.
