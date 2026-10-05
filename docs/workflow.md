# EnglishHelper workflow

The user types a word, phrase, or sentence and picks a target language. Streamlit validates the input. The app builds a prompt and calls the local Ollama model with `format="json"`. The model returns one JSON object. The app parses and validates that object, renders a flashcard, and appends it to `data/flashcards.json`. The Study List tab reads that file.

The model runs only inside `lookup` and `call_model` in `llm_client.py`. JSON mode makes valid JSON more likely. `parse_flashcard` still checks the keys and types, and `lookup` retries once before showing an error.

```mermaid
flowchart TD
    A[User enters word/phrase/sentence + target language] --> B{Input valid?}
    B -- No --> A
    B -- Yes --> C[Build prompt from template]
    C --> D[Call Ollama - local LLM<br/>format=json]
    D --> E{Valid JSON response?}
    E -- No / error --> F[Show friendly error,<br/>offer retry]
    F --> A
    E -- Yes --> G[Render flashcard:<br/>definition, synonyms/antonyms,<br/>examples, sentiment, translation]
    G --> H[Auto-save to data/flashcards.json]
    H --> I[Study List view reads JSON file]
    I --> J[User browses saved flashcards]
```
