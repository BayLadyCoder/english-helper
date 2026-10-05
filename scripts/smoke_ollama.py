"""Send one prompt to the local Ollama model and print the reply."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

from config import MODEL_NAME, OLLAMA_HOST


def main() -> None:
    """Ask the model for one English sentence and print it."""
    response = ollama.Client(host=OLLAMA_HOST).chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply with one short English sentence."}],
    )
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
