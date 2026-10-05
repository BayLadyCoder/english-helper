"""Runtime constants for the local model, languages, and storage path."""

from pathlib import Path

MODEL_NAME = "qwen2.5"
OLLAMA_HOST = "http://localhost:11434"
MAX_INPUT_LENGTH = 300
HISTORY_PATH = Path("data/history.json")
REQUIRED_FIELDS = (
    "senses",
    "sentiment",
)
SENSE_FIELDS = (
    "part_of_speech",
    "definition",
    "synonyms",
    "antonyms",
    "examples",
)
MAX_SENSES = 4
TONE_OPTIONS = (
    "Neutral",
    "Professional",
    "Casual",
    "Gen-Z",
)
MAX_TONE_LENGTH = 1000
MAX_GRAMMAR_LENGTH = 1000
MAX_SPEECH_LENGTH = 1000
KOKORO_REPO = "hexgrad/Kokoro-82M"
TTS_SAMPLE_RATE = 24000
ACCENTS = {
    "American": {
        "lang_code": "a",
        "voices": {
            "Female": "af_heart",
            # "Bella (female)": "af_bella",
            "Male": "am_michael",
            # "Fenrir (male)": "am_fenrir",
        },
    },
    "British": {
        "lang_code": "b",
        "voices": {
            "Female": "bf_emma",
            # "Male": "bm_george",
            "Male": "bm_fable",
        },
    },
}
DEFAULT_ACCENT = "American"
