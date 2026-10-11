"""Local text-to-speech with the Kokoro model."""

from functools import lru_cache

import numpy as np

from config import ACCENTS, KOKORO_REPO, TTS_SAMPLE_RATE


@lru_cache(maxsize=len(ACCENTS))
def _pipeline(lang_code: str):
    """Load the Kokoro pipeline once per accent; the first call downloads the model."""
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO)


def synthesize(text: str, accent: str, voice: str) -> tuple[np.ndarray, int]:
    """Return mono audio samples and their sample rate for the given text."""
    pipeline = _pipeline(ACCENTS[accent]["lang_code"])
    chunks = [result.audio.numpy() for result in pipeline(text, voice=voice) if result.audio is not None]
    if not chunks:
        raise ValueError("Nothing to pronounce. Please enter some words.")
    return np.concatenate(chunks), TTS_SAMPLE_RATE


def synthesize_parts(
    parts: list[tuple[str, str]],
    accent: str,
    pause_seconds: float = 0.35,
) -> tuple[np.ndarray, int]:
    """Read each (text, voice) part in turn and join them with a short pause."""
    clips: list[np.ndarray] = []
    sample_rate = TTS_SAMPLE_RATE
    for text, voice in parts:
        samples, sample_rate = synthesize(text, accent, voice)
        if clips:
            clips.append(np.zeros(int(sample_rate * pause_seconds), dtype=samples.dtype))
        clips.append(samples)
    if not clips:
        raise ValueError("Nothing to pronounce. Please enter some words.")
    return np.concatenate(clips), sample_rate
