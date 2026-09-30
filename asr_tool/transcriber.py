"""Core transcription engine wrapping faster-whisper (OpenAI Whisper)."""
from __future__ import annotations

import os
from typing import Optional

from .formats import Segment, TranscriptionResult

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm", ".mkv"}


class Transcriber:
    """Loads a Whisper model lazily and transcribes audio files.

    A pre-built `model` may be injected (used by the unit tests).
    """

    def __init__(self, model_size: str = "base", device: str = "auto",
                 compute_type: str = "int8", model=None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import: heavy
            self._model = WhisperModel(self.model_size, device=self.device,
                                       compute_type=self.compute_type)
        return self._model

    @staticmethod
    def validate_audio_path(path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    def transcribe(self, path: str, language: Optional[str] = None,
                   vad_filter: bool = True) -> TranscriptionResult:
        self.validate_audio_path(path)
        segments, info = self.model.transcribe(
            path, language=language, vad_filter=vad_filter, beam_size=5
        )
        return TranscriptionResult(
            language=info.language,
            duration=float(info.duration),
            segments=[Segment(s.start, s.end, s.text.strip()) for s in segments],
        )
