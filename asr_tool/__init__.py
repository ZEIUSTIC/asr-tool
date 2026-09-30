"""ASR Tool: a small speech-to-text toolkit built on faster-whisper."""
from .formats import Segment, TranscriptionResult
from .transcriber import Transcriber
from .wer import word_error_rate

__all__ = ["Segment", "TranscriptionResult", "Transcriber", "word_error_rate"]
__version__ = "1.0.0"
