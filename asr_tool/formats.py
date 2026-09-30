"""Data structures and output formatters (txt, srt, vtt, json)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    duration: float
    segments: List[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """Convert seconds to HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(result: TranscriptionResult) -> str:
    return result.text + "\n"


def to_srt(result: TranscriptionResult) -> str:
    blocks = []
    for i, seg in enumerate(result.segments, 1):
        blocks.append(
            f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n"
        )
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(
            f"{format_timestamp(seg.start, '.')} --> {format_timestamp(seg.end, '.')}"
        )
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def to_json(result: TranscriptionResult) -> str:
    payload = {
        "language": result.language,
        "duration": result.duration,
        "text": result.text,
        "segments": [asdict(s) for s in result.segments],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


FORMATTERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}
