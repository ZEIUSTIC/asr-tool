import json, math, struct, wave
from types import SimpleNamespace as NS

import pytest

from asr_tool import Segment, TranscriptionResult, Transcriber, word_error_rate
from asr_tool.cli import main
from asr_tool.formats import format_timestamp, to_srt, to_vtt, to_json, to_txt


class FakeModel:
    """Stands in for faster_whisper.WhisperModel so tests run offline."""
    def transcribe(self, path, language=None, vad_filter=True, beam_size=5):
        segs = [NS(start=0.0, end=1.5, text=" Hello world."),
                NS(start=1.5, end=3.25, text=" This is a test.")]
        return iter(segs), NS(language=language or "en", duration=3.25)


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "tone.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000)))
                               for i in range(16000)))
    return str(p)


@pytest.fixture
def result():
    return TranscriptionResult("en", 3.25, [Segment(0, 1.5, "Hello world."), Segment(1.5, 3.25, "This is a test.")])


def test_timestamp():
    assert format_timestamp(3725.5) == "01:02:05,500"
    assert format_timestamp(0.0, ".") == "00:00:00.000"
    with pytest.raises(ValueError):
        format_timestamp(-1)


def test_formats(result):
    assert to_txt(result) == "Hello world. This is a test.\n"
    assert "1\n00:00:00,000 --> 00:00:01,500\nHello world." in to_srt(result)
    assert to_vtt(result).startswith("WEBVTT")
    data = json.loads(to_json(result))
    assert data["language"] == "en" and len(data["segments"]) == 2


def test_wer():
    assert word_error_rate("the cat sat", "the cat sat") == 0
    assert word_error_rate("the cat sat", "the cat") == pytest.approx(1 / 3)
    assert word_error_rate("Hello, World!", "hello world") == 0
    assert word_error_rate("a b c d", "a x c d e") == pytest.approx(2 / 4)
    with pytest.raises(ValueError):
        word_error_rate("", "x")


def test_transcriber(wav):
    r = Transcriber(model=FakeModel()).transcribe(wav)
    assert r.text == "Hello world. This is a test."
    assert r.duration == 3.25 and len(r.segments) == 2


def test_missing_and_bad_files(tmp_path):
    t = Transcriber(model=FakeModel())
    with pytest.raises(FileNotFoundError):
        t.transcribe("nope.wav")
    bad = tmp_path / "x.txt"; bad.write_text("hi")
    with pytest.raises(ValueError):
        t.transcribe(str(bad))


def test_cli_srt_and_wer(wav, tmp_path, capsys):
    ref = tmp_path / "ref.txt"; ref.write_text("hello world this is a test")
    out = tmp_path / "out.srt"
    code = main([wav, "-f", "srt", "-o", str(out), "--reference", str(ref)],
                transcriber=Transcriber(model=FakeModel()))
    assert code == 0
    assert "Hello world." in out.read_text()
    assert "WER: 0.00%" in capsys.readouterr().err


def test_cli_missing_file(capsys):
    assert main(["missing.wav"], transcriber=Transcriber(model=FakeModel())) == 1
    assert "Error" in capsys.readouterr().err
