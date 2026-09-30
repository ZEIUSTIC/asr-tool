# ASR Tool – Automatic Speech Recognition (Whisper)

A command-line speech-to-text tool. It transcribes audio/video files to text, subtitles (SRT/VTT) or JSON, auto-detects the spoken language, and can score its own accuracy with **Word Error Rate (WER)** against a reference transcript.

## Features
- Transcribe `.wav .mp3 .m4a .flac .ogg .opus .mp4 .webm .mkv`
- Automatic language detection (or force one with `--language`)
- Output formats: `txt`, `srt`, `vtt`, `json` (with timestamps)
- Voice-activity detection to skip silence
- Built-in WER evaluation (`--reference`)
- Runs on CPU (int8) or GPU (CUDA)
- Unit-tested with a mocked model (runs offline, no download needed)

## Tools & Technologies
| Purpose | Technology |
|---|---|
| Language | Python 3.9+ |
| ASR model | OpenAI **Whisper** (open-source) |
| Inference engine | **faster-whisper** (CTranslate2 backend) |
| Audio decoding | PyAV (bundled with faster-whisper, no separate FFmpeg install needed) |
| VAD | Silero VAD (bundled with faster-whisper) |
| Testing | pytest |
| Version control | Git / GitHub |

## Project Structure
```
asr-tool/
├── asr_tool/
│   ├── __init__.py
│   ├── __main__.py        # enables `python -m asr_tool`
│   ├── cli.py             # command-line interface
│   ├── transcriber.py     # Whisper wrapper
│   ├── formats.py         # txt / srt / vtt / json writers
│   └── wer.py             # Word Error Rate metric
├── tests/test_asr.py
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Installation
```bash
git clone https://github.com/ZEIUSTIC/asr-tool.git
cd asr-tool

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```
The first run downloads the chosen Whisper model (~140 MB for `base`) and caches it.

## Usage
```bash
# Print transcript to the terminal
python -m asr_tool sample.wav

# Save subtitles
python -m asr_tool sample.mp3 -f srt -o sample.srt

# Choose a bigger model and force the language (Malayalam)
python -m asr_tool talk.m4a -m small -l ml -f json -o talk.json

# Measure accuracy against a ground-truth transcript
python -m asr_tool sample.wav --reference sample_truth.txt
```

### Options
| Flag | Description | Default |
|---|---|---|
| `-m, --model` | `tiny`, `base`, `small`, `medium`, `large-v3` | `base` |
| `-l, --language` | Language code (`en`, `hi`, `ml`, …) | auto-detect |
| `-f, --format` | `txt`, `srt`, `vtt`, `json` | `txt` |
| `-o, --output` | Output file | stdout |
| `--device` | `auto`, `cpu`, `cuda` | `auto` |
| `--compute-type` | `int8`, `float16`, `float32` | `int8` |
| `--no-vad` | Disable silence filtering | off |
| `--reference` | Reference transcript → prints WER | – |

### Use as a library
```python
from asr_tool import Transcriber

result = Transcriber("base").transcribe("sample.wav")
print(result.language, result.text)
for seg in result.segments:
    print(seg.start, seg.end, seg.text)
```

## Testing
```bash
pip install -r requirements-dev.txt
python -m pytest -v
```
Tests cover timestamp formatting, all four output formats, WER calculation, input validation, and the CLI end to end. They use a fake model, so no download or internet is required.

**Real-audio check:** record a short clip (e.g. read a sentence aloud, save as `sample.wav`), write what you said into `sample_truth.txt`, then run:
```bash
python -m asr_tool sample.wav --reference sample_truth.txt
```
Clear English speech with the `base` model typically gives a WER under ~10%. Use `small`/`medium` for noisier audio or other languages.

## How it works
1. Validate the input file and extension.
2. Decode audio to 16 kHz mono, filter silence with VAD.
3. Whisper (encoder–decoder Transformer) predicts text tokens with timestamps, using beam search (beam = 5).
4. Segments are formatted as TXT/SRT/VTT/JSON.
5. Optionally, WER = (substitutions + deletions + insertions) / reference words, computed via Levenshtein distance.

## Limitations
- Accuracy drops with heavy noise, overlapping speakers or strong accents; use a larger model.
- No speaker diarization (who said what).
- Not real-time streaming; it processes complete files.

## License
MIT
## Test Results
- Model: base (CPU, macOS)
- Audio: 15 s English recording
- Transcript: "<your output>"
- WER: X.XX%
