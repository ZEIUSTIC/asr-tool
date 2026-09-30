"""Command-line interface: python -m asr_tool audio.wav --format srt"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .formats import FORMATTERS
from .transcriber import Transcriber
from .wer import word_error_rate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asr_tool", description="Speech-to-text with Whisper.")
    p.add_argument("audio", help="Path to an audio/video file")
    p.add_argument("-m", "--model", default="base",
                   help="tiny | base | small | medium | large-v3 (default: base)")
    p.add_argument("-l", "--language", default=None, help="Language code, e.g. en, hi, ml (default: auto)")
    p.add_argument("-f", "--format", choices=sorted(FORMATTERS), default="txt")
    p.add_argument("-o", "--output", default=None, help="Output file (default: print to stdout)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--compute-type", default="int8", help="int8 | float16 | float32")
    p.add_argument("--no-vad", action="store_true", help="Disable voice-activity filter")
    p.add_argument("--reference", default=None,
                   help="Path to a ground-truth transcript; prints Word Error Rate")
    return p


def main(argv=None, transcriber: Transcriber | None = None) -> int:
    args = build_parser().parse_args(argv)
    tr = transcriber or Transcriber(args.model, args.device, args.compute_type)
    try:
        start = time.time()
        result = tr.transcribe(args.audio, language=args.language, vad_filter=not args.no_vad)
        elapsed = time.time() - start
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    output = FORMATTERS[args.format](result)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved {args.format.upper()} to {args.output}", file=sys.stderr)
    else:
        print(output)

    print(f"[language={result.language}, audio={result.duration:.1f}s, "
          f"processing={elapsed:.1f}s]", file=sys.stderr)

    if args.reference:
        if not os.path.isfile(args.reference):
            print(f"Error: reference file not found: {args.reference}", file=sys.stderr)
            return 1
        with open(args.reference, encoding="utf-8") as f:
            wer = word_error_rate(f.read(), result.text)
        print(f"WER: {wer:.2%}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
