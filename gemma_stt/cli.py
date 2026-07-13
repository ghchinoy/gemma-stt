"""gemma-stt: transcribe audio locally using Gemma 4's native audio encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gemma_stt import __version__
from gemma_stt.models import list_models, resolve_model
from gemma_stt.transcribe import DEFAULT_PROMPT, GemmaSTT

SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _collect_audio_files(inputs: list[str]) -> list[str]:
    files: list[str] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(
                str(f) for f in sorted(p.iterdir()) if f.suffix.lower() in SUPPORTED_EXTS
            )
        else:
            files.append(str(p))
    return files


def cmd_transcribe(args: argparse.Namespace) -> int:
    audio_files = _collect_audio_files(args.audio)
    if not audio_files:
        print("No audio files found.", file=sys.stderr)
        return 1

    try:
        resolved = resolve_model(args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Loading model '{resolved.alias}' from {resolved.path} ...", file=sys.stderr)
    try:
        engine = GemmaSTT(resolved.path, resolved.alias).load()
    except Exception as e:  # noqa: BLE001 - surface load errors without a raw traceback
        print(f"Error: failed to load model from '{resolved.path}': {e}", file=sys.stderr)
        print(
            "Note: gemma-stt requires an MLX or HF safetensors checkpoint directory "
            "(with config.json). A single-file .gguf path will not work -- see "
            "docs/FINDINGS.md.",
            file=sys.stderr,
        )
        return 1
    print(f"Model loaded in {engine.load_seconds:.1f}s", file=sys.stderr)

    results = []
    for f in audio_files:
        print(f"Transcribing {f} ...", file=sys.stderr)
        result = engine.transcribe(
            f,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        results.append(result)

    if args.output == "json":
        payload = [
            {
                "file": r.file,
                "text": r.text,
                "model": r.model_alias,
                "model_path": r.model_path,
                "load_seconds": round(r.load_seconds, 3),
                "generate_seconds": round(r.generate_seconds, 3),
                "error": r.error,
            }
            for r in results
        ]
        out = json.dumps(payload, indent=2)
    else:
        lines = []
        for r in results:
            if r.error:
                lines.append(f"# {r.file}\n[ERROR] {r.error}\n")
            else:
                lines.append(f"# {r.file} ({r.generate_seconds:.1f}s)\n{r.text}\n")
        out = "\n".join(lines)

    if args.out_file:
        Path(args.out_file).write_text(out)
        print(f"Wrote output to {args.out_file}", file=sys.stderr)
    else:
        print(out)

    return 1 if any(r.error for r in results) else 0


def cmd_models(_args: argparse.Namespace) -> int:
    for alias, info in list_models().items():
        status = "OK" if info["exists"] else "MISSING"
        print(f"{alias:>4}  [{status}]  {info['path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma-stt",
        description="Local speech-to-text using Gemma 4's native audio encoder (via mlx-vlm).",
    )
    parser.add_argument("--version", action="version", version=f"gemma-stt {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    p_tx = sub.add_parser("transcribe", help="Transcribe one or more audio files")
    p_tx.add_argument(
        "audio", nargs="+", help="Audio file(s) or directory of audio files (.wav, .flac, .mp3, .m4a, .ogg)"
    )
    p_tx.add_argument(
        "--model",
        default="e2b",
        help="Model alias ('e2b', 'e4b') or a path/HF repo id to an MLX Gemma 4 checkpoint. Default: e2b",
    )
    p_tx.add_argument("--prompt", default=DEFAULT_PROMPT, help="Override the transcription prompt")
    p_tx.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    p_tx.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    p_tx.add_argument("--out-file", help="Write output to this file instead of stdout")
    p_tx.add_argument("--verbose", action="store_true", help="Show mlx_vlm generation timing metrics")
    p_tx.set_defaults(func=cmd_transcribe)

    p_models = sub.add_parser("models", help="List configured model aliases and their resolved paths")
    p_models.set_defaults(func=cmd_models)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
