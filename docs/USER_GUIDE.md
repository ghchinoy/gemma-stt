# User Guide

Full command reference, configuration, and troubleshooting for `gemma-stt`.
For a quick pitch and 60-second quickstart, see the top-level
[`README.md`](../README.md). For accuracy/latency test results, see
[`FINDINGS.md`](FINDINGS.md).

## Contents

- [Installation](#installation)
- [CLI reference](#cli-reference)
  - [`gemma-stt transcribe`](#gemma-stt-transcribe)
  - [`gemma-stt models`](#gemma-stt-models)
- [Configuration (environment variables)](#configuration-environment-variables)
- [Output formats](#output-formats)
- [Choosing E2B vs E4B](#choosing-e2b-vs-e4b)
- [Exit codes](#exit-codes)
- [Troubleshooting](#troubleshooting)

## Installation

Requirements:

- macOS on Apple Silicon (mlx-vlm is Metal-based; this will not run on Intel Macs or Linux/Windows)
- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/) (or plain `pip`, if you prefer)
- Local Gemma 4 checkpoints with audio support (see [Configuration](#configuration-environment-variables) below if yours aren't in the default location)

```bash
git clone <this-repo> gemma-stt   # or use your existing local checkout
cd gemma-stt
uv venv
uv pip install -e .
```

This installs the pinned dependency set from `pyproject.toml`, notably
`mlx==0.31.1` and `mlx-vlm==0.4.4` — see
[Troubleshooting](#troubleshooting) for why those exact versions matter.

Verify the install and that your models resolve:

```bash
gemma-stt --version
gemma-stt models
```

## CLI reference

### `gemma-stt transcribe`

```
gemma-stt transcribe AUDIO [AUDIO ...] [OPTIONS]
```

| Argument/Flag | Default | Description |
|---|---|---|
| `audio` (positional, one or more) | required | Audio file path(s), and/or a directory. Directories are expanded to every `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg` file inside them (sorted, non-recursive). |
| `--model` | `e2b` | Model alias (`e2b`, `e4b`), a local directory path, or a Hugging Face repo id. See [Choosing E2B vs E4B](#choosing-e2b-vs-e4b). |
| `--prompt` | see below | Override the instruction sent alongside the audio. Default: `"Transcribe the following audio verbatim. Output only the spoken words as plain text, with no commentary, no speaker labels, and no additional formatting."` |
| `--max-tokens` | `256` | Max tokens to generate. Raise this for long audio; the model will truncate output if it runs out. |
| `--output` | `text` | `text` (human-readable, one block per file) or `json` (structured, includes timing — see [Output formats](#output-formats)). |
| `--out-file` | stdout | Write output to a file instead of printing it. |
| `--verbose` | off | Also print mlx_vlm's own generation debug info (prompt token count, tokens/sec, peak memory) to stdout. |

Examples:

```bash
# Single file, default model (E2B)
gemma-stt transcribe call.wav

# Better accuracy, worse latency
gemma-stt transcribe call.wav --model e4b

# Whole directory, JSON to a file
gemma-stt transcribe recordings/ --model e4b --output json --out-file results.json

# Custom prompt (e.g. ask for translation instead of transcription)
gemma-stt transcribe call.wav --prompt "Transcribe this audio and translate it to English."

# Any other MLX-format Gemma 4 checkpoint with audio support
gemma-stt transcribe call.wav --model /path/to/other/mlx-gemma-4-checkpoint
gemma-stt transcribe call.wav --model some-org/some-gemma4-mlx-repo
```

Model loading happens once per invocation, even with multiple files — pass
all your files in one `transcribe` call rather than looping the CLI in a
shell script, to avoid repeatedly paying the multi-second model load cost.

### `gemma-stt models`

Lists the built-in aliases and whether they currently resolve to an existing
path:

```bash
$ gemma-stt models
 e2b  [OK]  /Users/you/projects/gemma/mlx-gemma-4-e2b
 e4b  [MISSING]  /Users/you/projects/gemma/mlx-gemma-4-e4b
```

Useful as a first check before filing a bug — if a model shows `MISSING`,
fix your paths/env vars before anything else.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `GEMMA_STT_MODELS_ROOT` | `~/projects/gemma` | Parent directory. `e2b`/`e4b` resolve to `<root>/mlx-gemma-4-e2b` and `<root>/mlx-gemma-4-e4b` unless overridden individually. |
| `GEMMA_STT_E2B_PATH` | `<models-root>/mlx-gemma-4-e2b` | Explicit override for the `e2b` alias. |
| `GEMMA_STT_E4B_PATH` | `<models-root>/mlx-gemma-4-e4b` | Explicit override for the `e4b` alias. |

```bash
# Everything under one custom root
export GEMMA_STT_MODELS_ROOT=/Volumes/models/gemma

# Or point each alias somewhere different
export GEMMA_STT_E2B_PATH=/Volumes/models/mlx-gemma-4-e2b
export GEMMA_STT_E4B_PATH=/Volumes/models/mlx-gemma-4-e4b
```

You don't need either alias configured if you always pass `--model
<explicit-path-or-repo-id>` directly.

## Output formats

**`text`** (default) — one block per file, separated by blank lines:

```
# call.wav (1.8s)
Hi, I have a bill to pay.
```

If a file failed, its block shows the error instead of transcript text:

```
# broken.wav
[ERROR] <error message>
```

**`json`** — an array with one object per input file:

```json
[
  {
    "file": "call.wav",
    "text": "Hi, I have a bill to pay.",
    "model": "e2b",
    "model_path": "/Users/you/projects/gemma/mlx-gemma-4-e2b",
    "load_seconds": 3.05,
    "generate_seconds": 1.84,
    "error": null
  }
]
```

`load_seconds` is the same for every entry in a batch (model is loaded once
per invocation, not per file). `error` is `null` on success or a string on
failure — check it programmatically rather than relying on `text` being
empty.

## Choosing E2B vs E4B

Based on the (small) test set in [`FINDINGS.md`](FINDINGS.md):

- **E2B**: faster to load (~3-6s) and generate, lower peak memory (~10.5GB).
  Good default; made a couple of minor word-level errors on harder/longer
  clips in testing.
- **E4B**: ~2x generation latency, higher memory (~15GB), consistently more
  accurate on the same clips, especially on longer/more complex audio.

If accuracy matters more than latency (e.g. offline batch transcription),
default to `--model e4b`. For quick interactive checks, `e2b` (the CLI
default) is usually fine.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All files transcribed successfully |
| `1` | No audio files found, model failed to load, or one or more files errored during transcription |

`--output json` is the more reliable way to distinguish partial failures in
a batch — check each entry's `error` field rather than relying solely on
the process exit code.

## Troubleshooting

**`ValueError: Expected shape (128, 3, 3, 1) but received shape (128, 3, 1, 3) for parameter audio_tower.subsample_conv_projection.layer0.conv.weight`**

You have an incompatible `mlx-vlm` version installed (a newer release than
0.4.4 changed the Gemma 4 audio tower's conv weight layout). Reinstall with
the pinned versions:

```bash
uv pip install "mlx==0.31.1" "mlx-vlm==0.4.4"
```

**`NotADirectoryError: [Errno 20] Not a directory: '.../config.json'`**

You pointed `--model` at a single-file checkpoint (typically a `.gguf`
file). `gemma-stt` requires a checkpoint *directory* containing a
`config.json` — either MLX format (`mlx-gemma-4-e2b/`) or HF safetensors
format (`google-gemma-4-e2b/`). GGUF files also do not currently carry the
audio tower weights needed for this to work even if pointed at a directory
containing one — see [`FINDINGS.md`](FINDINGS.md#format-compatibility).

**`404 Client Error ... Repository Not Found`**

`--model` didn't match a built-in alias (`e2b`/`e4b`) or an existing local
path, so it was treated as a Hugging Face repo id and looked up remotely.
Check for typos, or run `gemma-stt models` to confirm your aliases resolve.

**`UserWarning: At least one mel filter has all zero values`**

Seen when loading the HF safetensors-format checkpoints directly (not the
MLX-converted ones). Cosmetic in testing so far — transcription quality
wasn't visibly affected — but worth watching for edge cases with unusual
input audio (e.g. very low sample rates or heavy clipping).

**Slow / high memory usage**

Expected — these are multi-billion-parameter multimodal models running
locally. E4B in particular needs ~15GB of unified memory headroom. Use
`e2b` if you're memory-constrained, and avoid running other large models
concurrently.

**Long audio produces truncated or incomplete transcripts**

Not yet handled — the audio processor is documented to cap around 750
audio tokens (~30s at 40ms/token). Longer files need chunking, which isn't
implemented in this CLI yet. See [`FINDINGS.md`](FINDINGS.md#things-not-tested--open-questions).
