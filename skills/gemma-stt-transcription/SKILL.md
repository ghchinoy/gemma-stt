---
name: gemma-stt-transcription
description: Transcribes audio (.wav/.flac/.mp3/.m4a/.ogg) locally using Gemma 4's native audio encoder via the gemma-stt CLI. Use when asked to transcribe, caption, or extract text from speech/audio files, or about local/offline speech-to-text with Gemma models.
compatibility: Requires macOS on Apple Silicon, the gemma-stt CLI installed (uv pip install -e .), and local Gemma 4 E2B or E4B MLX checkpoints.
license: Apache-2.0
---

# Gemma 4 Speech-to-Text (gemma-stt) Skill

This skill provides instructions for using the `gemma-stt` CLI to transcribe
audio locally on Apple Silicon, using Gemma 4's own native audio encoder
(no Whisper, no cloud calls, no separate ASR model). Gemma 4 is a general
multimodal LLM prompted to transcribe -- not a dedicated ASR head -- which
is why prompt choice matters more here than with a typical STT tool.

## ⚠️ The One Thing to Get Right: Model Selection

**Only `--model e2b` and `--model e4b` work.** Do not use `--model` with any
other local Gemma 4 checkpoint (12B, 26B A4B, 31B), even though Google's own
model cards say 12B "Unified" is also audio-capable -- it uses a different,
encoder-free architecture that `mlx_vlm` (this CLI's inference backend)
doesn't implement. Pointing `--model` at it will fail with a missing/
shape-mismatched audio-tower error, not a helpful message telling you why.
Full reasoning: [`docs/MODEL_SUPPORT.md`](../../docs/MODEL_SUPPORT.md).

- `e2b`: faster, lower memory, the CLI default. Good for quick/interactive use.
- `e4b`: ~2x slower, more accurate, especially on longer/harder audio. Prefer
  this for anything where accuracy matters more than latency.

## Step-by-Step Workflow

### 1. Verify the environment
```bash
gemma-stt models
```
Both `e2b` and `e4b` should show `[OK]`. If either shows `[MISSING]`, fix
`GEMMA_STT_MODELS_ROOT`/`GEMMA_STT_E2B_PATH`/`GEMMA_STT_E4B_PATH` before
doing anything else -- see
[`docs/USER_GUIDE.md`](../../docs/USER_GUIDE.md#configuration-environment-variables).

### 2. Transcribe
```bash
# Single file
gemma-stt transcribe path/to/audio.wav --model e4b

# Every audio file in a directory
gemma-stt transcribe path/to/audio_dir/ --model e4b

# Structured output (includes per-file timing, easy to parse)
gemma-stt transcribe path/to/audio.wav --model e4b --output json
```
Supported formats: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`. No manual
resampling needed -- mlx-vlm resamples to 16kHz mono internally.

### 3. (Optional) Add domain context to the prompt
`--prompt` accepts any instruction, so you can bias transcription toward a
known domain:
```bash
gemma-stt transcribe consult.wav --model e4b \
  --prompt "Transcribe the following audio verbatim. This is a doctor-patient \
medical consultation. Use correct clinical and pharmaceutical terminology."
```
**Caveat -- this was tested empirically, not assumed to work:** domain
prompting fixed some real errors (a jargon homophone, dollar-figure
formatting) but did **not** fix rare proper nouns, and on the smaller E2B
model it once caused a *new* hallucination (fabricated clinical detail not
in the audio at all). Treat it as a targeted fix for an error you've
actually observed, not a default-on accuracy boost, especially on E2B. Full
results: [`docs/DOMAIN_SHOWCASE.md`](../../docs/DOMAIN_SHOWCASE.md).

## Common Edge Cases & Workarounds

- **`ValueError: Expected shape ... audio_tower.subsample_conv_projection...`**
  Wrong `mlx-vlm`/`mlx` version installed. This project pins
  `mlx==0.31.1` / `mlx-vlm==0.4.4` in `pyproject.toml` deliberately -- newer
  releases changed the audio-tower weight layout. Reinstall the pinned
  versions rather than upgrading.
- **`NotADirectoryError: .../config.json`**
  `--model` was pointed at a single-file checkpoint (typically a `.gguf`).
  This CLI needs a checkpoint *directory* (MLX or HF safetensors format).
  GGUF files in `~/projects/gemma` are also text-decoder only regardless.
- **Long audio produces truncated/incomplete output**
  No chunking is implemented; the audio processor caps around ~30s
  (750 tokens at 40ms/token). Split longer files before transcribing.
- **No timestamps, diarization, or live/streaming transcription**
  This CLI is batch-file-only and returns plain text (or JSON with
  per-file timing, not word-level timestamps). Don't promise these
  capabilities to a user without checking first.

## See Also

- [`README.md`](../../README.md) -- overview, install, quickstart
- [`docs/USER_GUIDE.md`](../../docs/USER_GUIDE.md) -- full CLI reference, env vars, output schema, exit codes, troubleshooting
- [`docs/MODEL_SUPPORT.md`](../../docs/MODEL_SUPPORT.md) -- why only E2B/E4B work, with citations to Google's model cards
- [`docs/FINDINGS.md`](../../docs/FINDINGS.md) -- accuracy/latency data behind the E2B-vs-E4B guidance
- [`docs/DOMAIN_SHOWCASE.md`](../../docs/DOMAIN_SHOWCASE.md) -- full domain-prompting methodology and results
