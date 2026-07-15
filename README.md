# gemma-stt

A standalone speech-to-text (transcription) CLI powered by **Gemma 4's native
audio encoder**, running fully locally on Apple Silicon via
[`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm). No cloud calls, no separate
ASR model (Whisper, etc.) -- Gemma 4's E2B/E4B checkpoints ship a built-in
Conformer-style audio tower that was trained for ASR and speech-to-translated
-text, and this tool just prompts it to transcribe.

This project is a companion to two sibling directories:

- `~/projects/gemma` -- the local Gemma 4 model zoo (GGUF, HF safetensors,
  and MLX checkpoints for E2B/E4B/12B).
- `~/projects/gemmma` -- an MLX-based LoRA fine-tuning pipeline for Gemma 4
  (the `mlxtune` CLI), which is where the original proof-of-concept audio
  script (`scripts/ask_audio.py`) came from.

`gemma-stt` extracts that proof of concept into a dedicated, reusable CLI.

Only the **E2B** and **E4B** checkpoints work with this tool -- see
[Model support](#model-support) below before pointing `--model` at anything
else in `~/projects/gemma`.

## Contents

- [Status](#status)
- [Model support](#model-support)
- [Requirements](#requirements)
- [Install](#install)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Domain-specific prompting](#domain-specific-prompting)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

## Status

Working prototype, validated against both **E2B** and **E4B** MLX
checkpoints. See [`docs/FINDINGS.md`](docs/FINDINGS.md) for accuracy/latency
notes, known issues, and format gotchas (short version: GGUF files do **not**
work for audio -- you need the MLX or HF safetensors checkpoints).

## Model support

Google's own Gemma 4 model cards document audio support on **three**
checkpoints -- E2B, E4B, and 12B "Unified" (see
[`docs/MODEL_SUPPORT.md`](docs/MODEL_SUPPORT.md) for exact citations and
line numbers). This CLI only works with **E2B and E4B**:

| Checkpoint | Officially audio-capable? | Works with `gemma-stt`? |
|---|---|---|
| E2B | Yes (dedicated audio encoder) | **Yes** |
| E4B | Yes (dedicated audio encoder) | **Yes** |
| 12B "Unified" | Yes (different, encoder-free architecture) | **No** -- `mlx_vlm` has no implementation of this architecture |
| 26B A4B / 31B Dense | No (Text + Image only, per Google's own cards) | N/A |

The 12B gap is a tooling limitation, not a broken checkpoint -- its
"encoder-free" design is real and officially benchmarked, `mlx_vlm` (this
tool's inference backend) just doesn't implement it. Full technical
writeup, including direct tensor-level confirmation, in
[`docs/MODEL_SUPPORT.md`](docs/MODEL_SUPPORT.md). The 12B checkpoint is
still perfectly usable for text/coding/reasoning (and for the `gemmma`
fine-tuning pipeline) -- just not for audio, through this CLI.

## Requirements

- macOS + Apple Silicon (mlx-vlm is Metal-based)
- Python >= 3.10
- Local Gemma 4 **E2B or E4B** MLX checkpoints. By default this tool looks
  for `~/projects/gemma/mlx-gemma-4-e2b` and `~/projects/gemma/mlx-gemma-4-e4b`
  -- see the [User Guide](docs/USER_GUIDE.md#configuration-environment-variables)
  if yours live elsewhere.

## Install

```bash
git clone <this-repo> gemma-stt   # or use your existing local checkout
cd gemma-stt
uv venv
uv pip install -e .
# or: make install
```

This pins `mlx==0.31.1` and `mlx-vlm==0.4.4` deliberately -- newer mlx-vlm
releases changed a weight layout in Gemma 4's audio tower and fail to load
these checkpoints. See the [User Guide's Troubleshooting section](docs/USER_GUIDE.md#troubleshooting)
if you hit a shape-mismatch error.

## Usage

```bash
# Transcribe a single file with the smaller/faster E2B model (default)
gemma-stt transcribe path/to/audio.wav

# Use the larger E4B model for better accuracy
gemma-stt transcribe path/to/audio.wav --model e4b

# Transcribe every audio file in a directory, JSON output
gemma-stt transcribe path/to/audio_dir/ --model e4b --output json
```

Supported input formats: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`. Audio does
**not** need to be pre-resampled -- mlx-vlm resamples to the model's
expected 16kHz mono internally.

For the full CLI reference (all flags, environment variables, output JSON
schema, exit codes, E2B-vs-E4B guidance) and a troubleshooting guide, see
**[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**.

### Test fixtures

`tests/fixtures/**/*.wav` are **not committed** -- only the JSON manifests
(ground truth + source citations) are, to keep the repo lean. Fetch the
actual audio with:

```bash
make fixtures            # everything (domains + minds14)
make fixtures-domains    # legal/medical/financial clips only (curl + ffmpeg, no extra deps)
make fixtures-minds14    # E2B-vs-E4B comparison clips (installs the 'fixtures' extra: HF `datasets`)
```

See `scripts/fetch_fixtures.py` and `make help` for details.

## How it works

Gemma 4's chat format supports interleaved content blocks. For audio input,
the prompt looks like:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Transcribe the following audio verbatim..."},
  {"type": "audio"}
]}
```

`gemma_stt/transcribe.py` builds this message, applies the model's chat
template via its `AutoProcessor`, and calls `mlx_vlm.generate(...,
audio=[path])`. The model itself decides how to align its Conformer audio
tower output with the text decoder -- there's no separate alignment/ASR
pipeline involved.

## Domain-specific prompting

Because Gemma 4 is a general multimodal LLM rather than a fixed ASR head,
`--prompt` can be used to add domain context (e.g. "this is a medical
consultation, use correct clinical terminology"). `tests/fixtures/domains/`
has a real, sourced, cited test suite of legal, medical, and financial audio
clips (Supreme Court oral arguments and mock clinical consultations, with
official/professional ground-truth transcripts) used to test this claim
empirically rather than assume it.

**Result: it's a real but narrow effect, not a universal accuracy boost.**
Domain prompting fixed some jargon-homophone errors ("antibodies" ->
"antibiotics") and formatting conventions (adding `$` to dollar figures),
but did not fix rare proper nouns (case names), and on the smaller E2B
model, it once caused a *new* hallucination (fabricating "erectile
dysfunction" that wasn't in the audio at all). Full methodology, results
tables, and sourcing/licensing details for every clip:
**[docs/DOMAIN_SHOWCASE.md](docs/DOMAIN_SHOWCASE.md)**.

## Known limitations

See [`docs/FINDINGS.md`](docs/FINDINGS.md) for the full list, and
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md#troubleshooting) for how to work
around the ones that come up in practice. Headlines:

- No word-level timestamps (the model outputs text only; `.srt` generation
  would require estimating timing from `audio_ms_per_token`, which is
  unverified).
- GGUF checkpoints in `~/projects/gemma` are text-decoder only; audio
  requires MLX or HF safetensors format.
- 12B "Unified" is officially audio-capable but unusable here -- `mlx_vlm`
  has no implementation of its encoder-free architecture. See
  [Model support](#model-support).
- Long audio files may need chunking -- not yet implemented/tested here.

## Contributing

This is a personal/internal tool, built for experimenting with Gemma 4's
local audio capabilities alongside `~/projects/gemma` and `~/projects/gemmma`.
It's not currently soliciting outside contributions, but issues/PRs with
concrete fixes (especially around the open questions in
[`docs/FINDINGS.md`](docs/FINDINGS.md#things-not-tested--open-questions))
are welcome if you're using it too.

## License

[Apache License 2.0](LICENSE)
