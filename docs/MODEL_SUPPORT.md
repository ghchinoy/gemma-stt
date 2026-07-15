# Gemma 4 model support: what's official vs. what works in this CLI

This CLI uses `mlx_vlm` for local Apple Silicon (MLX/Metal) inference. That
is a narrower set of models than "everything Google's Gemma 4 model cards
say is multimodal" — this doc separates the two clearly, with citations to
the actual model cards bundled alongside the checkpoints in
`~/projects/gemma`, so future-you (or anyone else) doesn't have to
re-derive this.

## Official capability matrix (per Google's own model cards)

The most complete/current card is the one bundled with the 12B checkpoint
(`~/projects/gemma/google-gemma-4-12B-it-qat-q4_0-unquantized/README.md`),
which documents the full 5-size family. The E2B/E4B-bundled cards are an
earlier snapshot from before the 12B model existed and say "audio: E2B and
E4B only" — treat the 12B card as authoritative since it's the newer,
complete one.

| Model | Supported modalities | Audio encoder | Source |
|---|---|---|---|
| E2B | Text, Image, Audio | ~300M params, dedicated Conformer-style encoder | `google-gemma-4-12B-it-qat-q4_0-unquantized/README.md:60-69` |
| E4B | Text, Image, Audio | ~300M params, dedicated Conformer-style encoder | same |
| **12B "Unified"** | Text, Image, Audio | **None -- encoder-free**, direct waveform-to-embedding projection | same |
| 26B A4B (MoE) | Text, Image only | N/A -- no audio, ever | `README.md:77-86` (no Audio row in modality table) |
| 31B Dense | Text, Image only | N/A -- "No Audio" | `README.md:60-69` (`Audio Encoder Parameters: No Audio`) |

Direct quotes (line numbers from `google-gemma-4-12B-it-qat-q4_0-unquantized/README.md`):

> Line 32: "Gemma 4 models are multimodal, handling text and image input (with audio supported on E2B, E4B, and 12B)..."
>
> Line 73: "The 'Unified' in Gemma 4 12B Unified refers to its encoder-free architecture. Other Gemma 4 models use dedicated encoders to process multimodal data before passing it to the LLM. Gemma 4 12B eliminates these encoders entirely, projecting raw image patches and audio waveforms directly into the LLM's embedding space through lightweight linear layers."
>
> Line 132: "Audio (E2B, E4B, and 12B only) -- Automatic speech recognition (ASR) and speech-to-translated-text translation across multiple languages."

The benchmark table (`README.md:100-104`) backs this up with real numbers,
not just marketing copy -- 12B Unified actually scores *better* than E4B on
FLEURS:

| | 12B Unified | E4B | E2B |
|---|---|---|---|
| CoVoST | 38.5 | 35.54 | 33.47 |
| FLEURS (lower is better) | 0.069 | 0.08 | 0.09 |

26B A4B and 31B Dense show `-` (not evaluated / not applicable) on both
audio rows.

**Bottom line from Google's own docs: audio is officially supported on
E2B, E4B, *and* 12B.** The 12B just does it with a fundamentally different,
encoder-free architecture instead of a dedicated audio tower.

## Why this CLI only supports E2B and E4B today

This is a tooling gap, not a model limitation. Specifically:

1. **`mlx_vlm` has no `gemma4_unified` implementation.** Its
   `mlx_vlm/models/` directory ships `gemma3`, `gemma3n`, `gemma4`,
   `paligemma`, and dozens of other architectures -- but no
   `gemma4_unified`. Confirmed by directly listing the installed package's
   model directory; there is no code path for the encoder-free design 12B
   uses.
2. **HF `transformers` *does* have a real implementation** -- a
   `gemma4_unified` model package exists
   (`transformers/models/gemma4_unified/modeling_gemma4_unified.py`,
   auto-generated from `modular_gemma4_unified.py`, alongside
   `gemma4_unified_assistant`). So 12B's audio path is real and loadable in
   principle -- just via a **PyTorch** backend (`import torch` at the top
   of that file), not MLX. Using it would mean a completely separate
   pipeline from this CLI (plain `transformers` + `torch`, CPU/MPS, no
   Metal-optimized MLX inference), which is out of scope for `gemma-stt` as
   designed.
3. **The local 12B checkpoint's `config.json` was hand-patched.** The
   original (`config.json.bak`) correctly declares
   `Gemma4UnifiedForConditionalGeneration` / `model_type: gemma4_unified`.
   The active `config.json` was edited to claim the plain `gemma4`
   architecture instead -- almost certainly to satisfy
   `convert_hf_to_gguf.py`, which only recognizes `gemma4`. Any tool that
   trusts the active config (including a hypothetical `transformers`/
   PyTorch attempt) would be routed to the *wrong* model class -- the
   encoder-based `Gemma4ForConditionalGeneration`, which expects hundreds
   of `audio_tower.*` weights this checkpoint doesn't have, since it was
   never meant to have them.
4. **The checkpoint's actual tensors confirm the "encoder-free" design is
   real, not incomplete.** Direct inspection of
   `model.safetensors`'s header shows exactly one audio-related tensor
   (`model.embed_audio.embedding_projection.weight`) and exactly one
   vision-related tensor (`model.embed_vision.embedding_projection.weight`)
   -- precisely matching the model card's "lightweight linear layers"
   description. This is not a broken or stripped-down checkpoint; it's
   correctly saved for the architecture Google describes. It just isn't an
   architecture any tool in this project's stack (`mlx_vlm`) can run.
5. **No MLX-converted copy of 12B exists locally** (`~/projects/gemma` has
   `mlx-gemma-4-e2b` and `mlx-gemma-4-e4b`, but no `mlx-gemma-4-12b`), and
   the HF checkpoint is a single unsharded 23.8GB bf16 `model.safetensors`
   file -- a tight fit even for text-only use on a 32GB unified-memory
   Mac, before considering that `mlx_vlm.load()` defaults to eager
   (non-lazy) loading.

### If you ever want 12B's audio specifically

It would need a different toolchain than `gemma-stt`: restore
`config.json.bak` -> `config.json` in the 12B checkpoint directory, install
`transformers`, `torch`, `torchvision`, `librosa`, `accelerate`, and use
`AutoModelForMultimodalLM` directly (per the pattern documented in that
checkpoint's own bundled `README.md`). This is not implemented here and
not currently planned -- flagging it only so the option is documented if a
future need arises.

## Practical recommendation

- **Use `--model e2b` or `--model e4b` with `gemma-stt`** for audio --
  these are the only Gemma 4 checkpoints this CLI can actually run, and
  they're the ones validated in [`FINDINGS.md`](FINDINGS.md).
- **Use the 12B checkpoint for text, coding, and reasoning** (via the
  [gemma4-tuning](https://github.com/ghchinoy/gemma4-tuning) fine-tuning
  pipeline or a plain-text MLX/transformers workflow) -- its text decoder
  is intact and unaffected by any of the above; only its multimodal
  (audio/vision) path is unusable in this project's toolchain today.
- **Don't point `gemma-stt --model` at the 12B directory.** It will fail
  to load with a missing/shape-mismatched audio tower error (the same
  class of error documented in [`FINDINGS.md`](FINDINGS.md)'s mlx-vlm
  version-mismatch section, but for a structural reason this time, not a
  version pin).
- **26B A4B and 31B Dense are text+image only, per Google's own cards** --
  not audio-capable at all, regardless of tooling. Not relevant to this
  CLI under any circumstance.
