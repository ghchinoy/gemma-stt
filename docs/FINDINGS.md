# Findings: Gemma 4 as a local STT engine

Pragmatic notes from building and testing `gemma-stt` against the local
Gemma 4 checkpoints in `~/projects/gemma`. Environment: macOS, Apple Silicon,
`mlx-vlm` 0.4.4 / `mlx` 0.31.1.

## TL;DR

- **It works.** Both E2B and E4B correctly transcribe real speech via their
  native audio encoder, with no separate ASR model involved.
- **Version pinning matters.** `mlx-vlm` 0.6.4 (latest at time of testing)
  fails to load the audio tower for these checkpoints — see below. Pin to
  `mlx-vlm==0.4.4` / `mlx==0.31.1`.
- **Format matters.** Only the MLX-format (`mlx-gemma-4-e2b/e4b`) and HF
  safetensors-format (`google-gemma-4-e2b/e4b`) directories work. The
  standalone `.gguf` files in `~/projects/gemma` do **not** — see below.
- **E4B is meaningfully more accurate than E2B** on the (small) test set
  used here, at roughly 2x the latency and ~1.5x the peak memory.

## Test methodology

No local `.wav` fixtures existed in `gemmma/data_audio` (only a `train.jsonl`
referencing filenames from the original PolyAI/MInDS-14 dataset, without the
audio itself). Since that dataset was already cached locally
(`~/.cache/huggingface/datasets/PolyAI___minds14`, used by `gemmma`'s
`mlxtune prep` for audio fine-tuning), 5 real labeled banking-call-center
clips were extracted from it as ground-truth test fixtures
(`tests/fixtures/sample_*.wav` + `manifest.json`), all originally 8kHz mono,
4.5–17s long, all around the `joint_account` intent. mlx-vlm resamples audio
internally, so no manual resampling was needed to feed 8kHz source files in.

Caveat: the MInDS-14 `transcription` field is itself a rough
crowd-transcription and in at least one case (`sample_33`, see below)
appears truncated relative to the actual audio content, so treat differences
from ground truth as directional, not a rigorous WER benchmark.

## Results: E2B vs E4B (MLX format)

| file | duration | ground truth | E2B output | E4B output |
|---|---|---|---|---|
| sample_5.wav | 4.6s | "how to set up a joint account" | "How to set up a **join** account" | "How to set up a **joint** account?" |
| sample_20.wav | 8.3s | "I would like to set up a joint account can I do that in the app" | "I would like to **sort that** joint account. Can I do that in the app?" | "I would like to **set up a** joint account. Can I do that in the app?" |
| sample_0.wav | 10.8s | "I would like to set up a joint account with my partner" | "I would like to set up a joint account with my partner. **How do I proceed with doing that?**" | (identical to E2B) |
| sample_12.wav | 10.9s | "hi I was trying to set up a joint account is there a somewhere on the app that I can do that or do I have to do that through you" | "Hi, I was trying to set up a joint account. Is there somewhere on the app that I can do that or do I have to do that through you?" (matches, cleaned up grammar) | (identical to E2B) |
| sample_33.wav | 17.0s | "hello yes my son is going off to college and I'd like to set up a joint account so that emergency" *(truncated in dataset)* | "...so that we can both access it **but funny into it** for there's emergencies." (garbled) | "...so that we can both access it **and put money into it**. And there's that for any emergencies." (coherent, plausibly correct) |

Takeaways:
- On the shortest/simplest clip (`sample_5`), E2B made a homophone-style
  error ("join" vs "joint") that E4B got right.
- On `sample_20`, E2B substituted a wrong phrase ("sort that") where E4B
  matched ground truth exactly.
- On the longest/hardest clip (`sample_33`), E2B produced a garbled clause
  while E4B produced a coherent, plausible continuation — this is also the
  clip where the dataset's own ground truth is truncated, so the models are
  actually transcribing *more* real speech than the label captures.
- On the two "easy"/clear clips (`sample_0`, `sample_12`), both models
  performed identically, including both appending an extra trailing
  question ("How do I proceed with doing that?") that isn't in the labeled
  ground truth — worth verifying against the actual audio; MInDS-14 labels
  are known to sometimes omit trailing speech.

## Performance (Apple Silicon, this machine)

| model | format | load time | avg generation time (short clips) | peak memory |
|---|---|---|---|---|
| E2B | MLX | ~3–6s | ~0.6–2.1s | ~10.5 GB |
| E4B | MLX | ~5s | ~1.0–4.7s | ~15 GB (approx, not precisely measured) |

Load time and memory scale with checkpoint size (E2B ≈ 9.6GB on disk, E4B ≈
15GB). Both are comfortably usable interactively on a machine with enough
unified memory; E4B roughly doubles generation latency for a modest but real
accuracy improvement.

## Format compatibility

| checkpoint | works with `mlx_vlm.load()`? | notes |
|---|---|---|
| `mlx-gemma-4-e2b/` , `mlx-gemma-4-e4b/` | Yes | Primary target format, fastest to load |
| `google-gemma-4-e2b/` , `google-gemma-4-e4b/` (HF bf16 safetensors) | Yes | mlx_vlm loads HF-format directories directly too; comparable results in spot checks. Emits a `UserWarning` about mel filter bank zero values (`num_mel_filters`=128 vs `num_frequency_bins`=257) — cosmetic, didn't affect transcription quality in testing, but worth monitoring for numerical edge cases with different audio |
| `gemma-4-E2B-it-Q4_K_M.gguf`, `gemma-4-E*-it-qat-q4_0.gguf` | **No** | `mlx_vlm.load()` expects a directory containing `config.json`; a single `.gguf` file path fails immediately with `NotADirectoryError`. Confirmed via direct testing. These GGUF files are also confirmed (from earlier exploration) to contain **text-decoder tensors only** — no `audio_tower`/`vision_tower` weights — so even llama.cpp-based inference would need a separately-extracted `mmproj` file that has not been generated yet in `~/projects/gemma`. |
| `google-gemma-4-12B-it-qat-q4_0-unquantized/` | Untested | Config was hand-patched (`gemma4_unified` → `gemma4`) per earlier exploration and only has 1 audio-related tensor (missing a full audio tower) — likely **not** usable for audio; not tested here to avoid wasting time on a 45GB checkpoint that's already flagged as suspect. |

## mlx-vlm version incompatibility (important)

Installing `gemma-stt` with an unpinned `mlx-vlm>=0.4.4` resolves to the
latest release (0.6.4 at test time) and **fails to load the audio tower**:

```
ValueError: Expected shape (128, 3, 3, 1) but received shape (128, 3, 1, 3)
for parameter audio_tower.subsample_conv_projection.layer0.conv.weight
```

This is a conv weight layout change between mlx-vlm's Gemma 4 audio
implementations across versions — the checkpoints on disk match the layout
mlx-vlm 0.4.4 expects (the same version already used successfully in
`gemmma`'s `.venv` for `scripts/ask_audio.py`), not 0.6.4's. `gemma-stt`
pins `mlx-vlm==0.4.4` and `mlx==0.31.1` in `pyproject.toml` for this reason.
If you need a newer mlx-vlm for other reasons, re-verify audio loading
against these checkpoints first, or plan to reconvert the checkpoints.

## Things not tested / open questions

- **Timestamps / SRT output**: not implemented. `processor_config.json`
  advertises `audio_ms_per_token: 40`, which suggests token-position-based
  timestamp estimation might be feasible, but the mapping from generated
  text tokens back to audio-token positions isn't straightforward with the
  current `mlx_vlm.generate()` API and wasn't investigated here.
- **Long-form audio / chunking**: all test clips were under 17s. The
  processor config caps `audio_seq_length` at 750 tokens (~30s at 40ms/token
  per `Gemma4AudioFeatureExtractor`); longer files likely need chunking that
  isn't implemented in this CLI yet.
- **Non-English / accented speech, background noise, overlapping speakers**:
  not tested — the MInDS-14 sample used is clean, single-speaker, call-center
  audio.
- **12B "Unified" variant**: flagged as likely audio-incomplete based on
  config/tensor inspection; not benchmarked.
- **Quantized MLX checkpoints**: only tested against full bf16-derived MLX
  checkpoints. A 4-bit MLX quantization of the audio tower specifically was
  not tested and might behave differently (faster/smaller, possibly less
  accurate).
