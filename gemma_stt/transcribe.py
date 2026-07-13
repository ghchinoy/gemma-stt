"""Core transcription logic: load a Gemma 4 MLX model once, transcribe N files."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_PROMPT = (
    "Transcribe the following audio verbatim. "
    "Output only the spoken words as plain text, with no commentary, "
    "no speaker labels, and no additional formatting."
)


@dataclass
class TranscriptionResult:
    file: str
    text: str
    model_alias: str
    model_path: str
    prompt: str
    load_seconds: float
    generate_seconds: float
    error: Optional[str] = None
    raw_response: str = field(default="", repr=False)


class GemmaSTT:
    """Thin wrapper around mlx_vlm that keeps a model loaded across calls."""

    def __init__(self, model_path: str, model_alias: str):
        self.model_path = model_path
        self.model_alias = model_alias
        self._model = None
        self._processor = None
        self.load_seconds = 0.0

    def load(self):
        from mlx_vlm import load  # imported lazily so --help stays fast

        start = time.perf_counter()
        self._model, self._processor = load(self.model_path)
        self.load_seconds = time.perf_counter() - start
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        audio_path: str,
        prompt: str = DEFAULT_PROMPT,
        max_tokens: int = 256,
        verbose: bool = False,
    ) -> TranscriptionResult:
        if not self.loaded:
            self.load()

        from mlx_vlm import generate

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio"},
                ],
            }
        ]

        try:
            formatted_prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = f"User: {prompt} <audio>\nAssistant:"

        start = time.perf_counter()
        try:
            response = generate(
                self._model,
                self._processor,
                prompt=formatted_prompt,
                audio=[audio_path],
                max_tokens=max_tokens,
                verbose=verbose,
            )
            gen_seconds = time.perf_counter() - start
            text = _clean_response(response)
            return TranscriptionResult(
                file=audio_path,
                text=text,
                model_alias=self.model_alias,
                model_path=self.model_path,
                prompt=prompt,
                load_seconds=self.load_seconds,
                generate_seconds=gen_seconds,
                raw_response=str(response),
            )
        except Exception as e:  # noqa: BLE001 - surface any mlx_vlm/model error to caller
            gen_seconds = time.perf_counter() - start
            return TranscriptionResult(
                file=audio_path,
                text="",
                model_alias=self.model_alias,
                model_path=self.model_path,
                prompt=prompt,
                load_seconds=self.load_seconds,
                generate_seconds=gen_seconds,
                error=str(e),
            )


def _clean_response(response) -> str:
    """mlx_vlm's generate() may return a str or a GenerationResult-like object."""
    text = getattr(response, "text", None)
    if text is None:
        text = str(response)
    return text.strip()
