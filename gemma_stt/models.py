"""Model registry for gemma-stt.

Resolves short names (e2b / e4b) to local MLX model directories. Paths can be
overridden via environment variables, which is useful if your Gemma 4
checkpoints live somewhere other than the conventional ``~/projects/gemma``
layout this tool was originally built against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Default root where the sibling "model zoo" directory lives. Override with
# GEMMA_STT_MODELS_ROOT if your checkpoints are stored elsewhere.
DEFAULT_MODELS_ROOT = Path(
    os.environ.get("GEMMA_STT_MODELS_ROOT", str(Path.home() / "projects" / "gemma"))
)

# Known-good MLX-format checkpoints (mlx-vlm loads these directly, no
# conversion needed). GGUF files in the same directory are text-only and do
# NOT work here -- see docs/FINDINGS.md.
MODEL_ALIASES = {
    "e2b": os.environ.get(
        "GEMMA_STT_E2B_PATH", str(DEFAULT_MODELS_ROOT / "mlx-gemma-4-e2b")
    ),
    "e4b": os.environ.get(
        "GEMMA_STT_E4B_PATH", str(DEFAULT_MODELS_ROOT / "mlx-gemma-4-e4b")
    ),
}


@dataclass(frozen=True)
class ResolvedModel:
    alias: str
    path: str


def resolve_model(name: str) -> ResolvedModel:
    """Resolve a model alias (e2b/e4b) or raw path to a ResolvedModel.

    Any string not matching a known alias is treated as a literal path (local
    directory or a Hugging Face repo id), so users aren't locked into the
    two built-in checkpoints.
    """
    key = name.lower()
    if key in MODEL_ALIASES:
        path = MODEL_ALIASES[key]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model alias '{key}' points to '{path}', which does not exist. "
                f"Set GEMMA_STT_{key.upper()}_PATH or GEMMA_STT_MODELS_ROOT to override."
            )
        return ResolvedModel(alias=key, path=path)
    return ResolvedModel(alias=name, path=name)


def list_models() -> dict:
    """Return alias -> (path, exists) for display purposes."""
    return {
        alias: {"path": path, "exists": Path(path).exists()}
        for alias, path in MODEL_ALIASES.items()
    }
