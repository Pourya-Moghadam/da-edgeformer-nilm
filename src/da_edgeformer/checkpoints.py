from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from da_edgeformer.config import ExperimentConfig
from da_edgeformer.models.edgeformer import DAEdgeFormer


def save_checkpoint(
    path: str | Path,
    model: DAEdgeFormer,
    config: ExperimentConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": config.to_dict(), "metadata": metadata or {}}, path
    )


def load_model_checkpoint(
    path: str | Path, model: DAEdgeFormer, device: torch.device
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    return payload.get("metadata", {})
