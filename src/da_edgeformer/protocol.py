from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from da_edgeformer.config import ExperimentConfig


@dataclass(frozen=True)
class Protocol:
    path: Path
    protocol_id: str
    sha256: str
    values: dict[str, Any]


def load_protocol(path: str | Path, config: ExperimentConfig | None = None) -> Protocol:
    protocol_path = Path(path).resolve()
    content = protocol_path.read_bytes()
    values = yaml.safe_load(content)
    if values.get("schema_version") != 1 or not values.get("protocol_id"):
        raise ValueError("unsupported or unnamed protocol")
    protocol = Protocol(
        path=protocol_path,
        protocol_id=str(values["protocol_id"]),
        sha256=hashlib.sha256(content).hexdigest(),
        values=values,
    )
    if config is not None:
        _validate_against_config(protocol, config)
    return protocol


def _validate_against_config(protocol: Protocol, config: ExperimentConfig) -> None:
    architecture = protocol.values["architecture"]
    training = protocol.values["training"]
    expected = {
        "branch_width": config.model.branch_width,
        "head_hidden_width": config.model.head_hidden_width,
        "detector_projection_seed": config.model.detector_projection_seed,
    }
    for key, actual in expected.items():
        if architecture[key] != actual:
            raise ValueError(f"protocol {key}={architecture[key]} but config has {actual}")
    training_expected = {
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "weight_decay": config.training.weight_decay,
        "gradient_clipping_norm": config.training.gradient_clip_norm,
        "seeds": config.training.seeds,
    }
    for key, actual in training_expected.items():
        if training[key] != actual:
            raise ValueError(f"protocol training.{key}={training[key]} but config has {actual}")


def protocol_metadata(protocol: Protocol) -> dict[str, str]:
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
    }
