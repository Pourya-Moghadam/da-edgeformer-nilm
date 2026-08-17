from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExternalBaseline:
    name: str
    deployment_mode: str
    citation_key: str


EXTERNAL_BASELINES = {
    item.name: item
    for item in [
        ExternalBaseline("STNILM", "static", "Varanasi2024STNILM"),
        ExternalBaseline("AugLPN-NILM", "static", "Yu2024AugLPN"),
        ExternalBaseline("Energformer", "static", "energFormer2023"),
        ExternalBaseline("ConvTransNILM", "static", "JIANG2025116361"),
        ExternalBaseline("Metric Meta-NILM", "sparse_adaptation", "10090473"),
        ExternalBaseline("RTNILM", "transfer_initialized_static", "pan2025rtnilm"),
    ]
}


def load_external_predictions(
    path: str | Path,
    expected_samples: int,
    expected_appliances: list[str],
    expected_visible_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate predictions produced by a separately licensed baseline implementation."""
    payload = np.load(path, allow_pickle=False)
    appliances = payload["appliances"].astype(str).tolist()
    if appliances != expected_appliances:
        raise ValueError(f"appliance order {appliances} != expected {expected_appliances}")
    power = payload["power"]
    state = payload["state"]
    expected_shape = (expected_samples, len(expected_appliances))
    if power.shape != expected_shape or state.shape != expected_shape:
        raise ValueError(f"baseline arrays must have shape {expected_shape}")
    if expected_visible_labels is not None:
        if "visible_labels" not in payload:
            raise ValueError("sparse-adaptation artifact must contain visible_labels")
        np.testing.assert_array_equal(
            payload["visible_labels"].astype(bool), expected_visible_labels
        )
    return power.astype(np.float32), state.astype(bool)
