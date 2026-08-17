from __future__ import annotations

import hashlib

import numpy as np

from da_edgeformer.adaptation.controller import TokenBucketController
from da_edgeformer.config import ControllerConfig


def maximum_update_indices(
    timestamps: np.ndarray, label_mask: np.ndarray, config: ControllerConfig
) -> np.ndarray:
    """Return the deterministic maximum-rate update schedule allowed by the gate."""
    if len(timestamps) != len(label_mask):
        raise ValueError("timestamps and label_mask must have equal length")
    controller = TokenBucketController(config)
    new_labels = 0
    indices: list[int] = []
    for index, (timestamp, visible) in enumerate(zip(timestamps, label_mask, strict=True)):
        new_labels += int(visible)
        decision = controller.decide(
            float(timestamp), new_labels >= config.minimum_new_labels, new_labels
        )
        if decision.permitted:
            indices.append(index)
            new_labels = 0
    return np.asarray(indices, dtype=np.int64)


def matched_update_schedule(
    timestamps: np.ndarray,
    label_mask: np.ndarray,
    target_count: int,
    mode: str,
    config: ControllerConfig,
    seed: int,
    stream_id: str,
) -> set[int]:
    """Choose exactly ``target_count`` gate-feasible periodic or seeded-random alarms."""
    if target_count < 0:
        raise ValueError("target_count cannot be negative")
    if mode not in {"periodic", "random"}:
        raise ValueError("matched schedules are defined only for periodic and random triggers")
    eligible = maximum_update_indices(timestamps, label_mask, config)
    if target_count > len(eligible):
        raise ValueError(
            f"requested {target_count} updates, but only {len(eligible)} are gate-feasible"
        )
    if target_count == 0:
        return set()
    if mode == "periodic":
        positions = np.linspace(0, len(eligible) - 1, target_count, dtype=np.int64)
    else:
        digest = hashlib.sha256(f"matched:{seed}:{stream_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        positions = np.sort(rng.choice(len(eligible), target_count, replace=False))
    return {int(eligible[position]) for position in positions}
