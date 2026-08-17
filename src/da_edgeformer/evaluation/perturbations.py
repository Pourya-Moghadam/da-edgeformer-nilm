from __future__ import annotations

import numpy as np


def apply_controlled_shift(
    aggregate: np.ndarray,
    start: int,
    kind: str,
    magnitude: float,
    seed: int = 0,
) -> np.ndarray:
    """Secondary detector diagnostics; targets are intentionally unchanged."""
    shifted = np.asarray(aggregate, dtype=np.float32).copy()
    if not 0 <= start < len(shifted):
        raise ValueError("start must index the aggregate series")
    if kind == "gain":
        shifted[start:] *= magnitude
    elif kind == "noise":
        rng = np.random.default_rng(seed)
        shifted[start:] += rng.normal(0.0, magnitude, len(shifted) - start)
    elif kind == "missing_block":
        length = min(int(magnitude), len(shifted) - start)
        shifted[start : start + length] = np.nan
    elif kind == "gradual_gain":
        ramp = np.linspace(1.0, magnitude, len(shifted) - start)
        shifted[start:] *= ramp
    else:
        raise ValueError(f"unknown controlled shift: {kind}")
    return shifted
