from __future__ import annotations

import hashlib

import numpy as np


def deterministic_label_mask(length: int, fraction: float, seed: int, stream_id: str) -> np.ndarray:
    """Choose exactly round(fraction * length) post-prediction labels reproducibly."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    digest = hashlib.sha256(f"{seed}:{stream_id}".encode()).digest()
    derived_seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(derived_seed)
    count = int(round(fraction * length))
    mask = np.zeros(length, dtype=bool)
    if count:
        mask[rng.choice(length, count, replace=False)] = True
    return mask
