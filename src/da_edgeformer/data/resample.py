from __future__ import annotations

import numpy as np


def time_weighted_mean(
    timestamps: np.ndarray,
    values: np.ndarray,
    grid_seconds: int,
    native_interval_seconds: float,
    max_gap_intervals: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a piecewise-constant signal without bridging long gaps.

    Each observation is held until the next timestamp only when that interval
    is no longer than ``max_gap_intervals * native_interval_seconds``.
    Empty bins are NaN and are never interpolated.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or values.ndim != 1 or len(timestamps) != len(values):
        raise ValueError("timestamps and values must be one-dimensional and equal length")
    if len(timestamps) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[order]
    keep = np.r_[np.diff(timestamps) > 0, True]
    timestamps, values = timestamps[keep], values[keep]
    if len(timestamps) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    start = int(np.floor(timestamps[0] / grid_seconds) * grid_seconds)
    stop = int(np.ceil(timestamps[-1] / grid_seconds) * grid_seconds)
    edges = np.arange(start, stop + grid_seconds, grid_seconds, dtype=np.int64)
    weighted = np.zeros(len(edges) - 1, dtype=np.float64)
    duration = np.zeros(len(edges) - 1, dtype=np.float64)
    max_gap = native_interval_seconds * max_gap_intervals

    for left, right, value in zip(timestamps[:-1], timestamps[1:], values[:-1], strict=True):
        if not np.isfinite(value) or right <= left or right - left > max_gap:
            continue
        cursor = left
        while cursor < right:
            index = int((cursor - start) // grid_seconds)
            if index < 0 or index >= len(weighted):
                break
            boundary = min(right, start + (index + 1) * grid_seconds)
            span = boundary - cursor
            weighted[index] += value * span
            duration[index] += span
            cursor = boundary

    result = np.full_like(weighted, np.nan, dtype=np.float64)
    valid = duration > 0
    result[valid] = weighted[valid] / duration[valid]
    return edges[:-1], result.astype(np.float32)
