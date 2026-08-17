from __future__ import annotations

import numpy as np


def _confusion(true: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    true = true.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.sum(true & predicted))
    tn = int(np.sum(~true & ~predicted))
    fp = int(np.sum(~true & predicted))
    fn = int(np.sum(true & ~predicted))
    return tp, tn, fp, fn


def binary_f1(true: np.ndarray, predicted: np.ndarray) -> float:
    tp, _, fp, fn = _confusion(true, predicted)
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def binary_mcc(true: np.ndarray, predicted: np.ndarray) -> float:
    tp, tn, fp, fn = _confusion(true, predicted)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator == 0 else float((tp * tn - fp * fn) / denominator)


def nilm_metrics(
    true_power: np.ndarray,
    predicted_power: np.ndarray,
    true_state: np.ndarray,
    predicted_state: np.ndarray,
    target_mask: np.ndarray | None = None,
) -> dict[str, float | list[float]]:
    true_power = np.asarray(true_power, dtype=np.float64)
    predicted_power = np.asarray(predicted_power, dtype=np.float64)
    mask = (
        np.ones_like(true_power, dtype=bool)
        if target_mask is None
        else np.asarray(target_mask, dtype=bool)
    )
    absolute_error = np.where(mask, np.abs(true_power - predicted_power), np.nan)
    counts = mask.sum(axis=0)
    per_appliance_mae = np.divide(
        np.nansum(absolute_error, axis=0),
        counts,
        out=np.full(true_power.shape[1], np.nan),
        where=counts > 0,
    )
    mean_power = np.divide(
        np.where(mask, np.abs(true_power), 0.0).sum(axis=0),
        counts,
        out=np.full(true_power.shape[1], np.nan),
        where=counts > 0,
    )
    scale = np.maximum(mean_power, 1e-8)
    nmae = per_appliance_mae / scale
    available = np.any(mask, axis=0)
    true_energy = np.where(mask, true_power, 0.0).sum(axis=0)
    predicted_energy = np.where(mask, predicted_power, 0.0).sum(axis=0)
    sae = (predicted_energy - true_energy) / np.maximum(np.abs(true_energy), 1e-8)
    sae = np.where(available, sae, np.nan)
    f1 = [
        binary_f1(true_state[mask[:, i], i], predicted_state[mask[:, i], i])
        if np.any(mask[:, i])
        else float("nan")
        for i in range(true_state.shape[1])
    ]
    mcc = [
        binary_mcc(true_state[mask[:, i], i], predicted_state[mask[:, i], i])
        if np.any(mask[:, i])
        else float("nan")
        for i in range(true_state.shape[1])
    ]
    return {
        "mae_w": float(np.nanmean(per_appliance_mae)),
        "nmae": float(np.nanmean(nmae)),
        "sae": float(np.nanmean(np.abs(sae))),
        "f1": float(np.nanmean(f1)),
        "mcc": float(np.nanmean(mcc)),
        "per_appliance_mae_w": per_appliance_mae.tolist(),
        "per_appliance_f1": f1,
    }


def household_macro_metrics(
    true_power: np.ndarray,
    predicted_power: np.ndarray,
    true_state: np.ndarray,
    predicted_state: np.ndarray,
    target_mask: np.ndarray,
    household_lengths: list[int],
    household_ids: list[str],
) -> tuple[dict[str, float | list[float]], list[dict[str, float | list[float] | str]]]:
    if sum(household_lengths) != len(true_power) or len(household_lengths) != len(household_ids):
        raise ValueError("household boundaries do not match evaluation arrays")
    records: list[dict[str, float | list[float] | str]] = []
    offset = 0
    for household_id, length in zip(household_ids, household_lengths, strict=True):
        selection = slice(offset, offset + length)
        metrics = nilm_metrics(
            true_power[selection],
            predicted_power[selection],
            true_state[selection],
            predicted_state[selection],
            target_mask[selection],
        )
        records.append({"household_id": household_id, **metrics})
        offset += length
    scalar_keys = ["mae_w", "nmae", "sae", "f1", "mcc"]
    macro: dict[str, float | list[float]] = {
        key: float(np.nanmean([float(record[key]) for record in records])) for key in scalar_keys
    }
    for key in ("per_appliance_mae_w", "per_appliance_f1"):
        macro[key] = np.nanmean(
            np.asarray([record[key] for record in records], dtype=np.float64), axis=0
        ).tolist()
    return macro, records


def final_forgetting(response_matrix: np.ndarray) -> float:
    """Compute manuscript Eq. 8 from a square lower-triangular response matrix."""
    response_matrix = np.asarray(response_matrix, dtype=np.float64)
    tasks = response_matrix.shape[0]
    if response_matrix.shape != (tasks, tasks) or tasks < 2:
        raise ValueError("response_matrix must be square with at least two tasks")
    values = []
    for task in range(tasks - 1):
        previous_best = np.nanmax(response_matrix[task : tasks - 1, task])
        values.append(previous_best - response_matrix[tasks - 1, task])
    return float(np.mean(values))
