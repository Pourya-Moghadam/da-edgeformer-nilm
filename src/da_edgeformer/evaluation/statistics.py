from __future__ import annotations

import numpy as np


def hierarchical_bootstrap_difference(
    differences_by_household: list[np.ndarray],
    iterations: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap households first and seed/order replicates second."""
    if not differences_by_household:
        raise ValueError("at least one household is required")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    household_count = len(differences_by_household)
    for iteration in range(iterations):
        selected = rng.integers(0, household_count, household_count)
        household_means = []
        for index in selected:
            values = np.asarray(differences_by_household[int(index)], dtype=np.float64)
            replicate = rng.choice(values, len(values), replace=True)
            household_means.append(replicate.mean())
        estimates[iteration] = np.mean(household_means)
    point = float(np.mean([np.mean(values) for values in differences_by_household]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def paired_permutation_test(
    differences: np.ndarray, iterations: int = 100_000, seed: int = 0
) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    if not differences.size:
        raise ValueError("differences cannot be empty")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        exceed += abs(float(np.mean(signs * differences))) >= observed
    return (exceed + 1) / (iterations + 1)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def paired_effect_size(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    standard_deviation = differences.std(ddof=1)
    return 0.0 if standard_deviation == 0 else float(differences.mean() / standard_deviation)
