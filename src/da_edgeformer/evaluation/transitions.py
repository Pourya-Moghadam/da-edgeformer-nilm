from __future__ import annotations

import numpy as np


def detector_event_metrics(
    alarms: list[int],
    transitions: list[int],
    tolerance_samples: int,
    timestamps: np.ndarray,
) -> dict[str, float | int]:
    """One-to-one, post-transition alarm matching within a declared tolerance."""
    unmatched = set(int(alarm) for alarm in alarms)
    delays: list[int] = []
    for transition in sorted(transitions):
        candidates = sorted(
            alarm for alarm in unmatched if transition <= alarm <= transition + tolerance_samples
        )
        if candidates:
            matched = candidates[0]
            unmatched.remove(matched)
            delays.append(matched - transition)
    true_positives = len(delays)
    false_positives = len(unmatched)
    false_negatives = len(transitions) - true_positives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    event_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    duration_days = max((float(timestamps[-1]) - float(timestamps[0])) / 86_400, 1 / 86_400)
    return {
        "precision": precision,
        "recall": recall,
        "event_f1": event_f1,
        "median_delay_samples": float(np.median(delays)) if delays else float("nan"),
        "false_alarms_per_day": false_positives / duration_days,
        "missed_transitions": false_negatives,
    }


def post_shift_mae_increase(
    absolute_errors: np.ndarray,
    transitions: list[int],
    reference_samples: int,
    post_samples: int,
) -> float:
    increases = []
    errors = np.asarray(absolute_errors, dtype=np.float64)
    if errors.ndim > 1:
        errors = np.nanmean(errors, axis=tuple(range(1, errors.ndim)))
    for transition in transitions:
        before = errors[max(0, transition - reference_samples) : transition]
        after = errors[transition : transition + post_samples]
        if len(before) and len(after) and before.mean() > 0:
            increases.append((after.mean() - before.mean()) / before.mean())
    return float(np.mean(increases)) if increases else float("nan")


def recovery_delay(
    absolute_errors: np.ndarray,
    transition: int,
    reference_samples: int,
    block_samples: int,
    consecutive_blocks: int = 5,
    tolerance: float = 0.05,
) -> int | None:
    errors = np.asarray(absolute_errors, dtype=np.float64)
    if errors.ndim > 1:
        errors = np.nanmean(errors, axis=tuple(range(1, errors.ndim)))
    reference = np.nanmean(errors[max(0, transition - reference_samples) : transition])
    qualifying = 0
    for end in range(transition + block_samples, len(errors) + 1, block_samples):
        block = errors[end - block_samples : end]
        qualifying = qualifying + 1 if np.nanmean(block) <= reference * (1 + tolerance) else 0
        if qualifying >= consecutive_blocks:
            return end - transition
    return None
