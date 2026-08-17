import numpy as np

from da_edgeformer.evaluation.transitions import (
    detector_event_metrics,
    post_shift_mae_increase,
    recovery_delay,
)


def test_detector_event_matching() -> None:
    timestamps = np.arange(100) * 10
    metrics = detector_event_metrics([12, 55, 90], [10, 50], 10, timestamps)
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["median_delay_samples"] == 3.5
    assert metrics["missed_transitions"] == 0


def test_shift_and_recovery_metrics() -> None:
    errors = np.r_[np.ones(20), np.ones(10) * 2, np.ones(30)]
    increase = post_shift_mae_increase(errors, [20], 10, 10)
    assert increase == 1.0
    delay = recovery_delay(errors, 20, 10, block_samples=5, consecutive_blocks=2)
    assert delay == 20
