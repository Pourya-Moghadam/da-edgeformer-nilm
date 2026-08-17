import numpy as np

from da_edgeformer.config import ControllerConfig
from da_edgeformer.evaluation.schedules import matched_update_schedule, maximum_update_indices


def test_matched_schedules_have_exact_count_and_are_deterministic() -> None:
    config = ControllerConfig(
        capacity=2,
        refill_seconds=20,
        cooldown_seconds=20,
        minimum_new_labels=2,
    )
    timestamps = np.arange(100, dtype=np.float64) * 10
    labels = np.ones(100, dtype=bool)
    maximum = maximum_update_indices(timestamps, labels, config)
    assert len(maximum) > 5
    periodic = matched_update_schedule(timestamps, labels, 5, "periodic", config, 11, "s")
    random_a = matched_update_schedule(timestamps, labels, 5, "random", config, 11, "s")
    random_b = matched_update_schedule(timestamps, labels, 5, "random", config, 11, "s")
    assert len(periodic) == len(random_a) == 5
    assert random_a == random_b
