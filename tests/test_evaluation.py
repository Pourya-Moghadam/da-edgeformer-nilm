import numpy as np

from da_edgeformer.evaluation.labels import deterministic_label_mask
from da_edgeformer.evaluation.metrics import final_forgetting, nilm_metrics
from da_edgeformer.evaluation.statistics import holm_adjust


def test_label_mask_is_exact_and_reproducible() -> None:
    first = deterministic_label_mask(101, 0.05, 11, "stream-a")
    second = deterministic_label_mask(101, 0.05, 11, "stream-a")
    assert first.sum() == 5
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, deterministic_label_mask(101, 0.05, 12, "stream-a"))


def test_metrics_perfect_prediction() -> None:
    power = np.array([[0.0, 10.0], [5.0, 0.0]])
    state = power > 0
    metrics = nilm_metrics(power, power, state, state)
    assert metrics["mae_w"] == 0.0
    assert metrics["f1"] == 1.0
    assert metrics["sae"] == 0.0


def test_metrics_ignore_unavailable_appliance() -> None:
    true = np.array([[10.0, 0.0], [20.0, 0.0]])
    predicted = np.array([[12.0, 1000.0], [18.0, 1000.0]])
    state = true > 0
    mask = np.array([[True, False], [True, False]])
    metrics = nilm_metrics(true, predicted, state, predicted > 0, mask)
    assert metrics["mae_w"] == 2.0
    assert metrics["per_appliance_mae_w"][0] == 2.0
    assert np.isnan(metrics["per_appliance_mae_w"][1])


def test_forgetting_and_holm() -> None:
    response = np.array([[0.8, np.nan, np.nan], [0.9, 0.7, np.nan], [0.7, 0.6, 0.8]])
    assert np.isclose(final_forgetting(response), 0.15)
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
