import numpy as np

from da_edgeformer.adaptation.controller import TokenBucketController
from da_edgeformer.adaptation.drift import StableFeatureDriftDetector
from da_edgeformer.config import ControllerConfig, DriftConfig


def test_controller_requires_alarm_labels_token_and_cooldown() -> None:
    controller = TokenBucketController(
        ControllerConfig(
            capacity=1,
            refill_seconds=100,
            cooldown_seconds=50,
            minimum_new_labels=2,
        )
    )
    assert controller.decide(0, False, 2).reason == "no_alarm"
    assert controller.decide(1, True, 1).reason == "insufficient_labels"
    assert controller.decide(2, False, 2).permitted
    assert controller.decide(10, True, 2).reason == "no_token"
    assert controller.decide(110, False, 2).permitted


def test_drift_calibration_and_alarm() -> None:
    config = DriftConfig(
        beta_short=0.5,
        beta_long=0.99,
        threshold_quantile=0.9,
        consecutive=2,
        warmup=5,
    )
    detector = StableFeatureDriftDetector(2, config)
    detector.calibrate(np.zeros((20, 2)))
    for _ in range(5):
        detector.observe(np.zeros(2))
    alarms = [detector.observe(np.ones(2) * 10).alarm for _ in range(10)]
    assert any(alarms)
