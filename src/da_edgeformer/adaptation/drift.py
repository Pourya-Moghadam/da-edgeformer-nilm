from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da_edgeformer.config import DriftConfig


@dataclass
class DriftObservation:
    score: float
    alarm: bool


class StableFeatureDriftDetector:
    """Short/long EWMA mean-shift monitor in a fixed feature space."""

    def __init__(self, dimension: int, config: DriftConfig) -> None:
        self.dimension = dimension
        self.config = config
        self.threshold = config.threshold
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.exceedances = 0
        self.short_mean = np.zeros(self.dimension, dtype=np.float64)
        self.long_mean = np.zeros(self.dimension, dtype=np.float64)
        self.short_var = np.ones(self.dimension, dtype=np.float64)
        self.long_var = np.ones(self.dimension, dtype=np.float64)

    @staticmethod
    def _ewma_update(
        value: np.ndarray, mean: np.ndarray, variance: np.ndarray, beta: float
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = value - mean
        updated_mean = beta * mean + (1.0 - beta) * value
        # Stable exponentially weighted central second moment.
        updated_variance = beta * variance + (1.0 - beta) * delta * (value - updated_mean)
        return updated_mean, np.maximum(updated_variance, 0.0)

    def observe(self, feature: np.ndarray) -> DriftObservation:
        feature = np.asarray(feature, dtype=np.float64).reshape(-1)
        if feature.shape != (self.dimension,):
            raise ValueError(f"expected feature dimension {self.dimension}")
        if self.count == 0:
            self.short_mean[:] = feature
            self.long_mean[:] = feature
        else:
            self.short_mean, self.short_var = self._ewma_update(
                feature, self.short_mean, self.short_var, self.config.beta_short
            )
            self.long_mean, self.long_var = self._ewma_update(
                feature, self.long_mean, self.long_var, self.config.beta_long
            )
        self.count += 1
        score = float(
            np.mean(
                np.square(self.short_mean - self.long_mean) / (self.long_var + self.config.epsilon)
            )
        )
        above = (
            self.threshold is not None
            and self.count >= self.config.warmup
            and score > self.threshold
        )
        self.exceedances = self.exceedances + 1 if above else 0
        alarm = self.exceedances >= self.config.consecutive
        if alarm:
            self.exceedances = 0
        return DriftObservation(score=score, alarm=alarm)

    def calibrate(self, features: np.ndarray) -> float:
        """Set threshold from a validation stream and reset online state."""
        original = self.threshold
        self.threshold = np.inf
        scores = np.asarray([self.observe(feature).score for feature in features])
        usable = scores[self.config.warmup :]
        if not len(usable):
            self.threshold = original
            self.reset()
            raise ValueError("validation feature stream is shorter than drift warmup")
        self.threshold = float(np.quantile(usable, self.config.threshold_quantile))
        self.reset()
        return self.threshold
