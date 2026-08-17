from __future__ import annotations

import math
from collections import deque

import numpy as np


class ADWINDetector:
    """Compact adaptive-window mean-change detector for trigger comparisons.

    This implementation checks regularly spaced window cuts with a Hoeffding-style
    bound. Its exact version and parameters are recorded so it is not confused with
    another library's ADWIN implementation.
    """

    def __init__(
        self, delta: float = 0.002, min_subwindow: int = 32, max_window: int = 1024
    ) -> None:
        self.delta = delta
        self.min_subwindow = min_subwindow
        self.values: deque[float] = deque(maxlen=max_window)

    def observe(self, value: float) -> bool:
        self.values.append(float(value))
        if len(self.values) < 2 * self.min_subwindow:
            return False
        array = np.asarray(self.values, dtype=np.float64)
        candidates = range(self.min_subwindow, len(array) - self.min_subwindow + 1, 16)
        log_term = math.log(4.0 * len(array) / self.delta)
        for cut in candidates:
            left, right = array[:cut], array[cut:]
            epsilon = math.sqrt(0.5 * log_term * (1 / len(left) + 1 / len(right)))
            if abs(float(left.mean() - right.mean())) > epsilon:
                for _ in range(cut):
                    self.values.popleft()
                return True
        return False
