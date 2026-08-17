from __future__ import annotations

from dataclasses import dataclass

from da_edgeformer.config import ControllerConfig


@dataclass
class ControllerDecision:
    permitted: bool
    reason: str
    tokens: float


class TokenBucketController:
    """Stream-time token bucket; decisions do not depend on wall-clock execution."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.tokens = float(config.capacity)
        self.last_refill_time: float | None = None
        self.last_update_time: float | None = None
        self.pending_alarm = False

    def _refill(self, timestamp: float) -> None:
        if self.last_refill_time is None:
            self.last_refill_time = timestamp
            return
        elapsed = max(0.0, timestamp - self.last_refill_time)
        self.tokens = min(
            float(self.config.capacity), self.tokens + elapsed / self.config.refill_seconds
        )
        self.last_refill_time = timestamp

    def decide(self, timestamp: float, alarm: bool, newly_labeled: int) -> ControllerDecision:
        self._refill(timestamp)
        self.pending_alarm = self.pending_alarm or alarm
        if not self.pending_alarm:
            return ControllerDecision(False, "no_alarm", self.tokens)
        if newly_labeled < self.config.minimum_new_labels:
            return ControllerDecision(False, "insufficient_labels", self.tokens)
        if self.tokens < 1.0:
            return ControllerDecision(False, "no_token", self.tokens)
        if (
            self.last_update_time is not None
            and timestamp - self.last_update_time < self.config.cooldown_seconds
        ):
            return ControllerDecision(False, "cooldown", self.tokens)
        self.tokens -= 1.0
        self.last_update_time = timestamp
        self.pending_alarm = False
        return ControllerDecision(True, "permitted", self.tokens)
