from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class LabeledWindow:
    aggregate: Tensor
    power: Tensor
    state: Tensor
    target_mask: Tensor
    key: int | None = None


class ReplayBuffer:
    """Bounded reservoir sample of post-prediction labeled windows."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.items: list[LabeledWindow] = []
        self.seen = 0

    def add(self, item: LabeledWindow) -> None:
        detached = LabeledWindow(
            item.aggregate.detach().cpu().clone(),
            item.power.detach().cpu().clone(),
            item.state.detach().cpu().clone(),
            item.target_mask.detach().cpu().clone(),
            item.key,
        )
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(detached)
            return
        replacement = int(self.rng.integers(0, self.seen))
        if replacement < self.capacity:
            self.items[replacement] = detached

    def sample(self, count: int, exclude_keys: set[int] | None = None) -> list[LabeledWindow]:
        candidates = [
            item
            for item in self.items
            if item.key is None or item.key not in (exclude_keys or set())
        ]
        if not candidates or count <= 0:
            return []
        indices = self.rng.choice(len(candidates), min(count, len(candidates)), replace=False)
        return [candidates[int(index)] for index in indices]


def stack_windows(
    items: list[LabeledWindow], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not items:
        raise ValueError("cannot stack an empty window list")
    return (
        torch.stack([item.aggregate for item in items]).to(device),
        torch.stack([item.power for item in items]).to(device),
        torch.stack([item.state for item in items]).to(device),
        torch.stack([item.target_mask for item in items]).to(device),
    )
