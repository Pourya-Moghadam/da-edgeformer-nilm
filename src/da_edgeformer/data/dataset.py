from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class PreparedHousehold:
    dataset: str
    household_id: str
    timestamps: np.ndarray
    aggregate: np.ndarray
    appliance_power: np.ndarray
    appliance_state: np.ndarray
    target_mask: np.ndarray
    appliances: list[str]
    normalization_mean: float
    normalization_std: float

    @classmethod
    def load(cls, path: str | Path) -> PreparedHousehold:
        payload = np.load(path, allow_pickle=False)
        return cls(
            dataset=str(payload["dataset"]),
            household_id=str(payload["household_id"]),
            timestamps=payload["timestamps"],
            aggregate=payload["aggregate"],
            appliance_power=payload["appliance_power"],
            appliance_state=payload["appliance_state"],
            target_mask=payload["target_mask"],
            appliances=payload["appliances"].astype(str).tolist(),
            normalization_mean=float(payload["normalization_mean"]),
            normalization_std=float(payload["normalization_std"]),
        )


class CausalWindowDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, household: PreparedHousehold, window_size: int) -> None:
        self.household = household
        self.window_size = window_size
        valid_rows = np.isfinite(household.aggregate)
        convolution = np.convolve(valid_rows.astype(np.int32), np.ones(window_size), mode="valid")
        self.end_indices = np.flatnonzero(convolution == window_size) + window_size - 1

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        end = int(self.end_indices[index])
        start = end - self.window_size + 1
        aggregate = self.household.aggregate[start : end + 1]
        aggregate = (aggregate - self.household.normalization_mean) / max(
            self.household.normalization_std, 1e-6
        )
        return (
            torch.from_numpy(aggregate.astype(np.float32)),
            torch.from_numpy(self.household.appliance_power[end].astype(np.float32)),
            torch.from_numpy(self.household.appliance_state[end].astype(np.float32)),
            torch.from_numpy(self.household.target_mask[end].astype(bool)),
            torch.tensor(self.household.timestamps[end], dtype=torch.int64),
        )
