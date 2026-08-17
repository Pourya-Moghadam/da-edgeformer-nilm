from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    appliances: list[str] = field(
        default_factory=lambda: ["refrigerator", "washing_machine", "microwave"]
    )
    window_size: int = 256
    branch_width: int = 372
    width: int = 64
    conv_kernels: list[int] = field(default_factory=lambda: [3, 5, 7])
    conv_dilations: list[int] = field(default_factory=lambda: [1, 2, 4])
    attention_blocks: int = 2
    attention_heads: int = 4
    local_span: int = 64
    feedforward_width: int = 128
    adapter_bottleneck: int = 8
    head_hidden_width: int = 516
    detector_dim: int = 16
    detector_projection_seed: int = 1729
    dropout: float = 0.0


@dataclass
class LossConfig:
    huber_delta_w: float = 20.0
    classification_weight: float = 0.25
    replay_weight: float = 1.0
    stability_weight: float = 0.5
    retention_weight: float = 0.5


@dataclass
class DriftConfig:
    beta_short: float = 0.90
    beta_long: float = 0.995
    threshold_quantile: float = 0.995
    threshold: float | None = None
    consecutive: int = 3
    epsilon: float = 1e-6
    warmup: int = 256


@dataclass
class ControllerConfig:
    capacity: int = 4
    refill_seconds: float = 900.0
    cooldown_seconds: float = 600.0
    minimum_new_labels: int = 32
    replay_capacity: int = 256
    update_steps: int = 2


@dataclass
class TrainingConfig:
    seeds: list[int] = field(default_factory=lambda: [11, 23, 37, 53, 71])
    batch_size: int = 64
    epochs: int = 50
    inner_steps: int = 2
    inner_learning_rate: float = 1e-3
    outer_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    label_budget: float = 0.05


@dataclass
class DataConfig:
    grid_seconds: int = 10
    max_gap_native_intervals: float = 2.0
    purge_samples: int = 255
    state_thresholds_w: dict[str, float] = field(
        default_factory=lambda: {
            "refrigerator": 50.0,
            "washing_machine": 20.0,
            "microwave": 200.0,
        }
    )


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        if self.model.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.model.width % self.model.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if self.model.branch_width < self.model.width:
            raise ValueError("branch_width must be at least width")
        if len(self.model.conv_kernels) != len(self.model.conv_dilations):
            raise ValueError("conv_kernels and conv_dilations must have equal length")
        if self.data.purge_samples < self.model.window_size - 1:
            raise ValueError("purge_samples must be at least window_size - 1")
        if not 0.0 <= self.training.label_budget <= 1.0:
            raise ValueError("label_budget must be in [0, 1]")
        if not 0.0 < self.drift.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[Any], values: dict[str, Any] | None) -> Any:
    return cls(**(values or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = ExperimentConfig(
        model=_construct(ModelConfig, raw.get("model")),
        loss=_construct(LossConfig, raw.get("loss")),
        drift=_construct(DriftConfig, raw.get("drift")),
        controller=_construct(ControllerConfig, raw.get("controller")),
        training=_construct(TrainingConfig, raw.get("training")),
        data=_construct(DataConfig, raw.get("data")),
    )
    config.validate()
    return config


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
