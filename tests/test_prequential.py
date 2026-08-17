import numpy as np
import torch

from da_edgeformer.config import ExperimentConfig
from da_edgeformer.data.dataset import CausalWindowDataset, PreparedHousehold
from da_edgeformer.evaluation.prequential import ABLATIONS, PrequentialEvaluator, UpdatePolicy
from da_edgeformer.models.edgeformer import DAEdgeFormer


def test_synthetic_prequential_static_run() -> None:
    config = ExperimentConfig()
    config.model.window_size = 8
    config.model.branch_width = 8
    config.model.width = 8
    config.model.attention_blocks = 1
    config.model.attention_heads = 2
    config.model.local_span = 4
    config.model.feedforward_width = 16
    config.model.adapter_bottleneck = 2
    config.model.detector_dim = 4
    config.data.purge_samples = 7
    config.drift.threshold = 1.0
    config.drift.warmup = 4
    length = 32
    aggregate = np.linspace(0, 100, length, dtype=np.float32)
    power = np.stack([aggregate * 0.1, aggregate * 0.2, aggregate * 0.05], axis=1)
    household = PreparedHousehold(
        dataset="synthetic",
        household_id="1",
        timestamps=np.arange(length) * 10,
        aggregate=aggregate,
        appliance_power=power,
        appliance_state=(power > 5).astype(np.uint8),
        target_mask=np.ones_like(power, dtype=bool),
        appliances=config.model.appliances,
        normalization_mean=float(aggregate.mean()),
        normalization_std=float(aggregate.std()),
    )
    stream = CausalWindowDataset(household, config.model.window_size)
    evaluator = PrequentialEvaluator(
        DAEdgeFormer(config.model), config, ABLATIONS["B0"], torch.device("cpu"), seed=11
    )
    result = evaluator.run(stream, "synthetic", label_fraction=0.25)
    assert result.predictions.shape == (len(stream), 3)
    assert result.visible_labels.sum() == round(0.25 * len(stream))
    assert not any(event.get("decision") == "permitted" for event in result.events)


def test_synthetic_prequential_adapter_update() -> None:
    config = ExperimentConfig()
    config.model.window_size = 8
    config.model.branch_width = 8
    config.model.width = 8
    config.model.attention_blocks = 1
    config.model.attention_heads = 2
    config.model.local_span = 4
    config.model.feedforward_width = 16
    config.model.adapter_bottleneck = 2
    config.model.detector_dim = 4
    config.data.purge_samples = 7
    config.controller.minimum_new_labels = 1
    config.controller.cooldown_seconds = 0
    config.controller.refill_seconds = 1
    config.controller.update_steps = 1
    length = 20
    aggregate = np.linspace(0, 50, length, dtype=np.float32)
    power = np.stack([aggregate * 0.1, aggregate * 0.2, aggregate * 0.05], axis=1)
    household = PreparedHousehold(
        dataset="synthetic",
        household_id="2",
        timestamps=np.arange(length) * 10,
        aggregate=aggregate,
        appliance_power=power,
        appliance_state=(power > 2).astype(np.uint8),
        target_mask=np.ones_like(power, dtype=bool),
        appliances=config.model.appliances,
        normalization_mean=float(aggregate.mean()),
        normalization_std=float(aggregate.std()),
    )
    stream = CausalWindowDataset(household, config.model.window_size)
    policy = UpdatePolicy(trigger="periodic", periodic_samples=2)
    evaluator = PrequentialEvaluator(
        DAEdgeFormer(config.model), config, policy, torch.device("cpu"), seed=11
    )
    result = evaluator.run(stream, "synthetic-update", label_fraction=1.0)
    assert any(event.get("decision") == "permitted" for event in result.events)
    assert all(
        "update_loss" in event for event in result.events if event["decision"] == "permitted"
    )
