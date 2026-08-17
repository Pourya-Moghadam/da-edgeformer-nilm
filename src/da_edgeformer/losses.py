from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from da_edgeformer.config import LossConfig
from da_edgeformer.models.edgeformer import ModelOutput


def prediction_loss(
    output: ModelOutput,
    target_power: Tensor,
    target_state: Tensor,
    config: LossConfig,
    target_mask: Tensor | None = None,
) -> Tensor:
    mask = (
        torch.ones_like(target_power, dtype=torch.bool)
        if target_mask is None
        else target_mask.bool()
    )
    if not torch.any(mask):
        raise ValueError("prediction loss requires at least one visible appliance target")
    regression_values = F.huber_loss(
        output.power, target_power, delta=config.huber_delta_w, reduction="none"
    )
    classification_values = F.binary_cross_entropy_with_logits(
        output.state_logits, target_state.float(), reduction="none"
    )
    regression = regression_values[mask].mean()
    classification = classification_values[mask].mean()
    return regression + config.classification_weight * classification


def stability_penalty(
    named_parameters: dict[str, Tensor], anchor: dict[str, Tensor], importance: dict[str, Tensor]
) -> Tensor:
    terms = [
        (importance[name] * (parameter - anchor[name]).square()).sum()
        for name, parameter in named_parameters.items()
        if name in anchor and name in importance
    ]
    if terms:
        return torch.stack(terms).sum()
    parameter = next(iter(named_parameters.values()))
    return parameter.new_zeros(())
