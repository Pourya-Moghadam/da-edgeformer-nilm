from __future__ import annotations

import torch
from torch import Tensor, nn

from da_edgeformer.config import LossConfig
from da_edgeformer.losses import prediction_loss
from da_edgeformer.models.edgeformer import DAEdgeFormer


def online_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def snapshot_online_parameters(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def empirical_importance(
    model: DAEdgeFormer,
    aggregate: Tensor,
    power: Tensor,
    state: Tensor,
    target_mask: Tensor,
    loss_config: LossConfig,
) -> dict[str, Tensor]:
    model.zero_grad(set_to_none=True)
    loss = prediction_loss(model(aggregate), power, state, loss_config, target_mask)
    parameters = online_named_parameters(model)
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
    return {
        name: torch.zeros_like(parameter) if gradient is None else gradient.detach().square()
        for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True)
    }


def merge_importance(
    previous: dict[str, Tensor], current: dict[str, Tensor], decay: float = 0.9
) -> dict[str, Tensor]:
    if not previous:
        return current
    return {
        name: decay * previous.get(name, torch.zeros_like(value)) + (1.0 - decay) * value
        for name, value in current.items()
    }
