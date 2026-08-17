from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from da_edgeformer.adaptation.ewc import (
    empirical_importance,
    merge_importance,
    online_named_parameters,
    snapshot_online_parameters,
)
from da_edgeformer.adaptation.replay import LabeledWindow, stack_windows
from da_edgeformer.config import ExperimentConfig
from da_edgeformer.losses import prediction_loss, stability_penalty
from da_edgeformer.models.edgeformer import DAEdgeFormer


@dataclass
class UpdateState:
    anchor: dict[str, Tensor]
    importance: dict[str, Tensor]

    @classmethod
    def initialize(cls, model: DAEdgeFormer) -> UpdateState:
        return cls(snapshot_online_parameters(model), {})


def adapt_online(
    model: DAEdgeFormer,
    new_items: list[LabeledWindow],
    replay_items: list[LabeledWindow],
    state: UpdateState,
    config: ExperimentConfig,
    device: torch.device,
    full_finetune: bool = False,
) -> tuple[UpdateState, list[float]]:
    if not new_items:
        return state, []
    if full_finetune:
        model.unfreeze_all()
    else:
        model.freeze_backbone()
    optimizer = torch.optim.Adam(model.online_parameters(), lr=config.training.inner_learning_rate)
    new_x, new_y, new_s, new_mask = stack_windows(new_items, device)
    replay_batch = stack_windows(replay_items, device) if replay_items else None
    losses: list[float] = []
    for _ in range(config.controller.update_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = prediction_loss(model(new_x), new_y, new_s, config.loss, new_mask)
        if replay_batch is not None:
            rep_x, rep_y, rep_s, rep_mask = replay_batch
            loss = loss + config.loss.replay_weight * prediction_loss(
                model(rep_x), rep_y, rep_s, config.loss, rep_mask
            )
        if state.importance:
            penalty = stability_penalty(
                online_named_parameters(model), state.anchor, state.importance
            )
            loss = loss + config.loss.stability_weight * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.online_parameters(), config.training.gradient_clip_norm
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    current_importance = empirical_importance(model, new_x, new_y, new_s, new_mask, config.loss)
    return UpdateState(
        anchor=snapshot_online_parameters(model),
        importance=merge_importance(state.importance, current_importance),
    ), losses
