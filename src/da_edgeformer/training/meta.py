from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from da_edgeformer.config import ExperimentConfig
from da_edgeformer.losses import prediction_loss
from da_edgeformer.models.edgeformer import DAEdgeFormer


@dataclass
class Episode:
    support: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    query: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    retained: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _stack(dataset: Dataset, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    rows = [dataset[int(index)] for index in indices]
    return tuple(torch.stack([row[position] for row in rows]).to(device) for position in range(4))


class FirstOrderMetaTrainer:
    """Household-task first-order MAML with chronological support/query episodes."""

    def __init__(
        self, model: DAEdgeFormer, config: ExperimentConfig, device: torch.device, seed: int
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.model.unfreeze_all()
        self.outer_optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=config.training.outer_learning_rate,
            weight_decay=config.training.weight_decay,
        )

    def episode(
        self, dataset: Dataset, support_size: int, query_size: int, retained_size: int
    ) -> Episode:
        purge = self.config.data.purge_samples
        required = support_size + purge + query_size
        if len(dataset) < required:
            raise ValueError(f"task has {len(dataset)} windows; {required} are required")
        start = int(self.rng.integers(0, len(dataset) - required + 1))
        support_indices = np.arange(start, start + support_size)
        query_start = start + support_size + purge
        query_indices = np.arange(query_start, query_start + query_size)
        visible_count = max(1, round(self.config.training.label_budget * support_size))
        support_indices = np.sort(self.rng.choice(support_indices, visible_count, replace=False))
        retained_count = min(retained_size, len(support_indices))
        retained_indices = self.rng.choice(support_indices, retained_count, replace=False)
        return Episode(
            support=_stack(dataset, support_indices, self.device),
            query=_stack(dataset, query_indices, self.device),
            retained=_stack(dataset, retained_indices, self.device),
        )

    def step(self, episodes: list[Episode]) -> float:
        self.outer_optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, torch.Tensor] = {}
        losses: list[float] = []
        for episode in episodes:
            fast_model = copy.deepcopy(self.model)
            fast_model.freeze_backbone()
            inner_optimizer = torch.optim.SGD(
                fast_model.online_parameters(), lr=self.config.training.inner_learning_rate
            )
            support_x, support_y, support_s, support_mask = episode.support
            for _ in range(self.config.training.inner_steps):
                inner_optimizer.zero_grad(set_to_none=True)
                inner_loss = prediction_loss(
                    fast_model(support_x),
                    support_y,
                    support_s,
                    self.config.loss,
                    support_mask,
                )
                inner_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    fast_model.online_parameters(), self.config.training.gradient_clip_norm
                )
                inner_optimizer.step()

            fast_model.unfreeze_all()
            query_x, query_y, query_s, query_mask = episode.query
            retained_x, retained_y, retained_s, retained_mask = episode.retained
            outer_loss = prediction_loss(
                fast_model(query_x), query_y, query_s, self.config.loss, query_mask
            ) + self.config.loss.retention_weight * prediction_loss(
                fast_model(retained_x),
                retained_y,
                retained_s,
                self.config.loss,
                retained_mask,
            )
            outer_loss.backward()
            losses.append(float(outer_loss.detach()))
            for name, parameter in fast_model.named_parameters():
                if parameter.grad is not None:
                    accumulated[name] = (
                        accumulated.get(name, torch.zeros_like(parameter.grad)) + parameter.grad
                    )

        for name, parameter in self.model.named_parameters():
            if name in accumulated:
                parameter.grad = accumulated[name] / len(episodes)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.training.gradient_clip_norm,
        )
        self.outer_optimizer.step()
        return float(np.mean(losses))

    def validation_mae(self, episodes: list[Episode]) -> float:
        household_mae: list[float] = []
        for episode in episodes:
            fast_model = copy.deepcopy(self.model)
            fast_model.freeze_backbone()
            optimizer = torch.optim.SGD(
                fast_model.online_parameters(), lr=self.config.training.inner_learning_rate
            )
            support_x, support_y, support_s, support_mask = episode.support
            for _ in range(self.config.training.inner_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = prediction_loss(
                    fast_model(support_x),
                    support_y,
                    support_s,
                    self.config.loss,
                    support_mask,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    fast_model.online_parameters(), self.config.training.gradient_clip_norm
                )
                optimizer.step()
            query_x, query_y, _, query_mask = episode.query
            with torch.no_grad():
                prediction = fast_model(query_x).power
                errors = (prediction - query_y).abs()[query_mask]
                household_mae.append(float(errors.mean()))
        return float(np.mean(household_mae))

    def fit(
        self,
        tasks: list[Dataset],
        validation_tasks: list[Dataset] | None = None,
        support_size: int = 256,
        query_size: int = 256,
        retained_size: int = 64,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        validation_episodes = (
            [
                self.episode(task, support_size, query_size, retained_size)
                for task in validation_tasks
            ]
            if validation_tasks
            else []
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_mae = float("inf")
        for _ in range(self.config.training.epochs):
            episodes = [
                self.episode(task, support_size, query_size, retained_size) for task in tasks
            ]
            train_loss = self.step(episodes)
            validation_mae = (
                self.validation_mae(validation_episodes) if validation_episodes else train_loss
            )
            history.append({"train_loss": train_loss, "validation_mae_w": validation_mae})
            if validation_mae < best_mae:
                best_mae = validation_mae
                best_state = copy.deepcopy(self.model.state_dict())
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history
