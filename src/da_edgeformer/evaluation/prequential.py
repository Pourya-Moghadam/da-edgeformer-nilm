from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from da_edgeformer.adaptation.adwin import ADWINDetector
from da_edgeformer.adaptation.controller import TokenBucketController
from da_edgeformer.adaptation.drift import StableFeatureDriftDetector
from da_edgeformer.adaptation.replay import LabeledWindow, ReplayBuffer
from da_edgeformer.adaptation.update import UpdateState, adapt_online
from da_edgeformer.config import ExperimentConfig
from da_edgeformer.evaluation.labels import deterministic_label_mask
from da_edgeformer.evaluation.metrics import nilm_metrics
from da_edgeformer.models.edgeformer import DAEdgeFormer

TriggerMode = Literal["none", "drift", "periodic", "random", "raw", "adwin", "oracle"]


@dataclass
class UpdatePolicy:
    trigger: TriggerMode = "drift"
    adapters: bool = True
    replay: bool = True
    stability: bool = True
    periodic_samples: int = 360
    random_rate: float = 1 / 360


ABLATIONS: dict[str, UpdatePolicy] = {
    "B0": UpdatePolicy(trigger="none", replay=False, stability=False),
    "B1": UpdatePolicy(trigger="periodic", adapters=False, replay=False, stability=False),
    "B2": UpdatePolicy(trigger="drift", adapters=False, replay=False, stability=False),
    "B3": UpdatePolicy(trigger="periodic", replay=False, stability=False),
    "B4": UpdatePolicy(trigger="drift", replay=False, stability=False),
    "B5": UpdatePolicy(trigger="drift", replay=True, stability=False),
    "B6": UpdatePolicy(trigger="drift", replay=False, stability=True),
    "B7": UpdatePolicy(trigger="drift", replay=True, stability=True),
    "B8": UpdatePolicy(trigger="periodic", replay=True, stability=True),
    "B9": UpdatePolicy(trigger="drift", replay=True, stability=True),
}


@dataclass
class PrequentialResult:
    metrics: dict[str, float | list[float]]
    predictions: np.ndarray
    targets: np.ndarray
    predicted_states: np.ndarray
    target_states: np.ndarray
    target_mask: np.ndarray
    drift_scores: np.ndarray
    visible_labels: np.ndarray
    events: list[dict[str, float | int | str | bool]] = field(default_factory=list)


class PrequentialEvaluator:
    def __init__(
        self,
        model: DAEdgeFormer,
        config: ExperimentConfig,
        policy: UpdatePolicy,
        device: torch.device,
        seed: int,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.policy = policy
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.detector = StableFeatureDriftDetector(config.model.detector_dim, config.drift)
        self.raw_detector = StableFeatureDriftDetector(2, config.drift)
        self.adwin = ADWINDetector()
        self.controller = TokenBucketController(config.controller)
        self.replay = ReplayBuffer(config.controller.replay_capacity, seed)
        self.update_state: UpdateState | None = None

    def calibrate(self, streams: list[Dataset]) -> float | None:
        """Calibrate the detector using validation streams only."""
        features: list[np.ndarray] = []
        raw_features: list[np.ndarray] = []
        self.model.freeze_backbone()
        self.model.eval()
        with torch.no_grad():
            for stream in streams:
                for index in range(len(stream)):
                    aggregate = stream[index][0].unsqueeze(0).to(self.device)
                    features.append(self.model(aggregate).detector_features[0].cpu().numpy())
                    raw = aggregate[0].cpu().numpy()
                    raw_features.append(np.asarray([raw.mean(), raw.std()]))
        if self.policy.trigger == "raw":
            return self.raw_detector.calibrate(np.asarray(raw_features))
        if self.policy.trigger in {"none", "periodic", "random", "adwin", "oracle"}:
            return None
        return self.detector.calibrate(np.asarray(features))

    def _trigger(
        self,
        index: int,
        drift_alarm: bool,
        raw_alarm: bool,
        adwin_alarm: bool,
        transition_indices: set[int],
    ) -> bool:
        if self.policy.trigger == "none":
            return False
        if self.policy.trigger == "drift":
            return drift_alarm
        if self.policy.trigger == "periodic":
            return (index + 1) % self.policy.periodic_samples == 0
        if self.policy.trigger == "random":
            return bool(self.rng.random() < self.policy.random_rate)
        if self.policy.trigger == "raw":
            return raw_alarm
        if self.policy.trigger == "adwin":
            return adwin_alarm
        return index in transition_indices

    def run(
        self,
        stream: Dataset[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        stream_id: str,
        label_fraction: float | None = None,
        transition_indices: set[int] | None = None,
        scheduled_alarm_indices: set[int] | None = None,
    ) -> PrequentialResult:
        fraction = self.config.training.label_budget if label_fraction is None else label_fraction
        label_mask = deterministic_label_mask(len(stream), fraction, self.seed, stream_id)
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        predicted_states: list[np.ndarray] = []
        target_states: list[np.ndarray] = []
        target_masks: list[np.ndarray] = []
        scores: list[float] = []
        events: list[dict[str, float | int | str | bool]] = []
        newly_labeled: list[LabeledWindow] = []
        transition_indices = transition_indices or set()

        self.model.freeze_backbone()
        self.model.eval()
        self.update_state = UpdateState.initialize(self.model)
        for index in range(len(stream)):
            aggregate, power, state, target_mask, timestamp = stream[index]
            batch = aggregate.unsqueeze(0).to(self.device)

            # Prediction is recorded before any label at this index is exposed.
            with torch.no_grad():
                output = self.model(batch)
            predictions.append(output.power[0].cpu().numpy())
            targets.append(power.numpy())
            predicted_states.append((output.state_logits[0] >= 0).cpu().numpy())
            target_states.append(state.numpy())
            target_masks.append(target_mask.numpy())

            observation = self.detector.observe(output.detector_features[0].cpu().numpy())
            raw_feature = np.asarray([float(aggregate.mean()), float(aggregate.std())])
            raw_observation = self.raw_detector.observe(raw_feature)
            adwin_alarm = self.adwin.observe(float(aggregate[-1]))
            scores.append(observation.score)
            if label_mask[index]:
                item = LabeledWindow(aggregate, power, state, target_mask, key=index)
                newly_labeled.append(item)
                self.replay.add(item)

            alarm = (
                index in scheduled_alarm_indices
                if scheduled_alarm_indices is not None
                else self._trigger(
                    index,
                    observation.alarm,
                    raw_observation.alarm,
                    adwin_alarm,
                    transition_indices,
                )
            )
            decision = self.controller.decide(
                float(timestamp), alarm, newly_labeled=len(newly_labeled)
            )
            if alarm or decision.permitted:
                event: dict[str, float | int | str | bool] = {
                    "index": index,
                    "timestamp": int(timestamp),
                    "drift_score": observation.score,
                    "alarm": alarm,
                    "decision": decision.reason,
                    "tokens": decision.tokens,
                }
                if decision.permitted and fraction > 0:
                    started = time.perf_counter()
                    replay_items = (
                        self.replay.sample(
                            self.config.controller.replay_capacity,
                            exclude_keys={
                                item.key for item in newly_labeled if item.key is not None
                            },
                        )
                        if self.policy.replay
                        else []
                    )
                    if not self.policy.stability:
                        self.update_state.importance = {}
                    self.model.train()
                    self.update_state, losses = adapt_online(
                        self.model,
                        newly_labeled,
                        replay_items,
                        self.update_state,
                        self.config,
                        self.device,
                        full_finetune=not self.policy.adapters,
                    )
                    self.model.eval()
                    event["update_seconds"] = time.perf_counter() - started
                    event["update_loss"] = losses[-1]
                    event["new_labels"] = len(newly_labeled)
                    newly_labeled.clear()
                events.append(event)

        prediction_array = np.asarray(predictions)
        target_array = np.asarray(targets)
        state_array = np.asarray(predicted_states)
        target_state_array = np.asarray(target_states)
        target_mask_array = np.asarray(target_masks)
        return PrequentialResult(
            metrics=nilm_metrics(
                target_array,
                prediction_array,
                target_state_array,
                state_array,
                target_mask_array,
            ),
            predictions=prediction_array,
            targets=target_array,
            predicted_states=state_array,
            target_states=target_state_array,
            target_mask=target_mask_array,
            drift_scores=np.asarray(scores),
            visible_labels=label_mask,
            events=events,
        )
