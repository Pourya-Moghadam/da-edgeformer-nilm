from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from da_edgeformer.config import ExperimentConfig
from da_edgeformer.models.edgeformer import DAEdgeFormer
from da_edgeformer.protocol import Protocol

EXPECTED_SPLITS: dict[str, dict[str, list[str]]] = {
    "REDD": {"train": ["1", "2", "3"], "validation": ["4"], "test": ["5", "6"]},
    "UK-DALE": {"train": ["1", "2", "3"], "validation": ["4"], "test": ["5"]},
    "REFIT": {
        "train": [str(value) for value in range(1, 13)],
        "validation": ["13", "15", "16", "17"],
        "test": ["18", "19", "20", "21"],
    },
    "ENERTALK": {
        "train": [f"{value:02d}" for value in range(14)],
        "validation": [f"{value:02d}" for value in range(14, 18)],
        "test": [f"{value:02d}" for value in range(18, 22)],
    },
}

EXPECTED_CONFIG = {
    "model.window_size": 256,
    "model.width": 64,
    "model.attention_blocks": 2,
    "model.attention_heads": 4,
    "model.local_span": 64,
    "model.feedforward_width": 128,
    "model.adapter_bottleneck": 8,
    "model.detector_dim": 16,
    "loss.huber_delta_w": 20.0,
    "loss.classification_weight": 0.25,
    "loss.replay_weight": 1.0,
    "loss.stability_weight": 0.5,
    "loss.retention_weight": 0.5,
    "drift.beta_short": 0.90,
    "drift.beta_long": 0.995,
    "drift.threshold_quantile": 0.995,
    "drift.consecutive": 3,
    "controller.capacity": 4,
    "controller.refill_seconds": 900,
    "controller.cooldown_seconds": 600,
    "controller.minimum_new_labels": 32,
    "controller.replay_capacity": 256,
    "controller.update_steps": 2,
    "training.inner_steps": 2,
    "training.inner_learning_rate": 0.001,
    "training.outer_learning_rate": 0.0001,
    "data.grid_seconds": 10,
    "data.purge_samples": 255,
}
EXPECTED_BASELINES = {
    "STNILM",
    "AugLPN-NILM",
    "Energformer",
    "ConvTransNILM",
    "Metric Meta-NILM",
    "RTNILM",
}


def _config_value(config: ExperimentConfig, dotted_name: str) -> Any:
    value: Any = config
    for part in dotted_name.split("."):
        value = getattr(value, part)
    return value


def audit_protocol(
    config: ExperimentConfig,
    protocol: Protocol,
    dataset_manifest_paths: list[str | Path],
    orders_path: str | Path,
) -> dict[str, Any]:
    failures: list[str] = []
    for name, expected in EXPECTED_CONFIG.items():
        actual = _config_value(config, name)
        if actual != expected:
            failures.append(f"{name}={actual}, expected {expected}")
    model = DAEdgeFormer(config.model)
    total = model.parameter_counts()["total"]
    model.freeze_backbone()
    online = model.parameter_counts()["trainable"]
    if round(total / 1_000_000, 2) != 0.64:
        failures.append(f"total parameters {total} do not round to 0.64 M")
    if round(online / 1_000_000, 3) != 0.074:
        failures.append(f"online parameters {online} do not round to 0.074 M")

    test_keys: set[str] = set()
    seen_datasets: set[str] = set()
    manifest_records = []
    for path_value in dataset_manifest_paths:
        path = Path(path_value)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset = str(raw["dataset"])
        seen_datasets.add(dataset)
        splits = {name: [str(value) for value in values] for name, values in raw["splits"].items()}
        counts = {name: len(values) for name, values in splits.items()}
        expected_splits = EXPECTED_SPLITS.get(dataset)
        if splits != expected_splits:
            failures.append(f"{dataset} split membership differs from the fixed protocol")
        household_ids = [str(entry["id"]) for entry in raw["households"]]
        flattened = [value for values in splits.values() for value in values]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(household_ids):
            failures.append(f"{dataset} split membership is not a partition of households")
        test_keys.update(f"{dataset}:{value}" for value in splits["test"])
        manifest_records.append(
            {"dataset": dataset, "counts": counts, "households": len(household_ids)}
        )
    if seen_datasets != set(EXPECTED_SPLITS):
        failures.append(
            f"dataset manifests are {sorted(seen_datasets)}, expected {sorted(EXPECTED_SPLITS)}"
        )

    orders = yaml.safe_load(Path(orders_path).read_text(encoding="utf-8"))["orders"]
    if len(orders) != 10:
        failures.append(f"found {len(orders)} stream orders, expected 10")
    for index, order in enumerate(orders, start=1):
        if len(order) != len(set(order)) or set(order) != test_keys:
            failures.append(f"stream order {index} is not a permutation of all test households")

    baseline_path = Path(orders_path).with_name("baselines.yaml")
    baselines = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))["comparators"]
    if set(baselines) != EXPECTED_BASELINES:
        failures.append("external baseline registry does not contain the six manuscript methods")
    for name, record in baselines.items():
        if not record.get("citation_doi") or not record.get("implementation"):
            failures.append(f"{name} is missing citation or implementation provenance")
        if "repository" in record and len(str(record.get("commit", ""))) != 40:
            failures.append(f"{name} public repository is not pinned to a full commit")

    return {
        "ready": not failures,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "parameters": {"total": total, "online_trainable": online},
        "dataset_manifests": manifest_records,
        "stream_orders": len(orders),
        "external_baselines": len(baselines),
        "failures": failures,
    }
