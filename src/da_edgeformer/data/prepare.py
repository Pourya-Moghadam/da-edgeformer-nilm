from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from da_edgeformer.config import ExperimentConfig
from da_edgeformer.data.resample import time_weighted_mean
from da_edgeformer.data.sources import load_household, load_source_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    source = root / entry["path"]
    files = (
        [source]
        if source.is_file()
        else sorted(path for path in source.rglob("*") if path.is_file())
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return {"path": str(entry["path"]), "files": len(files), "sha256": digest.hexdigest()}


def _resample_channel(
    timestamps: np.ndarray,
    values: np.ndarray,
    config: ExperimentConfig,
    native_interval: float,
) -> tuple[np.ndarray, np.ndarray]:
    return time_weighted_mean(
        timestamps,
        values,
        config.data.grid_seconds,
        native_interval,
        config.data.max_gap_native_intervals,
    )


def _align(raw: Any, config: ExperimentConfig, native_interval: float) -> tuple[Any, ...]:
    resampled = {
        name: _resample_channel(raw.timestamps[name], values, config, native_interval)
        for name, values in raw.channels.items()
    }
    common = resampled["aggregate"][0]
    for name, (timestamps, _) in resampled.items():
        if name != "aggregate":
            common = np.intersect1d(common, timestamps, assume_unique=True)
    aligned: dict[str, np.ndarray] = {}
    for name, (timestamps, values) in resampled.items():
        indices = np.searchsorted(timestamps, common)
        aligned[name] = values[indices]
    return common, aligned


def _split_ids(manifest: dict[str, Any]) -> dict[str, list[str]]:
    explicit = manifest.get("splits")
    if explicit:
        return {key: [str(value) for value in values] for key, values in explicit.items()}
    households = [str(entry["id"]) for entry in manifest["households"]]
    counts = manifest.get("split_counts")
    if not counts or sum(counts.values()) != len(households):
        raise ValueError("provide explicit splits or split_counts totaling all households")
    result: dict[str, list[str]] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        count = int(counts[split])
        result[split] = households[offset : offset + count]
        offset += count
    return result


def prepare_dataset(
    source_manifest: str | Path,
    raw_root: str | Path,
    output_root: str | Path,
    config: ExperimentConfig,
    protocol: dict[str, str] | None = None,
) -> Path:
    """Prepare one dataset and write arrays plus a provenance manifest."""
    source_path = Path(source_manifest).resolve()
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    manifest = load_source_manifest(source_path)
    dataset_name = str(manifest["dataset"])
    destination = output_root / dataset_name.lower()
    destination.mkdir(parents=True, exist_ok=True)
    splits = _split_ids(manifest)
    split_by_id = {household: split for split, ids in splits.items() for household in ids}
    native_interval = float(manifest["native_interval_seconds"])
    appliances = config.model.appliances
    prepared: list[dict[str, Any]] = []
    train_aggregate: list[np.ndarray] = []

    staged: list[tuple[Any, np.ndarray, dict[str, np.ndarray], dict[str, Any]]] = []
    for entry in manifest["households"]:
        raw = load_household(raw_root, manifest["format"], entry)
        timestamps, channels = _align(raw, config, native_interval)
        staged.append((raw, timestamps, channels, _source_digest(raw_root, entry)))
        if split_by_id[raw.household_id] == "train":
            train_aggregate.append(channels["aggregate"])

    finite_train = np.concatenate([values[np.isfinite(values)] for values in train_aggregate])
    normalization_mean = float(finite_train.mean())
    normalization_std = float(finite_train.std())

    for raw, timestamps, channels, source_record in staged:
        power = np.stack(
            [
                channels.get(name, np.full(len(timestamps), np.nan, dtype=np.float32))
                for name in appliances
            ],
            axis=1,
        ).astype(np.float32)
        target_mask = np.isfinite(power)
        thresholds = np.asarray(
            [config.data.state_thresholds_w[name] for name in appliances], dtype=np.float32
        )
        states = (np.nan_to_num(power) >= thresholds).astype(np.uint8)
        power = np.nan_to_num(power).astype(np.float32)
        output_path = destination / f"household_{raw.household_id}.npz"
        np.savez_compressed(
            output_path,
            dataset=np.asarray(dataset_name),
            household_id=np.asarray(raw.household_id),
            timestamps=timestamps.astype(np.int64),
            aggregate=channels["aggregate"].astype(np.float32),
            appliance_power=power,
            appliance_state=states,
            target_mask=target_mask,
            appliances=np.asarray(appliances),
            normalization_mean=np.asarray(normalization_mean),
            normalization_std=np.asarray(normalization_std),
        )
        prepared.append(
            {
                "household_id": raw.household_id,
                "split": split_by_id[raw.household_id],
                "path": output_path.name,
                "samples": len(timestamps),
                "available_appliances": [
                    name
                    for name, available in zip(appliances, target_mask.any(axis=0), strict=True)
                    if available
                ],
                "source": source_record,
                "sha256": _sha256(output_path),
            }
        )

    output_manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "source_manifest": source_path.name,
        "source_manifest_sha256": _sha256(source_path),
        "grid_seconds": config.data.grid_seconds,
        "max_gap_native_intervals": config.data.max_gap_native_intervals,
        "window_size": config.model.window_size,
        "purge_samples": config.data.purge_samples,
        "appliances": appliances,
        "state_thresholds_w": config.data.state_thresholds_w,
        "normalization": {"mean": normalization_mean, "std": normalization_std},
        "splits": splits,
        "households": prepared,
        **(protocol or {}),
    }
    output_manifest_path = destination / "manifest.json"
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    return output_manifest_path
