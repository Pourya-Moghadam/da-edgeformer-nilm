from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ALIASES = {
    "refrigerator": {"refrigerator", "fridge", "fridge freezer", "freezer"},
    "washing_machine": {
        "washing machine",
        "washer",
        "clothes washer",
        "washer dryer",
    },
    "microwave": {"microwave", "microwave oven"},
}
ALIAS_PRIORITY = {
    "refrigerator": ["refrigerator", "fridge", "fridge freezer", "freezer"],
    "washing_machine": ["washing machine", "washer", "clothes washer", "washer dryer"],
    "microwave": ["microwave", "microwave oven"],
}


@dataclass
class RawHousehold:
    household_id: str
    timestamps: dict[str, np.ndarray]
    channels: dict[str, np.ndarray]


def _canonical(label: str) -> str | None:
    normalized = _normalize_label(label)
    for canonical, aliases in ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def _normalize_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").replace("-", " ").split())


def _read_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["timestamp", "power"])
    return frame["timestamp"].to_numpy(), frame["power"].to_numpy(dtype=np.float32)


def _read_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            channel, label = line.strip().split(maxsplit=1)
            labels[int(channel)] = label
    return labels


def _sum_dat_channels(directory: Path, channel_numbers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    parts = [_read_dat(directory / f"channel_{number}.dat") for number in channel_numbers]
    union = np.unique(np.concatenate([timestamps for timestamps, _ in parts]))
    aligned = [pd.Series(values, index=timestamps).reindex(union) for timestamps, values in parts]
    frame = pd.concat(aligned, axis=1)
    return union, frame.sum(axis=1, min_count=len(parts)).to_numpy(dtype=np.float32)


def load_channel_dat(root: Path, household: dict[str, Any]) -> RawHousehold:
    """Read REDD/UK-DALE-style ``channel_N.dat`` source files."""
    directory = root / household["path"]
    labels = _read_labels(directory / household.get("labels_file", "labels.dat"))
    mains_channels = household.get("aggregate_channels", [1])
    timestamps: dict[str, np.ndarray] = {}
    channels: dict[str, np.ndarray] = {}

    aggregate_times, aggregate = _sum_dat_channels(directory, list(mains_channels))
    timestamps["aggregate"] = aggregate_times
    channels["aggregate"] = aggregate

    requested = household.get("appliance_channels", {})
    if requested:
        appliance_channels = {
            name: [int(value) for value in number] if isinstance(number, list) else [int(number)]
            for name, number in requested.items()
        }
    else:
        candidates: dict[str, dict[str, list[int]]] = {}
        for number, label in labels.items():
            canonical = _canonical(label)
            if canonical:
                normalized = _normalize_label(label)
                candidates.setdefault(canonical, {}).setdefault(normalized, []).append(number)
        appliance_channels = {}
        for canonical, label_groups in candidates.items():
            selected = next(
                (label for label in ALIAS_PRIORITY[canonical] if label in label_groups), None
            )
            if selected is not None:
                appliance_channels[canonical] = label_groups[selected]
    for name, numbers in appliance_channels.items():
        times, values = _sum_dat_channels(directory, numbers)
        timestamps[name] = times
        channels[name] = values
    return RawHousehold(str(household["id"]), timestamps, channels)


def load_csv(root: Path, household: dict[str, Any]) -> RawHousehold:
    path = root / household["path"]
    frame = pd.read_csv(path)
    timestamp_column = household.get("timestamp_column", "timestamp")
    unit = household.get("timestamp_unit", "s")
    raw_time = frame[timestamp_column]
    if np.issubdtype(raw_time.dtype, np.number):
        timestamps = raw_time.to_numpy(dtype=np.float64)
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}[unit]
        timestamps *= scale
    else:
        timestamps = pd.to_datetime(raw_time, utc=True).astype("int64").to_numpy() / 1e9
    column_map = household["columns"]
    channel_times = {name: timestamps.copy() for name in column_map}
    channels = {}
    for name, columns in column_map.items():
        column_list = columns if isinstance(columns, list) else [columns]
        channels[name] = (
            frame[column_list].sum(axis=1, min_count=len(column_list)).to_numpy(dtype=np.float32)
        )
    if "aggregate" not in channels:
        raise ValueError(f"{path}: columns mapping must include aggregate")
    return RawHousehold(str(household["id"]), channel_times, channels)


def load_parquet_directory(root: Path, household: dict[str, Any]) -> RawHousehold:
    """Read the native ENERTALK house/date/device Parquet hierarchy."""
    directory = root / household["path"]
    files = sorted(directory.glob("*/*.parquet.gzip"))
    if not files:
        files = sorted(directory.glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no ENERTALK Parquet files found under {directory}")
    grouped: dict[str, list[Path]] = {}
    for path in files:
        parts = path.name.split("_", maxsplit=1)
        if len(parts) != 2:
            continue
        device, label = parts
        label = label.split(".parquet", maxsplit=1)[0].replace("-", " ")
        canonical = "aggregate" if int(device) == 0 else _canonical(label)
        if canonical and canonical not in grouped:
            # Use the lowest-numbered instance for a canonical appliance.
            grouped[canonical] = []
        if canonical and (
            canonical == "aggregate"
            or not grouped[canonical]
            or device == grouped[canonical][0].name.split("_")[0]
        ):
            grouped[canonical].append(path)
    timestamps: dict[str, np.ndarray] = {}
    channels: dict[str, np.ndarray] = {}
    for name, paths in grouped.items():
        frames = [pd.read_parquet(path, columns=["timestamp", "active_power"]) for path in paths]
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        timestamps[name] = frame["timestamp"].to_numpy(dtype=np.float64) / 1000.0
        channels[name] = frame["active_power"].abs().to_numpy(dtype=np.float32)
    if "aggregate" not in channels:
        raise ValueError(f"{directory}: ENERTALK device 00 aggregate was not found")
    return RawHousehold(str(household["id"]), timestamps, channels)


def load_source_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    required = {"dataset", "format", "native_interval_seconds", "households"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"source manifest missing: {', '.join(sorted(missing))}")
    return manifest


def load_household(root: Path, format_name: str, household: dict[str, Any]) -> RawHousehold:
    if format_name == "channel_dat":
        return load_channel_dat(root, household)
    if format_name == "csv":
        return load_csv(root, household)
    if format_name == "parquet_directory":
        return load_parquet_directory(root, household)
    raise ValueError(f"unsupported source format: {format_name}")
