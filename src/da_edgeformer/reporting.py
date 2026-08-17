from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPORT_METRICS = ("mae_w", "nmae", "sae", "f1", "mcc")


def discover_summaries(root: str | Path) -> list[dict[str, Any]]:
    paths = sorted(Path(root).rglob("summary.json"))
    if not paths:
        raise ValueError(f"no summary.json files found below {root}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    identities = {(item.get("protocol_id"), item.get("protocol_sha256")) for item in summaries}
    if len(identities) != 1:
        raise ValueError("result tree mixes protocol identities")
    return summaries


def aggregate_summaries(
    summaries: list[dict[str, Any]], iterations: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        key = (
            summary.get("experiment", "evaluation"),
            summary.get("ablation"),
            summary.get("trigger"),
            summary.get("label_budget"),
        )
        grouped[key].append(summary)

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for key, runs in sorted(grouped.items(), key=lambda item: str(item[0])):
        record: dict[str, Any] = {
            "experiment": key[0],
            "ablation": key[1],
            "trigger": key[2],
            "label_budget": key[3],
            "runs": len(runs),
        }
        household_values: dict[str, dict[str, list[float]]] = {
            metric: defaultdict(list) for metric in REPORT_METRICS
        }
        for run in runs:
            for household in run["household_metrics"]:
                for metric in REPORT_METRICS:
                    value = household[metric]
                    household_values[metric][household["household_id"]].append(float(value))
        for metric, by_household in household_values.items():
            values = [np.asarray(item, dtype=np.float64) for item in by_household.values()]
            point = float(np.mean([item.mean() for item in values]))
            bootstrap = np.empty(iterations, dtype=np.float64)
            for iteration in range(iterations):
                selected = rng.integers(0, len(values), len(values))
                bootstrap[iteration] = np.mean(
                    [
                        rng.choice(values[index], len(values[index]), replace=True).mean()
                        for index in selected
                    ]
                )
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            record[metric] = point
            record[f"{metric}_ci_low"] = float(low)
            record[f"{metric}_ci_high"] = float(high)
        records.append(record)
    return records


def write_report(records: list[dict[str, Any]], output: str | Path) -> tuple[Path, Path]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "aggregate.json"
    csv_path = output_path / "aggregate.csv"
    json_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path
