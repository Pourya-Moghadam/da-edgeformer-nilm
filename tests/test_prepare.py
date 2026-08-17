import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from da_edgeformer.config import ExperimentConfig
from da_edgeformer.data.dataset import CausalWindowDataset, PreparedHousehold
from da_edgeformer.data.prepare import prepare_dataset


def test_csv_prepare_pipeline(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    timestamps = np.arange(0, 200, 5)
    for household_id in (1, 2, 3):
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "aggregate": 100 + timestamps,
                "fridge": np.where(timestamps % 20 < 10, 80, 0),
                "washer": np.where(timestamps > 100, 200, 0),
                "microwave": np.where((timestamps >= 50) & (timestamps < 70), 500, 0),
            }
        )
        frame.to_csv(raw / f"house_{household_id}.csv", index=False)
    manifest = {
        "dataset": "synthetic",
        "format": "csv",
        "native_interval_seconds": 5,
        "split_counts": {"train": 1, "validation": 1, "test": 1},
        "households": [
            {
                "id": str(household_id),
                "path": f"house_{household_id}.csv",
                "columns": {
                    "aggregate": "aggregate",
                    "refrigerator": "fridge",
                    "washing_machine": "washer",
                    "microwave": "microwave",
                },
            }
            for household_id in (1, 2, 3)
        ],
    }
    source_manifest = tmp_path / "source.yaml"
    source_manifest.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    config = ExperimentConfig()
    config.model.window_size = 4
    config.data.purge_samples = 3
    output_manifest = prepare_dataset(source_manifest, raw, tmp_path / "prepared", config)
    output = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert output["splits"]["train"] == ["1"]
    household = PreparedHousehold.load(output_manifest.parent / "household_3.npz")
    assert len(CausalWindowDataset(household, 4)) > 0
