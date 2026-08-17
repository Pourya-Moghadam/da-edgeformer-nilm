import json

from da_edgeformer.reporting import aggregate_summaries, discover_summaries, write_report


def test_reporting_discovers_and_aggregates_households(tmp_path) -> None:
    summary = {
        "protocol_id": "p",
        "protocol_sha256": "h",
        "experiment": "natural-stream",
        "ablation": "B9",
        "trigger": "drift",
        "label_budget": 0.05,
        "household_metrics": [
            {
                "household_id": "d:1",
                "mae_w": 1,
                "nmae": 2,
                "sae": 3,
                "f1": 0.8,
                "mcc": 0.7,
            }
        ],
    }
    path = tmp_path / "run" / "summary.json"
    path.parent.mkdir()
    path.write_text(json.dumps(summary), encoding="utf-8")
    records = aggregate_summaries(discover_summaries(tmp_path), iterations=20, seed=1)
    assert records[0]["mae_w"] == 1
    json_path, csv_path = write_report(records, tmp_path / "report")
    assert json_path.exists() and csv_path.exists()
