from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import TrainingConfig
from pipeline.responsible_ai import build_fairness_report
from pipeline.run_pipeline import run_pipeline
from tests.helpers import write_synthetic_dataset


def test_build_fairness_report_computes_group_disparities() -> None:
    prediction_frame = pd.DataFrame(
        [
            {
                "state_id": "CA",
                "store_id": "CA_1",
                "cat_id": "FOODS",
                "dept_id": "FOODS_1",
                "item_id": "FOODS_1_001",
                "actual_demand": 10.0,
                "predicted_demand": 9.0,
            },
            {
                "state_id": "CA",
                "store_id": "CA_1",
                "cat_id": "FOODS",
                "dept_id": "FOODS_1",
                "item_id": "FOODS_1_002",
                "actual_demand": 11.0,
                "predicted_demand": 10.5,
            },
            {
                "state_id": "TX",
                "store_id": "TX_1",
                "cat_id": "HOUSEHOLD",
                "dept_id": "HOUSEHOLD_1",
                "item_id": "HOUSEHOLD_1_001",
                "actual_demand": 10.0,
                "predicted_demand": 6.0,
            },
            {
                "state_id": "TX",
                "store_id": "TX_1",
                "cat_id": "HOUSEHOLD",
                "dept_id": "HOUSEHOLD_1",
                "item_id": "HOUSEHOLD_1_002",
                "actual_demand": 12.0,
                "predicted_demand": 7.0,
            },
        ]
    )

    fairness_frame, summary = build_fairness_report(prediction_frame)

    assert not fairness_frame.empty
    assert "state_id" in summary["group_disparities"]
    assert summary["group_disparities"]["state_id"]["worst_group"] == "TX"
    assert summary["group_disparities"]["state_id"]["best_group"] == "CA"
    assert summary["group_disparities"]["state_id"]["mae_gap"] > 0


def test_run_pipeline_writes_responsible_ai_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "m5_data"
    model_dir = tmp_path / "models"
    write_synthetic_dataset(data_dir)

    config = TrainingConfig(
        data_dir=data_dir,
        model_dir=model_dir,
        max_series=2,
        recent_days=30,
        validation_days=7,
        enable_mlflow=False,
    )
    result = run_pipeline(config)

    responsible_ai = result["responsible_ai"]
    output_dir = Path(responsible_ai["output_dir"])

    assert output_dir.exists()
    assert Path(responsible_ai["artifacts"]["fairness_group_metrics_csv"]).exists()
    assert Path(responsible_ai["artifacts"]["fairness_summary_json"]).exists()
    assert Path(responsible_ai["artifacts"]["explainability_csv"]).exists()
    assert Path(responsible_ai["artifacts"]["explainability_summary_json"]).exists()
    assert Path(responsible_ai["artifacts"]["explainability_plot"]).exists()
