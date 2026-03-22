from __future__ import annotations

from pathlib import Path

from pipeline.config import TrainingConfig
from pipeline.run_pipeline import run_pipeline
from tests.helpers import write_synthetic_dataset


def test_run_pipeline_creates_model_artifact(tmp_path: Path) -> None:
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

    assert Path(result["artifact_path"]).exists()
    assert result["train_rows"] > 0
    assert result["validation_rows"] > 0
    assert "rmse" in result["metrics"]
