from __future__ import annotations

from datetime import datetime, timezone
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline.config import TrainingConfig


def evaluate_model(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    validation_meta: pd.DataFrame,
    config: TrainingConfig,
    *,
    run_id: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = np.clip(model.predict(X_valid), 0.0, None)
    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))
    mae = float(mean_absolute_error(y_valid, predictions))
    denominator = float(np.abs(y_valid).sum()) or 1.0
    wape = float(np.abs(y_valid - predictions).sum() / denominator)

    metrics = {"rmse": rmse, "mae": mae, "wape": wape}
    prediction_frame = validation_meta.copy()
    prediction_frame["actual_demand"] = y_valid.to_numpy()
    prediction_frame["predicted_demand"] = predictions

    if run_id and config.enable_mlflow:
        try:
            import mlflow
        except ImportError:
            return metrics, prediction_frame

        output_dir = config.model_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / "validation_predictions.csv"
        metrics_path = output_dir / "validation_metrics.json"
        figure_path = output_dir / "validation_forecast.png"

        prediction_frame.to_csv(predictions_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2))

        figure = plt.figure(figsize=(10, 4))
        preview = prediction_frame.sort_values("date").head(100)
        plt.plot(preview["date"], preview["actual_demand"], label="actual")
        plt.plot(preview["date"], preview["predicted_demand"], label="predicted")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.legend()
        figure.savefig(figure_path)
        plt.close(figure)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
            mlflow.log_artifact(str(figure_path), artifact_path="evaluation")

    return metrics, prediction_frame


def save_model_bundle(
    model,
    feature_columns: list[str],
    metrics: dict[str, float],
    config: TrainingConfig,
    *,
    params: dict[str, float | int],
    run_id: str | None,
) -> Path:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "model_version": timestamp.strftime("%Y%m%d%H%M%S"),
        "trained_at": timestamp.isoformat(),
        "training_config": config.to_dict(),
        "training_params": params,
        "run_id": run_id,
    }
    with config.model_artifact_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return config.model_artifact_path


def log_run_artifacts(
    artifact_path: Path,
    config: TrainingConfig,
    *,
    run_id: str | None,
) -> None:
    if not run_id or not config.enable_mlflow:
        return

    try:
        import mlflow
    except ImportError:
        return

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(artifact_path), artifact_path="bundle")
