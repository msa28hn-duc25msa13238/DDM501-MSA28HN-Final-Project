from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pipeline.config import TrainingConfig


FAIRNESS_GROUP_COLUMNS = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]


def _metric_triplet(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    denominator = float(np.abs(actual).sum()) or 1.0
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "wape": float(np.abs(actual - predicted).sum() / denominator),
    }


def build_fairness_report(
    prediction_frame: pd.DataFrame,
    *,
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    groups = group_columns or FAIRNESS_GROUP_COLUMNS
    actual = prediction_frame["actual_demand"].astype(float)
    predicted = prediction_frame["predicted_demand"].astype(float)
    overall = _metric_triplet(actual, predicted)

    rows: list[dict[str, object]] = []
    for group_column in groups:
        if group_column not in prediction_frame.columns:
            continue
        grouped = prediction_frame.groupby(group_column, dropna=False, sort=True)
        for group_value, group_frame in grouped:
            metrics = _metric_triplet(
                group_frame["actual_demand"].astype(float),
                group_frame["predicted_demand"].astype(float),
            )
            rows.append(
                {
                    "group_column": group_column,
                    "group_value": str(group_value),
                    "row_count": int(len(group_frame)),
                    **metrics,
                }
            )

    fairness_frame = pd.DataFrame(rows)
    if fairness_frame.empty:
        return fairness_frame, {"overall_metrics": overall, "group_disparities": {}}

    disparity_summary: dict[str, object] = {}
    for group_column, group_frame in fairness_frame.groupby("group_column", sort=False):
        worst_mae = group_frame.sort_values("mae", ascending=False).iloc[0]
        best_mae = group_frame.sort_values("mae", ascending=True).iloc[0]
        disparity_summary[group_column] = {
            "mae_gap": float(worst_mae["mae"] - best_mae["mae"]),
            "worst_group": worst_mae["group_value"],
            "best_group": best_mae["group_value"],
            "worst_group_mae": float(worst_mae["mae"]),
            "best_group_mae": float(best_mae["mae"]),
        }

    summary = {
        "overall_metrics": overall,
        "group_disparities": disparity_summary,
    }
    return fairness_frame.sort_values(["group_column", "mae"], ascending=[True, False]), summary


def build_explainability_report(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    random_state: int,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = permutation_importance(
        model,
        X_valid,
        y_valid,
        n_repeats=5,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    importance_frame = pd.DataFrame(
        {
            "feature": X_valid.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    top_features = importance_frame.head(top_n)
    summary = {
        "method": "permutation_importance",
        "scoring": "neg_mean_absolute_error",
        "top_features": top_features.to_dict(orient="records"),
    }
    return importance_frame, summary


def _write_importance_plot(importance_frame: pd.DataFrame, output_path: Path) -> None:
    top_features = importance_frame.head(10).iloc[::-1]
    figure = plt.figure(figsize=(10, 5))
    plt.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
    )
    plt.xlabel("Permutation importance (mean decrease in score)")
    plt.ylabel("Feature")
    plt.title("Model Explainability: Top Validation Features")
    plt.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def evaluate_responsible_ai(
    model,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    validation_meta: pd.DataFrame,
    config: TrainingConfig,
    *,
    run_id: str | None = None,
) -> dict[str, object]:
    output_dir = config.model_dir / "responsible_ai"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame = validation_meta.copy()
    prediction_frame["actual_demand"] = y_valid.to_numpy()
    prediction_frame["predicted_demand"] = np.clip(model.predict(X_valid), 0.0, None)

    fairness_frame, fairness_summary = build_fairness_report(prediction_frame)
    explainability_frame, explainability_summary = build_explainability_report(
        model,
        X_valid,
        y_valid,
        random_state=config.random_state,
    )

    fairness_csv = output_dir / "fairness_group_metrics.csv"
    fairness_json = output_dir / "fairness_summary.json"
    explainability_csv = output_dir / "explainability_permutation_importance.csv"
    explainability_json = output_dir / "explainability_summary.json"
    explainability_png = output_dir / "explainability_top_features.png"

    fairness_frame.to_csv(fairness_csv, index=False)
    fairness_json.write_text(json.dumps(fairness_summary, indent=2))
    explainability_frame.to_csv(explainability_csv, index=False)
    explainability_json.write_text(json.dumps(explainability_summary, indent=2))
    _write_importance_plot(explainability_frame, explainability_png)

    if run_id and config.enable_mlflow:
        try:
            import mlflow
        except ImportError:
            pass
        else:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(fairness_csv), artifact_path="responsible_ai")
                mlflow.log_artifact(str(fairness_json), artifact_path="responsible_ai")
                mlflow.log_artifact(
                    str(explainability_csv), artifact_path="responsible_ai"
                )
                mlflow.log_artifact(
                    str(explainability_json), artifact_path="responsible_ai"
                )
                mlflow.log_artifact(
                    str(explainability_png), artifact_path="responsible_ai"
                )

    return {
        "output_dir": str(output_dir),
        "fairness_summary": fairness_summary,
        "explainability_summary": explainability_summary,
        "artifacts": {
            "fairness_group_metrics_csv": str(fairness_csv),
            "fairness_summary_json": str(fairness_json),
            "explainability_csv": str(explainability_csv),
            "explainability_summary_json": str(explainability_json),
            "explainability_plot": str(explainability_png),
        },
    }
