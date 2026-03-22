from __future__ import annotations

import argparse
import json

from pipeline.config import TrainingConfig
from pipeline.data_ingestion import load_modeling_frame
from pipeline.evaluation import evaluate_model, save_model_bundle
from pipeline.features import build_feature_frame, select_feature_columns, split_train_validation
from pipeline.registry import register_best_model
from pipeline.training import train_model


def run_pipeline(
    config: TrainingConfig | None = None,
    *,
    run_name: str | None = None,
    model_params: dict[str, float | int] | None = None,
) -> dict[str, object]:
    training_config = config or TrainingConfig()

    raw_frame = load_modeling_frame(training_config)
    feature_frame = build_feature_frame(raw_frame)
    X_train, y_train, X_valid, y_valid, validation_meta = split_train_validation(
        feature_frame,
        validation_days=training_config.validation_days,
        include_price=training_config.include_price,
    )

    train_result = train_model(
        X_train,
        y_train,
        training_config,
        run_name=run_name,
        model_params=model_params,
    )
    metrics, _ = evaluate_model(
        train_result.model,
        X_valid,
        y_valid,
        validation_meta,
        training_config,
        run_id=train_result.run_id,
    )
    artifact_path = save_model_bundle(
        train_result.model,
        select_feature_columns(include_price=training_config.include_price),
        metrics,
        training_config,
        params=train_result.params,
        run_id=train_result.run_id,
    )

    registered_version = None
    if training_config.register_model and training_config.enable_mlflow:
        registered_version = register_best_model(training_config)

    return {
        "metrics": metrics,
        "artifact_path": str(artifact_path),
        "run_id": train_result.run_id,
        "registered_version": registered_version,
        "train_rows": len(X_train),
        "validation_rows": len(X_valid),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline M5 demand forecasting pipeline.")
    parser.add_argument("--max-series", type=int, default=300)
    parser.add_argument("--recent-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=28)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--disable-price", action="store_true")
    parser.add_argument("--register-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        max_series=args.max_series,
        recent_days=args.recent_days,
        validation_days=args.validation_days,
        enable_mlflow=not args.disable_mlflow,
        include_price=not args.disable_price,
        register_model=args.register_model,
    )
    result = run_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
