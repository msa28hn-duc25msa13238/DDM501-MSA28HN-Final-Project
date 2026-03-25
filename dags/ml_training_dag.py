from __future__ import annotations

from datetime import datetime, timedelta
import pickle
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline.config import TrainingConfig
from pipeline.data_ingestion import load_modeling_frame
from pipeline.evaluation import evaluate_model, save_model_bundle
from pipeline.features import (
    build_feature_frame,
    select_feature_columns,
    split_train_validation,
)
from pipeline.registry import register_best_model
from pipeline.training import train_model


TEMP_DIR = Path("/tmp/ddm501_airflow")
FEATURES_PATH = TEMP_DIR / "feature_frame.pkl"
TRAINING_PATH = TEMP_DIR / "training_output.pkl"
CONFIG = TrainingConfig()


def prepare_data_fn() -> str:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    raw_frame = load_modeling_frame(CONFIG)
    feature_frame = build_feature_frame(raw_frame)
    with FEATURES_PATH.open("wb") as handle:
        pickle.dump(feature_frame, handle)
    return str(FEATURES_PATH)


def train_model_fn() -> str:
    with FEATURES_PATH.open("rb") as handle:
        feature_frame = pickle.load(handle)

    X_train, y_train, X_valid, y_valid, validation_meta = split_train_validation(
        feature_frame,
        validation_days=CONFIG.validation_days,
        include_price=CONFIG.include_price,
    )
    train_result = train_model(
        X_train, y_train, CONFIG, run_name="airflow_weekly_training"
    )

    with TRAINING_PATH.open("wb") as handle:
        pickle.dump(
            {
                "model": train_result.model,
                "run_id": train_result.run_id,
                "params": train_result.params,
                "X_valid": X_valid,
                "y_valid": y_valid,
                "validation_meta": validation_meta,
            },
            handle,
        )
    return str(TRAINING_PATH)


def evaluate_model_fn() -> str:
    with TRAINING_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    metrics, _ = evaluate_model(
        payload["model"],
        payload["X_valid"],
        payload["y_valid"],
        payload["validation_meta"],
        CONFIG,
        run_id=payload["run_id"],
    )
    artifact_path = save_model_bundle(
        payload["model"],
        select_feature_columns(include_price=CONFIG.include_price),
        metrics,
        CONFIG,
        params=payload["params"],
        run_id=payload["run_id"],
    )
    payload["artifact_path"] = str(artifact_path)
    with TRAINING_PATH.open("wb") as handle:
        pickle.dump(payload, handle)
    return str(artifact_path)


def register_model_fn() -> str:
    return register_best_model(CONFIG)


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="m5_demand_training",
    default_args=default_args,
    description="Weekly retraining pipeline for the M5 demand forecast model.",
    schedule_interval="@weekly",
    catchup=False,
) as dag:
    prepare_data = PythonOperator(
        task_id="prepare_data", python_callable=prepare_data_fn
    )
    train = PythonOperator(task_id="train_model", python_callable=train_model_fn)
    evaluate = PythonOperator(
        task_id="evaluate_model", python_callable=evaluate_model_fn
    )
    register = PythonOperator(
        task_id="register_model", python_callable=register_model_fn
    )

    prepare_data >> train >> evaluate >> register
