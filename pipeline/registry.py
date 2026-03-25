from __future__ import annotations

from pipeline.config import TrainingConfig


def register_best_model(config: TrainingConfig, stage: str = "Production") -> str:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("MLflow is not installed.") from exc

    mlflow.set_tracking_uri(config.tracking_uri)
    client = MlflowClient(tracking_uri=config.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment {config.experiment_name!r} does not exist.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No completed runs found for registration.")

    best_run = runs[0]
    registration = mlflow.register_model(
        f"runs:/{best_run.info.run_id}/model", config.registry_model_name
    )
    client.transition_model_version_stage(
        name=config.registry_model_name,
        version=registration.version,
        stage=stage,
    )
    return str(registration.version)
