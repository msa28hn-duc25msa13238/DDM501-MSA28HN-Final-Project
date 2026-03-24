from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pipeline.config import TrainingConfig
from pipeline.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass
class TrainResult:
    model: Pipeline
    run_id: str | None
    params: dict[str, float | int]


def _resolve_experiment_name(mlflow_module, config: TrainingConfig) -> str:
    client = mlflow_module.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        artifact_root = str((Path.cwd() / "mlruns" / config.experiment_name).resolve())
        client.create_experiment(
            config.experiment_name, artifact_location=artifact_root
        )
        return config.experiment_name

    if config.tracking_uri.startswith(
        "sqlite:"
    ) and experiment.artifact_location.startswith("/app/"):
        local_name = f"{config.experiment_name}-local"
        local_experiment = client.get_experiment_by_name(local_name)
        if local_experiment is None:
            artifact_root = str((Path.cwd() / "mlruns" / local_name).resolve())
            client.create_experiment(local_name, artifact_location=artifact_root)
        return local_name

    return config.experiment_name


@contextmanager
def _maybe_mlflow_run(config: TrainingConfig, run_name: str | None):
    if not config.enable_mlflow:
        yield None
        return

    try:
        import mlflow
    except ImportError:
        yield None
        return

    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        experiment_name = _resolve_experiment_name(mlflow, config)
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize MLflow tracking at {config.tracking_uri}. "
            "Check the tracking URI or override MLFLOW_TRACKING_URI."
        ) from exc
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def build_estimator(
    model_params: dict[str, float | int] | None = None,
    include_price: bool = True,
    *,
    random_state: int = 42,
) -> Pipeline:
    params = {
        "learning_rate": 0.05,
        "max_depth": 8,
        "max_iter": 300,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
    }
    if model_params:
        params.update(model_params)

    numeric_features = list(NUMERIC_FEATURES)
    if not include_price:
        numeric_features.remove("sell_price")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    regressor = HistGradientBoostingRegressor(
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        max_iter=int(params["max_iter"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        l2_regularization=float(params["l2_regularization"]),
        random_state=int(random_state),
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def train_model(
    X_train,
    y_train,
    config: TrainingConfig,
    *,
    run_name: str | None = None,
    model_params: dict[str, float | int] | None = None,
) -> TrainResult:
    params = dict(config.model_params)
    if model_params:
        params.update(model_params)

    estimator = build_estimator(
        params,
        include_price=config.include_price,
        random_state=config.random_state,
    )

    with _maybe_mlflow_run(config, run_name) as run:
        if run is not None:
            import mlflow
            import mlflow.sklearn

            mlflow.log_params(params)
            mlflow.log_param("random_state", config.random_state)
            mlflow.log_param("include_price", config.include_price)
            mlflow.log_param("max_series", config.max_series)
            mlflow.log_param("recent_days", config.recent_days)
            mlflow.log_param("validation_days", config.validation_days)
            mlflow.log_param("train_rows", len(X_train))

        estimator.fit(X_train, y_train)

        if run is not None:
            import mlflow.sklearn

            mlflow.sklearn.log_model(estimator, artifact_path="model")
            return TrainResult(model=estimator, run_id=run.info.run_id, params=params)

    return TrainResult(model=estimator, run_id=None, params=params)
