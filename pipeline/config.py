from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os


@dataclass
class TrainingConfig:
    data_dir: Path = Path("m5_data")
    model_dir: Path = Path("models")
    model_artifact_name: str = "forecast_model.pkl"
    experiment_name: str = "m5-demand-forecast"
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    registry_model_name: str = "m5-demand-forecast-model"
    enable_mlflow: bool = True
    include_price: bool = True
    max_series: int = 300
    recent_days: int = 365
    validation_days: int = 28
    max_lag_days: int = 28
    random_state: int = 42
    register_model: bool = False
    model_params: dict[str, float | int] = field(
        default_factory=lambda: {
            "learning_rate": 0.05,
            "max_depth": 8,
            "max_iter": 300,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        }
    )

    @property
    def model_artifact_path(self) -> Path:
        return self.model_dir / self.model_artifact_name

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["model_dir"] = str(self.model_dir)
        payload["model_artifact_path"] = str(self.model_artifact_path)
        return payload
