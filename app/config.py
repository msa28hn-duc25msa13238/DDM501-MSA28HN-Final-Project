from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class AppSettings:
    title: str = "M5 Demand Forecast API"
    description: str = (
        "API for serving recursive demand forecasts on M5-style retail data."
    )
    version: str = "1.0.0"
    model_path: Path = Path("models/forecast_model.pkl")
    data_dir: Path = Path("m5_data")

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            title=os.getenv("APP_TITLE", cls.title),
            description=os.getenv("APP_DESCRIPTION", cls.description),
            version=os.getenv("APP_VERSION", cls.version),
            model_path=Path(os.getenv("MODEL_PATH", str(cls.model_path))),
            data_dir=Path(os.getenv("DATA_DIR", str(cls.data_dir))),
        )
