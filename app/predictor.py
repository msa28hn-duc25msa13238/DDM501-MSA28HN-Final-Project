from __future__ import annotations

from datetime import timedelta
import pickle
from typing import Any

import pandas as pd

from app.config import AppSettings
from app.schemas import PredictionRequest
from pipeline.features import build_inference_row


class DemandForecaster:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.model: Any | None = None
        self.bundle: dict[str, Any] | None = None
        self.calendar: pd.DataFrame | None = None
        self.model_version: str | None = None
        self.model_load_error: str | None = None
        self._load_calendar()
        self.reload_model()

    def _load_calendar(self) -> None:
        calendar_path = self.settings.data_dir / "calendar.csv"
        if not calendar_path.exists():
            self.calendar = None
            return
        calendar = pd.read_csv(calendar_path, parse_dates=["date"])
        for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            calendar[column] = calendar[column].fillna("None")
        self.calendar = calendar.set_index("date")

    def reload_model(self) -> None:
        model_path = self.settings.model_path
        if not model_path.exists():
            self.bundle = None
            self.model = None
            self.model_version = None
            self.model_load_error = None
            return
        try:
            with model_path.open("rb") as handle:
                self.bundle = pickle.load(handle)
        except Exception as exc:
            self.bundle = None
            self.model = None
            self.model_version = None
            self.model_load_error = str(exc)
            return
        self.model = self.bundle["model"]
        self.model_version = self.bundle.get("model_version", "unknown")
        self.model_load_error = None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.calendar is not None

    def get_model_info(self) -> dict[str, Any]:
        if not self.bundle:
            if self.model_load_error:
                raise RuntimeError(
                    f"Model artifact could not be loaded: {self.model_load_error}"
                )
            raise RuntimeError("Model artifact has not been created yet.")
        return {
            "model_version": self.bundle.get("model_version", "unknown"),
            "trained_at": self.bundle.get("trained_at"),
            "metrics": self.bundle.get("metrics", {}),
            "feature_columns": self.bundle.get("feature_columns", []),
            "artifact_path": str(self.settings.model_path),
        }

    def _calendar_row(self, forecast_date: pd.Timestamp) -> pd.Series:
        if self.calendar is None:
            raise RuntimeError("calendar.csv is not available.")
        try:
            return self.calendar.loc[forecast_date]
        except KeyError as exc:
            raise RuntimeError(
                f"Date {forecast_date.date()} does not exist in calendar.csv."
            ) from exc

    def predict(self, request: PredictionRequest) -> list[dict[str, Any]]:
        if not self.model_loaded or not self.bundle:
            if self.model_load_error:
                raise RuntimeError(
                    f"Model could not be loaded: {self.model_load_error}"
                )
            raise RuntimeError(
                "Model is not loaded. Train a model before serving predictions."
            )

        history = [float(value) for value in request.recent_demand]
        feature_columns = self.bundle["feature_columns"]
        results: list[dict[str, Any]] = []
        current_date = pd.Timestamp(request.forecast_start_date)

        for _ in range(request.horizon):
            calendar_row = self._calendar_row(current_date)
            feature_row = build_inference_row(
                item_id=request.item_id,
                dept_id=request.dept_id,
                cat_id=request.cat_id,
                store_id=request.store_id,
                state_id=request.state_id,
                history=history,
                calendar_row=calendar_row,
                current_price=request.current_price,
            )
            frame = pd.DataFrame([feature_row], columns=feature_columns)
            raw_prediction = float(self.model.predict(frame)[0])
            prediction = max(0.0, raw_prediction)
            results.append(
                {
                    "date": current_date.date(),
                    "predicted_demand": round(prediction, 4),
                }
            )
            history.append(prediction)
            current_date = current_date + timedelta(days=1)

        return results
