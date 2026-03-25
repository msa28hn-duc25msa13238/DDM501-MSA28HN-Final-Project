from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ForecastPoint(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    date: date
    predicted_demand: float


class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    item_id: str = Field(..., examples=["FOODS_1_001"])
    dept_id: str = Field(..., examples=["FOODS_1"])
    cat_id: str = Field(..., examples=["FOODS"])
    store_id: str = Field(..., examples=["CA_1"])
    state_id: str = Field(..., examples=["CA"])
    forecast_start_date: date = Field(..., examples=["2016-04-25"])
    horizon: int = Field(default=1, ge=1, le=28)
    recent_demand: List[float] = Field(
        ...,
        min_length=28,
        description="Last 28 or more daily unit sales values in chronological order.",
    )
    current_price: Optional[float] = Field(default=None, ge=0)

    @field_validator("state_id")
    @classmethod
    def validate_state(cls, value: str) -> str:
        if value not in {"CA", "TX", "WI"}:
            raise ValueError("state_id must be one of: CA, TX, WI")
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    item_id: str
    store_id: str
    horizon: int
    forecasts: List[ForecastPoint]
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    trained_at: Optional[str] = None
    metrics: dict[str, float]
    responsible_ai: dict[str, object] = Field(default_factory=dict)
    feature_columns: List[str]
    artifact_path: str
