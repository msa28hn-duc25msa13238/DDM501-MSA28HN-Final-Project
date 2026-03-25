from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

NUMERIC_FEATURES = [
    "wday",
    "month",
    "year",
    "snap",
    "sell_price",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
]

REQUIRED_HISTORY = 28


def select_feature_columns(include_price: bool = True) -> list[str]:
    numeric = list(NUMERIC_FEATURES)
    if not include_price:
        numeric.remove("sell_price")
    return CATEGORICAL_FEATURES + numeric


def add_demand_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.sort_values(["id", "date"]).copy()
    grouped = enriched.groupby("id", sort=False)["demand"]
    enriched["lag_1"] = grouped.shift(1)
    enriched["lag_7"] = grouped.shift(7)
    enriched["lag_28"] = grouped.shift(28)
    enriched["rolling_mean_7"] = grouped.transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    enriched["rolling_mean_28"] = grouped.transform(
        lambda s: s.shift(1).rolling(28).mean()
    )
    enriched["rolling_std_28"] = grouped.transform(
        lambda s: s.shift(1).rolling(28).std()
    )
    return enriched


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = add_demand_features(frame)
    feature_frame["rolling_std_28"] = feature_frame["rolling_std_28"].fillna(0.0)
    feature_frame = feature_frame.dropna(
        subset=["lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"]
    )
    return feature_frame.reset_index(drop=True)


def split_train_validation(
    feature_frame: pd.DataFrame,
    validation_days: int,
    include_price: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    cutoff = feature_frame["date"].max() - pd.Timedelta(days=validation_days - 1)
    train_frame = feature_frame[feature_frame["date"] < cutoff].copy()
    validation_frame = feature_frame[feature_frame["date"] >= cutoff].copy()
    feature_columns = select_feature_columns(include_price=include_price)

    if train_frame.empty or validation_frame.empty:
        raise ValueError(
            "Train/validation split is empty. Increase available history or reduce validation_days."
        )

    validation_meta = validation_frame[
        ["date", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ].copy()
    return (
        train_frame[feature_columns],
        train_frame["demand"].astype(float),
        validation_frame[feature_columns],
        validation_frame["demand"].astype(float),
        validation_meta,
    )


def _safe_event_value(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and np.isnan(value):
        return "None"
    text = str(value).strip()
    return text if text else "None"


def _snap_value(calendar_row: pd.Series, state_id: str) -> int:
    column = f"snap_{state_id}"
    if column in calendar_row:
        return int(calendar_row[column])
    if "snap" in calendar_row:
        return int(calendar_row["snap"])
    return 0


def build_inference_row(
    *,
    item_id: str,
    dept_id: str,
    cat_id: str,
    store_id: str,
    state_id: str,
    history: Iterable[float],
    calendar_row: pd.Series,
    current_price: float | None,
) -> dict[str, float | int | str]:
    demand_history = [float(value) for value in history]
    if len(demand_history) < REQUIRED_HISTORY:
        raise ValueError(
            f"At least {REQUIRED_HISTORY} days of recent demand are required."
        )

    last_7 = demand_history[-7:]
    last_28 = demand_history[-28:]
    return {
        "item_id": item_id,
        "dept_id": dept_id,
        "cat_id": cat_id,
        "store_id": store_id,
        "state_id": state_id,
        "weekday": str(calendar_row["weekday"]),
        "event_name_1": _safe_event_value(calendar_row.get("event_name_1")),
        "event_type_1": _safe_event_value(calendar_row.get("event_type_1")),
        "event_name_2": _safe_event_value(calendar_row.get("event_name_2")),
        "event_type_2": _safe_event_value(calendar_row.get("event_type_2")),
        "wday": int(calendar_row["wday"]),
        "month": int(calendar_row["month"]),
        "year": int(calendar_row["year"]),
        "snap": _snap_value(calendar_row, state_id),
        "sell_price": float(current_price or 0.0),
        "lag_1": float(demand_history[-1]),
        "lag_7": float(demand_history[-7]),
        "lag_28": float(demand_history[-28]),
        "rolling_mean_7": float(np.mean(last_7)),
        "rolling_mean_28": float(np.mean(last_28)),
        "rolling_std_28": float(np.std(last_28)),
    }
