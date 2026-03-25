from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pipeline.config import TrainingConfig
from pipeline.evaluation import save_model_bundle
from pipeline.features import select_feature_columns
from pipeline.training import build_estimator


def write_calendar_csv(
    path: Path, days: int = 40, start: date = date(2016, 1, 1)
) -> None:
    rows = []
    for offset in range(days):
        current = start + timedelta(days=offset)
        rows.append(
            {
                "date": current.isoformat(),
                "wm_yr_wk": 201600 + (offset // 7),
                "weekday": current.strftime("%A"),
                "wday": current.isoweekday(),
                "month": current.month,
                "year": current.year,
                "d": f"d_{offset + 1}",
                "event_name_1": "None",
                "event_type_1": "None",
                "event_name_2": "None",
                "event_type_2": "None",
                "snap_CA": 1 if offset % 5 == 0 else 0,
                "snap_TX": 0,
                "snap_WI": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def create_test_bundle(model_path: Path) -> None:
    feature_columns = select_feature_columns(include_price=True)
    training_frame = pd.DataFrame(
        [
            {
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "weekday": "Monday",
                "event_name_1": "None",
                "event_type_1": "None",
                "event_name_2": "None",
                "event_type_2": "None",
                "wday": 1,
                "month": 1,
                "year": 2016,
                "snap": 1,
                "sell_price": 4.5,
                "lag_1": 3,
                "lag_7": 2,
                "lag_28": 1,
                "rolling_mean_7": 2.0,
                "rolling_mean_28": 1.5,
                "rolling_std_28": 0.5,
            },
            {
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "weekday": "Tuesday",
                "event_name_1": "None",
                "event_type_1": "None",
                "event_name_2": "None",
                "event_type_2": "None",
                "wday": 2,
                "month": 1,
                "year": 2016,
                "snap": 0,
                "sell_price": 4.5,
                "lag_1": 4,
                "lag_7": 3,
                "lag_28": 1,
                "rolling_mean_7": 2.2,
                "rolling_mean_28": 1.6,
                "rolling_std_28": 0.6,
            },
            {
                "item_id": "FOODS_1_001",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": "CA_1",
                "state_id": "CA",
                "weekday": "Wednesday",
                "event_name_1": "None",
                "event_type_1": "None",
                "event_name_2": "None",
                "event_type_2": "None",
                "wday": 3,
                "month": 1,
                "year": 2016,
                "snap": 0,
                "sell_price": 4.5,
                "lag_1": 5,
                "lag_7": 4,
                "lag_28": 2,
                "rolling_mean_7": 2.8,
                "rolling_mean_28": 1.9,
                "rolling_std_28": 0.7,
            },
        ],
        columns=feature_columns,
    )
    target = pd.Series([3.2, 4.1, 5.0])
    model = build_estimator(include_price=True)
    model.fit(training_frame, target)
    config = TrainingConfig(model_dir=model_path.parent)
    save_model_bundle(
        model,
        feature_columns,
        {"rmse": 0.1, "mae": 0.1, "wape": 0.05},
        config,
        params=config.model_params,
        run_id=None,
    )


def write_synthetic_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    write_calendar_csv(data_dir / "calendar.csv")

    sales_rows = [
        {
            "id": "FOODS_1_001_CA_1_validation",
            "item_id": "FOODS_1_001",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
        },
        {
            "id": "FOODS_1_002_CA_1_validation",
            "item_id": "FOODS_1_002",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
        },
    ]
    for index, row in enumerate(sales_rows):
        for day in range(1, 41):
            row[f"d_{day}"] = (day + index) % 6
    pd.DataFrame(sales_rows).to_csv(
        data_dir / "sales_train_validation.csv", index=False
    )

    price_rows = []
    for item_id in ["FOODS_1_001", "FOODS_1_002"]:
        for week in range(201600, 201606):
            price_rows.append(
                {
                    "store_id": "CA_1",
                    "item_id": item_id,
                    "wm_yr_wk": week,
                    "sell_price": 3.5 if item_id.endswith("1") else 4.0,
                }
            )
    pd.DataFrame(price_rows).to_csv(data_dir / "sell_prices.csv", index=False)
