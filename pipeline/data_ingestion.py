from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import TrainingConfig


ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _day_number(day_id: str) -> int:
    return int(day_id.split("_")[1])


def _snap_for_state(frame: pd.DataFrame) -> pd.Series:
    return np.select(
        [
            frame["state_id"].eq("CA"),
            frame["state_id"].eq("TX"),
            frame["state_id"].eq("WI"),
        ],
        [frame["snap_CA"], frame["snap_TX"], frame["snap_WI"]],
        default=0,
    )


def _select_day_columns(
    calendar: pd.DataFrame, available_days: set[str], config: TrainingConfig
) -> list[str]:
    total_days = config.recent_days + config.validation_days + config.max_lag_days
    day_frame = calendar[["d"]].copy()
    day_frame["day_number"] = day_frame["d"].map(_day_number)
    day_frame = day_frame[day_frame["d"].isin(available_days)]
    return day_frame.sort_values("day_number").tail(total_days)["d"].tolist()


def load_modeling_frame(config: TrainingConfig) -> pd.DataFrame:
    calendar_path = config.data_dir / "calendar.csv"
    sales_path = config.data_dir / "sales_train_validation.csv"
    prices_path = config.data_dir / "sell_prices.csv"

    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    sales_columns = pd.read_csv(sales_path, nrows=0).columns.tolist()
    available_days = {column for column in sales_columns if column.startswith("d_")}
    selected_days = _select_day_columns(calendar, available_days, config)

    sales = pd.read_csv(sales_path, usecols=ID_COLUMNS + selected_days)
    if config.max_series and config.max_series < len(sales):
        sales = sales.sample(
            n=config.max_series, random_state=config.random_state
        ).reset_index(drop=True)

    long_frame = sales.melt(
        id_vars=ID_COLUMNS,
        value_vars=selected_days,
        var_name="d",
        value_name="demand",
    )

    calendar_subset = calendar[calendar["d"].isin(selected_days)].copy()
    merged = long_frame.merge(calendar_subset, on="d", how="left")

    prices = pd.read_csv(prices_path)
    merged = merged.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["sell_price"] = merged.groupby(["store_id", "item_id"], sort=False)[
        "sell_price"
    ].transform(lambda s: s.ffill().bfill())
    merged["sell_price"] = merged["sell_price"].fillna(0.0)
    merged["snap"] = _snap_for_state(merged)

    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        merged[column] = merged[column].fillna("None")

    merged = merged.sort_values(["id", "date"]).reset_index(drop=True)
    return merged
