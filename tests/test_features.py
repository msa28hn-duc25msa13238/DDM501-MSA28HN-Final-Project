from __future__ import annotations

import pandas as pd

from pipeline.features import REQUIRED_HISTORY, build_inference_row


def test_build_inference_row_uses_expected_lags() -> None:
    history = list(range(REQUIRED_HISTORY))
    calendar_row = pd.Series(
        {
            "weekday": "Monday",
            "wday": 1,
            "month": 1,
            "year": 2016,
            "event_name_1": "None",
            "event_type_1": "None",
            "event_name_2": "None",
            "event_type_2": "None",
            "snap_CA": 1,
        }
    )

    row = build_inference_row(
        item_id="FOODS_1_001",
        dept_id="FOODS_1",
        cat_id="FOODS",
        store_id="CA_1",
        state_id="CA",
        history=history,
        calendar_row=calendar_row,
        current_price=4.5,
    )

    assert row["lag_1"] == 27.0
    assert row["lag_7"] == 21.0
    assert row["lag_28"] == 0.0
    assert row["snap"] == 1
