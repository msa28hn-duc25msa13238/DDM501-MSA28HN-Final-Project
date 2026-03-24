"""Measure offline inference latency by calling DemandForecaster.predict in a loop."""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.config import AppSettings
from app.schemas import PredictionRequest
from app.predictor import DemandForecaster


def _ensure_minimal_calendar(data_dir: Path, days: int = 60) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cal = data_dir / "calendar.csv"
    if cal.exists():
        return
    rows = []
    start = date(2016, 1, 1)
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
    pd.DataFrame(rows).to_csv(cal, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline benchmark for DemandForecaster.predict."
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/forecast_model.pkl")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("m5_data"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()

    _ensure_minimal_calendar(args.data_dir)

    forecaster = DemandForecaster(
        AppSettings(model_path=args.model_path, data_dir=args.data_dir)
    )
    if not forecaster.model_loaded:
        raise SystemExit(
            f"Model not loaded. Train first or set MODEL_PATH ({args.model_path})."
        )

    request = PredictionRequest(
        item_id="FOODS_1_001",
        dept_id="FOODS_1",
        cat_id="FOODS",
        store_id="CA_1",
        state_id="CA",
        forecast_start_date=date(2016, 1, 29),
        horizon=args.horizon,
        recent_demand=[float(i % 5) for i in range(28)],
        current_price=4.5,
    )

    # Warmup
    forecaster.predict(request)

    t0 = time.perf_counter()
    for _ in range(args.iterations):
        forecaster.predict(request)
    elapsed = time.perf_counter() - t0
    print(
        f"iterations={args.iterations} horizon={args.horizon} total_s={elapsed:.4f} avg_ms={1000 * elapsed / args.iterations:.3f}"
    )


if __name__ == "__main__":
    main()
