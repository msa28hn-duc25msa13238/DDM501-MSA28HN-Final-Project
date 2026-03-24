"""CLI to generate an Evidently data drift HTML report from feature CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from pipeline.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    select_feature_columns,
)


def _schema(include_price: bool) -> DataDefinition:
    numeric = list(NUMERIC_FEATURES)
    if not include_price:
        numeric.remove("sell_price")
    return DataDefinition(
        numerical_columns=numeric,
        categorical_columns=list(CATEGORICAL_FEATURES),
    )


def _load_features(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns: {sorted(missing)}")
    return frame[columns]


def _write_demo_pair(out_dir: Path) -> tuple[Path, Path]:
    """Create tiny reference/current CSVs with intentional distribution shift for smoke tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = select_feature_columns(include_price=True)
    base_row = {
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
        "lag_1": 2.0,
        "lag_7": 2.0,
        "lag_28": 1.0,
        "rolling_mean_7": 2.0,
        "rolling_mean_28": 1.5,
        "rolling_std_28": 0.5,
    }
    ref_rows = []
    cur_rows = []
    for i in range(80):
        r = dict(base_row)
        r["lag_1"] = float(i % 5)
        r["wday"] = (i % 7) + 1
        ref_rows.append(r)
    for i in range(80):
        r = dict(base_row)
        r["lag_1"] = float(20 + (i % 5))
        r["wday"] = (i % 7) + 1
        cur_rows.append(r)
    ref_path = out_dir / "demo_reference_features.csv"
    cur_path = out_dir / "demo_current_features.csv"
    pd.DataFrame(ref_rows, columns=columns).to_csv(ref_path, index=False)
    pd.DataFrame(cur_rows, columns=columns).to_csv(cur_path, index=False)
    return ref_path, cur_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Evidently DataDrift HTML report."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference feature CSV (training baseline).",
    )
    parser.add_argument(
        "--current", type=Path, default=None, help="Current batch feature CSV."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/evidently_drift.html")
    )
    parser.add_argument(
        "--no-price",
        action="store_true",
        help="Drop sell_price to match no-price training.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write synthetic reference/current under reports/ and run drift report.",
    )
    args = parser.parse_args()

    if args.demo:
        ref, cur = _write_demo_pair(Path("reports"))
    else:
        if args.reference is None or args.current is None:
            parser.error("--reference and --current are required unless --demo is set.")
        ref, cur = args.reference, args.current

    include_price = not args.no_price
    columns = select_feature_columns(include_price=include_price)
    schema = _schema(include_price)

    ref_frame = _load_features(ref, columns)
    cur_frame = _load_features(cur, columns)

    reference_ds = Dataset.from_pandas(ref_frame, data_definition=schema)
    current_ds = Dataset.from_pandas(cur_frame, data_definition=schema)

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_ds, reference_ds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.output))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
