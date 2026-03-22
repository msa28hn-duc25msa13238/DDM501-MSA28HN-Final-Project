from __future__ import annotations

import argparse
import json

from pipeline.config import TrainingConfig
from pipeline.run_pipeline import run_pipeline


EXPERIMENTS = [
    {"name": "baseline_with_price", "include_price": True, "model_params": {"max_depth": 8, "max_iter": 250}},
    {"name": "baseline_without_price", "include_price": False, "model_params": {"max_depth": 8, "max_iter": 250}},
    {"name": "shallower_model", "include_price": True, "model_params": {"max_depth": 6, "max_iter": 250}},
    {"name": "faster_learning_rate", "include_price": True, "model_params": {"learning_rate": 0.08, "max_iter": 220}},
    {"name": "deeper_model", "include_price": True, "model_params": {"max_depth": 10, "max_iter": 320}},
]


def run_all_experiments(
    *,
    max_series: int = 300,
    recent_days: int = 365,
    validation_days: int = 28,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        config = TrainingConfig(
            include_price=experiment["include_price"],
            max_series=max_series,
            recent_days=recent_days,
            validation_days=validation_days,
        )
        outcome = run_pipeline(
            config,
            run_name=experiment["name"],
            model_params=experiment["model_params"],
        )
        results.append({"experiment": experiment["name"], **outcome})
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Lab 2 MLflow experiment batch.")
    parser.add_argument("--max-series", type=int, default=300)
    parser.add_argument("--recent-days", type=int, default=365)
    parser.add_argument("--validation-days", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_all_experiments(
                max_series=args.max_series,
                recent_days=args.recent_days,
                validation_days=args.validation_days,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
