# M5 Demand Forecast

This repository implements an MLOps-first M5 demand forecasting project that
covers the main deliverables from DDM501 Lab 1 and Lab 2:

- Lab 1: train a demand model, expose it through FastAPI, dockerize it, and test it
- Lab 2: refactor training into a modular pipeline, track experiments with MLflow,
  and orchestrate retraining with Airflow

The goal is not to win the M5 competition. The goal is to build a clean,
reproducible ML product around the M5 data.

## Project Structure

```text
.
├── app/                  # FastAPI inference service
├── pipeline/             # Training, evaluation, and registry logic
├── experiments/          # Repeated MLflow experiment runs
├── dags/                 # Airflow DAGs
├── scripts/              # Convenience entrypoints
├── tests/                # Unit tests
├── models/               # Local model artifacts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Data

The project expects the original M5 CSV files under `m5_data/`:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`
- `sales_train_evaluation.csv`
- `sample_submission.csv`

The raw dataset is ignored by git because several files exceed GitHub's normal
file size limit.

## Model Design

The baseline model is a single global regressor trained on sampled
`item_id + store_id + date` rows. It uses:

- static IDs: item, department, category, store, state
- calendar features: weekday, month, year, events, SNAP
- demand lags: 1, 7, 28 days
- rolling demand statistics: 7-day and 28-day windows
- optional `sell_price`

The service predicts next-day demand and can recursively forecast up to 28 days.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train a baseline model artifact:

```bash
python -m scripts.train_baseline
```

4. Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. Open Swagger:

```text
http://localhost:8000/docs
```

## API

### `GET /health`

Health check for the service and model artifact.

### `GET /model/info`

Returns model metadata, metrics, and artifact path.

### `POST /predict`

Forecast demand from a recent demand history window.

Example request:

```json
{
  "item_id": "FOODS_1_001",
  "dept_id": "FOODS_1",
  "cat_id": "FOODS",
  "store_id": "CA_1",
  "state_id": "CA",
  "forecast_start_date": "2016-04-25",
  "horizon": 7,
  "recent_demand": [0, 1, 0, 2, 1, 0, 0, 3, 2, 1, 0, 1, 2, 0, 0, 1, 1, 0, 2, 2, 1, 0, 1, 3, 1, 0, 2, 2],
  "current_price": 4.99
}
```

## Training Pipeline

Run a single training pipeline:

```bash
python -m pipeline.run_pipeline
```

Run the MLflow experiment batch:

```bash
python -m experiments.run_experiments
```

Start MLflow locally:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```

MLflow UI:

```text
http://localhost:5001
```

## Airflow

The repository includes a weekly retraining DAG in
[`dags/ml_training_dag.py`](/Users/albertdinh/Programming/DDM501/ddm501_msa28hn_demand_forecast/dags/ml_training_dag.py).

The simplest way to run Airflow is through Docker Compose:

```bash
docker compose up --build
```

Airflow UI:

```text
http://localhost:8080
```

Default credentials are set in `docker-compose.yml` for local development only.

## Docker

Build and run the API:

```bash
docker compose up --build api
```

Run the supporting MLOps stack:

```bash
docker compose up --build
```

## Tests

```bash
pytest -v
```

## Lab Mapping

Lab 1 deliverables covered by this repo:

- trained model artifact and loading code
- FastAPI app with `/health`, `/predict`, and `/model/info`
- Dockerfile and Compose service
- pytest test suite
- README and Swagger docs

Lab 2 deliverables covered by this repo:

- modular pipeline under `pipeline/`
- MLflow experiment tracking and model logging
- experiment runner with multiple configurations
- Airflow retraining DAG
- model registry integration helper
