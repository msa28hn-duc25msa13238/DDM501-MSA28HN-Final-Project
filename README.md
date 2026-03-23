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

Airflow UI:

```text
http://localhost:8080
```

Default credentials are set in `docker-compose.yml` for local development only.

## Docker Demo Workflow

For the lab demo, use Docker Compose in modular steps rather than one large
`docker compose up` command. Long-running services should be started with
`docker compose up`, while one-off jobs such as experiment runs and training
should be executed with `docker compose run --rm`.

Run these commands from the project root:

1. Start MLflow:

```bash
docker compose up --build -d mlflow
```

2. Initialize Airflow metadata and create the local admin user:

```bash
docker compose up --build airflow-init
```

3. Run the experiment batch and log runs to MLflow:

```bash
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 api python -m experiments.run_experiments
```

4. Train the final model artifact used by the API:

```bash
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 api python -m pipeline.run_pipeline
```

5. Start the API and Airflow services:

```bash
docker compose up -d api airflow-webserver airflow-scheduler
```

6. Verify container status:

```bash
docker compose ps
```

Useful URLs after startup:

- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`
- MLflow UI: `http://localhost:5001`
- Airflow UI: `http://localhost:8080`
- Airflow login: `admin` / `admin`

If `models/forecast_model.pkl` already exists and you only want to bring the
stack back up without rerunning experiments and training:

```bash
docker compose up -d mlflow
docker compose up airflow-init
docker compose up -d api airflow-webserver airflow-scheduler
docker compose ps
```

To stop the stack:

```bash
docker compose down
```

## Tests

```bash
pytest -v
```

## CI/CD (GitHub Actions)

This repository now includes 3 workflows under `.github/workflows/`:

- `ci.yml`: runs on pull requests and pushes to `main`
  - install dependencies with Python 3.12
  - run `pytest -v`
  - build both Docker images (`Dockerfile` and `Dockerfile.airflow`)
- `cd.yml`: runs on pushes to `main` and manual dispatch
  - build and push images to GitHub Container Registry (GHCR)
  - tags both `latest` and `sha-<commit>`
  - optionally triggers staging deploy webhook when configured
- `retrain.yml`: runs weekly and manual dispatch
  - runs on a self-hosted Windows runner with labels `self-hosted`, `Windows`, `X64`, `m5-local`
  - trains the model if a data directory is configured on that runner
  - checks a WAPE quality gate
  - uploads the retrained model artifact

### Required Repository Settings

Configure the following repository secrets in GitHub:

- `DEPLOY_WEBHOOK_URL` (optional): webhook endpoint used by `cd.yml`
- `M5_DATA_DIR` (required for `retrain.yml`): absolute path to M5 CSV directory on the self-hosted runner, for example `D:\datasets\m5_data`

For retraining, `retrain.yml` uses local MLflow on the self-hosted runner:

- `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`

Start MLflow on that machine before running retrain:

```bash
docker compose up -d mlflow
```

If `M5_DATA_DIR` is not set, the retraining workflow exits early without training.

### Container Images

`cd.yml` publishes:

- `ghcr.io/<owner>/m5-forecast-api:latest`
- `ghcr.io/<owner>/m5-forecast-api:sha-<commit>`
- `ghcr.io/<owner>/m5-forecast-airflow:latest`
- `ghcr.io/<owner>/m5-forecast-airflow:sha-<commit>`

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
