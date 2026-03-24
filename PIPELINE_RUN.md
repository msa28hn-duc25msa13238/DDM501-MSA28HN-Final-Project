# Hướng dẫn chạy pipeline & thử nghiệm (M5 Demand Forecast)

Tài liệu này mô tả **luồng hoàn chỉnh**: huấn luyện, API, giám sát Prometheus, báo cáo Evidently, kiểm thử tải, pre-commit và CI. Chi tiết kiến trúc: [ARCHITECTURE.md](./ARCHITECTURE.md). Monitoring vận hành: [MONITORING.md](./MONITORING.md). Mở rộng kỹ thuật: [PIPELINE_EXTENSIONS.md](./PIPELINE_EXTENSIONS.md).

---

## 1. Chuẩn bị môi trường

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

- Sao chép [`.env.example`](./.env.example) thành `.env` nếu cần chỉnh biến môi trường (ứng dụng load qua `python-dotenv` khi chạy API và các entrypoint pipeline).
- Đặt dữ liệu M5 thô vào `m5_data/` theo [README.md](./README.md).

---

## 2. Huấn luyện và tái lập (reproducibility)

- **Seed:** `RANDOM_STATE` trong `.env` hoặc mặc định `42`. Pipeline gọi `set_global_seed` trước khi nạp dữ liệu; `HistGradientBoostingRegressor` dùng cùng giá trị từ `TrainingConfig`. MLflow log tham số `random_state`.

```bash
python -m pipeline.run_pipeline --max-series 300 --random-state 42
# hoặc
python -m scripts.train_baseline
```

- **Batch thí nghiệm MLflow:**

```bash
python -m experiments.run_experiments
```

---

## 3. Chạy API cục bộ

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | Mục đích |
|----------|----------|
| `GET /health` | Liveness + trạng thái model |
| `GET /model/info` | Metadata / metrics artifact |
| `GET /metrics` | Prometheus (HTTP histogram, counters) |
| `POST /predict` | Dự báo (xem README cho JSON mẫu) |
| `GET /docs` | Swagger |

Kiểm nhanh:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/metrics | head
```

---

## 4. Docker Compose: API + MLflow + Airflow + Prometheus

Từ thư mục gốc repo:

1. `docker compose up --build -d mlflow`
2. `docker compose up --build airflow-init`
3. Huấn luyện / thí nghiệm (tùy nhu cầu), ví dụ:
   - `docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000 api python -m pipeline.run_pipeline`
4. `docker compose up -d api prometheus airflow-webserver airflow-scheduler`

URL sau khi chạy:

- API: http://localhost:8000  
- Prometheus UI: http://localhost:9090 (job scrape `api:8000/metrics`, cấu hình trong [`monitoring/prometheus.yml`](./monitoring/prometheus.yml))  
- MLflow: http://localhost:5001  
- Airflow: http://localhost:8080 (mặc định local: `admin` / `admin`)

---

## 5. Báo cáo drift (Evidently)

Cần hai file CSV cùng tập cột feature với `pipeline.features` (có hoặc không `sell_price` tùy `--no-price`).

```bash
python -m scripts.evidently_report --reference path/to/ref_features.csv --current path/to/current_features.csv --output reports/evidently_drift.html
```

Chạy thử nhanh **không cần dữ liệu M5** (tạo CSV demo + HTML trong `reports/`):

```bash
python -m scripts.evidently_report --demo
```

Nếu gặp lỗi import Evidently do môi trường trộn phiên bản, hãy dùng **venv sạch** và cài lại: `pip install --force-reinstall -r requirements.txt`.

---

## 6. Mô phỏng tải (Locust) và benchmark offline

**Locust** (cài thêm: `pip install locust` — không bắt buộc trong `requirements.txt` để giảm phụ thuộc):

```bash
locust -f simulations/locustfile.py --host http://localhost:8000
```

Mở UI Locust (mặc định http://localhost:8089), đặt số user và spawn rate, quan sát latency; đối chiếu với metric histogram trên `/metrics` và Prometheus.

**Benchmark gọi trực tiếp `DemandForecaster`** (không qua HTTP):

```bash
python -m simulations.benchmark_offline --model-path models/forecast_model.pkl --data-dir m5_data --iterations 200
```

Script sẽ tạo `m5_data/calendar.csv` tối thiểu nếu thiếu (chỉ phục vụ demo benchmark).

---

## 7. Pre-commit (local)

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Cấu hình: [`.pre-commit-config.yaml`](./.pre-commit-config.yaml) (Ruff lint + format).

---

## 8. CI (GitHub Actions)

Workflow: [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) — cài `requirements.txt`, **Ruff** (`check` + `format --check`), **pip-audit**, **pytest**.

---

## 9. Checklist smoke test trước demo

1. `pytest -v` pass.  
2. `python -m pipeline.run_pipeline` (hoặc Docker tương đương) tạo `models/forecast_model.pkl`.  
3. `GET /health` → `healthy`, `GET /metrics` trả về text Prometheus.  
4. Prometheus **Targets** thấy job `api` **UP** (nếu dùng Compose service `prometheus`).  
5. (Tuỳ chọn) `python -m scripts.evidently_report --demo` tạo HTML trong `reports/`.
