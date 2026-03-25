# Monitoring Guide

This project includes a local monitoring stack for the FastAPI inference
service:

- **Prometheus** for scraping and storing `/metrics`
- **Grafana** for pre-provisioned dashboards
- **Alertmanager** for viewing routed alerts
- **Prometheus alert rules** for basic API health, latency, and error checks

## Services

Run the monitoring stack with the API:

```bash
docker compose up -d api prometheus grafana alertmanager
```

If you also want the full demo stack:

```bash
docker compose up -d mlflow
docker compose up airflow-init
docker compose up -d api prometheus grafana alertmanager airflow-webserver airflow-scheduler
```

## URLs

- API metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Alertmanager: `http://localhost:9093`

Default local login:

- Grafana: `admin` / `admin`

## What To Explore

### Prometheus metrics collection

1. Open Prometheus and go to `Status -> Targets`.
2. Confirm the `api` job is `UP`.
3. Try queries such as:

```promql
up{job="api"}
sum(rate(http_requests_total{job="api"}[5m])) by (handler, method, status)
histogram_quantile(0.95, sum by (le, handler, method) (rate(http_request_duration_seconds_bucket{job="api"}[5m])))
```

### Grafana dashboards

Grafana auto-loads the `M5 API Observability` dashboard from:

- `monitoring/grafana/dashboards/api-observability.json`

The dashboard includes:

- request rate
- API up status
- 5xx error ratio
- `POST /predict` p95 latency
- endpoint p95 latency trends
- requests by status

### Alerting rules

Prometheus loads alert rules from:

- `monitoring/alerts/api-alerts.yml`

Starter alerts included:

- `ApiDown`
- `PredictEndpointHighP95Latency`
- `High5xxErrorRate`
- `LowTrafficToPredictEndpoint`

Open `Alerts` in Prometheus to inspect rule state. Alertmanager receives routed
alerts and exposes them in its own UI.

## Notes

- The starter Alertmanager configuration is intentionally minimal and keeps
  alerts local for demo use.
- If you want email, Slack, or webhook notifications, extend
  `monitoring/alertmanager/alertmanager.yml`.
