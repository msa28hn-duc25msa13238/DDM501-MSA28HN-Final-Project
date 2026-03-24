"""Locust load test for POST /predict (install locust separately: pip install locust)."""

from __future__ import annotations

from locust import HttpUser, between, task


class ForecastApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def predict(self) -> None:
        payload = {
            "item_id": "FOODS_1_001",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
            "forecast_start_date": "2016-04-25",
            "horizon": 7,
            "recent_demand": [float(i % 5) for i in range(28)],
            "current_price": 4.99,
        }
        self.client.post("/predict", json=payload, name="/predict")
