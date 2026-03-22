from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.config import AppSettings
from app.main import create_app
from tests.helpers import create_test_bundle, write_calendar_csv


async def _request(app, method: str, path: str, payload: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, json=payload)


def test_health_and_model_info(tmp_path: Path) -> None:
    data_dir = tmp_path / "m5_data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    write_calendar_csv(data_dir / "calendar.csv")
    create_test_bundle(model_dir / "forecast_model.pkl")

    app = create_app(AppSettings(model_path=model_dir / "forecast_model.pkl", data_dir=data_dir))
    health_response = asyncio.run(_request(app, "GET", "/health"))
    assert health_response.status_code == 200
    assert health_response.json()["model_loaded"] is True

    info_response = asyncio.run(_request(app, "GET", "/model/info"))
    assert info_response.status_code == 200
    assert "rmse" in info_response.json()["metrics"]


def test_predict_endpoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "m5_data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    write_calendar_csv(data_dir / "calendar.csv", days=60)
    create_test_bundle(model_dir / "forecast_model.pkl")

    app = create_app(AppSettings(model_path=model_dir / "forecast_model.pkl", data_dir=data_dir))
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/predict",
            {
            "item_id": "FOODS_1_001",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
            "forecast_start_date": "2016-01-29",
            "horizon": 3,
            "recent_demand": [float(index % 5) for index in range(28)],
            "current_price": 4.5,
            },
        )
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["horizon"] == 3
    assert len(payload["forecasts"]) == 3


def test_predict_validation_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "m5_data"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()
    write_calendar_csv(data_dir / "calendar.csv")
    create_test_bundle(model_dir / "forecast_model.pkl")

    app = create_app(AppSettings(model_path=model_dir / "forecast_model.pkl", data_dir=data_dir))
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/predict",
            {
            "item_id": "FOODS_1_001",
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "store_id": "CA_1",
            "state_id": "CA",
            "forecast_start_date": "2016-01-29",
            "horizon": 3,
            "recent_demand": [1.0, 2.0],
            },
        )
    )
    assert response.status_code == 422
