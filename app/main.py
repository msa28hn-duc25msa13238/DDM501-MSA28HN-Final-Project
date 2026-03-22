from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import AppSettings
from app.predictor import DemandForecaster
from app.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse


def create_app(settings: AppSettings | None = None) -> FastAPI:
    app_settings = settings or AppSettings.from_env()
    forecaster = DemandForecaster(app_settings)

    app = FastAPI(
        title=app_settings.title,
        description=app_settings.description,
        version=app_settings.version,
    )
    app.state.forecaster = forecaster

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Return service health and whether the trained model artifact is loaded."""
        status = "healthy" if app.state.forecaster.model_loaded else "degraded"
        return HealthResponse(
            status=status,
            model_loaded=app.state.forecaster.model_loaded,
            model_version=app.state.forecaster.model_version,
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        """Return metadata about the currently loaded model artifact."""
        try:
            info = app.state.forecaster.get_model_info()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ModelInfoResponse(**info)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse:
        """Generate recursive demand forecasts from a recent demand history window."""
        try:
            forecasts = app.state.forecaster.predict(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive API wrapper
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PredictionResponse(
            item_id=request.item_id,
            store_id=request.store_id,
            horizon=request.horizon,
            forecasts=forecasts,
            model_version=app.state.forecaster.model_version or "unknown",
        )

    return app


app = create_app()
