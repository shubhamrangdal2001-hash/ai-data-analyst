"""Machine Learning endpoints."""
from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter

from backend.models.schemas import APIResponse
from backend.services.ml_service import forecast_time_series, train_and_compare

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


class TrainRequest(BaseModel):
    target_column: str
    task_type: str = "auto"
    test_size: float = 0.2
    tune: bool = False
    n_trials: int = 20


class ForecastRequest(BaseModel):
    date_col: str
    value_col: str
    periods: int = 30
    freq: str = "D"


@router.post("/{dataset_id}/train", response_model=APIResponse)
async def train_models(dataset_id: str, req: TrainRequest):
    """Train and compare multiple ML models on the dataset."""
    result = train_and_compare(
        dataset_id=dataset_id,
        target_column=req.target_column,
        task_type=req.task_type,
        test_size=req.test_size,
        tune=req.tune,
        n_trials=req.n_trials,
    )
    return APIResponse(success=True, data=result.model_dump(), message=f"Best model: {result.best_model}")


@router.post("/{dataset_id}/forecast", response_model=APIResponse)
async def forecast(dataset_id: str, req: ForecastRequest):
    """Time-series forecast."""
    result = forecast_time_series(
        dataset_id=dataset_id,
        date_col=req.date_col,
        value_col=req.value_col,
        periods=req.periods,
        freq=req.freq,
    )
    return APIResponse(success=True, data=result, message="Forecast complete")
