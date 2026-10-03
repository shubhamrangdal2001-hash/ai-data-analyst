"""Exploratory Data Analysis endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from backend.models.schemas import APIResponse
from backend.services.eda_service import (
    get_categorical_counts,
    get_distribution_data,
    get_time_series_data,
    run_eda,
)

router = APIRouter(prefix="/eda", tags=["EDA"])


@router.get("/{dataset_id}", response_model=APIResponse)
async def exploratory_analysis(dataset_id: str):
    """Run full EDA on a dataset."""
    result = run_eda(dataset_id)
    return APIResponse(success=True, data=result.model_dump(), message="EDA complete")


@router.get("/{dataset_id}/distribution/{column}", response_model=APIResponse)
async def column_distribution(dataset_id: str, column: str):
    data = get_distribution_data(dataset_id, column)
    return APIResponse(success=True, data=data)


@router.get("/{dataset_id}/categorical/{column}", response_model=APIResponse)
async def categorical_counts(dataset_id: str, column: str, top_n: int = 15):
    data = get_categorical_counts(dataset_id, column, top_n)
    return APIResponse(success=True, data=data)


@router.get("/{dataset_id}/timeseries", response_model=APIResponse)
async def time_series(dataset_id: str, date_col: str, value_col: str, freq: str = "M"):
    data = get_time_series_data(dataset_id, date_col, value_col, freq)
    return APIResponse(success=True, data=data)
