"""Shared Pydantic response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]


class DatasetMeta(BaseModel):
    filename: str
    rows: int
    columns: int
    size_kb: float
    column_info: list[ColumnInfo]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    memory_mb: float


class StatsSummary(BaseModel):
    column: str
    count: float
    mean: float | None
    std: float | None
    min: float | None
    q25: float | None
    median: float | None
    q75: float | None
    max: float | None
    skewness: float | None
    kurtosis: float | None


class MissingValueInfo(BaseModel):
    column: str
    missing_count: int
    missing_pct: float
    dtype: str


class OutlierInfo(BaseModel):
    column: str
    method: str
    outlier_count: int
    outlier_pct: float
    lower_bound: float | None
    upper_bound: float | None


class CorrelationResult(BaseModel):
    method: str
    matrix: dict[str, dict[str, float]]
    top_pairs: list[dict[str, Any]]


class EDAResult(BaseModel):
    dataset_meta: DatasetMeta
    stats_summary: list[StatsSummary]
    missing_values: list[MissingValueInfo]
    outliers: list[OutlierInfo]
    correlation: CorrelationResult


class ModelResult(BaseModel):
    model_name: str
    task_type: str
    metrics: dict[str, float]
    feature_importance: dict[str, float] | None
    best_params: dict[str, Any] | None
    training_time_sec: float


class ModelComparisonResult(BaseModel):
    task_type: str
    target_column: str
    results: list[ModelResult]
    best_model: str
    leaderboard: list[dict[str, Any]]


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    dataset_id: str
    system_context: str | None = None


class ChatResponse(BaseModel):
    answer: str
    code: str | None = None
    chart_json: str | None = None
    tokens_used: int


class ReportRequest(BaseModel):
    dataset_id: str
    include_eda: bool = True
    include_ml: bool = False
    model_id: str | None = None
    executive_summary: bool = True


class APIResponse(BaseModel):
    success: bool
    data: Any = None
    message: str = ""
