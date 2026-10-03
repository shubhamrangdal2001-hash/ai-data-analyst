"""Data ingestion, validation and in-memory store."""
from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from backend.core.config import settings
from backend.core.exceptions import (
    DataLoadError,
    DataTooLargeError,
    UnsupportedFileTypeError,
)
from backend.models.schemas import ColumnInfo, DatasetMeta

# In-process store: {dataset_id: {"df": pd.DataFrame, "meta": DatasetMeta, "filename": str}}
_DATASET_STORE: dict[str, dict[str, Any]] = {}

SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx", ".tsv"}
_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


# ── Loading ───────────────────────────────────────────────────────────────────

def _read_delimited(file_bytes: bytes, sep: str) -> pd.DataFrame:
    """Read CSV/TSV bytes, trying common encodings before failing."""
    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), sep=sep, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DataLoadError(
        "Could not decode file. Try saving it as UTF-8.",
        detail=str(last_error) if last_error else None,
    ) from last_error


def load_dataset(file_bytes: bytes, filename: str) -> tuple[str, DatasetMeta]:
    """Parse raw bytes into a DataFrame and register it in the store."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File type '{suffix}' is not supported.",
            detail=f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    size_kb = len(file_bytes) / 1024
    max_kb = settings.max_upload_size_mb * 1024
    if size_kb > max_kb:
        raise DataTooLargeError(
            f"File is {size_kb / 1024:.1f} MB – exceeds the {settings.max_upload_size_mb} MB limit."
        )

    try:
        if suffix == ".csv":
            df = _read_delimited(file_bytes, sep=",")
        elif suffix == ".tsv":
            df = _read_delimited(file_bytes, sep="\t")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Failed to parse '{filename}': {exc}") from exc

    # Infer datetime columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True)
                df[col] = converted
            except Exception:
                pass

    dataset_id = _make_id(file_bytes, filename)
    meta = _build_meta(df, filename, size_kb)
    _DATASET_STORE[dataset_id] = {"df": df, "meta": meta, "filename": filename}
    logger.info("Dataset loaded | id={} rows={} cols={}", dataset_id, df.shape[0], df.shape[1])
    return dataset_id, meta


def get_dataset(dataset_id: str) -> pd.DataFrame:
    entry = _DATASET_STORE.get(dataset_id)
    if entry is None:
        from backend.core.exceptions import ModelNotFoundError
        raise ModelNotFoundError(f"Dataset '{dataset_id}' not found. Please re-upload.")
    return entry["df"]


def get_meta(dataset_id: str) -> DatasetMeta:
    entry = _DATASET_STORE.get(dataset_id)
    if entry is None:
        from backend.core.exceptions import ModelNotFoundError
        raise ModelNotFoundError(f"Dataset '{dataset_id}' not found.")
    return entry["meta"]


def list_datasets() -> list[dict[str, Any]]:
    return [
        {"dataset_id": k, "filename": v["filename"], "rows": v["meta"].rows, "columns": v["meta"].columns}
        for k, v in _DATASET_STORE.items()
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(file_bytes: bytes, filename: str) -> str:
    h = hashlib.md5(file_bytes[:4096]).hexdigest()[:8]
    return f"{Path(filename).stem[:16]}_{h}"


def _build_meta(df: pd.DataFrame, filename: str, size_kb: float) -> DatasetMeta:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols += [c for c in df.columns if c not in cat_cols and pd.api.types.is_string_dtype(df[c])]
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    col_info = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        samples = series.dropna().head(3).tolist()
        col_info.append(
            ColumnInfo(
                name=col,
                dtype=str(series.dtype),
                non_null_count=int(series.notna().sum()),
                null_count=null_count,
                null_pct=round(null_count / max(len(series), 1) * 100, 2),
                unique_count=int(series.nunique()),
                sample_values=[str(s) for s in samples],
            )
        )

    return DatasetMeta(
        filename=filename,
        rows=len(df),
        columns=len(df.columns),
        size_kb=round(size_kb, 2),
        column_info=col_info,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        datetime_columns=dt_cols,
        memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 3),
    )
