"""Automated Exploratory Data Analysis service."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from backend.core.exceptions import AnalysisError, InsufficientDataError
from backend.models.schemas import (
    CorrelationResult,
    EDAResult,
    MissingValueInfo,
    OutlierInfo,
    StatsSummary,
)
from backend.services.data_service import get_dataset, get_meta


def run_eda(dataset_id: str) -> EDAResult:
    """Run full EDA pipeline on a registered dataset."""
    df = get_dataset(dataset_id)
    meta = get_meta(dataset_id)
    logger.info("Running EDA | dataset_id={} shape={}", dataset_id, df.shape)

    try:
        stats_summary = _compute_stats(df, meta.numeric_columns)
        missing = _analyse_missing(df)
        outliers = _detect_outliers(df, meta.numeric_columns)
        correlation = _compute_correlation(df, meta.numeric_columns)
    except Exception as exc:
        raise AnalysisError(f"EDA failed: {exc}") from exc

    return EDAResult(
        dataset_meta=meta,
        stats_summary=stats_summary,
        missing_values=missing,
        outliers=outliers,
        correlation=correlation,
    )


# ── Statistical Summary ───────────────────────────────────────────────────────

def _compute_stats(df: pd.DataFrame, numeric_cols: list[str]) -> list[StatsSummary]:
    results = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        desc = s.describe()
        results.append(
            StatsSummary(
                column=col,
                count=float(desc["count"]),
                mean=float(desc["mean"]),
                std=float(desc["std"]),
                min=float(desc["min"]),
                q25=float(desc["25%"]),
                median=float(desc["50%"]),
                q75=float(desc["75%"]),
                max=float(desc["max"]),
                skewness=float(s.skew()),
                kurtosis=float(s.kurtosis()),
            )
        )
    return results


# ── Missing Value Analysis ────────────────────────────────────────────────────

def _analyse_missing(df: pd.DataFrame) -> list[MissingValueInfo]:
    results = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue
        results.append(
            MissingValueInfo(
                column=col,
                missing_count=missing_count,
                missing_pct=round(missing_count / len(df) * 100, 2),
                dtype=str(df[col].dtype),
            )
        )
    return sorted(results, key=lambda x: x.missing_pct, reverse=True)


# ── Outlier Detection ─────────────────────────────────────────────────────────

def _detect_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> list[OutlierInfo]:
    results = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue

        # IQR method
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (s < lower) | (s > upper)
        count = int(outlier_mask.sum())
        results.append(
            OutlierInfo(
                column=col,
                method="IQR",
                outlier_count=count,
                outlier_pct=round(count / len(s) * 100, 2),
                lower_bound=round(float(lower), 4),
                upper_bound=round(float(upper), 4),
            )
        )

        # Z-score method
        z_scores = np.abs(stats.zscore(s))
        z_outliers = int((z_scores > 3).sum())
        results.append(
            OutlierInfo(
                column=col,
                method="Z-Score",
                outlier_count=z_outliers,
                outlier_pct=round(z_outliers / len(s) * 100, 2),
                lower_bound=None,
                upper_bound=None,
            )
        )
    return results


# ── Correlation Analysis ──────────────────────────────────────────────────────

def _compute_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> CorrelationResult:
    if len(numeric_cols) < 2:
        return CorrelationResult(method="pearson", matrix={}, top_pairs=[])

    corr_df = df[numeric_cols].corr(method="pearson").round(4)
    matrix = {col: corr_df[col].to_dict() for col in corr_df.columns}

    # Extract top absolute correlations (excluding self-correlations)
    pairs = []
    cols = list(corr_df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = float(corr_df.iloc[i, j])
            if not np.isnan(val):
                pairs.append({"col_a": cols[i], "col_b": cols[j], "correlation": round(val, 4)})

    top_pairs = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:10]

    return CorrelationResult(method="pearson", matrix=matrix, top_pairs=top_pairs)


# ── Visualisation Data Helpers ────────────────────────────────────────────────

def get_distribution_data(dataset_id: str, column: str) -> dict[str, Any]:
    df = get_dataset(dataset_id)
    s = df[column].dropna()
    return {
        "values": s.tolist(),
        "column": column,
        "dtype": str(s.dtype),
        "stats": {
            "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) else None,
            "median": float(s.median()) if pd.api.types.is_numeric_dtype(s) else None,
            "std": float(s.std()) if pd.api.types.is_numeric_dtype(s) else None,
        },
    }


def get_categorical_counts(dataset_id: str, column: str, top_n: int = 15) -> dict[str, Any]:
    df = get_dataset(dataset_id)
    counts = df[column].value_counts().head(top_n)
    return {"labels": counts.index.tolist(), "values": counts.values.tolist(), "column": column}


def get_time_series_data(dataset_id: str, date_col: str, value_col: str, freq: str = "M") -> dict[str, Any]:
    df = get_dataset(dataset_id)
    ts = df.set_index(date_col)[value_col].resample(freq).sum().reset_index()
    return {
        "dates": ts[date_col].astype(str).tolist(),
        "values": ts[value_col].tolist(),
        "date_col": date_col,
        "value_col": value_col,
    }
