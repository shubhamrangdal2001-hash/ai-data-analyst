"""Machine Learning service: classification, regression, time-series, hyperparameter tuning."""
from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from backend.core.exceptions import InsufficientDataError, ModelTrainingError, TargetColumnError
from backend.models.schemas import ModelComparisonResult, ModelResult
from backend.services.data_service import get_dataset

# In-process model registry
_MODEL_REGISTRY: dict[str, dict[str, Any]] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_compare(
    dataset_id: str,
    target_column: str,
    task_type: str = "auto",
    test_size: float = 0.2,
    tune: bool = False,
    n_trials: int = 20,
) -> ModelComparisonResult:
    df = get_dataset(dataset_id)
    _validate_target(df, target_column)

    X, y, task_type, label_enc = _prepare_data(df, target_column, task_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    logger.info("Training | task={} target={} rows={} features={}", task_type, target_column, len(X), X.shape[1])

    models = _get_model_zoo(task_type)
    results: list[ModelResult] = []

    for name, model in models.items():
        result = _train_single(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            task_type=task_type,
            tune=tune,
            n_trials=n_trials,
        )
        results.append(result)
        _MODEL_REGISTRY[f"{dataset_id}_{name}"] = {
            "model": model,
            "task_type": task_type,
            "target": target_column,
            "features": X.columns.tolist(),
            "label_encoder": label_enc,
        }

    # Sort by primary metric
    sort_key = "roc_auc" if task_type == "classification" else "r2"
    results.sort(key=lambda r: r.metrics.get(sort_key, 0), reverse=True)
    best_model = results[0].model_name

    leaderboard = [
        {"rank": i + 1, "model": r.model_name, **r.metrics, "train_time_s": r.training_time_sec}
        for i, r in enumerate(results)
    ]

    return ModelComparisonResult(
        task_type=task_type,
        target_column=target_column,
        results=results,
        best_model=best_model,
        leaderboard=leaderboard,
    )


def forecast_time_series(
    dataset_id: str,
    date_col: str,
    value_col: str,
    periods: int = 30,
    freq: str = "D",
) -> dict[str, Any]:
    """Simple time-series forecast using Prophet or statsmodels as fallback."""
    df = get_dataset(dataset_id)
    try:
        from prophet import Prophet
        ts = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"}).dropna()
        ts["ds"] = pd.to_datetime(ts["ds"])
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(ts)
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)
        return {
            "dates": forecast["ds"].astype(str).tolist(),
            "yhat": forecast["yhat"].round(2).tolist(),
            "yhat_lower": forecast["yhat_lower"].round(2).tolist(),
            "yhat_upper": forecast["yhat_upper"].round(2).tolist(),
            "historical_dates": ts["ds"].astype(str).tolist(),
            "historical_values": ts["y"].tolist(),
            "method": "prophet",
        }
    except Exception as exc:
        logger.warning("Prophet failed ({}), using ETS fallback", exc)
        return _ets_forecast(df, date_col, value_col, periods, freq)


# ── Private helpers ────────────────────────────────────────────────────────────

def _validate_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise TargetColumnError(f"Target column '{target}' not found in dataset.")
    if df[target].isna().all():
        raise TargetColumnError(f"Target column '{target}' is entirely null.")
    if len(df) < 30:
        raise InsufficientDataError("Need at least 30 rows to train a model.")


def _prepare_data(
    df: pd.DataFrame, target: str, task_type: str
) -> tuple[pd.DataFrame, pd.Series, str, LabelEncoder | None]:
    df = df.copy().dropna(subset=[target])
    y = df[target]
    X = df.drop(columns=[target])

    # Determine task type
    if task_type == "auto":
        unique_ratio = y.nunique() / len(y)
        task_type = "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"

    label_enc = None
    if task_type == "classification" and y.dtype == object:
        label_enc = LabelEncoder()
        y = pd.Series(label_enc.fit_transform(y), name=target)

    # Encode features
    X = X.select_dtypes(exclude=["datetime", "datetimetz"])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols += [c for c in X.columns if c not in cat_cols and pd.api.types.is_string_dtype(X[c])]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))

    return X, y, task_type, label_enc


def _get_model_zoo(task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        return {
            "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=42))]),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
    else:
        return {
            "Ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }


def _train_single(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
    tune: bool,
    n_trials: int,
) -> ModelResult:
    t0 = time.time()
    try:
        if tune and name in ("RandomForest", "GradientBoosting"):
            model = _tune_model(model, X_train, y_train, task_type, n_trials)
        model.fit(X_train, y_train)
        metrics = _compute_metrics(model, X_test, y_test, task_type)
        fi = _get_feature_importance(model, X_train.columns.tolist())
        params = model.get_params() if hasattr(model, "get_params") else {}
    except Exception as exc:
        raise ModelTrainingError(f"Model '{name}' failed: {exc}") from exc

    return ModelResult(
        model_name=name,
        task_type=task_type,
        metrics=metrics,
        feature_importance=fi,
        best_params={k: str(v) for k, v in list(params.items())[:10]},
        training_time_sec=round(time.time() - t0, 3),
    )


def _compute_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series, task_type: str) -> dict[str, float]:
    y_pred = model.predict(X_test)
    if task_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
            else:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")), 4)
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(rmse, 4),
        }
    return metrics


def _get_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float] | None:
    inner = model
    if hasattr(model, "named_steps"):
        for v in model.named_steps.values():
            if hasattr(v, "feature_importances_") or hasattr(v, "coef_"):
                inner = v
                break
    if hasattr(inner, "feature_importances_"):
        fi = dict(zip(feature_names, inner.feature_importances_.tolist()))
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20])
    if hasattr(inner, "coef_"):
        coef = inner.coef_.flatten()[:len(feature_names)]
        fi = dict(zip(feature_names, np.abs(coef).tolist()))
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:20])
    return None


def _tune_model(model: Any, X: pd.DataFrame, y: pd.Series, task_type: str, n_trials: int) -> Any:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
        model.set_params(**params)
        scoring = "roc_auc" if task_type == "classification" else "r2"
        scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    model.set_params(**study.best_params)
    return model


def _ets_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods: int, freq: str) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ts = df[[date_col, value_col]].dropna()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.sort_values(date_col).set_index(date_col)[value_col]
    ts = ts.resample(freq).sum()
    try:
        model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12)
        fit = model.fit()
        fc = fit.forecast(periods)
    except Exception:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        fit = SimpleExpSmoothing(ts).fit()
        fc = fit.forecast(periods)

    future_dates = pd.date_range(ts.index[-1], periods=periods + 1, freq=freq)[1:]
    return {
        "dates": [str(ts.index[-1])] + [str(d) for d in future_dates],
        "yhat": ts.tolist()[-1:] + fc.round(2).tolist(),
        "yhat_lower": None,
        "yhat_upper": None,
        "historical_dates": ts.index.astype(str).tolist(),
        "historical_values": ts.tolist(),
        "method": "ets",
    }
