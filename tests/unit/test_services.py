"""Unit tests for data loading, EDA, and ML services."""
import io
import pytest
import pandas as pd
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "income": np.random.normal(50000, 15000, n),
        "score": np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "label": np.random.choice([0, 1], n),
    })


@pytest.fixture
def csv_bytes(sample_df):
    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    return buf.getvalue()


# ── Data Service ──────────────────────────────────────────────────────────────

def test_load_csv(csv_bytes):
    from backend.services.data_service import load_dataset, get_dataset
    dataset_id, meta = load_dataset(csv_bytes, "test.csv")
    assert meta.rows == 200
    assert meta.columns == 5
    assert "age" in meta.numeric_columns
    assert "category" in meta.categorical_columns
    df = get_dataset(dataset_id)
    assert len(df) == 200


def test_unsupported_file_type():
    from backend.core.exceptions import UnsupportedFileTypeError
    from backend.services.data_service import load_dataset
    with pytest.raises(UnsupportedFileTypeError):
        load_dataset(b"data", "file.json")


def test_missing_dataset():
    from backend.core.exceptions import ModelNotFoundError
    from backend.services.data_service import get_dataset
    with pytest.raises(ModelNotFoundError):
        get_dataset("nonexistent_id")


# ── EDA Service ───────────────────────────────────────────────────────────────

def test_run_eda(csv_bytes):
    from backend.services.data_service import load_dataset
    from backend.services.eda_service import run_eda
    dataset_id, _ = load_dataset(csv_bytes, "eda_test.csv")
    result = run_eda(dataset_id)
    assert result.dataset_meta.rows == 200
    assert len(result.stats_summary) > 0
    assert len(result.correlation.top_pairs) > 0


def test_missing_value_detection():
    import io
    import numpy as np
    np.random.seed(1)
    df = pd.DataFrame({
        "a": np.random.randn(100),
        "b": [None if i % 5 == 0 else float(i) for i in range(100)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)

    from backend.services.data_service import load_dataset
    from backend.services.eda_service import run_eda
    did, _ = load_dataset(buf.getvalue(), "missing_test.csv")
    result = run_eda(did)
    missing_cols = [m.column for m in result.missing_values]
    assert "b" in missing_cols


# ── ML Service ────────────────────────────────────────────────────────────────

def test_classification(csv_bytes):
    from backend.services.data_service import load_dataset
    from backend.services.ml_service import train_and_compare
    dataset_id, _ = load_dataset(csv_bytes, "ml_test.csv")
    result = train_and_compare(dataset_id, target_column="label", task_type="classification")
    assert result.task_type == "classification"
    assert len(result.results) >= 2
    assert result.best_model in [r.model_name for r in result.results]
    for r in result.results:
        assert "accuracy" in r.metrics or "roc_auc" in r.metrics


def test_regression(csv_bytes):
    from backend.services.data_service import load_dataset
    from backend.services.ml_service import train_and_compare
    dataset_id, _ = load_dataset(csv_bytes, "reg_test.csv")
    result = train_and_compare(dataset_id, target_column="income", task_type="regression")
    assert result.task_type == "regression"
    for r in result.results:
        assert "r2" in r.metrics


def test_invalid_target(csv_bytes):
    from backend.core.exceptions import TargetColumnError
    from backend.services.data_service import load_dataset
    from backend.services.ml_service import train_and_compare
    dataset_id, _ = load_dataset(csv_bytes, "bad_target.csv")
    with pytest.raises(TargetColumnError):
        train_and_compare(dataset_id, target_column="nonexistent_column")
