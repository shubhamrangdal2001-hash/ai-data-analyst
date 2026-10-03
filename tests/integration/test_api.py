"""Integration tests — FastAPI endpoints via TestClient."""
import io
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def csv_bytes():
    np.random.seed(0)
    n = 150
    df = pd.DataFrame({
        "age":      np.random.randint(18, 70, n),
        "income":   np.random.normal(55000, 12000, n),
        "score":    np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "label":    np.random.choice([0, 1], n),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture(scope="module")
def dataset_id(client, csv_bytes):
    resp = client.post(
        "/api/v1/data/upload",
        files={"file": ("integration.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    return resp.json()["data"]["dataset_id"]


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload(client, csv_bytes):
    r = client.post(
        "/api/v1/data/upload",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["success"] is True
    assert "dataset_id" in d["data"]
    assert d["data"]["meta"]["rows"] == 150


def test_upload_bad_type(client):
    r = client.post(
        "/api/v1/data/upload",
        files={"file": ("bad.json", b"{}", "application/json")},
    )
    assert r.status_code == 415


# ── Dataset listing & sample ──────────────────────────────────────────────────

def test_list_datasets(client, dataset_id):
    r = client.get("/api/v1/data/datasets")
    assert r.status_code == 200
    ids = [d["dataset_id"] for d in r.json()["data"]]
    assert dataset_id in ids


def test_get_sample(client, dataset_id):
    r = client.get(f"/api/v1/data/datasets/{dataset_id}/sample?rows=10")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 10


# ── EDA ───────────────────────────────────────────────────────────────────────

def test_eda(client, dataset_id):
    r = client.get(f"/api/v1/eda/{dataset_id}")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["dataset_meta"]["rows"] == 150
    assert len(data["stats_summary"]) > 0
    assert "matrix" in data["correlation"]


def test_eda_distribution(client, dataset_id):
    r = client.get(f"/api/v1/eda/{dataset_id}/distribution/income")
    assert r.status_code == 200
    assert "values" in r.json()["data"]


def test_eda_categorical(client, dataset_id):
    r = client.get(f"/api/v1/eda/{dataset_id}/categorical/category")
    assert r.status_code == 200
    data = r.json()["data"]
    assert "labels" in data
    assert "values" in data


# ── ML ────────────────────────────────────────────────────────────────────────

def test_ml_classification(client, dataset_id):
    r = client.post(
        f"/api/v1/ml/{dataset_id}/train",
        json={"target_column": "label", "task_type": "classification", "tune": False},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["task_type"] == "classification"
    assert len(data["results"]) >= 2
    assert data["best_model"] in [x["model_name"] for x in data["results"]]


def test_ml_regression(client, dataset_id):
    r = client.post(
        f"/api/v1/ml/{dataset_id}/train",
        json={"target_column": "income", "task_type": "regression", "tune": False},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["task_type"] == "regression"
    for result in data["results"]:
        assert "r2" in result["metrics"]


def test_ml_bad_target(client, dataset_id):
    r = client.post(
        f"/api/v1/ml/{dataset_id}/train",
        json={"target_column": "nonexistent", "task_type": "auto"},
    )
    assert r.status_code == 422


# ── Missing dataset ───────────────────────────────────────────────────────────

def test_eda_missing_dataset(client):
    r = client.get("/api/v1/eda/does_not_exist")
    assert r.status_code == 404


def test_ml_missing_dataset(client):
    r = client.post(
        "/api/v1/ml/does_not_exist/train",
        json={"target_column": "label"},
    )
    assert r.status_code == 404
