"""HTTP client for the FastAPI backend."""
from __future__ import annotations

import os
from typing import Any

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BASE = f"{BACKEND_URL}/api/v1"
TIMEOUT = 120.0


def _get(path: str, **params) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.get(f"{BASE}{path}", params=params)
        r.raise_for_status()
        return r.json()


def _raise_api_error(response: httpx.Response) -> None:
    try:
        body = response.json()
        msg = body.get("message") or body.get("detail")
        if msg:
            raise httpx.HTTPStatusError(msg, request=response.request, response=response)
    except (ValueError, TypeError):
        pass
    response.raise_for_status()


def _post(path: str, json: dict | None = None, files=None) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as c:
        if files:
            r = c.post(f"{BASE}{path}", files=files)
        else:
            r = c.post(f"{BASE}{path}", json=json)
        if not r.is_success:
            _raise_api_error(r)
        return r.json()


def _get_bytes(path: str) -> bytes:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.get(f"{BASE}{path}")
        r.raise_for_status()
        return r.content


# ── Data ─────────────────────────────────────────────────────────────────────

def _mime_for(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "csv": "text/csv",
        "tsv": "text/tab-separated-values",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
    }.get(ext, "application/octet-stream")


def upload_file(file_bytes: bytes, filename: str) -> dict[str, Any]:
    return _post("/data/upload", files={"file": (filename, file_bytes, _mime_for(filename))})


def get_datasets() -> list[dict]:
    return _get("/data/datasets").get("data", [])


def get_dataset_meta(dataset_id: str) -> dict:
    return _get(f"/data/datasets/{dataset_id}/meta").get("data", {})


def get_sample(dataset_id: str, rows: int = 20) -> list[dict]:
    return _get(f"/data/datasets/{dataset_id}/sample", rows=rows).get("data", [])


# ── EDA ───────────────────────────────────────────────────────────────────────

def run_eda(dataset_id: str) -> dict:
    return _get(f"/eda/{dataset_id}").get("data", {})


def get_distribution(dataset_id: str, column: str) -> dict:
    return _get(f"/eda/{dataset_id}/distribution/{column}").get("data", {})


def get_categorical(dataset_id: str, column: str, top_n: int = 15) -> dict:
    return _get(f"/eda/{dataset_id}/categorical/{column}", top_n=top_n).get("data", {})


def get_timeseries(dataset_id: str, date_col: str, value_col: str, freq: str = "M") -> dict:
    return _get(f"/eda/{dataset_id}/timeseries", date_col=date_col, value_col=value_col, freq=freq).get("data", {})


# ── ML ────────────────────────────────────────────────────────────────────────

def train_models(dataset_id: str, target: str, task_type: str = "auto", tune: bool = False) -> dict:
    return _post(f"/ml/{dataset_id}/train", json={"target_column": target, "task_type": task_type, "tune": tune}).get("data", {})


def forecast(dataset_id: str, date_col: str, value_col: str, periods: int = 30, freq: str = "D") -> dict:
    return _post(f"/ml/{dataset_id}/forecast", json={"date_col": date_col, "value_col": value_col, "periods": periods, "freq": freq}).get("data", {})


# ── LLM ───────────────────────────────────────────────────────────────────────

def chat(dataset_id: str, messages: list[dict]) -> dict:
    return _post(f"/llm/{dataset_id}/chat", json={"messages": messages, "dataset_id": dataset_id}).get("data", {})


def get_insights(dataset_id: str) -> list[dict]:
    return _get(f"/llm/{dataset_id}/insights").get("data", [])


def download_report(dataset_id: str) -> bytes:
    return _get_bytes(f"/llm/{dataset_id}/report/pdf")
