"""Data upload and management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from backend.models.schemas import APIResponse, DatasetMeta
from backend.services.data_service import list_datasets, load_dataset

router = APIRouter(prefix="/data", tags=["Data Management"])


@router.post("/upload", response_model=APIResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV or Excel file and return a dataset_id."""
    raw = await file.read()
    dataset_id, meta = load_dataset(raw, file.filename or "upload")
    return APIResponse(
        success=True,
        data={"dataset_id": dataset_id, "meta": meta.model_dump()},
        message=f"Loaded {meta.rows:,} rows × {meta.columns} columns.",
    )


@router.get("/datasets", response_model=APIResponse)
async def list_all_datasets():
    return APIResponse(success=True, data=list_datasets())


@router.get("/datasets/{dataset_id}/meta", response_model=APIResponse)
async def get_dataset_meta(dataset_id: str):
    from backend.services.data_service import get_meta
    meta = get_meta(dataset_id)
    return APIResponse(success=True, data=meta.model_dump())


@router.get("/datasets/{dataset_id}/sample", response_model=APIResponse)
async def get_sample(dataset_id: str, rows: int = 20):
    from backend.services.data_service import get_dataset
    df = get_dataset(dataset_id)
    return APIResponse(success=True, data=df.head(rows).to_dict(orient="records"))
