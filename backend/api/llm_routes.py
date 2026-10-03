"""LLM chat, insights, and report generation endpoints."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from backend.models.schemas import APIResponse, ChatRequest
from backend.services.eda_service import run_eda
from backend.services.llm_service import chat, generate_insights, generate_report_narrative
from backend.services.report_service import generate_pdf_report

router = APIRouter(prefix="/llm", tags=["LLM / AI"])


@router.post("/{dataset_id}/chat", response_model=APIResponse)
async def chat_endpoint(dataset_id: str, req: ChatRequest):
    """Natural language chat with the dataset."""
    result = chat(dataset_id, req.messages, req.system_context)
    return APIResponse(success=True, data=result.model_dump())


@router.get("/{dataset_id}/insights", response_model=APIResponse)
async def get_insights(dataset_id: str):
    """Generate AI-powered business insights."""
    eda = run_eda(dataset_id)
    summary = {
        "missing_values": [m.model_dump() for m in eda.missing_values[:5]],
        "outliers": [o.model_dump() for o in eda.outliers[:5]],
        "correlations": eda.correlation.top_pairs[:5],
        "stats": [s.model_dump() for s in eda.stats_summary[:5]],
    }
    insights = generate_insights(dataset_id, summary)
    return APIResponse(success=True, data=insights)


@router.get("/{dataset_id}/report/pdf")
async def download_pdf_report(dataset_id: str):
    """Generate and download a PDF report."""
    eda = run_eda(dataset_id)
    narrative = generate_report_narrative(dataset_id, eda)
    summary = {
        "missing": [m.model_dump() for m in eda.missing_values[:5]],
        "stats": [s.model_dump() for s in eda.stats_summary[:5]],
    }
    insights = generate_insights(dataset_id, summary)
    pdf_bytes = generate_pdf_report(eda, narrative, insights)
    filename = f"report_{eda.dataset_meta.filename.replace('.', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
