"""LLM service: Groq-powered natural-language querying, code generation, insights."""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.core.config import settings
from backend.core.exceptions import LLMError, LLMRateLimitError
from backend.models.schemas import ChatMessage, ChatResponse
from backend.services.data_service import get_dataset, get_meta


# ── Groq client (OpenAI-compatible SDK) ──────────────────────────────────────

def _get_groq_client():
    """
    Groq uses the OpenAI SDK pointed at their base URL.
    Install: pip install groq  OR  pip install openai
    The `groq` package is a thin wrapper; both work identically.
    """
    try:
        from groq import Groq
        return Groq(api_key=settings.groq_api_key), "groq"
    except ImportError:
        # Fallback: use openai SDK with Groq base URL
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        return client, "openai_compat"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_llm(system: str, messages: list[dict], max_tokens: int | None = None) -> str:
    """
    Unified Groq LLM call.
    Groq API is OpenAI-compatible: chat.completions.create().
    """
    max_tokens = max_tokens or settings.llm_max_tokens
    try:
        client, sdk = _get_groq_client()
        all_messages = [{"role": "system", "content": system}] + messages

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=settings.llm_temperature,
        )
        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        if usage:
            logger.debug(
                "Groq | model={} prompt_tokens={} completion_tokens={} total={}",
                settings.llm_model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
        return text
    except Exception as exc:
        err_str = str(exc).lower()
        if "rate" in err_str or "429" in err_str or "rate_limit" in err_str:
            raise LLMRateLimitError(
                "Groq rate limit reached. Please wait a moment and retry."
            )
        if "authentication" in err_str or "401" in err_str:
            raise LLMError(
                "Groq authentication failed. Check your GROQ_API_KEY in .env."
            )
        raise LLMError(f"Groq LLM call failed: {exc}") from exc


# ── Dataset context builder ───────────────────────────────────────────────────

def _build_dataset_context(dataset_id: str) -> str:
    meta = get_meta(dataset_id)
    df = get_dataset(dataset_id)
    col_details = "\n".join(
        f"  - {c.name} ({c.dtype}): {c.non_null_count} non-null, "
        f"{c.null_count} missing, {c.unique_count} unique. "
        f"Sample: {c.sample_values}"
        for c in meta.column_info
    )
    sample_rows = df.head(3).to_csv(index=False)
    return f"""
Dataset: {meta.filename}
Shape: {meta.rows} rows × {meta.columns} columns
Numeric columns: {meta.numeric_columns}
Categorical columns: {meta.categorical_columns}
DateTime columns: {meta.datetime_columns}

Column Details:
{col_details}

First 3 rows (CSV):
{sample_rows}
""".strip()


# ── Chat / Q&A ────────────────────────────────────────────────────────────────

SYSTEM_ANALYST = """You are an expert Senior Data Analyst and Data Scientist.
You have access to the user's dataset described below.
When asked a question:
1. Provide a clear, concise, data-driven answer.
2. If useful, generate executable Python/pandas code in a ```python block.
3. Always mention key numbers or insights from the dataset.
4. If you generate a Plotly chart, include the fig.to_json() call inside a ```plotly_json block.
5. Structure long answers with clear headers.
6. Be honest about uncertainty; never hallucinate numbers.

DATASET CONTEXT:
{context}
"""


def chat(
    dataset_id: str,
    messages: list[ChatMessage],
    system_context: str | None = None,
) -> ChatResponse:
    ctx = _build_dataset_context(dataset_id)
    system = SYSTEM_ANALYST.format(context=ctx)
    if system_context:
        system += f"\n\nAdditional context:\n{system_context}"

    llm_messages = [{"role": m.role, "content": m.content} for m in messages]
    answer = _call_llm(system, llm_messages)

    code = _extract_code_block(answer, "python")
    chart_json = _extract_code_block(answer, "plotly_json")

    exec_result = None
    if code:
        exec_result = _safe_exec_code(code, dataset_id)

    return ChatResponse(
        answer=answer,
        code=code,
        chart_json=exec_result if exec_result else chart_json,
        tokens_used=0,
    )


def _extract_code_block(text: str, lang: str) -> str | None:
    pattern = rf"```{lang}\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _safe_exec_code(code: str, dataset_id: str) -> str | None:
    """Execute generated pandas code and capture Plotly chart JSON."""
    try:
        df = get_dataset(dataset_id)
        local_ns: dict[str, Any] = {"df": df, "pd": pd}
        exec(code, local_ns)  # noqa: S102
        fig = local_ns.get("fig")
        if fig is not None:
            return fig.to_json()
    except Exception as exc:
        logger.warning("Generated code execution failed: {}", exc)
    return None


# ── Business Insight Generation ───────────────────────────────────────────────

INSIGHT_PROMPT = """You are a senior business intelligence analyst.
Analyse the following dataset statistics and generate 5-7 actionable business insights.

For each insight:
- Give it a clear title
- Explain the finding in 2-3 sentences
- State the business implication
- Suggest a recommended action

Format your response as a JSON array with keys: title, finding, implication, action.

Dataset statistics:
{stats}

Return ONLY valid JSON array, no markdown fences, no preamble.
"""


def generate_insights(
    dataset_id: str, eda_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    stats_str = json.dumps(eda_summary, indent=2, default=str)[:4000]
    prompt = INSIGHT_PROMPT.format(stats=stats_str)
    raw = _call_llm(
        "You are a business analyst. Return only a valid JSON array.",
        [{"role": "user", "content": prompt}],
    )
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return [{"title": "Insights", "finding": raw, "implication": "", "action": ""}]


# ── Executive Report Narrative ────────────────────────────────────────────────

REPORT_PROMPT = """Write a professional executive data analysis report.

Dataset: {filename}
Rows: {rows}, Columns: {columns}

Key findings:
{findings}

Write the report with:
1. Executive Summary (2-3 sentences)
2. Dataset Overview
3. Key Findings (bullet points)
4. Data Quality Notes
5. Recommendations
6. Next Steps

Use professional business language. Be concise but comprehensive. Max 600 words.
"""


def generate_report_narrative(dataset_id: str, eda_result: Any) -> str:
    meta = eda_result.dataset_meta
    findings = {
        "missing_columns": [m.column for m in eda_result.missing_values[:5]],
        "top_outlier_columns": [
            o.column for o in eda_result.outliers[:5] if o.method == "IQR"
        ],
        "top_correlations": eda_result.correlation.top_pairs[:3],
        "numeric_summary": [
            {"col": s.column, "mean": s.mean, "std": s.std}
            for s in eda_result.stats_summary[:5]
        ],
    }
    prompt = REPORT_PROMPT.format(
        filename=meta.filename,
        rows=meta.rows,
        columns=meta.columns,
        findings=json.dumps(findings, indent=2, default=str),
    )
    return _call_llm(
        "You are a professional data analyst writing an executive report.",
        [{"role": "user", "content": prompt}],
    )
