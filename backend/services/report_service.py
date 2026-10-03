"""PDF report generation using ReportLab."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from backend.core.config import settings
from backend.models.schemas import EDAResult


def generate_pdf_report(
    eda_result: EDAResult,
    narrative: str,
    insights: list[dict[str, Any]],
    charts: list[bytes] | None = None,
) -> bytes:
    """Generate a professional PDF report and return raw bytes."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title="AI Data Analyst Report",
        )
        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"))
        h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=4, textColor=colors.HexColor("#16213e"))
        body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
        label_style = ParagraphStyle("Label", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#666666"))

        # ── Cover ─────────────────────────────────────────────────────────────
        story.append(Spacer(1, 1 * cm))
        story.append(Paragraph("AI Data Analyst", title_style))
        story.append(Paragraph("Automated Analytics Report", styles["Heading2"]))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4f46e5")))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(f"Dataset: {eda_result.dataset_meta.filename}", label_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}", label_style))
        story.append(Spacer(1, 0.8 * cm))

        # ── Dataset Overview ──────────────────────────────────────────────────
        story.append(Paragraph("Dataset Overview", h1_style))
        meta = eda_result.dataset_meta
        overview_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{meta.rows:,}"],
            ["Total Columns", str(meta.columns)],
            ["File Size", f"{meta.size_kb:.1f} KB"],
            ["Memory Usage", f"{meta.memory_mb:.2f} MB"],
            ["Numeric Columns", str(len(meta.numeric_columns))],
            ["Categorical Columns", str(len(meta.categorical_columns))],
        ]
        t = Table(overview_data, colWidths=[6 * cm, 6 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f8ff"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.6 * cm))

        # ── Executive Narrative ───────────────────────────────────────────────
        story.append(Paragraph("Executive Summary", h1_style))
        for para in narrative.split("\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para, body_style))
                story.append(Spacer(1, 0.15 * cm))
        story.append(Spacer(1, 0.4 * cm))

        # ── Missing Values ────────────────────────────────────────────────────
        if eda_result.missing_values:
            story.append(Paragraph("Data Quality — Missing Values", h1_style))
            mv_data = [["Column", "Missing Count", "Missing %", "Data Type"]]
            for mv in eda_result.missing_values[:15]:
                mv_data.append([mv.column, str(mv.missing_count), f"{mv.missing_pct:.1f}%", mv.dtype])
            t2 = Table(mv_data, colWidths=[5 * cm, 3.5 * cm, 3 * cm, 3.5 * cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e11d48")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fff1f2"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.6 * cm))

        # ── Business Insights ─────────────────────────────────────────────────
        if insights:
            story.append(Paragraph("Business Insights", h1_style))
            for i, ins in enumerate(insights[:6], 1):
                story.append(Paragraph(f"{i}. {ins.get('title', 'Insight')}", styles["Heading3"]))
                story.append(Paragraph(ins.get("finding", ""), body_style))
                if ins.get("action"):
                    story.append(Paragraph(f"→ Action: {ins['action']}", label_style))
                story.append(Spacer(1, 0.3 * cm))

        # ── Footer ────────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Generated by AI Data Analyst Agent | Confidential", label_style))

        doc.build(story)
        pdf_bytes = buf.getvalue()
        logger.info("PDF report generated | size={} bytes", len(pdf_bytes))
        return pdf_bytes

    except Exception as exc:
        logger.exception("PDF generation failed: {}", exc)
        raise
