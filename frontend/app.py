"""
AI Data Analyst Agent — Streamlit Frontend
Production-grade dashboard with upload, EDA, ML, Chat, and Reporting.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(79,70,229,0.2) !important;
    border: 1px solid rgba(79,70,229,0.5) !important;
    color: #a5b4fc !important;
    width: 100%;
    border-radius: 8px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e1e3a 0%, #2d2d5e 100%);
    border: 1px solid rgba(79,70,229,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #818cf8; }
.metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white; padding: 0.8rem 1.2rem; border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0; max-width: 75%; margin-left: auto;
}
.chat-bot {
    background: #1e293b; color: #e2e8f0; padding: 0.8rem 1.2rem;
    border-radius: 18px 18px 18px 4px; margin: 0.4rem 0; max-width: 75%;
    border-left: 3px solid #4f46e5;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-size: 0.9rem; font-weight: 600; padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] { color: #4f46e5 !important; }

/* Section headers */
.section-header {
    font-size: 1.4rem; font-weight: 700; color: #1e293b;
    border-left: 4px solid #4f46e5; padding-left: 0.8rem; margin: 1.5rem 0 1rem;
}

/* Insight card */
.insight-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.2rem; margin: 0.6rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-top: 3px solid #4f46e5;
}
.insight-title { font-weight: 700; color: #1e293b; font-size: 1rem; margin-bottom: 0.4rem; }
.insight-action { color: #059669; font-size: 0.85rem; margin-top: 0.4rem; }

/* Status badges */
.badge-success { background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.badge-warning { background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.badge-danger  { background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }

/* Main header */
.main-title {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Import API client and charts ──────────────────────────────────────────────
try:
    from frontend.utils.api_client import (
        chat, download_report, forecast, get_categorical, get_distribution,
        get_insights, get_sample, get_timeseries, run_eda, train_models, upload_file,
    )
    from frontend.utils.charts import (
        fig_bar, fig_correlation_heatmap, fig_distribution, fig_feature_importance,
        fig_forecast, fig_missing_values, fig_model_comparison, fig_scatter,
    )
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


# ── Session State ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "dataset_id": None,
        "dataset_meta": None,
        "eda_result": None,
        "ml_result": None,
        "chat_history": [],
        "insights": None,
        "active_tab": "Upload",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:1rem 0;">'
                '<div style="font-size:2.5rem">🧠</div>'
                '<div style="font-size:1.2rem;font-weight:800;color:#818cf8">AI Data Analyst</div>'
                '<div style="font-size:0.7rem;color:#475569;margin-top:2px">v1.0 · Production</div>'
                '</div>', unsafe_allow_html=True)
    st.divider()

    nav = st.radio(
        "Navigation",
        ["📤 Upload", "🔍 EDA", "🤖 Machine Learning", "💬 AI Chat", "💡 Insights", "📄 Report"],
        label_visibility="collapsed",
    )

    st.divider()

    if st.session_state.dataset_id:
        meta = st.session_state.dataset_meta or {}
        st.markdown("**Active Dataset**")
        st.caption(f"📁 {meta.get('filename', 'Unknown')}")
        st.caption(f"📊 {meta.get('rows', 0):,} rows × {meta.get('columns', 0)} cols")
        st.caption(f"💾 {meta.get('size_kb', 0):.1f} KB")
    else:
        st.info("No dataset loaded", icon="📂")

    st.divider()
    st.caption("Powered by Groq LLM (LLaMA 3.3 · Mixtral)")
    st.caption("FastAPI · Streamlit · Plotly · Groq")


# ── Main content ──────────────────────────────────────────────────────────────
section = nav.split(" ", 1)[1]  # strip emoji

st.markdown(f'<div class="main-title">AI Data Analyst Agent</div>', unsafe_allow_html=True)
st.markdown(f'<div style="color:#64748b;margin-bottom:1.5rem">Automated analytics, ML, and AI-powered insights</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if section == "Upload":
    st.markdown('<div class="section-header">📤 Upload Dataset</div>', unsafe_allow_html=True)

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Drop a CSV or Excel file here",
            type=["csv", "xlsx", "xls", "tsv"],
            help="Max 100 MB",
        )

        if uploaded:
            with st.spinner("⚙️ Loading and profiling dataset…"):
                try:
                    resp = upload_file(uploaded.read(), uploaded.name)
                    if resp.get("success"):
                        d = resp["data"]
                        st.session_state.dataset_id = d["dataset_id"]
                        st.session_state.dataset_meta = d["meta"]
                        st.session_state.eda_result = None
                        st.session_state.ml_result = None
                        st.session_state.insights = None
                        st.success(f"✅ {resp['message']}")
                    else:
                        st.error(f"Upload failed: {resp.get('message')}")
                except Exception as e:
                    st.error(f"❌ {e}")

    with col_info:
        st.markdown("""
        **Supported formats**
        - CSV (`.csv`)
        - Excel (`.xlsx`, `.xls`)
        - Tab-separated (`.tsv`)

        **Max size:** 100 MB
        **Auto-detected:**
        - Data types
        - DateTime columns
        - Missing values
        """)

    # Dataset preview
    if st.session_state.dataset_id:
        st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
        meta = st.session_state.dataset_meta

        m1, m2, m3, m4 = st.columns(4)
        for col_obj, label, val in zip(
            [m1, m2, m3, m4],
            ["Rows", "Columns", "Size (KB)", "Memory (MB)"],
            [f"{meta['rows']:,}", str(meta['columns']), f"{meta['size_kb']:.1f}", f"{meta['memory_mb']:.2f}"],
        ):
            with col_obj:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                            f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        try:
            sample = get_sample(st.session_state.dataset_id, rows=50)
            st.dataframe(pd.DataFrame(sample), use_container_width=True, height=300)
        except Exception as e:
            st.warning(f"Could not load preview: {e}")

        # Column info
        with st.expander("📋 Column Details"):
            col_df = pd.DataFrame(meta.get("column_info", []))
            if not col_df.empty:
                st.dataframe(col_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif section == "EDA":
    if not st.session_state.dataset_id:
        st.warning("Please upload a dataset first.")
        st.stop()

    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if st.button("▶️ Run Full EDA", type="primary"):
        with st.spinner("🔬 Analysing dataset…"):
            try:
                st.session_state.eda_result = run_eda(st.session_state.dataset_id)
                st.success("✅ EDA complete!")
            except Exception as e:
                st.error(f"EDA failed: {e}")

    eda = st.session_state.eda_result
    if not eda:
        st.info("Click **Run Full EDA** to start analysis.", icon="ℹ️")
        st.stop()

    # Tabs
    t_stats, t_missing, t_outliers, t_corr, t_dist, t_cat = st.tabs([
        "📊 Statistics", "⚠️ Missing Values", "🎯 Outliers",
        "🔗 Correlations", "📈 Distributions", "🏷️ Categorical"
    ])

    with t_stats:
        st.markdown("#### Statistical Summary")
        stats_df = pd.DataFrame([s for s in eda.get("stats_summary", [])])
        if not stats_df.empty:
            st.dataframe(stats_df.round(4), use_container_width=True)

    with t_missing:
        missing = eda.get("missing_values", [])
        if missing:
            fig = fig_missing_values(missing)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame(missing), use_container_width=True)
        else:
            st.success("🎉 No missing values found!", icon="✅")

    with t_outliers:
        outliers = [o for o in eda.get("outliers", []) if o["method"] == "IQR"]
        if outliers:
            st.dataframe(pd.DataFrame(outliers), use_container_width=True)
        else:
            st.success("No outliers detected via IQR method.")

    with t_corr:
        matrix = eda.get("correlation", {}).get("matrix", {})
        if matrix:
            fig = fig_correlation_heatmap(matrix)
            st.plotly_chart(fig, use_container_width=True)
        top_pairs = eda.get("correlation", {}).get("top_pairs", [])
        if top_pairs:
            st.markdown("**Top Correlations**")
            st.dataframe(pd.DataFrame(top_pairs), use_container_width=True)

    with t_dist:
        meta = st.session_state.dataset_meta
        num_cols = meta.get("numeric_columns", [])
        if num_cols:
            sel = st.selectbox("Select column", num_cols)
            if sel:
                try:
                    dist_data = get_distribution(st.session_state.dataset_id, sel)
                    st.plotly_chart(fig_distribution(dist_data), use_container_width=True)
                except Exception as e:
                    st.error(str(e))

    with t_cat:
        meta = st.session_state.dataset_meta
        cat_cols = meta.get("categorical_columns", [])
        if cat_cols:
            sel = st.selectbox("Select column", cat_cols, key="cat_sel")
            if sel:
                try:
                    cat_data = get_categorical(st.session_state.dataset_id, sel)
                    st.plotly_chart(fig_bar(cat_data), use_container_width=True)
                except Exception as e:
                    st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Machine Learning":
    if not st.session_state.dataset_id:
        st.warning("Please upload a dataset first.")
        st.stop()

    st.markdown('<div class="section-header">🤖 Machine Learning</div>', unsafe_allow_html=True)
    meta = st.session_state.dataset_meta

    ml_tab, forecast_tab = st.tabs(["📊 Classification / Regression", "📈 Time-Series Forecast"])

    with ml_tab:
        all_cols = [c["name"] for c in meta.get("column_info", [])]
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            target = st.selectbox("🎯 Target column", all_cols)
        with c2:
            task = st.selectbox("Task type", ["auto", "classification", "regression"])
        with c3:
            tune = st.checkbox("Hyperparameter tuning", value=False)

        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("🏋️ Training models… this may take a minute"):
                try:
                    result = train_models(st.session_state.dataset_id, target, task, tune)
                    st.session_state.ml_result = result
                    st.success(f"✅ Best model: **{result.get('best_model')}**")
                except Exception as e:
                    st.error(f"Training failed: {e}")

        ml = st.session_state.ml_result
        if ml:
            st.markdown(f"**Task:** `{ml.get('task_type')}` | **Target:** `{ml.get('target_column')}`")
            st.markdown(f"🏆 **Best Model:** `{ml.get('best_model')}`")

            lb = ml.get("leaderboard", [])
            if lb:
                st.plotly_chart(fig_model_comparison(lb, ml.get("task_type", "")), use_container_width=True)
                st.dataframe(pd.DataFrame(lb), use_container_width=True)

            # Feature importance for best model
            for r in ml.get("results", []):
                if r["model_name"] == ml.get("best_model") and r.get("feature_importance"):
                    st.plotly_chart(fig_feature_importance(r["feature_importance"], f"Feature Importance — {r['model_name']}"), use_container_width=True)
                    break

    with forecast_tab:
        dt_cols = meta.get("datetime_columns", [])
        num_cols = meta.get("numeric_columns", [])
        if not dt_cols:
            st.warning("No datetime columns detected in your dataset.")
        else:
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                date_col = st.selectbox("Date column", dt_cols)
            with fc2:
                value_col = st.selectbox("Value column", num_cols)
            with fc3:
                periods = st.number_input("Forecast periods", 7, 365, 30)
            with fc4:
                freq = st.selectbox("Frequency", ["D", "W", "M", "Q"])

            if st.button("📈 Generate Forecast", type="primary"):
                with st.spinner("🔮 Forecasting…"):
                    try:
                        fc_data = forecast(st.session_state.dataset_id, date_col, value_col, periods, freq)
                        st.plotly_chart(fig_forecast(fc_data), use_container_width=True)
                        st.info(f"Method used: **{fc_data.get('method', 'N/A')}**")
                    except Exception as e:
                        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# AI CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif section == "AI Chat":
    if not st.session_state.dataset_id:
        st.warning("Please upload a dataset first.")
        st.stop()

    st.markdown('<div class="section-header">💬 Chat with Your Data</div>', unsafe_allow_html=True)
    st.caption("Ask questions in plain English. The AI can query, analyse, and visualise your data.")

    # Quick prompts
    st.markdown("**Quick prompts:**")
    qcols = st.columns(4)
    quick_prompts = [
        "What are the key statistics?",
        "Show distribution of numeric columns",
        "Which columns have most missing data?",
        "What are the top correlations?",
    ]
    for i, (col_obj, prompt) in enumerate(zip(qcols, quick_prompts)):
        with col_obj:
            if st.button(prompt, key=f"qp_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})

    st.divider()

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                st.markdown(f'<div class="chat-bot">🧠 {content}</div>', unsafe_allow_html=True)
                if msg.get("chart_json"):
                    try:
                        fig = go.Figure(json.loads(msg["chart_json"]))
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

    # Input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Your question", placeholder="e.g. What is the average revenue by region?", height=80)
        submitted = st.form_submit_button("Send 🚀", type="primary")

    if submitted and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("🤔 Thinking…"):
            try:
                result = chat(st.session_state.dataset_id, st.session_state.chat_history)
                bot_msg = {"role": "assistant", "content": result.get("answer", ""), "chart_json": result.get("chart_json")}
                st.session_state.chat_history.append(bot_msg)
            except Exception as e:
                st.error(f"Chat error: {e}")
        st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Insights":
    if not st.session_state.dataset_id:
        st.warning("Please upload a dataset first.")
        st.stop()

    st.markdown('<div class="section-header">💡 AI Business Insights</div>', unsafe_allow_html=True)
    st.caption("Automatically generated business insights powered by Claude AI.")

    if st.button("✨ Generate Insights", type="primary"):
        with st.spinner("🧠 Analysing patterns and generating insights…"):
            try:
                st.session_state.insights = get_insights(st.session_state.dataset_id)
            except Exception as e:
                st.error(f"Insight generation failed: {e}")

    insights = st.session_state.insights
    if insights:
        st.markdown(f"**{len(insights)} insights generated**")
        for ins in insights:
            title = ins.get("title", "Insight")
            finding = ins.get("finding", "")
            implication = ins.get("implication", "")
            action = ins.get("action", "")
            st.markdown(
                f'<div class="insight-card">'
                f'<div class="insight-title">💡 {title}</div>'
                f'<div style="color:#475569;font-size:0.9rem;margin:0.3rem 0">{finding}</div>'
                f'<div style="color:#6366f1;font-size:0.85rem"><b>Implication:</b> {implication}</div>'
                f'<div class="insight-action">→ <b>Action:</b> {action}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    elif st.session_state.insights is not None:
        st.info("No insights could be generated. Try running EDA first.")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Report":
    if not st.session_state.dataset_id:
        st.warning("Please upload a dataset first.")
        st.stop()

    st.markdown('<div class="section-header">📄 Generate Report</div>', unsafe_allow_html=True)
    st.markdown("Generate a professional PDF report with EDA results, business insights, and executive summary.")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        inc_eda = st.checkbox("Include EDA summary", value=True)
        inc_insights = st.checkbox("Include AI insights", value=True)
    with col_r2:
        exec_summary = st.checkbox("Executive summary", value=True)
        inc_ml = st.checkbox("Include ML results", value=False)

    if st.button("📥 Generate & Download PDF Report", type="primary"):
        with st.spinner("📝 Generating professional PDF report… (this may take 30–60 seconds)"):
            try:
                pdf_bytes = download_report(st.session_state.dataset_id)
                meta = st.session_state.dataset_meta
                filename = f"report_{meta.get('filename', 'data').replace('.', '_')}.pdf"
                st.download_button(
                    label="💾 Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    type="primary",
                )
                st.success("✅ Report generated successfully!")
                st.info(f"Report size: {len(pdf_bytes) / 1024:.1f} KB")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.caption("Make sure EDA has been run and LLM API key is configured.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#94a3b8;font-size:0.78rem">'
    'AI Data Analyst Agent · Built with FastAPI, Streamlit & Groq LLM · '
    f'Dataset: {"✅ Loaded" if st.session_state.dataset_id else "❌ None"}'
    '</div>',
    unsafe_allow_html=True,
)
