# 🧠 AI Data Analyst Agent (Groq Edition)

> Production-grade AI-powered data analysis platform combining automated EDA,
> machine learning, and **Groq-accelerated LLM** for natural language querying,
> insight generation, and PDF reporting.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?logo=lightning)](https://groq.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![Azure](https://img.shields.io/badge/Azure-App_Service-0089D6?logo=microsoft-azure)](https://azure.microsoft.com)

---

## ⚡ Why Groq?

| Feature | Groq | Others |
|---|---|---|
| Inference speed | **~300 tokens/s** (LPU™) | ~30-80 tokens/s |
| Free tier | ✅ Generous limits | Limited / paid |
| OpenAI-compatible API | ✅ Drop-in | Varies |
| Models | LLaMA 3.3 70B, Mixtral 8x7B, Gemma 2 9B | GPT-4o, Claude, etc. |
| Latency | **<1 s first token** | 2-5 s |

Get a free API key at **https://console.groq.com/keys**

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND  :8501                      │
│  Upload │ EDA Dashboard │ ML Training │ AI Chat │ Reports       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / REST
┌───────────────────────────▼─────────────────────────────────────┐
│                   FASTAPI BACKEND  :8000                         │
│  /api/v1/data  │  /api/v1/eda  │  /api/v1/ml  │  /api/v1/llm  │
└──────┬──────────────┬───────────────┬───────────────┬───────────┘
       │              │               │               │
  Data Service   EDA Service    ML Service      LLM Service
  (Upload/Store) (Stats/Outliers)(Train/Forecast)(Groq API ⚡)
                                                      │
                                          ┌───────────▼───────────┐
                                          │   api.groq.com/openai │
                                          │  llama-3.3-70b        │
                                          │  mixtral-8x7b-32768   │
                                          │  llama-3.1-8b-instant │
                                          └───────────────────────┘
```

---

## 🗂️ Project Structure

```
ai-data-analyst/
├── backend/
│   ├── api/
│   │   ├── data_routes.py      Upload & dataset management
│   │   ├── eda_routes.py       EDA endpoints
│   │   ├── ml_routes.py        ML training & forecasting
│   │   └── llm_routes.py       Chat, insights, PDF report
│   ├── core/
│   │   ├── config.py           Pydantic settings (Groq keys)
│   │   ├── logging.py          Loguru structured logging
│   │   └── exceptions.py       Domain exceptions + handlers
│   ├── models/schemas.py       Pydantic request/response models
│   └── services/
│       ├── data_service.py     Loading, validation, in-memory store
│       ├── eda_service.py      Stats, missing values, outliers, correlation
│       ├── ml_service.py       Classification, regression, time-series
│       ├── llm_service.py      ⚡ Groq chat + insights + report narrative
│       └── report_service.py   PDF generation (ReportLab)
│
├── frontend/
│   ├── app.py                  Streamlit dashboard (5 tabs)
│   └── utils/
│       ├── api_client.py       HTTP client for backend
│       └── charts.py           Plotly chart builders
│
├── data/sample/                1,000-row sales_data.csv + .xlsx
├── deployment/
│   ├── docker/entrypoint.sh    Start API + frontend
│   ├── azure/deploy.sh         Full Azure provisioning script
│   └── github/ci-cd.yml        GitHub Actions pipeline
├── .github/workflows/ci-cd.yml
├── tests/unit/                 8 unit tests
├── tests/integration/          13 API integration tests
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## ✨ Features

### 📊 Data Processing
- Upload CSV, Excel (.xlsx/.xls), TSV — up to 100 MB
- Auto type inference (datetime detection, numeric/categorical)
- Interactive preview with column profiling

### 🔍 Automated EDA
- Descriptive stats (mean, std, skewness, kurtosis)
- Missing value analysis with % breakdown
- Outlier detection: IQR + Z-Score
- Pearson correlation matrix + top pairs
- Distribution, bar, and time-series charts

### 🤖 Machine Learning

| Task | Models |
|---|---|
| Classification | Logistic Regression, Random Forest, Gradient Boosting |
| Regression | Ridge, Random Forest, Gradient Boosting |
| Time-Series | Prophet (primary), ETS/Holt-Winters (fallback) |

- Auto task detection
- Optuna hyperparameter tuning
- Feature importance charts
- Model comparison leaderboard

### 💬 AI Chat — Powered by Groq ⚡
- Natural language dataset querying (~300 tokens/s)
- Automatic pandas code generation + execution
- Plotly chart generation from natural language
- Model selectable: `llama-3.3-70b-versatile` (default), `mixtral-8x7b-32768`, `llama-3.1-8b-instant`

### 📄 Reporting
- AI executive narrative (via Groq LLaMA 3.3 70B)
- 5-7 actionable business insights
- Professional PDF (ReportLab)

---

## 🚀 Quick Start

### 1. Get a free Groq API key
```
https://console.groq.com/keys
```

### 2. Clone & configure
```bash
git clone https://github.com/your-org/ai-data-analyst.git
cd ai-data-analyst
cp .env.example .env
# Edit .env — set GROQ_API_KEY=gsk_...
```

### 3. Docker Compose (recommended)
```bash
docker-compose up --build
# Frontend : http://localhost:8501
# API Docs : http://localhost:8000/api/docs
```

### 4. Local development
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make dev-api        # Terminal 1 — FastAPI :8000
make dev-frontend   # Terminal 2 — Streamlit :8501
```

### 5. Run tests
```bash
make test           # 21/21 ✅
```

---

## ⚙️ Groq Model Selection

Change `LLM_MODEL` in `.env`:

| Model | Speed | Context | Best For |
|---|---|---|---|
| `llama-3.3-70b-versatile` | Fast | 128K | Default — best quality |
| `llama-3.1-8b-instant` | Fastest | 128K | Low-latency chat |
| `mixtral-8x7b-32768` | Fast | 32K | Long document analysis |
| `gemma2-9b-it` | Fast | 8K | Lightweight tasks |

---

## ☁️ Azure Deployment

### GitHub Secrets required

| Secret | Description |
|---|---|
| `AZURE_CREDENTIALS` | Service principal JSON |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription ID |
| `AZURE_ACR_NAME` | Container registry name |
| `ACR_USERNAME` | ACR admin username |
| `ACR_PASSWORD` | ACR admin password |
| `AZURE_WEBAPP_NAME` | Frontend Web App name |
| `AZURE_WEBAPP_BACKEND` | API Web App name |
| `GROQ_API_KEY` | Groq API key (`gsk_...`) |

### Manual deploy
```bash
export AZURE_SUBSCRIPTION_ID="..."
export AZURE_ACR_NAME="acrdataanalyst"
export GROQ_API_KEY="gsk_..."
chmod +x deployment/azure/deploy.sh
./deployment/azure/deploy.sh
```

### CI/CD Flow
```
Push to main
    ↓
Lint (ruff) + Tests (pytest) — 21 tests
    ↓
Build Docker image (multi-stage)
    ↓
Push to Azure Container Registry
    ↓
Deploy API + Frontend to Azure App Service
    ↓
Health check /health endpoint
```

---

## 🔐 Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_...                # Free from console.groq.com

# Optional overrides
LLM_MODEL=llama-3.3-70b-versatile   # or llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1
APP_ENV=production
MAX_UPLOAD_SIZE_MB=100
LOG_LEVEL=INFO
```

---

## 📈 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/data/upload` | Upload CSV/Excel |
| `GET` | `/api/v1/data/datasets` | List datasets |
| `GET` | `/api/v1/eda/{id}` | Full EDA |
| `POST` | `/api/v1/ml/{id}/train` | Train & compare models |
| `POST` | `/api/v1/ml/{id}/forecast` | Time-series forecast |
| `POST` | `/api/v1/llm/{id}/chat` | ⚡ Groq NL chat |
| `GET` | `/api/v1/llm/{id}/insights` | ⚡ AI business insights |
| `GET` | `/api/v1/llm/{id}/report/pdf` | Download PDF report |

Swagger UI: `http://localhost:8000/api/docs`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.35, Plotly 5.22 |
| Backend | FastAPI 0.111, Uvicorn |
| LLM | **Groq** (LLaMA 3.3 70B / Mixtral 8x7B / Gemma 2) |
| ML | scikit-learn, XGBoost, LightGBM, Prophet, Optuna |
| Reporting | ReportLab |
| Logging | Loguru |
| Validation | Pydantic v2 |
| Container | Docker (multi-stage) + Docker Compose |
| CI/CD | GitHub Actions |
| Cloud | Azure App Service + Azure Container Registry |

---

## 📄 License

MIT License

---

*Built with ⚡ Groq · FastAPI · Streamlit · scikit-learn*
