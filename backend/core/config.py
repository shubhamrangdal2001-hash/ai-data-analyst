"""Application configuration loaded from environment variables."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "AI Data Analyst Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    app_version: str = "1.0.0"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = True

    # ── Frontend ──────────────────────────────────────────────────────────────
    streamlit_port: int = 8501
    backend_url: str = "http://localhost:8000"

    # ── LLM — Groq ───────────────────────────────────────────────────────────
    # Primary provider: Groq (ultra-fast inference via llama / mixtral)
    # Fallback: OpenAI-compatible endpoint (set groq_base_url accordingly)
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Recommended Groq models (free tier):
    #   llama-3.3-70b-versatile   ← best quality, default
    #   llama-3.1-8b-instant      ← fastest / lowest latency
    #   mixtral-8x7b-32768        ← long-context tasks
    #   gemma2-9b-it              ← lightweight
    llm_model: str = "llama-3.3-70b-versatile"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1

    # ── Storage ───────────────────────────────────────────────────────────────
    upload_dir: Path = Path("./data/uploads")
    reports_dir: Path = Path("./data/reports")
    max_upload_size_mb: int = 100

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"
    log_dir: Path = Path("./logs")

    @field_validator("upload_dir", "reports_dir", "log_dir", mode="before")
    @classmethod
    def create_dirs(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
