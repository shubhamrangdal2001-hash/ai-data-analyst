"""FastAPI application entry point."""
from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.api.data_routes import router as data_router
from backend.api.eda_routes import router as eda_router
from backend.api.llm_routes import router as llm_router
from backend.api.ml_routes import router as ml_router
from backend.core.config import settings
from backend.core.exceptions import register_exception_handlers
from backend.core.logging import setup_logging

setup_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade AI Data Analyst Agent API",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        logger.debug("{} {} → {} ({}ms)", request.method, request.url.path, response.status_code, elapsed)
        return response

    # Exception handlers
    register_exception_handlers(app)

    # Routers
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(eda_router, prefix="/api/v1")
    app.include_router(ml_router, prefix="/api/v1")
    app.include_router(llm_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": settings.app_version, "environment": settings.app_env}

    @app.on_event("startup")
    async def startup():
        logger.info("🚀 {} v{} started | env={}", settings.app_name, settings.app_version, settings.app_env)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload)
