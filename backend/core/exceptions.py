"""Domain-specific exception hierarchy and FastAPI exception handlers."""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger


# ── Base ─────────────────────────────────────────────────────────────────────
class AppBaseException(Exception):
    """Root exception for the application."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail or message
        super().__init__(self.message)


# ── Data Layer ────────────────────────────────────────────────────────────────
class DataLoadError(AppBaseException):
    status_code = 422
    error_code = "DATA_LOAD_ERROR"


class UnsupportedFileTypeError(AppBaseException):
    status_code = 415
    error_code = "UNSUPPORTED_FILE_TYPE"


class DataValidationError(AppBaseException):
    status_code = 422
    error_code = "DATA_VALIDATION_ERROR"


class DataTooLargeError(AppBaseException):
    status_code = 413
    error_code = "DATA_TOO_LARGE"


# ── Analysis Layer ────────────────────────────────────────────────────────────
class AnalysisError(AppBaseException):
    status_code = 500
    error_code = "ANALYSIS_ERROR"


class InsufficientDataError(AppBaseException):
    status_code = 422
    error_code = "INSUFFICIENT_DATA"


# ── ML Layer ─────────────────────────────────────────────────────────────────
class ModelTrainingError(AppBaseException):
    status_code = 500
    error_code = "MODEL_TRAINING_ERROR"


class ModelNotFoundError(AppBaseException):
    status_code = 404
    error_code = "MODEL_NOT_FOUND"


class TargetColumnError(AppBaseException):
    status_code = 422
    error_code = "TARGET_COLUMN_ERROR"


# ── LLM Layer ─────────────────────────────────────────────────────────────────
class LLMError(AppBaseException):
    status_code = 503
    error_code = "LLM_ERROR"


class LLMRateLimitError(LLMError):
    status_code = 429
    error_code = "LLM_RATE_LIMIT"


# ── Handlers ──────────────────────────────────────────────────────────────────
def _error_response(exc: AppBaseException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "detail": exc.detail,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppBaseException)
    async def app_exception_handler(request: Request, exc: AppBaseException):
        logger.warning("AppException {} | path={} | msg={}", exc.error_code, request.url.path, exc.message)
        return _error_response(exc)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception | path={}", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred."},
        )
