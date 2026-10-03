"""Structured logging setup using loguru."""
import sys
from pathlib import Path

from loguru import logger

from backend.core.config import settings


def setup_logging() -> None:
    """Configure loguru for the application."""
    logger.remove()

    log_format_text = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    log_format_json = (
        '{{"time":"{time:YYYY-MM-DD HH:mm:ss.SSS}",'
        '"level":"{level}",'
        '"module":"{name}",'
        '"function":"{function}",'
        '"line":{line},'
        '"message":"{message}"}}'
    )

    fmt = log_format_json if settings.log_format == "json" else log_format_text

    # Console handler
    logger.add(
        sys.stdout,
        format=fmt,
        level=settings.log_level,
        colorize=(settings.log_format == "text"),
        backtrace=settings.debug,
        diagnose=settings.debug,
    )

    # File handler – rotated daily, retained 30 days
    log_file = settings.log_dir / "app_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        format=fmt,
        level=settings.log_level,
        rotation="00:00",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
    )

    # Error-only file
    error_file = settings.log_dir / "error_{time:YYYY-MM-DD}.log"
    logger.add(
        str(error_file),
        format=fmt,
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="gz",
    )

    logger.info(
        "Logging initialised | env={} level={} format={}",
        settings.app_env,
        settings.log_level,
        settings.log_format,
    )


# Expose a named logger for imports across the app
def get_logger(name: str):
    return logger.bind(module=name)
