"""Structured logging configuration (Loguru)."""

from __future__ import annotations

import sys

from loguru import logger

from backend.config import settings


def setup_logging() -> None:
    """Configure Loguru for production-grade structured logging."""
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=fmt,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )

    logger.add(
        "logs/dextera_{time:YYYY-MM-DD}.log",
        format=fmt,
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        enqueue=True,
    )

    logger.info(
        "Logging ready  |  level={}  env={}  version={}",
        settings.log_level,
        settings.app_env,
        settings.app_version,
    )
