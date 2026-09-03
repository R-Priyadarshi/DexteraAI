# ============================================================
#  Dextera AI — FastAPI Application Factory
# ============================================================
"""
Production FastAPI application with:
  • REST endpoints for health, prediction, model info
  • WebSocket endpoint for real-time streaming
  • CORS, security headers, request ID middleware
  • Structured logging integration
  • Graceful startup / shutdown lifecycle

NOTE: This is OPTIONAL. The core ML pipeline (core/) runs without a server.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator  # noqa: TCH003 — needed at runtime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.apps.api.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from backend.apps.api.routes import router as api_router
from backend.config import settings
from backend.logging_config import setup_logging

_start_time: float = 0.0


def get_uptime() -> float:
    return time.time() - _start_time


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ANN001
    """Startup / shutdown lifecycle."""
    global _start_time
    _start_time = time.time()
    setup_logging()
    logger.info(
        "🤟 {} v{} starting  |  env={}  debug={}",
        settings.app_name,
        settings.app_version,
        settings.app_env,
        settings.debug,
    )
    yield
    logger.info("Dextera AI shutting down gracefully")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "🤟 **Dextera AI** — Industrial-grade, privacy-preserving "
            "gesture intelligence platform.\n\n"
            "All inference runs 100% on-device. No data leaves your machine."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    # ── Routes ───────────────────────────────────────────────
    app.include_router(api_router, prefix="/api")

    return app


app = create_app()
