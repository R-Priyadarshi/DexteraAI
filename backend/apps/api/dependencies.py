# ============================================================
#  Dextera AI — Dependency Injection
# ============================================================
"""
FastAPI dependency providers for inference engine, settings, etc.
Ensures single engine instance across the application lifetime.
"""
from __future__ import annotations

from functools import lru_cache

from backend.config import Settings, settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return settings



