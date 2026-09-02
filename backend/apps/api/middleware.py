# ============================================================
#  Dextera AI — Custom Middleware
# ============================================================
from __future__ import annotations

import time
import uuid
from collections import deque
from typing import TYPE_CHECKING

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID header to every request/response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject standard security headers for production hardening."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Powered-By"] = "Dextera AI"
        return response


RATE_LIMIT = 100  # requests per minute
rate_limit_store: dict[str, int] = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Limit requests per client IP, per minute.

    NOTE: not mounted by default (see backend/apps/api/main.py). It is
    process-local, so it does not hold across multiple workers or replicas.
    Use a shared store (Redis) or an upstream proxy for real rate limiting.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = int(time.time() / 60)  # minute bucket
        key = f"{client_ip}:{now}"

        # Drop expired buckets so the store cannot grow without bound.
        if len(rate_limit_store) > 10_000:
            for stale in [k for k in rate_limit_store if not k.endswith(f":{now}")]:
                del rate_limit_store[stale]

        count = rate_limit_store.get(key, 0)
        if count >= RATE_LIMIT:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
        rate_limit_store[key] = count + 1
        return await call_next(request)


# Usage logging middleware. Bounded so a long-running process cannot leak memory.
MAX_USAGE_LOG_ENTRIES = 1000
usage_log: deque[dict[str, object]] = deque(maxlen=MAX_USAGE_LOG_ENTRIES)


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    """Log usage data for requests."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        usage_log.append({
            "path": request.url.path,
            "method": request.method,
            "timestamp": time.time(),
            # request.client is None on some transports; don't crash the request.
            "client_ip": request.client.host if request.client else None,
        })
        response = await call_next(request)
        return response
