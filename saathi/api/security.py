"""API-key auth and per-client rate limiting (FastAPI dependencies)."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from ..metrics import RATE_LIMITED
from ..settings import Settings
from .deps import get_app_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Depends(api_key_header), settings: Settings = Depends(get_app_settings)) -> None:
    allowed = settings.api_key_set
    if not allowed:  # auth disabled (local dev)
        return
    if not key or not any(hmac.compare_digest(key, k) for k in allowed):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"}
        )


class SlidingWindowLimiter:
    """In-memory sliding window per client.

    Fine for a single instance. For several replicas, swap for a Redis-backed limiter
    (same interface) — see docs/DEPLOYMENT.md.
    """

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Return None if allowed, else seconds until the next slot frees up."""
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return self.window - (now - q[0])
            q.append(now)
            if len(self.hits) > 50_000:  # bound memory under abuse
                self.hits = defaultdict(deque, {k: v for k, v in self.hits.items() if v})
            return None


def _client_key(request: Request, settings: Settings) -> str:
    """Anonymous callers are limited per IP. A trusted caller (valid API key, e.g. our own UI server)
    may pass X-Client-Session so each end user gets their own bucket instead of sharing one."""
    key = request.headers.get("x-api-key")
    session = request.headers.get("x-client-session", "")[:64]
    if key and session and key in settings.api_key_set:
        return f"session:{session}"
    # Behind a proxy (Render, HF, nginx) uvicorn --proxy-headers sets request.client from X-Forwarded-For.
    return f"ip:{request.client.host if request.client else 'anon'}"


def rate_limit(request: Request, settings: Settings = Depends(get_app_settings)) -> None:
    limiter: SlidingWindowLimiter | None = request.app.state.limiter
    if limiter is None:
        return
    retry = limiter.check(_client_key(request, settings))
    if retry is not None:
        RATE_LIMITED.inc()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many requests, slow down",
            headers={"Retry-After": str(int(retry) + 1)},
        )
