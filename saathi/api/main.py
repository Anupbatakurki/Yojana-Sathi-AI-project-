"""FastAPI application factory.

Run:  uvicorn saathi.api.main:app --host 0.0.0.0 --port 8000 --proxy-headers
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..logging_config import configure_logging, request_id_var
from ..pipeline import SaathiPipeline
from ..settings import Settings, get_settings
from .middleware import RequestContextMiddleware
from .routes import ops, v1
from .security import SlidingWindowLimiter

log = logging.getLogger(__name__)

DESCRIPTION = """
Multilingual (English / Hindi / Kannada) assistant for Indian government welfare schemes.

* **/v1/eligibility** — deterministic, explainable eligibility screening
* **/v1/ask** and **/v1/ask/stream** — grounded RAG answers with citations
* Authenticate with the `X-API-Key` header when the server has `API_KEYS` set.
"""


def _error(
    status_code: int, error: str, detail: object = None, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={"error": error, "detail": detail, "request_id": request_id_var.get()},
    )


def create_app(settings: Settings | None = None, pipeline: SaathiPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load models off the event loop; /readyz returns 503 until this finishes.
        app.state.pipeline = pipeline or await run_in_threadpool(SaathiPipeline, settings)
        if settings.env == "prod" and not settings.api_key_set:
            log.warning("api_auth_disabled_in_prod")
        yield

    app = FastAPI(
        title="Yojana Saathi API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.pipeline = None
    app.state.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute else None

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "X-Client-Session"],
            expose_headers=["X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        return _error(422, "Invalid request", detail)

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_error")
        return _error(500, "Internal server error")

    app.include_router(ops)
    app.include_router(v1)
    return app


app = create_app()
