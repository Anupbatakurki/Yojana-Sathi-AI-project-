"""Dependency providers — overridable in tests via app.dependency_overrides."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from ..pipeline import SaathiPipeline
from ..settings import Settings


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_pipeline(request: Request) -> SaathiPipeline:
    pipe: SaathiPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service is starting")
    return pipe


def get_request_id(request: Request) -> str:
    return request.scope.get("state", {}).get("request_id", "-")
