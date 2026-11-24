"""HTTP routes. Versioned under /v1; ops endpoints (/healthz, /readyz, /metrics) at the root."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..eligibility import Verdict
from ..feedback import log_feedback
from ..llm import LLMError
from ..pipeline import Prepared, SaathiPipeline
from ..settings import Settings
from .deps import get_app_settings, get_pipeline, get_request_id
from .schemas import (
    AskMeta,
    AskRequest,
    AskResponse,
    EligibilityOut,
    EligibilityResponse,
    ErrorBody,
    FeedbackRequest,
    Health,
    Profile,
    SchemeOut,
    Source,
)
from .security import rate_limit, require_api_key

log = logging.getLogger(__name__)

ops = APIRouter(tags=["ops"])
v1 = APIRouter(
    prefix="/v1",
    dependencies=[Depends(require_api_key)],
    responses={401: {"model": ErrorBody}, 429: {"model": ErrorBody}, 503: {"model": ErrorBody}},
)


# ------------------------------------------------------------------ ops
@ops.get("/healthz", response_model=Health, summary="Liveness probe")
def healthz() -> Health:
    return Health(status="ok", version=__version__)


@ops.get("/readyz", response_model=Health, summary="Readiness probe (pipeline loaded)")
def readyz(request: Request, response: Response) -> Health:
    pipe: SaathiPipeline | None = getattr(request.app.state, "pipeline", None)
    if pipe is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Health(status="starting", version=__version__)
    return Health(
        status="ok",
        version=__version__,
        schemes=len(pipe.schemes),
        retrieval=pipe.retriever.mode,
        llm_configured=pipe.llm.available,
    )


@ops.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------ schemes
@v1.get("/schemes", response_model=list[SchemeOut], tags=["schemes"], summary="List all schemes")
def list_schemes(pipe: SaathiPipeline = Depends(get_pipeline)) -> list[SchemeOut]:
    return [SchemeOut.of(s) for s in pipe.schemes]


@v1.get("/schemes/{scheme_id}", response_model=SchemeOut, tags=["schemes"], responses={404: {"model": ErrorBody}})
def get_scheme(scheme_id: str, pipe: SaathiPipeline = Depends(get_pipeline)) -> SchemeOut:
    s = pipe.by_id.get(scheme_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown scheme '{scheme_id}'")
    return SchemeOut.of(s)


# ------------------------------------------------------------------ eligibility
@v1.post(
    "/eligibility",
    response_model=EligibilityResponse,
    tags=["eligibility"],
    summary="Screen a citizen profile against every scheme (deterministic rules engine)",
)
def eligibility(profile: Profile, pipe: SaathiPipeline = Depends(get_pipeline)) -> EligibilityResponse:
    results = pipe.screen(profile.to_domain())
    counts = {v.value: sum(r.verdict == v for r in results) for v in Verdict}
    return EligibilityResponse(results=[EligibilityOut.of(r) for r in results], counts=counts)


# ------------------------------------------------------------------ ask
def _meta(p: Prepared, mode: str, rid: str) -> AskMeta:
    return AskMeta(
        request_id=rid,
        language=p.language,
        search_query=p.search_query,
        mode=mode,
        sources=Source.from_chunks(p.sources),
        eligibility=[EligibilityOut.of(r) for r in p.eligibility],
    )


def _prepare(pipe: SaathiPipeline, body: AskRequest) -> Prepared:
    return pipe.prepare(body.question, body.language, body.profile.to_domain() if body.profile else None)


@v1.post(
    "/ask",
    response_model=AskResponse,
    tags=["assistant"],
    dependencies=[Depends(rate_limit)],
    summary="Ask a question; returns the full grounded answer with sources",
)
def ask(
    body: AskRequest, pipe: SaathiPipeline = Depends(get_pipeline), rid: str = Depends(get_request_id)
) -> AskResponse:
    p = _prepare(pipe, body)
    mode = pipe.mode_for(p)
    history = [t.model_dump() for t in body.history]
    try:
        answer = "".join(pipe.generate(p, history))
    except LLMError:
        log.exception("llm_failed_fallback_extractive")
        mode, answer = "extractive", pipe.extractive_answer(p.sources, p.eligibility)
    return AskResponse(**_meta(p, mode, rid).model_dump(), answer=answer)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@v1.post(
    "/ask/stream",
    tags=["assistant"],
    dependencies=[Depends(rate_limit)],
    summary="Ask a question; Server-Sent Events stream",
    response_description="text/event-stream with events: meta, token, done, error",
    responses={200: {"content": {"text/event-stream": {}}}},
)
def ask_stream(
    body: AskRequest, pipe: SaathiPipeline = Depends(get_pipeline), rid: str = Depends(get_request_id)
) -> StreamingResponse:
    p = _prepare(pipe, body)
    history = [t.model_dump() for t in body.history]

    def events() -> Iterator[str]:
        yield _sse("meta", _meta(p, pipe.mode_for(p), rid).model_dump())
        sent_any = False
        try:
            for token in pipe.generate(p, history):
                sent_any = True
                yield _sse("token", {"text": token})
        except LLMError:
            log.exception("llm_failed_during_stream")
            if sent_any:
                yield _sse("error", {"message": "The AI service stopped responding. Please try again."})
            else:  # nothing sent yet -> degrade gracefully to the extractive answer
                yield _sse("token", {"text": pipe.extractive_answer(p.sources, p.eligibility)})
        yield _sse("done", {"request_id": rid})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
    )


# ------------------------------------------------------------------ feedback
@v1.post("/feedback", status_code=status.HTTP_204_NO_CONTENT, tags=["assistant"], dependencies=[Depends(rate_limit)])
def feedback(body: FeedbackRequest, settings: Settings = Depends(get_app_settings)) -> Response:
    log_feedback(settings.feedback_path, body.request_id, body.question, body.answer, body.rating, body.comment)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
