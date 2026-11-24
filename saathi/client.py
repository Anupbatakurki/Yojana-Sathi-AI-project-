"""Clients used by the UI. Both speak the same JSON contract as the HTTP API.

* HttpClient  — talks to the deployed FastAPI service (production: UI and API scale separately).
* LocalClient — runs the pipeline in-process (single-container hosts like Streamlit Cloud / HF Spaces).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any, Protocol

import httpx

Event = tuple[str, dict[str, Any]]


class ClientError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class SaathiClient(Protocol):
    def health(self) -> dict[str, Any]: ...
    def eligibility(self, profile: dict[str, Any]) -> dict[str, Any]: ...
    def ask_stream(self, payload: dict[str, Any]) -> Iterator[Event]: ...
    def feedback(self, payload: dict[str, Any]) -> None: ...


class HttpClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        session_id: str | None = None,
        timeout: float = 60.0,
        http: httpx.Client | None = None,
    ):
        headers = {"X-Client-Session": session_id or uuid.uuid4().hex}
        if api_key:
            headers["X-API-Key"] = api_key
        self.http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.http.headers.update(headers)

    def _check(self, r: httpx.Response) -> None:
        if r.is_error:
            try:
                msg = r.json().get("error", r.text)
            except ValueError:
                msg = r.text
            raise ClientError(msg, r.status_code)

    def health(self) -> dict[str, Any]:
        r = self.http.get("/readyz")
        self._check(r)
        return r.json()

    def eligibility(self, profile: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post("/v1/eligibility", json=profile)
        self._check(r)
        return r.json()

    def ask_stream(self, payload: dict[str, Any]) -> Iterator[Event]:
        with self.http.stream("POST", "/v1/ask/stream", json=payload) as r:
            if r.is_error:
                r.read()
                self._check(r)
            event, data = "message", ""
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data += line[5:].strip()
                elif line == "" and data:
                    yield event, json.loads(data)
                    event, data = "message", ""

    def feedback(self, payload: dict[str, Any]) -> None:
        self._check(self.http.post("/v1/feedback", json=payload))


class LocalClient:
    """Same contract, no network. Reuses the API's schemas so the output shape is identical."""

    def __init__(self, pipeline: Any | None = None):
        from .pipeline import SaathiPipeline

        self.pipe = pipeline or SaathiPipeline()

    def health(self) -> dict[str, Any]:
        from . import __version__

        return {
            "status": "ok",
            "version": __version__,
            "schemes": len(self.pipe.schemes),
            "retrieval": self.pipe.retriever.mode,
            "llm_configured": self.pipe.llm.available,
        }

    def eligibility(self, profile: dict[str, Any]) -> dict[str, Any]:
        from .api.schemas import EligibilityOut, Profile
        from .eligibility import Verdict

        results = self.pipe.screen(Profile(**profile).to_domain())
        return {
            "results": [EligibilityOut.of(r).model_dump() for r in results],
            "counts": {v.value: sum(r.verdict == v for r in results) for v in Verdict},
        }

    def ask_stream(self, payload: dict[str, Any]) -> Iterator[Event]:
        from .api.schemas import AskMeta, AskRequest, EligibilityOut, Source
        from .llm import LLMError

        body = AskRequest(**payload)
        rid = uuid.uuid4().hex
        p = self.pipe.prepare(body.question, body.language, body.profile.to_domain() if body.profile else None)
        yield (
            "meta",
            AskMeta(
                request_id=rid,
                language=p.language,
                search_query=p.search_query,
                mode=self.pipe.mode_for(p),
                sources=Source.from_chunks(p.sources),
                eligibility=[EligibilityOut.of(r) for r in p.eligibility],
            ).model_dump(),
        )
        try:
            for tok in self.pipe.generate(p, [t.model_dump() for t in body.history]):
                yield "token", {"text": tok}
        except LLMError:
            yield "token", {"text": self.pipe.extractive_answer(p.sources, p.eligibility)}
        yield "done", {"request_id": rid}

    def feedback(self, payload: dict[str, Any]) -> None:
        from .api.schemas import FeedbackRequest
        from .feedback import log_feedback

        b = FeedbackRequest(**payload)
        log_feedback(self.pipe.s.feedback_path, b.request_id, b.question, b.answer, b.rating, b.comment)
