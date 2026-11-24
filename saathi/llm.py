"""Client for any OpenAI-compatible chat API : timeouts, retries with backoff, metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

from .metrics import LLM_CALLS, LLM_LATENCY
from .settings import Settings, get_settings

log = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}


class LLMError(RuntimeError):
    """Raised when the upstream LLM fails after all retries."""


def detect_language(text: str) -> str:
    """Script-based detection — cheap and reliable for English / Hindi / Kannada."""
    kn = sum(0x0C80 <= ord(c) <= 0x0CFF for c in text)
    hi = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    if max(kn, hi) == 0:
        return "en"
    return "kn" if kn >= hi else "hi"


class LLM:
    def __init__(self, settings: Settings | None = None, client: Any | None = None):
        self.s = settings or get_settings()
        self.model = self.s.llm_model
        self.client = client
        if self.client is None and self.s.llm_configured:
            from openai import OpenAI

            # SDK-level retries off: we retry ourselves so we can log and count attempts.
            self.client = OpenAI(
                api_key=self.s.llm_api_key.get_secret_value(),
                base_url=self.s.llm_base_url,
                timeout=self.s.llm_timeout_s,
                max_retries=0,
            )

    @property
    def available(self) -> bool:
        return self.client is not None

    def _call(self, kind: str, messages: list[dict], stream: bool, temperature: float) -> Any:
        if self.client is None:
            raise LLMError("LLM is not configured")
        delay = 1.0
        for attempt in range(1, self.s.llm_max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream,
                    max_tokens=self.s.llm_max_tokens,
                )
                LLM_LATENCY.labels(kind).observe(time.perf_counter() - start)
                LLM_CALLS.labels(kind, "ok").inc()
                return resp
            except Exception as e:  # 429 / 5xx / timeouts -> exponential backoff
                LLM_CALLS.labels(kind, "error").inc()
                log.warning("llm_call_failed", extra={"kind": kind, "attempt": attempt, "error": type(e).__name__})
                if attempt == self.s.llm_max_retries:
                    raise LLMError(f"LLM unavailable: {type(e).__name__}") from e
                time.sleep(delay)
                delay *= 2
        raise LLMError("unreachable")

    def complete(self, messages: list[dict], kind: str = "complete") -> str:
        resp = self._call(kind, messages, stream=False, temperature=0.0)
        return (resp.choices[0].message.content or "").strip()

    def stream(self, messages: list[dict]) -> Iterator[str]:
        for event in self._call("answer", messages, stream=True, temperature=0.2):
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta
