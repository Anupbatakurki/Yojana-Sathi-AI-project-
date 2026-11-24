"""End-to-end RAG pipeline.

sanitize + redact PII -> detect language -> translate query -> hybrid retrieve
-> rules-engine eligibility -> grounded, cited answer (LLM, or extractive fallback)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

from .eligibility import CitizenProfile, EligibilityResult, Verdict, check, screen
from .llm import LANGUAGES, LLM, LLMError, detect_language
from .metrics import ANSWERS
from .prompts import SYSTEM_PROMPT, TRANSLATE_PROMPT, USER_TEMPLATE
from .retriever import Embedder, FastEmbedder, HybridRetriever
from .schemes import Chunk, Scheme, chunk_schemes, load_schemes
from .settings import Settings, get_settings

log = logging.getLogger(__name__)

NO_MATCH_TEXT = (
    "Sorry, I could not find a matching scheme. Please visit https://www.myscheme.gov.in "
    "or your nearest Common Service Centre (CSC)."
)

_PII_PATTERNS = [
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
]
_PROMPT_TAGS = re.compile(r"</?\s*(question|context|eligibility)\s*>", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def redact_pii(text: str) -> str:
    """Citizens often paste Aadhaar / phone numbers. Never send them to a third-party LLM or to logs."""
    for pat, repl in _PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


def sanitize(text: str, max_chars: int) -> str:
    """Strip control chars and our own prompt delimiters so user text cannot break out of <question>."""
    text = _CONTROL.sub("", text)
    text = _PROMPT_TAGS.sub("", text)
    return redact_pii(text.strip())[:max_chars]


AnswerMode = Literal["llm", "extractive", "no_match"]


@dataclass
class Prepared:
    question: str
    search_query: str
    language: str
    sources: list[Chunk]
    eligibility: list[EligibilityResult] = field(default_factory=list)


class SaathiPipeline:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None, embedder: Embedder | None = None):
        self.s = settings or get_settings()
        self.schemes: list[Scheme] = load_schemes(self.s.schemes_path)
        self.by_id = {s.id: s for s in self.schemes}
        self.chunks = chunk_schemes(self.schemes)
        if embedder is None and self.s.use_dense:
            try:
                embedder = FastEmbedder(self.s.embed_model, cache_dir=self.s.cache_dir / "models")
            except Exception as e:
                log.warning("embedding_model_unavailable", extra={"error": str(e)})
        self.retriever = HybridRetriever(self.chunks, embedder, cache_dir=self.s.cache_dir)
        self.llm = llm if llm is not None else LLM(self.s)
        log.info(
            "pipeline_ready",
            extra={"schemes": len(self.schemes), "retrieval": self.retriever.mode, "llm": self.llm.available},
        )

    # ---------- eligibility ----------
    def screen(self, profile: CitizenProfile) -> list[EligibilityResult]:
        return screen(self.schemes, profile)

    # ---------- retrieval ----------
    def to_search_query(self, question: str, lang: str) -> str:
        if lang == "en" or not self.llm.available:
            return question
        try:
            return self.llm.complete(
                [{"role": "user", "content": TRANSLATE_PROMPT.format(question=question)}], kind="translate"
            )
        except LLMError:
            log.warning("translation_failed_using_original")
            return question

    def retrieve(self, query: str) -> list[Chunk]:
        return [c for c, _ in self.retriever.search(query, k=self.s.top_k)]

    # ---------- answering ----------
    def prepare(self, question: str, language: str | None = None, profile: CitizenProfile | None = None) -> Prepared:
        q = sanitize(question, self.s.max_question_chars)
        detected = detect_language(q)
        lang = language or detected
        search_query = self.to_search_query(q, detected) if q else ""
        sources = self.retrieve(search_query) if search_query else []
        elig: list[EligibilityResult] = []
        if profile is not None:
            elig = [check(self.by_id[sid], profile) for sid in dict.fromkeys(c.scheme_id for c in sources)]
        return Prepared(q, search_query, lang, sources, elig)

    def mode_for(self, p: Prepared) -> AnswerMode:
        if not p.sources:
            return "no_match"
        return "llm" if self.llm.available else "extractive"

    def generate(self, p: Prepared, history: list[dict] | None = None) -> Iterator[str]:
        mode = self.mode_for(p)
        ANSWERS.labels(mode, p.language).inc()
        if mode == "no_match":
            yield NO_MATCH_TEXT
        elif mode == "extractive":
            yield self.extractive_answer(p.sources, p.eligibility)
        else:
            yield from self.llm.stream(self.build_messages(p, history))

    def ask(
        self,
        question: str,
        language: str | None = None,
        profile: CitizenProfile | None = None,
        history: list[dict] | None = None,
    ) -> tuple[Prepared, str]:
        p = self.prepare(question, language, profile)
        return p, "".join(self.generate(p, history))

    # ---------- prompt building ----------
    def build_messages(self, p: Prepared, history: list[dict] | None) -> list[dict]:
        context = "\n\n".join(f"[S{i + 1}] {c.text} (Source: {c.url})" for i, c in enumerate(p.sources))
        elig_block = ""
        if p.eligibility:
            lines = [
                f"- {r.scheme.name}: {r.verdict.value.replace('_', ' ')}"
                f" ({'; '.join(r.reasons) or 'no specific rules'})"
                + (f"; missing info: {', '.join(r.missing)}" if r.missing else "")
                for r in p.eligibility
            ]
            elig_block = "<eligibility>\n" + "\n".join(lines) + "\n</eligibility>"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(language=LANGUAGES.get(p.language, "English"))}]
        turns = (history or [])[-self.s.history_turns :] if self.s.history_turns else []
        for turn in turns:
            if turn.get("role") in ("user", "assistant"):
                content = turn["content"]
                if turn["role"] == "user":
                    content = sanitize(content, self.s.max_question_chars)
                msgs.append({"role": turn["role"], "content": content[:4000]})
        msgs.append(
            {
                "role": "user",
                "content": USER_TEMPLATE.format(context=context, eligibility=elig_block, question=p.question),
            }
        )
        return msgs

    @staticmethod
    def extractive_answer(sources: list[Chunk], elig: list[EligibilityResult]) -> str:
        """Works with no API key at all — useful for demos, outages and offline kiosks."""
        verdicts = {r.scheme.id: r.verdict for r in elig}
        tags = {
            Verdict.ELIGIBLE: " ✅ likely eligible",
            Verdict.MAYBE: " ❔ need more info",
            Verdict.NOT_ELIGIBLE: " ❌ likely not eligible",
        }
        lines = ["**Most relevant information found** (AI answers unavailable — showing source text):", ""]
        for i, c in enumerate(sources, 1):
            tag = tags[verdicts[c.scheme_id]] if c.scheme_id in verdicts else ""
            lines.append(f"- {c.text} [S{i}]{tag}")
        lines += ["", "_Rules can change — please confirm on the official website._"]
        return "\n".join(lines)
