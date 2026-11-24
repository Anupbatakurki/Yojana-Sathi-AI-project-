"""Hybrid retrieval: BM25 (keyword) + dense embeddings, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .metrics import RETRIEVAL_LATENCY
from .schemes import Chunk

log = logging.getLogger(__name__)

_STOP = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "for",
    "and",
    "or",
    "in",
    "on",
    "is",
    "are",
    "be",
    "by",
    "with",
    "as",
    "at",
    "from",
    "it",
    "this",
    "that",
    "what",
    "which",
    "who",
    "how",
    "do",
    "does",
    "i",
    "my",
    "me",
    "can",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedder:
    """Small ONNX embedding model (no GPU / torch) — fits 512 MB free hosting tiers."""

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        from fastembed import TextEmbedding

        self.name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, cache_dir: Path | None = None):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.embedder: Embedder | None = embedder
        self.vectors: np.ndarray | None = None
        if embedder is not None:
            try:
                self.vectors = self._load_or_build(embedder, cache_dir)
            except Exception as e:  # model download blocked, etc. -> degrade gracefully
                log.warning("dense_index_disabled", extra={"error": str(e)})
                self.embedder = None

    def _load_or_build(self, embedder: Embedder, cache_dir: Path | None) -> np.ndarray:
        """Cache the index on disk, keyed by model + exact corpus, so restarts are instant."""
        texts = [c.text for c in self.chunks]
        key = hashlib.sha256((getattr(embedder, "name", "?") + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
        path = cache_dir / f"index-{key}.npy" if cache_dir else None
        if path is not None and path.exists():
            log.info("dense_index_loaded", extra={"path": str(path)})
            return np.load(path)
        vecs = embedder.embed(texts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vecs)
        return vecs

    @property
    def mode(self) -> str:
        return "hybrid" if self.vectors is not None else "bm25"

    def search(self, query: str, k: int = 6, pool: int = 20) -> list[tuple[Chunk, float]]:
        start = time.perf_counter()
        idxs = list(range(len(self.chunks)))
        bm = self.bm25.get_scores(tokenize(query))
        rankings = [sorted(idxs, key=lambda i: -bm[i])[:pool]]

        if self.vectors is not None and self.embedder is not None:
            q = self.embedder.embed([query])[0]
            dense = self.vectors @ q
            rankings.append(sorted(idxs, key=lambda i: -dense[i])[:pool])

        fused = rrf(rankings)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        RETRIEVAL_LATENCY.observe(time.perf_counter() - start)
        return [(self.chunks[i], fused[i]) for i in top]
