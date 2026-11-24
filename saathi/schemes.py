"""Load the scheme knowledge base and turn it into retrievable text chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Scheme:
    id: str
    name: str
    level: str
    category: str
    summary: str
    benefits: str
    eligibility: str
    documents: list[str]
    how_to_apply: str
    official_url: str
    rules: dict = field(default_factory=dict)

    def full_text(self) -> str:
        return (
            f"Scheme: {self.name} ({self.level} government, {self.category})\n"
            f"Summary: {self.summary}\n"
            f"Benefits: {self.benefits}\n"
            f"Eligibility: {self.eligibility}\n"
            f"Documents required: {', '.join(self.documents)}\n"
            f"How to apply: {self.how_to_apply}\n"
            f"Official website: {self.official_url}"
        )


@dataclass
class Chunk:
    """One retrievable unit of text, tagged with the scheme it came from."""

    chunk_id: str
    scheme_id: str
    scheme_name: str
    section: str
    text: str
    url: str


REQUIRED_FIELDS = [
    "id",
    "name",
    "level",
    "category",
    "summary",
    "benefits",
    "eligibility",
    "documents",
    "how_to_apply",
    "official_url",
]


def load_schemes(path: str | Path) -> list[Scheme]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    schemes, seen = [], set()
    for raw in data["schemes"]:
        missing = [f for f in REQUIRED_FIELDS if f not in raw]
        if missing:
            raise ValueError(f"Scheme {raw.get('id', '?')} is missing fields: {missing}")
        if raw["id"] in seen:
            raise ValueError(f"Duplicate scheme id: {raw['id']}")
        seen.add(raw["id"])
        schemes.append(Scheme(**{k: raw[k] for k in REQUIRED_FIELDS}, rules=raw.get("rules", {})))
    return schemes


def chunk_schemes(schemes: list[Scheme]) -> list[Chunk]:
    """Section-level chunks: each section is prefixed with the scheme name so a
    chunk is self-contained (helps both BM25 and embeddings)."""
    chunks: list[Chunk] = []
    for s in schemes:
        sections = {
            "overview": f"{s.summary} Category: {s.category}. Level: {s.level}.",
            "benefits": s.benefits,
            "eligibility": s.eligibility,
            "documents": "Documents required: " + ", ".join(s.documents),
            "apply": s.how_to_apply,
        }
        for sec, body in sections.items():
            chunks.append(
                Chunk(
                    chunk_id=f"{s.id}::{sec}",
                    scheme_id=s.id,
                    scheme_name=s.name,
                    section=sec,
                    text=f"{s.name} — {sec}: {body}",
                    url=s.official_url,
                )
            )
    return chunks
