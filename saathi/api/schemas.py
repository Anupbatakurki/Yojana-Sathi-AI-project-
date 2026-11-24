"""Public API contract (request / response models). Changing these = changing the API version."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..eligibility import FLAG_LABELS, CitizenProfile, EligibilityResult
from ..schemes import Chunk, Scheme

Lang = Literal["en", "hi", "kn"]


class Profile(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 45,
                "gender": "female",
                "state": "Karnataka",
                "residence": "rural",
                "social_category": "obc",
                "annual_income": 150000,
                "flags": ["farmer", "bpl"],
            }
        },
    )

    age: int | None = Field(None, ge=0, le=120)
    gender: Literal["male", "female", "transgender"] | None = None
    state: str | None = Field(None, max_length=60)
    residence: Literal["rural", "urban"] | None = None
    social_category: Literal["general", "obc", "sc", "st"] | None = None
    annual_income: int | None = Field(None, ge=0, le=10**10)
    flags: list[str] = Field(
        default_factory=list, max_length=len(FLAG_LABELS), description="Any of: " + ", ".join(FLAG_LABELS)
    )

    @field_validator("flags")
    @classmethod
    def _known_flags(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(FLAG_LABELS))
        if unknown:
            raise ValueError(f"unknown flags: {unknown}")
        return v

    def to_domain(self) -> CitizenProfile:
        return CitizenProfile(
            age=self.age,
            gender=self.gender,
            state=self.state,
            residence=self.residence,
            social_category=self.social_category,
            annual_income=self.annual_income,
            flags=set(self.flags),
        )


class SchemeOut(BaseModel):
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

    @classmethod
    def of(cls, s: Scheme) -> SchemeOut:
        return cls(
            id=s.id,
            name=s.name,
            level=s.level,
            category=s.category,
            summary=s.summary,
            benefits=s.benefits,
            eligibility=s.eligibility,
            documents=s.documents,
            how_to_apply=s.how_to_apply,
            official_url=s.official_url,
        )


class EligibilityOut(BaseModel):
    scheme: SchemeOut
    verdict: Literal["eligible", "maybe", "not_eligible"]
    reasons: list[str]
    missing: list[str]

    @classmethod
    def of(cls, r: EligibilityResult) -> EligibilityOut:
        return cls(scheme=SchemeOut.of(r.scheme), verdict=r.verdict.value, reasons=r.reasons, missing=r.missing)


class EligibilityResponse(BaseModel):
    results: list[EligibilityOut]
    counts: dict[str, int]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"question": "What documents do I need for PM-KISAN?", "language": "en"}},
    )

    question: str = Field(..., min_length=2, max_length=2000)
    language: Lang | None = Field(None, description="Answer language; auto-detected if omitted")
    profile: Profile | None = None
    history: list[Turn] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    tag: str
    scheme_id: str
    scheme_name: str
    section: str
    url: str

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> list[Source]:
        return [
            cls(tag=f"S{i}", scheme_id=c.scheme_id, scheme_name=c.scheme_name, section=c.section, url=c.url)
            for i, c in enumerate(chunks, 1)
        ]


class AskMeta(BaseModel):
    request_id: str
    language: Lang
    search_query: str
    mode: Literal["llm", "extractive", "no_match"]
    sources: list[Source]
    eligibility: list[EligibilityOut]


class AskResponse(AskMeta):
    answer: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(..., max_length=64)
    rating: Literal[-1, 1]
    question: str = Field(..., max_length=2000)
    answer: str = Field(..., max_length=8000)
    comment: str | None = Field(None, max_length=1000)


class Health(BaseModel):
    status: Literal["ok", "starting"]
    version: str
    schemes: int = 0
    retrieval: str = ""
    llm_configured: bool = False


class ErrorBody(BaseModel):
    error: str
    detail: str | None = None
    request_id: str
