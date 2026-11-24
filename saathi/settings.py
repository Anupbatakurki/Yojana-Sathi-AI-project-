"""Typed, validated configuration (12-factor: everything comes from the environment).

Invalid values fail fast at startup instead of at the first request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- runtime ---
    env: Literal["dev", "test", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True

    # --- LLM (any OpenAI-compatible endpoint: Groq, OpenAI, Together, vLLM, Ollama...) ---
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_s: float = Field(30.0, gt=0, le=300)
    llm_max_retries: int = Field(3, ge=1, le=10)
    llm_max_tokens: int = Field(1200, ge=64, le=8192)

    # --- retrieval ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    use_dense: bool = True
    cache_dir: Path = ROOT / ".cache"
    top_k: int = Field(6, ge=1, le=20)

    # --- data ---
    schemes_path: Path = ROOT / "data" / "schemes.json"
    feedback_path: Path = ROOT / "data" / "feedback.jsonl"

    # --- API protection ---
    api_keys: str = ""  # comma-separated; empty = no auth (dev only)
    rate_limit_per_minute: int = Field(30, ge=0)  # per client IP; 0 disables
    cors_origins: str = ""  # comma-separated allowed origins
    max_question_chars: int = Field(1000, ge=50, le=5000)
    history_turns: int = Field(6, ge=0, le=20)

    @field_validator("schemes_path")
    @classmethod
    def _schemes_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"schemes file not found: {v}")
        return v

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
