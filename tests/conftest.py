from __future__ import annotations

import os

# Hermetic tests: never call a real LLM or download models, whatever the developer's .env says.
os.environ.update(
    {
        "USE_DENSE": "false",
        "LLM_API_KEY": "",
        "LOG_JSON": "false",
        "API_KEYS": "",
        "RATE_LIMIT_PER_MINUTE": "0",
        "API_URL": "",
    }
)

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from saathi.llm import LLM
from saathi.pipeline import SaathiPipeline
from saathi.settings import Settings

DATA = Path(__file__).resolve().parent.parent / "data" / "schemes.json"


def make_settings(tmp_path: Path, **kw) -> Settings:
    base = dict(
        env="test",
        use_dense=False,
        schemes_path=DATA,
        cache_dir=tmp_path / "cache",
        feedback_path=tmp_path / "feedback.jsonl",
        llm_api_key="",
        log_json=False,
        rate_limit_per_minute=0,
        llm_max_retries=1,
    )
    base.update(kw)
    return Settings(**base)


class FakeCompletions:
    """Stands in for openai's chat.completions; records calls, can be told to fail."""

    def __init__(
        self, tokens=("Shakti ", "gives ", "free travel [S1]."), translation="free bus travel for women", fail=False
    ):
        self.tokens, self.translation, self.fail = tokens, translation, fail
        self.calls: list[list[dict]] = []

    def create(self, model, messages, stream, **kw):
        self.calls.append(messages)
        if self.fail:
            raise TimeoutError("upstream timeout")
        if not stream:
            return NS(choices=[NS(message=NS(content=self.translation))])
        return iter([NS(choices=[NS(delta=NS(content=t))]) for t in self.tokens])


def fake_llm(settings: Settings, **kw) -> tuple[LLM, FakeCompletions]:
    comp = FakeCompletions(**kw)
    return LLM(settings, client=NS(chat=NS(completions=comp))), comp


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def offline_pipe(settings):
    return SaathiPipeline(settings)


@pytest.fixture
def schemes(offline_pipe):
    return offline_pipe.by_id
