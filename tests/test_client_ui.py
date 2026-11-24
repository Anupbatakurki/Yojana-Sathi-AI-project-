"""Both UI clients must produce the same contract; the Streamlit UI must render in both modes."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from saathi.api.main import create_app
from saathi.client import ClientError, HttpClient, LocalClient

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def http_client(settings, offline_pipe):
    with TestClient(create_app(settings, offline_pipe)) as tc:
        yield HttpClient("http://testserver", http=tc)


@pytest.fixture
def local_client(offline_pipe):
    return LocalClient(offline_pipe)


@pytest.mark.parametrize("which", ["http_client", "local_client"])
def test_clients_share_contract(request, which):
    c = request.getfixturevalue(which)
    assert c.health()["schemes"] == 20
    elig = c.eligibility({"age": 45, "flags": ["farmer"]})
    assert elig["results"][0]["scheme"]["id"] and set(elig["counts"]) == {"eligible", "maybe", "not_eligible"}
    events = list(c.ask_stream({"question": "documents for PM-KISAN"}))
    assert events[0][0] == "meta" and events[-1][0] == "done"
    assert events[0][1]["sources"][0]["scheme_id"] == "pm-kisan"
    c.feedback({"request_id": events[-1][1]["request_id"], "rating": 1, "question": "q", "answer": "a"})


def test_http_client_raises_on_validation_error(http_client):
    with pytest.raises(ClientError) as e:
        http_client.eligibility({"age": -5})
    assert e.value.status == 422


def test_ui_renders_and_answers_in_process(monkeypatch):
    monkeypatch.setenv("USE_DENSE", "false")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("API_URL", "")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Yojana Saathi" in t.value for t in at.title)
    at.chat_input[0].set_value("What documents are needed for PM-KISAN?").run()
    assert not at.exception
    assert any("[S1]" in m.value for m in at.markdown)
