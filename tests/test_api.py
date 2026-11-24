import json

import pytest
from fastapi.testclient import TestClient

from saathi.api.main import create_app
from saathi.pipeline import SaathiPipeline

from .conftest import fake_llm, make_settings


@pytest.fixture
def client(settings, offline_pipe):
    with TestClient(create_app(settings, offline_pipe)) as c:
        yield c


def sse(resp) -> list[tuple[str, dict]]:
    out, event = [], None
    for line in resp.iter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            out.append((event, json.loads(line[5:])))
    return out


# ---------- ops ----------
def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["schemes"] == 20 and ready["retrieval"] == "bm25" and ready["llm_configured"] is False


def test_request_id_and_security_headers(client):
    r = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
    assert r.headers["x-content-type-options"] == "nosniff"
    bad = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces"})
    assert bad.headers["x-request-id"] != "bad id with spaces"


def test_metrics_exposed(client):
    client.get("/v1/schemes")
    body = client.get("/metrics").text
    assert 'saathi_http_requests_total{method="GET",route="/v1/schemes",status="200"}' in body


def test_openapi_documents_v1(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/ask", "/v1/ask/stream", "/v1/eligibility", "/v1/schemes"} <= set(paths)


# ---------- schemes & eligibility ----------
def test_schemes(client):
    assert len(client.get("/v1/schemes").json()) == 20
    assert client.get("/v1/schemes/pm-kisan").json()["name"].startswith("PM-KISAN")
    r = client.get("/v1/schemes/nope")
    assert r.status_code == 404 and r.json()["error"].startswith("Unknown scheme")


def test_eligibility(client):
    r = client.post("/v1/eligibility", json={"age": 45, "flags": ["farmer"]})
    data = r.json()
    assert r.status_code == 200 and sum(data["counts"].values()) == 20
    assert data["results"][0]["verdict"] == "eligible"


@pytest.mark.parametrize("bad", [{"age": -1}, {"flags": ["astronaut"]}, {"gender": "x"}, {"unknown": 1}])
def test_eligibility_validation(client, bad):
    r = client.post("/v1/eligibility", json=bad)
    assert r.status_code == 422 and r.json()["error"] == "Invalid request" and r.json()["request_id"]


# ---------- ask ----------
def test_ask_offline(client):
    r = client.post("/v1/ask", json={"question": "documents for PM-KISAN", "profile": {"age": 40, "flags": ["farmer"]}})
    j = r.json()
    assert j["mode"] == "extractive" and j["sources"][0]["scheme_id"] == "pm-kisan"
    assert j["eligibility"][0]["verdict"] == "eligible" and "[S1]" in j["answer"]


def test_ask_stream_with_llm(tmp_path):
    s = make_settings(tmp_path)
    llm, _ = fake_llm(s)
    with (
        TestClient(create_app(s, SaathiPipeline(s, llm=llm))) as c,
        c.stream("POST", "/v1/ask/stream", json={"question": "ಮಹಿಳೆಯರಿಗೆ ಉಚಿತ ಬಸ್ ಪ್ರಯಾಣ"}) as r,
    ):
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse(r)
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and kinds.count("token") == 3
    assert events[0][1]["language"] == "kn" and events[0][1]["mode"] == "llm"


def test_llm_outage_falls_back_to_extractive(tmp_path):
    s = make_settings(tmp_path)
    llm, _ = fake_llm(s, fail=True)
    with TestClient(create_app(s, SaathiPipeline(s, llm=llm))) as c:
        j = c.post("/v1/ask", json={"question": "free bus travel for women"}).json()
        assert j["mode"] == "extractive" and "[S1]" in j["answer"]
        with c.stream("POST", "/v1/ask/stream", json={"question": "free bus travel for women"}) as r:
            events = sse(r)
    assert any(e == "token" and "[S1]" in d["text"] for e, d in events)


def test_feedback(client, settings):
    r = client.post(
        "/v1/feedback", json={"request_id": "r1", "rating": -1, "question": "my no 9876543210", "answer": "x"}
    )
    assert r.status_code == 204
    line = json.loads(settings.feedback_path.read_text().strip())
    assert line["rating"] == -1 and "9876543210" not in line["question"]


# ---------- protection ----------
def test_api_key_required_when_configured(tmp_path, offline_pipe):
    s = make_settings(tmp_path, api_keys="k1,k2")
    with TestClient(create_app(s, offline_pipe)) as c:
        assert c.get("/v1/schemes").status_code == 401
        assert c.get("/v1/schemes", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/schemes", headers={"X-API-Key": "k2"}).status_code == 200
        assert c.get("/healthz").status_code == 200  # probes stay open


def test_rate_limit_per_ip_and_per_session(tmp_path, offline_pipe):
    s = make_settings(tmp_path, rate_limit_per_minute=2, api_keys="k")
    with TestClient(create_app(s, offline_pipe)) as c:
        body = {"question": "pension"}
        h = {"X-API-Key": "k"}
        assert [c.post("/v1/ask", json=body, headers=h).status_code for _ in range(3)] == [200, 200, 429]
        r = c.post("/v1/ask", json=body, headers=h)
        assert "retry-after" in r.headers
        # a trusted UI passing distinct end-user sessions gets separate buckets
        assert c.post("/v1/ask", json=body, headers={**h, "X-Client-Session": "u1"}).status_code == 200
        assert c.post("/v1/ask", json=body, headers={**h, "X-Client-Session": "u2"}).status_code == 200


def test_not_ready_returns_503(settings):
    app = create_app(settings, pipeline=None)  # lifespan not run -> pipeline missing
    c = TestClient(app)
    assert c.get("/readyz").status_code == 503
    assert c.get("/v1/schemes").status_code == 503
