"""Prometheus metrics (scraped from GET /metrics)."""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter("saathi_http_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LATENCY = Histogram("saathi_http_request_seconds", "HTTP request latency", ["route"])
LLM_CALLS = Counter("saathi_llm_calls_total", "LLM API calls", ["kind", "outcome"])
LLM_LATENCY = Histogram(
    "saathi_llm_seconds",
    "LLM call latency (time to first byte for streams)",
    ["kind"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
RETRIEVAL_LATENCY = Histogram(
    "saathi_retrieval_seconds", "Hybrid retrieval latency", buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
)
ANSWERS = Counter("saathi_answers_total", "Answers served", ["mode", "language"])
FEEDBACK = Counter("saathi_feedback_total", "User feedback", ["rating"])
RATE_LIMITED = Counter("saathi_rate_limited_total", "Requests rejected by rate limiter")
