# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com); versioning: [SemVer](https://semver.org).

## [Unreleased]

## [2.0.0] — 2026-11-18
### Added
- FastAPI service with versioned `/v1` API, OpenAPI docs, SSE streaming (`/v1/ask/stream`).
- API-key auth, per-IP / per-session rate limiting, request IDs, security headers, structured error bodies.
- Structured JSON logging and Prometheus metrics; Grafana/Prometheus compose profile and alert rules.
- Typed, validated settings (pydantic-settings); fail-fast startup.
- On-disk vector index cache; embedding model and index baked into Docker images.
- Prompt-injection hardening and control-character sanitisation.
- Automatic fallback to extractive answers when the LLM provider fails.
- HTTP and in-process UI clients sharing one contract.
- Hardened multi-stage Docker images, docker-compose, Render blueprint.
- CI/CD: lint, mypy, tests on 3.11/3.12 with coverage gate, retrieval quality gate, pip-audit, CodeQL, image build + smoke test, GHCR publishing, Dependabot.
- Pinned lockfiles, `pyproject.toml`, Makefile, pre-commit, deployment/architecture/security docs.

### Changed
- UI is now a client of the API (`streamlit_app.py`); run in-process when `API_URL` is empty.
- Test suite reorganised by layer (54 tests, 93% coverage).

## [1.0.0] — 2026-11-18
### Added
- Initial Streamlit app: eligibility rules engine, hybrid RAG, EN/HI/KN support, PII redaction, evaluation script.
