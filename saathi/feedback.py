"""Append-only feedback store (JSONL). Swap for a database table in larger deployments."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from .metrics import FEEDBACK
from .pipeline import redact_pii

_lock = threading.Lock()


def log_feedback(
    path: Path, request_id: str, question: str, answer: str, rating: int, comment: str | None = None
) -> None:
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "rating": rating,  # +1 helpful, -1 not helpful
        "question": redact_pii(question),
        "answer": answer[:4000],
        "comment": redact_pii(comment) if comment else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    FEEDBACK.labels("up" if rating > 0 else "down").inc()
