"""Evaluate retrieval quality (Hit@1, Hit@3, MRR) and, optionally, answer groundedness.

Doubles as a CI quality gate: exits non-zero if Hit@3 drops below --min-hit3.

Usage:
    python -m eval.evaluate                       # retrieval metrics
    python -m eval.evaluate --min-hit3 0.9        # fail if quality regresses
    python -m eval.evaluate --judge 8             # + LLM-as-judge on 8 answers (needs LLM_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from saathi.llm import LLMError, detect_language
from saathi.pipeline import SaathiPipeline
from saathi.settings import get_settings

HERE = Path(__file__).resolve().parent

JUDGE_PROMPT = """You are grading a government-scheme assistant.
<context>
{context}
</context>
<answer>
{answer}
</answer>
Score how well every factual claim in the answer is supported by the context, from 1 (mostly unsupported or
invented) to 5 (fully supported, nothing invented). Reply with only the number."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=int, default=0, help="number of answers to grade with LLM-as-judge")
    ap.add_argument("--min-hit3", type=float, default=0.0, help="fail (exit 1) if Hit@3 is below this")
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args()

    pipe = SaathiPipeline(get_settings())
    rows = [json.loads(line) for line in (HERE / "qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    hit1 = hit3 = n = skipped = 0
    rr = 0.0
    misses = []
    for row in rows:
        lang = detect_language(row["q"])
        if lang != "en" and not pipe.llm.available:
            skipped += 1
            continue
        query = pipe.to_search_query(row["q"], lang)
        ranking = list(dict.fromkeys(c.scheme_id for c in pipe.retrieve(query)))
        n += 1
        if row["scheme"] in ranking:
            pos = ranking.index(row["scheme"]) + 1
            hit1 += pos == 1
            hit3 += pos <= 3
            rr += 1 / pos
        if not ranking or ranking[0] != row["scheme"]:
            misses.append({"q": row["q"], "expected": row["scheme"], "got": ranking[:3]})

    report: dict = {
        "retrieval_mode": pipe.retriever.mode,
        "llm_model": pipe.llm.model if pipe.llm.available else None,
        "questions_evaluated": n,
        "skipped_non_english_without_llm": skipped,
        "hit@1": round(hit1 / n, 3),
        "hit@3": round(hit3 / n, 3),
        "mrr": round(rr / n, 3),
        "misses": misses,
    }

    if args.judge and pipe.llm.available:
        scores = []
        for row in rows[: args.judge]:
            try:
                p, text = pipe.ask(row["q"], language="en")
                ctx = "\n".join(c.text for c in p.sources)
                raw = pipe.llm.complete(
                    [{"role": "user", "content": JUDGE_PROMPT.format(context=ctx, answer=text)}], kind="judge"
                )
            except LLMError as e:
                print(f"judge skipped one item: {e}", file=sys.stderr)
                continue
            m = re.search(r"[1-5]", raw)
            if m:
                scores.append(int(m.group()))
        report["groundedness_avg_1to5"] = round(sum(scores) / len(scores), 2) if scores else None

    print(json.dumps(report, indent=2, ensure_ascii=False))
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if report["hit@3"] < args.min_hit3:
        print(f"QUALITY GATE FAILED: hit@3 {report['hit@3']} < {args.min_hit3}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
