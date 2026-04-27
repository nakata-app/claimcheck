"""Production middleware pattern: gate an LLM response on trust_score.

Framework-agnostic — works with any code path that has (1) an LLM
producing an answer and (2) a corpus you trust. Drop the
`verify_response()` function in front of your output handler.

Run from the repo root:

    pip install -e ../adaptmem ../halluguard -e .
    python examples/middleware.py
"""
from __future__ import annotations

from dataclasses import dataclass

from claimcheck import Pipeline, Verdict


@dataclass
class GatedResponse:
    """Result wrapper your service hands to the caller."""

    answer: str
    blocked: bool
    trust_score: float
    flagged_claims: list[str]
    timing_ms: float


# Trust thresholds — tune per product. Defaults are intentionally cautious.
BLOCK_BELOW = 0.40       # auto-block: response is mostly unsupported
WARN_BELOW = 0.70        # show with a warning banner
SAFE_AT_OR_ABOVE = 0.70  # surface as-is


def verify_response(pipeline: Pipeline, answer: str, question: str | None = None) -> GatedResponse:
    """Run the response through claimcheck, return a routing decision.

    Uses `profile=True` so latency is recorded — the calling layer can emit
    a metric for p50/p95 dashboards without re-instrumenting.
    """
    verdict: Verdict = pipeline.check(answer, question=question, profile=True)
    timing_ms = float(verdict.timing["total_ms"]) if verdict.timing else 0.0

    return GatedResponse(
        answer=answer,
        blocked=verdict.trust_score < BLOCK_BELOW,
        trust_score=verdict.trust_score,
        flagged_claims=verdict.flagged_claims,
        timing_ms=timing_ms,
    )


def main() -> None:
    documents = [
        "Apple released the iPhone in 2007.",
        "The Tesla Model S launched in 2012.",
        "Google's Android was first released in 2008.",
    ]
    labelled = [
        {"query": "When was the iPhone released?", "relevant_ids": ["doc0"]},
        {"query": "When did the Model S launch?", "relevant_ids": ["doc1"]},
        {"query": "When was Android released?", "relevant_ids": ["doc2"]},
    ]

    pipeline = Pipeline.from_corpus(documents, labelled, train=True, enable_nli=True, device="cpu")

    cases = [
        # (answer, question)
        ("The iPhone was released in 2007.", "When?"),
        ("The iPhone was released in 1995, the same year as Windows 95.", "When?"),  # hallucination
        ("Android first shipped in 2008. The Tesla Model S launched in 2012.", "Timeline?"),
    ]

    for answer, question in cases:
        gated = verify_response(pipeline, answer, question=question)
        verdict_label = (
            "BLOCKED"
            if gated.blocked
            else ("WARN" if gated.trust_score < WARN_BELOW else "OK")
        )
        print(f"[{verdict_label:>7}] trust={gated.trust_score:.2f}  {gated.timing_ms:.0f}ms  {gated.answer!r}")
        for c in gated.flagged_claims:
            print(f"        flagged: {c}")


if __name__ == "__main__":
    main()
