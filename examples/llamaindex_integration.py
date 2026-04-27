"""LlamaIndex integration: verify any LlamaIndex query response with claimcheck.

Pattern: a `Pipeline` from claimcheck wraps the LlamaIndex query
engine's response — the engine retrieves + answers, claimcheck
verifies the answer against the same source documents.

Requires `llama-index-core` (no LLM provider needed for the demo —
we use a hand-rolled fake retrieval to avoid burning API credit).

Install:

    pip install -e ../adaptmem ../halluguard -e .
    pip install llama-index-core

Run:

    python examples/llamaindex_integration.py
"""
from __future__ import annotations

from typing import Any

from claimcheck import Pipeline


def build_pipeline() -> Pipeline:
    documents = [
        "PostgreSQL added native JSON column support in version 9.4 (2014).",
        "MySQL gained a JSON column type in version 5.7.7, released August 2015.",
        "MongoDB stores documents as BSON, a binary-encoded superset of JSON.",
        "Redis can store JSON only via the RedisJSON module — no native type.",
    ]
    labelled = [
        {"query": "Which databases have native JSON?", "relevant_ids": ["doc0", "doc1"]},
        {"query": "What does MongoDB use?", "relevant_ids": ["doc2"]},
    ]
    return Pipeline.from_corpus(documents, labelled, train=True, enable_nli=True, device="cpu")


def make_verified_query_engine(pipeline: Pipeline, llama_engine: Any) -> Any:
    """Wrap a LlamaIndex query engine so every response is verified.

    Pattern: same shape as the native LlamaIndex postprocessing — call
    .query(), then walk the returned Response.source_nodes (or
    response.response text directly) through claimcheck. Attach the
    verdict to the Response object as a custom attribute the caller
    can inspect.
    """
    original_query = llama_engine.query

    def verified_query(query_str: str) -> Any:
        response = original_query(query_str)
        # Get the answer text. LlamaIndex Response objects expose
        # `.response` for the natural-language answer.
        answer = getattr(response, "response", str(response))
        verdict = pipeline.check(answer, question=query_str)
        # Attach verdict to the response object — caller decides whether
        # to show, redact, or block based on `trust_score`.
        response.claimcheck_verdict = verdict  # type: ignore[attr-defined]
        return response

    llama_engine.query = verified_query  # monkey-patch
    return llama_engine


# ---- Fake LlamaIndex pieces for a self-contained demo --------------------

class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.response = text
        self.source_nodes: list[Any] = []


class _FakeEngine:
    """Minimal stand-in that returns canned answers — replace with a real
    LlamaIndex `VectorStoreIndex.as_query_engine()` in production."""

    def __init__(self) -> None:
        self._answers = {
            "moon_landing": "Apollo 11 landed on the Moon on July 20, 1969.",
            "json_support": "PostgreSQL has had native JSON since 9.4. MongoDB also has a native JSON column type.",
            "redis_native": "Redis ships a native JSON column out of the box.",
        }

    def query(self, query_str: str) -> _FakeResponse:
        # Pick the canned answer keyed off a marker in the query.
        for marker, answer in self._answers.items():
            if marker in query_str.lower():
                return _FakeResponse(answer)
        return _FakeResponse("(no answer)")


def main() -> None:
    pipeline = build_pipeline()
    engine = make_verified_query_engine(pipeline, _FakeEngine())

    cases = [
        "json_support — Which databases have native JSON?",  # mixed (Postgres OK, Mongo wrong)
        "redis_native — Does Redis support JSON natively?",  # fully wrong
    ]

    for q in cases:
        response = engine.query(q)
        v = response.claimcheck_verdict  # type: ignore[attr-defined]
        verdict_label = (
            "BLOCK" if v.trust_score < 0.4 else f"trust={v.trust_score:.2f}"
        )
        print(f"[{verdict_label:>10}]  Q: {q!r}")
        print(f"             A: {response.response!r}")
        for c in v.flagged_claims:
            print(f"      flagged: {c}")
        print()


if __name__ == "__main__":
    main()
