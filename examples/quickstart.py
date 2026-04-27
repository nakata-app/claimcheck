"""Quickstart: train a Pipeline on a small corpus, score an answer.

Run from the repo root:

    pip install -e ../adaptmem ../halluguard -e .
    python examples/quickstart.py

The first run downloads MiniLM (~90MB) and the NLI cross-encoder (~700MB) on
demand. CPU-only — no GPU required. Takes ~1-2 minutes end-to-end on a Mac
mini, mostly model download.
"""
from claimcheck import Pipeline


def main() -> None:
    documents = [
        "PostgreSQL ships native JSON and JSONB column types since version 9.4.",
        "MySQL added a JSON column type in version 5.7.7, released in 2015.",
        "SQLite has no native JSON column type; JSON is stored as TEXT and queried via the JSON1 extension.",
        "MongoDB stores documents in BSON, a binary-encoded superset of JSON.",
        "Redis can store JSON via the RedisJSON module but does not have a native JSON type.",
    ]

    labelled_queries = [
        {"query": "Which databases have a native JSON type?", "relevant_ids": ["doc0", "doc1"]},
        {"query": "Where is JSON stored as TEXT?", "relevant_ids": ["doc2"]},
        {"query": "What is BSON?", "relevant_ids": ["doc3"]},
    ]

    pipeline = Pipeline.from_corpus(
        documents=documents,
        labelled_queries=labelled_queries,
        train=True,
        enable_nli=True,
        device="cpu",
    )

    # 1. Answer that is fully grounded in the corpus → high trust score.
    grounded = pipeline.check(
        answer="PostgreSQL has native JSON support since version 9.4, and MySQL added JSON in 5.7.7.",
        question="Which databases have native JSON types?",
    )
    print(f"GROUNDED  trust_score={grounded.trust_score:.3f}")
    for c in grounded.supported_claims:
        print(f"  ok    {c}")
    for c in grounded.flagged_claims:
        print(f"  FLAG  {c}")

    print()

    # 2. Answer that mixes a fact with a hallucination → flagged claim.
    mixed = pipeline.check(
        answer=(
            "PostgreSQL has native JSON since 9.4. "
            "Redis has had a native JSON column since version 4.0."  # false: Redis only via RedisJSON module
        ),
        question="Which databases have native JSON types?",
    )
    print(f"MIXED     trust_score={mixed.trust_score:.3f}")
    for c in mixed.supported_claims:
        print(f"  ok    {c}")
    for c in mixed.flagged_claims:
        print(f"  FLAG  {c}")


if __name__ == "__main__":
    main()
