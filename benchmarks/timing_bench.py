"""End-to-end timing bench: how long does Pipeline.check take in practice.

Measures retrieval + NLI verification per claim on a synthetic but
realistic workload — a small fixed corpus, a stream of (question,
answer) pairs spanning grounded / mixed / hallucinated, and timing
captured via the built-in `profile=True` flag.

Reports p50 / p90 / p99 / mean / max in milliseconds, plus per-claim
breakdown. The numbers a middleware operator would put on a dashboard.

Run from the repo root:

    pip install -e ../adaptmem ../halluguard -e ".[dev]"
    python benchmarks/timing_bench.py --n 50 --out benchmarks/results_timing.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from claimcheck import Pipeline


CORPUS = [
    "PostgreSQL added native JSON in 9.4 and JSONB shortly after.",
    "MySQL gained a JSON column type in version 5.7.7 (2015).",
    "SQLite has no native JSON type; the JSON1 extension queries TEXT.",
    "MongoDB stores documents as BSON, a binary JSON superset.",
    "Redis stores JSON via the RedisJSON module, not a native type.",
    "ChromaDB is an embedding-first vector database with a Python client.",
    "Pinecone is a managed vector database serving billion-scale indexes.",
    "Qdrant is an open-source vector database written in Rust.",
    "FAISS is a similarity search library from Meta AI for dense vectors.",
    "Annoy is Spotify's approximate nearest-neighbor library.",
]

LABELLED = [
    {"query": "Which databases have native JSON?", "relevant_ids": ["doc0", "doc1"]},
    {"query": "Where is JSON stored as TEXT?", "relevant_ids": ["doc2"]},
    {"query": "Vector databases?", "relevant_ids": ["doc5", "doc6", "doc7"]},
    {"query": "Similarity search libraries?", "relevant_ids": ["doc8", "doc9"]},
]

WORKLOAD = [
    # (question, answer, expected_label)
    (
        "Which databases have native JSON?",
        "PostgreSQL has native JSON since 9.4. MySQL added JSON in 5.7.7.",
        "grounded",
    ),
    (
        "Vector databases?",
        "Pinecone is a managed vector database. ChromaDB is embedding-first. "
        "Cassandra is a vector-native key-value store.",  # Cassandra: hallucinated
        "mixed",
    ),
    (
        "Where is JSON stored as TEXT?",
        "Postgres stores all JSON as TEXT. MySQL has no JSON support whatsoever.",  # both wrong
        "hallucinated",
    ),
    (
        "Similarity search libraries?",
        "FAISS is Meta AI's library. Annoy is from Spotify.",
        "grounded",
    ),
    (
        "Vector databases?",
        "Qdrant is written in Rust. Weaviate ships its own LLM internally for query rewriting.",  # Weaviate llm: false
        "mixed",
    ),
]


def run(n: int, warmup: int = 2) -> dict:
    print(f"building Pipeline (cpu, MiniLM raw + NLI cross-encoder)…", flush=True)
    t_build = time.perf_counter()
    # train=False: no fine-tune — measures the runtime cost users see when
    # they ship the pipeline, not the one-off training cost. Mac/Py3.14
    # has a known sentence-transformers train-loop deadlock; this path
    # also dodges that.
    pipeline = Pipeline.from_corpus(
        CORPUS,
        LABELLED,
        train=False,
        enable_nli=True,
        device="cpu",
    )
    build_s = time.perf_counter() - t_build
    print(f"  built in {build_s:.1f}s", flush=True)

    # Warm up: trigger any lazy NLI model load + tokenizer cache.
    for _ in range(warmup):
        pipeline.check(WORKLOAD[0][1], question=WORKLOAD[0][0], profile=False)

    timings_ms: list[float] = []
    per_label_ms: dict[str, list[float]] = {"grounded": [], "mixed": [], "hallucinated": []}
    n_claims_total = 0

    for i in range(n):
        question, answer, label = WORKLOAD[i % len(WORKLOAD)]
        v = pipeline.check(answer, question=question, profile=True)
        assert v.timing is not None
        ms = float(v.timing["total_ms"])
        timings_ms.append(ms)
        per_label_ms[label].append(ms)
        n_claims_total += int(v.timing["n_claims"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}  last={ms:.0f}ms", flush=True)

    timings_ms.sort()
    p50 = statistics.median(timings_ms)
    p90 = timings_ms[int(0.90 * len(timings_ms))]
    p99 = timings_ms[min(int(0.99 * len(timings_ms)), len(timings_ms) - 1)]
    mean = statistics.fmean(timings_ms)

    return {
        "n_calls": n,
        "n_claims_total": n_claims_total,
        "ms_per_call_p50": round(p50, 2),
        "ms_per_call_p90": round(p90, 2),
        "ms_per_call_p99": round(p99, 2),
        "ms_per_call_mean": round(mean, 2),
        "ms_per_call_max": round(max(timings_ms), 2),
        "ms_per_claim_mean": round(mean / max(1, n_claims_total / n), 2),
        "build_time_s": round(build_s, 2),
        "device": "cpu",
        "encoder": "all-MiniLM-L6-v2 (raw, no fine-tune for timing isolation)",
        "verifier": "cross-encoder/nli-deberta-v3-base (default)",
        "per_label_p50_ms": {k: round(statistics.median(v), 2) if v else 0.0 for k, v in per_label_ms.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of check() calls (post-warmup)")
    ap.add_argument("--warmup", type=int, default=2, help="warmup calls (not counted)")
    ap.add_argument("--out", default="benchmarks/results_timing.json")
    args = ap.parse_args()

    result = run(args.n, args.warmup)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\n→ wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
