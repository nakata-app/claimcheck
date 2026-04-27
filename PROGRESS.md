# claimcheck — progress & resume notes

**Last updated:** 2026-04-27 (post-public-push session).

This file is the resume contract: open the repo, read this, you know the
state of play. Updated at the end of each working session.

**Public:** https://github.com/nakata-app/claimcheck (main, MIT, CI green).

## What claimcheck is

A thin orchestration layer over two siblings:
- [`adaptmem`](https://github.com/nakata-app/adaptmem) — domain-tuned bi-encoder retrieval.
- [`halluguard`](https://github.com/nakata-app/halluguard) — reverse-RAG hallucination detection.

`Pipeline.from_corpus(...)` does the wiring; `pipeline.check(answer, question)`
returns a `Verdict` with response-level `trust_score` + per-claim
flagged/supported lists.

## Where we are

```
v0.1 init               ████████████  done   (Pipeline API, save/load, 4 tests)
v0.1.1 polish           ████████████  done   (CHANGELOG, ROADMAP, CI matrix,
                                              release.yml, pre-commit, .gitignore)
v0.1.2 surface           ████████████  done   (--profile flag, check_stream, mypy
                                              --strict, comparison table, 4 examples,
                                              timing bench, Pipeline.from_daemon)
v0.2 multi-bench         ░░░░░░░░░░░░  0%     (LongMemEval/HaluEval paired runs —
                                              halluguard already ablated encoder swap,
                                              null result; pair-bench would re-confirm)
v0.3 release + recipes   ████░░░░░░░░  ~30%   (timing bench done, comparison table done,
                                              langchain example done; PyPI gating on
                                              siblings publishing first)
```

## Public API surface (v0.1.2)

```python
from claimcheck import Pipeline, Verdict

# 1. Build from corpus
pipeline = Pipeline.from_corpus(
    documents=[...],
    labelled_queries=[{"query": "...", "relevant_ids": [...]}],
    train=True,                # fine-tune retriever (off → no-train path)
    enable_nli=True,           # NLI cross-encoder verifier (off → cosine-only)
    base_model="all-MiniLM-L6-v2",
    device="cpu",
    threshold=0.55,
    entail_threshold=0.5,
    min_entail_votes=1,
)

# 2. One-shot check
verdict: Verdict = pipeline.check(
    answer="...",
    question="...",
    profile=False,             # True → populate verdict.timing
)
verdict.trust_score            # mean per-claim entailment, 0-1
verdict.flagged_claims         # list[str]
verdict.supported_claims       # list[str]
verdict.timing                 # {"total_ms", "n_claims", "ms_per_claim"} when profile=True
verdict.raw_report             # halluguard SupportReport for full detail

# 3. Streaming — flag a sentence the moment it lands
for claim in pipeline.check_stream(token_iter, question="..."):
    if claim.status.value == "HALLUCINATION_FLAG": ...

# 4. Persistence
pipeline.save("./model")
pipeline2 = Pipeline.load("./model", enable_nli=True)

# 5. Daemon-backed (one model load shared across processes)
# Prerequisite: `pip install "adaptmem[server]" && adaptmem serve`
pipeline = Pipeline.from_daemon(
    documents=[...],
    daemon_url="http://127.0.0.1:7800",
    enable_nli=True,
)
```

## Examples (runnable)

- `examples/quickstart.py` — train + grounded vs mixed answers.
- `examples/streaming.py` — fake LLM token feed, sentence-by-sentence verdict.
- `examples/middleware.py` — production gate (block / warn / pass).
- `examples/langchain_integration.py` — RunnableLambda post-step.

## How to resume (next session)

1. Read this file + `README.md` + `ROADMAP.md`.
2. Open work, in order of value:
   - **Multi-bench harness** (`benchmarks/longmemeval_paired_eval.py`) —
     measures Pipeline.check across LongMemEval-style retrieval tasks.
     Expected outcome: roughly equal to halluguard standalone since the
     encoder swap ablation already showed no F1 lift on HaluEval QA;
     the new value is **paired latency** (which `timing_bench.py` already
     gives, so this is mostly box-checking).
   - **PyPI release** — gating on adaptmem + halluguard publishing
     first. Once they're on PyPI, drop `dependencies = []` →
     `dependencies = ["adaptmem>=0.5", "halluguard>=0.3"]`.
   - **mypy strict in CI** — the local `mypy --strict claimcheck` passes;
     wire it into `.github/workflows/ci.yml` as a step after lint.
3. Atakan-gated work:
   - PyPI token (whoever owns the namespace).
   - Real-world corpus choice for the LangChain example.

## Toolchain

- Same shared venv as adaptmem + halluguard:
  `~/Projects/metis-pair/benchmarks/.venv`.
- Tests: `cd ~/Projects/claimcheck && ../metis-pair/benchmarks/.venv/bin/pytest -q`
- Current suite: **8/8 pass**, lint clean, mypy --strict clean.
- Timing bench: `python benchmarks/timing_bench.py --n 30 --out benchmarks/results_timing.json`
