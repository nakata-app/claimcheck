# Changelog

All notable changes to claimcheck are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — v0.1.2

### Added
- **`Pipeline.from_daemon(documents, daemon_url)`** — daemon-backed
  factory; encoder lives in a long-lived `adaptmem serve` process,
  NLI verifier stays local (cross-encoder batches don't fit the
  daemon's `/embed` shape). `pipeline.save()` raises a clear error
  for daemon-backed pipelines (the model has nowhere local to go).
  Construction calls `/healthz` so misconfig fails loudly.
- **`Pipeline.check(profile=True)`** — populates `Verdict.timing`
  with `total_ms`, `n_claims`, `ms_per_claim`. Default off — zero
  overhead path. Lets middleware emit p50/p95 metrics without
  instrumenting the call site.
- **`Pipeline.check_stream(answer_chunks, question)`** — forwards
  to halluguard's streaming Guard; yields a Claim per completed
  sentence. Lets a service flag a hallucinated sentence mid-
  generation instead of waiting for the full answer.
- **`Verdict.timing`** field — populated only when
  `check(profile=True)` is set.
- **`benchmarks/timing_bench.py`** — measures Pipeline.check end-to-
  end across grounded / mixed / hallucinated workloads. Reports
  p50/p90/p99/mean/max ms + per-label breakdown. Uses `train=False`
  so timing isolates retrieval+verify cost (not the one-off train
  deadlock on Mac/Py3.14).
- **`PROGRESS.md`** — resume contract for next session: state, full
  public API surface, ordered open work, Atakan-gated items.
- **`examples/`** — quickstart, streaming, middleware (production
  gate pattern), langchain_integration (RunnableLambda post-step).
  Each runs as a self-contained script.
- **README "Daemon mode" section** — documents
  `Pipeline.from_daemon` and links the [adaptmem ADR](https://github.com/nakata-app/adaptmem/blob/master/docs/metis_integration.md).
- **README "How it compares to LLM-as-judge tools"** — 10-feature
  table vs Patronus / Galileo / CleanLab / Guardrails. Honest
  tradeoffs in both directions; "they're complementary" closing.
- **`release.yml`** + **`pre-commit-config.yaml`** + **`ci.yml`** —
  full CI/release matrix (3.10/3.11/3.12), tag-gated PyPI publish
  with shell guard for missing token.
- **`tool.mypy` overrides** in `pyproject.toml` — sibling editable
  installs ignored for missing-stubs (they ship `py.typed` but
  editable install hides them from mypy).
- **mypy --strict pass.** `dict[str, Any]` annotations,
  `Iterator[Any]` local in `check_stream` to satisfy
  `no-any-return`.
- **8 unit tests** (was 4) — added daemon-factory delegation, save-
  without-adaptmem guard, profile timing, streaming chunk count.

## [0.1.0] — 2026-04-27

### Added
- **`Pipeline` orchestration layer.** Single API that wires
  `adaptmem.AdaptMem` (domain-tuned bi-encoder retrieval) into
  `halluguard.Guard` (NLI-based claim verification). Two factories:
  `Pipeline.from_corpus(...)` for one-shot setup with optional
  fine-tuning, and `Pipeline(retriever, guard)` for reuse of existing
  components.
- **`pipeline.check(answer, question)`** — returns a
  `halluguard.SupportReport` enriched with the trust score and
  per-claim metadata already produced by halluguard. Question-aware
  premise threaded through when supplied.
- **`pipeline.train(labelled_queries)`** — re-fine-tune the retriever
  on the same corpus without re-instantiating the whole pipeline.
- **Editable-install workflow.** `dependencies = []` until adaptmem
  and halluguard ship to PyPI; install with
  `pip install -e ../adaptmem ../halluguard ../claimcheck`.
- **`py.typed`** (PEP 561) marker — downstream type-checkers see the
  public surface.
- **4 unit tests** covering pipeline construction, retrieval pass-
  through, NLI flagging, and the train re-entry path.

### Notes
- Pre-PyPI: `adaptmem` and `halluguard` resolve via local editable
  installs. Once both publish, they move into `dependencies = [...]`.
- API surface is intentionally narrow (`Pipeline`, `from_corpus`,
  `check`, `train`). Sibling repos remain the source of truth for
  retrieval and verification internals.
