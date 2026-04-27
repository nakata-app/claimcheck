# claimcheck roadmap

Status — `v0.1.0` (April 27 2026): `Pipeline.from_corpus(...).check(...)` API shipped, py.typed marker, 4 passing unit tests, README + CHANGELOG.

claimcheck is a thin orchestration layer over two siblings:
- [`adaptmem`](https://github.com/nakata-app/adaptmem) — domain-tuned bi-encoder retrieval (R@5=0.995 on LongMemEval).
- [`halluguard`](https://github.com/nakata-app/halluguard) — reverse-RAG hallucination detection with NLI verifier.

The roadmap below picks up from a working init commit. Each milestone has a concrete exit criterion.

---

## v0.1.x — quality gates (target: 1 week)

**Goal:** make the init commit something a stranger can install and run end-to-end.

- [x] `Pipeline` API + 4 unit tests — `258ee42`.
- [x] py.typed marker.
- [ ] **GitHub Actions CI** — Python 3.10/3.11/3.12 matrix, ruff lint + pytest. Mirror the adaptmem / halluguard pattern.
- [ ] **Release workflow** — wheel build + sdist + tag-gated PyPI publish step (skipped when `PYPI_API_TOKEN` is absent so initial tags don't fail).
- [ ] **Pre-commit hooks** — ruff + standard hygiene, matching siblings.
- [ ] **CLI smoke test** — `python -m claimcheck` example or `examples/quickstart.py` so the README is verifiable.
- [ ] **End-to-end example** — runnable script that pulls a small public corpus, fine-tunes, runs `check`, prints trust score.

**Exit:** CI green on push, `git clone && pip install -e ".[dev]" && pytest` works on a fresh machine, README "Quickstart" copy-pastes cleanly.

---

## v0.2 — multi-bench harness (target: 2-3 weeks)

**Goal:** measure the *combined* pipeline on the same benchmarks the siblings already track. Today we have separate numbers — adaptmem R@5 on LongMemEval, halluguard F1 on HaluEval QA. claimcheck's value claim is "retrieval + verification together"; that needs a paired number.

- [ ] **End-to-end LongMemEval bench.** Stack `Pipeline.check()` on top of LongMemEval queries; report retrieval R@5 *and* per-claim trust scores. Honest table even if the trust-score signal is weak on memory-style queries.
- [ ] **HaluEval QA paired.** Same idea: pipeline.check on each (question, answer, corpus) triple; F1 / precision / recall against the gold hallucination labels. Compare to halluguard-standalone to measure whether the AdaptMem-tuned encoder lifts retrieval quality enough to change the verification verdict.
- [ ] **Per-stage timing.** Ship a `--profile` flag that prints `retrieval_ms`, `nli_ms`, `total_ms` per call. Rough heuristic: under 100ms per claim on CPU with small models, otherwise the pipeline isn't a viable middleware.
- [ ] **Streaming pipeline.** Wrap `Guard.check_stream` (already shipped in halluguard v0.4) so claimcheck can yield verdicts as the LLM produces tokens. Useful for live agents.

**Exit:** README has a paired benchmark table for both LongMemEval and HaluEval QA, profile snapshots at p50/p95.

---

## v0.3 — release + ecosystem (target: 1-2 weeks)

**Goal:** ship a real `pip install claimcheck` and document the integration story.

- [ ] **PyPI release.** Move `dependencies = []` → `dependencies = ["adaptmem>=X.Y", "halluguard>=A.B"]` once both publish. Tag a `v0.3.0` and let the gated publish job fire.
- [ ] **mypy --strict pass.** py.typed is the marker; this is the gate.
- [ ] **Drop-in middleware recipe.** One end-to-end LangChain or LlamaIndex example showing the pipeline as a verification step on `LLMChain` output.
- [ ] **Comparison table** in README: claimcheck (retrieval-tuned + NLI) vs LLM-as-judge (Patronus, Galileo, CleanLab) — F1, latency, cost, vendor lock-in. Honest tradeoffs only.

**Exit:** `pip install claimcheck` works, the LangChain example runs end-to-end, the comparison table is published.

---

## Non-goals (until further notice)

- **Claimcheck-internal retrieval.** Retrieval is adaptmem's job. If a feature is missing, it goes upstream.
- **Claimcheck-internal verifier.** Same for halluguard. Keep this layer as glue, not a third reimplementation.
- **LLM-as-judge fallback.** The siblings' selling point is "no LLM in the loop"; claimcheck preserves that.

---

## What needs Atakan's hand

- **PyPI release** for adaptmem and halluguard first (claimcheck depends on them via editable install today).
- **API token** in repo secrets when ready to publish.
- **Real-world corpus** for the LangChain example — a small public dataset is fine, but choosing it is a curation call.
