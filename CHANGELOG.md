# Changelog

All notable changes to claimcheck are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
