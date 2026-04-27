# Contributing to claimcheck

Thanks for considering a contribution. claimcheck is a thin
orchestration layer — most behaviour lives in the two siblings
([`adaptmem`](https://github.com/nakata-app/adaptmem) and
[`halluguard`](https://github.com/nakata-app/halluguard)). Bug fixes
and new factories land easily here; behaviour changes usually belong
in a sibling.

## Quickstart for a local dev loop

```bash
git clone https://github.com/nakata-app/claimcheck.git
cd claimcheck
python -m venv .venv && source .venv/bin/activate
# Siblings (adaptmem, halluguard) come from PyPI as regular dependencies.
pip install -e ".[dev]"
pre-commit install
```

For local sibling development (editing adaptmem or halluguard side-by-side):

```bash
pip install -e ../adaptmem ../halluguard -e ".[dev]"
```

## What we run before every commit

```bash
ruff check claimcheck tests
mypy --strict claimcheck
pytest -q
```

CI runs the same three on Python 3.10 / 3.11 / 3.12, with the siblings
checked out from `nakata-app/{adaptmem,halluguard}` and editable-
installed.

## What lands easily

- Bug fixes with a regression test.
- New `Pipeline.from_*` factories (e.g. `from_redis`, `from_pgvector`)
  that compose the siblings differently.
- New examples under `examples/` (each runnable, each self-contained,
  no API keys).
- Production-pattern improvements (e.g. richer `Verdict.timing`).

## What probably belongs in a sibling instead

- **Retrieval changes** → `adaptmem`. Encoder swap, training tricks,
  ANN backends, on-disk indexes.
- **Verification changes** → `halluguard`. NLI prompts, vote policies,
  segmentation, span-level labels.
- **New benchmark harnesses** → whichever sibling produces the number.
  Pair-benchmarks live here only when claimcheck composes the two
  in a way the siblings can't reach individually.

## Style

- Match the existing code. Type hints on public surfaces; no
  speculative abstractions; comments only for non-obvious WHY.
- Lazy imports of siblings — claimcheck shouldn't crash at import
  time if a user only needs one sibling.

## Reporting bugs

GitHub Issues. Include:
- Python version + OS.
- adaptmem + halluguard versions (or git SHAs if editable-installed).
- A minimum reproduction.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md). Don't open a public issue for an
unpatched vulnerability.
