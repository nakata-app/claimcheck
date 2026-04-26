# claimcheck

**Domain-tuned retrieval + zero-LLM claim verification, in one pipeline.**

`claimcheck` glues two siblings — [`adaptmem`](../adaptmem) (domain-adapted bi-encoder retrieval) and [`halluguard`](../halluguard) (reverse-RAG hallucination detection) — into a single API:

```python
from claimcheck import Pipeline

pipeline = Pipeline.from_corpus(
    documents=["..."],
    labelled_queries=[{"query": "...", "relevant_ids": [...]}],
    train=True,        # fine-tune the retriever on the labelled set
    enable_nli=True,   # add NLI verification on top of cosine retrieval
)

verdict = pipeline.check(
    answer="The user prefers PostgreSQL because it has better JSON support.",
    question="What database does the user prefer?",
)

print(verdict.trust_score)         # 0.84
print(verdict.flagged_claims)       # ["...because it has better JSON support"]
```

## What it is

A thin orchestration layer over the two siblings:

1. **adaptmem** trains a domain-adapted bi-encoder on your corpus + labelled queries.
2. **halluguard** wraps the trained encoder in a `Guard` with NLI verification, surfaces a per-claim and per-response trust score.

The same `Pipeline` object can be saved + reloaded as a unit, so a downstream service has one model directory to manage.

## Why one package

Adaptmem and halluguard are independently useful:
- adaptmem alone is a retrieval-quality lift (any domain).
- halluguard alone is a verification layer (any encoder).

But the most common deployment shape pairs them — domain-tuned retrieval for the cosine gate, claim-level NLI on top. `claimcheck` saves you the wiring.

## What it is NOT

- **Not a wrapper around any LLM.** Both siblings are explicitly LLM-free.
- **Not a vector database.** Bring your own; `claimcheck` is the *encoder + verifier* layer.
- **Not a replacement for either sibling.** If you only need adaptmem (no verification) or only halluguard (with a generic encoder), use them directly.

## Status

`v0.1` skeleton. Public API decided, integration tests landing next. The two siblings (`adaptmem` v0.4-shipped, `halluguard` v0.2-ext) are mature enough to compose; this repo just wires them.

## License

MIT.
