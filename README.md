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

## Daemon mode (`Pipeline.from_daemon`)

For deployments where you'd rather not load a SentenceTransformer in
every Python process (claimcheck + halluguard + a third service each
paying the same model cost), point claimcheck at a long-lived
[`adaptmem serve`](https://github.com/nakata-app/adaptmem#daemon-mode-adaptmem-serve)
process:

```python
from claimcheck import Pipeline

# Daemon must be running: `adaptmem serve --port 7800`
pipeline = Pipeline.from_daemon(
    documents=[...],
    daemon_url="http://127.0.0.1:7800",
    enable_nli=True,   # NLI verifier still runs in-process
)
verdict = pipeline.check("an answer", question="...")
```

The encoder hop crosses HTTP; cosine search and NLI verification stay
local. `pipeline.save()` is not supported for daemon-backed pipelines
(the model lives in the daemon). `Pipeline.from_daemon` calls
`/healthz` first so a misconfigured URL fails loudly at construction
time, not deep inside the first `.check()`.

## How it compares to LLM-as-judge tools

The closest commercial / open-source category is "LLM-as-judge" — a separate large-model call grades each claim. Claimcheck is the **no-LLM-judge** branch: a deterministic NLI cross-encoder + retrieval-augmented gate. The tradeoffs are real and shape what you should use it for.

| Feature | claimcheck | LLM-as-judge<br/>(Patronus, Galileo, CleanLab, Guardrails) |
|---|---|---|
| Judge model | NLI cross-encoder (≈90M params, local) | LLM call (GPT-4 / Claude / open-source 7-70B) |
| Cost per claim | $0 (local CPU/GPU) | $0.001-0.05 (API token cost) |
| Latency per claim (CPU) | 50-200ms | 500-3000ms (network + LLM inference) |
| Determinism | yes — same input → same score | partial — depends on model temperature, version, drift |
| Vendor lock-in | none | judge model API, often a single provider |
| Audit trail | claim → cited chunk → entail/contradict score | claim → judge prompt + judge response (opaque reasoning) |
| Domain tuning | yes — retriever fine-tuned on your corpus (adaptmem) | usually no — judge is generic |
| Customising the judge | swap any HuggingFace cross-encoder | retrain or fine-tune the LLM (rarely practical) |
| Streaming | yes — sentence-by-sentence verdict (`check_stream`) | yes for some, but each judge call is heavier |
| Privacy | data stays local | claims and context sent to judge provider |
| Best at | budget-bound CI/middleware, per-domain accuracy, audit | general-purpose judgement, "did the model do something obviously bad" |

**When claimcheck wins:**
- High-throughput middleware where per-claim cost matters (every chatbot turn checked).
- Privacy-bound deployments (medical, legal, internal tools) where claims can't leave the perimeter.
- Domain-specific RAG where a tuned retriever beats a generic LLM judge that doesn't know your jargon.
- Streaming UX where users see the verdict as the LLM types.

**When LLM-as-judge wins:**
- Open-ended quality assessment ("is this answer helpful, safe, polite?") that isn't really a hallucination check.
- Few-shot domains with no labelled training queries to fine-tune the retriever.
- One-off audits where a $0.05 model call is cheaper than building infrastructure.

The two are **complementary**, not exclusive. A reasonable production stack runs claimcheck in-line on every response (cheap, deterministic, blocks the worst), and an LLM judge in a sampled audit (expensive, broader, catches subtler issues).

## Status

`v0.1.1` shipped. Public API decided (`Pipeline.from_corpus`, `check`, `check_stream`, `check(profile=True)`, `save/load`), 6 unit tests passing, mypy --strict clean, CI matrix on 3.10/3.11/3.12. The two siblings (`adaptmem` v0.4-shipped, `halluguard` v0.2-ext-shipped) are mature enough to compose; this repo just wires them.

Pre-PyPI: install via local editable until siblings publish.
```bash
pip install -e ../adaptmem ../halluguard ../claimcheck
```

## License

MIT.
