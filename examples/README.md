# claimcheck examples

Four runnable scripts covering the typical integration paths. Each is
self-contained — no external services, no API keys.

| File | What it shows | Extra deps |
|---|---|---|
| `quickstart.py` | Train a Pipeline on a 5-doc corpus, score one grounded answer + one mixed answer | none |
| `streaming.py` | `check_stream` — flag a hallucinated sentence the moment it lands | none |
| `middleware.py` | Production gate pattern — block / warn / pass on `trust_score`, with `profile=True` timing | none |
| `langchain_integration.py` | LangChain `RunnableLambda` post-step that verifies any chain's output | `langchain-core` |
| `llamaindex_integration.py` | LlamaIndex query-engine wrapper that verifies every `.query()` response | `llama-index-core` (demo uses a fake engine to skip API costs) |
| `daemon_integration.py` | `Pipeline.from_daemon` — encoder lives in a long-lived `adaptmem serve` process | `adaptmem[server]`, `requests`, daemon running |
| `slack_bot.py` | Slash command (`/factcheck`) + @mention handler — verify any teammate-posted claim against your team's docs | `slack_bolt`, Slack tokens |
| `ecommerce_product_rag.py` | Customer asks → LLM answers → claimcheck guards against invented SKUs / prices / attributes | none |

## Prerequisites

```bash
# From PyPI (siblings come along automatically):
pip install claimcheck

# Or, for local development across all three repos:
pip install -e ../adaptmem ../halluguard -e .
```

The first run downloads MiniLM (≈90MB) and the NLI cross-encoder (≈700MB)
on demand. CPU-only, no GPU required. Total wall-clock for `quickstart.py`
on a Mac mini: 1-2 minutes (mostly model download), then sub-second after.

## Mental model

```
documents ─┐
            ├─→ adaptmem (encoder fine-tune) ─→ trained encoder ─┐
labelled ──┘                                                      │
                                                                  ▼
answer ────────────────────────────────────→ halluguard (NLI gate) ─→ Verdict
                                                                  ▲
                                                          retrieval│
                                                          + verify │
```

`Pipeline.from_corpus(...)` does both training steps and returns a single
object. `pipeline.check(answer)` does the retrieval + verify path on every
call.

## Choosing a threshold

`trust_score` is the mean per-claim entailment, in [0, 1]:

- `< 0.4` — most claims unsupported. Block by default.
- `0.4 – 0.7` — mixed. Warn or surface the flagged subset.
- `≥ 0.7` — supported.

These are starting points, not commandments. The right cutoff depends on
your tolerance for false flags vs missed hallucinations — calibrate on a
held-out set.
