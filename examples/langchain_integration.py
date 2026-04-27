"""LangChain integration: wrap an LLM chain with a claimcheck verifier.

Pattern: any LangChain `Runnable` chain → `RunnableLambda` post-step that
runs `Pipeline.check` on the chain's output and either passes the answer
through, attaches a warning, or replaces it with a refusal.

Requires `langchain-core` (no LLM provider needed — this example uses a
fake chain so you don't burn API credits to see the wiring).

Install:

    pip install -e ../adaptmem ../halluguard -e .
    pip install langchain-core

Run:

    python examples/langchain_integration.py
"""
from __future__ import annotations

from typing import Any

from claimcheck import Pipeline


def build_pipeline() -> Pipeline:
    documents = [
        "The Apollo 11 mission landed on the Moon on July 20, 1969.",
        "Neil Armstrong was the first person to walk on the Moon, followed by Buzz Aldrin.",
        "The Apollo program ran from 1961 to 1972 with six successful crewed lunar landings.",
    ]
    labelled = [
        {"query": "When did Apollo 11 land?", "relevant_ids": ["doc0"]},
        {"query": "Who walked on the Moon first?", "relevant_ids": ["doc1"]},
        {"query": "How many Apollo landings were successful?", "relevant_ids": ["doc2"]},
    ]
    return Pipeline.from_corpus(documents, labelled, train=True, enable_nli=True, device="cpu")


def make_verified_chain(pipeline: Pipeline) -> Any:
    """Build `fake_llm | claimcheck_verify` as a LangChain Runnable.

    Replace `fake_llm` with any real chain (RAG, agent, raw model) — the
    verification step is a drop-in `RunnableLambda` and does not care
    what produced the answer text.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as e:
        raise SystemExit(
            "langchain-core is not installed. Run `pip install langchain-core` first."
        ) from e

    def fake_llm(input_dict: dict[str, str]) -> dict[str, str]:
        # Replace this with your actual chain (e.g. ChatOpenAI() | StrOutputParser()).
        # Hard-coded answers selected to demonstrate one supported, one mixed,
        # one fully hallucinated response.
        question = input_dict["question"]
        canned = {
            "moon_landing": "Apollo 11 landed on the Moon on July 20, 1969.",
            "first_walker": "Neil Armstrong walked on the Moon first, followed by Buzz Aldrin.",
            "fake": "Apollo 11 landed in 1955, and Yuri Gagarin was the first to walk on the Moon.",
        }
        key = input_dict.get("key", "moon_landing")
        return {"answer": canned[key], "question": question}

    def claimcheck_verify(io: dict[str, str]) -> dict[str, Any]:
        verdict = pipeline.check(io["answer"], question=io.get("question"))
        return {
            "answer": io["answer"],
            "trust_score": verdict.trust_score,
            "flagged_claims": verdict.flagged_claims,
            "supported_claims": verdict.supported_claims,
            "block": verdict.trust_score < 0.4,
        }

    chain = RunnableLambda(fake_llm) | RunnableLambda(claimcheck_verify)
    return chain


def main() -> None:
    pipeline = build_pipeline()
    chain = make_verified_chain(pipeline)

    cases = [
        {"question": "When did Apollo 11 land?", "key": "moon_landing"},
        {"question": "Who walked on the Moon first?", "key": "first_walker"},
        {"question": "When did Apollo 11 land?", "key": "fake"},
    ]

    for case in cases:
        result = chain.invoke(case)
        verdict = "BLOCK" if result["block"] else f"trust={result['trust_score']:.2f}"
        print(f"[{verdict:>10}]  Q: {case['question']!r}")
        print(f"             A: {result['answer']!r}")
        if result["flagged_claims"]:
            for c in result["flagged_claims"]:
                print(f"      flagged: {c}")
        print()


if __name__ == "__main__":
    main()
