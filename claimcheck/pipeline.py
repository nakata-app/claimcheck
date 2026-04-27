"""Pipeline: orchestrate adaptmem (retrieval) + halluguard (verification).

Lazy imports of both siblings so claimcheck doesn't impose a hard dependency
on either at install time — the user opts in by training a Pipeline. Lets
this package install in environments that only have one sibling available
(e.g. retrieval-only or verification-only deployments).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Verdict:
    """High-level result of a claim check.

    Wraps halluguard's SupportReport with a single response-level
    `trust_score` and a list of flagged claim texts — the two fields a
    middleware actually routes on.
    """

    answer: str
    trust_score: float
    flagged_claims: list[str] = field(default_factory=list)
    supported_claims: list[str] = field(default_factory=list)
    raw_report: Any = None  # the underlying SupportReport, for callers that want the full detail
    timing: dict[str, float] | None = None  # populated when Pipeline.check(profile=True)


class Pipeline:
    """Orchestrates AdaptMem-tuned retrieval + halluguard verification.

    Typical use:
        pipeline = Pipeline.from_corpus(documents, labelled_queries, train=True)
        verdict = pipeline.check(answer, question="...")
        if verdict.trust_score < 0.7: ...

    Implementation note: lazy imports of adaptmem / halluguard. The
    Pipeline only has direct dependencies on whichever sibling it actually
    invokes for a given configuration.
    """

    def __init__(self, adaptmem: Any, guard: Any):
        self._adaptmem = adaptmem
        self._guard = guard

    @classmethod
    def from_corpus(
        cls,
        documents: list[str],
        labelled_queries: list[dict[str, Any]] | None = None,
        *,
        train: bool = True,
        enable_nli: bool = True,
        base_model: str = "all-MiniLM-L6-v2",
        device: str | None = "cpu",
        threshold: float = 0.55,
        entail_threshold: float = 0.5,
        min_entail_votes: int = 1,
    ) -> "Pipeline":
        """Build a Pipeline from a raw corpus + (optional) labelled queries.

        - When `train=True` and `labelled_queries` is supplied, runs adaptmem
          fine-tune. Otherwise uses the `base_model` as-is.
        - When `enable_nli=True`, halluguard's NLI cross-encoder is added.
        """
        from adaptmem import AdaptMem
        from halluguard import Guard
        from halluguard.verifier import NLIVerifier

        am = AdaptMem(base_model=base_model, device=device)
        if train and labelled_queries:
            corpus = [{"id": f"doc{i}", "text": d} for i, d in enumerate(documents)]
            am.train(corpus=corpus, labelled=labelled_queries)
        else:
            # No-train path: encode the corpus with the base model directly.
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(base_model, device=device) if device else SentenceTransformer(base_model)
            am._model = model
            from adaptmem.miner import CorpusEntry
            am._corpus = [CorpusEntry(id=f"doc{i}", text=d) for i, d in enumerate(documents)]
            am._embeddings = model.encode(
                list(documents),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )

        verifier = NLIVerifier() if enable_nli else None
        guard = Guard.from_adaptmem(
            am,
            threshold=threshold,
            verifier=verifier,
            entail_threshold=entail_threshold,
            min_entail_votes=min_entail_votes,
        )
        return cls(adaptmem=am, guard=guard)

    def check(
        self,
        answer: str,
        question: str | None = None,
        *,
        profile: bool = False,
    ) -> Verdict:
        """Score `answer` against the indexed corpus.

        Returns a Verdict with response-level trust_score and the list of
        flagged claim texts. The full SupportReport is in `verdict.raw_report`.

        When `profile=True`, populates `verdict.timing` with `total_ms`,
        `n_claims`, and `ms_per_claim` so middleware can trace latency
        without instrumenting the call site.
        """
        if profile:
            import time
            t0 = time.perf_counter()
            report = self._guard.check(answer, question=question)
            total_ms = (time.perf_counter() - t0) * 1000.0
        else:
            report = self._guard.check(answer, question=question)

        flagged: list[str] = []
        supported: list[str] = []
        for c in report.claims:
            (flagged if c.status.value == "HALLUCINATION_FLAG" else supported).append(c.text)

        timing: dict[str, float] | None = None
        if profile:
            n_claims = len(report.claims)
            timing = {
                "total_ms": round(total_ms, 2),
                "n_claims": float(n_claims),
                "ms_per_claim": round(total_ms / n_claims, 2) if n_claims else 0.0,
            }

        return Verdict(
            answer=answer,
            trust_score=report.trust_score,
            flagged_claims=flagged,
            supported_claims=supported,
            raw_report=report,
            timing=timing,
        )

    def check_stream(
        self,
        answer_chunks: Iterable[str],
        question: str | None = None,
    ) -> Iterator[Any]:
        """Streaming variant of `check`.

        Feeds the LLM's output to halluguard's streaming Guard sentence-by-
        sentence; yields a Claim object as each completed sentence is
        verified. Lets a middleware flag a hallucinated sentence mid-
        generation instead of waiting for the full answer.

        The yielded objects are halluguard Claims (not Verdicts) — their
        `.status`, `.support_score`, and `.entail_votes` are populated.
        Aggregate them yourself if you need a final response-level
        trust_score.
        """
        result: Iterator[Any] = self._guard.check_stream(answer_chunks, question=question)
        return result

    def save(self, path: str | Path) -> None:
        """Persist the trained pipeline to disk. Reloadable via Pipeline.load()."""
        if self._adaptmem is None:
            raise RuntimeError(
                "Pipeline.save() is only supported for in-process pipelines "
                "(Pipeline.from_corpus). Daemon-backed pipelines have nothing "
                "to persist locally — the encoder lives in the daemon."
            )
        self._adaptmem.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        enable_nli: bool = True,
        threshold: float = 0.55,
        entail_threshold: float = 0.5,
        min_entail_votes: int = 1,
    ) -> "Pipeline":
        from adaptmem import AdaptMem
        from halluguard import Guard
        from halluguard.verifier import NLIVerifier

        am = AdaptMem.load(path)
        verifier = NLIVerifier() if enable_nli else None
        guard = Guard.from_adaptmem(
            am,
            threshold=threshold,
            verifier=verifier,
            entail_threshold=entail_threshold,
            min_entail_votes=min_entail_votes,
        )
        return cls(adaptmem=am, guard=guard)

    @classmethod
    def from_daemon(
        cls,
        documents: list[str],
        daemon_url: str = "http://127.0.0.1:7800",
        *,
        timeout_s: float = 10.0,
        enable_nli: bool = True,
        threshold: float = 0.55,
        entail_threshold: float = 0.5,
        min_entail_votes: int = 1,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ) -> "Pipeline":
        """Build a Pipeline whose encoder is an `adaptmem serve` daemon.

        Same surface as `from_corpus(..., train=False)`, but the
        SentenceTransformer load happens once inside the daemon —
        across many Pipeline instances if you want. Saves the per-
        process model load cost.

        Trade-off: every retrieval pays one HTTP round-trip to the
        daemon for query encoding. On localhost this is ~1ms; for
        latency-sensitive paths, prefer `from_corpus(...)` which keeps
        the encoder in the same process.

        NLI verifier (the slow part) stays local because cross-encoder
        rerank batches don't fit the daemon's `/embed` shape.
        """
        from halluguard import Guard
        from halluguard.verifier import NLIVerifier

        verifier = NLIVerifier() if enable_nli else None
        guard = Guard.from_daemon(
            documents,
            daemon_url=daemon_url,
            timeout_s=timeout_s,
            threshold=threshold,
            verifier=verifier,
            entail_threshold=entail_threshold,
            min_entail_votes=min_entail_votes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        # The retriever side carries everything we need; AdaptMem is only
        # needed when the caller wants `pipeline.train()` later, which the
        # daemon path doesn't support today.
        return cls(adaptmem=None, guard=guard)
