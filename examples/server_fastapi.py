"""Run claimcheck as an HTTP service — minimal FastAPI server.

Use case: you've fine-tuned a Pipeline once on your corpus, and now
want to expose `pipeline.check(answer)` over HTTP so any process
(non-Python included) can call it. The adaptmem daemon handles the
encoder side; this example wraps the *full* claimcheck pipeline
(retrieval + NLI + verdict aggregation) into a single endpoint.

Endpoints:
    GET  /healthz                 — liveness
    GET  /readyz                  — pipeline initialised + NLI loaded
    POST /verify                  — body: {answer, question} → Verdict

Install:

    pip install -e ../adaptmem ../halluguard -e .
    pip install fastapi uvicorn

Run:

    python examples/server_fastapi.py
    # Or directly with uvicorn for prod:
    uvicorn examples.server_fastapi:app --host 0.0.0.0 --port 8000

For deployment behind a reverse proxy / k8s, see the patterns in
adaptmem/Dockerfile and adaptmem/charts/adaptmem/.
"""
from __future__ import annotations

from typing import Any

from claimcheck import Pipeline


# ---- Pipeline factory: keep the corpus + labelled queries here -----------

def _build_pipeline() -> Pipeline:
    """Replace these placeholders with your own corpus + labelled set.

    For larger corpora, train + save once with `pipeline.save(path)` and
    reload with `Pipeline.load(path)` here so the service starts fast.
    """
    documents = [
        "PostgreSQL has native JSON since version 9.4 (2014).",
        "MySQL added a JSON column type in version 5.7.7 (2015).",
        "MongoDB stores documents in BSON, a binary JSON superset.",
        "Redis stores JSON via the RedisJSON module — no native column type.",
    ]
    labelled = [
        {"query": "Which databases have native JSON?", "relevant_ids": ["doc0", "doc1"]},
        {"query": "What does MongoDB use?", "relevant_ids": ["doc2"]},
    ]
    return Pipeline.from_corpus(documents, labelled, train=True, enable_nli=True, device="cpu")


# ---- FastAPI app ---------------------------------------------------------

def _build_app() -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as e:
        raise SystemExit(
            "Server example needs FastAPI. Run `pip install fastapi uvicorn`."
        ) from e

    class VerifyRequest(BaseModel):
        answer: str
        question: str | None = None

    class VerifyResponse(BaseModel):
        trust_score: float
        flagged_claims: list[str]
        supported_claims: list[str]

    state: dict[str, Any] = {"pipeline": None}
    app = FastAPI(title="claimcheck-server", description="claimcheck as an HTTP service")

    @app.on_event("startup")
    def _build_on_startup() -> None:
        # Build the pipeline lazily so /healthz answers fast even before
        # the encoder has loaded.
        state["pipeline"] = _build_pipeline()

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> dict[str, bool]:
        return {"ready": state["pipeline"] is not None}

    @app.post("/verify", response_model=VerifyResponse)
    def verify(req: VerifyRequest) -> VerifyResponse:
        pipeline = state["pipeline"]
        if pipeline is None:
            raise HTTPException(status_code=503, detail="pipeline not yet initialised")
        verdict = pipeline.check(req.answer, question=req.question)
        return VerifyResponse(
            trust_score=verdict.trust_score,
            flagged_claims=verdict.flagged_claims,
            supported_claims=verdict.supported_claims,
        )

    return app


app = _build_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit("`pip install uvicorn`") from e

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
