"""Pipeline tests — model-free.

Stubs both siblings (AdaptMem and Guard) so the test suite doesn't pull
weights or trigger contrastive fine-tunes. Validates the orchestration
contract (what gets called, what comes out).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# ---- Shim: stub adaptmem + halluguard before claimcheck imports them ------
class _StubAdaptMem:
    instances: list = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)
        self.trained = False

    def train(self, corpus, labelled, **k):
        self.trained = True
        self.corpus_arg = corpus
        self.labelled_arg = labelled
        return {"n_pairs": len(labelled), "runtime_s": 0.0, "n_steps": 1}

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path):
        am = cls()
        am.trained = True
        return am


class _StubReport:
    def __init__(self, claims):
        self.answer = "stub"
        self.threshold = 0.5
        self.claims = claims
        self.trust_score = sum(c.support_score for c in claims) / max(1, len(claims))


class _StubClaim:
    def __init__(self, text, supported=True):
        self.text = text
        self.support_score = 0.9 if supported else 0.1
        self.status = types.SimpleNamespace(value="SUPPORTED" if supported else "HALLUCINATION_FLAG")


class _StubGuard:
    @classmethod
    def from_adaptmem(cls, am, **kwargs):
        g = cls()
        g.am = am
        g.kwargs = kwargs
        return g

    def check(self, answer, question=None):
        # 2 claims: first supported, second flagged.
        return _StubReport(
            claims=[
                _StubClaim("first claim", supported=True),
                _StubClaim("second claim", supported=False),
            ]
        )

    def check_stream(self, chunks, question=None):
        # Yield one claim per chunk so the test can count.
        for chunk in chunks:
            yield _StubClaim(chunk, supported=True)


def _install_stubs():
    sys.modules["adaptmem"] = types.SimpleNamespace(AdaptMem=_StubAdaptMem)
    sys.modules["adaptmem.miner"] = types.SimpleNamespace(
        CorpusEntry=lambda **k: types.SimpleNamespace(**k)
    )
    sys.modules["halluguard"] = types.SimpleNamespace(Guard=_StubGuard)
    sys.modules["halluguard.verifier"] = types.SimpleNamespace(
        NLIVerifier=lambda *a, **k: object()
    )
    # Also stub sentence_transformers for the no-train path.
    class _STStub:
        def __init__(self, *a, **k): pass
        def encode(self, texts, **k):
            import numpy as np
            return np.zeros((len(texts), 4), dtype="float32")
    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=_STStub)


_install_stubs()
from claimcheck import Pipeline, Verdict  # noqa: E402


def test_pipeline_train_path_invokes_adaptmem_train():
    docs = ["doc1", "doc2"]
    labelled = [{"query": "q1", "relevant_ids": ["doc0"]}]
    Pipeline.from_corpus(docs, labelled, train=True, enable_nli=False)
    am = _StubAdaptMem.instances[-1]
    assert am.trained is True
    assert am.labelled_arg == labelled


def test_pipeline_check_returns_verdict_with_trust_score():
    p = Pipeline.from_corpus(["doc1"], [{"query": "q", "relevant_ids": ["doc0"]}], train=True)
    v = p.check("the answer here", question="what?")
    assert isinstance(v, Verdict)
    assert 0.0 <= v.trust_score <= 1.0
    # First stub claim supported, second flagged → lists split correctly
    assert v.flagged_claims == ["second claim"]
    assert v.supported_claims == ["first claim"]
    assert v.raw_report is not None


def test_pipeline_passes_through_guard_kwargs():
    p = Pipeline.from_corpus(
        ["d"], [{"query": "q", "relevant_ids": ["doc0"]}],
        train=True, threshold=0.42, entail_threshold=0.7, min_entail_votes=2,
    )
    assert p._guard.kwargs["threshold"] == 0.42
    assert p._guard.kwargs["entail_threshold"] == 0.7
    assert p._guard.kwargs["min_entail_votes"] == 2


def test_pipeline_save_then_load_round_trip(tmp_path: Path):
    p = Pipeline.from_corpus(["d"], [{"query": "q", "relevant_ids": ["doc0"]}], train=True)
    p.save(tmp_path / "model")
    p2 = Pipeline.load(tmp_path / "model", enable_nli=False)
    assert p2._adaptmem.trained is True


def test_pipeline_check_stream_yields_claims_per_chunk():
    p = Pipeline.from_corpus(["doc1"], [{"query": "q", "relevant_ids": ["doc0"]}], train=True)
    chunks = ["sentence one. ", "sentence two."]
    claims = list(p.check_stream(chunks, question="q"))
    assert len(claims) == 2
    assert claims[0].text == "sentence one. "
    assert claims[1].text == "sentence two."


def test_pipeline_from_daemon_delegates_to_guard_factory(monkeypatch):
    """`Pipeline.from_daemon` should call `Guard.from_daemon` and skip AdaptMem."""
    captured: dict = {}

    def fake_guard_from_daemon(documents, daemon_url, **kwargs):
        captured["documents"] = documents
        captured["daemon_url"] = daemon_url
        captured["kwargs"] = kwargs
        g = _StubGuard()
        g.kwargs = kwargs
        return g

    # Patch the Guard symbol that Pipeline.from_daemon reaches for.
    import sys
    sys.modules["halluguard"].Guard.from_daemon = staticmethod(fake_guard_from_daemon)

    try:
        p = Pipeline.from_daemon(
            ["doc1", "doc2"],
            daemon_url="http://example.invalid:7800",
            threshold=0.42,
        )
        assert captured["documents"] == ["doc1", "doc2"]
        assert captured["daemon_url"] == "http://example.invalid:7800"
        assert captured["kwargs"]["threshold"] == 0.42
        # Daemon path leaves _adaptmem as None.
        assert p._adaptmem is None
    finally:
        # Don't leak the stub into other tests.
        del sys.modules["halluguard"].Guard.from_daemon


def test_pipeline_save_raises_when_no_adaptmem():
    p = Pipeline(adaptmem=None, guard=_StubGuard())
    with pytest.raises(RuntimeError, match="Daemon-backed"):
        p.save("/tmp/should_not_be_written")


def test_pipeline_check_profile_flag_populates_timing():
    p = Pipeline.from_corpus(["doc1"], [{"query": "q", "relevant_ids": ["doc0"]}], train=True)
    v_off = p.check("answer", question="q")
    assert v_off.timing is None

    v_on = p.check("answer", question="q", profile=True)
    assert v_on.timing is not None
    assert v_on.timing["n_claims"] == 2.0
    assert v_on.timing["total_ms"] >= 0.0
    assert v_on.timing["ms_per_claim"] >= 0.0
