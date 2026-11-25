"""
An upload's embedding yields to a question's model work (Decision 124).

On the emulated two-core host, while ingest embedded a 10-K's chunks in one call, a
question's retrieval model work took 6.5-9.5 s at the median instead of 0.33 s. Ingest
now embeds in slices and, before each, waits (capped) while a question's embedding,
rerank or NLI call is running. These tests pin the mechanism with a stand-in model;
the measurement is in DECISIONS_LOG.md.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

import src.data_processing.ingest_pipeline as ip
import src.vector_db.reranker as rr
from src.utils import model_priority as mp
from src.verification import nli
from tests.fixtures import ingestion_docs as fx
from tests.fixtures import reingest_docs as rd
from tests.fixtures.reingest_docs import default_bm25_is_a_stub  # noqa: F401  (fixture)


class _Device:
    def __init__(self, kind: str):
        self.type = kind


class FakeModel:
    """Stands in for bge-m3: a vector from the text, every call recorded with its time."""

    def __init__(self, device: str = "cpu"):
        self.device = _Device(device)
        self.calls: list[tuple[float, list[str], bool]] = []

    def encode(self, texts, **_kwargs):
        self.calls.append((time.monotonic(), list(texts), mp.foreground_active()))
        rows = [[float(len(t)), float(sum(map(ord, t)) % 97), 1.0] for t in texts]
        out = np.asarray(rows, dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(rr, "_get_embedding_model", lambda: fake)
    monkeypatch.delenv("INGEST_EMBED_SLICE", raising=False)
    monkeypatch.delenv("INGEST_YIELD_MAX_WAIT_S", raising=False)
    return fake


TEXTS = ["a" * 5, "b" * 40, "c" * 12, "d" * 40, "e"]


def _expected(texts):
    return FakeModel().encode(texts)


# ── the count ────────────────────────────────────────────────────────────────


def test_foreground_nests_and_is_released_on_error():
    assert not mp.foreground_active()
    with mp.foreground():
        with mp.foreground():
            assert mp.foreground_active()
        assert mp.foreground_active()
    assert not mp.foreground_active()
    with pytest.raises(RuntimeError):
        with mp.foreground():
            raise RuntimeError("model failed")
    assert not mp.foreground_active()


@pytest.mark.asyncio
async def test_waiting_ends_when_the_question_call_ends_or_at_the_cap():
    assert await mp.wait_for_foreground(5.0) < 0.05  # nothing running: no wait

    async def question(seconds):
        with mp.foreground():
            await asyncio.sleep(seconds)

    task = asyncio.create_task(question(0.3))
    await asyncio.sleep(0.01)
    waited = await mp.wait_for_foreground(5.0)
    await task
    assert 0.2 < waited < 1.0

    with mp.foreground():
        waited = await mp.wait_for_foreground(0.2)
    assert 0.2 <= waited < 0.6


# ── ingest's embedding ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_the_cpu_one_text_a_call_longest_first_back_in_input_order(model):
    out = await rr.embed_texts_background(TEXTS)
    assert np.allclose(out, _expected(TEXTS))
    assert [c[1] for c in model.calls] == [["b" * 40], ["d" * 40], ["c" * 12], ["a" * 5], ["e"]]


@pytest.mark.asyncio
async def test_on_a_gpu_every_text_in_one_call(monkeypatch, model):
    model.device = _Device("cuda")
    out = await rr.embed_texts_background(TEXTS)
    assert np.allclose(out, _expected(TEXTS))
    assert [c[1] for c in model.calls] == [TEXTS]


@pytest.mark.parametrize("value, sizes", [("2", [2, 2, 1]), ("0", [5]), ("9", [5]), ("x", [1] * 5)])
@pytest.mark.asyncio
async def test_the_slice_follows_INGEST_EMBED_SLICE(monkeypatch, model, value, sizes):
    monkeypatch.setenv("INGEST_EMBED_SLICE", value)
    out = await rr.embed_texts_background(TEXTS)
    assert np.allclose(out, _expected(TEXTS))
    assert [len(c[1]) for c in model.calls] == sizes


@pytest.mark.asyncio
async def test_a_slice_waits_for_a_question_model_call(model):
    released = {}

    async def question():
        with mp.foreground():
            await asyncio.sleep(0.4)
            released["at"] = time.monotonic()

    task = asyncio.create_task(question())
    await asyncio.sleep(0.01)
    await rr.embed_texts_background(TEXTS)
    await task
    assert model.calls[0][0] >= released["at"]
    assert not any(active for _, _, active in model.calls)


@pytest.mark.asyncio
async def test_questions_back_to_back_hold_an_upload_back_at_most_the_cap(monkeypatch, model):
    monkeypatch.setenv("INGEST_YIELD_MAX_WAIT_S", "0.1")
    started = time.monotonic()
    with mp.foreground():  # a question that never ends
        out = await rr.embed_texts_background(TEXTS)
    assert np.allclose(out, _expected(TEXTS))
    assert time.monotonic() - started < 0.1 * len(TEXTS) + 2.0


@pytest.mark.asyncio
async def test_empty_input_is_refused(model):
    with pytest.raises(ValueError):
        await rr.embed_texts_background([])


# ── a question's calls count ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_question_embedding_runs_marked(model):
    await rr.embed_texts_async(["what is the cap?"])
    assert model.calls[-1][2] is True
    assert not mp.foreground_active()


@pytest.mark.asyncio
async def test_a_rerank_runs_marked(monkeypatch):
    seen = []

    def score(pairs):
        seen.append(mp.foreground_active())
        return np.zeros(len(pairs), dtype=np.float32)

    monkeypatch.setattr(rr, "_score_pairs", score)
    await rr.rerank_pairs_async([("q", "p1"), ("q", "p2")])
    assert seen == [True] and not mp.foreground_active()


@pytest.mark.asyncio
async def test_an_nli_scoring_runs_marked(monkeypatch):
    seen = []

    def score(pairs):
        seen.append(mp.foreground_active())
        return []

    monkeypatch.setattr(nli, "score_pairs", score)
    await nli.ascore_pairs([("evidence", "claim")])
    assert seen == [True] and not mp.foreground_active()


# ── ingest uses it ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_embeds_through_the_yielding_path_by_default(monkeypatch, tmp_path):
    calls = []

    async def background(texts):
        calls.append(len(texts))
        return await rd.stub_embed(texts)

    # By path: a test elsewhere may have replaced the module in sys.modules, and ingest
    # imports the name from whichever module is there when it runs.
    monkeypatch.setattr("src.vector_db.reranker.embed_texts_background", background)
    client = await rd.new_client()
    path = fx.write_txt(tmp_path, "a.txt")
    result = await ip.index_document(str(path), "a.txt", rd.DEAL, client=client)
    assert calls == [result["chunks_created"]]  # one call: the document's chunks
