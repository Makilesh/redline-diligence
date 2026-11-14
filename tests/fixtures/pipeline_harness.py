"""
Offline harness for driving the real query pipeline end to end.

Everything between the HTTP boundary and the model weights runs for real: the
LangGraph graph, every agent node, the LLM wrapper and model ladder, the budget
tracker, ingestion (extract, chunk, classify, PII, risk), Qdrant (in memory),
hybrid search, RRF, reranking, context expansion and claim verification. Only
the things that need a network, a GPU or a database are replaced, each at the
outermost seam available:

    litellm.acompletion          -> ScriptedLLM (routes by the prompt's output
                                    contract, records every call)
    asyncpg.create_pool          -> refuses, so BudgetTracker takes its own
                                    documented in-memory fallback
    bge-m3 / bge-reranker / BM25 -> small deterministic lexical models
    Qdrant server                -> AsyncQdrantClient(":memory:")

The seams are listed once, in SEAMS, and patched through _patch_seam. When a
refactor moves one, the failure names the seam and this file — fix it here and
every full-graph test follows. Tests themselves must assert on observable
behaviour only (final state, the `stage` events stream_query emits, what the
scripted LLM was asked and how often), never on these internals.

The lexical models make relevance a function of shared vocabulary, which is all
the tests need: a question that shares its content words with a passage scores
near 1.0, one that shares none scores 0.0, and the quality gate's thresholds
(0.05 refuse, 0.30 pass) sit between them with room on both sides.
"""

from __future__ import annotations

import html
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

# (module path, attribute) for everything the harness replaces. Private names are
# used only where no public seam exists; see the module docstring.
SEAMS = {
    "llm": ("litellm", "acompletion"),
    "postgres": ("asyncpg", "create_pool"),
    "embedding_model": ("src.vector_db.reranker", "_get_embedding_model"),
    "reranker_model": ("src.vector_db.reranker", "_get_reranker_model"),
    "bm25_model": ("src.vector_db.hybrid_search", "_get_bm25_model"),
    "qdrant_singleton": ("src.vector_db.qdrant_client", "_qdrant_client"),
    # The query-time decisions (answerability gate) reach Laya through
    # laya_client.adecide_batch, which looks this name up at call time.
    "laya": ("src.decisions.laya_client", "decide_batch"),
    # Ingestion imported the function by name, so it holds its own reference.
    "laya_ingest": ("src.decisions.ingest_signals", "decide_batch"),
    "nli": ("src.verification.nli", "ascore_pairs"),
}

# Long enough to pass BudgetTracker's fragment filter; obviously not credentials.
FAKE_API_KEYS = (
    "offline-suite-key-000000000000000000000001",
    "offline-suite-key-000000000000000000000002",
)

DEAL = "deal-alpha"
OTHER_DEAL = "deal-beta"


class HarnessSeamMoved(AssertionError):
    """A production seam the harness patches no longer exists under that name."""


def _patch_seam(monkeypatch: pytest.MonkeyPatch, seam: str, value: Any) -> None:
    import importlib

    module_path, attribute = SEAMS[seam]
    module = importlib.import_module(module_path)
    if not hasattr(module, attribute):
        raise HarnessSeamMoved(
            f"{module_path}.{attribute} no longer exists. The offline pipeline harness "
            f"replaces it (seam '{seam}'); update SEAMS in tests/fixtures/pipeline_harness.py "
            f"to wherever that responsibility moved."
        )
    monkeypatch.setattr(module, attribute, value)


# ==============================================================================
# Deterministic lexical stand-ins for the three local models
# ==============================================================================

_STOPWORDS = frozenset(
    """
    a an and any are as at be by did do does for from had has have how in into is it its
    much many of on or per that the their this to under was were what when where which
    who whom whose why will with would
    """.split()
)
_WORD = re.compile(r"[a-z0-9]+")
_VECTOR_SIZE = 1024


def content_tokens(text: str) -> list[str]:
    """Lower-cased words of three or more characters, minus stopwords."""
    return [t for t in _WORD.findall((text or "").lower()) if len(t) >= 3 and t not in _STOPWORDS]


class LexicalEmbeddingModel:
    """Stands in for bge-m3: a hashed bag of words, L2-normalised to 1024 dims."""

    def encode(self, texts, **_kwargs) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(_VECTOR_SIZE, dtype=np.float32)
            for token in content_tokens(text):
                vec[zlib.crc32(token.encode()) % _VECTOR_SIZE] += 1.0
            if not vec.any():
                vec[0] = 1.0  # cosine distance is undefined for a zero vector
            rows.append(vec / np.linalg.norm(vec))
        return np.asarray(rows, dtype=np.float32)


class LexicalReranker:
    """
    Stands in for bge-reranker-v2-m3: the share of the query's content words the
    passage contains, in [0, 1] like the real model's sigmoid output.
    """

    def predict(self, pairs, **_kwargs) -> np.ndarray:
        scores = []
        for query, passage in pairs:
            wanted = set(content_tokens(query))
            have = set(content_tokens(passage))
            scores.append(len(wanted & have) / len(wanted) if wanted else 0.0)
        return np.asarray(scores, dtype=np.float32)


class LexicalBM25:
    """Stands in for FastEmbed's Qdrant/bm25: term-frequency sparse vectors."""

    def embed(self, texts, **_kwargs):
        for text in texts:
            counts: dict[int, float] = {}
            for token in content_tokens(text) or ["__empty__"]:
                index = zlib.crc32(token.encode())
                counts[index] = counts.get(index, 0.0) + 1.0
            indices = sorted(counts)
            yield SimpleNamespace(
                indices=np.asarray(indices, dtype=np.int64),
                values=np.asarray([counts[i] for i in indices], dtype=np.float32),
            )


# ==============================================================================
# Scripted LLM
# ==============================================================================

ROLES = ("query_intelligence", "query_rewriter", "quality_assessor", "judge", "synthesis")

# Each JSON-mode agent is recognised by a key of the output contract its system
# prompt spells out. That key is what the node reads from the reply, so prompt
# and node cannot drift apart on it without breaking production too. The answer
# synthesizer is the only call made without JSON mode.
_CONTRACT_KEYS = (
    ("rewritten_query", "query_rewriter"),
    ("verdicts", "judge"),
    ("context_quality_score", "quality_assessor"),
    ("sub_questions", "query_intelligence"),
)

_DOCUMENT = re.compile(r'<document index="(\d+)"([^>]*)>\n(.*?)\n</document>', re.DOTALL)
_ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')

DECLINE = (
    "### 1. Direct Answer\n\nThe provided documents do not contain the information "
    "needed to answer this question. No passage in the retrieved context states the "
    "requested fact, so no figure or term can be given without going beyond what the "
    "data room supports."
)


class ScriptedRateLimitError(Exception):
    """Shaped like a provider 429 so the ladder's quota handling recognises it."""

    def __init__(self, model: str = ""):
        super().__init__(f"429 RESOURCE_EXHAUSTED: quota exceeded for {model or 'model'} (scripted)")


@dataclass
class Document:
    """One <document> element as the model was shown it."""

    index: int
    attributes: dict[str, str]
    body: str

    @property
    def source(self) -> str:
        return self.attributes.get("source", "")

    @property
    def section(self) -> str:
        return self.attributes.get("section", "")


@dataclass
class LLMCall:
    """One call the pipeline made to the (scripted) LLM."""

    role: str
    model: str
    system: str
    user: str
    stream: bool
    api_key: str | None

    @property
    def documents(self) -> list[Document]:
        return [
            Document(
                index=int(index),
                attributes={k: html.unescape(v) for k, v in _ATTRIBUTE.findall(attrs)},
                body=body,
            )
            for index, attrs, body in _DOCUMENT.findall(self.user)
        ]

    @property
    def sources(self) -> set[str]:
        return {d.source for d in self.documents}


Responder = Callable[[LLMCall], Any]


def quote(*needles: str, extra: str = "") -> Responder:
    """
    A synthesis reply built from the context the model was actually given.

    For each needle, the line of the first document containing it is quoted with
    a citation to that document. Because the text comes from the prompt, the
    reply stays valid whatever the chunker or the context formatter does — and
    if retrieval did not surface a needle, the reply simply lacks it, which is
    what a grounded model would do.

    The synthesizer rejects a reply under 220 characters as a truncated
    generation, so quote at least two facts when the answer is meant to stand.

    Args:
        *needles: Substrings identifying the facts to state.
        extra: Raw text appended to the answer (e.g. an unsupported claim).

    Returns:
        A responder for ScriptedLLM.script("synthesis", ...).
    """

    def respond(call: LLMCall) -> str:
        lines = []
        for needle in needles:
            for doc in call.documents:
                hit = next((ln.strip() for ln in doc.body.splitlines() if needle in ln), None)
                if hit:
                    lines.append(f"- {hit} [{doc.source} | {doc.section or 'document'}]")
                    break
        if extra:
            lines.append(extra)
        if not lines:
            return DECLINE
        return "### 1. Direct Answer\n\n" + "\n".join(lines)

    return respond


class ScriptedLLM:
    """
    Replacement for litellm.acompletion.

    Each role has a script: a sequence of steps consumed one per call, the last
    repeating. A step is a dict (returned as JSON), a str (returned as prose), an
    Exception (raised), or a callable taking the LLMCall and returning any of
    those. Roles without a script get an inert default and the call is recorded
    in `unscripted`, so a test can require that nothing unexpected was asked.
    """

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []
        self.unscripted: list[LLMCall] = []
        self._scripts: dict[str, list[Any]] = {}
        self._defaults: dict[str, Any] = {
            # Omitting reformulated_query makes the node keep the user's query.
            "query_intelligence": {
                "query_type": "summary",
                "primary_intent": "scripted",
                "metadata_filters": {},
                "sub_questions": [],
            },
            # Omitting rewritten_query makes the node keep the current query.
            "query_rewriter": {"rewrite_reasoning": "scripted"},
            "quality_assessor": {
                "context_quality_score": 0.1,
                "quality_breakdown": {"relevance": 0.1, "completeness": 0.0, "precision": 0.0},
                "missing_aspects": ["scripted: context does not answer the question"],
                "force_refusal": True,
            },
            "judge": _all_supported,
            "synthesis": DECLINE,
        }

    def script(self, role: str, *steps: Any) -> "ScriptedLLM":
        assert role in ROLES, f"unknown role {role!r}; expected one of {ROLES}"
        assert steps, "a script needs at least one step"
        self._scripts[role] = list(steps)
        return self

    def of(self, role: str) -> list[LLMCall]:
        return [c for c in self.calls if c.role == role]

    def count(self, role: str) -> int:
        return len(self.of(role))

    def saw(self, text: str) -> bool:
        """True if `text` appeared in any prompt sent so far, to any agent."""
        return any(text in c.system or text in c.user for c in self.calls)

    @staticmethod
    def _role(kwargs: dict) -> str:
        if kwargs.get("response_format") is None:
            return "synthesis"
        system = kwargs["messages"][0]["content"]
        for key, role in _CONTRACT_KEYS:
            if key in system:
                return role
        raise AssertionError(
            "ScriptedLLM could not tell which agent made a JSON-mode call: its system "
            f"prompt names none of {[k for k, _ in _CONTRACT_KEYS]}. If a prompt's output "
            "contract changed, update _CONTRACT_KEYS in tests/fixtures/pipeline_harness.py."
        )

    async def __call__(self, **kwargs):
        messages = kwargs["messages"]
        call = LLMCall(
            role=self._role(kwargs),
            model=kwargs.get("model", ""),
            system=messages[0]["content"],
            user=messages[-1]["content"],
            stream=bool(kwargs.get("stream")),
            api_key=kwargs.get("api_key"),
        )
        self.calls.append(call)

        script = self._scripts.get(call.role)
        if script is None:
            self.unscripted.append(call)
            step = self._defaults[call.role]
        else:
            step = script.pop(0) if len(script) > 1 else script[0]
        if callable(step) and not isinstance(step, Exception):
            step = step(call)
        if isinstance(step, Exception):
            raise step
        content = step if isinstance(step, str) else json.dumps(step)

        if call.stream:
            return _stream(content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None
        )


def _all_supported(call: LLMCall) -> dict:
    """Default judge: every numbered claim in the prompt is supported by document 1."""
    block = call.user.split("Claims to check:", 1)[-1].split("Source documents:", 1)[0]
    ids = [int(n) for n in re.findall(r"^(\d+)\. ", block, flags=re.MULTILINE)]
    return {
        "verdicts": [
            {"id": i, "status": "supported", "document": 1, "reason": "scripted"} for i in ids
        ]
    }


async def _stream(content: str):
    for start in range(0, len(content), 48):
        piece = content[start : start + 48]
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))], usage=None)


# ==============================================================================
# The pipeline under test
# ==============================================================================


@dataclass
class Run:
    """What one query did, as an outside observer sees it."""

    state: dict
    # Graph nodes in completion order, from stream_query's `stage` events.
    # None for a blocking run, which emits no events.
    stages: list[str] | None = None
    events: list[tuple[str, dict]] = field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.state.get("generated_answer", "")

    @property
    def tokens(self) -> str:
        return "".join(p.get("text", "") for e, p in self.events if e == "token")

    @property
    def draft(self) -> str:
        """
        What a streaming client is displaying when `result` arrives: the tokens
        since the last `answer_reset`.
        """
        text = ""
        for event, payload in self.events:
            if event == "answer_reset":
                text = ""
            elif event == "token":
                text += payload.get("text", "")
        return text

    @property
    def resets(self) -> list[str]:
        """Reasons of the `answer_reset` events, in order."""
        return [p.get("reason", "") for e, p in self.events if e == "answer_reset"]

    @property
    def context_sources(self) -> set[str]:
        chunks = self.state.get("expanded_context") or self.state.get("reranked_results") or []
        return {c.get("source_file", "") for c in chunks}

    @property
    def cited_sources(self) -> set[str]:
        return {c.get("source_file", "") for c in self.state.get("citations", [])}


class OfflinePipeline:
    """The compiled graph over an in-memory index, with a scripted LLM."""

    def __init__(self, llm: ScriptedLLM, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.llm = llm
        self._workdir = workdir
        self._monkeypatch = monkeypatch
        self._client = None
        self._graph = None

    async def client(self):
        """The in-memory Qdrant client, created on first use inside the test's loop."""
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            from src.vector_db import qdrant_client as qdrant_module
            from src.vector_db.collection_manager import setup_collections

            self._client = AsyncQdrantClient(location=":memory:")
            await setup_collections(self._client)
            # get_qdrant_client() hands out this module-level singleton.
            module_path, attribute = SEAMS["qdrant_singleton"]
            if not hasattr(qdrant_module, attribute):
                raise HarnessSeamMoved(f"{module_path}.{attribute} no longer exists")
            setattr(qdrant_module, attribute, self._client)
        return self._client

    @property
    def graph(self):
        """The production graph, compiled with an in-memory checkpointer."""
        if self._graph is None:
            from langgraph.checkpoint.memory import MemorySaver

            from src.workflow.orchestrator import build_graph

            self._graph = build_graph().compile(checkpointer=MemorySaver())
        return self._graph

    async def ingest(
        self,
        name: str,
        text: str,
        deal_id: str = DEAL,
        *,
        category: str | None = None,
        supersedes: dict | None = None,
    ) -> dict:
        """
        Indexes one .txt document through the production ingestion path.

        Args:
            name: Filename (drives format detection and version resolution).
            text: Document body.
            deal_id: Deal to index into.
            category: Category override; auto-classified when None.
            supersedes: The index_document result of the version this replaces.

        Returns:
            index_document's result dict.
        """
        from src.data_processing.ingest_pipeline import index_document, mark_superseded

        client = await self.client()
        folder = self._workdir / deal_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text(text, encoding="utf-8")
        result = await index_document(
            str(path),
            name,
            deal_id,
            category,
            supersedes_doc_id=supersedes["doc_id"] if supersedes else None,
            client=client,
        )
        if supersedes:
            await mark_superseded(deal_id, supersedes["doc_id"], result["doc_id"], client=client)
        return result

    async def ask(
        self,
        query: str,
        deal_id: str = DEAL,
        *,
        include_pii: bool = False,
        stream: bool = True,
    ) -> Run:
        """
        Runs one query through the compiled graph.

        Args:
            query: The question.
            deal_id: Deal scope.
            include_pii: The authenticated caller's PII authorization.
            stream: Use stream_query (records stages and tokens) rather than
                run_query.

        Returns:
            The Run.
        """
        from src.workflow.orchestrator import run_query, stream_query

        await self.client()
        if not stream:
            state = await run_query(self.graph, query, deal_id, include_pii=include_pii)
            assert state.get("status") == "completed", f"pipeline errored: {state.get('error')}"
            return Run(state=state)

        events = [
            (name, payload)
            async for name, payload in stream_query(
                self.graph, query, deal_id, include_pii=include_pii
            )
        ]
        errors = [p for e, p in events if e == "error"]
        assert not errors, f"pipeline errored: {errors}"
        results = [p for e, p in events if e == "result"]
        assert len(results) == 1, f"expected exactly one result event, got {len(results)}"
        return Run(
            state=results[0],
            stages=[p["agent"] for e, p in events if e == "stage"],
            events=events,
        )

    async def load_data_room(self) -> dict[str, dict]:
        """Indexes the standard mini data room (see DATA_ROOM). Returns results by name."""
        indexed: dict[str, dict] = {}
        for spec in DATA_ROOM:
            indexed[spec["name"]] = await self.ingest(
                spec["name"],
                spec["text"],
                spec.get("deal_id", DEAL),
                category=spec.get("category"),
                supersedes=indexed[spec["supersedes"]] if spec.get("supersedes") else None,
            )
        return indexed

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()

    def use_laya(self, decide: Callable[[dict, dict], dict]) -> None:
        """
        Swaps in a scripted Laya decision model for the query-time decisions.

        Ingestion keeps its regex-only path (its Laya seam stays unloadable), so
        what is in the index does not depend on whether a test scripts Laya.

        Args:
            decide: (state, questions) -> {question name: answer dict}, e.g.
                {"stated": {"noul": 0.02}} for the answerability gate.
        """
        self._monkeypatch.setenv("LAYA_ENABLED", "1")
        _patch_seam(
            self._monkeypatch, "laya", lambda requests: [decide(s, q) for s, q in requests]
        )

    def use_nli(self, entailment: Callable[[str, str], float]) -> None:
        """
        Swaps in a scripted NLI scorer and turns NLI verification on.

        Args:
            entailment: (premise, hypothesis) -> P(entailment); the remainder is
                reported as neutral, never contradiction.
        """
        from src.verification.nli import NLIScore

        async def score(pairs):
            out = []
            for premise, hypothesis in pairs:
                p = float(entailment(premise, hypothesis))
                out.append(NLIScore(entailment=p, contradiction=0.0, neutral=1.0 - p))
            return out

        self._monkeypatch.setenv("VERIFY_NLI", "1")
        _patch_seam(self._monkeypatch, "nli", score)


# ==============================================================================
# A data room small enough to reason about
# ==============================================================================
# One sentence per line, so a scripted answer can quote a fact as one line. The
# same question ("termination fee") has a different answer in the current
# agreement, its superseded draft, and another deal — which is what lets a test
# tell a filter that holds from one that leaked.

FINANCIALS = """\
NORTHWIND SYSTEMS INC.
ANNUAL FINANCIAL SUMMARY

================================================================================
REVENUE AND PROFITABILITY
================================================================================

Total revenue for FY2023 was $452.8 million, compared with $387.1 million in FY2022.
Subscription revenue for FY2023 was $318.4 million, which is the largest component of total revenue.
Adjusted EBITDA for FY2023 was $96.3 million after restructuring charges of $4.5 million.

================================================================================
LIQUIDITY
================================================================================

Cash and cash equivalents at the end of FY2023 were $61.2 million.
The revolving credit facility of $150.0 million was undrawn at the end of FY2023.
"""

AGREEMENT_DRAFT = """\
AGREEMENT AND PLAN OF MERGER
DRAFT OF JANUARY 2024

================================================================================
ARTICLE IX - TERMINATION
================================================================================

Section 9.3 - Termination Fee
The termination fee payable by the Company to Buyer upon a superior proposal is $18.0 million.
The reverse termination fee payable by Buyer upon a financing failure is $30.0 million.
"""

AGREEMENT_FINAL = """\
AGREEMENT AND PLAN OF MERGER
EXECUTION VERSION

================================================================================
ARTICLE IX - TERMINATION
================================================================================

Section 9.3 - Termination Fee
The termination fee payable by the Company to Buyer upon a superior proposal is $25.0 million.
The reverse termination fee payable by Buyer upon a financing failure is $42.0 million.

================================================================================
ARTICLE VIII - INDEMNIFICATION
================================================================================

Section 8.2 - Indemnification Cap
The indemnification cap for breaches of representations and warranties is $174.0 million.
The survival period for general representations and warranties is eighteen months after Closing.
"""

RETENTION_SCHEDULE = """\
NORTHWIND SYSTEMS INC.
EXECUTIVE RETENTION SCHEDULE

================================================================================
RETENTION AWARDS
================================================================================

Jane Roe, Chief Financial Officer, Social Security Number 536-22-7914, receives a retention bonus: $1,250,000 payable at Closing.
Jane Roe may be contacted at jane.roe@northwind.example or 415-555-0137 regarding the retention award.
"""

OTHER_DEAL_AGREEMENT = """\
AGREEMENT AND PLAN OF MERGER
EXECUTION VERSION

================================================================================
ARTICLE IX - TERMINATION
================================================================================

Section 9.3 - Termination Fee
The termination fee payable by the Company to Buyer upon a superior proposal is $77.0 million.
Total revenue for FY2023 was $999.9 million according to the seller's management accounts.
"""

DATA_ROOM: list[dict] = [
    {"name": "northwind_financial_summary.txt", "text": FINANCIALS, "category": "financial"},
    {"name": "merger_agreement_v1.txt", "text": AGREEMENT_DRAFT, "category": "legal"},
    {
        "name": "merger_agreement_v2.txt",
        "text": AGREEMENT_FINAL,
        "category": "legal",
        "supersedes": "merger_agreement_v1.txt",
    },
    {"name": "executive_retention_schedule.txt", "text": RETENTION_SCHEDULE, "category": "legal"},
    {
        "name": "contoso_merger_agreement.txt",
        "text": OTHER_DEAL_AGREEMENT,
        "category": "legal",
        "deal_id": OTHER_DEAL,
    },
]

# Figures that must never reach a reader of DEAL who is not entitled to them.
SUPERSEDED_FEE = "$18.0 million"
OTHER_DEAL_FEE = "$77.0 million"
OTHER_DEAL_REVENUE = "$999.9 million"
PII_MARKERS = ("536-22-7914", "$1,250,000", "jane.roe@northwind.example")


# ==============================================================================
# Fixture
# ==============================================================================

# Every variable production code reads that changes what a query does. Set
# explicitly so a developer's shell (or .env) cannot change what is under test.
_ENVIRONMENT = {
    "GEMINI_API_KEYS": ",".join(FAKE_API_KEYS),
    "GEMINI_API_KEY": "",
    "LAYA_ENABLED": "0",
    "LAYA_GATE": "1",
    "LAYA_QUERY_GUARD": "1",
    "VERIFY_NLI": "0",
    "VERIFICATION_BACKEND": "cloud",
    "SYNTHESIS_MODEL_PIN": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "ENVIRONMENT": "production",
    "QDRANT_API_KEY": "",
}


def install(monkeypatch: pytest.MonkeyPatch, workdir: Path) -> OfflinePipeline:
    """
    Replaces the external seams and returns the pipeline. Use the
    `offline_pipeline` fixture in tests/conftest.py rather than calling this.
    """
    for name, value in _ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    for i in range(1, 10):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)

    llm = ScriptedLLM()
    _patch_seam(monkeypatch, "llm", llm)

    async def refuse(*_args, **_kwargs):
        raise ConnectionRefusedError("no Postgres in the offline suite")

    _patch_seam(monkeypatch, "postgres", refuse)

    embedding, reranker, bm25 = LexicalEmbeddingModel(), LexicalReranker(), LexicalBM25()
    _patch_seam(monkeypatch, "embedding_model", lambda: embedding)
    _patch_seam(monkeypatch, "reranker_model", lambda: reranker)
    _patch_seam(monkeypatch, "bm25_model", lambda: bm25)

    # Restores whatever client the singleton held once the test ends; the
    # pipeline points it at the in-memory index on first use.
    _patch_seam(monkeypatch, "qdrant_singleton", None)

    # The two remaining local models are off by environment above. They are
    # also made unloadable, so a test that switches one on without scripting it
    # (use_laya / use_nli) gets the pipeline's own "unavailable" fallback rather
    # than a multi-GB download — and so these seams are checked on every run,
    # not only by the tests that script them.
    from src.decisions.laya_client import LayaUnavailable

    def no_laya(_requests):
        raise LayaUnavailable("Laya is not loaded in the offline suite")

    async def no_nli(_pairs):
        raise RuntimeError("the NLI model is not loaded in the offline suite")

    _patch_seam(monkeypatch, "laya", no_laya)
    _patch_seam(monkeypatch, "laya_ingest", no_laya)
    _patch_seam(monkeypatch, "nli", no_nli)

    # A fresh tracker per test: quota and cooldown state must not carry over.
    from src.llm.budget_tracker import BudgetTracker

    monkeypatch.setattr(BudgetTracker, "_instance", None)
    monkeypatch.setattr(BudgetTracker, "_instance_lock", None)

    return OfflinePipeline(llm, workdir, monkeypatch)


async def start_tracker() -> None:
    """Initialises BudgetTracker the way the API lifespan does (in-memory here)."""
    from src.llm.budget_tracker import BudgetTracker

    tracker = await BudgetTracker.get_instance("postgresql://offline-suite/none")
    assert tracker._is_mock, "the offline harness must never reach a real Postgres"
