"""
Pydantic response models for the API.
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A citation reference in the answer."""
    chunk_id: str = ""
    source_file: str = ""
    page_number: int | None = None
    section_heading: str = ""
    is_current_version: bool = True
    content_type: str = "text"
    is_redline: bool = False
    superseded_by: str = ""


class PassageLocation(BaseModel):
    """
    Where a passage with instruction-like text sits — never what it says
    (src/data_processing/injection_screen.flagged_locations).
    """
    page_number: int | None = None
    section_heading: str = ""
    signals: list[str] = Field(
        default_factory=list,
        description=(
            "instruction_override | model_address | role_marker | forged_marker | "
            "supersession_claim | invisible_text"
        ),
    )


class FlaggedPassage(PassageLocation):
    """A retrieved passage with instruction-like text, by document and place."""
    source_file: str = ""


class InjectionScreen(BaseModel):
    """
    What the ingest screen found in an upload (Decisions 76, 82). A label only:
    the file is indexed exactly as written either way.
    """
    flagged_passages: int = Field(
        0, description="Indexed passages whose text reads like instructions to an AI model"
    )
    signals: list[str] = Field(default_factory=list, description="Every signal found")
    locations: list[PassageLocation] = Field(
        default_factory=list,
        description="Distinct pages/sections holding them, at most 10; no document text",
    )


class InjectionReview(BaseModel):
    """
    What the validator found about instruction-like text behind one answer
    (Decisions 76, 82). Empty when no retrieved passage carried any.
    """
    flagged_passages: list[FlaggedPassage] = Field(
        default_factory=list,
        description="Retrieved passages with instruction-like text, by location, at most 10",
    )
    cited_flagged_passages: int = Field(
        0, description="How many of them the answer cites (each makes 'passed' a 'warning')"
    )
    tainted_figures: list[str] = Field(
        default_factory=list,
        description=(
            "Answer figures that only instruction-like text states (each makes 'passed' a "
            "'warning'; the claim carries injection_suspect)"
        ),
    )


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    answer: str = Field(..., description="Generated answer with citations")
    answer_truncated: bool = Field(
        False,
        description=(
            "True when the answer stopped at the model's output limit rather than "
            "where the model ended it (streamed route only); validation_status is "
            "then never 'passed'"
        ),
    )
    query_type: str = Field(..., description="Detected query type")
    confidence_score: float = Field(..., description="Answer confidence 0.0-1.0")
    validation_status: str = Field(..., description="passed|warning|failed")
    citations: list[Citation] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    claim_checks: list[dict] = Field(
        default_factory=list,
        description="Per-claim verification: claim, status, method, evidence source, score",
    )
    total_latency_ms: float = Field(..., description="Total pipeline latency")
    session_id: str = Field(..., description="Session ID for this query")
    rewrite_iterations: int = Field(0, description="Number of query rewrites performed")
    agent_trace: list[dict] = Field(default_factory=list, description="Agent execution trace")
    is_refusal: bool = Field(
        False,
        description="True when the pipeline refused to answer for lack of usable context",
    )
    context_quality_score: float = Field(
        0.0, description="Best context quality score achieved across retrieval attempts"
    )
    outcome: str = Field(
        "answered",
        description=(
            "How the run ended: answered | answered_unverified (failed validation "
            "after the retry; shipped with flags) | refused (no usable context) | "
            "synthesis_failed (no model returned a usable answer)"
        ),
    )
    node_timings: list[dict] = Field(
        default_factory=list,
        description=(
            "One row per node execution, in order: agent, duration_ms, llm_calls, "
            "prompt_tokens, completion_tokens, models"
        ),
    )
    llm_calls: int = Field(0, description="Completed LLM calls this query made, all nodes")
    request_id: str = Field(
        "", description="Correlation id of the request, also sent as X-Request-ID"
    )
    injection_review: InjectionReview = Field(
        default_factory=InjectionReview,
        description=(
            "Retrieved passages whose text reads like instructions to an AI model, whether "
            "the answer cites them, and figures only they state. Locations only, never the text"
        ),
    )


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    doc_id: str = Field(..., description="Assigned document ID")
    deal_id: str = Field(..., description="Deal identifier")
    document_category: str = Field(..., description="Detected/assigned category")
    chunks_created: int = Field(..., description="Number of chunks indexed")
    status: str = Field("success", description="Ingestion status")
    notices: list[str] = Field(
        default_factory=list,
        description="Why a 'partial' ingestion was partial, when the cause is known",
    )
    version_candidates: list[dict] = Field(
        default_factory=list,
        description=(
            "Indexed documents this upload looks related to: [{doc_id, source_file, "
            "relation: newer|older|same_document}]. A proposal only — nothing was "
            "superseded; send supersedes_doc_id to act on it."
        ),
    )
    injection_screen: InjectionScreen = Field(
        default_factory=InjectionScreen,
        description=(
            "Passages of this upload whose text reads like instructions to an AI model. A "
            "label only — nothing was blocked or changed, and status is unaffected"
        ),
    )


class DealResponse(BaseModel):
    """Response model for deal operations."""
    deal_id: str
    deal_name: str
    description: str = ""
    document_count: int = 0
    status: str = "active"
    is_sandbox: bool = False
    expires_at: str = Field(
        "",
        description="ISO-8601 TTL deadline for sandbox deals; empty for permanent ones",
    )


class DocumentRecord(BaseModel):
    """A single ingested document and its place in the version chain."""
    doc_id: str
    filename: str
    document_category: str = ""
    chunks_created: int = 0
    version_label: str = ""
    upload_date: str = ""
    is_current_version: bool = True
    supersedes_doc_id: str = ""
    superseded_by: str = ""
    has_redline: bool = False
    instruction_like_passages: int = Field(
        0,
        description=(
            "Indexed passages of this document whose text reads like instructions to an AI "
            "model (Decisions 76, 82). A label only; the document is searched as usual"
        ),
    )
    instruction_signals: list[str] = Field(
        default_factory=list, description="The screen's signals found in those passages"
    )


class RiskSignal(BaseModel):
    """A risk signal detected in a deal document during ingestion."""
    signal_type: str
    severity: str = "low"
    source_file: str = ""
    description: str = ""
    page_number: int | None = None


class BudgetStatusResponse(BaseModel):
    """Response model for budget status."""
    synthesis_primary: dict = Field(default_factory=dict)
    agent_workhorse: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "healthy"
    service: str = "manda-rag"
