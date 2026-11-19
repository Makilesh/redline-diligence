"""
Typed shapes for what the structured agents return.

JSON mode guarantees syntax, not shape. A model can return `"metadata_filters":
null`, a query type that does not exist, or `"reformulated_query": null`, and
each of those used to travel into the pipeline as-is: the first crashed Agent 1
on `.pop`, the last handed `None` to the embedder as the query.

These models are deliberately LENIENT. A field the model got wrong is replaced
by its safe default rather than rejected, because the alternative — a repair
request, or failing the query — costs more than the field is worth: every one
of them has a default the pipeline already ran with. What the schema buys is
that the bad states are no longer representable downstream. Code that reads
`parsed.metadata_filters` gets a dict, always.

Nothing here decides what an LLM is ALLOWED to set. A well-formed
`metadata_filters` is still only a proposal; the whitelist that turns it into
real filters lives in retrieval_strategy.apply_filter_overrides.
"""

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

QueryType = Literal["financial", "legal", "comparative", "summary", "multi_hop"]
QUERY_TYPES: tuple[str, ...] = get_args(QueryType)
DEFAULT_QUERY_TYPE = "summary"


def _text(value: Any) -> str | None:
    """A non-blank string, stripped — or None for anything else."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _text_list(value: Any) -> list[str]:
    """The non-blank strings in a list — or [] for anything that is not a list."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _flag(value: Any) -> bool:
    """True only for a JSON true (or the string a sloppy model writes for it)."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


def _mapping(value: Any) -> dict:
    """A dict — or {} for null, a list, a string."""
    return dict(value) if isinstance(value, dict) else {}


class _AgentOutput(BaseModel):
    """Base for agent outputs: unknown keys are ignored, never an error."""

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_llm(cls, result: Any):
        """
        Builds the typed output from whatever the model returned.

        Never raises: a top-level value that is not an object, or a field no
        validator anticipated, yields the defaults.

        Args:
            result: The parsed JSON from the agent call.

        Returns:
            An instance of the output model.
        """
        try:
            return cls.model_validate(result if isinstance(result, dict) else {})
        except ValidationError:
            return cls()


class QueryIntelligenceOutput(_AgentOutput):
    """Agent 1 — the fields the pipeline reads."""

    query_type: QueryType = DEFAULT_QUERY_TYPE
    metadata_filters: dict[str, Any] = {}
    requires_numerical_precision: bool = False
    requires_cross_document: bool = False
    sub_questions: list[str] = []
    reformulated_query: str | None = None

    @field_validator("query_type", mode="before")
    @classmethod
    def _known_query_type(cls, value: Any) -> str:
        text = _text(value)
        normalised = text.lower() if text else ""
        return normalised if normalised in QUERY_TYPES else DEFAULT_QUERY_TYPE

    @field_validator("metadata_filters", mode="before")
    @classmethod
    def _filters_are_a_mapping(cls, value: Any) -> dict:
        return _mapping(value)

    @field_validator("requires_numerical_precision", "requires_cross_document", mode="before")
    @classmethod
    def _flags(cls, value: Any) -> bool:
        return _flag(value)

    @field_validator("sub_questions", mode="before")
    @classmethod
    def _sub_questions(cls, value: Any) -> list[str]:
        return _text_list(value)

    @field_validator("reformulated_query", mode="before")
    @classmethod
    def _reformulated_query(cls, value: Any) -> str | None:
        return _text(value)


class QueryRewriterOutput(_AgentOutput):
    """Agent 6 — the fields the pipeline reads."""

    rewritten_query: str | None = None
    updated_retrieval_config: dict[str, Any] = {}
    updated_metadata_filters: dict[str, Any] = {}
    rewrite_reasoning: str = ""
    alternative_formulations: list[str] = []

    @field_validator("rewritten_query", mode="before")
    @classmethod
    def _rewritten_query(cls, value: Any) -> str | None:
        return _text(value)

    @field_validator("updated_retrieval_config", "updated_metadata_filters", mode="before")
    @classmethod
    def _mappings(cls, value: Any) -> dict:
        return _mapping(value)

    @field_validator("rewrite_reasoning", mode="before")
    @classmethod
    def _reasoning(cls, value: Any) -> str:
        return _text(value) or ""

    @field_validator("alternative_formulations", mode="before")
    @classmethod
    def _alternatives(cls, value: Any) -> list[str]:
        return _text_list(value)


QUALITY_DIMENSIONS: tuple[str, ...] = ("relevance", "completeness", "precision")


def _unit_score(value: Any) -> float | None:
    """A finite number in [0, 1] (a numeric string counts) — or None.

    Out-of-range is None, not clamped: a model scoring on a 10-point scale
    would otherwise have every 0.8+ reading clamp to a perfect 1.0.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:  # an integer of hundreds of digits: no score (Decision 88)
        return None
    return value if 0.0 <= value <= 1.0 else None


class QualityAssessorOutput(_AgentOutput):
    """Agent 5's LLM branch — the fields the quality gate reads.

    Unlike the other agents, a field here has no safe default: a missing
    precision is not 0.0 (that silently refuses) and not 1.0 (that admits).
    `complete` says whether the reply can be used at all; when it cannot, the
    node falls back to its heuristic numbers.
    """

    context_quality_score: float | None = None
    quality_breakdown: dict[str, float | None] = {}
    missing_aspects: list[str] = []
    force_refusal: bool = False

    @field_validator("context_quality_score", mode="before")
    @classmethod
    def _score(cls, value: Any) -> float | None:
        return _unit_score(value)

    @field_validator("quality_breakdown", mode="before")
    @classmethod
    def _breakdown(cls, value: Any) -> dict[str, float | None]:
        given = _mapping(value)
        return {dim: _unit_score(given.get(dim)) for dim in QUALITY_DIMENSIONS}

    @field_validator("missing_aspects", mode="before")
    @classmethod
    def _missing(cls, value: Any) -> list[str]:
        return _text_list(value)

    @field_validator("force_refusal", mode="before")
    @classmethod
    def _refusal(cls, value: Any) -> bool:
        return _flag(value)

    @property
    def complete(self) -> bool:
        """True when the score and all three dimensions are usable."""
        return self.context_quality_score is not None and all(
            self.quality_breakdown.get(dim) is not None for dim in QUALITY_DIMENSIONS
        )
