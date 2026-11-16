"""
Reranker calibration profiles — the score scale the context gate reads.

The Quality Assessor's floors and the router's thresholds were measured on
BAAI/bge-reranker-v2-m3's output (DECISIONS_LOG Decision 17). A cross-encoder's
sigmoid output is not a probability shared between models: on the same 1145
dev pairs ms-marco-MiniLM-L-6-v2 puts the median pair at 0.0002 where bge puts
it at 0.0026, and the 90th percentile at 0.33 against 0.22. The thresholds
therefore do not transfer. Measured on the held-out golden set with MiniLM
scored raw against bge's floors: heuristic controls admitted/LLM/refused went
3/1/2 -> 1/4/1 and answerables 33/1/1 -> 31/3/1.

So the gate never reads a raw score. Every reranker has a profile that maps
its raw score onto ONE common scale — bge-reranker-v2-m3's, where the
thresholds below were measured — and the thresholds live once, on that scale.
The bge profile is the identity; other profiles carry a monotone map derived on
the dev set (never the golden set) by eval/decisions/run_reranker_calibration.py.
A monotone map changes no ranking, only the numbers the thresholds compare.
The one threshold a profile may lower is the outright-refusal ceiling, for a
reranker that misses evidence bge finds (see MINILM_PROFILE).

A profile is keyed by model AND sequence length, because truncation changes
scores. An unlisted reranker, or a listed one at another length, raises
UncalibratedReranker when the model loads (so startup fails with
WARM_MODELS=1, the first query otherwise). There is deliberately no fallback
scale: reading an unknown model's scores against bge's floors is exactly the
silent miscalibration this module exists to prevent.

Env:
    RERANKER_PROFILE  Name of a profile to apply to the configured reranker,
                      bypassing the model/length check. For a fine-tune or a
                      re-hosted copy of a calibrated model only; it asserts the
                      scores are on that profile's scale, so measure first.
"""

from __future__ import annotations

import bisect
import contextlib
import contextvars
import os
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ==============================================================================
# Gate thresholds — on the common (bge-reranker-v2-m3) scale
#
# Measured on bge, see DECISIONS_LOG Decision 17 and the calibration notes in
# src/agents/quality_assessor.py and src/workflow/conditional_edges.py. They
# are the same for every reranker because every reranker's scores are mapped
# onto this scale first.
# ==============================================================================

# Score at or above which a chunk counts as usable evidence (peak-F1 from sweep).
RELEVANCE_FLOOR = 0.10

# Below this, the best chunk is noise and no amount of context will help.
CONFIDENT_REFUSAL_CEILING = 0.05

# Above this, the heuristic is confident enough to skip the LLM assessor.
CONFIDENT_PASS_FLOOR = 0.30

# The router's per-dimension floors and its floor on the overall score.
ROUTER_THRESHOLDS: dict[str, float] = {"relevance": 0.30, "completeness": 0.30, "precision": 0.25}
ROUTER_MIN_SCORE = 0.30


class UncalibratedReranker(RuntimeError):
    """The configured reranker has no calibration profile at this sequence length."""


@dataclass(frozen=True)
class RerankerProfile:
    """
    How one reranker's scores reach the common scale.

    Attributes:
        name: Profile name (what RERANKER_PROFILE selects).
        model: Hub id of the cross-encoder it was measured with.
        max_length: Sequence length (tokens) it was measured at.
        knots: (raw, common-scale) points of a strictly increasing piecewise-
            linear map from (0, 0) to (1, 1). Empty = already on the scale.
        refusal_ceiling: Common-scale best score below which the heuristic
            refuses without asking the LLM. A map can align two rerankers'
            score distributions but not their mistakes: a reranker that scores
            some genuinely relevant passages near zero needs a lower ceiling
            than bge, or those questions are refused outright.
        derivation: Where the map and the ceiling come from.
        document_focus_k: Most in-document candidates the legal document-focus
            pass may add to the rerank pool with this reranker, or None for
            the query type's own depth (src/agents/retrieval_strategy.py).
            More candidates help only as far as this model orders them
            well, and each is one more cross-encoder pair on its host.
    """

    name: str
    model: str
    max_length: int
    knots: tuple[tuple[float, float], ...] = ()
    refusal_ceiling: float = CONFIDENT_REFUSAL_CEILING
    derivation: str = ""
    document_focus_k: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.refusal_ceiling <= CONFIDENT_REFUSAL_CEILING:
            # Raising it would refuse more than the measured reference does.
            raise ValueError(f"{self.name}: refusal_ceiling must be in (0, "
                             f"{CONFIDENT_REFUSAL_CEILING}]")
        if self.document_focus_k is not None and self.document_focus_k < 0:
            raise ValueError(f"{self.name}: document_focus_k must be >= 0 or None")
        if self.knots:
            xs = [k[0] for k in self.knots]
            ys = [k[1] for k in self.knots]
            if self.knots[0] != (0.0, 0.0) or self.knots[-1] != (1.0, 1.0):
                raise ValueError(f"{self.name}: a score map must run from (0, 0) to (1, 1)")
            if any(b <= a for a, b in zip(xs, xs[1:])) or any(b <= a for a, b in zip(ys, ys[1:])):
                raise ValueError(f"{self.name}: a score map must be strictly increasing")

    def calibrate(self, raw: float) -> float:
        """
        The common-scale equivalent of a raw score from this profile's model.

        Args:
            raw: Sigmoid-normalised reranker score in [0, 1].

        Returns:
            The score bge-reranker-v2-m3 would need to sit at the same point of
            the distribution, in [0, 1].
        """
        score = min(max(float(raw), 0.0), 1.0)
        if not self.knots:
            return score
        xs = [k[0] for k in self.knots]
        i = min(max(bisect.bisect_right(xs, score), 1), len(self.knots) - 1)
        (x0, y0), (x1, y1) = self.knots[i - 1], self.knots[i]
        return y0 + (y1 - y0) * (score - x0) / (x1 - x0)


REFERENCE_PROFILE = RerankerProfile(
    name="bge-reranker-v2-m3",
    model="BAAI/bge-reranker-v2-m3",
    max_length=1024,
    derivation="The reference scale: every threshold above was measured on it (Decision 17).",
)

# The CPU host's reranker. Derived by `python -m eval.decisions.run_reranker_calibration`
# on eval/answerability_dev.json (2026-11-15): the 1145 (query, passage) pairs
# the cross-encoder scores for the 58 dev questions, scored by both models.
#
# Map: quantile mapping, MiniLM's raw score -> the bge score at the same point
# of the pooled distribution. Chosen over isotonic and Platt calibration onto a
# common P(passage states the answer), which the dev set can also fit (27
# positive pairs): under 5-fold cross-validation by minimal pair, both
# probability fits admitted 5 unanswerable dev questions bge sends to the LLM
# or refuses, while the quantile map admitted none — it is label-free, uses
# every pair, and was as good out of sample as in.
#
# Ceiling: a map can align the distributions but not the mistakes. MiniLM
# scored some evidence passages bge scores 0.60-0.75 at 0.001-0.03 (rank
# correlation with bge 0.82), so at bge's 0.05 two answerable dev questions
# (dev_a34 best 0.024 mapped, dev_a39 0.036) were refused outright. The
# ceiling is the largest grid value at most 0.8x the lowest such best score
# (0.0244): 0.0175. Dev, heuristic gate, admitted/LLM/refused — bge: answerable
# 25/2/0, unanswerable 19/7/2; MiniLM raw: 21/2/4, 21/3/4; mapped + ceiling:
# 21/5/1, 16/10/2 (the one refusal, dev_a41, has no evidence in MiniLM's
# context). All 24 off-topic/jailbreak/injection prompts stay refused.
_MINILM_KNOTS = (
    (0.0, 0.0), (1e-05, 1.6e-05), (1.2e-05, 5.8e-05), (1.4e-05, 0.000136),
    (1.8e-05, 0.000218), (2.3e-05, 0.000354), (2.9e-05, 0.000515), (3.7e-05, 0.000707),
    (5e-05, 0.000964), (7.4e-05, 0.001336), (0.000119, 0.001866), (0.000199, 0.002621),
    (0.000364, 0.003786), (0.000456, 0.004807), (0.000633, 0.005651), (0.000924, 0.006771),
    (0.001463, 0.008972), (0.002381, 0.011517), (0.004065, 0.014107), (0.008282, 0.017986),
    (0.013612, 0.024099), (0.022937, 0.031297), (0.033304, 0.045066), (0.056346, 0.06657),
    (0.099354, 0.096501), (0.180805, 0.143068), (0.334914, 0.219975), (0.422197, 0.279864),
    (0.538052, 0.318714), (0.677247, 0.465336), (0.752603, 0.535851), (0.803109, 0.680027),
    (0.877229, 0.762417), (0.942212, 0.886532), (0.985228, 0.958679), (0.996033, 0.987272),
    (0.997981, 0.996085), (0.999904, 0.99877), (1.0, 1.0),
)

MINILM_PROFILE = RerankerProfile(
    name="ms-marco-MiniLM-L-6-v2",
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    knots=_MINILM_KNOTS,
    refusal_ceiling=0.0175,
    derivation="Quantile map + refusal ceiling on eval/answerability_dev.json pairs "
               "(eval/decisions/run_reranker_calibration.py, fp32, max_length 512).",
    # Decision 71: on LegalBench-RAG's dev documents, run on CPU with two
    # threads, MiniLM put more of the ground truth into the final context with
    # 10 focus candidates than with 20 or 30 (78.6% against 75.2 / 75.3; 30 -> 10
    # +3.3 points [+0.5, +6.6], MAUD 23.4 -> 33.8%): past the first few, the
    # extra passages of the same agreement are ones it ranks above the clause
    # that answers. The cost would have been acceptable (+0.8 s at p50 for 30
    # against +0.3 s for 10, emulated 2-core host); the ranking was not.
    document_focus_k=10,
)

PROFILES: tuple[RerankerProfile, ...] = (REFERENCE_PROFILE, MINILM_PROFILE)


def resolve_profile(model: str, max_length: int | None = None) -> RerankerProfile:
    """
    The profile for `model` at `max_length`, or UncalibratedReranker.

    Args:
        model: The cross-encoder's Hub id (RERANKER_MODEL).
        max_length: Its effective sequence length; None skips the length check
            (a stand-in model that does not say).

    Returns:
        The matching profile.

    Raises:
        UncalibratedReranker: No profile for this model at this length, or
            RERANKER_PROFILE names no profile.
    """
    override = os.getenv("RERANKER_PROFILE", "").strip()
    if override:
        for profile in PROFILES:
            if override in (profile.name, profile.model):
                if profile.model != model:
                    logger.warning(
                        "RERANKER_PROFILE applies another model's calibration",
                        extra={"model": model, "profile": profile.name},
                    )
                return profile
        raise UncalibratedReranker(
            f"RERANKER_PROFILE={override!r} names no profile; known: "
            + ", ".join(p.name for p in PROFILES)
        )
    for profile in PROFILES:
        if profile.model == model:
            if max_length is not None and max_length != profile.max_length:
                raise UncalibratedReranker(
                    f"{model} is calibrated at max_length {profile.max_length}, configured "
                    f"{max_length}. Truncation moves its scores; re-run "
                    "eval/decisions/run_reranker_calibration.py at this length, or unset "
                    "RERANKER_MAX_LENGTH."
                )
            return profile
    raise UncalibratedReranker(
        f"No calibration profile for reranker {model!r}. The context gate's thresholds sit on "
        f"{REFERENCE_PROFILE.model}'s score scale and do not transfer between cross-encoders; "
        "derive a profile with eval/decisions/run_reranker_calibration.py, or set "
        "RERANKER_PROFILE to assert one."
    )


_forced: contextvars.ContextVar[RerankerProfile | None] = contextvars.ContextVar(
    "forced_reranker_profile", default=None
)


@contextlib.contextmanager
def use_profile(profile: RerankerProfile):
    """Evaluates the gate on `profile`'s scale (eval harnesses replaying stored scores)."""
    token = _forced.set(profile)
    try:
        yield profile
    finally:
        _forced.reset(token)


def active_profile() -> RerankerProfile:
    """
    The profile of the reranker that produced the scores being gated.

    The profile resolved when the model loaded (with its real sequence length)
    if it has loaded, else the configured model's profile.

    Raises:
        UncalibratedReranker: As resolve_profile.
    """
    forced = _forced.get()
    if forced is not None:
        return forced
    from src.vector_db import reranker

    loaded = getattr(reranker, "_reranker_profile", None)
    if loaded is not None:
        return loaded
    return resolve_profile(reranker.RERANKER_MODEL_NAME)
