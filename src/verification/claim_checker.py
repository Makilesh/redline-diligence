"""
Claim-level answer verification.

Pipeline, cheapest first:

1. Split the answer into atomic claims (src/verification/claims.py).
2. Ground every figure deterministically (numeric_grounding.py).
3. Score each claim against its evidence with a local model (nli.py): the
   FactCG grounding checker on GPU, an NLI cross-encoder on CPU. "Entailment"
   below is the checker's support probability; each model has its own
   thresholds (thresholds_for). Evidence is sentence-sized windows from the
   chunks the claim cites, plus the
   best lexical matches across all retrieved chunks, so a correct claim with a
   sloppy citation is not failed for the citation alone.
4. Only claims neither step could decide go to an LLM judge — batched into ONE
   call, and only when at least one such claim exists. The common case makes
   zero LLM calls; the previous validator made one on every query.

Per-claim decision (first matching rule wins; "figure" = amount or ratio,
"quantity" = "24 months", "45 days"; see numeric_grounding for "confirmed" —
found at a unit the document states AND tied to the row and period the claim
names):

    an amount or ratio not in the context          -> unsupported   (numeric)
    beyond the checking budget (MAX_CLAIMS claims
      the figures do not settle)                   -> unverified ("not checked")
    a statement of absence ("X is not disclosed")  -> LLM judge only
    a figure the context gives for another row
      or period                                    -> LLM judge
    a quantity the context does not state          -> LLM judge
    NLI entailment >= entail threshold and no
      figure merely present elsewhere              -> supported     (nli)
    every figure confirmed                         -> supported     (numeric)
    NLI contradiction >= contradiction threshold
      and entailment < 0.1                         -> LLM judge (suspected
                                                      contradiction)
    NLI clearly neutral (entailment below the
      clear-neutral line, contradiction low,
      no figures)                                  -> unsupported   (nli)
    otherwise                                      -> LLM judge

A claim sent to the judge stays "unverified" when no judge answers. So
"contradicted" is only ever an LLM judge's verdict, confirmed against the
documents; the NLI model and the figure binding raise the question.

Answer-level outcome:

    failed   any contradicted claim, or any figure not found in the context
    warning  any unsupported or unverified claim, or an unverifiable
             calculation
    passed   otherwise

Confidence (shown to users) is COMPUTED, not self-reported by a model:

    claim_support   = (supported + 0.5 * unverified) / claims
    figure_support  = figures confirmed, or in a claim the NLI model or the
                      judge supported / figures
    confidence      = claim_support                       if no figures
                    = (claim_support + figure_support) / 2 otherwise

Nothing is exempt: a statement that the documents lack something is a claim
like any other (it is the designed way to decline, and it can be false). An
answer with no claims at all — only headings and fragments — scores 0.0.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from src.utils.logger import setup_logger
from src.verification import nli
from src.verification.claims import Claim, resolve_cited_chunks, split_claims
from src.verification.evidence import chunk_evidence_text
from src.verification.numeric_grounding import (
    ContextIndex,
    ground_texts,
    summarize_numeric,
)

logger = setup_logger(__name__)

# Thresholds on the NLI softmax. Entailment >= 0.5 means entailment is the
# model's most likely label by a margin; contradiction is held to a much higher
# bar because a false "contradicted" fails the answer and triggers a re-synthesis
# on a scarce reasoning-model quota.
ENTAIL_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.9
CLEAR_NEUTRAL_MAX_ENTAIL = 0.02
CLEAR_NEUTRAL_MAX_CONTRADICTION = 0.5


@dataclass(frozen=True)
class Thresholds:
    """Where the decision rules draw their lines on one model's scores."""

    entail: float
    contradiction: float
    clear_neutral_entail: float
    clear_neutral_contradiction: float


# The values above are the cross-encoders'. A grounding checker's support
# probability is on another scale. FactCG's were chosen on the dev half of
# eval/verification_set.json by a rule fixed before any checker was scored —
# the fewest wrong claims passed while passing at least as many correct claims
# locally as nli-deberta-v3-small — then the warn line that catches the most
# wrong claims with no more false alarms than -small's; reported once on the
# test half (DECISIONS_LOG). It has no contradiction output, so that rule never
# fires for it. A low support score is a far better "not supported" signal
# than a cross-encoder's neutral: on dev, wrong claims caught locally (warn or
# fail) rose from 46 to 102 of 167 at the same three false alarms, so they no
# longer wait for the judge.
CROSS_ENCODER_THRESHOLDS = Thresholds(
    ENTAIL_THRESHOLD, CONTRADICTION_THRESHOLD,
    CLEAR_NEUTRAL_MAX_ENTAIL, CLEAR_NEUTRAL_MAX_CONTRADICTION,
)
MODEL_THRESHOLDS = {
    # 0.432 sits in a narrow gap on dev: a wrong claim scores 0.4305 and a
    # correct one 0.4326. fp16 moves scores by at most 0.002 on the benchmark
    # and changes no decision; a recalibration should start here.
    nli.FACTCG: Thresholds(
        entail=0.432, contradiction=CONTRADICTION_THRESHOLD,
        clear_neutral_entail=0.2, clear_neutral_contradiction=CLEAR_NEUTRAL_MAX_CONTRADICTION,
    ),
}


def thresholds_for(model: str | None) -> Thresholds:
    """
    The thresholds calibrated for `model`'s scores.

    Args:
        model: The model that produced the scores (nli.loaded_model_name());
            None or an unlisted model gets the cross-encoder thresholds — which
            is also what an injected scorer gets when no model is loaded.

    Returns:
        Thresholds.
    """
    return MODEL_THRESHOLDS.get(model or "", CROSS_ENCODER_THRESHOLDS)
# Share of a claim's content tokens a premise window must contain before its
# contradiction score is believed (see the note where it is applied).
CONTRADICTION_MIN_OVERLAP = 0.5

# Evidence selection per claim: best windows (by lexical overlap) from the
# chunks the claim cites, plus from the top retrieved chunks.
CITED_WINDOWS_PER_CLAIM = 2
OTHER_WINDOWS_PER_CLAIM = 2
FALLBACK_TOP_K = 6

# Bounds on work per answer.
MAX_CLAIMS = 40
MAX_JUDGE_CLAIMS = 15
MAX_JUDGE_DOCUMENTS = 8

Scorer = Callable[[list[tuple[str, str]]], Awaitable[list[nli.NLIScore]]]
# judge(query, [{"id", "claim"}], [chunk dicts]) -> ({id: {"status", "reason"}}, model)
Judge = Callable[[str, list[dict], list[dict]], Awaitable[tuple[dict[int, dict], str]]]

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "was", "were", "are",
    "has", "had", "have", "its", "which", "into", "than", "their", "also", "been",
    "under", "over", "per", "all", "any", "not", "but", "such", "these", "those",
    "fy", "company", "total",
}
_TOKEN = re.compile(r"[a-z][a-z\-]{2,}|\d+(?:\.\d+)?")


@dataclass
class VerificationReport:
    """Everything the validator node needs to populate state."""

    claim_checks: list[dict] = field(default_factory=list)
    numeric_checks: list[dict] = field(default_factory=list)
    validation_status: str = "passed"
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)
    status_counts: dict = field(default_factory=dict)
    method_counts: dict = field(default_factory=dict)
    numeric_counts: dict = field(default_factory=dict)
    nli_model: str | None = None
    llm_model: str | None = None
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


class _FigureView:
    """
    What a claim's figures say about it, before any model is asked.

    Quantities ("24 months", "45 days") are checked one way only. One that the
    context does not contain stops the claim being passed locally; one that it
    does contain proves little — "18 months" and "36 months" sit in the same
    survival clause — so quantities alone never confirm a claim. And a missing
    one is not an automatic failure the way a missing amount is: documents
    spell durations out ("eighteen months") and answers compute them from dates.

    Attributes:
        unsupported: Raw text of amounts and ratios not found in the context.
        misattributed: Figures the context holds for another row or period.
        missing_quantities: Raw text of quantities the context does not state.
        unbound: Raw text of figures present, in a place the claim does not name.
        confirmed: True when the figures alone settle the claim as supported.
        settled: True when the figures alone settle it either way — so the
            entailment model and the judge have nothing to add.
    """

    def __init__(self, nums: list):
        figures = [n for n in nums if n.kind != "quantity"]
        self.unsupported = [n.raw for n in figures if n.status == "unsupported"]
        self.misattributed = [n for n in figures if n.binding == "conflict"]
        self.missing_quantities = [
            n.raw for n in nums if n.kind == "quantity" and n.status != "grounded"
        ]
        self.unbound = [n.raw for n in figures if n.binding == "unbound"]
        self.confirmed = (
            bool(figures)
            and all(n.is_confirmed for n in figures)
            and not self.missing_quantities
        )
        self.settled = bool(self.unsupported) or self.confirmed


def _tokens(text: str) -> set[str]:
    """
    Content tokens for lexical evidence ranking.

    Numbers are de-comma'd and a plural "s" is dropped, so "waivers" finds
    "No waiver or amendment has been requested" — the kind of near-verbatim
    evidence a stricter match missed during calibration.
    """
    lowered = (text or "").lower().replace(",", "")
    out = set()
    for t in _TOKEN.findall(lowered):
        if t in _STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss") and t[0].isalpha():
            t = t[:-1]
        out.add(t)
    return out


_LABEL_PREFIX = re.compile(r"^([^:]{2,60}):\s+(.+)$")


def _nli_hypothesis(claim_text: str) -> str:
    """
    Rewrites a claim into the plain-sentence form NLI models were trained on.

    Synthesis answers lean on "Label: statement" bullets. Measured on the sample
    data room, the label alone flipped verdicts: "Reportability & Valuation: The
    transaction is reportable under the HSR Act" scored 0.986 CONTRADICTION
    against "The transaction is reportable under the Hart-Scott-Rodino Antitrust
    Improvements Act." When the statement after the label is a sentence in its
    own right, the label is dropped; when it is a fragment ("36 months
    post-closing") the pair is joined with "is" so it still reads as a claim.

    Args:
        claim_text: Clean claim text.

    Returns:
        Hypothesis text for NLI scoring.
    """
    m = _LABEL_PREFIX.match(claim_text.strip())
    if not m or len(m.group(1).split()) > 6:
        return claim_text
    label, rest = m.group(1).strip(), m.group(2).strip().rstrip(" .")
    if len(rest.split()) >= 5 and rest[:1].isupper():
        return rest
    return f"{label} is {rest}"


def _build_windows(chunks: list[dict]) -> dict[int, list[tuple[str, set[str]]]]:
    """
    Premise windows (with their tokens) for every chunk.

    Evidence text is the same the synthesizer saw (chunk_evidence_text). Chunks
    that expand to the same parent section share one computation.

    Returns:
        {chunk index: [(window text, window tokens)]}.
    """
    cache: dict[str, list[tuple[str, set[str]]]] = {}
    out = {}
    for ci, chunk in enumerate(chunks):
        body = chunk_evidence_text(chunk)
        if body not in cache:
            cache[body] = [(w, _tokens(w)) for w in nli.split_premises(body)]
        out[ci] = cache[body]
    return out


def _best_windows(
    claim_tokens: set[str],
    chunk_ids: list[int],
    windows: dict[int, list[tuple[str, set[str]]]],
    limit: int,
) -> list[tuple[int, str, float]]:
    """
    Picks the windows most likely to hold a claim's evidence, by token overlap.

    Lexical preselection keeps NLI cost at a few pairs per claim instead of
    every window of every chunk — the difference between milliseconds and many
    seconds on a CPU-only host.

    Returns:
        (chunk index, window text, overlap) tuples, best first; zero-overlap
        windows are dropped, and a window shared by sibling chunks (same
        parent section) is returned once.
    """
    scored = []
    seen: set[str] = set()
    for ci in chunk_ids:
        for w, toks in windows.get(ci, []):
            if w in seen:
                continue
            overlap = len(claim_tokens & toks) / max(len(claim_tokens), 1)
            if overlap > 0:
                seen.add(w)
                scored.append((ci, w, overlap))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:limit]


def _snippet(window: str, claim_tokens: set[str], width: int = 280) -> str:
    """The part of a window around its first claim-token hit."""
    lowered = window.lower()
    hit = min(
        (lowered.find(t) for t in claim_tokens if len(t) > 3 and lowered.find(t) >= 0),
        default=0,
    )
    start = max(0, hit - 60)
    return window[start: start + width].strip()


def _evidence_fields(chunk: dict | None) -> dict:
    """Source metadata for a claim check's evidence."""
    if not chunk:
        return {"evidence_source": "", "evidence_page": None, "evidence_section": ""}
    return {
        "evidence_source": chunk.get("source_file", "") or "",
        "evidence_page": chunk.get("page_number") or None,
        "evidence_section": chunk.get("section_heading", "") or "",
    }


def _short(text: str, limit: int = 200) -> str:
    """Truncates claim text for flags."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def verify_answer(
    answer: str,
    chunks: list[dict],
    query: str,
    *,
    scorer: Scorer | None = None,
    judge: Judge | None = None,
    use_nli: bool | None = None,
) -> VerificationReport:
    """
    Verifies an answer claim by claim against its retrieved context.

    Args:
        answer: Generated answer (markdown with inline citation markers).
        chunks: Retrieved context the answer was written from.
        query: The user's original question (for the judge prompt).
        scorer: NLI scorer; defaults to the local model. Injected in tests.
        judge: LLM judge for undecided claims; None disables LLM judging.
        use_nli: Force NLI on/off; defaults to VERIFY_NLI.

    Returns:
        VerificationReport.
    """
    start = time.monotonic()
    report = VerificationReport()

    claims: list[Claim] = split_claims(answer)
    if not claims:
        report.notes.append("no verifiable claims")
        report.elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return report

    # ── 1. Numeric grounding ────────────────────────────────────────────────
    # Every claim's figures are grounded: it is deterministic and cheap. The
    # claim cap used to be applied first, and a fabricated figure in the 41st
    # claim of a long answer came back "passed" at confidence 1.0.
    index = ContextIndex(chunks)
    numeric = ground_texts([c.text for c in claims], index)
    report.numeric_checks = [n.to_dict() for group in numeric for n in group]
    report.numeric_counts = summarize_numeric(report.numeric_checks)
    figure_view = [_FigureView(group) for group in numeric]

    # MAX_CLAIMS bounds the entailment and judge work, and it is spent only on
    # claims whose outcome that work can change. A claim settled by its figures
    # — confirmed, or not found — does not use it, so padding an answer with
    # grounded bullet points cannot push a false statement past the budget.
    in_budget: set[int] = set()
    for claim in claims:
        if figure_view[claim.index].settled:
            continue
        if len(in_budget) < MAX_CLAIMS:
            in_budget.add(claim.index)

    # ── 2. NLI ──────────────────────────────────────────────────────────────
    if use_nli is None:
        use_nli = nli.nli_enabled()
    if scorer is None and use_nli:
        scorer = nli.ascore_pairs

    windows = _build_windows(chunks)
    top_k = list(range(min(len(chunks), FALLBACK_TOP_K)))

    # (claim idx, chunk idx, cited, window, lexical overlap)
    pair_owner: list[tuple[int, int, bool, str, float]] = []
    pairs: list[tuple[str, str]] = []
    cited_by_claim: dict[int, list[int]] = {}
    claim_tokens: dict[int, set[str]] = {}

    for claim in claims:
        toks = _tokens(claim.text)
        claim_tokens[claim.index] = toks
        cited = resolve_cited_chunks(claim.markers, chunks)
        cited_by_claim[claim.index] = cited
        # No entailment pairs for a claim the figures already settled, one past
        # the budget, or a statement of absence — "X is not disclosed" is not
        # entailed by any passage, only by the lack of one.
        if claim.index not in in_budget or claim.is_absence:
            continue
        # Uncited evidence is searched across ALL retrieved chunks: lexical
        # ranking is cheap, and a correct claim with a wrong or missing
        # citation should still find its passage.
        others = [ci for ci in range(len(chunks)) if ci not in cited]
        chosen = _best_windows(toks, cited, windows, CITED_WINDOWS_PER_CLAIM)
        chosen += _best_windows(
            toks, others, windows,
            OTHER_WINDOWS_PER_CLAIM + (0 if cited else 1),
        )
        hypothesis = _nli_hypothesis(claim.text)
        for ci, w, overlap in chosen:
            pair_owner.append((claim.index, ci, ci in cited, w, overlap))
            pairs.append((w, hypothesis))

    nli_scores: list[nli.NLIScore] = []
    nli_available = False
    if use_nli and scorer is not None and pairs:
        try:
            nli_scores = await scorer(pairs)
            nli_available = True
            report.nli_model = nli.loaded_model_name() or nli.nli_model_name()
        except Exception as e:
            logger.warning("NLI scoring unavailable; falling back", extra={"error": str(e)})
            report.notes.append(f"NLI unavailable: {e}")
    elif not use_nli:
        report.notes.append("NLI disabled (VERIFY_NLI=0)")
    # The lines follow the model that actually scored: a checker that failed
    # to load was replaced by a cross-encoder (nli._get_model).
    th = thresholds_for(nli.loaded_model_name() if nli_available else None)

    # Best evidence per claim.
    best: dict[int, dict] = {}
    for (claim_idx, ci, _cited, window, overlap), score in zip(pair_owner, nli_scores):
        entry = best.setdefault(claim_idx, {
            "entail": 0.0, "entail_chunk": None, "entail_window": "",
            "contra": 0.0, "contra_chunk": None, "contra_window": "",
        })
        if score.entailment > entry["entail"]:
            entry.update(entail=score.entailment, entail_chunk=ci, entail_window=window)
        # A contradiction only counts from a window that is plainly about the
        # same thing. Small NLI models return confident "contradiction" for a
        # premise that merely sits near the topic — measured on the sample data
        # room, "Aurora has not sought a waiver" scored 0.997 contradiction
        # against the one-line document title "AURORA TECHNOLOGIES INC.".
        if overlap >= CONTRADICTION_MIN_OVERLAP and score.contradiction > entry["contra"]:
            entry.update(contra=score.contradiction, contra_chunk=ci, contra_window=window)

    # ── 3. Decide ───────────────────────────────────────────────────────────
    checks: list[dict] = []
    undecided: list[int] = []

    for claim in claims:
        nums = numeric[claim.index]
        num_dicts = [
            {
                k: v for k, v in n.to_dict().items()
                if k in ("raw", "status", "formula", "evidence_source", "binding", "binding_note") and v
            }
            for n in nums
        ]
        view = figure_view[claim.index]
        unsupported_nums = view.unsupported
        misattributed = view.misattributed
        missing_quantities = view.missing_quantities
        unbound = view.unbound
        all_grounded = view.confirmed
        beyond_cap = not view.settled and claim.index not in in_budget
        ev = best.get(claim.index) if nli_available else None
        cited = cited_by_claim[claim.index]
        contra = ev["contra"] if ev else 0.0
        entail = ev["entail"] if ev else 0.0

        check = {
            "claim": claim.text,
            "status": "unverified",
            "method": "none",
            "score": None,
            "numbers": num_dicts,
            "cited": bool(cited),
            "reason": "",
            "evidence": "",
            **_evidence_fields(None),
        }
        toks = claim_tokens[claim.index]

        if unsupported_nums:
            check.update(
                status="unsupported",
                method="numeric",
                reason="figure(s) not found in the retrieved documents: " + ", ".join(unsupported_nums),
            )
        elif beyond_cap:
            # The entailment and judge budget is spent. The claim is reported as
            # not checked — never as supported — and the answer cannot pass.
            check["reason"] = (
                f"not checked: beyond the first {MAX_CLAIMS} statements that needed "
                "checking against the documents' text"
            )
        elif claim.is_absence:
            # "The indemnity cap is not specified." No passage entails an
            # absence, so the entailment model is not asked; the judge reads
            # the documents and says whether one of them states the thing after
            # all. Until it has, the statement is unverified — it is the
            # designed way to decline, so it never fails an answer by itself,
            # but a false one tells a reviewer a term or a risk does not exist.
            undecided.append(claim.index)
            check.update(
                reason="states that the documents do not contain something; not confirmed",
                absence=True,
            )
        elif misattributed:
            # The value exists, but where the documents put it is not where the
            # claim puts it ("Revenue was $387.1 million in FY2023" — that is
            # the FY2022 column). Deterministic evidence of a wrong figure, yet
            # it rests on reading a table's layout, so like an NLI contradiction
            # it is confirmed by the judge before it can fail the answer.
            undecided.append(claim.index)
            first = misattributed[0]
            check.update(
                score=round(entail, 3) if ev else None,
                reason=f"{first.raw} is in the documents, but {first.binding_note}",
                evidence=first.evidence_snippet,
                evidence_source=first.evidence_source,
                suspected_misattribution=True,
            )
        elif missing_quantities:
            undecided.append(claim.index)
            check.update(
                score=round(entail, 3) if ev else None,
                reason="not stated in the retrieved documents: " + ", ".join(missing_quantities),
                suspected_misattribution=True,
            )
        elif ev and entail >= th.entail and not unbound:
            # Not when a figure of the claim sits in the context only somewhere
            # the claim does not name. The entailment model reads a window that
            # holds the value and calls the claim entailed whatever the value is
            # attached to there: "The termination fee payable on a Superior
            # Proposal is $13.92 million" scored 0.98 against the sentence that
            # gives $13.92 million as the REVERSE fee. Measured on the dev half of
            # eval/verification_set.json with nli-deberta-v3-small: wrong claims
            # passed 22/167 -> 15/167, supported claims passed locally 68.8% ->
            # 66.1% (three more go to the judge), false alarms unchanged.
            chunk = chunks[ev["entail_chunk"]]
            check.update(
                status="supported",
                method="nli",
                score=round(entail, 3),
                evidence=_snippet(ev["entail_window"], toks),
                **_evidence_fields(chunk),
            )
        elif all_grounded:
            # Exact figure matches outrank the NLI model on figure-dense claims.
            # Measured on the 35 RESULTS.md answers: of the claims whose every
            # figure was grounded yet NLI scored >= 0.9 contradiction, the large
            # majority were table rows and "Label: $X" lines the model misreads
            # ("Minimum Fixed Charge Coverage Ratio: >= 1.25x"). The score is
            # kept on the check so a reviewer can still see the disagreement.
            first = next((n for n in nums if n.evidence_source), None)
            check.update(
                status="supported",
                method="numeric",
                score=round(entail, 3) if ev else None,
                reason="every figure appears in, or is computed from, the retrieved documents",
                evidence=first.evidence_snippet if first else "",
                evidence_source=first.evidence_source if first else "",
            )
            if contra >= th.contradiction:
                check["nli_contradiction"] = round(contra, 3)
        elif ev and contra >= th.contradiction and entail < 0.1:
            # A suspected contradiction, confirmed by the judge before it can
            # fail the answer. Measured on the 35 RESULTS.md answers against the
            # sample data room: every one of the NLI model's >= 0.9
            # contradictions (5 of 5) was wrong — "36 months (Fundamental
            # Representation)" against "18 months, except that Fundamental
            # Representations … 36 months", for instance. A false "contradicted"
            # fails the answer and spends a reasoning-model re-synthesis, so the
            # small model is trusted to raise the question, not to settle it.
            chunk = chunks[ev["contra_chunk"]]
            undecided.append(claim.index)
            check.update(
                score=round(contra, 3),
                reason="NLI suggests a contradiction; not confirmed",
                evidence=_snippet(ev["contra_window"], toks),
                suspected_contradiction=True,
                **_evidence_fields(chunk),
            )
        elif (
            ev
            and entail < th.clear_neutral_entail
            and contra < th.clear_neutral_contradiction
            and not nums
        ):
            check.update(
                status="unsupported",
                method="nli",
                score=round(entail, 3),
                reason="no retrieved passage entails this statement",
            )
        else:
            undecided.append(claim.index)
            check["score"] = round(entail, 3) if ev else None
            if unbound:
                check["reason"] = (
                    "figure(s) present in the documents but not in a passage this "
                    "statement names: " + ", ".join(unbound)
                )
            elif nli_available and ev is None:
                check["reason"] = "no retrieved passage shares vocabulary with this claim"
        checks.append(check)

    # ── 4. LLM judge for what is left ───────────────────────────────────────
    if undecided:
        # When the cap bites, the claims with something specific against them
        # go first: a figure attached to the wrong row, a suspected contradiction.
        undecided.sort(key=lambda idx: not (
            checks[idx].get("suspected_misattribution") or checks[idx].get("suspected_contradiction")
        ))
        to_judge = undecided[:MAX_JUDGE_CLAIMS]
        if len(undecided) > MAX_JUDGE_CLAIMS:
            report.notes.append(
                f"{len(undecided) - MAX_JUDGE_CLAIMS} undecided claims beyond the judge cap left unverified"
            )
        if judge is None:
            report.notes.append("validation unavailable: no LLM judge configured for undecided claims")
        else:
            # Documents in priority order: each claim's best NLI evidence, then
            # what it cites, then the top retrieved chunks. Siblings expanding
            # to the same parent section are sent once.
            doc_ids: list[int] = []
            seen_bodies: set[str] = set()

            def add(ci: int | None) -> None:
                if ci is None or ci in doc_ids or len(doc_ids) >= MAX_JUDGE_DOCUMENTS:
                    return
                body = chunk_evidence_text(chunks[ci])
                if body in seen_bodies:
                    return
                seen_bodies.add(body)
                doc_ids.append(ci)

            for idx in to_judge:
                ev = best.get(idx)
                if ev:
                    add(ev["entail_chunk"])
                    add(ev["contra_chunk"])
            for idx in to_judge:
                for ci in cited_by_claim[idx]:
                    add(ci)
            for ci in top_k:
                add(ci)
            documents = [chunks[ci] for ci in sorted(doc_ids)]
            payload = [{"id": i + 1, "claim": claims[idx].text} for i, idx in enumerate(to_judge)]

            try:
                verdicts, model = await judge(query, payload, documents)
                report.llm_calls = 1
                report.llm_model = model
                for i, idx in enumerate(to_judge):
                    verdict = verdicts.get(i + 1) or {}
                    status = str(verdict.get("status", "")).lower()
                    if status not in ("supported", "contradicted", "unsupported"):
                        continue
                    # "Unsupported" is no verdict on a statement of absence:
                    # either a document states the thing or none does.
                    if status == "unsupported" and checks[idx].get("absence"):
                        continue
                    doc_no = verdict.get("document")
                    doc = None
                    try:
                        if doc_no:
                            doc = documents[int(doc_no) - 1]
                    except (ValueError, IndexError, TypeError):
                        doc = None
                    checks[idx].update(
                        status=status,
                        method="llm",
                        reason=str(verdict.get("reason", ""))[:300],
                        **_evidence_fields(doc),
                    )
            except Exception as e:
                logger.warning("LLM judge unavailable", extra={"error": str(e)})
                report.notes.append(f"validation unavailable: LLM judge failed ({type(e).__name__}: {e})")

    # ── 5. Aggregate ────────────────────────────────────────────────────────
    report.claim_checks = checks
    status_counts = Counter(c["status"] for c in checks)
    method_counts = Counter(c["method"] for c in checks if c["method"] != "none")
    report.status_counts = {
        s: status_counts.get(s, 0)
        for s in ("supported", "contradicted", "unsupported", "unverified")
    }
    report.method_counts = dict(method_counts)

    n = len(checks)
    claim_support = (status_counts["supported"] + 0.5 * status_counts["unverified"]) / n
    # A figure counts when it is confirmed (found AND tied to what the claim
    # says it is, or reproduced by arithmetic), or when its claim was confirmed
    # by the entailment model or the judge. A value that merely occurs
    # somewhere, and a calculation nobody could reproduce, count for nothing —
    # the latter used to earn half credit.
    figures_total = sum(len(group) for group in numeric)
    if figures_total:
        figures_confirmed = sum(
            1
            for claim, check in zip(claims, checks)
            for figure in numeric[claim.index]
            if figure.status in ("grounded", "derived")
            and (figure.is_confirmed or check["status"] == "supported")
        )
        report.confidence = round((claim_support + figures_confirmed / figures_total) / 2, 3)
    else:
        report.confidence = round(claim_support, 3)

    flags: list[str] = []
    for c in checks:
        if c["status"] == "contradicted":
            src = c["evidence_source"] or "the retrieved documents"
            flags.append(f"Contradicted by {src}: {_short(c['claim'])}")
        elif c["status"] == "unsupported" and c["method"] == "numeric":
            flags.append(f"Figure not found in the documents — {_short(c['claim'])}")
        elif c["status"] == "unsupported":
            flags.append(f"Not supported by the retrieved documents: {_short(c['claim'])}")
    unverifiable_calcs = [
        n["raw"] for n in report.numeric_checks if n["status"] == "derived_unverified"
    ]
    if unverifiable_calcs:
        flags.append(
            "Calculation not reproducible from the documents: "
            + ", ".join(dict.fromkeys(unverifiable_calcs))
        )
    unchecked = sum(1 for c in checks if c["reason"].startswith("not checked: beyond"))
    if unchecked:
        report.notes.append(f"{unchecked} of {len(checks)} claims beyond the checking budget")
        flags.append(
            f"{unchecked} statement(s) beyond the first {MAX_CLAIMS} were checked for "
            "figures only, not against the documents' text"
        )
    unconfirmed_absences = sum(
        1 for c in checks if c.get("absence") and c["status"] == "unverified"
    )
    if unconfirmed_absences:
        flags.append(
            f"{unconfirmed_absences} statement(s) that the documents do not contain "
            "something could not be confirmed"
        )
    unavailable = [note for note in report.notes if note.startswith("validation unavailable")]
    if unavailable and status_counts["unverified"]:
        flags.append(
            f"validation unavailable: {status_counts['unverified']} claim(s) could not be verified"
        )
    report.flags = flags

    numeric_failure = any(
        c["status"] == "unsupported" and c["method"] == "numeric" for c in checks
    )
    if status_counts["contradicted"] or numeric_failure:
        report.validation_status = "failed"
    elif (
        status_counts["unsupported"]
        or status_counts["unverified"]
        or unverifiable_calcs
    ):
        report.validation_status = "warning"
    else:
        report.validation_status = "passed"

    report.elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    return report


def revision_feedback(claim_checks: list[dict], flags: list[str] | None = None) -> list[str]:
    """
    Builds the "fix these" list fed back into a re-synthesis.

    Args:
        claim_checks: claim_checks from a failed validation.
        flags: hallucination_flags, used when claim_checks is empty.

    Returns:
        One line per contradicted or unsupported claim, with the reason.
    """
    lines = []
    for c in claim_checks or []:
        if c.get("status") not in ("contradicted", "unsupported"):
            continue
        reason = c.get("reason") or (
            f"contradicted by {c.get('evidence_source')}" if c.get("status") == "contradicted"
            else "not supported by the documents"
        )
        lines.append(f"\"{_short(c.get('claim', ''), 300)}\" — {reason}")
    if not lines and flags:
        lines = [f for f in flags if not f.startswith("validation unavailable")]
    return lines
