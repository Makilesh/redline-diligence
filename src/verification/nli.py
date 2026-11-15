"""
Local claim-support scoring for claim verification.

A local model scores whether an evidence passage supports a claim. It replaces
the per-query LLM judge for the common case: it runs on the same machine as the
embedding and reranker models, costs no API quota, and is deterministic for a
given input. Two kinds of model sit behind the same interface:

- a cross-encoder NLI model (DeBERTa-v3 fine-tuned on SNLI + MultiNLI), which
  returns entailment / contradiction / neutral probabilities;
- a purpose-trained grounding checker (FactCG-DeBERTa-v3-Large, NAACL 2025,
  MIT), which returns one probability that the passage supports the claim. It
  is reported as an NLIScore with that probability as `entailment`, no
  contradiction signal, and the remainder as `neutral`.

Model choice (env NLI_MODEL overrides):
    GPU: yaxili96/FactCG-DeBERTa-v3-Large     (~435M params, fp16)
    CPU: cross-encoder/nli-deberta-v3-xsmall  (~70M params)
Measured on eval/verification_set.json (thresholds chosen on the dev half,
reported once on the test half; DECISIONS_LOG): against nli-deberta-v3-small,
FactCG passed 11/165 wrong claims instead of 16/165 — the 11 the deterministic
stages already pass, so it added none — flagged 3/101 correct claims instead
of 5/101, and sent 35% of claims to the LLM judge instead of 48% — it catches
flipped negations the cross-encoder entailed. The cost is speed and memory:
31 ms per claim on GPU in fp16 (-small: 12 ms) with a 1.2GB peak (0.6GB), and
2.5 s per claim on two x86 CPU threads (-xsmall: 0.2 s), so a CPU host keeps
-xsmall. If the checker cannot be loaded (not downloaded, out of memory) the
process falls back to nli-deberta-v3-small and says so in /health
(runtime_status "nli_checker_fallback").

Each model has its own decision thresholds (claim_checker.thresholds_for): the
two kinds' scores are not on the same scale.

Set VERIFY_NLI=0 to disable it entirely on hosts that cannot spare the memory;
verification then falls back to numeric grounding plus the LLM judge.

CRITICAL: inference is synchronous and CPU/GPU bound. Every call from async
code goes through asyncio.to_thread so it never blocks the event loop, and
model loading happens on that worker thread too.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FACTCG = "yaxili96/FactCG-DeBERTa-v3-Large"
_GPU_DEFAULT = FACTCG
# What the GPU falls back to when the checker cannot be loaded: the previous
# GPU default, whose thresholds were calibrated on this corpus.
_GPU_FALLBACK = "cross-encoder/nli-deberta-v3-small"
_CPU_DEFAULT = "cross-encoder/nli-deberta-v3-xsmall"

# Grounding checkers, by Hugging Face id: how each is called and the revision
# its thresholds were calibrated on (a new upload could move the scores).
# FactCG's input format is its reference implementation's
# (github.com/derenlei/FactCG, factcg/inference.py, Hugging Face checkpoint
# path): one sequence built from an instruction template, the tokenizer's pad
# token set to its EOS token, softmax over two labels, label 1 = supported.
_CHECKERS: dict[str, dict] = {
    FACTCG: {
        "revision": "0430e3509dbd28d2dff7a117c0eae25359ff3e80",
        "max_length": 2048,
    },
}
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above can we conclude "
    "that \"{text_b}\"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
)

# Premise windows are one to three sentences, capped well inside the model's
# 512-token limit (see split_premises for why short windows).
PREMISE_MAX_WORDS = 90

_RULE_LINE = re.compile(r"^\s*[=\-_*~]{4,}\s*$")
_COLUMN_GAP = re.compile(r"\S\s{2,}\S")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'$])")
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16

_model = None
_model_name: str | None = None
# "nli" (a CrossEncoder with three labels) or "checker" (a _Checker).
_model_kind: str | None = None
_label_index: dict[str, int] = {}
_load_lock = threading.Lock()
# One inference at a time: the reranker and embedder share the device, and
# concurrent predict() calls on one GPU only contend for the same memory.
_predict_lock = threading.Lock()
_load_failed = False


@dataclass(frozen=True)
class NLIScore:
    """Softmax probabilities for one (premise, hypothesis) pair."""

    entailment: float
    contradiction: float
    neutral: float


def nli_enabled() -> bool:
    """True unless VERIFY_NLI is set to a falsy value."""
    return os.getenv("VERIFY_NLI", "1").strip().lower() not in ("0", "false", "no", "off")


def nli_model_name() -> str:
    """
    The model this process is configured to use: NLI_MODEL if set, else a
    device default. What actually loaded is loaded_model_name() — a checker
    that fails to load is replaced by a cross-encoder.

    Returns:
        Hugging Face model id.
    """
    configured = os.getenv("NLI_MODEL", "").strip()
    if configured:
        return configured
    try:
        import torch

        return _GPU_DEFAULT if torch.cuda.is_available() else _CPU_DEFAULT
    except Exception:
        return _CPU_DEFAULT


def is_checker(name: str | None) -> bool:
    """True when `name` is a grounding checker (one support score), not an NLI model."""
    return bool(name) and name in _CHECKERS


def _checker_half_precision() -> bool:
    """fp16 for the checker on CUDA unless NLI_FP16=0 (as RERANKER_FP16)."""
    return os.getenv("NLI_FP16", "1").strip().lower() not in ("0", "false", "no", "off")


class _Checker:
    """
    A grounding checker loaded with transformers and called as its authors do.

    Pairs are scored in batches sorted by length (less padding — the cost of a
    batch is its longest member) and returned in input order.
    """

    def __init__(self, name: str, device: str):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        spec = _CHECKERS[name]
        revision = spec["revision"]
        config = AutoConfig.from_pretrained(
            name, revision=revision, num_labels=2, finetuning_task="text-classification"
        )
        config.problem_type = "single_label_classification"
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        # fp16 on CUDA only: on the benchmark it moves scores by at most 0.002,
        # changes no decision and halves latency (60 -> 31 ms per claim). CPU
        # half precision was not measured.
        dtype = torch.float16 if device == "cuda" and _checker_half_precision() else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            name, revision=revision, config=config, torch_dtype=dtype
        )
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = spec["max_length"]
        self.dtype = str(dtype).replace("torch.", "")
        self._torch = torch

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(supported) per (premise, hypothesis) pair, in input order."""
        torch = self._torch
        texts = [_FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [0.0] * len(texts)
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start: start + BATCH_SIZE]
            enc = self.tokenizer(
                [texts[i] for i in idx], max_length=self.max_length, truncation=True,
                padding="longest", return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            for i, p in zip(idx, probs):
                out[i] = float(p)
        return out


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(name: str) -> None:
    """Loads `name` into the module globals. Caller holds _load_lock."""
    global _model, _model_name, _model_kind, _label_index

    device = _device()
    logger.info(f"Loading NLI model {name} to {device.upper()}")
    if is_checker(name):
        model = _Checker(name, device)
        _label_index = {}
        _model, _model_kind = model, "checker"
        _model_name = name
        logger.info(f"Grounding checker {name} loaded on {device.upper()}",
                    extra={"dtype": model.dtype})
        return

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name, max_length=MAX_SEQ_LENGTH, device=device)
    id2label = getattr(model.config, "id2label", None) or {}
    labels = {str(v).lower(): int(k) for k, v in id2label.items()}
    required = {"entailment", "contradiction", "neutral"}
    if not required.issubset(labels):
        raise RuntimeError(
            f"NLI model {name} labels {sorted(labels)} lack {sorted(required)}"
        )
    _label_index = labels
    _model, _model_kind = model, "nli"
    _model_name = name
    logger.info(f"NLI model {name} loaded on {device.upper()}", extra={"labels": labels})


def _release_gpu_memory() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _fallback_for(name: str) -> str | None:
    """The cross-encoder to use when checker `name` cannot be loaded."""
    if not is_checker(name):
        return None
    try:
        return _GPU_FALLBACK if _device() == "cuda" else _CPU_DEFAULT
    except Exception:
        return _CPU_DEFAULT


def _get_model():
    """
    Loads the configured model once per process (thread-safe, lazy).

    For a cross-encoder the label order is read from the model config rather
    than assumed: the cross-encoder NLI checkpoints use {0: contradiction,
    1: entailment, 2: neutral}, which differs from the MNLI convention most
    people remember, and a silently swapped label would invert every verdict.

    A grounding checker that cannot be loaded (not downloaded on an offline
    host, out of GPU memory) is replaced by the cross-encoder fallback, and the
    substitution is recorded as a degraded mode so /health shows it. The
    decision thresholds follow the model actually loaded (loaded_model_name).

    Returns:
        The loaded model (CrossEncoder or _Checker).

    Raises:
        RuntimeError: If no model can be loaded (cached after first failure,
            so a missing model is not re-downloaded on every query).
    """
    global _load_failed

    if _model is not None:
        return _model
    if _load_failed:
        raise RuntimeError("NLI model failed to load earlier in this process")

    with _load_lock:
        if _model is not None:
            return _model
        name = nli_model_name()
        try:
            _load(name)
            return _model
        except Exception as first:
            fallback = _fallback_for(name)
            if fallback is None:
                _load_failed = True
                raise
            logger.warning(
                f"Grounding checker {name} could not be loaded; falling back to {fallback}",
                extra={"error_type": type(first).__name__, "error": str(first)[:300]},
            )
            _release_gpu_memory()  # an out-of-memory load can leave blocks held
            try:
                _load(fallback)
            except Exception:
                _load_failed = True
                raise
            from src.utils import runtime_status

            runtime_status.mark_degraded(
                "nli_checker_fallback",
                f"The grounding checker {name} could not be loaded "
                f"({type(first).__name__}); claims are scored by {fallback}, which "
                "passes more wrong claims on the verification benchmark.",
            )
            return _model


def loaded_model_name() -> str | None:
    """Name of the model actually loaded, or None if not loaded yet."""
    return _model_name


def score_pairs(pairs: list[tuple[str, str]]) -> list[NLIScore]:
    """
    Scores (premise, hypothesis) pairs synchronously. Call via ascore_pairs.

    Args:
        pairs: (evidence passage, claim) tuples.

    Returns:
        One NLIScore per pair, in order. A grounding checker's support
        probability is the entailment; it has no contradiction output.
    """
    if not pairs:
        return []
    model = _get_model()
    if _model_kind == "checker":
        with _predict_lock:
            support = model.predict(pairs)
        return [NLIScore(entailment=p, contradiction=0.0, neutral=1.0 - p) for p in support]
    with _predict_lock:
        probs = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            apply_softmax=True,
            show_progress_bar=False,
        )
    e, c, n = (
        _label_index["entailment"],
        _label_index["contradiction"],
        _label_index["neutral"],
    )
    return [
        NLIScore(entailment=float(p[e]), contradiction=float(p[c]), neutral=float(p[n]))
        for p in probs
    ]


async def ascore_pairs(pairs: list[tuple[str, str]]) -> list[NLIScore]:
    """
    Scores pairs off the event loop.

    Args:
        pairs: (evidence passage, claim) tuples.

    Returns:
        One NLIScore per pair, in order.
    """
    return await asyncio.to_thread(score_pairs, pairs)


async def warm_nli_model() -> float | None:
    """
    Loads and exercises the NLI model so the first query does not pay for it.

    Mirrors src.vector_db.reranker.warm_models: one real forward pass, failures
    logged and swallowed — a warmup must never stop the API from starting.

    Returns:
        Seconds taken, or None when disabled or failed.
    """
    if not nli_enabled():
        return None
    start = time.monotonic()
    probe = "Aurora Technologies reported total revenue of $452.8 million in FY2023."
    try:
        await ascore_pairs([(probe, probe)])
    except Exception as e:
        logger.warning(f"NLI model warmup failed: {e}")
        return None
    elapsed = round(time.monotonic() - start, 2)
    logger.info("NLI model warmed", extra={"model": _model_name, "seconds": elapsed})
    return elapsed


def _premise_units(text: str) -> list[str]:
    """
    Splits evidence text into sentence-sized units.

    Data-room text is hard-wrapped prose interleaved with fixed-width tables, so
    neither newlines nor full stops alone mark a unit. Paragraphs (blank-line
    separated) whose lines look tabular — a gap of 2+ spaces between columns —
    keep one unit per line; other paragraphs are unwrapped and sentence-split.
    Separator rules ("=====", "-----") are dropped.
    """
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text or ""):
        lines = [ln for ln in para.splitlines() if ln.strip() and not _RULE_LINE.match(ln)]
        if not lines:
            continue
        tabular = sum(1 for ln in lines if len(_COLUMN_GAP.findall(ln.strip())) >= 1)
        if tabular >= max(1, len(lines) // 2):
            units.extend(re.sub(r"\s+", " ", ln).strip() for ln in lines)
            continue
        joined = re.sub(r"\s+", " ", " ".join(lines)).strip()
        units.extend(s.strip() for s in _SENTENCE_END.split(joined) if s.strip())
    return units


def split_premises(text: str, max_words: int = PREMISE_MAX_WORDS) -> list[str]:
    """
    Builds short premise windows from evidence text.

    Long premises make small NLI models unreliable: measured on the sample data
    room, 180-word windows produced confident "contradiction" for claims whose
    exact supporting sentence sat inside the window, because the same window
    also held a neighbouring clause about a different jurisdiction or period.
    Scoring against one to three consecutive sentences (the standard
    sentence-level approach, e.g. SummaC) and taking the best window removes
    most of that noise. Windows of 2-3 units cover claims that fuse adjacent
    sentences.

    Args:
        text: Evidence text.
        max_words: Upper bound on window length.

    Returns:
        Windows, each 1-3 consecutive units within max_words.
    """
    units = _premise_units(text)
    windows: list[str] = []
    for i in range(len(units)):
        words = 0
        for k in range(i, min(i + 3, len(units))):
            words += len(units[k].split())
            if words > max_words and k > i:
                break
            windows.append(" ".join(units[i: k + 1]))
    return windows
