"""
The grounding checker behind the NLI interface (src/verification/nli.py) and
the per-model decision thresholds (claim_checker.thresholds_for).

No weights are downloaded: the model objects are stubs. The one test that loads
FactCG is marked `models` and deselected in CI.
"""

import asyncio

import pytest

from src.utils import runtime_status
from src.verification import claim_checker, nli
from src.verification.nli import NLIScore

CHUNKS = [{
    "chunk_id": "c1",
    "text": "There is no financing condition. There is no condition relating to "
            "obtaining customer consents.",
    "source_file": "regulatory_and_data_privacy_memo.txt",
    "page_number": None,
    "section_heading": "Closing conditions",
}]
ANSWER = ("The merger agreement contains no financing condition "
          "[📄 regulatory_and_data_privacy_memo.txt | Closing conditions].")


@pytest.fixture
def fresh_nli(monkeypatch):
    """Module state as in a new process; restored afterwards."""
    for name, value in (("_model", None), ("_model_name", None), ("_model_kind", None),
                        ("_label_index", {}), ("_load_failed", False)):
        monkeypatch.setattr(nli, name, value)
    runtime_status.reset_status()
    yield
    runtime_status.reset_status()


class TestModelChoice:
    def test_gpu_defaults_to_the_checker_and_cpu_to_xsmall(self, monkeypatch):
        import torch

        monkeypatch.delenv("NLI_MODEL", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert nli.nli_model_name() == nli.FACTCG
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert nli.nli_model_name() == "cross-encoder/nli-deberta-v3-xsmall"

    def test_nli_model_overrides(self, monkeypatch):
        monkeypatch.setenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
        assert nli.nli_model_name() == "cross-encoder/nli-deberta-v3-small"
        assert not nli.is_checker(nli.nli_model_name())
        assert nli.is_checker(nli.FACTCG)


class TestThresholds:
    def test_each_model_has_its_own_lines(self):
        ce = claim_checker.thresholds_for("cross-encoder/some-other-nli-model")
        assert ce == claim_checker.CROSS_ENCODER_THRESHOLDS
        assert ce.entail == claim_checker.ENTAIL_THRESHOLD == 0.5
        fc = claim_checker.thresholds_for(nli.FACTCG)
        assert fc.entail == 0.432 and fc.clear_neutral_entail == 0.2

    def test_nothing_loaded_means_cross_encoder_lines(self):
        # An injected scorer with no model loaded (every offline test) keeps
        # the thresholds the stubs were written against, on any machine.
        assert claim_checker.thresholds_for(None) == claim_checker.CROSS_ENCODER_THRESHOLDS


class _StubChecker:
    def __init__(self, support):
        self.support = support
        self.seen = []

    def predict(self, pairs):
        self.seen.extend(pairs)
        return [self.support for _ in pairs]


class TestCheckerScores:
    def test_support_is_entailment_and_there_is_no_contradiction(self, fresh_nli, monkeypatch):
        monkeypatch.setattr(nli, "_model", _StubChecker(0.7))
        monkeypatch.setattr(nli, "_model_kind", "checker")
        monkeypatch.setattr(nli, "_model_name", nli.FACTCG)
        [score] = nli.score_pairs([("premise", "hypothesis")])
        assert score == NLIScore(entailment=0.7, contradiction=0.0, neutral=pytest.approx(0.3))

    def test_input_is_the_reference_template_and_order_is_kept(self):
        import torch

        class Tok:
            texts = []

            def __call__(self, texts, **kw):
                Tok.texts.extend(texts)
                assert kw["truncation"] is True and kw["max_length"] == 2048

                class Enc(dict):
                    def to(self, device):
                        return self
                return Enc(n=len(texts))

        class Model:
            def __call__(self, n):
                # label 1 = supported; make the long text the supported one
                rows = [[0.0, 5.0] if len(t) > 150 else [5.0, 0.0] for t in Tok.texts[-n:]]

                class Out:
                    logits = torch.tensor(rows)
                return Out()

        checker = nli._Checker.__new__(nli._Checker)
        checker.tokenizer, checker.model = Tok(), Model()
        checker.device, checker.max_length, checker._torch = "cpu", 2048, torch
        long_premise = "Revenue was $452.8 million in FY2023, up 17.0% year over year. " * 2
        out = checker.predict([(long_premise, "Revenue grew."), ("Short.", "Short.")])
        assert out[0] > 0.99 and out[1] < 0.01  # input order, not length order
        assert Tok.texts[0].startswith("Short.\n\nChoose your answer: based on the paragraph above")
        assert Tok.texts[0].endswith('can we conclude that "Short."?\n\nOPTIONS:\n- Yes\n- No\n'
                                     "I think the answer is ")


class TestFallback:
    def test_checker_that_cannot_load_falls_back_and_says_so(self, fresh_nli, monkeypatch):
        monkeypatch.setenv("NLI_MODEL", nli.FACTCG)
        monkeypatch.setattr(nli, "_device", lambda: "cuda")
        loaded = []

        def fake_load(name):
            if nli.is_checker(name):
                raise OSError("not in the local cache")
            loaded.append(name)
            nli._model, nli._model_kind, nli._model_name = object(), "nli", name

        monkeypatch.setattr(nli, "_load", fake_load)
        nli._get_model()
        assert loaded == ["cross-encoder/nli-deberta-v3-small"]
        assert nli.loaded_model_name() == "cross-encoder/nli-deberta-v3-small"
        assert "nli_checker_fallback" in runtime_status.describe()["degraded"]
        # ... and the decisions use the fallback's thresholds, not FactCG's.
        assert (claim_checker.thresholds_for(nli.loaded_model_name())
                != claim_checker.thresholds_for(nli.FACTCG))

    def test_cpu_fallback_is_xsmall(self, fresh_nli, monkeypatch):
        monkeypatch.setenv("NLI_MODEL", nli.FACTCG)
        monkeypatch.setattr(nli, "_device", lambda: "cpu")
        assert nli._fallback_for(nli.FACTCG) == "cross-encoder/nli-deberta-v3-xsmall"

    def test_a_cross_encoder_failure_is_not_papered_over(self, fresh_nli, monkeypatch):
        monkeypatch.setenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

        def boom(name):
            raise OSError("missing")

        monkeypatch.setattr(nli, "_load", boom)
        with pytest.raises(OSError):
            nli._get_model()
        with pytest.raises(RuntimeError, match="failed to load earlier"):
            nli._get_model()
        assert "nli_checker_fallback" not in runtime_status.describe()["degraded"]


def _verify(monkeypatch, model, support):
    async def scorer(pairs):
        return [NLIScore(support, 0.0, 1.0 - support) for _ in pairs]

    monkeypatch.setattr(nli, "ascore_pairs", scorer)
    monkeypatch.setattr(nli, "loaded_model_name", lambda: model)
    report = asyncio.run(claim_checker.verify_answer(ANSWER, CHUNKS, "financing?", judge=None, use_nli=True))
    [check] = report.claim_checks
    return check


class TestDecisionsFollowTheLoadedModel:
    def test_moderate_support_passes_under_the_checker_only(self, monkeypatch):
        check = _verify(monkeypatch, nli.FACTCG, 0.45)
        assert check["status"] == "supported" and check["method"] == "nli"
        check = _verify(monkeypatch, "cross-encoder/nli-deberta-v3-small", 0.45)
        assert check["status"] == "unverified"  # left to the judge

    def test_low_support_is_flagged_under_the_checker_only(self, monkeypatch):
        check = _verify(monkeypatch, nli.FACTCG, 0.15)
        assert check["status"] == "unsupported" and check["method"] == "nli"
        check = _verify(monkeypatch, "cross-encoder/nli-deberta-v3-small", 0.15)
        assert check["status"] == "unverified"


@pytest.mark.models
class TestRealFactCG:
    def test_negation_is_not_supported(self, fresh_nli, monkeypatch):
        # The benchmark's own pair (f083.4): nli-deberta-v3-small entailed it
        # at 0.954 from the sentence that says the opposite.
        monkeypatch.setenv("NLI_MODEL", nli.FACTCG)
        premise = "There is no financing condition. There is no condition relating to obtaining customer consents"
        scores = nli.score_pairs([
            (premise, "Closing is not conditional on financing."),
            (premise, "The merger agreement contains a financing condition in favour of the Buyer."),
        ])
        assert nli.loaded_model_name() == nli.FACTCG
        assert scores[0].entailment > 0.432
        assert scores[1].entailment < 0.2
