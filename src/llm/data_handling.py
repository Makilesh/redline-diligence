"""
Where document text goes: the notice shown where a user decides to upload.

With a Gemini key configured, retrieved passages and the question are sent to
Google's Gemini API to write and check answers (every agent; the local model is
only the ladder's last rung). The deployment runs on the free tier (Decision 107),
whose terms (aistudio.google.com/docs/terms, effective 2026-03-23) say Google uses
what is submitted to "improve, and develop Google products", that "human reviewers
may read, annotate, and process your API input and output", and "Do not submit
sensitive, confidential, or personal information to the Unpaid Services". So the
upload panels say so before the upload, not in a footer (Decision 116).

With no key, nothing leaves the server and no notice is shown. UPLOAD_NOTICE
replaces the wording (the owner's to set; for example after moving to a paid tier
under a data processing agreement); it is shown only while a hosted model is
configured.
"""

from __future__ import annotations

import os

HOSTED_PROVIDER = "Google Gemini API"

DEFAULT_UPLOAD_NOTICE = (
    "Passages from uploaded documents, and your questions, are sent to Google's Gemini "
    "API to write and check answers. This deployment uses Gemini's free tier: Google "
    "may use what is sent to improve its products, and people at Google may read it. "
    "Do not upload confidential, client or personal documents."
)


def hosted_llm_configured() -> bool:
    """True when a Gemini key is configured, so document text can leave the server."""
    from src.llm.budget_tracker import BudgetTracker

    instance = BudgetTracker._instance
    keys = instance._api_keys if instance is not None else BudgetTracker._load_api_keys()
    return bool(keys)


def upload_notice() -> dict:
    """
    {"hosted_llm": provider or None, "upload_notice": text or None}.

    Says nothing about the keys themselves: only whether any is configured.
    """
    if not hosted_llm_configured():
        return {"hosted_llm": None, "upload_notice": None}
    text = os.getenv("UPLOAD_NOTICE", "").strip() or DEFAULT_UPLOAD_NOTICE
    return {"hosted_llm": HOSTED_PROVIDER, "upload_notice": text}
