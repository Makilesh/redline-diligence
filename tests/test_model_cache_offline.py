"""
A Hub outage drill on a real cached model (Decision 84).

With the Hub unreachable, transformers 4.57 fails to load a large-vocabulary
tokenizer by Hub id (its Mistral-regex check calls model_info and does not
catch the error). The app's loader must still load it from the cache. The
bge-m3 tokenizer is the cheapest model that triggers the check (a 17 MB file,
no weights), so it stands in for the four models that do.

Runs in a subprocess, because transformers reads the endpoint and offline mode
at import. Needs the tokenizer in the local cache, so it is a `models` test
(deselected in CI) and skips when the file is not there.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.model_cache import cached_snapshot

pytestmark = pytest.mark.models

ROOT = Path(__file__).resolve().parent.parent
# A closed local port: every Hub request is refused, as in a DNS or network outage.
UNREACHABLE = {"HF_ENDPOINT": "http://127.0.0.1:9", "HF_HUB_DISABLE_TELEMETRY": "1"}


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, **UNREACHABLE, "PYTHONPATH": str(ROOT)}
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.mark.skipif(cached_snapshot("BAAI/bge-m3") is None, reason="bge-m3 is not in the local cache")
def test_the_app_loads_a_cached_tokenizer_with_the_hub_unreachable():
    done = _run(
        "from src.utils.token_counter import get_tokenizer, count_tokens\n"
        "get_tokenizer()\n"
        "print('tokens', count_tokens('Purchase price is $34.50 per share.'))\n"
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "tokens " in done.stdout
