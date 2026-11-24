"""
The Streamlit console says where uploaded text goes before an upload (Decision 116).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

pytest.importorskip("streamlit")

import app.components.document_uploader as uploader  # noqa: E402

NOTICE = "Passages from uploaded documents are sent to Google's Gemini API."


def _serve(monkeypatch, status=200, body=None, error=None):
    def get(url, timeout):
        assert url == "http://engine.test/api/v1/data-handling" and timeout <= 5
        if error:
            raise error
        return SimpleNamespace(status_code=status, json=lambda: body)

    monkeypatch.setattr(uploader.requests, "get", get)


def test_the_servers_wording_when_a_hosted_model_is_configured(monkeypatch):
    _serve(monkeypatch, body={"hosted_llm": "Google Gemini API", "upload_notice": NOTICE})
    assert uploader.data_handling_notice("http://engine.test/api/v1") == NOTICE


def test_nothing_when_text_stays_on_the_server(monkeypatch):
    _serve(monkeypatch, body={"hosted_llm": None, "upload_notice": None})
    assert uploader.data_handling_notice("http://engine.test/api/v1") is None


@pytest.mark.parametrize("kwargs", [
    {"status": 404, "body": {"detail": "Not Found"}},  # a server from before the endpoint
    {"status": 200, "body": ["not", "an", "object"]},
    {"error": ConnectionError("refused")},
], ids=["old-server", "malformed", "unreachable"])
def test_assumes_the_text_leaves_when_the_server_cannot_say(monkeypatch, kwargs):
    _serve(monkeypatch, **kwargs)
    assert uploader.data_handling_notice("http://engine.test/api/v1") == uploader.DATA_HANDLING_UNKNOWN


def test_it_is_shown_above_the_file_picker(monkeypatch):
    from streamlit.testing.v1 import AppTest

    _serve(monkeypatch, body={"hosted_llm": "Google Gemini API", "upload_notice": NOTICE})

    def script():
        from app.components.document_uploader import render_document_uploader

        render_document_uploader("http://engine.test/api/v1", "deal-1")

    at = AppTest.from_function(script, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert [w.value for w in at.warning] == [NOTICE]


def test_a_parser_child_still_starts_after_an_app_test():
    # Runs after the AppTest above. AppTest leaves its generated script installed as
    # __main__, which a spawned child runs first and dies on, unless conftest's
    # _main_module_is_restored puts the real one back.
    import os

    import src.data_processing.parse_isolation as iso

    assert iso.warm_up() != os.getpid()
