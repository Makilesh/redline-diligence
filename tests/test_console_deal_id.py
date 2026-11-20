"""
The Streamlit console accepts only a well-formed deal id from its text box and
its `?deal_id=` deep link: the id goes into the API paths the console requests
with the admin key (`/deals/{deal_id}/documents`, `/tie-out`, ...), so a value
holding "/", ".." or "?" could point those requests at another endpoint.
Found by the follow-up review of Decision 96.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.models.request_models import DEAL_ID_PATTERN  # noqa: E402
from app.components.deal_manager import _DEAL_ID, valid_deal_id  # noqa: E402

APP_PATH = str(_PROJECT_ROOT / "app" / "streamlit_app.py")


def test_the_console_uses_the_apis_own_deal_id_pattern():
    assert _DEAL_ID.pattern == DEAL_ID_PATTERN


@pytest.mark.parametrize("value", ["aurora_vertex_2024", "a", "A-1_b", "x" * 64, "  aurora_vertex_2024 "])
def test_a_well_formed_deal_id_is_kept(value):
    assert valid_deal_id(value) == value.strip()


@pytest.mark.parametrize("value", [
    "../budget", "x/../../budget", "aurora?include=all", "aurora#x", "a b", "x" * 65,
    "aurora%2F..%2Fbudget", "deal\nid", "délai",
])
def test_anything_else_is_refused(value):
    assert valid_deal_id(value) is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_no_value_is_no_deal(value):
    assert valid_deal_id(value) is None


def test_a_crafted_deep_link_selects_no_deal():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params["deal_id"] = "../../budget"
    at.run()
    assert not at.exception, at.exception
    assert any("not a deal id" in w.value for w in at.warning)
    # No deal selected: the uploader is not offered, the sidebar asks for a deal.
    assert any("Select or create a Deal ID" in i.value for i in at.info)


def test_a_well_formed_deep_link_still_selects_its_deal():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.query_params["deal_id"] = "aurora_vertex_2024"
    at.run()
    assert not at.exception, at.exception
    assert not any("not a deal id" in w.value for w in at.warning)
    assert not any("Select or create a Deal ID" in i.value for i in at.info)
