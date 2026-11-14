"""A run that carries no verification verdict is never reported as verified."""

from api.routes.query import _build_response


def test_missing_validation_status_is_not_reported_as_passed():
    result = {"generated_answer": "Revenue was $452.8M [aurora_financials_fy2023.txt]."}
    response = _build_response(result, session_id="s1")
    assert response.validation_status != "passed"
    assert response.validation_status == "warning"


def test_a_stated_validation_status_is_reported_as_is():
    for status in ("passed", "warning", "failed"):
        result = {"generated_answer": "x", "validation_status": status}
        assert _build_response(result, session_id="s1").validation_status == status
