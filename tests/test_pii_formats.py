"""
PII formats the regex used to miss (audit I11), and the business text it must
keep passing. Strings are the audit's probes plus the hard negatives the new
patterns were checked against; numbers are specimen values.
"""

import pytest

from src.data_processing.pii_detector import PIIDetector


@pytest.fixture(scope="module")
def detector():
    return PIIDetector()


@pytest.mark.parametrize(
    "text, pii_type",
    [
        ("Social Security Number 123.45.6789", "ssn"),
        ("SSN 123 45 6789", "ssn"),
        ("She will receive an annual base salary of $525,000 and a target bonus of $310,000.",
         "salary_data"),
        ("Name      Title   Base Salary   Cash Bonus\nJane Roe  CFO       $445,000     $267,000",
         "salary_data"),
        ("Account No. 4532-1234-5678-9010", "bank_account"),
        ("Wire to IBAN GB82 WEST 1234 5698 7654 32", "bank_account"),
        ("Card on file: 4111 1111 1111 1111", "payment_card"),
    ],
)
def test_previously_missed_formats_are_flagged(detector, text, pii_type):
    result = detector.detect(text)
    assert result.contains_pii == 1
    assert pii_type in result.pii_types


@pytest.mark.parametrize(
    "text",
    [
        # Aggregates headed by pay words.
        "Stock-based compensation of $4.2 million was added back.",
        "The bonus pool of $3.1 million is accrued in Q4.",
        "Bonus accrual (FY2023): $1,515,000 total for the executive team",
        "any employee whose annual compensation is in excess of $250,000;",
        "Compensation Above Market (Note 4)      $1.9            $1.9",
        # Reference numbers that fail the checksums.
        "Order reference 4532 1234 5678 9011",
        "Purchase order 1234567890123 issued March 2024",
        "Section 2.1(b)(iii) applies to GB82 transfers",
        "Invoice 123456789",
    ],
)
def test_business_text_is_not_flagged(detector, text):
    assert detector.detect(text).contains_pii == 0


def test_millions_cannot_be_backtracked_into_a_salary(detector):
    # Without the atomic group the engine retries "$4.2 million" as "$4" and
    # the not-millions lookahead then passes.
    assert detector.detect("Her bonus was $4.2 million in aggregate").contains_pii == 0
    assert detector.detect("His bonus was $42,000.").contains_pii == 1


def test_checksums(detector):
    assert PIIDetector._valid_iban("GB82WEST12345698765432")
    assert not PIIDetector._valid_iban("GB82WEST12345698765433")
    assert PIIDetector._luhn_valid("4111 1111 1111 1111")
    assert not PIIDetector._luhn_valid("4111 1111 1111 1112")
    # A run of one digit passes Luhn trivially for zeros; it is never a card.
    assert not PIIDetector._luhn_valid("0000 0000 0000 0000")


def test_payment_card_is_a_strong_type():
    # Checksum-validated, so Laya (LAYA_PII=1) must not overrule it.
    result = PIIDetector().detect("Card on file: 4111 1111 1111 1111", laya_probability=0.0)
    assert result.contains_pii == 1
