# src/data_processing/excel_normalizer.py
"""
Excel scale and currency normalization for financial data.

Detects unit scale (thousands, millions, billions) and currency from
column/row headers before value extraction.

CRITICAL: Without normalization, cross-document comparison produces false
inconsistency alarms. A value of 45,200 (scale=thousands) and 45.2 (scale=millions)
are identical when normalized — the Financial Verifier must compare normalized
values only, never raw values across documents.

All Financial Verification Agent comparisons MUST use normalized_value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ─── Scale detection patterns (from p4.md spec) ──────────────────────────────
# Checked in this order, first match wins. Abbreviations are anchored to a
# currency sign or code: a bare "MM\b" also matched the end of "Telecomm" and
# "m"/"k" alone would match almost any header.
_CCY = r"(?:\$|€|£|¥|₹|\b(?:usd|eur|gbp|cad|aud|chf|jpy|inr|cny))"
SCALE_PATTERNS: dict[str, float] = {
    rf"in\s+thousands?|\(\s*thousands\s*\)|{_CCY}\s*0{{3}}s?\b|\b0{{3}}s\b|\(\s*0{{3}}'?s?\s*\)"
    rf"|{_CCY}\s*(?:k|thousands?)\b|×\s*1[,.]?000(?!\s*[,.]?000)": 1_000.0,
    rf"in\s+millions?|\(\s*millions\s*\)|{_CCY}\s*(?:mm?|mn|millions?)\b|\bmm\b"
    rf"|×\s*1[,.]?000[,.]?000(?!\s*[,.]?000)": 1_000_000.0,
    rf"in\s+billions?|\(\s*billions\s*\)|{_CCY}\s*(?:bn?|billions?)\b"
    rf"|×\s*1[,.]?000[,.]?000[,.]?000": 1_000_000_000.0,
}

# Currency markers, most specific first. Codes are matched as whole words:
# a substring test read "Audited" as AUD, "Europe" as EUR and "Cadence" as CAD.
_CURRENCY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"c\$|\bcad\b", "CAD"),
    (r"a\$|\baud\b", "AUD"),
    (r"hk\$|\bhkd\b", "HKD"),
    (r"s\$|\bsgd\b", "SGD"),
    (r"\$|\busd\b|\bus dollars?\b", "USD"),
    (r"€|\beur\b|\beuros?\b", "EUR"),
    (r"£|\bgbp\b|\bsterling\b", "GBP"),
    (r"¥|\bjpy\b|\byen\b", "JPY"),
    (r"\bchf\b", "CHF"),
    (r"₹|\binr\b", "INR"),
    (r"\bcny\b|\brmb\b|元", "CNY"),
)


@dataclass
class TableNormalizationMeta:
    """
    Metadata about detected scale and currency for a financial table.

    Attributes:
        currency: ISO 4217 currency code (e.g., "USD", "EUR") or "UNKNOWN".
        scale_factor: Multiplier for raw values (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0).
        scale_label: Human-readable scale ("units", "thousands", "millions", "billions").
    """
    currency: str           # "USD", "EUR", "GBP", etc.
    scale_factor: float     # 1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0
    scale_label: str        # "units", "thousands", "millions", "billions"


class ExcelNormalizer:
    """
    Detects unit scale and currency from column/row headers before value extraction.

    CRITICAL: Without normalization, cross-document comparison produces false
    inconsistency alarms. A value of 45,200 (scale=thousands) and 45.2 (scale=millions)
    are identical when normalized — the Financial Verifier must compare normalized
    values only, never raw values across documents.

    All Financial Verification Agent comparisons MUST use normalized_value.
    """

    def detect_scale(self, header_cells: list[str]) -> TableNormalizationMeta:
        """
        Scans header cells for scale indicators.

        Returns TableNormalizationMeta with scale_factor and scale_label.
        Defaults to scale_factor=1.0 if no indicator found.

        Args:
            header_cells: List of header cell values from the table.

        Returns:
            TableNormalizationMeta with detected currency, scale_factor, and scale_label.
        """
        logger.info(
            "Detecting scale from headers",
            extra={"num_headers": len(header_cells)},
        )

        combined = " ".join(str(c) for c in header_cells if c is not None).lower()

        # The scale stated FIRST is the table's. Apple's 10-K caption reads "(In
        # millions, except number of shares, which are reflected in thousands,
        # and per-share amounts)": taking patterns in dictionary order found
        # "in thousands" and made every amount a thousand times too small.
        found = [
            (match.start(), pattern)
            for pattern in SCALE_PATTERNS
            if (match := re.search(pattern, combined, re.IGNORECASE))
        ]
        first = min(found)[1] if found else None

        for pattern, factor in SCALE_PATTERNS.items():
            if pattern == first:
                label = {
                    1_000.0: "thousands",
                    1_000_000.0: "millions",
                    1_000_000_000.0: "billions",
                }.get(factor, "units")

                result = TableNormalizationMeta(
                    currency=self._detect_currency(combined),
                    scale_factor=factor,
                    scale_label=label,
                )
                logger.info(
                    "Scale detected",
                    extra={
                        "scale_factor": result.scale_factor,
                        "scale_label": result.scale_label,
                        "currency": result.currency,
                    },
                )
                return result

        result = TableNormalizationMeta(
            currency=self._detect_currency(combined),
            scale_factor=1.0,
            scale_label="units",
        )
        logger.info(
            "No scale indicator found, defaulting to units",
            extra={"currency": result.currency},
        )
        return result

    def _detect_currency(self, text: str) -> str:
        """
        Detects currency from header text.

        Returns ISO 4217 code or 'UNKNOWN'.
        Return UNKNOWN instead of silently assuming USD.
        The Financial Verifier will flag UNKNOWN currencies for analyst review
        rather than producing silent misclassification in cross-border deals.

        Args:
            text: Combined header text (already lowercased).

        Returns:
            ISO 4217 currency code string.
        """
        lowered = text.lower()
        for pattern, code in _CURRENCY_PATTERNS:
            if re.search(pattern, lowered):
                return code
        # Return UNKNOWN instead of silently assuming USD.
        # The Financial Verifier will flag UNKNOWN currencies for analyst review
        # rather than producing silent misclassification in cross-border deals.
        return "UNKNOWN"

    def normalize_value(
        self,
        raw_value: float | int | str,
        meta: TableNormalizationMeta,
    ) -> dict:
        """
        Normalizes a single raw value using detected scale and currency.

        CRITICAL: Never round or approximate verbatim numbers.
        Store raw_value, normalized_value, currency, scale_factor in payload.

        Args:
            raw_value: The original value from the spreadsheet.
            meta: The normalization metadata for this table.

        Returns:
            Dict with raw_value, normalized_value, currency, scale_factor.

        Raises:
            ValueError: If raw_value cannot be converted to a numeric type.
        """
        numeric_value = self._parse_numeric(raw_value)

        return {
            "raw_value": raw_value,
            "normalized_value": numeric_value * meta.scale_factor,
            "currency": meta.currency,
            "scale_factor": meta.scale_factor,
            "scale_label": meta.scale_label,
        }

    def _parse_numeric(self, value: float | int | str) -> float:
        """
        Parse a potentially formatted numeric value into a float.

        Handles common formatting: commas, parentheses for negatives,
        percentage signs, currency symbols.

        Args:
            value: Raw value to parse.

        Returns:
            Parsed float value.

        Raises:
            ValueError: If value cannot be parsed as numeric.
        """
        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            raise ValueError(f"Cannot parse non-string/non-numeric value: {type(value)}")

        cleaned = value.strip()

        # Handle empty or dash-only values
        if not cleaned or cleaned in ("-", "—", "–", "N/A", "n/a", "-", ""):
            return 0.0

        # Remove currency symbols
        for symbol in ("$", "€", "£", "¥", "₹"):
            cleaned = cleaned.replace(symbol, "")

        # Handle percentage
        is_percentage = cleaned.endswith("%")
        if is_percentage:
            cleaned = cleaned[:-1]

        # Handle parentheses for negative numbers: (1,234) → -1234
        is_negative = cleaned.startswith("(") and cleaned.endswith(")")
        if is_negative:
            cleaned = cleaned[1:-1]

        # Remove commas and whitespace
        cleaned = cleaned.replace(",", "").replace(" ", "").strip()

        if not cleaned:
            return 0.0

        try:
            result = float(cleaned)
            if is_negative:
                result = -result
            if is_percentage:
                result = result / 100.0
            return result
        except ValueError:
            raise ValueError(f"Cannot parse '{value}' as numeric") from None
