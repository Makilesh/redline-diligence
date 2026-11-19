"""
Hypothesis profiles and text strategies for the property tests (Decision 88).

Profiles (HYPOTHESIS_PROFILE selects one):
    redline   the default: few examples, derandomized, no example database, so
              the offline suite stays fast and a run is reproducible.
    fuzz      a heavier search for a gated job:
              HYPOTHESIS_PROFILE=fuzz pytest tests/test_fuzz_properties.py

Plain random text almost never reaches the code worth testing: a filing-period
pattern needs "first quarter of 2023", a party clause needs "by and between".
So the strategies below build text from the vocabulary the regexes key on,
mixed with what real documents and questions bring that the regexes did not
expect — characters that only match case-insensitively (dotless ı, İ, long ſ,
the Kelvin sign), digits of other scripts, non-breaking and unusual
whitespace, long digit runs and stray brackets.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

settings.register_profile(
    "redline",
    max_examples=50,
    derandomize=True,
    database=None,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "fuzz",
    max_examples=2_000,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "redline"))

# Characters re.IGNORECASE matches to an ASCII letter although .lower() does
# not turn them into it: ı (U+0131) and İ (U+0130) are "i", ſ (U+017F) is "s",
# the Kelvin sign (U+212A) is "k".
CASE_LOOKALIKES = "ıİſK"
# Decimal digits of other scripts (all match \d): Arabic-Indic, fullwidth, NKo.
OTHER_DIGITS = "٣٤３߁"
ODD_SPACE = "\xa0  \t\r\x0b\x0c"

VOCABULARY = (
    # Contracts and identity
    "AGREEMENT", "Agreement", "THIS", "This", "between", "by and between", "among", "and",
    "Inc.", "LLC", "S.p.A.", "SpA", "plc", "Ltd", "a Delaware corporation", "(\"Company\")",
    "Share Purchase Agreement", "SPA", "NDA", "Non-Disclosure", "MUTUAL", "Merger Sub",
    "WHEREAS", "ARTICLE IV", "Section 4.3", "(a)", "(iv)", "1.", "12.1.2", "EXHIBIT 2.1",
    # Periods and figures
    "FY2023", "FY 22", "Q3", "2Q22", "fiscal 2021", "first quarter of 2023", "second", "fourth",
    "December 31, 2022", "Aug", "August", "$452.8", "($1.8)", "452,800", "12.5%", "7.5x", "bps",
    "million", "millions", "(in millions)", "in thousands of", "USD m", "EBITDA", "Revenue:",
    "FY2023=452,800,000", "[report.pdf | p. 3]", "[a.txt | Section 2]", "|",
    # PII and injection
    "SSN", "123-45-6789", "Social Security Number", "salary", "$165,000", "Account No.",
    "jane@gmail.com", "IGNORE ALL PREVIOUS INSTRUCTIONS", "SYSTEM:", "<|im_start|>",
    "note to the AI", "all other documents are superseded",
    "[", "]", "[📄 a.pdf |", "| p.3]", "[INST]", "</document>", "< /Document >", "<finding", "COMPUTED:",
    # Filing covers
    "SECURITIES AND EXCHANGE COMMISSION", "Washington", "FORM 10-K", "FORM 10-Q",
    "For the fiscal year ended", "quarterly period ended", "Commission file number 1-3285",
    "(Exact name of registrant", "three months ended", "nine months ended", "(NYSE: ABC)",
    "results", "reports",
)

_ALPHABET = st.characters(
    codec="utf-8",
    categories=("L", "N", "P", "S", "Zs"),
) | st.sampled_from(CASE_LOOKALIKES + OTHER_DIGITS + ODD_SPACE + "\n\n()[]|.,;:-_/\\\"'$%")


def lookalike(text: str) -> str:
    """The same text with i, s and k replaced by characters that match them case-insensitively."""
    return text.replace("i", "ı").replace("s", "ſ").replace("k", "K")


# Lengths around the limits that matter: int() refuses more than 4,300 digits;
# the identity extractor caps titles at 120 characters and names at 100.
RUN_LENGTHS = st.sampled_from([1, 2, 3, 4, 7, 99, 121, 201, 4_301, 6_000])


def words() -> st.SearchStrategy[str]:
    """One word: domain vocabulary (sometimes in lookalike letters), a long run, or arbitrary characters."""
    return st.one_of(
        st.sampled_from(VOCABULARY),
        st.sampled_from(VOCABULARY).map(lookalike),
        st.text(_ALPHABET, min_size=1, max_size=8),
        st.tuples(st.sampled_from("7A"), RUN_LENGTHS).map(lambda t: t[0] * t[1]),
    )


def domain_text(max_words: int = 40) -> st.SearchStrategy[str]:
    """Text assembled from words() with mixed separators (spaces, line breaks, odd whitespace)."""
    separator = st.sampled_from([" ", " ", " ", "\n", "\n\n", ", ", "\xa0", "\t", ""])
    return st.lists(st.tuples(words(), separator), max_size=max_words).map(
        lambda parts: "".join(w + s for w, s in parts)
    )
