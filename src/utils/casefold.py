"""
The ASCII lower-case form of a word that a re.IGNORECASE pattern matched.

Python's re.IGNORECASE compares by Unicode case mapping, so a pattern written
in ASCII also matches four non-ASCII letters: dotless ı (U+0131) and İ
(U+0130) match "i", long ſ (U+017F) matches "s", and the Kelvin sign
(U+212A) matches "k" (an exhaustive search of the code space finds no
others). str.lower() turns none of them into ASCII, so a dict keyed by the
pattern's own words raised KeyError on what the pattern had just accepted:
"fırst quarter of 2023" — a dotless ı from a Turkish keyboard — failed the
query, and a table header "(ın mıllıons)" — "IN MILLIONS" lower-cased under
Turkish rules — failed numeric grounding (Decision 88).
"""

_FOLD = str.maketrans({"ı": "i", "İ": "i", "ſ": "s", "K": "k"})


def ascii_lower(word: str) -> str:
    """`word` lower-cased, the letters re.IGNORECASE reads as i, s and k made ASCII."""
    return word.translate(_FOLD).lower()
