"""
A question's model work goes before an upload's embedding (Decision 124).

On the CPU host the models share two cores. While ingest embedded a document's chunks
in one call, a question's retrieval model work (embedding its search texts, reranking
twenty passages) took 6.5-9.5 s at the median and up to 20 s, against 0.33 s idle
(emulated two-core host, 48 chunks of a 10-K). A call already running cannot be
paused, so ingest embeds in small slices instead and, before each, waits while a
question's model call is running. The wait is capped: questions arriving back to back
must not hold an upload back for good.

Model calls made for a question run inside `foreground()`; ingest calls
`wait_for_foreground()` between slices. The count is shared by every thread of the
process (the wrappers run on the event loop, the models on executor threads).
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import Iterator

_lock = threading.Lock()
_active = 0


@contextlib.contextmanager
def foreground() -> Iterator[None]:
    """Marks a question's model call as running for as long as the block runs."""
    global _active
    with _lock:
        _active += 1
    try:
        yield
    finally:
        with _lock:
            _active -= 1


def foreground_active() -> bool:
    """Whether a question's model call is running now."""
    return _active > 0


async def wait_for_foreground(max_wait_s: float, poll_s: float = 0.05) -> float:
    """
    Waits while a question's model call is running, for at most max_wait_s.

    Polls rather than waiting on an asyncio.Event: an Event binds to one event
    loop, and the count is changed from whichever loop or thread runs the call.

    Returns:
        The seconds waited.
    """
    start = time.monotonic()
    while _active and time.monotonic() - start < max_wait_s:
        await asyncio.sleep(poll_s)
    return time.monotonic() - start
