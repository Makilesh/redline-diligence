"""
An asyncio event loop that runs on a virtual clock.

Deadlines measured in tens of seconds cannot be tested in real time, and
patching asyncio.sleep alone does not move the clock asyncio.wait / wait_for
read. This loop replaces both at the source: `loop.time()` is a counter, and
when nothing is ready the selector jumps the counter straight to the next
timer instead of blocking. Every sleep, timeout and deadline therefore runs
instantly and in the right order, and `loop.time()` reports the wall time the
code would have taken.

Only for code with no real I/O (scripted providers, in-memory tracker): a
coroutine waiting on a socket or a thread would wait for ever, and select()
raises instead of hanging when nothing at all is scheduled.
"""

import asyncio
import selectors


class _VirtualSelector(selectors.BaseSelector):
    def __init__(self, clock: list[float]):
        self._clock = clock
        self._keys: dict = {}

    def register(self, fileobj, events, data=None):
        key = selectors.SelectorKey(fileobj, fileobj if isinstance(fileobj, int)
                                    else fileobj.fileno(), events, data)
        self._keys[fileobj] = key
        return key

    def unregister(self, fileobj):
        return self._keys.pop(fileobj)

    def select(self, timeout=None):
        if timeout is None:
            raise RuntimeError(
                "virtual-time loop is idle with nothing scheduled: the code under "
                "test is waiting on real I/O or on an event nobody will set"
            )
        if timeout > 0:
            self._clock[0] += timeout
        return []

    def get_map(self):
        return self._keys

    def close(self):
        self._keys.clear()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """SelectorEventLoop whose time() is virtual and whose idle waits are free."""

    def __init__(self):
        self._virtual_now = [0.0]
        super().__init__(selector=_VirtualSelector(self._virtual_now))

    def time(self) -> float:
        return self._virtual_now[0]


def run_virtual(coro):
    """
    Runs `coro` to completion on a fresh virtual-time loop.

    Returns:
        (result, elapsed virtual seconds). Exceptions propagate.
    """
    loop = VirtualTimeLoop()
    try:
        start = loop.time()
        result = loop.run_until_complete(coro)
        return result, loop.time() - start
    finally:
        loop.close()
