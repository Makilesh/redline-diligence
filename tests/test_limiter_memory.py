"""
The per-client limiter keeps only as much history as its limits can count.

A bucket with no daily cap (the read bucket: 120 a minute, no day limit) kept
every admitted hit for 24 hours anyway: one client staying just under the
minute limit held ~7,200 timestamps an hour (~5.6 MB a day), and each request
scanned them all on the event loop for the per-minute count.
"""
from __future__ import annotations

from api.security import _DAY, _MINUTE, _SlidingWindowLimiter


def _drive(limiter, key, per_minute, per_day, seconds, rate_per_minute):
    """Admits hits at a steady rate below the minute limit; returns the last time used."""
    step = _MINUTE / rate_per_minute
    now = 0.0
    while now < seconds:
        assert limiter.hit(key, per_minute=per_minute, per_day=per_day, now=now) is None
        now += step
    return now


def test_no_daily_cap_keeps_a_minute_of_history():
    limiter = _SlidingWindowLimiter()
    _drive(limiter, "read:1.2.3.4", per_minute=120, per_day=0, seconds=3600, rate_per_minute=119)
    assert len(limiter._hits["read:1.2.3.4"]) <= 120


def test_the_minute_limit_still_holds_without_a_daily_cap():
    limiter = _SlidingWindowLimiter()
    for i in range(5):
        assert limiter.hit("k", per_minute=5, per_day=0, now=100.0 + i) is None
    retry = limiter.hit("k", per_minute=5, per_day=0, now=110.0)
    assert retry is not None and 0 < retry <= _MINUTE
    assert limiter.hit("k", per_minute=5, per_day=0, now=100.0 + _MINUTE + 0.1) is None


def test_a_daily_cap_still_counts_the_whole_day():
    limiter = _SlidingWindowLimiter()
    for i in range(3):
        assert limiter.hit("k", per_minute=0, per_day=3, now=i * 3600.0) is None
    assert limiter.hit("k", per_minute=0, per_day=3, now=5 * 3600.0) is not None
    assert limiter.hit("k", per_minute=0, per_day=3, now=_DAY + 1.0) is None


def test_both_limits_off_records_nothing():
    limiter = _SlidingWindowLimiter()
    for i in range(1000):
        assert limiter.hit("k", per_minute=0, per_day=0, now=float(i)) is None
    assert not limiter._hits.get("k")
