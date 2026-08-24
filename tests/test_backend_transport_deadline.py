"""The transport read deadline must not be shorter than the request's deadline.

WHAT WENT WRONG (measured 2026-08-24, tier3 / DeepSeek-V4-Flash-0731). The
pooled backend client is built with a FLAT ``read=600.0``. A non-streaming call
was handed its real deadline through ``asyncio.wait_for`` — but httpx stopped
reading at 600s first, and a ``ReadTimeout`` is an ``httpx.HTTPError``, so the
call surfaced as ``BackendError(502)``. Two separate defects in one line:

  * a legitimately long generation could NEVER exceed 600s, whatever it
    declared. A native-reasoning song-authoring call (~15k prompt, 12k
    max_tokens after the proxy's own reasoning budget) was dispatched with a
    ~1230s deadline and died at exactly 600.0s, twice, with ~10 minutes still
    left on the scheduler's clock;
  * the failure was MISTYPED. A deadline the caller set is a timeout — a
    retryable, deferrable condition the caller can reason about. A 502 says the
    backend is broken, which it was not.

The fix is not "make the constant bigger": that would re-create the same cliff
one number further out, and the 600s floor is still right for a caller that
declares less. The transport deadline TRACKS the request's, plus a margin, so
``asyncio.wait_for`` is always the layer that fires.
"""

from __future__ import annotations

import httpx

from originfleet.llmproxy.backend import _transport_timeout


def test_a_short_deadline_keeps_the_historic_read_floor():
    """A caller with a small budget must not shrink the read below what the
    pool was built for — that would turn a slow-but-fine backend into a fault."""
    t = _transport_timeout(180.0)
    assert isinstance(t, httpx.Timeout)
    assert t.read == 600.0


def test_a_long_deadline_extends_the_read_past_the_flat_600():
    """The case that was broken: 1230s declared, 600s enforced."""
    t = _transport_timeout(1230.0)
    assert t.read > 600.0, (
        "the transport still cuts at the flat 600s — a declared deadline above "
        "it is unreachable, and the failure surfaces as a 502 rather than a "
        "timeout")
    assert t.read > 1230.0, (
        "the transport must outlast the request deadline, so asyncio.wait_for "
        "fires first and the failure is typed as a TIMEOUT")


def test_connect_and_pool_deadlines_are_untouched():
    """Only the READ deadline is deadline-dependent. Widening connect/pool would
    change how fast an unreachable or saturated backend is detected, which is a
    different decision with a different blast radius."""
    for budget in (10.0, 600.0, 2400.0):
        t = _transport_timeout(budget)
        assert t.connect == 5.0
        assert t.pool == 5.0
        assert t.write == 10.0


def test_the_read_deadline_grows_monotonically_with_the_request():
    reads = [_transport_timeout(b).read for b in (60.0, 600.0, 1200.0, 2400.0)]
    assert reads == sorted(reads)
    assert reads[-1] >= 2400.0
