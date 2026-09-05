"""`?window=` on the analytics plane, pinned at the producer.

Two call sites passed `_clamp_window(s, default, *, cap=7*86400)` a bound in the
DEFAULT slot, where it was read as the fallback for an unparseable string and
then clamped by the untouched 7-day keyword cap (fixed 2026-09-05):

  * `/v1/fleet/cache-stats` passed `30 * 86400` positionally, so the cap stayed
    at a week. `?window=30d` parsed cleanly, clamped to 604800, and returned
    seven days of snapshots labelled `window_s: 604800` — a caller asking for a
    month of cache drift got a week and no error to read.
  * `/v1/fleet/cache-attribution` passed `7 * 86400` as its default, so
    `?window=abc` ran a WEEK-wide `GROUP BY` over `proxy_completions` instead of
    the documented hour. The widest scan on the plane, off a typo.

Both are invisible from the response: the payload echoes the window it actually
used, so the wrong number describes itself correctly. The assertions therefore
sit on what the PRODUCER was handed, driven through the real ASGI app so the
query-param parsing, the handler and the clamp are all in the path.

The cap-still-bites cases are asserted beside them. A fix that removed the cap
would satisfy the headline assertions and quietly hand an operator an unbounded
scan.
"""
from __future__ import annotations

import httpx
import pytest

from roadstead.config import ProxyConfig
from roadstead.__main__ import build_app


#: Loopback → the ACL's `internal` identity, as in `tests/e2e/conftest.py`.
_INTERNAL_CLIENT = ("127.0.0.1", 41999)


@pytest.fixture
def proxy_app(tmp_path):
    """The real app with a real (empty) queue DB, no startup.

    These handlers are pure reads: no poller, no scheduler loop and no backend
    is needed to reach the clamp, and not starting the service keeps the test
    off every timing-dependent path it would otherwise drag in.
    """
    app = build_app(ProxyConfig(queue_db_path=str(tmp_path / "queue.db")))
    try:
        yield app
    finally:
        app.state.proxy_service._queue_db.close()


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=app, raise_app_exceptions=True, client=_INTERNAL_CLIENT),
        base_url="http://proxy")


def _record(app, method: str, returns):
    """Replace one queue-DB producer with a recorder of its first argument."""
    seen: list = []

    def _fake(window_s, *args, **kwargs):
        seen.append(window_s)
        return returns

    setattr(app.state.proxy_service._queue_db, method, _fake)
    return seen


@pytest.mark.parametrize("window,expected", [
    ("30d", 30 * 86400),   # the regression: used to arrive as 604800
    ("7d", 7 * 86400),
    ("90d", 30 * 86400),   # the cap still bites, one month above it
    ("abc", 7 * 86400),    # unparseable → the documented 7d default
])
async def test_cache_stats_window_reaches_thirty_days(proxy_app, window, expected):
    seen = _record(proxy_app, "cache_stats_snapshots", [])
    async with _client(proxy_app) as client:
        resp = await client.get(f"/v1/fleet/cache-stats?window={window}")
    assert resp.status_code == 200
    assert seen == [expected]


@pytest.mark.parametrize("window,expected", [
    ("abc", 3600),         # the regression: used to arrive as 604800
    ("", 3600),
    ("6h", 6 * 3600),
    ("30d", 7 * 86400),    # this route's cap is a week, and stays one
])
async def test_cache_attribution_falls_back_to_one_hour(proxy_app, window, expected):
    seen = _record(proxy_app, "cache_attribution",
                   {"window_s": 0, "now": None, "by_call_site": [],
                    "by_endpoint": [], "fleet": None})
    async with _client(proxy_app) as client:
        resp = await client.get(f"/v1/fleet/cache-attribution?window={window}")
    assert resp.status_code == 200
    assert seen == [expected]


async def test_the_untouched_analytics_windows_keep_their_defaults(proxy_app):
    """The other three `_clamp_window` sites, asserted rather than eyeballed —
    the same argument-slot mistake in any of them would be just as silent."""
    for route, method, returns, expected in (
        ("/v1/fleet/activity", "fleet_activity",
         {"window_s": 0, "bin_s": 0, "calls": [], "by_endpoint_1h": []}, 86400),
        ("/v1/fleet/top-callers", "top_callers", {"window_s": 0, "providers": {}}, 3600),
    ):
        seen = _record(proxy_app, method, returns)
        async with _client(proxy_app) as client:
            assert (await client.get(route)).status_code == 200
        assert seen == [expected], route

    # /v1/series takes its window the same way but needs the required endpoint.
    seen: list = []

    def _series(endpoint, window_s, bin_s):
        seen.append(window_s)
        return {"endpoint": endpoint, "window_s": window_s, "bin_s": bin_s,
                "now": 0.0, "calls_series": []}

    proxy_app.state.proxy_service._queue_db.endpoint_series = _series
    async with _client(proxy_app) as client:
        assert (await client.get("/v1/series?endpoint=tier3")).status_code == 200
    assert seen == [86400]
