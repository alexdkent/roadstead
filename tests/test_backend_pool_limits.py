"""Phase 1 hardening — backend connection-pool sizing.

The historic flat ``max_connections=20`` starved the 32-slot thinker: at
>20 concurrent dispatches httpx queued requests on its own pool (pool=5.0s)
and failed them as PoolTimeout even though the backend had free slots. Pins:

  - the pool is sized to the caller's concurrency (effective_max_slots +
    headroom) with the old 20 as the floor;
  - a later, BIGGER requirement (slot discovery raised max_slots) retires
    the old client (kept open for its in-flight requests) and rebuilds;
  - call()/stream() pass their endpoint's slot count;
  - PoolTimeout maps to BackendUnavailable (transient → in-proxy retry +
    caller deferral), never a hard BackendError 502.
"""

from __future__ import annotations

import httpx
import pytest

from originfleet.llmproxy import backend as backend_mod
from originfleet.llmproxy.backend import (
    BackendClientPool,
    BackendUnavailable,
)
from originfleet.llmproxy.config import EndpointConfig


def _thinker_cfg(max_slots=32) -> EndpointConfig:
    return EndpointConfig(
        endpoint_class="thinker", role="llama-thinker",
        max_slots=max_slots, context_per_slot=131072,
        host="127.0.0.1", port=1, backend_engine="vllm",
    )


def test_pool_sized_to_slots_with_floor(monkeypatch):
    created: list[dict] = []
    real_client = httpx.AsyncClient

    def spy_client(**kwargs):
        created.append(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr(backend_mod.httpx, "AsyncClient", spy_client)
    pool = BackendClientPool()

    # Small endpoint: floor of 20 applies.
    pool._client_for("h1", 1, min_pool=3 + 4)
    assert created[-1]["limits"].max_connections == 20

    # Thinker-sized endpoint: pool >= admission ceiling.
    pool._client_for("h2", 1, min_pool=32 + 4)
    assert created[-1]["limits"].max_connections == 36
    assert created[-1]["limits"].max_keepalive_connections >= 18


def test_pool_grows_and_retires_old_client(monkeypatch):
    created: list[dict] = []
    real_client = httpx.AsyncClient

    def spy_client(**kwargs):
        created.append(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr(backend_mod.httpx, "AsyncClient", spy_client)
    pool = BackendClientPool()

    small = pool._client_for("h", 1, min_pool=0)          # built at 20
    same = pool._client_for("h", 1, min_pool=10)          # 20 suffices → reused
    assert same is small
    grown = pool._client_for("h", 1, min_pool=36)         # needs rebuild
    assert grown is not small
    assert created[-1]["limits"].max_connections == 36
    # Old client is retired (still open for in-flight), not closed in place.
    assert small in pool._retired
    # And a subsequent smaller requirement reuses the grown client.
    assert pool._client_for("h", 1, min_pool=5) is grown


@pytest.mark.asyncio
async def test_call_and_stream_request_slot_sized_pool():
    pool = BackendClientPool()
    seen: list[int] = []

    def spy(host, port, min_pool=0):
        seen.append(min_pool)
        raise RuntimeError("stop before any network I/O")

    pool._client_for = spy
    cfg = _thinker_cfg()

    with pytest.raises(RuntimeError):
        await pool.call(cfg, {"messages": []}, "chat_completion", "r1")
    agen = pool.stream(cfg, {"messages": []}, "chat_completion", "r2")
    with pytest.raises(RuntimeError):
        await agen.__anext__()
    assert seen == [36, 36]  # effective_max_slots + 4, both paths


@pytest.mark.asyncio
async def test_pool_timeout_maps_to_backend_unavailable():
    pool = BackendClientPool()
    cfg = _thinker_cfg()

    class _ExhaustedClient:
        async def post(self, *a, **k):
            raise httpx.PoolTimeout("pool exhausted")

    pool._client_for = lambda *a, **k: _ExhaustedClient()
    with pytest.raises(BackendUnavailable) as ei:
        await pool.call(cfg, {"messages": []}, "chat_completion", "r1")
    assert "pool exhausted" in str(ei.value)

    class _ExhaustedStreamClient:
        def stream(self, *a, **k):
            raise httpx.PoolTimeout("pool exhausted")

    pool._client_for = lambda *a, **k: _ExhaustedStreamClient()
    agen = pool.stream(cfg, {"messages": []}, "chat_completion", "r2")
    with pytest.raises(BackendUnavailable):
        await agen.__anext__()
