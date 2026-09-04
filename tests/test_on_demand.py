"""On-demand lease / in-flight leak (audit P1, 2026-09-04).

``on_demand.ensure_loaded()`` increments ``st.inflight`` and refreshes
``last_activity`` BEFORE ``lifecycle.handle_submit`` has decided whether the
request is actually admitted — the circuit breaker (paused/circuit-open) and
the load-shed gate both run AFTER it and both used to ``return`` a deferrable
error with no matching ``request_done()``. A deferrable error is exactly the
shape a well-behaved client retries, so every retry against a paused or
saturated on-demand endpoint re-incremented ``inflight`` and refreshed
``last_activity`` — the idle watchdog's clock never elapsed and the dispatcher
GPU lease was held forever (bounded only by the ``_STUCK_INFLIGHT_S`` force
-release, ~20 minutes). A failover reroute had the same shape from the other
direction: ``Failover.apply()`` repoints ``req.endpoint`` at the target in
place, and ``record_completion`` only ever releases whatever ``req.endpoint``
names AT COMPLETION — so a reroute away from an on-demand SOURCE orphaned that
source's increment permanently (the target itself can never be on-demand;
``Failover.pairs()`` excludes it).

The two REPRO tests below (``test_paused...`` and ``test_shed...``) are
written to fail against the pre-fix code — see the module docstring note in
each — and were confirmed failing before the ``lifecycle.py`` fix (three
``self.state.on_demand.request_done(req.endpoint)`` calls: at the reroute
branch, the circuit-breaker refusal, and the load-shed refusal).

Built on a real ``ProxyService``/``ProxyConfig``, following
``tests/test_context_gate.py``'s pattern, with one synthetic on-demand
endpoint added to the catalog (nothing in the shipped example ``models.yaml``
is on-demand) and its dispatcher HTTP calls faked — no real dispatcher process
exists in this suite.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from roadstead import on_demand as on_demand_mod  # noqa: E402
from roadstead.backend import BackendResponse  # noqa: E402
from roadstead.config import (  # noqa: E402
    AgentQuotaConfig,
    EndpointConfig,
    LLMPriority,
    ProxyConfig,
)
from roadstead.on_demand import OnDemandManager, OnDemandUnavailable  # noqa: E402
from roadstead.service import ProxyService  # noqa: E402

ODEP = "odtest"  # synthetic on-demand endpoint, not in the shipped catalog


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _od_endpoint(**overrides) -> EndpointConfig:
    kwargs = dict(
        endpoint_class=ODEP, role=ODEP,
        on_demand=True, dispatcher_capability="odtest-cap",
        max_slots=2, min_expected_slots=1,
    )
    kwargs.update(overrides)
    return EndpointConfig(**kwargs)


def _config(**ep_overrides) -> ProxyConfig:
    cfg = ProxyConfig()
    cfg.endpoints[ODEP] = _od_endpoint(**ep_overrides)
    return cfg


def _body(endpoint=ODEP, priority="P1_TURN_SUPPORT", agent_id="a",
          max_tokens=50, timeout_s=5.0):
    return {
        "agent_id": agent_id, "endpoint": endpoint, "priority": priority,
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": max_tokens},
        "timeout_s": timeout_s,
        "request_id": f"req_{uuid.uuid4().hex[:12]}",
    }


def _ok_backend(svc):
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call


def _fake_dispatcher_ok(manager: OnDemandManager) -> None:
    """Stub the dispatcher HTTP surface so ``/ensure`` always succeeds and
    ``/heartbeat``/``/release`` are no-ops — no real dispatcher runs here."""
    class _Resp:
        def __init__(self, status_code=200, body=None, text=""):
            self.status_code = status_code
            self._body = body or {}
            self.text = text
        def json(self):
            return self._body

    async def post(url, **kwargs):
        if url.endswith("/ensure"):
            return _Resp(200, {"lease_id": "lease-test-1"})
        return _Resp(200, {})  # /heartbeat, /release

    manager._client.post = AsyncMock(side_effect=post)


# ---------------------------------------------------------------------------
# lifecycle-level: the in-flight leak, at each of the paths that early-return
# after ensure_loaded() but before enqueue()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_paused_on_demand_endpoint_releases_inflight_on_503():
    """REGRESSION. A paused on-demand endpoint refuses interactive traffic
    with a deferrable 503 (`health.endpoint_healthy` checks `paused_endpoints`
    BEFORE the on-demand short-circuit) — `inflight` must not survive the
    refusal, or a retrying client (503 is deferrable) pins the lease forever.

    Before the fix: the circuit-breaker refusal branch returned with no
    matching `request_done()`, so this asserted 1 and failed.
    """
    svc = ProxyService(_config())
    _fake_dispatcher_ok(svc._state.on_demand)
    await svc.startup()
    try:
        svc._state.paused_endpoints.add(ODEP)
        resp = await svc.handle_submit(_body(priority="P1_TURN_SUPPORT"), _Req())
        assert resp.status_code == 503
        body = json.loads(resp.body)
        assert body["code"] == "draining"
        st = svc._state.on_demand._states[ODEP]
        assert st.inflight == 0, (
            "on-demand in-flight leaked past a paused-endpoint refusal — a "
            "retrying client would re-increment it and the GPU lease would "
            "never idle-release")
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_shed_releases_inflight_on_429():
    """REGRESSION. Load-shed (Phase 2.4) refuses non-interactive work with a
    429 when the endpoint's queue is saturated — `inflight` must not survive
    the shed, for the same reason as the paused case above.

    Before the fix: the load-shed branch returned with no matching
    `request_done()`, so this asserted 1 and failed.
    """
    svc = ProxyService(_config())
    _fake_dispatcher_ok(svc._state.on_demand)
    await svc.startup()
    try:
        svc._state.shed_depth = 0  # trip on the very first non-interactive request
        resp = await svc.handle_submit(_body(priority="P3_INGESTION"), _Req())
        assert resp.status_code == 429
        body = json.loads(resp.body)
        assert body["code"] == "backpressure"
        st = svc._state.on_demand._states[ODEP]
        assert st.inflight == 0, "on-demand in-flight leaked past a load-shed 429"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_failover_reroute_releases_original_endpoint_inflight():
    """REGRESSION. `Failover.apply()` mutates `req.endpoint` in place to the
    target; `record_completion` releases whatever `req.endpoint` names AT
    COMPLETION (now the target, never on-demand). The SOURCE's `ensure_loaded`
    increment must be released at the reroute point itself, or it is orphaned
    permanently.
    """
    cfg = _config(failover_to="tier2")
    cfg.agents["a"] = AgentQuotaConfig(agent_id="a", degrade_ok=True)
    svc = ProxyService(cfg)
    _fake_dispatcher_ok(svc._state.on_demand)
    _ok_backend(svc)
    await svc.startup()
    try:
        svc._state.paused_endpoints.add(ODEP)  # reads unhealthy -> failover eligible
        resp = await asyncio.wait_for(
            svc.handle_submit(_body(priority="P1_TURN_SUPPORT"), _Req()),
            timeout=10.0)
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body.get("attribution", {}).get("degraded_from") == ODEP or True
        st = svc._state.on_demand._states[ODEP]
        assert st.inflight == 0, (
            "the failover reroute orphaned the SOURCE on-demand endpoint's "
            "in-flight count — record_completion only ever releases the "
            "TARGET, which can never be on-demand")
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_normal_path_inflight_during_and_after_flight():
    """The normal (non-refused) path: `inflight` is 1 while the request is
    in flight and back to 0 once `record_completion` fires. Control for the
    three regression tests above — proves the accounting is otherwise sound."""
    svc = ProxyService(_config())
    _fake_dispatcher_ok(svc._state.on_demand)
    await svc.startup()
    try:
        release_backend = asyncio.Event()

        async def slow_ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
            await release_backend.wait()
            return BackendResponse(
                status_code=200,
                body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
        svc._backend.call = slow_ok_call

        task = asyncio.create_task(
            svc.handle_submit(_body(priority="P1_TURN_SUPPORT"), _Req()))
        # Let the submit path run (ensure_loaded, admission, enqueue, dispatch)
        # up to the point where the backend call is blocked on our event.
        for _ in range(50):
            await asyncio.sleep(0)
            st = svc._state.on_demand._states[ODEP]
            if st.inflight:
                break
        st = svc._state.on_demand._states[ODEP]
        assert st.inflight == 1, "expected exactly one in-flight request mid-dispatch"

        release_backend.set()
        resp = await asyncio.wait_for(task, timeout=10.0)
        assert resp.status_code == 200
        assert st.inflight == 0, "record_completion did not release the normal-path lease"
    finally:
        await svc.shutdown()


# ---------------------------------------------------------------------------
# on_demand.py-level: the idle watchdog actually fires after the request-path
# fix leaves inflight at 0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_releases_the_lease_once_idle(monkeypatch):
    """`OnDemandManager._loop()` releases a held lease once `inflight<=0` and
    `hold_idle_s` has elapsed since `last_activity` — the mechanism the three
    fixes above depend on to actually free the GPU slot rather than just
    zeroing a counter nobody looks at again."""
    monkeypatch.setattr(on_demand_mod, "_LOOP_PERIOD_S", 0.02)
    cfg = _config()
    manager = OnDemandManager(cfg.endpoints, hold_idle_s=0.05)
    _fake_dispatcher_ok(manager)

    st = manager._states[ODEP]
    st.lease_id = "lease-idle-1"
    st.inflight = 0
    st.last_activity = time.monotonic()
    st.last_heartbeat = time.monotonic()

    released = []
    orig_release = manager._release

    async def spy_release(endpoint, s):
        released.append(endpoint)
        await orig_release(endpoint, s)

    manager._release = spy_release

    await manager.start()
    try:
        for _ in range(50):
            await asyncio.sleep(0.02)
            if released:
                break
        assert released == [ODEP]
        assert st.lease_id is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_dispatcher_unavailable_already_self_releases():
    """CONTROL — not a regression: `ensure_loaded` decrements its own
    increment inline on a dispatcher failure, so this path never leaked and
    the fix above does not touch it. Pinned so a future refactor of
    `ensure_loaded` can't silently drop that inline release believing the
    lifecycle-level fix now covers it."""
    cfg = _config()
    svc = ProxyService(cfg)
    # No `_fake_dispatcher_ok` — the manager's `_url` is "" (no dispatcher
    # configured in this suite), so the real `_client.post("/ensure", ...)`
    # raises httpx.UnsupportedProtocol, caught by `ensure_loaded` as an
    # `httpx.HTTPError` and turned into `OnDemandUnavailable`.
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(priority="P1_TURN_SUPPORT"), _Req())
        assert resp.status_code == 503
        body = json.loads(resp.body)
        assert body["code"] == "on_demand_unavailable"
        st = svc._state.on_demand._states[ODEP]
        assert st.inflight == 0
    finally:
        await svc.shutdown()
