"""Phase 0 hardening — critical-loop resilience + liveness surfacing.

The scheduler loop and capacity poller are single points of failure that
uvicorn happily serves 503s through (the process never exits, so the
run_agent.sh respawner never fires). These tests pin:

  - a poisoned scheduler tick does NOT kill dispatching (guarded iteration);
  - a poisoned poller chore does NOT kill probing/alerting (guarded iteration);
  - a genuinely dead loop is VISIBLE: /health degrades, /v1/status grows a
    read-time CRITICAL alert (the poller can't report its own death through
    the poller-evaluated alert list);
  - skip_discovery endpoints (embed/rerank FastAPI shims) probe /health only —
    no /props ÷ /v1/models 404 churn — and the circuit still trips/recovers.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.backend import BackendResponse
from roadstead.config import ProxyConfig
from roadstead.service import ProxyService


class _FakeRequest:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


async def _none(*a, **k):
    return None


def _stub_probes(svc: ProxyService) -> None:
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none


def _ok_backend(svc: ProxyService) -> None:
    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=3, output_tokens=1,
            finish_reason="stop",
        )
    svc._backend.call = fake_call


def _body(timeout_s=5.0):
    return {
        "agent_id": "a", "endpoint": "tier3", "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}]},
        "timeout_s": timeout_s,
    }


# --- scheduler-loop guard ----------------------------------------------------

@pytest.mark.asyncio
async def test_poisoned_tick_does_not_kill_dispatch():
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    _ok_backend(svc)

    real_tick = svc._scheduler.tick
    state = {"raised": False}

    def poisoned_tick(now):
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("poisoned tick")
        return real_tick(now)

    svc._scheduler.tick = poisoned_tick
    await svc.startup()
    try:
        # First iteration raises (guard logs CRITICAL + 1s backoff), but the
        # loop must survive and the NEXT submit must dispatch + complete.
        svc._dispatch_event.set()
        await asyncio.sleep(0.05)
        assert state["raised"]
        assert svc._scheduler_loop_alive()

        resp = await asyncio.wait_for(
            svc.handle_submit(_body(), _FakeRequest()), timeout=10.0)
        assert resp.status_code == 200
        assert json.loads(resp.body)["status"] == "ok"
    finally:
        await svc.shutdown()


# --- poller guard -------------------------------------------------------------

@pytest.mark.asyncio
async def test_poisoned_poller_chore_does_not_kill_poller():
    cfg = ProxyConfig(poller_interval_s=0.01)
    svc = ProxyService(cfg)
    _stub_probes(svc)

    calls = {"n": 0}
    real_prune = svc._timeout_model.prune

    def poisoned_prune(now):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("poisoned chore")
        return real_prune(now)

    svc._timeout_model.prune = poisoned_prune
    await svc.startup()
    try:
        # Wait until the poller has run the poisoned iteration AND a later one.
        for _ in range(200):
            if calls["n"] >= 3:
                break
            await asyncio.sleep(0.01)
        assert calls["n"] >= 3, "poller did not keep cycling after a poisoned iteration"
        assert svc._poller_alive()
    finally:
        await svc.shutdown()


# --- liveness surfacing -------------------------------------------------------

@pytest.mark.asyncio
async def test_dead_poller_degrades_health_and_alerts():
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    await svc.startup()
    try:
        svc._poller_task.cancel()
        await asyncio.sleep(0.02)

        health = json.loads((await svc.handle_health(_FakeRequest())).body)
        assert health["poller_alive"] is False
        assert health["scheduler_alive"] is True
        assert health["status"] == "degraded"

        status = json.loads((await svc.handle_status(_FakeRequest())).body)
        names = {a["name"] for a in status["alerts"]}
        assert "poller_dead" in names
        assert status["reliability"]["poller_alive"] is False
        assert status["reliability"]["scheduler_alive"] is True
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_dead_scheduler_is_down_503():
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    await svc.startup()
    try:
        svc._scheduler_task.cancel()
        await asyncio.sleep(0.02)

        resp = await svc.handle_health(_FakeRequest())
        health = json.loads(resp.body)
        assert resp.status_code == 503
        assert health["status"] == "down"

        status = json.loads((await svc.handle_status(_FakeRequest())).body)
        assert "scheduler_loop_dead" in {a["name"] for a in status["alerts"]}
    finally:
        await svc.shutdown()


# --- skip_discovery (embed/rerank FastAPI shims) -------------------------------

@pytest.mark.asyncio
async def test_skip_discovery_probes_health_only():
    svc = ProxyService(ProxyConfig())
    probes = {"health": [], "props": 0, "models": 0, "vllm": 0}

    async def fake_health(ep_cfg):
        probes["health"].append(ep_cfg.endpoint_class)
        return True

    async def fake_props(ep_cfg):
        probes["props"] += 1
        return None

    async def fake_models(ep_cfg):
        probes["models"] += 1
        return None

    async def fake_vllm(ep_cfg):
        probes["vllm"] += 1
        return None

    svc._backend.probe_health = fake_health
    svc._backend.probe_props = fake_props
    svc._backend.probe_models = fake_models
    svc._backend.probe_vllm_capacity = fake_vllm

    ep = svc._config.endpoints["embed"]
    assert ep.skip_discovery  # config invariant: the shims skip discovery
    assert svc._config.endpoints["rerank"].skip_discovery

    await svc._poll_endpoint_once("embed", ep)
    assert probes["health"] == ["embed"]
    assert probes["props"] == 0 and probes["models"] == 0 and probes["vllm"] == 0
    assert svc._endpoint_health["embed"]["healthy"] is True


@pytest.mark.asyncio
async def test_skip_discovery_circuit_still_trips_and_recovers():
    svc = ProxyService(ProxyConfig())
    health_state = {"up": False}

    async def fake_health(ep_cfg):
        return health_state["up"]

    svc._backend.probe_health = fake_health
    ep = svc._config.endpoints["embed"]

    # Health down: consecutive failures climb to the threshold, the confirming
    # /health probe also fails → circuit trips.
    for _ in range(svc._health_fail_threshold):
        await svc._poll_endpoint_once("embed", ep)
    assert svc._endpoint_health["embed"]["healthy"] is False

    # Health back: the unhealthy endpoint recovers on liveness alone.
    health_state["up"] = True
    await svc._poll_endpoint_once("embed", ep)
    assert svc._endpoint_health["embed"]["healthy"] is True
