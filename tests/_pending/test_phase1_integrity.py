"""Phase 1 — integrity & correctness for the LLM proxy.

Covers:
  1.1 truncation: structured finish_reason=length fails loud (deferrable),
      free-form truncation is returned normally.
  1.2 circuit breaker: an unhealthy backend fast-fails interactive submits and
      defers background.
  1.3 transient retry/defer: unreachable/empty-completion retry within the
      deadline; 4xx don't; the proxy's error strings classify as deferrable.
  1.4 band inversion: a 1-slot endpoint no longer lets queued background block
      interactive.
  1.5 slot-leak: the backend call is bounded to the caller's remaining deadline.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from originfleet.framework.nexus_errors import is_deferrable_llm_error
from roadstead.backend import (
    BackendError,
    BackendResponse,
    BackendUnavailable,
)
from roadstead.config import (
    DEFAULT_ENDPOINTS,
    EndpointConfig,
    PriorityBand,
    ProxyConfig,
)
from roadstead.scheduler import (
    DispatchDecision,
    QueuedRequest,
    Scheduler,
)
from roadstead.service import ProxyService


_COMPLETION = {
    "id": "c", "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
}


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


# --- 1.3 deferral classifier ------------------------------------------------

def test_proxy_error_strings_are_deferrable():
    deferrable = [
        "LLM proxy error 502: upstream",
        "LLM proxy dispatch error: backend thinker returned empty completion",
        "LLM proxy unreachable: connection refused",
        "backend thinker truncated structured output (finish_reason=length)",
        "backend thinker unavailable (circuit open)",
    ]
    for msg in deferrable:
        assert is_deferrable_llm_error(ConnectionError(msg)) is True, msg
    # A genuine content/request error stays NON-deferrable.
    assert is_deferrable_llm_error(ValueError("invalid request: bad field")) is False


# --- 1.4 band inversion on a 1-slot endpoint --------------------------------

def test_interactive_not_blocked_by_background_on_one_slot_endpoint():
    ep = EndpointConfig(endpoint_class="rerank", role="r",
                        max_slots=1, background_floor_pct=0.0)
    assert ep.background_floor_slots == 1
    eq = SimpleNamespace(band_depth=lambda b: 1)  # background has queued work
    fake = SimpleNamespace(_active={})
    avail = Scheduler._available_for_band(
        fake, ep, PriorityBand.INTERACTIVE, 0, eq)
    assert avail == 1  # was 0 before the fix (max_slots - floor = 0 → blocked)


# --- direct _execute_sync harness -------------------------------------------

def _svc_with_call(call):
    svc = ProxyService(ProxyConfig())
    svc._backend.call = call
    return svc


def _mk_req(*, timeout_s=60.0, payload=None):
    return QueuedRequest.create(
        agent_id="a", endpoint="llama-thinker", priority="P3_INGESTION",
        call_site="t", payload_type="chat_completion",
        payload=payload if payload is not None else {"messages": [{"role": "user", "content": "x"}]},
        timeout_s=timeout_s,
    )


def _register_future(svc, req):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    svc._pending_futures[req.request_id] = fut
    return fut


# --- 1.5 deadline-bound backend call ----------------------------------------

@pytest.mark.asyncio
async def test_backend_call_bounded_to_remaining_deadline():
    captured = {}

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        captured["timeout_s"] = timeout_s
        return BackendResponse(200, _COMPLETION, 0.01, 5, 2, finish_reason="stop")

    svc = _svc_with_call(fake_call)
    req = _mk_req(timeout_s=100.0)
    req.timeout_deadline = time.monotonic() + 3.0  # only 3s of the 100s budget remains
    _register_future(svc, req)
    decision = DispatchDecision(request=req, queue_wait_ms=0.0, occupancy_at_dispatch=0)
    await svc._execute_sync(req, svc._config.endpoints["thinker"], decision)
    # bounded to the remaining deadline (~3s), NOT a fresh 100s
    assert 2.0 < captured["timeout_s"] <= 3.0, captured


# --- 1.3 transient retry / 4xx no-retry -------------------------------------

@pytest.mark.asyncio
async def test_transient_unavailable_retried_then_ok():
    calls = {"n": 0}

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BackendUnavailable("backend thinker unreachable")
        return BackendResponse(200, _COMPLETION, 0.01, 5, 2, finish_reason="stop")

    svc = _svc_with_call(fake_call)
    req = _mk_req(timeout_s=60.0)
    fut = _register_future(svc, req)
    decision = DispatchDecision(request=req, queue_wait_ms=0.0, occupancy_at_dispatch=0)
    await svc._execute_sync(req, svc._config.endpoints["thinker"], decision)
    assert calls["n"] == 2          # retried once
    assert fut.result()["status"] == "ok"


@pytest.mark.asyncio
async def test_empty_completion_retried():
    calls = {"n": 0}

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BackendError(502, "backend thinker returned empty completion (no content)")
        return BackendResponse(200, _COMPLETION, 0.01, 5, 2, finish_reason="stop")

    svc = _svc_with_call(fake_call)
    req = _mk_req(timeout_s=60.0)
    fut = _register_future(svc, req)
    decision = DispatchDecision(request=req, queue_wait_ms=0.0, occupancy_at_dispatch=0)
    await svc._execute_sync(req, svc._config.endpoints["thinker"], decision)
    assert calls["n"] == 2
    assert fut.result()["status"] == "ok"


@pytest.mark.asyncio
async def test_4xx_not_retried():
    calls = {"n": 0}

    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        calls["n"] += 1
        raise BackendError(400, "bad request: malformed field")

    svc = _svc_with_call(fake_call)
    req = _mk_req(timeout_s=60.0)
    fut = _register_future(svc, req)
    decision = DispatchDecision(request=req, queue_wait_ms=0.0, occupancy_at_dispatch=0)
    await svc._execute_sync(req, svc._config.endpoints["thinker"], decision)
    assert calls["n"] == 1          # deterministic 4xx → no retry
    assert fut.result()["status"] == "error"


# --- 1.1 truncation (direct _execute_sync) ----------------------------------

@pytest.mark.asyncio
async def test_structured_truncation_fails_loud():
    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            200, {"choices": [{"message": {"content": '{"a":'}, "finish_reason": "length"}]},
            0.01, 5, 16, finish_reason="length")

    svc = _svc_with_call(fake_call)
    # structured: response_format present
    req = _mk_req(payload={"messages": [{"role": "user", "content": "x"}],
                           "response_format": {"type": "json_object"}})
    fut = _register_future(svc, req)
    decision = DispatchDecision(request=req, queue_wait_ms=0.0, occupancy_at_dispatch=0)
    await svc._execute_sync(req, svc._config.endpoints["thinker"], decision)
    r = fut.result()
    assert r["status"] == "error"
    assert "truncated structured output" in r["error"]


@pytest.mark.asyncio
async def test_freeform_truncation_returned_ok():
    async def fake_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            200, {"choices": [{"message": {"content": "a long capped reply"},
                               "finish_reason": "length"}]},
            0.01, 5, 16, finish_reason="length")

    svc = _svc_with_call(fake_call)
    req = _mk_req(payload={"messages": [{"role": "user", "content": "x"}]})  # free-form
    fut = _register_future(svc, req)
    decision = DispatchDecision(request=req, queue_wait_ms=0.0, occupancy_at_dispatch=0)
    await svc._execute_sync(req, svc._config.endpoints["thinker"], decision)
    r = fut.result()
    assert r["status"] == "ok"      # free-form truncation is benign — return it
    assert r["response"]["choices"][0]["message"]["content"] == "a long capped reply"


# --- 1.2 circuit breaker (integration) --------------------------------------

async def _started_svc():
    svc = ProxyService(ProxyConfig())

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(200, _COMPLETION, 0.01, 5, 2, finish_reason="stop")

    async def _none(*a, **k):
        return None

    svc._backend.call = ok_call
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    return svc


@pytest.mark.asyncio
async def test_circuit_open_fast_fails_interactive():
    svc = await _started_svc()
    try:
        svc._endpoint_health["thinker"] = {
            "healthy": False, "consecutive_failures": 5, "unhealthy_since": time.monotonic()}
        body = {
            "agent_id": "a", "endpoint": "llama-thinker", "priority": "P0_REALTIME",
            "call_site": "t", "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "x"}]}, "timeout_s": 10.0,
        }
        resp = await svc.handle_submit(body, _FakeRequest())
        assert resp.status_code == 503
        env = json.loads(resp.body.decode())
        assert "circuit open" in env["error"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_circuit_open_defers_background_to_timeout():
    svc = await _started_svc()
    try:
        # Phase 5C: the circuit recovers on /health alone (decoupled from
        # capacity-discovery). To keep this endpoint genuinely DOWN — so the
        # poller can't recover it mid-test — make /health fail too.
        async def _health_down(*a, **k):
            return False
        svc._backend.probe_health = _health_down
        svc._endpoint_health["thinker"] = {
            "healthy": False, "consecutive_failures": 5, "unhealthy_since": time.monotonic()}
        body = {
            "agent_id": "a", "endpoint": "llama-thinker", "priority": "P3_INGESTION",
            "call_site": "t", "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "x"}]}, "timeout_s": 0.3,
        }
        resp = await svc.handle_submit(body, _FakeRequest())
        # background is NOT circuit-rejected (503): it queues + defers, and since
        # the scheduler skips the unhealthy endpoint it expires to a 504 timeout
        # (proving it was deferred, not dispatched — the ok backend would 200).
        assert resp.status_code == 504
    finally:
        await svc.shutdown()
