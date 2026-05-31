"""Phase 5B — proxy reliability hardening.

  5B.1 streaming slot-leak: when the SSE consumer disconnects, the producer
       dispatch task is cancelled so it can't wedge on a full stream_q.put and
       hold the scheduler slot forever (the live companion-jam bug, 2026-05-31).
  5B.2 drain: a straggler that exceeds the drain deadline is cancelled AND its
       CancelledError handler frees the slot (records the completion).
  5B.4 budget restore: a configured agent's cap/weight win over the persisted
       row on restart.
"""

from __future__ import annotations

import asyncio

import pytest

from originfleet.llmproxy import service as service_mod
from originfleet.llmproxy.backend import BackendStreamEvent
from originfleet.llmproxy.config import AgentQuotaConfig, ProxyConfig
from originfleet.llmproxy.observability import (
    MetricsSample, RollingMetrics, check_alerts,
)
from originfleet.llmproxy.queue import PersistentQueue
from originfleet.llmproxy.service import ProxyService


class _FakeRequest:
    class _Client:
        host = "172.16.0.5"

    client = _Client()
    headers: dict = {}


async def _none(*a, **k):
    return None


def _stub_probes(svc: ProxyService) -> None:
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none


def _body(stream=False, timeout_s=10.0):
    return {
        "agent_id": "a", "endpoint": "llama-thinker", "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}], "stream": stream},
        "timeout_s": timeout_s,
    }


# --- 5B.1 streaming slot-leak ----------------------------------------------

@pytest.mark.asyncio
async def test_consumer_disconnect_cancels_producer_and_frees_slot():
    svc = ProxyService(ProxyConfig())

    async def slow_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        # Produces indefinitely so the producer is still streaming when the
        # consumer disconnects (and would wedge on a full stream_q without the fix).
        for i in range(100000):
            await asyncio.sleep(0.005)
            yield BackendStreamEvent(
                "chunk", '{"choices":[{"delta":{"content":"x"}}]}',
                {"choices": [{"delta": {"content": "x"}}]})

    svc._backend.stream = slow_stream
    _stub_probes(svc)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(stream=True), _FakeRequest())
        it = resp.body_iterator

        # Read frames until a chunk arrives → the producer is dispatched, live,
        # and holding a scheduler slot.
        async def read_until_chunk():
            async for ch in it:
                t = ch.decode() if isinstance(ch, (bytes, bytearray)) else ch
                if '"chunk"' in t or '"delta"' in t:
                    return
        await asyncio.wait_for(read_until_chunk(), timeout=5.0)
        assert svc._scheduler.endpoint_snapshot("thinker")["in_flight"] >= 1

        # Consumer disconnects: close the SSE generator → finally cancels the
        # producer.
        await it.aclose()

        # Let the cancellation + CancelledError handler propagate.
        async def wait_freed():
            while svc._scheduler.endpoint_snapshot("thinker")["in_flight"] > 0:
                await asyncio.sleep(0.02)
        await asyncio.wait_for(wait_freed(), timeout=5.0)

        assert svc._slot_leak_reclaimed >= 1, "producer cancel must reclaim the slot"
        assert svc._scheduler.endpoint_snapshot("thinker")["in_flight"] == 0
    finally:
        await svc.shutdown()


# --- 5B.2 drain straggler cancel frees the slot -----------------------------

@pytest.mark.asyncio
async def test_drain_cancels_straggler_and_counts(monkeypatch):
    monkeypatch.setattr(service_mod, "_DRAIN_DEADLINE_S", 0.1)
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    await svc.startup()

    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(30)  # far exceeds the 0.1s drain deadline

    task = asyncio.create_task(slow())
    svc._inflight_tasks["straggler"] = task
    task.add_done_callback(lambda t: svc._inflight_tasks.pop("straggler", None))
    await asyncio.wait_for(started.wait(), timeout=2.0)

    await svc.shutdown()  # drain deadline fires → cancel + awaited gather
    assert svc._drain_straggler_cancelled == 1
    assert task.cancelled() or task.done()


# --- 5B.4 budget restore honors configured cap/weight -----------------------

@pytest.mark.asyncio
async def test_budget_restore_uses_configured_cap(tmp_path):
    db = str(tmp_path / "q.db")
    # Persist a balance ABOVE the default 60.0 cap with a stale weight.
    pq = PersistentQueue(db)
    pq.save_budgets([{
        "agent_id": "agentX", "weight": 9.0,
        "balance_ss": 500.0, "total_consumed_ss": 12.0,
    }])
    pq.close()

    cfg = ProxyConfig(queue_db_path=db)
    cfg.agents["agentX"] = AgentQuotaConfig(
        agent_id="agentX", weight=2.0, max_balance_ss=300.0)
    svc = ProxyService(cfg)
    _stub_probes(svc)
    await svc.startup()
    try:
        b = svc._budget_mgr.get_or_create("agentX")
        assert b.max_balance == 300.0, "configured cap must win over default 60"
        assert b.weight == 2.0, "configured weight must win over persisted 9.0"
        assert b.balance == 300.0, "balance clamped to the configured cap (was 500)"
    finally:
        await svc.shutdown()


# --- 5C.1 circuit recovers on /health even when discovery stays flaky -------

@pytest.mark.asyncio
async def test_circuit_recovers_on_health_even_if_discovery_fails():
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)

    async def _health_up(*a, **k):
        return True
    svc._backend.probe_health = _health_up
    await svc.startup()
    try:
        # Tripped (e.g. discovery had been failing); /health is back but
        # capacity-discovery still fails. Old code latched open forever.
        svc._endpoint_health["thinker"] = {
            "healthy": False, "consecutive_failures": 9,
            "unhealthy_since": 1.0}
        ep_cfg = svc._config.endpoints["thinker"]
        await svc._update_endpoint_health("thinker", ep_cfg, probe_ok=False)
        assert svc._endpoint_health["thinker"]["healthy"] is True, (
            "must recover on /health alone, not latch open on discovery failure")
    finally:
        await svc.shutdown()


# --- 5C TTFT fast-fail: a 0-token hang aborts fast + frees the slot ---------

@pytest.mark.asyncio
async def test_ttft_fastfail_aborts_zero_token_hang(monkeypatch):
    monkeypatch.setattr(service_mod, "_STREAM_TTFT_DEADLINE_S", 0.3)
    svc = ProxyService(ProxyConfig())

    async def hang_stream(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        await asyncio.sleep(30)   # never produces a first token
        yield BackendStreamEvent("done", "[DONE]")  # unreachable

    svc._backend.stream = hang_stream
    _stub_probes(svc)
    await svc.startup()
    try:
        resp = await svc.handle_submit(_body(stream=True), _FakeRequest())
        frames = []

        async def drain():
            async for ch in resp.body_iterator:
                frames.append(ch.decode() if isinstance(ch, (bytes, bytearray)) else ch)
        await asyncio.wait_for(drain(), timeout=5.0)
        joined = "".join(frames)
        assert "ttft" in joined or "backpressure" in joined, joined

        # Slot freed promptly (NOT held for the full deadline).
        async def wait_freed():
            while svc._scheduler.endpoint_snapshot("thinker")["in_flight"] > 0:
                await asyncio.sleep(0.02)
        await asyncio.wait_for(wait_freed(), timeout=5.0)
        assert svc._scheduler.endpoint_snapshot("thinker")["in_flight"] == 0
    finally:
        await svc.shutdown()


# --- 5C drr_imbalance gated on real queued work -----------------------------

def _imbalanced_metrics(now):
    m = RollingMetrics()
    m.record(MetricsSample(now, "companion", "heavy", "P3_INGESTION",
                           0.1, 1.0, "ok", 100.0))
    m.record(MetricsSample(now, "companion", "light", "P3_INGESTION",
                           0.1, 1.0, "ok", 1.0))
    return m


def test_drr_imbalance_silent_without_queue():
    now = 1000.0
    alerts = check_alerts(
        endpoint_snapshots={"companion": {"queued": 0}}, agent_budgets=[],
        metrics=_imbalanced_metrics(now), cost_model_samples={},
        queue_wal_size=0, now=now)
    assert not any(a.name == "drr_imbalance" for a in alerts), \
        "imbalance with no queued work is harmless — must not page"


def test_drr_imbalance_fires_with_queue():
    now = 1000.0
    alerts = check_alerts(
        endpoint_snapshots={"companion": {"queued": 3}}, agent_budgets=[],
        metrics=_imbalanced_metrics(now), cost_model_samples={},
        queue_wal_size=0, now=now)
    assert any(a.name == "drr_imbalance" for a in alerts), \
        "imbalance WITH queued work is actionable — must page"


# --- 5C drain-503 carries a deferrable marker -------------------------------

@pytest.mark.asyncio
async def test_drain_503_body_is_deferrable():
    from originfleet.framework.nexus_errors import is_deferrable_llm_error
    import json as _json
    svc = ProxyService(ProxyConfig())
    _stub_probes(svc)
    await svc.startup()
    try:
        svc._draining.set()
        resp = await svc.handle_submit(_body(), _FakeRequest())
        assert resp.status_code == 503
        err = _json.loads(resp.body.decode())["error"]
        # A streaming/sync caller wraps this body; it MUST classify deferrable so
        # a turn caught mid-SIGTERM defers instead of surfacing a hard error.
        assert is_deferrable_llm_error(ConnectionError(err)), err
    finally:
        svc._draining.clear()
        await svc.shutdown()
