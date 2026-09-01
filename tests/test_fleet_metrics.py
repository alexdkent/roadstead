"""Phase 1 — proxy as the fleet call-metrics authority.

Covers the non-LLM ingest writer, the usage/activity/savings rollups, the
cloud-rate table, the SSE hub, and the ingest + stream HTTP handlers.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from roadstead.config import ProxyConfig
from roadstead.queue import PersistentQueue
from roadstead.service import ProxyService
from roadstead.sse_hub import DROP_SENTINEL, SSEHub
from roadstead.usage_rates import cloud_cost_usd, cloud_rate

from tests.admin_key import ADMIN_HEADERS, enrol_admin


# ----- queue: non-LLM ingest + rollups -----

def _ext(pq, rid, endpoint, kind="audio", in_tok=0, out_tok=0, dur=1.0,
         status="ok", agent="tideway"):
    pq.persist_external_call(
        request_id=rid, agent_id=agent, endpoint=endpoint, call_site=kind,
        kind=kind, input_tokens=in_tok, output_tokens=out_tok,
        duration_s=dur, status=status)


def _llm(pq, rid, endpoint="tier3", in_tok=100, out_tok=50, dur=1.0,
         status="ok", agent="knowledge"):
    pq.persist_complete(rid, agent, endpoint, "site", 3, in_tok, out_tok,
                        dur, 2.0, status, kind="chat")


def test_persist_external_call_and_kind(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _ext(pq, "e1", "whisper-1", kind="audio", in_tok=0, out_tok=0)
    _llm(pq, "l1", endpoint="tier3")
    rows = pq.recent_requests(10)
    assert {r["request_id"] for r in rows} == {"e1", "l1"}
    # kind column persisted distinctly
    kinds = dict(pq._reader().execute(
        "SELECT request_id, kind FROM proxy_completions").fetchall())
    assert kinds["e1"] == "audio" and kinds["l1"] == "chat"
    pq.close()


def test_fleet_activity_aggregates(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _llm(pq, "l1", endpoint="tier3", status="ok")
    _llm(pq, "l2", endpoint="tier3", status="error")
    _ext(pq, "e1", "whisper-1", kind="audio")
    act = pq.fleet_activity(window_s=3600, bin_s=60)
    total_n = sum(b["n"] for b in act["calls"])
    total_fails = sum(b["fails"] for b in act["calls"])
    assert total_n == 3 and total_fails == 1
    eps = {r["endpoint"] for r in act["by_endpoint_1h"]}
    assert "tier3" in eps and "whisper-1" in eps
    pq.close()


def test_usage_rollup_by_agent_with_cost(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    # 2026-08-22: tier3 cut over from Laguna S 2.1 INT4 to DeepSeek-V4-Flash-0731
    # (0.14/0.28 per 1M in/out, its own published rate) after composer/tier3 were
    # repointed onto it.
    _llm(pq, "l1", endpoint="tier3", in_tok=1_000_000, out_tok=1_000_000, agent="knowledge")
    _ext(pq, "e1", "whisper-1", kind="audio", in_tok=1_000_000, agent="tideway")
    rows = pq.usage_rollup("agent", hours=1)
    by_agent = {r["key"]: r for r in rows}
    # tier3: 0.14/M in + 0.28/M out = 0.42
    assert by_agent["knowledge"]["cost_usd"] == pytest.approx(0.42, abs=0.01)
    # whisper (stt): input_tokens = audio_seconds*100 → 1M = 10_000s = 2.778 hr
    #                × $0.111/hr = $0.308
    assert by_agent["tideway"]["cost_usd"] == pytest.approx(0.308, abs=0.01)
    assert by_agent["knowledge"]["requests"] == 1
    pq.close()


def test_savings_summary(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _llm(pq, "l1", endpoint="tier1", in_tok=1_000_000, out_tok=0)  # 0.04/M in = $0.04
    s = pq.savings_summary(today_start=0.0)  # everything counts as "today"
    assert s["total_usd"] == pytest.approx(0.04, abs=0.001)
    assert s["today_usd"] == pytest.approx(0.04, abs=0.001)
    assert s["total_tokens_in"] == 1_000_000
    pq.close()


def test_top_callers_groups_by_endpoint(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _llm(pq, "l1", endpoint="tier3", agent="knowledge")
    _llm(pq, "l2", endpoint="tier3", agent="knowledge")
    _llm(pq, "l3", endpoint="tier3", agent="forum-agent")
    tc = pq.top_callers(window_s=3600, per_endpoint=5)
    callers = {c["agent"]: c["n"] for c in tc["providers"]["tier3"]}
    assert callers["knowledge"] == 2 and callers["forum-agent"] == 1
    pq.close()


def test_endpoint_series(tmp_path):
    pq = PersistentQueue(str(tmp_path / "q.db"))
    _llm(pq, "l1", endpoint="tier2", dur=1.0, status="ok")
    _llm(pq, "l2", endpoint="tier2", dur=2.0, status="error")
    # classify/analyst RE-HOMED onto the boxa `tier2` endpoint 2026-07-11 (three-role
    # consolidation) — tier2 is now a legacy alias resolving to `tier2`.
    out = pq.endpoint_series("tier2", window_s=3600, bin_s=60)  # role → class
    assert out["endpoint"] == "tier2"
    total = sum(b["n"] for b in out["calls_series"])
    assert total == 2
    pq.close()


# ----- usage rate table -----

def test_cloud_rate_resolves_class_unit_and_role():
    """A class, an alias of a class, a per-unit capability, and a name nobody
    knows — the four cases the resolver has to keep apart. A class that resolves
    to nothing bills $0, which reads as "this lane is free" rather than "nobody
    priced it"."""
    assert cloud_rate("tier3") == (0.14, 0.28)
    assert cloud_rate("reasoner") == (0.14, 0.28)   # alias -> tier3
    assert cloud_rate("composer") == (0.14, 0.28)   # alias -> tier3
    assert cloud_rate("tier2") == (0.15, 0.55)
    assert cloud_rate("vision") == (0.14, 0.28)     # alias -> tier3
    assert cloud_rate("tier1") == (0.04, 0.08)
    assert cloud_rate("classify") == (0.04, 0.08)   # alias -> tier1
    # Per-unit / no-analog endpoints carry NO token rate (cost via cloud_cost_usd).
    assert cloud_rate("tts") == (0.0, 0.0)          # per-char, not per-token
    assert cloud_rate("whisper-1") == (0.0, 0.0)    # per-audio-hour
    assert cloud_rate("ocr") == (0.0, 0.0)          # no analog
    assert cloud_rate("nonsense") == (0.0, 0.0)


def test_a_remote_class_is_unmetered_not_free():
    """🚨 The avoided-cost table only makes sense for capacity you OWN. A remote
    provider's endpoint costs actual money, and pricing it from this table would
    credit the fleet with saving money it is in fact spending. Explicitly None
    until Workstream D meters it from what the provider publishes."""
    from roadstead.usage_rates import _ENDPOINT_CLASS
    assert _ENDPOINT_CLASS["spill-chat"] is None
    assert cloud_rate("spill-chat") == (0.0, 0.0)


def test_cloud_cost_usd_token_and_per_unit():
    # 1. Token-native: rate x tokens.
    assert cloud_cost_usd("tier3", 1_000_000, 1_000_000) == pytest.approx(0.42)
    assert cloud_cost_usd("classify", 1_000_000, 0) == pytest.approx(0.04)  # alias -> tier1
    assert cloud_cost_usd("tier2", 1_000_000, 0) == pytest.approx(0.15)
    # 2. Per-audio-hour: input_tokens = audio_seconds * 100, so 360_000 = 1 hour.
    assert cloud_cost_usd("whisper-1", 360_000, 0) == pytest.approx(0.111)     # stt
    assert cloud_cost_usd("diarize", 360_000, 0) == pytest.approx(0.12)      # diarize
    # 3. Per-character TTS: input_tokens = characters.
    assert cloud_cost_usd("tts", 1_000_000, 5_000_000) == pytest.approx(15.0)  # 1M chars; out ignored
    # 4. No-analog / not-yet-metered -> $0 (stream avoids double-counting stt).
    assert cloud_cost_usd("stream", 10_000_000, 0) == 0.0
    assert cloud_cost_usd("imagegen", 999, 999) == 0.0
    assert cloud_cost_usd("nonsense", 1_000_000, 1_000_000) == 0.0


# ----- SSE hub -----

@pytest.mark.asyncio
async def test_sse_hub_publish_and_drop():
    hub = SSEHub(queue_maxsize=2)
    q = hub.subscribe()
    hub.publish("call.completed", {"a": 1})
    ev, data = await q.get()
    assert ev == "call.completed" and '"a":1' in data
    # overflow → slow client gets dropped with the sentinel
    hub.publish("x", {"n": 1})
    hub.publish("x", {"n": 2})
    hub.publish("x", {"n": 3})   # overflow triggers drop sentinel
    seen_sentinel = False
    for _ in range(4):
        if q.empty():
            break
        item = await q.get()
        if item == DROP_SENTINEL:
            seen_sentinel = True
    assert seen_sentinel
    hub.unsubscribe(q)
    assert hub.client_count == 0


# ----- service: ingest + stream handlers -----

class _FakeReq:
    class _Client:
        host = "172.16.0.5"   # internal → ACL-allowed
    client = _Client()
    headers: dict = ADMIN_HEADERS

    def __init__(self, body=None):
        self._body = body or {}

    async def json(self):
        return self._body


async def _none(*a, **k):
    return None


def _svc_request_disconnected_false():
    pass


@pytest.mark.asyncio
async def test_calls_log_ingest_records_and_emits(tmp_path):
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    enrol_admin(svc)
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    try:
        sub = svc._sse.subscribe()
        req = _FakeReq({
            "unit": "whisper-1", "kind": "audio", "agent": "tideway",
            "latency_ms": 1200, "input_tokens": 0, "output_tokens": 0,
            "success": True,
        })
        resp = await svc.handle_calls_log(req)
        assert resp.status_code == 200
        # recorded
        svc._queue_db.flush(2.0)
        rows = svc._queue_db.recent_requests(10)
        assert any(r["endpoint"] == "whisper-1" for r in rows)
        # fanned out
        ev, _data = await asyncio.wait_for(sub.get(), timeout=2.0)
        assert ev == "call.completed"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_calls_log_rejects_llm_kind():
    svc = ProxyService(ProxyConfig())
    enrol_admin(svc)
    svc._backend.probe_vllm_capacity = _none
    svc._backend.probe_props = _none
    svc._backend.probe_models = _none
    await svc.startup()
    try:
        req = _FakeReq({"unit": "tier3", "kind": "chat", "agent": "x"})
        resp = await svc.handle_calls_log(req)
        assert resp.status_code == 409   # proxy-native; refuse double-count
    finally:
        await svc.shutdown()


# --- 2026-06-11: calls/log refuses pushes for proxy-native LLM endpoints -----

import pytest as _pt


class _LoopReq:
    def __init__(self, body):
        class _C:
            host = "127.0.0.1"

        self.client = _C()
        self.headers: dict = dict(ADMIN_HEADERS)
        self._body = body

    async def json(self):
        return self._body


@_pt.mark.asyncio
async def test_metrics_renders_truncation_gauge():
    """L-2 (audit 2026-07-12): the per-caller truncation tally must surface on
    /metrics so a TSDB rule can alert on structured truncations by caller."""
    svc = ProxyService(ProxyConfig())
    enrol_admin(svc)
    svc._state.truncation_by_model_caller["tier2|kv4_grader"] = {
        "count": 3, "structured": 2, "freetext": 1}
    resp = await svc.handle_prometheus_metrics(_LoopReq({}))
    text = resp.body.decode()
    assert "llmproxy_truncations_total" in text
    assert ('llmproxy_truncations_total{endpoint="tier2",caller="kv4_grader",'
            'structured="true"} 2') in text
    assert ('llmproxy_truncations_total{endpoint="tier2",caller="kv4_grader",'
            'structured="false"} 1') in text


@_pt.mark.asyncio
async def test_metrics_renders_empty_completion_gauge():
    """C-3 (audit 2026-07-12): the per-endpoint empty-completion counter must
    surface on /metrics so the post-boxa reliability signature is trendable."""
    svc = ProxyService(ProxyConfig())
    enrol_admin(svc)
    svc._state.empty_completion_by_endpoint["tier2"] = 5
    resp = await svc.handle_prometheus_metrics(_LoopReq({}))
    text = resp.body.decode()
    assert 'llmproxy_empty_completion_total{endpoint="tier2"} 5' in text


@_pt.mark.asyncio
async def test_calls_log_409_for_llm_class_endpoints():
    """A pushed call whose endpoint normalizes to a proxy-native LLM class
    duplicates a natively-recorded row (the 2026-06-11 rerank double-count
    arrived as kind='external' + endpoint='rerank') — refuse it regardless of
    the kind label. Genuine non-LLM units still ingest."""
    from roadstead.config import ProxyConfig
    from roadstead.service import ProxyService

    svc = ProxyService(ProxyConfig())

    enrol_admin(svc)
    for ep in ("rerank", "rerank", "rerank", "embed",
               "embed", "tier3", "chat"):
        resp = await svc.handle_calls_log(_LoopReq(
            {"endpoint": ep, "kind": "external", "duration_s": 0.1}))
        assert resp.status_code == 409, ep
    for ep in ("whisper-1", "tts", "diarize", "stream",
               "stt", "ocr"):
        resp = await svc.handle_calls_log(_LoopReq(
            {"endpoint": ep, "kind": "audio", "duration_s": 0.1}))
        assert resp.status_code == 200, ep
