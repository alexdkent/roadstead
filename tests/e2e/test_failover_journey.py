"""JOURNEY — § 9.9 of `docs/anvil2_tier3_deepseek_v4_flash_plan_2026-08.md`
(plan item D2, the tier3 → tier2 LLM-proxy failover).

    tier3 down -> served by tier2 -> response labelled and
    resp.model correct -> health-verifier chip raised -> an agent WITHOUT degrade_ok
    gets a clean 503 + Retry-After in the same window -> tier3 back ->
    drained -> next request on tier3, chip cleared.

This is deliberately NOT `test_failover.py` (19 unit tests already cover the
state machine, the two admission gates and the config plumbing in isolation).
Its job is the thing units cannot prove: that the capability is REACHED
through the real HTTP front door, against the real ``ProxyService``, real
scheduler and real ``Failover``/``Health`` wiring, with a REAL (if fake-
backed) round trip on the wire — and that what a caller actually receives on
that wire is correct, not merely what the code intended to send.

Real seams driven, nothing else faked:
  * tier3 is made sick the way an operator actually does it — POST
    ``/v1/admin/endpoints/tier3/pause`` (the same drain switch used for a
    live vLLM restart), not by touching ``Failover``/``ProxyState`` directly.
  * requests are submitted through the real ``/rs/v1/chat`` front door, which
    runs the full ``Lifecycle.handle_submit`` — circuit breaker, failover gates,
    scheduler, dispatch — exactly as production does.
  * the two callers are told apart by API KEY, not by a body field. Since
    Workstream B the fair-share key comes from the credential, and since C the
    body cannot claim one at all — so the opted-in and not-opted-in identities
    this journey turns on are established the way a deployment establishes
    them, which is the point of an end-to-end test.
  * two independent fake backends stand in for tier3 (``tier3``) and
    tier2 (``tier2``) so a captured ``resp.model`` can only have
    come from whichever one actually served — the identical-content "echo:"
    shape a single shared fake would produce can't prove that; a distinct
    ``served_model_id`` per backend can. (``fake_backend.py`` had to learn to
    stamp ``served_model_id`` into the chat-completion body's ``model`` field
    for this — it was previously hardcoded to the literal string
    "fake-model", the one piece of real backend behaviour this journey needs
    that the shared fixture didn't yet model. Default value unchanged, so
    every other e2e test's assertions are untouched — see
    ``test_e2e_happy.py``/``test_behavior_preservation.py``, still green.)

DWELL HANDLING: the plan sets `policy.failover_dwell_s: 120` in
models.yaml. This journey does NOT sleep 120s and does NOT monkeypatch
`Failover` — it builds its OWN `ProxyConfig` (same pattern as
`conftest.py::_repointed_config`) with the `tier3` stanza's
`failover_dwell_s` overridden to 0.3s via `dataclasses.replace`, i.e. driving
the config the way an operator would tune the dwell, not the clock. The
recovery LEAVE transition is then driven by the real poller loop (interval
0.05s, same trick `conftest.py` already uses) rather than a manual
`Failover.refresh()` call — nobody in this file calls into `failover.py`
directly.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import tempfile
from typing import AsyncIterator, Dict

import httpx
import pytest
import pytest_asyncio

from roadstead.config import (
    AgentQuotaConfig,
    EndpointConfig,
    ProxyConfig,
    load_agent_configs,
    normalize_endpoint,
)
from roadstead.__main__ import build_app

from roadstead.testing import FAULT_CAPACITY_DESYNC, FakeBackend, FakeBackendServer

from tests.admin_key import ADMIN_HEADERS, enrol_admin

SRC = normalize_endpoint("tier3")     # tier3
TGT = normalize_endpoint("tier2")    # tier2 (the boxa)

_INTERNAL_CLIENT = ("127.0.0.1", 41998)  # loopback -> ACL "internal" + admin

FAST_DWELL_S = 0.3  # real dwell is 120s (models.yaml); see module docstring


async def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.03):
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()


@pytest_asyncio.fixture
async def journey(caplog) -> AsyncIterator[dict]:
    """A real ProxyService with tier3 (`tier3`) and tier2
    (`tier2`) backed by TWO INDEPENDENT fake backends, real agents.yaml
    (`chat-assistant.degrade_ok=True`, `extractor.degrade_ok=False`), and the `tier3`
    dwell shrunk to FAST_DWELL_S. Every other endpoint is left pointed at the
    tier2 fake — unused by this journey, but must resolve to
    something so config construction doesn't 404 on itself.
    """
    caplog.set_level(logging.WARNING, logger="roadstead.failover")
    fake_thinker = FakeBackendServer(FakeBackend(served_model_id="llama-thinker-live")).start()
    fake_creative = FakeBackendServer(FakeBackend(served_model_id="qwen3-creative-live")).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = ProxyConfig(
                queue_db_path=f"{tmp}/queue.db",
                agents=load_agent_configs(),  # real agents.yaml — chat-assistant opted in, extractor not
            )
            endpoints: Dict[str, EndpointConfig] = {}
            for cls, ep in base.endpoints.items():
                host, port = (
                    (fake_thinker.host, fake_thinker.port) if cls == SRC
                    else (fake_creative.host, fake_creative.port)
                )
                endpoints[cls] = dataclasses.replace(
                    ep, host=host, port=port,
                    max_slots=ep.max_slots or 4,
                    context_per_slot=ep.context_per_slot or 8192,
                    failover_dwell_s=(FAST_DWELL_S if cls == SRC else ep.failover_dwell_s),
                )
            base.endpoints = endpoints
            base.poller_interval_s = 0.05  # real poller drives the LEAVE transition
            # Identity comes from the credential now, so the journey has to
            # supply one. Set on os.environ (not monkeypatch) because this is a
            # module-scoped async fixture, which pytest's function-scoped
            # monkeypatch cannot reach.
            os.environ["ROADSTEAD_API_KEYS"] = ",".join(
                f"{key}={agent}:{KEY_PRIORITY}" for agent, key in KEYS.items())
            app = build_app(base)
            svc = app.state.proxy_service

            async def _healthy(ep_cfg):
                return True
            svc._backend.probe_health = _healthy

            # This journey configures real caller keys, so the registry IS in
            # play and an unknown credential is a 401. The admin pause below
            # needs one the registry knows.
            enrol_admin(svc)

            await svc.startup()
            transport = httpx.ASGITransport(
                app=app, raise_app_exceptions=False, client=_INTERNAL_CLIENT)
            client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                                       timeout=30.0)
            try:
                yield {
                    "svc": svc, "client": client,
                    "fake_thinker": fake_thinker, "fake_creative": fake_creative,
                }
            finally:
                os.environ.pop("ROADSTEAD_API_KEYS", None)
                await client.aclose()
                await svc.shutdown()
    finally:
        fake_thinker.stop()
        fake_creative.stop()


#: One API key per caller, so identity arrives the way it does in production.
#: `chat-assistant` carries `degrade_ok: true` in the real agents.yaml and
#: `extractor` deliberately does not — that contrast is what this file tests.
#:
#: 🚨 Both keys declare P1. The band is load-bearing here and not incidental:
#: the failover gates run only for INTERACTIVE and FOREGROUND work, because a
#: BACKGROUND request falls through and queues to defer until the backend
#: recovers (``lifecycle.handle_submit``) — correct behaviour, and it would make
#: this journey hang rather than fail. A key with no declared priority defaults
#: to P3_INGESTION, which is exactly that case.
KEYS = {"chat-assistant": "key-chat", "extractor": "key-extract"}
KEY_PRIORITY = "P1_TURN_SUPPORT"


def _auth(agent: str) -> dict:
    return {"Authorization": f"Bearer {KEYS[agent]}"}


def _submit_body(agent: str, *, content: str = "hi", max_tokens: int = 16) -> dict:
    return {
        "model": SRC,
        "call_site": f"{agent}.journey",
        "payload_type": "chat_completion",
        "payload": {
            "model": SRC,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        },
    }


async def _status(client: httpx.AsyncClient) -> dict:
    resp = await client.get("/v1/status")
    assert resp.status_code == 200
    return resp.json()


async def test_tier3_failover_full_journey(journey, caplog):
    """The whole D2 arc, in order, over the real seams."""
    caplog.set_level(logging.WARNING, logger="roadstead.failover")
    client = journey["client"]
    svc = journey["svc"]
    fake_thinker = journey["fake_thinker"]
    fake_creative = journey["fake_creative"]

    # -- 0. baseline: tier3 healthy, served directly, no degraded marker ----
    resp = await client.post("/rs/v1/chat", json=_submit_body("chat-assistant"),
                             headers=_auth("chat-assistant"))
    assert resp.status_code == 200, resp.text
    env = resp.json()
    assert env["status"] == "ok"
    assert env["response"]["model"] == "llama-thinker-live"
    # Attribution says so explicitly rather than by the ABSENCE of a marker:
    # the enriched envelope always discloses, so "not substituted" is a value
    # a caller can read rather than a key it has to notice is missing.
    assert env["attribution"]["substituted"] is False
    assert env["attribution"]["substitution"] is None
    assert fake_thinker.controller.requests, "baseline call never reached tier3's fake"
    assert not fake_creative.controller.requests, "baseline call leaked onto tier2"

    # -- 1. tier3 down -- the way the system does it: an operator drain -----
    pause = await client.post(f"/v1/admin/endpoints/{SRC}/pause",
                              headers=dict(ADMIN_HEADERS),
                              json={"reason": "journey: simulate tier3 outage"})
    assert pause.status_code == 200, pause.text
    assert pause.json()["paused"] is True
    assert svc._health.endpoint_healthy(SRC) is False

    # -- 2. served by tier2; response labelled; resp.model correct --
    fake_creative.controller.reset()
    resp = await client.post("/rs/v1/chat", json=_submit_body("chat-assistant", content="degraded turn"),
                             headers=_auth("chat-assistant"))
    assert resp.status_code == 200, resp.text
    env = resp.json()
    assert env["status"] == "ok"
    assert env["attribution"]["substituted"] is True
    assert env["attribution"]["substitution"] == "failover"
    assert env["attribution"]["resolved"] == SRC
    assert env["attribution"]["endpoint"] == TGT
    # 🚨 the wire fact, not the intent: it must have come back FROM the
    # tier2 fake specifically, naming ITS served model.
    assert env["response"]["model"] == "qwen3-creative-live"
    assert env["response"]["choices"][0]["message"]["content"] == "echo: degraded turn"
    assert any(r.path == "/v1/chat/completions" for r in fake_creative.controller.requests), (
        "the degraded request never reached tier2's fake")

    # -- 3. health-verifier chip raised -------------------------------------------
    # health-verifier's llmproxy verifier chips off exactly two observable signals: the
    # CRITICAL ROADSTEAD_FAILOVER_ENTER log line (its alert path is filtered to
    # CRITICAL/ERROR — a WARNING here would be recorded and never chipped) and
    # the /v1/status `degraded_endpoints` SET (the same surface `paused_endpoints`
    # already uses). Both are asserted directly since health-verifier itself is a
    # separate agent/process, out of scope for an llmproxy journey.
    enter_lines = [r for r in caplog.records if "ROADSTEAD_FAILOVER_ENTER" in r.getMessage()]
    assert enter_lines, "no ROADSTEAD_FAILOVER_ENTER — the chip has nothing to fire on"
    assert enter_lines[0].levelno == logging.CRITICAL, (
        "ENTER logged below CRITICAL — health-verifier's alert filter would swallow it silently")
    status = await _status(client)
    assert SRC in status["degraded_endpoints"]
    assert status["failover_pairs"].get(SRC) == TGT
    assert status["degraded_rerouted"].get(SRC, 0) >= 1

    # -- 4. an agent WITHOUT degrade_ok gets a clean 503 + Retry-After, ------
    #       in the SAME window (tier3 still down, still degraded) -----------
    assert svc._state.config.agent_config("extractor").degrade_ok is False
    resp = await client.post("/rs/v1/chat", json=_submit_body("extractor", content="not opted in"),
                             headers=_auth("extractor"))
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["code"] == "draining"  # this outage is an operator drain
    assert body["degraded_refusal"] == "degraded_not_opted_in"
    assert "Retry-After" in resp.headers
    assert int(resp.headers["Retry-After"]) > 0
    # refused, never silently degraded — the request must not have reached
    # EITHER backend.
    assert not any("not opted in" in str(r.body) for r in fake_thinker.controller.requests)
    assert not any("not opted in" in str(r.body) for r in fake_creative.controller.requests)

    # -- 5. tier3 back -- drained -- next request on tier3, chip cleared ----
    # Hold ONE degraded request in flight on tier2 (CAPACITY_DESYNC's
    # sleep-then-serve happy path) so the LEAVE transition has a real,
    # observable cohort to drain rather than an instantaneous empty one.
    fake_creative.controller.set_fault(FAULT_CAPACITY_DESYNC, 0.5)
    inflight_task = asyncio.create_task(
        client.post("/rs/v1/chat", json=_submit_body("chat-assistant", content="in-flight during recovery"),
                             headers=_auth("chat-assistant")))
    assert await _wait_until(
        lambda: svc._scheduler.degraded_inflight(SRC) >= 1, timeout_s=2.0
    ), "the held request never registered as degraded in-flight"

    # 🚨 `json={}` rather than a bare POST: identity.py's CSRF gate requires
    # `Content-Type: application/json` on every mutating admin request, which
    # httpx only sets when a `json=` body is given.
    resume = await client.post(f"/v1/admin/endpoints/{SRC}/resume",
                               headers=dict(ADMIN_HEADERS), json={})
    assert resume.status_code == 200, resume.text
    assert resume.json()["paused"] is False
    assert svc._health.endpoint_healthy(SRC) is True  # health is instant; drain+dwell gate LEAVE

    # Recovered + past dwell, but the cohort hasn't drained yet -> must STAY.
    await asyncio.sleep(FAST_DWELL_S + 0.1)
    status = await _status(client)
    assert SRC in status["degraded_endpoints"], (
        "left degraded mode before the in-flight cohort drained")

    inflight_resp = await inflight_task
    assert inflight_resp.status_code == 200
    assert inflight_resp.json()["attribution"]["substituted"] is True

    # Drained AND past dwell -> the poller LEAVEs on its own cadence, no
    # request required to observe it (§ 9.7 — "must not require traffic").
    left = await _wait_until(lambda: SRC not in svc._state.degraded_endpoints, timeout_s=3.0)
    assert left, "never left degraded mode after drain + dwell"
    leave_lines = [r for r in caplog.records if "ROADSTEAD_FAILOVER_LEAVE" in r.getMessage()]
    assert leave_lines, "no ROADSTEAD_FAILOVER_LEAVE — the chip has nothing to clear on"

    status = await _status(client)
    assert SRC not in status["degraded_endpoints"], "chip surface not cleared"
    assert SRC not in status["degraded_for_s"]

    # Next request goes back to tier3, unlabelled.
    fake_thinker.controller.reset()
    fake_creative.controller.reset()
    resp = await client.post("/rs/v1/chat", json=_submit_body("chat-assistant", content="recovered turn"),
                             headers=_auth("chat-assistant"))
    assert resp.status_code == 200, resp.text
    env = resp.json()
    # Attribution says so explicitly rather than by the ABSENCE of a marker:
    # the enriched envelope always discloses, so "not substituted" is a value
    # a caller can read rather than a key it has to notice is missing.
    assert env["attribution"]["substituted"] is False
    assert env["attribution"]["substitution"] is None
    assert env["response"]["model"] == "llama-thinker-live"
    assert any(r.path == "/v1/chat/completions" for r in fake_thinker.controller.requests)
    assert not fake_creative.controller.requests, (
        "post-recovery request still landed on tier2")
