"""ACL identity map — was untested, and it assigns the priority band that drives
DRR (Phase 4.3 coverage)."""

from __future__ import annotations

import pytest

from originfleet.llmproxy.acl import IPIdentityMap
from originfleet.llmproxy.config import LLMPriority


def test_internal_nets_are_internal():
    acl = IPIdentityMap.from_env()
    for ip in ("127.0.0.1", "172.16.0.5", "172.31.255.1"):
        assert acl.identify(ip) == ("internal", LLMPriority.P1_TURN_SUPPORT), ip


def test_lan_generic_subnet():
    acl = IPIdentityMap.from_env()
    r = acl.identify("10.0.0.50")
    assert r == ("lan-generic", LLMPriority.P3_INGESTION)


def test_anvil_not_admin_but_priority_unchanged():
    # 2026-07-09: anvil (10.0.0.3) was granted admin (via _admin_nets) for the
    # classify-evict pause/resume. 2026-07-11: REVOKED — classify re-homed off
    # anvil onto the boxa `creative` endpoint and the eviction coordinator is
    # inert, so anvil no longer needs proxy admin (back to loopback-only). Pin:
    #   (1) anvil is NO LONGER admin,
    #   (2) revoking admin did NOT change anvil's inference priority — identify()
    #       still resolves it via its own registration (anvil-local / P3), not the
    #       P1 internal path.
    acl = IPIdentityMap.from_env()
    assert acl.is_admin("10.0.0.3") is False
    assert acl.identify("10.0.0.3") != ("internal", LLMPriority.P1_TURN_SUPPORT)
    # A different LAN host is neither admin nor internal.
    assert acl.is_admin("10.0.0.50") is False
    assert acl.is_admin("127.0.0.1") is True


def test_exact_registration_beats_subnet():
    # 10.0.0.9 is registered exact (tideway) AND falls in 10.0.0.0/24 (lan-generic);
    # the exact match must win.
    acl = IPIdentityMap.from_env()
    assert acl.identify("10.0.0.9") == ("tideway", LLMPriority.P3_INGESTION)


def test_unknown_ip_denied():
    acl = IPIdentityMap.from_env()
    assert acl.identify("8.8.8.8") is None
    assert acl.is_allowed("8.8.8.8") is False


def test_malformed_ip_does_not_crash():
    acl = IPIdentityMap.from_env()
    assert acl.identify("not-an-ip") is None


def test_env_parse_priority_and_default(monkeypatch):
    monkeypatch.setenv(
        "LLM_PROXY_ACL", "10.2.0.0/16=batch:P4_HYGIENE,10.3.3.3=special")
    acl = IPIdentityMap.from_env()
    assert acl.identify("10.2.5.5") == ("batch", LLMPriority.P4_HYGIENE)
    # no priority suffix → default P3
    assert acl.identify("10.3.3.3") == ("special", LLMPriority.P3_INGESTION)


def test_malformed_env_entry_skipped(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_ACL", "garbage-no-equals,10.4.0.0/16=ok")
    acl = IPIdentityMap.from_env()           # must not raise
    assert acl.identify("10.4.1.1") == ("ok", LLMPriority.P3_INGESTION)


# --- Phase 8 hardening: admin surfaces are loopback/docker-only ---------------

import json as _json

import pytest as _pytest

from originfleet.llmproxy.config import ProxyConfig as _PCfg
from originfleet.llmproxy.service import ProxyService as _PSvc


class _IpReq:
    def __init__(self, host, method="GET", body=None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.method = method
        self.headers: dict = {}
        self.query_params: dict = {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def test_is_admin_loopback_and_docker_only():
    acl = IPIdentityMap.from_env()
    assert acl.is_admin("127.0.0.1")
    assert acl.is_admin("172.17.0.5")      # docker bridge
    assert not acl.is_admin("10.0.0.42")   # LAN device — identify() passes, admin doesn't
    assert acl.identify("10.0.0.42") is not None
    assert not acl.is_admin("8.8.8.8")
    assert not acl.is_admin("not-an-ip")


@_pytest.mark.asyncio
async def test_admin_routes_deny_lan_allow_loopback():
    svc = _PSvc(_PCfg())
    lan, loop = "10.0.0.42", "127.0.0.1"

    # Pause/resume.
    resp = await svc.handle_admin_endpoint_pause("gemma", _IpReq(lan, "POST"), pause=True)
    assert resp.status_code == 403
    resp = await svc.handle_admin_endpoint_pause("gemma", _IpReq(loop, "POST"), pause=True)
    assert resp.status_code == 200
    await svc.handle_admin_endpoint_pause("gemma", _IpReq(loop, "POST"), pause=False)

    # Flags.
    assert (await svc.handle_admin_flags(_IpReq(lan, "GET"))).status_code == 403
    assert (await svc.handle_admin_flags(_IpReq(loop, "GET"))).status_code == 200

    # Maintenance annotate + list.
    assert (await svc.handle_maintenance(_IpReq(lan, "POST", {"endpoint": "gemma"}))).status_code == 403
    assert (await svc.handle_maintenance_list(_IpReq(lan, "GET"))).status_code == 403
    assert (await svc.handle_maintenance_list(_IpReq(loop, "GET"))).status_code == 200

    # calls/log ingest.
    body = {"endpoint": "whisper-1", "kind": "audio", "duration_s": 1.0}
    assert (await svc.handle_calls_log(_IpReq(lan, "POST", body))).status_code == 403
    assert (await svc.handle_calls_log(_IpReq(loop, "POST", body))).status_code == 200


@_pytest.mark.asyncio
async def test_inference_routes_still_open_to_lan():
    svc = _PSvc(_PCfg())

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        from originfleet.llmproxy.backend import BackendResponse
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")

    svc._backend.call = ok_call
    await svc.startup()
    try:
        import asyncio as _aio
        resp = await _aio.wait_for(svc.handle_openai_chat(
            {"model": "llama-thinker", "max_tokens": 8,
             "messages": [{"role": "user", "content": "hi"}]},
            _IpReq("10.0.0.42", "POST")), timeout=10.0)
        assert resp.status_code == 200  # LAN inference unaffected by the tightening
    finally:
        await svc.shutdown()
