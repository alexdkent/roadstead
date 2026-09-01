"""The address ACL — the SECOND identity factor, behind API keys.

It assigns the priority band that drives DRR, so it was worth testing even when
it was the only identity mechanism there was (Phase 4.3 coverage).

🚨 Every address here is RFC 5737 documentation space (``192.0.2.0/24``,
``198.51.100.0/24``). As of 2026-09-01 the ACL ships NO registrations — a
private fleet's LAN was compiled into it until then (scrub item S1) — so these
drive the ``ROADSTEAD_ACL`` env form, which is now the only way to enrol a host.
That is not merely a scrub artefact: it is the only form an operator has, so
testing it is testing the real path rather than a default nobody can reach.
"""

from __future__ import annotations

import pytest

from roadstead.acl import IPIdentityMap
from roadstead.config import LLMPriority

#: A stand-in for the enrolled fleet: one exact host inside a subnet that is
#: itself enrolled, so exact-beats-subnet is observable.
_ACL = (
    "192.0.2.9=tideway:P3_INGESTION,"
    "192.0.2.23=chat-brain:P1_TURN_SUPPORT,"
    "192.0.2.0/24=lan-generic:P3_INGESTION:1800"
)


@pytest.fixture
def acl(monkeypatch) -> IPIdentityMap:
    monkeypatch.setenv("ROADSTEAD_ACL", _ACL)
    monkeypatch.delenv("LLM_PROXY_ACL", raising=False)
    monkeypatch.delenv("ROADSTEAD_ADMIN_NETS", raising=False)
    return IPIdentityMap.from_env()


# --- what ships ---------------------------------------------------------------

def test_ships_no_registrations(monkeypatch):
    """🚨 The property scrub item S1 exists for, asserted rather than assumed.

    A default that happens to match somebody's LAN is worse than no default: it
    hands an identity — and with it a DRR share and a deadline floor — to
    whatever answers at an address we guessed.
    """
    monkeypatch.delenv("ROADSTEAD_ACL", raising=False)
    monkeypatch.delenv("LLM_PROXY_ACL", raising=False)
    bare = IPIdentityMap.from_env()
    for ip in ("192.0.2.9", "198.51.100.1", "10.0.0.2", "10.0.0.1", "8.8.8.8"):
        assert bare.identify(ip) is None, f"{ip} resolves out of the box"
    # …but the local-first case still needs no configuration whatsoever.
    assert bare.identify("127.0.0.1") == ("internal", LLMPriority.P1_TURN_SUPPORT)


def test_internal_nets_are_internal(acl):
    for ip in ("127.0.0.1", "172.16.0.5", "172.31.255.1"):
        assert acl.identify(ip) == ("internal", LLMPriority.P1_TURN_SUPPORT), ip


def test_subnet_registration_catches_the_rest_of_the_lan(acl):
    assert acl.identify("192.0.2.50") == ("lan-generic", LLMPriority.P3_INGESTION)


def test_exact_registration_beats_subnet(acl):
    # 192.0.2.9 is registered exact (tideway) AND falls in 192.0.2.0/24
    # (lan-generic); the exact match must win, for identity and for the floor.
    assert acl.identify("192.0.2.9") == ("tideway", LLMPriority.P3_INGESTION)
    assert acl.min_timeout_s("192.0.2.9") is None
    assert acl.min_timeout_s("192.0.2.50") == 1800.0


def test_an_exact_registration_is_not_a_catch_all(acl):
    """The .12/.18/.19/.86 mis-claim class, generically.

    An address registration is only ever about the address written down. This
    is the assertion that a registration cannot quietly widen — which is how a
    stale row survives a host being destroyed and mis-attributes whatever takes
    the address next.
    """
    assert acl.identify("198.51.100.9") is None


def test_unknown_ip_denied(acl):
    assert acl.identify("8.8.8.8") is None
    assert acl.is_allowed("8.8.8.8") is False


def test_malformed_ip_does_not_crash(acl):
    assert acl.identify("not-an-ip") is None


# --- the env grammar ----------------------------------------------------------

def test_env_parse_priority_and_default(monkeypatch):
    monkeypatch.setenv(
        "ROADSTEAD_ACL", "198.51.100.0/24=batch:P4_HYGIENE,203.0.113.3=special")
    acl = IPIdentityMap.from_env()
    assert acl.identify("198.51.100.5") == ("batch", LLMPriority.P4_HYGIENE)
    # no priority suffix → default P3
    assert acl.identify("203.0.113.3") == ("special", LLMPriority.P3_INGESTION)


def test_env_segments_are_order_independent(monkeypatch):
    """The spec grammar identifies a segment by SHAPE, not position, so an
    operator cannot get the order wrong — there is no order to get wrong."""
    monkeypatch.setenv(
        "ROADSTEAD_ACL",
        "192.0.2.1=a:P1_TURN_SUPPORT:600,192.0.2.2=b:600:P1_TURN_SUPPORT")
    acl = IPIdentityMap.from_env()
    assert acl.identify("192.0.2.1")[1] is LLMPriority.P1_TURN_SUPPORT
    assert acl.identify("192.0.2.2")[1] is LLMPriority.P1_TURN_SUPPORT
    assert acl.min_timeout_s("192.0.2.1") == acl.min_timeout_s("192.0.2.2") == 600.0


def test_malformed_env_entry_skipped(monkeypatch):
    monkeypatch.setenv("ROADSTEAD_ACL", "garbage-no-equals,198.51.100.0/24=ok")
    acl = IPIdentityMap.from_env()           # must not raise
    assert acl.identify("198.51.100.1") == ("ok", LLMPriority.P3_INGESTION)


def test_the_pre_rename_spelling_is_still_honoured(monkeypatch):
    """``LLM_PROXY_ACL`` is the ONE legacy env name kept, and deliberately.

    Every other ``COLLECTIVE_*``/``LLM_PROXY_*`` rename dropped the old spelling
    outright, because two names for one switch is how they come to disagree.
    This one configures ACCESS: a proxy that silently stops recognising its
    callers on upgrade fails closed in the most confusing way available.
    """
    monkeypatch.setenv("LLM_PROXY_ACL", "192.0.2.7=legacy:P3_INGESTION")
    monkeypatch.delenv("ROADSTEAD_ACL", raising=False)
    acl = IPIdentityMap.from_env()
    assert acl.identify("192.0.2.7") == ("legacy", LLMPriority.P3_INGESTION)


def test_the_new_spelling_wins_a_collision(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_ACL", "192.0.2.7=old:P4_HYGIENE")
    monkeypatch.setenv("ROADSTEAD_ACL", "192.0.2.7=new:P1_TURN_SUPPORT")
    acl = IPIdentityMap.from_env()
    assert acl.identify("192.0.2.7") == ("new", LLMPriority.P1_TURN_SUPPORT)


# --- admin surfaces are narrower than inference -------------------------------

def test_is_admin_is_loopback_and_docker_by_default(acl):
    assert acl.is_admin("127.0.0.1")
    assert acl.is_admin("172.17.0.5")        # docker bridge
    # An enrolled LAN device passes identify() and is NOT admin. Enrolling a
    # subnet for inference says nothing about who may pause a backend fleet-wide.
    assert acl.identify("192.0.2.42") is not None
    assert not acl.is_admin("192.0.2.42")
    assert not acl.is_admin("8.8.8.8")
    assert not acl.is_admin("not-an-ip")


def test_admin_nets_grant_admin_without_touching_priority(monkeypatch):
    """The grant must not promote a host's inference band as a side effect.

    ``_admin_nets`` is a separate list from ``_internal_nets`` for exactly this
    reason: ``identify()`` matches the internal nets BEFORE subnet entries, so
    granting admin through that list would silently re-band the host's traffic
    to the internal P1 default.
    """
    monkeypatch.setenv("ROADSTEAD_ACL", "198.51.100.0/24=ops-lan:P4_HYGIENE")
    monkeypatch.setenv("ROADSTEAD_ADMIN_NETS", "198.51.100.7/32")
    acl = IPIdentityMap.from_env()
    assert acl.is_admin("198.51.100.7")
    assert acl.identify("198.51.100.7") == ("ops-lan", LLMPriority.P4_HYGIENE)
    assert not acl.is_admin("198.51.100.8")


def test_admin_segment_on_an_acl_entry_grants_both(monkeypatch):
    """``=ops:admin`` is the one place the two decisions are made together,
    because an operator writing it plainly means both."""
    monkeypatch.setenv("ROADSTEAD_ACL", "192.0.2.5=ops:P3_INGESTION:admin")
    acl = IPIdentityMap.from_env()
    assert acl.is_admin("192.0.2.5")
    assert acl.identify("192.0.2.5") == ("ops", LLMPriority.P3_INGESTION)


# --- Phase 8 hardening: the admin routes, end to end --------------------------

import pytest as _pytest

from roadstead.config import ProxyConfig as _PCfg
from roadstead.service import ProxyService as _PSvc


class _IpReq:
    def __init__(self, host, method="GET", body=None, headers=None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.method = method
        self.headers: dict = headers or {}
        self.query_params: dict = {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


@_pytest.mark.asyncio
async def test_the_admin_plane_needs_BOTH_a_reachable_address_and_a_credential(monkeypatch):
    """🚨 The 2026-09-01 change, as an assertion. This test used to be called
    `test_admin_routes_deny_lan_allow_loopback` and it asserted that loopback
    with NO credential could pause a backend, set a flag and ingest a call log.
    That was true, and it was the bug: the audit trail recorded such changes as
    `key_id: null, source: "ip"` — the system knew nobody had authenticated.

    An address is a GATE now and never a grant. Both halves must pass, and the
    two refusals are different things:

    * off-net       — 403, and no credential answers it.
    * on-net, no credential — 403 from the scope check, answerable by presenting
      one. Through the UI door this same denial becomes a 401 + challenge, which
      is what makes a browser show a password box.
    """
    monkeypatch.setenv("ROADSTEAD_ACL", _ACL)
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "sk-ops=ops:admin")
    svc = _PSvc(_PCfg())
    lan, loop = "192.0.2.42", "127.0.0.1"
    key = {"X-API-Key": "sk-ops"}

    # Off-net, with a VALID admin key: still refused. The gate is the network.
    resp = await svc.handle_admin_endpoint_pause(
        "tier1", _IpReq(lan, "POST", headers=key), pause=True)
    assert resp.status_code == 403

    # On-net, no credential: refused. This is the line that used to be 200.
    resp = await svc.handle_admin_endpoint_pause("tier1", _IpReq(loop, "POST"), pause=True)
    assert resp.status_code == 403

    # On-net, with the credential: allowed.
    resp = await svc.handle_admin_endpoint_pause(
        "tier1", _IpReq(loop, "POST", headers=key), pause=True)
    assert resp.status_code == 200
    await svc.handle_admin_endpoint_pause(
        "tier1", _IpReq(loop, "POST", headers=key), pause=False)

    # Every other admin surface inherits it — the gate is one method, not a
    # per-route decision, so this is a check that nothing bypasses it.
    assert (await svc.handle_admin_flags(_IpReq(loop, "GET"))).status_code == 403
    assert (await svc.handle_admin_flags(_IpReq(loop, "GET", headers=key))).status_code == 200
    assert (await svc.handle_maintenance_list(_IpReq(loop, "GET"))).status_code == 403
    assert (await svc.handle_maintenance_list(_IpReq(loop, "GET", headers=key))).status_code == 200
    body = {"endpoint": "whisper-1", "kind": "audio", "duration_s": 1.0}
    assert (await svc.handle_calls_log(_IpReq(loop, "POST", body))).status_code == 403
    assert (await svc.handle_calls_log(_IpReq(loop, "POST", body, headers=key))).status_code == 200


@_pytest.mark.asyncio
async def test_inference_routes_still_open_to_an_enrolled_lan(monkeypatch):
    monkeypatch.setenv("ROADSTEAD_ACL", _ACL)
    svc = _PSvc(_PCfg())

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        from roadstead.backend import BackendResponse
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
            {"model": "tier3", "max_tokens": 8,
             "messages": [{"role": "user", "content": "hi"}]},
            _IpReq("192.0.2.42", "POST")), timeout=10.0)
        assert resp.status_code == 200  # LAN inference unaffected by the tightening
    finally:
        await svc.shutdown()
