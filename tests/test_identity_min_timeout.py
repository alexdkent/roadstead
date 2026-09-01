"""Per-identity MINIMUM deadline floor (2026-08-03).

A registered caller that supplies NO deadline of its own is entirely at the
mercy of the adaptive timeout model, and the model's ``size_stretch`` clamps at
3.0 — floor(tier3 180s) x surge(1.0) x 3.0 = a flat 540s wall for any tier3
prompt above ~82K tokens. Live evidence: three consecutive kills at
elapsed_s=539.999 against applied_timeout_s=540.0 on a 123,466-token prompt
(agent_id=lan-generic, layer=stream). ``timeout_ceiling_s`` cannot fix that —
the stretch clamp binds long before any ceiling — so the lever is a FLOOR.

The contract pinned here:
  * a registered identity WITH a floor and no caller timeout -> raised to floor
  * the same identity WITH an explicit caller timeout        -> caller wins
  * an identity with NO floor                                -> unchanged
  * both ``smart_default_timeout`` flag states                -> floor applies
  * the floor is EXTEND-ONLY — it never shortens a longer default
  * an API key carries its own floor, exactly as an address does

🚨 Addresses here are RFC 5737 documentation space, configured through
``ROADSTEAD_ACL``. The ACL shipped a private fleet's registrations until
2026-09-01 (scrub item S1) and now ships none, so the floor is an operator
decision in both of its forms — which is what these drive.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from roadstead import scheduler as _sched
from roadstead.acl import IPIdentityMap
from roadstead.backend import BackendResponse
from roadstead.config import LLMPriority, ProxyConfig
from roadstead.constants import _DEFAULT_TIMEOUT_S, _SMART_DEFAULT_CAP_S
from roadstead.service import ProxyService

# The smart default for "chat" (-> the tier2 endpoint class) on a cold model.
# Both it and the flat default are far below the 1800s floor, so the floor is
# visibly the binding constraint in BOTH flag states.
_CREATIVE_FLOOR_S = 120.0

#: A batch harness with the background floor, a floor-less exact registration
#: inside the same subnet, and the subnet catch-all carrying the floor too.
_ACL = (
    f"192.0.2.25=batch-harness:P3_INGESTION:{int(_SMART_DEFAULT_CAP_S)},"
    "192.0.2.9=no-floor:P3_INGESTION,"
    f"192.0.2.0/24=lan-generic:P3_INGESTION:{int(_SMART_DEFAULT_CAP_S)}"
)

#: The same floor, granted to a KEY instead of an address. ``…:1800`` is the
#: shared identity-spec grammar, so an operator writes the floor the same way in
#: both registries.
_KEYS = f"floored-key=batch-harness:P3_INGESTION:{int(_SMART_DEFAULT_CAP_S)}"


@pytest.fixture(autouse=True)
def _registrations(monkeypatch):
    """Both registries, for every test in this module. The ACL and the key
    registry are read at ``ProxyService`` construction, so this must run first."""
    monkeypatch.setenv("ROADSTEAD_ACL", _ACL)
    monkeypatch.setenv("ROADSTEAD_API_KEYS", _KEYS)
    monkeypatch.delenv("LLM_PROXY_ACL", raising=False)
    monkeypatch.delenv("ROADSTEAD_REQUIRE_API_KEY", raising=False)


class _Req:
    """Fake Starlette request with a settable source IP."""

    def __init__(self, host: str = "127.0.0.1", headers: dict | None = None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.headers = headers or {}
        self.method = "POST"
        self.query_params: dict = {}


def _submit_body(endpoint: str = "chat", *, timeout_s=None):
    body = {
        "agent_id": "a", "endpoint": endpoint, "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100},
    }
    if timeout_s is not None:
        body["timeout_s"] = timeout_s
    return body


def _ok_backend(svc):
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call


@contextlib.contextmanager
def _capture_applied_timeout():
    """Spy the applied deadline via ``QueuedRequest.create`` — the single point
    that stamps ``timeout_s`` onto the request."""
    created: list = []
    orig = _sched.QueuedRequest.create.__func__

    def spy(cls, **kw):
        req = orig(cls, **kw)
        created.append(req)
        return req

    _sched.QueuedRequest.create = classmethod(spy)
    try:
        yield created
    finally:
        _sched.QueuedRequest.create = classmethod(orig)


# --------------------------------------------------------------------------
# ACL — the registration surface.
# --------------------------------------------------------------------------

def test_batch_harnesses_and_lan_generic_carry_the_floor():
    acl = IPIdentityMap.from_env()
    # An agentic harness and the subnet catch-all both get the 1800s floor. The
    # FLOOR does not belong to any particular caller — it belongs to any client
    # that supplies no deadline of its own, which is why the catch-all carries
    # it as well as the named row.
    assert acl.min_timeout_s("192.0.2.25") == _SMART_DEFAULT_CAP_S
    assert acl.min_timeout_s("192.0.2.50") == _SMART_DEFAULT_CAP_S
    # Kept consistent with the cap the smart default may already reach.
    assert _SMART_DEFAULT_CAP_S == 1800.0


def test_identities_without_a_floor_report_none():
    acl = IPIdentityMap.from_env()
    # 192.0.2.9 is an EXACT registration inside 192.0.2.0/24 — exact beats
    # subnet for the floor exactly as it does for identity, so it does NOT
    # inherit lan-generic's floor.
    assert acl.min_timeout_s("192.0.2.9") is None
    assert acl.min_timeout_s("127.0.0.1") is None       # internal
    assert acl.min_timeout_s("172.17.0.5") is None      # docker bridge
    assert acl.min_timeout_s("8.8.8.8") is None         # unregistered
    assert acl.min_timeout_s("not-an-ip") is None       # malformed, no crash


def test_identify_tuple_shape_unchanged():
    # identify() deliberately still returns (agent_id, priority): the floor is
    # read through its own lookup so no existing call site had to widen.
    acl = IPIdentityMap.from_env()
    assert acl.identify("192.0.2.25") == ("batch-harness", LLMPriority.P3_INGESTION)
    assert acl.identify("192.0.2.9") == ("no-floor", LLMPriority.P3_INGESTION)
    assert acl.identify("127.0.0.1") == ("internal", LLMPriority.P1_TURN_SUPPORT)
    assert acl.identify("8.8.8.8") is None


def test_register_defaults_to_no_floor():
    acl = IPIdentityMap()
    acl.register("192.0.2.99", "plain", LLMPriority.P3_INGESTION)
    assert acl.min_timeout_s("192.0.2.99") is None


# --------------------------------------------------------------------------
# Wire-through — /v1/submit.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("smart", [False, True])
async def test_floor_raises_default_deadline_in_both_flag_states(smart):
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": smart})
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            r = await asyncio.wait_for(
                svc.handle_submit(_submit_body(), _Req("192.0.2.25")), timeout=10.0)
            assert r.status_code == 200
            # Without the floor this would be 180.0 (flag off) / 120.0 (flag on).
            assert created[-1].timeout_s == _SMART_DEFAULT_CAP_S
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("smart", [False, True])
async def test_explicit_caller_timeout_wins_over_the_floor(smart):
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": smart})
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            r = await asyncio.wait_for(
                svc.handle_submit(_submit_body(timeout_s=45.0), _Req("192.0.2.25")),
                timeout=10.0)
            assert r.status_code == 200
            # A caller that states its own deadline stays authoritative — even
            # far BELOW the floor. The floor is for deadlines the proxy chose.
            assert created[-1].timeout_s == 45.0
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("smart,expected", [(False, _DEFAULT_TIMEOUT_S),
                                            (True, _CREATIVE_FLOOR_S)])
async def test_identity_without_a_floor_is_byte_identical(smart, expected):
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": smart})
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            # 192.0.2.9 has no floor -> the pre-existing default.
            r = await asyncio.wait_for(
                svc.handle_submit(_submit_body(), _Req("192.0.2.9")), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == expected
            # Same for a container-internal caller (the local-first default).
            r = await asyncio.wait_for(
                svc.handle_submit(_submit_body(), _Req("127.0.0.1")), timeout=10.0)
            assert created[-1].timeout_s == expected
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_floor_is_extend_only_never_shortens():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": False})
    # A floor BELOW the resolved default must leave the default alone.
    svc._acl.register("198.51.100.7", "tiny-floor", LLMPriority.P3_INGESTION,
                      min_timeout_s=10.0)
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            r = await asyncio.wait_for(
                svc.handle_submit(_submit_body(), _Req("198.51.100.7")), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == _DEFAULT_TIMEOUT_S
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_malformed_caller_timeout_falls_to_default_and_gets_the_floor():
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            # A value we could not use is not a caller deadline — the request
            # is on the proxy's default, so the floor applies to it.
            r = await asyncio.wait_for(
                svc.handle_submit(_submit_body(timeout_s="garbage"), _Req("192.0.2.25")),
                timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == _SMART_DEFAULT_CAP_S
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------
# Wire-through — the OpenAI front door (how an agentic CLI actually arrives).
# --------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("smart", [False, True])
async def test_openai_door_batch_caller_gets_the_floor(smart):
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": smart})
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            body = {"model": "tier3",
                    "messages": [{"role": "user", "content": "hi"}]}
            r = await asyncio.wait_for(
                svc.handle_openai_chat(dict(body), _Req("192.0.2.25")), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == _SMART_DEFAULT_CAP_S
            assert created[-1].agent_id == "batch-harness"

            # An X-Timeout-S header is an explicit caller deadline -> it wins.
            r = await asyncio.wait_for(
                svc.handle_openai_chat(
                    dict(body), _Req("192.0.2.25", {"X-Timeout-S": "30"})),
                timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == 30.0
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------
# The same floor, carried by a KEY (Workstream B).
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_key_carries_the_floor_from_an_unenrolled_address():
    """The point of moving identity onto keys: the concession follows the
    CALLER, not the machine it happens to be running on.

    The source address here is not enrolled at all — under the address-only
    scheme this request is a 403, and under it the floor could never travel with
    a caller that moved host.
    """
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"smart_default_timeout": False})
    _ok_backend(svc)
    await svc.startup()
    try:
        with _capture_applied_timeout() as created:
            req = _Req("203.0.113.77", {"Authorization": "Bearer floored-key"})
            r = await asyncio.wait_for(
                svc.handle_submit(_submit_body(), req), timeout=10.0)
            assert r.status_code == 200
            assert created[-1].timeout_s == _SMART_DEFAULT_CAP_S
            # …and the key, not the body's self-declared "a", is the DRR key.
            assert created[-1].agent_id == "batch-harness"
    finally:
        await svc.shutdown()
