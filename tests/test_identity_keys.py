"""API keys as identity — Workstream B.

A caller's identity is the DRR fair-share key, the quota holder and the budget
holder. Until 2026-09-01 it came from a source address (which identifies a
*host*, survives no DHCP lease, and shipped with one fleet's LAN compiled in) or
from the request body on a ``/v1/submit`` door that had no access control at all
— so the fair-share key was, on that door, whatever the caller typed.

These pin the three rules in ``identity.py``'s docstring, each of which is a
decision that looks arbitrary until it bites:

1. a presented key that does not resolve is a **401**, never a fall-through to
   the address — a wrong credential must not silently become a weaker working
   one;
2. with **no keys configured** the registry is not in play, because every OpenAI
   client sends an ``Authorization`` header whether anybody meant it to or not;
3. a **key overrides** a body-declared identity; an **address only fills in** one
   the body omitted.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from roadstead.backend import BackendResponse
from roadstead.config import LLMPriority, ProxyConfig
from roadstead.identity import (
    IdentityResolver,
    KeyRegistry,
    Principal,
    presented_key,
)
from roadstead.acl import IPIdentityMap
from roadstead.service import ProxyService


class _Req:
    def __init__(self, host="203.0.113.10", headers=None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.headers = headers or {}
        self.method = "POST"
        self.query_params: dict = {}


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("ROADSTEAD_API_KEYS", "ROADSTEAD_API_KEYS_FILE", "ROADSTEAD_ACL",
                "LLM_PROXY_ACL", "ROADSTEAD_ADMIN_NETS", "ROADSTEAD_REQUIRE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# The registry.
# --------------------------------------------------------------------------

def test_a_key_resolves_to_its_principal():
    reg = KeyRegistry()
    reg.register(secret="s3cret", agent_id="chat-assistant",
                 priority=LLMPriority.P1_TURN_SUPPORT, min_timeout_s=600.0,
                 key_id="chat-assistant-prod")
    p = reg.resolve("s3cret")
    assert p == Principal(agent_id="chat-assistant", priority=LLMPriority.P1_TURN_SUPPORT,
                          min_timeout_s=600.0, admin=False, source="api_key",
                          key_id="chat-assistant-prod")
    assert p.authenticated is True
    assert reg.resolve("s3crets") is None
    assert reg.resolve("") is None


def test_a_precomputed_digest_is_equivalent_to_the_plaintext():
    """``key_sha256`` is the documented form precisely so a deployment never has
    to write the secret into a file that a git history will keep."""
    secret = "another-secret"
    digest = hashlib.sha256(secret.encode()).hexdigest()
    reg = KeyRegistry()
    reg.register(key_sha256=digest.upper(), agent_id="by-digest")
    assert reg.resolve(secret).agent_id == "by-digest"


def test_a_malformed_or_ambiguous_registration_is_refused_not_half_applied():
    reg = KeyRegistry()
    assert reg.register(agent_id="a") is None                       # neither form
    assert reg.register(secret="s", key_sha256="0" * 64, agent_id="a") is None
    assert reg.register(key_sha256="not-hex", agent_id="a") is None
    assert reg.register(secret="s", agent_id="") is None            # no identity
    assert len(reg) == 0


def test_one_secret_cannot_carry_two_identities():
    """Not a merge — a misconfiguration in which one of the two silently never
    applies, and the operator cannot tell which."""
    reg = KeyRegistry()
    assert reg.register(secret="dup", agent_id="first") is not None
    assert reg.register(secret="dup", agent_id="second") is None
    assert reg.resolve("dup").agent_id == "first"


def test_a_snapshot_never_carries_the_secret_or_its_digest():
    """🚨 A digest is not a secret cryptographically, but it IS a working
    credential for anyone who can compute one. An admin readout that prints it
    turns a read-only endpoint into a key store."""
    reg = KeyRegistry()
    reg.register(secret="top-secret", agent_id="a", key_id="a-key")
    blob = repr(reg.snapshot())
    assert "top-secret" not in blob
    assert hashlib.sha256(b"top-secret").hexdigest() not in blob
    # Equality, not a subset check: the point is that the field SET is pinned,
    # so a field added here is a deliberate decision about what an admin readout
    # publishes rather than something that arrived with a refactor. `source` and
    # `created_at` are Workstream E's — the management plane has to be able to
    # tell a runtime enrolment from a line in the environment.
    row = reg.snapshot()[0]
    created_at = row.pop("created_at")
    assert isinstance(created_at, float)
    assert row == {"key_id": "a-key", "agent_id": "a",
                   "priority": "P3_INGESTION", "min_timeout_s": None,
                   "admin": False, "admin_readonly": False, "may_write": False,
                   "expires_at": None, "expired": False, "bind": [],
                   "source": "file"}


def test_the_env_form_uses_the_shared_identity_grammar(monkeypatch):
    monkeypatch.setenv(
        "ROADSTEAD_API_KEYS",
        "k1=chat-assistant:P1_TURN_SUPPORT:600,k2=batch,k3=ops:admin")
    reg = KeyRegistry.from_env()
    assert reg.resolve("k1").priority is LLMPriority.P1_TURN_SUPPORT
    assert reg.resolve("k1").min_timeout_s == 600.0
    assert reg.resolve("k2") == Principal(
        agent_id="batch", priority=LLMPriority.P3_INGESTION, source="api_key",
        key_id=reg.resolve("k2").key_id)
    assert reg.resolve("k3").admin is True


def test_a_keys_file_reports_fields_it_cannot_use(tmp_path, monkeypatch, caplog):
    """🚨 The failure this repo has already had twice, in
    ``model_catalog._POLICY_PASSTHROUGH`` and ``load_agent_configs``: a knob the
    parser does not know is dropped in silence, and an unreachable knob looks
    exactly like a policy decision that the caller is not opted in."""
    f = tmp_path / "keys.yaml"
    f.write_text(
        "keys:\n"
        "  - id: batch-key\n"
        "    agent_id: batch\n"
        "    key: plain-secret\n"
        "    priority: P4_HYGIENE\n"
        "    weight: 4.0\n"          # not a keys-file field
    )
    monkeypatch.setenv("ROADSTEAD_API_KEYS_FILE", str(f))
    with caplog.at_level("WARNING"):
        reg = KeyRegistry.from_env()
    assert reg.resolve("plain-secret").priority is LLMPriority.P4_HYGIENE
    assert "weight" in caplog.text and "NO effect" in caplog.text


def test_every_field_of_a_principal_is_reachable_from_a_keys_file(tmp_path, monkeypatch):
    """The other half of the allowlist guard, and the half that actually bites.

    Reporting an unknown field catches a typo. It does NOT catch a knob added to
    ``Principal`` and never wired into the parser — that one is settable in code,
    unreachable by an operator, and looks exactly like a policy decision. So set
    every field at once and read them all back.
    """
    f = tmp_path / "keys.yaml"
    f.write_text(
        "keys:\n"
        "  - id: everything\n"
        "    agent_id: full-house\n"
        "    key: the-secret\n"
        "    priority: P0_REALTIME\n"
        "    min_timeout_s: 42.5\n"
        "    admin: true\n"
    )
    monkeypatch.setenv("ROADSTEAD_API_KEYS_FILE", str(f))
    assert KeyRegistry.from_env().resolve("the-secret") == Principal(
        agent_id="full-house", priority=LLMPriority.P0_REALTIME,
        min_timeout_s=42.5, admin=True, source="api_key", key_id="everything")


def test_a_missing_keys_file_is_loud_but_not_fatal(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("ROADSTEAD_API_KEYS_FILE", str(tmp_path / "nope.yaml"))
    with caplog.at_level("WARNING"):
        reg = KeyRegistry.from_env()
    assert not reg.configured
    assert "does not exist" in caplog.text


# --------------------------------------------------------------------------
# Rule 1 — a bad key is a 401 and never a demotion.
# --------------------------------------------------------------------------

def test_an_unknown_key_is_401_even_when_the_address_is_enrolled():
    """The doctrine test.

    The source address here resolves perfectly well. Falling back to it would
    mean a caller whose key was revoked keeps working as a *different, weaker*
    identity — indistinguishable from the credential being fine, which is the
    same shape as the ``finish_reason`` repair that became a silencer.
    """
    acl = IPIdentityMap()
    acl.register("203.0.113.10", "lan-generic", LLMPriority.P3_INGESTION)
    keys = KeyRegistry()
    keys.register(secret="good", agent_id="enrolled")
    res = IdentityResolver(acl, keys).resolve(_Req(headers=_bearer("wrong")))

    assert not res.ok
    assert res.denial.status == 401
    assert res.denial.code == "invalid_api_key"
    # And it SAYS so — an operator must not have to infer this from a graph.
    assert "not ignored in favour of the source address" in res.denial.message.lower()


def test_a_good_key_beats_an_unenrolled_address():
    """The other half: the credential is what is trusted, so a caller carrying
    one need not also be at an address somebody enrolled."""
    res = IdentityResolver(IPIdentityMap(),
                           _reg(secret="good", agent_id="mobile")).resolve(
        _Req(host="198.51.100.200", headers=_bearer("good")))
    assert res.ok and res.principal.agent_id == "mobile"


# --------------------------------------------------------------------------
# Rule 2 — with no keys configured, a presented key is invisible.
# --------------------------------------------------------------------------

def test_a_placeholder_key_is_ignored_when_no_keys_are_configured():
    """Every OpenAI SDK insists on an ``api_key``; users type ``EMPTY`` or
    ``sk-no-key-required``. Treating that as significant before an operator has
    configured any key would 401 the entire existing world on upgrade, over a
    credential nobody chose."""
    acl = IPIdentityMap()
    acl.register("203.0.113.10", "lan-generic", LLMPriority.P3_INGESTION)
    resolver = IdentityResolver(acl, KeyRegistry())
    for placeholder in ("EMPTY", "sk-no-key-required", "dummy"):
        res = resolver.resolve(_Req(headers=_bearer(placeholder)))
        assert res.ok, placeholder
        assert res.principal.agent_id == "lan-generic"
        assert res.principal.authenticated is False


def test_require_key_refuses_a_request_that_presents_none():
    resolver = IdentityResolver(IPIdentityMap(), _reg(secret="k", agent_id="a"),
                                require_key=True)
    res = resolver.resolve(_Req(host="127.0.0.1"))
    assert not res.ok and res.denial.status == 401
    assert resolver.resolve(_Req(headers=_bearer("k"))).ok


def test_require_key_without_any_key_configured_is_reported(caplog):
    """It cannot serve anybody, so it is certainly a misconfiguration and never
    a policy — say so at startup rather than 401-ing every caller in silence."""
    with caplog.at_level("ERROR"):
        IdentityResolver(IPIdentityMap(), KeyRegistry(), require_key=True)
    assert "NO keys are configured" in caplog.text


# --------------------------------------------------------------------------
# How a key arrives.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("headers,expected", [
    ({"Authorization": "Bearer abc"}, "abc"),
    ({"authorization": "bearer abc"}, "abc"),      # scheme is case-insensitive
    ({"X-API-Key": "abc"}, "abc"),
    # 🚨 CHANGED 2026-09-01 (Workstream G). This case asserted `""` — Basic was
    # "somebody else's auth". It is now OURS, and the key rides in the PASSWORD
    # half (`user:pw` → `pw`), because a browser cannot attach a bearer token to
    # a navigation and `EventSource` cannot set a header at all. The alternative
    # was minting a password, which is a second credential kind with its own
    # store, rotation and revocation, parallel to a registry that already does
    # all three. Recorded in CHANGELOG.md as the wire-contract change it is.
    ({"Authorization": "Basic dXNlcjpwdw=="}, "pw"),
    # A scheme we do not recognise is still ignored rather than treated as a
    # malformed key of ours. That claim is what the Basic case used to carry,
    # and it still needs carrying.
    ({"Authorization": "Negotiate YIIC…"}, ""),
    ({}, ""),
])
def test_where_a_key_may_be_presented(headers, expected):
    assert presented_key(_Req(headers=headers)) == expected


def test_a_request_double_without_headers_or_client_is_refused_not_crashed():
    """Fail closed. A request the proxy cannot identify is not a request it may
    serve, and it must certainly not be a 500."""
    class _Bare:
        pass

    res = IdentityResolver(IPIdentityMap(), KeyRegistry()).resolve(_Bare())
    assert not res.ok and res.denial.status == 403


# --------------------------------------------------------------------------
# Admin scope.
# --------------------------------------------------------------------------

def test_an_admin_key_grants_admin_from_anywhere():
    resolver = IdentityResolver(
        IPIdentityMap(), _reg(secret="ops", agent_id="ops", admin=True))
    assert resolver.is_admin(_Req(host="198.51.100.9", headers=_bearer("ops")))


def test_an_authenticated_non_admin_does_not_inherit_its_hosts_privileges():
    """🚨 The narrowing property. Loopback is in the admin nets, so under the
    address-only scheme this caller WAS admin. A scoped key that could only ever
    widen access and never narrow it is worthless on the machine it runs on."""
    resolver = IdentityResolver(
        IPIdentityMap(), _reg(secret="app", agent_id="app", admin=False))
    assert resolver.is_admin(_Req(host="127.0.0.1")) is True          # no key
    assert resolver.is_admin(
        _Req(host="127.0.0.1", headers=_bearer("app"))) is False      # keyed


# --------------------------------------------------------------------------
# Rule 3 — end to end through /v1/submit.
# --------------------------------------------------------------------------

def _reg(**kwargs) -> KeyRegistry:
    reg = KeyRegistry()
    reg.register(**kwargs)
    return reg


def _submit_body(agent_id: str | None = "self-declared"):
    body = {"endpoint": "chat", "call_site": "t", "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8}}
    if agent_id is not None:
        body["agent_id"] = agent_id
    return body


def _ok_backend(svc):
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call


async def _submit(svc, body, req):
    from roadstead import scheduler as _sched
    created: list = []
    orig = _sched.QueuedRequest.create.__func__
    _sched.QueuedRequest.create = classmethod(
        lambda cls, **kw: created.append(orig(cls, **kw)) or created[-1])
    try:
        resp = await asyncio.wait_for(svc.handle_submit(body, req), timeout=10.0)
    finally:
        _sched.QueuedRequest.create = classmethod(orig)
    return resp, created


@pytest.mark.asyncio
async def test_a_key_overrides_a_body_declared_agent_id(monkeypatch):
    """The fair-share key stops being caller-asserted. Before this, a caller on
    the ``/v1/submit`` door could name any ``agent_id`` — including one with a
    better DRR weight — and nothing checked."""
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "kk=真-caller:P4_HYGIENE")
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        resp, created = await _submit(
            svc, _submit_body("self-declared"),
            _Req(host="127.0.0.1", headers=_bearer("kk")))
        assert resp.status_code == 200
        assert created[-1].agent_id == "真-caller"
        assert created[-1].priority is LLMPriority.P4_HYGIENE
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_an_address_only_fills_in_an_agent_id_the_body_omitted():
    """The counterweight to the rule above. An address identifies a HOST, and
    several callers legitimately share one — so there the body knows better, and
    the registration supplies only what it left out."""
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        resp, created = await _submit(svc, _submit_body("declared"), _Req("127.0.0.1"))
        assert resp.status_code == 200
        assert created[-1].agent_id == "declared"

        resp, created = await _submit(svc, _submit_body(None), _Req("127.0.0.1"))
        assert resp.status_code == 200
        # Was the literal "unknown" before Workstream B — a junk bucket every
        # unattributed caller shared.
        assert created[-1].agent_id == "internal"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_declared_priority_still_wins_over_the_identitys_default(monkeypatch):
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "kk=batch:P4_HYGIENE")
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        body = _submit_body()
        body["priority"] = "P0_REALTIME"
        resp, created = await _submit(
            svc, body, _Req(host="127.0.0.1", headers=_bearer("kk")))
        assert resp.status_code == 200
        # 🚨 P0_REALTIME is 0. Comparing the declared priority truthily rather
        # than against None would silently demote every realtime call here.
        assert created[-1].priority is LLMPriority.P0_REALTIME
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_submit_is_closed_to_an_unidentified_caller():
    """The hole Workstream B closed: the OpenAI doors were ACL-gated and
    ``/v1/submit`` was not, on the same port."""
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit(_submit_body(), _Req("198.51.100.99")), timeout=10.0)
        assert resp.status_code == 403
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_the_openai_door_refuses_a_bad_key_in_the_openai_shape(monkeypatch):
    """A strict OpenAI client does ``resp.error.message``; a bare string body
    crashes it. Phase 5D fixed that for the 403 and the 401 must match."""
    import json

    monkeypatch.setenv("ROADSTEAD_API_KEYS", "kk=a")
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_openai_chat(
        {"model": "tier3", "messages": [{"role": "user", "content": "hi"}]},
        _Req(host="127.0.0.1", headers=_bearer("nope")))
    assert resp.status_code == 401
    err = json.loads(resp.body)["error"]
    assert err["code"] == "invalid_api_key"
    assert err["type"] == "invalid_request_error"
    assert isinstance(err["message"], str) and err["message"]
