"""🚨 The admin plane takes BOTH a network gate and a credential.

Changed 2026-09-01. Until then an ADDRESS granted admin: a request from loopback
with no credential at all could pause a backend, re-weight a caller's quota or
flip a runtime flag, and the audit trail recorded the change with
``key_id: null, source: "ip"`` — the system knew nobody had authenticated and
allowed it anyway.

That default was written for the INFERENCE door, where "already on the box" is a
fair proxy for "allowed" on a local-first proxy. The admin plane, and later the
UI, were added on the same port and inherited it. Nobody re-asked whether being
on the box should also mean being allowed to read every caller's traffic and
change policy.

The two questions are separate now and both must pass:

* **reach** — ``acl.may_reach_admin``, from ``ROADSTEAD_ADMIN_NETS``. Loopback is
  unconditional; docker-internal is a default that naming any net drops.
* **credential** — an ``admin``-scoped key, checked by the registry.

This file owns that doctrine. ``tests/admin_key.py`` authenticates the tests that
are about what the plane DOES, and it must never become the place the gate is
tested — a regression that reopened the plane would be masked by the helper the
plane's own tests depend on.
"""
from __future__ import annotations

import pytest

from roadstead.acl import IPIdentityMap
from roadstead.config import ProxyConfig
from roadstead.identity import IdentityResolver, KeyRegistry
from roadstead.service import ProxyService

from tests.admin_key import ADMIN_HEADERS, enrol_admin


class _Req:
    def __init__(self, *, host="127.0.0.1", headers=None, method="GET", body=None,
                 path_params=None):
        class _C:
            pass
        _C.host = host
        self.client = _C()
        self.headers = {} if headers is None else headers
        self.method = method
        self.query_params: dict = {}
        self.path_params = path_params or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _svc(tmp_path, **cfg) -> ProxyService:
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db"),
                                    admin_store_path=str(tmp_path / "o.json"), **cfg))


# ---------------------------------------------------------------------------
# 1 · An address is a gate, never a grant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loopback_without_a_credential_cannot_mutate(tmp_path):
    """🚨 THE regression this change exists to prevent, as one assertion.

    If this test ever passes with a 200, the hole is back and everything else
    in this file is decoration.
    """
    svc = _svc(tmp_path)
    enrol_admin(svc)                      # a key EXISTS; it is simply not presented
    resp = await svc.handle_admin_flags(
        _Req(method="POST", body={"context_gate_enforce": True}))
    assert resp.status_code == 403, (
        "an unauthenticated request from loopback mutated a runtime flag")
    assert svc._state.flags.get("context_gate_enforce") is False, (
        "the flag CHANGED despite the refusal — the gate refused after acting")


@pytest.mark.asyncio
async def test_loopback_without_a_credential_cannot_read_either(tmp_path):
    """Reads are gated too. `/rs/v1/admin/callers` names every caller, their
    endpoints, token counts and timings — it is the live form of the traffic
    log, and treating reads as harmless is how a control plane leaks."""
    svc = _svc(tmp_path)
    enrol_admin(svc)
    assert (await svc.handle_admin_callers(_Req())).status_code == 403


def test_an_address_alone_never_carries_the_admin_scope():
    """The unit form, at the resolver rather than through a handler."""
    acl = IPIdentityMap()
    acl.register_admin_net("127.0.0.1")          # the strongest thing an operator can write
    resolver = IdentityResolver(acl, KeyRegistry())
    assert resolver.is_admin(_Req(host="127.0.0.1")) is False


# ---------------------------------------------------------------------------
# 2 · The network gate refuses before any credential is read
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_valid_credential_does_not_answer_a_network_refusal(tmp_path):
    """🚨 The ordering, and it is the point of having two gates.

    A credential is a thing an attacker can steal; an address is a thing they
    have to be. Checking the network first also means a blocked address never
    learns whether the key it presented was valid.
    """
    svc = _svc(tmp_path)
    headers = enrol_admin(svc)
    outside = _Req(host="198.51.100.7", headers=headers, method="POST",
                   body={"context_gate_enforce": True})
    resp = await svc.handle_admin_flags(outside)
    assert resp.status_code == 403
    body = resp.body.decode()
    assert "198.51.100.7" in body
    assert "network" in body.lower(), (
        "the refusal did not say it was about the network — an operator with a "
        "valid key needs to be told no credential will fix this")


@pytest.mark.asyncio
async def test_naming_a_net_admits_it_and_drops_the_docker_default(tmp_path):
    """What `ROADSTEAD_ADMIN_NETS` now means, both halves of it."""
    svc = _svc(tmp_path)
    headers = enrol_admin(svc)
    svc._state.acl.register_admin_net("198.51.100.0/24")

    ok = await svc.handle_admin_flags(_Req(host="198.51.100.7", headers=headers))
    assert ok.status_code == 200
    # 🚨 A /12 is a weak gate, so naming a net drops it. An operator who has said
    # what they want has said it.
    assert svc._state.acl.may_reach_admin("172.17.0.5") is False
    # Loopback survives everything — it is where the bootstrap key is usable.
    assert svc._state.acl.may_reach_admin("127.0.0.1") is True


# ---------------------------------------------------------------------------
# 3 · The bootstrap key: never open, never locked out
# ---------------------------------------------------------------------------

def test_a_bootstrap_key_is_minted_when_nothing_is_configured(tmp_path):
    svc = _svc(tmp_path)
    assert len(svc._state.identity.keys) == 0
    svc._mint_bootstrap_admin_key()
    assert len(svc._state.identity.keys) == 1


def test_a_bootstrap_key_does_not_flip_the_inference_identity_regime(tmp_path):
    """🚨 §1.5 rule 2, protected against the fix for a different problem.

    `KeyRegistry.configured` governs whether a presented key is significant on
    the INFERENCE door. Every OpenAI client sends an `Authorization` header
    whether anybody meant it to or not, so treating one as significant before an
    operator configured any key would 401 the world. A key the proxy minted for
    its own dashboard is not an operator adopting API-key identity.
    """
    svc = _svc(tmp_path)
    svc._mint_bootstrap_admin_key()
    assert svc._state.identity.keys.configured is False
    # A placeholder credential on the inference door is still ignored, not refused.
    res = svc._state.identity.resolve(
        _Req(headers={"Authorization": "Bearer sk-placeholder"}))
    assert res.ok and res.principal.source == "ip"


@pytest.mark.asyncio
async def test_the_minted_key_actually_works_on_the_plane(tmp_path):
    """🚨 Driven through a real handler with the real secret.

    Written first as a check that a bootstrap digest had been recorded, which
    asserted nothing about whether the credential AUTHENTICATES — the failure
    mode being guarded against is precisely "minted a key that does not work",
    where the operator has no way in and a live secret is in the log for
    nothing. It now presents the key to the plane.
    """
    svc = _svc(tmp_path)
    secret = svc._mint_bootstrap_admin_key()
    assert secret, "nothing was minted"

    denied = await svc.handle_admin_flags(_Req())
    assert denied.status_code == 403

    ok = await svc.handle_admin_flags(
        _Req(method="POST", body={"context_gate_enforce": True},
             headers={"X-API-Key": secret, "Content-Type": "application/json"}))
    assert ok.status_code == 200, ok.body
    assert svc._state.flags.get("context_gate_enforce") is True


def test_no_second_key_is_minted_once_an_operator_configures_one(tmp_path):
    """A new admin credential appearing on every restart of a configured
    deployment would be a credential nobody expects and nobody revokes."""
    svc = _svc(tmp_path)
    svc._state.identity.keys.register(secret="sk-real", agent_id="ops", admin=True)
    svc._mint_bootstrap_admin_key()
    assert len(svc._state.identity.keys) == 1


# ---------------------------------------------------------------------------
# 4 · The UI door still answers a browser
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_ui_door_refuses_401_so_a_browser_can_answer(tmp_path, monkeypatch):
    """A 403 gives a browser no way to ask for a password. The plane answers 403
    and the door answers 401 + WWW-Authenticate; that split has to survive the
    gate being added in front of it."""
    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")
    svc = _svc(tmp_path)
    enrol_admin(svc)
    resp = await svc.handle_admin_ui(_Req())
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"].startswith("Basic ")
