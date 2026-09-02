"""Roadmap J1 — supplying a provider credential, and bringing an endpoint into
service from the management plane.

🚨 The whole design rests on one claim: `planned` -> `active` is NOT hot-adding
an endpoint. The stanza is already in `models.yaml`, already parsed and already
validated; the only thing that changes is membership of the routing table. These
tests hold that line from both ends — the promotion works and is real (the
router will actually dispatch to it), and the plane still cannot invent an
endpoint that nobody declared.

The load-bearing refusals, each of which looks like an inconvenience until you
read what it prevents:

* **No credential, no promotion.** `models.yaml` says in its own comment what
  `planned` is for: a deployment without the key "should not have an endpoint in
  its routing table that cannot serve". Promoting without it puts exactly that
  into the routing table, from the surface whose purpose is reporting gaps.
* **The same rule at STARTUP.** The status persists and the credential
  deliberately does not, so a restart is the one moment the rule could be
  bypassed — by the operator's own earlier promotion replaying against an
  environment that has since lost the variable.
* **No demotion under load.** The request path reads `config.endpoints.get(...)`
  after dispatch, so removing an entry from under live work is a null
  dereference. Pause drains; pause first.
"""
from __future__ import annotations

import json
import os
import pathlib
from pathlib import Path

import pytest

from roadstead import model_catalog
from roadstead.config import ProxyConfig
from roadstead.management import (
    endpoint_credential_gap,
    install_endpoint,
    remove_endpoint,
)
from roadstead.service import ProxyService

from .admin_key import ADMIN_HEADERS, enrol_admin
from .test_management_plane import _Req, _body, _svc

_ENV = "OPENROUTER_API_KEY"
_REMOTE = "spill-chat"          # `status: planned`, provider openrouter
_LOCAL = "tier1"                # routed at boot, no credential


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    """Every test states its own credential situation."""
    monkeypatch.delenv(_ENV, raising=False)


# --------------------------------------------------------------------------- #
# The mechanism
# --------------------------------------------------------------------------- #

def test_a_promoted_endpoint_is_built_the_way_a_restart_would_build_it():
    """🚨 Through `build_endpoint_kwargs`, not a second builder beside it.

    An endpoint configured one way when it boots active and another way when it
    is promoted is a drift nobody would look for — it would surface as a
    backend that behaves differently depending on how it entered service. The
    check is exact equality against what the catalog produces for the same
    entry, so a divergence cannot hide in a field neither of us thought to name.
    """
    cat = model_catalog.load_catalog()
    config = ProxyConfig(queue_db_path=":memory:")
    ep = install_endpoint(config, _REMOTE, cat)

    expected = model_catalog.build_endpoint_kwargs(cat, [cat.endpoints[_REMOTE]])[_REMOTE]
    for field, value in expected.items():
        assert getattr(ep, field) == value, field
    assert config.endpoints[_REMOTE] is ep

    remove_endpoint(config, _REMOTE)
    assert _REMOTE not in config.endpoints
    # Idempotent both ways — an operator double-clicking must not 500.
    remove_endpoint(config, _REMOTE)
    install_endpoint(config, _REMOTE, cat)
    install_endpoint(config, _REMOTE, cat)


def test_the_plane_cannot_install_an_endpoint_nobody_declared():
    """J2's boundary, held by construction: the name is resolved through the
    catalog, so there is nowhere to put an endpoint that is not in it."""
    config = ProxyConfig(queue_db_path=":memory:")
    with pytest.raises(KeyError):
        install_endpoint(config, "invented-by-the-api")


# --------------------------------------------------------------------------- #
# The credential
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_setting_a_credential_never_gives_it_back(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    secret = "sk-or-v1-THIS-MUST-NOT-APPEAR-ANYWHERE"
    resp = await svc.handle_admin_provider_credential(_Req(
        method="POST", path_params={"provider": "openrouter"},
        body={"value": secret}))
    assert resp.status_code == 200
    body = json.loads(resp.body)

    assert body["credential"] == {"env_var": _ENV, "present": True}
    assert os.environ[_ENV] == secret
    # 🚨 Not the value, and not a fragment of it. A prefix or a length narrows a
    # search; a digest IS a working credential to anyone who can compute one.
    assert secret not in resp.body.decode()
    for fragment in (secret[:8], secret[-8:], str(len(secret))):
        assert fragment not in json.dumps(body["credential"])

    # …and the trail names the variable, which is the actionable half.
    audit = json.loads((await svc.handle_admin_audit(_Req())).body)
    record = audit["entries"][-1]
    assert record["action"] == "provider.credential"
    assert record["detail"] == {"env_var": _ENV}
    assert secret not in json.dumps(audit)


@pytest.mark.asyncio
async def test_the_credential_says_it_will_not_survive_a_restart(tmp_path):
    """🚨 Disclosed in advance rather than discovered at the next restart.

    The overlay holds key digests and has never held a secret. Not persisting is
    the deliberate choice; a response that stayed quiet about it would be the
    surprising-and-correct behaviour this repo answers by DISCLOSING.
    """
    svc = _svc(tmp_path)
    body = json.loads((await svc.handle_admin_provider_credential(_Req(
        method="POST", path_params={"provider": "openrouter"},
        body={"value": "sk-x"}))).body)
    assert body["persisted"] is False
    assert any("restart" in w for w in body["warnings"])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,body,expect", [
    ("openrouter", {"value": ""}, 400),        # empty
    ("openrouter", {"value": 7}, 400),         # not a string
    ("openrouter", {"vale": "x"}, 400),        # typo, never a silent drop
    ("small-box", {"value": "x"}, 400),        # declares no api_key_env
    ("no-such-provider", {"value": "x"}, 404),
])
async def test_credential_refusals(tmp_path, provider, body, expect):
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_provider_credential(_Req(
        method="POST", path_params={"provider": provider}, body=body))
    assert resp.status_code == expect, resp.body


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_promoting_without_the_credential_is_refused(tmp_path):
    """🚨 The refusal `planned` exists to express, enforced where it is asked."""
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _REMOTE},
        body={"status": "active"}))
    assert resp.status_code == 400
    message = json.loads(resp.body)["error"]
    # It must name the variable, or an operator cannot act on it.
    assert _ENV in message
    assert _REMOTE not in svc._state.config.endpoints


@pytest.mark.asyncio
async def test_credential_then_promote_puts_it_in_the_routing_table(tmp_path):
    catalog_path = pathlib.Path(model_catalog.load_catalog.__module__ and
                                model_catalog._DEFAULT_PATH)
    catalog_before = catalog_path.read_bytes()
    svc = _svc(tmp_path)
    await svc.handle_admin_provider_credential(_Req(
        method="POST", path_params={"provider": "openrouter"},
        body={"value": "sk-or-v1-test"}))
    resp = await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _REMOTE},
        body={"status": "active"}))
    assert resp.status_code == 200, resp.body
    body = json.loads(resp.body)
    assert body["routed"] is True
    # 🚨 The FILE is unchanged, asserted on the file rather than on the
    # sentence. The response should also say so — an override that looked like
    # an edit to models.yaml would be a lie about where truth lives — but a test
    # that only reads the prose passes against a handler that writes the file
    # and apologises.
    assert body["declared_status"] == "planned"
    assert catalog_path.read_bytes() == catalog_before, (
        "the write touched models.yaml — the overlay is meant to be the only "
        "writer, which is what makes a bad save survivable")
    assert any("models.yaml" in w and "unchanged" in w for w in body["warnings"])
    assert _REMOTE in svc._state.config.endpoints


@pytest.mark.asyncio
async def test_a_promoted_endpoint_is_actually_ROUTED_not_just_recorded(tmp_path):
    """🚨 The half that a management-plane assertion alone would miss.

    `/rs/v1/admin/providers` saying `routed: true` proves the plane changed its
    own mind. What matters is whether the ROUTER agrees: `enriched.facts()`
    computed `routed` from the CATALOG entry, so a promoted endpoint reported as
    unrouted on `/rs/v1/models` — and `intent.py` filters on exactly that field,
    which made a pin to it 404 while the admin view called it live. The two
    sources could not disagree before J1, because the routing table was built
    from the catalog at startup; that is why the wrong branch survived.
    """
    svc = _svc(tmp_path)
    await svc.handle_admin_provider_credential(_Req(
        method="POST", path_params={"provider": "openrouter"},
        body={"value": "sk-or-v1-test"}))
    await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _REMOTE},
        body={"status": "active"}))

    facts = {f.endpoint: f.routed for f in svc._enriched.facts()}
    assert facts[_REMOTE] is True, (
        "the router does not consider the promoted endpoint routable, so a pin "
        "to it will 404 while the admin plane reports it live")


# --------------------------------------------------------------------------- #
# Demotion
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_demotion_takes_it_out(tmp_path):
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _LOCAL}, body={"status": "planned"}))
    assert resp.status_code == 200, resp.body
    assert _LOCAL not in svc._state.config.endpoints
    assert {f.endpoint for f in svc._enriched.facts() if f.routed} != set()


@pytest.mark.asyncio
async def test_demotion_is_refused_while_work_is_in_flight(tmp_path, monkeypatch):
    """🚨 Removal is the dangerous direction and the danger is not theoretical:
    `correction.py` alone reads `config.endpoints.get(req.endpoint)` at six
    sites AFTER dispatch. Pause already drains, so the safe order exists — this
    refuses rather than growing a second drain beside it."""
    svc = _svc(tmp_path)
    monkeypatch.setattr(svc._state.scheduler, "endpoint_snapshot",
                        lambda name: {"in_flight": 3, "max_slots": 4, "queued": 0})
    resp = await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _LOCAL}, body={"status": "planned"}))
    assert resp.status_code == 400
    message = json.loads(resp.body)["error"]
    assert "in flight" in message and "Pause" in message
    assert _LOCAL in svc._state.config.endpoints, "it was removed anyway"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint,body,expect", [
    ("nope", {"status": "active"}, 404),
    (_LOCAL, {"status": "retired"}, 400),      # catalog-only
    (_LOCAL, {"status": "on_demand"}, 400),    # catalog-only
    (_LOCAL, {"stats": "planned"}, 400),       # typo, never a silent drop
])
async def test_status_refusals(tmp_path, endpoint, body, expect):
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": endpoint}, body=body))
    assert resp.status_code == expect, resp.body


# --------------------------------------------------------------------------- #
# Across a restart
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_a_promotion_survives_a_restart_and_its_credential_does_not(tmp_path):
    """🚨 The asymmetry, and the hole it would have left.

    The status persists (an operator's routing decision should be findable
    again); the credential does not (a secret at rest is a different posture).
    Left there, a restart would be the one moment a routed endpoint that cannot
    serve appears — the promotion replaying against an environment that no
    longer has the variable. `endpoint_credential_gap` is consulted by BOTH
    paths so the rule cannot hold in one and lapse in the other.
    """
    store = tmp_path / "admin_overlay.json"
    svc = _svc(tmp_path)
    await svc.handle_admin_provider_credential(_Req(
        method="POST", path_params={"provider": "openrouter"},
        body={"value": "sk-or-v1-test"}))
    await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _REMOTE},
        body={"status": "active"}))
    saved = json.loads(store.read_text())
    assert saved["catalog"]["endpoints"][_REMOTE] == {"status": "active"}
    assert "sk-or-v1-test" not in store.read_text(), "the store holds a secret"

    # Restart WITH the variable still exported: the promotion comes back.
    again = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q2.db"),
                                     admin_store_path=str(store)))
    assert _REMOTE in again._state.config.endpoints

    # Restart WITHOUT it: the endpoint stays out, and the operator's decision is
    # still on record rather than quietly deleted.
    del os.environ[_ENV]
    third = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q3.db"),
                                     admin_store_path=str(store)))
    assert _REMOTE not in third._state.config.endpoints, (
        "a restart put an endpoint that cannot serve back into the routing "
        "table — the exact condition the promotion refusal prevents, arriving "
        "through the back door")
    assert third._state.admin_overlay.catalog["endpoints"][_REMOTE] == {"status": "active"}


def test_the_credential_gap_is_decided_in_one_place():
    """Both promote paths must consult the same function, or the rule holds in
    one and lapses in the other — which is the whole hazard above."""
    source = (Path(__file__).resolve().parents[1]
              / "roadstead" / "management.py").read_text()
    assert source.count("endpoint_credential_gap(") >= 3, (
        "expected the definition plus a call from the handler and from the "
        "overlay's startup replay")


@pytest.mark.asyncio
async def test_a_local_endpoint_needs_no_credential_to_be_promoted(tmp_path):
    """The gate is about a credential the provider DECLARES, not about being
    remote. A llama.cpp endpoint has no `api_key_env` and must not be blocked by
    a rule written for one that has."""
    svc = _svc(tmp_path)
    await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _LOCAL}, body={"status": "planned"}))
    resp = await svc.handle_admin_endpoint_status(_Req(
        method="POST", path_params={"endpoint": _LOCAL}, body={"status": "active"}))
    assert resp.status_code == 200, resp.body
    assert _LOCAL in svc._state.config.endpoints
    assert endpoint_credential_gap(_LOCAL) is None
