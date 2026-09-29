"""`calls_push` — how the scope is declared, kept, and refused a widening.

The route-level containment (the scope reaches ONE route and no other) is
driven over the real route table in `tests/e2e/test_calls_push_scope.py`. This
file pins the plumbing that decides whether the scope survives being issued:

* it is a complete credential and is **never combined with `admin`**, at the
  registry and at the management boundary alike;
* an **address can never confer it**, and the shared identity grammar cannot
  express it — a segment the address ACL parsed but had to ignore would read as
  a scope granted;
* it **survives a restart, a rotation and the keys file**, because a scope that
  lives only in memory replays a pusher as a plain key and the fleet goes back to
  silently refused pushes with no change anyone made;
* the admin gate itself is **byte-for-byte what it was** for every credential
  that is not the ingest scope.
"""
from __future__ import annotations

import json

import pytest

from roadstead.acl import IPIdentityMap
from roadstead.config import ProxyConfig
from roadstead.identity import (
    IdentityResolver, KeyRegistry, Principal, parse_identity_spec,
)
from roadstead.management import AdminOverlay, Invalid, validate_key_create
from roadstead.service import ProxyService

from tests.admin_key import ADMIN_HEADERS, enrol_admin


class _Req:
    def __init__(self, *, host="127.0.0.1", headers=None, method="POST",
                 body=None, path_params=None):
        class _C:
            pass
        _C.host = host
        self.client = _C()
        self.headers = {"Content-Type": "application/json",
                        **(headers if headers is not None else ADMIN_HEADERS)}
        self.method = method
        self.query_params: dict = {}
        self.path_params = path_params or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _svc(tmp_path) -> ProxyService:
    svc = ProxyService(ProxyConfig(
        queue_db_path=str(tmp_path / "q.db"),
        admin_store_path=str(tmp_path / "admin_overlay.json")))
    enrol_admin(svc)
    return svc


def _resolver(keys: KeyRegistry) -> IdentityResolver:
    return IdentityResolver(acl=IPIdentityMap(), keys=keys)


# ---------------------------------------------------------------------------
# 1 · A complete credential on its own, never an add-on to admin
# ---------------------------------------------------------------------------

def test_the_scope_is_off_by_default_and_grants_no_admin():
    keys = KeyRegistry()
    keys.register(secret="s", agent_id="a", key_id="a", calls_push=True)
    p = keys.resolve("s")
    assert p.calls_push is True
    assert p.admin is False and p.may_admin_write is False
    assert Principal(agent_id="x").calls_push is False


def test_the_registry_refuses_the_scope_beside_admin(caplog):
    """Refused, not tolerated: beside `admin_readonly` it would read as a
    read-only admin key that can write — a narrowing another field cancels."""
    keys = KeyRegistry()
    assert keys.register(secret="s", agent_id="a", key_id="a", admin=True,
                         admin_readonly=True, calls_push=True) is None
    assert keys.resolve("s") is None
    assert "never combined with admin" in caplog.text


def test_the_management_boundary_refuses_it_beside_admin():
    with pytest.raises(Invalid, match="never combined with admin"):
        validate_key_create({"agent_id": "a", "admin": True, "calls_push": True})
    with pytest.raises(Invalid, match="JSON boolean"):
        validate_key_create({"agent_id": "a", "calls_push": "yes"})
    assert validate_key_create(
        {"agent_id": "a", "calls_push": True})["calls_push"] is True


def test_an_address_and_the_shared_grammar_cannot_confer_it(caplog):
    """`ROADSTEAD_ACL` shares the grammar with `ROADSTEAD_API_KEYS`, so a
    `calls_push` segment would be parsed for addresses and could only be
    ignored — which reads as granted. It is refused as unrecognised instead."""
    parsed = parse_identity_spec("pusher:calls_push")
    assert len(parsed) == 6  # the tuple did not grow a scope for addresses
    assert "unrecognised segment 'calls_push'" in caplog.text

    keys = KeyRegistry()
    keys._load_env("k-env=pusher:calls_push")
    assert keys.resolve("k-env").calls_push is False


# ---------------------------------------------------------------------------
# 2 · The gate is what it was for everyone who is not the ingest scope
# ---------------------------------------------------------------------------

class _Peer:
    def __init__(self, headers, method="POST", host="127.0.0.1"):
        class _C:
            pass
        _C.host = host
        self.client = _C()
        self.headers = {"Content-Type": "application/json", **headers}
        self.method = method


def test_the_ingest_gate_agrees_with_the_admin_gate_for_every_other_credential():
    keys = KeyRegistry()
    keys.register(secret="adm", agent_id="ops", key_id="adm", admin=True)
    keys.register(secret="ro", agent_id="ops", key_id="ro", admin=True,
                  admin_readonly=True)
    keys.register(secret="plain", agent_id="p", key_id="plain")
    r = _resolver(keys)
    for headers in ({"X-API-Key": "adm"}, {"X-API-Key": "ro"},
                    {"X-API-Key": "plain"}, {"X-API-Key": "unknown"}, {}):
        for method in ("POST", "GET"):
            req = _Peer(headers, method)
            a, i = r.admin_denial(req), r.calls_push_denial(req)
            assert (a is None) == (i is None), (headers, method)
            if a is not None:
                assert (a.status, a.code, a.message) == (i.status, i.code, i.message)


def test_the_network_gate_is_shared():
    """The scope is narrower than admin and must not be reachable from anywhere
    admin is not."""
    keys = KeyRegistry()
    keys.register(secret="s", agent_id="a", key_id="a", calls_push=True)
    r = _resolver(keys)
    ok = r.calls_push_denial(_Peer({"X-API-Key": "s"}))
    assert ok is None
    far = r.calls_push_denial(_Peer({"X-API-Key": "s"}, host="198.51.100.4"))
    assert far is not None and "not reachable from 198.51.100.4" in far.message


def test_a_binding_narrows_the_scope_like_any_key():
    keys = KeyRegistry()
    keys.register(secret="s", agent_id="a", key_id="a", calls_push=True,
                  bind=["10.9.0.0/16"])
    r = _resolver(keys)
    denial = r.calls_push_denial(_Peer({"X-API-Key": "s"}))
    assert denial is not None and denial.status == 401
    assert "is bound to 10.9.0.0/16" in denial.message


# ---------------------------------------------------------------------------
# 3 · It survives being issued
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_runtime_enrolment_reports_the_scope_and_survives_a_restart(tmp_path):
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_keys(_Req(body={
        "agent_id": "calls-ingest", "calls_push": True,
        "id": "ingest", "bind": ["127.0.0.0/8"]}))
    assert resp.status_code == 201
    body = json.loads(resp.body)
    assert body["calls_push"] is True and body["admin"] is False
    secret = body["key"]

    listed = json.loads((await svc.handle_admin_keys(
        _Req(method="GET"))).body)
    (row,) = [k for k in listed["keys"] if k["key_id"] == "ingest"]
    assert row["calls_push"] is True and row["admin"] is False

    audit = svc._state.admin_overlay.audit[-1]
    assert audit["action"] == "key.enrol" and audit["detail"]["calls_push"] is True
    assert secret not in json.dumps(svc._state.admin_overlay.audit)

    # A restart: a fresh registry, the overlay replayed over it.
    restarted = KeyRegistry()
    overlay = AdminOverlay(tmp_path / "admin_overlay.json")
    overlay.apply(restarted, ProxyConfig())
    replayed = restarted.resolve(secret)
    assert replayed is not None, "the key did not come back after a restart"
    assert replayed.calls_push is True and replayed.admin is False
    assert restarted.binding("ingest") == ["127.0.0.0/8"]


@pytest.mark.asyncio
async def test_a_rotation_keeps_the_scope(tmp_path):
    """A rotation is a new SECRET for the same identity; dropping the scope
    would take the pusher back to refused at the moment the key changed."""
    svc = _svc(tmp_path)
    keys = svc._state.identity.keys
    keys.register(secret="old", agent_id="calls-ingest", key_id="old",
                  calls_push=True)
    resp = await svc.handle_admin_key_rotate(
        _Req(path_params={"key_id": "old"}, body={}))
    assert resp.status_code == 201
    body = json.loads(resp.body)
    assert body["calls_push"] is True
    successor = keys.resolve(body["key"])
    assert successor.calls_push is True and successor.admin is False


def test_the_keys_file_declares_it(tmp_path):
    f = tmp_path / "keys.yaml"
    f.write_text("keys:\n"
                 "  - {id: ingest, agent_id: calls-ingest, key: k-file,"
                 " calls_push: true}\n")
    keys = KeyRegistry()
    keys._load_file(f)
    p = keys.resolve("k-file")
    assert p is not None and p.calls_push is True and p.admin is False


def test_the_scope_is_published_per_key_and_never_the_secret():
    keys = KeyRegistry()
    keys.register(secret="s3cret", agent_id="a", key_id="a", calls_push=True)
    rows = keys.snapshot()
    assert rows[0]["calls_push"] is True
    assert "s3cret" not in json.dumps(rows)
