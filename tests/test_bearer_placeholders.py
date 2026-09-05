"""``ROADSTEAD_BEARER_PLACEHOLDERS`` — the literal an SDK insists on sending.

🚨 **The rollback this exists for.** An OpenAI SDK will not construct a client
with an empty ``api_key``, so a fleet that authorises by ADDRESS has to send
something, and two of its callers sent ``Authorization: Bearer not-needed`` on
every ``/v1/chat/completions`` call. The proxy they came from ignored any
bearer. This one applies §1.5 rule 1 — an unregistered credential is a 401 and
never falls back to the address — which is correct, and which took the
interactive chat path down for eleven minutes on 2026-09-04. The cutover was
rolled back.

So a deployment may DECLARE those literals and a declared one is read as though
the header had not been sent. What every test below is really guarding is that
the shim stayed narrow:

* only an **exact, case-sensitive** match qualifies — nothing is inferred from
  shape, because a heuristic that demotes a credential is a silencer;
* it yields the **address's** identity and therefore grants no admin, and
  ``ROADSTEAD_REQUIRE_API_KEY`` still refuses it;
* **Basic is untouched** — it is the management UI's channel and the CSRF gate
  keys off it;
* a placeholder colliding with a registered key **refuses to start**, because
  that key would keep working as a weaker identity with nothing to say so.

🚨 **Every test here configures an operator key.** Without one, §1.5 rule 2
already ignores a presented key and every assertion below would pass for the
wrong reason — which is exactly the state the fleet upgraded OUT of.
``test_the_fixture_is_not_vacuous`` is the counterweight that says so.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.acl import IPIdentityMap
from roadstead.backend import BackendResponse
from roadstead.config import LLMPriority, ProxyConfig
from roadstead.identity import (
    BearerPlaceholders,
    IdentityResolver,
    KeyRegistry,
)
from roadstead.service import ProxyService
from roadstead import scheduler as sched

from tests.admin_key import enrol_admin


class _Req:
    def __init__(self, host="127.0.0.1", headers=None, method="POST", body=None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.headers = dict(headers) if headers else {}
        # The CSRF gate on the mutating admin plane requires this on every
        # request it sees, so the double carries it unconditionally rather than
        # only when a test remembers to.
        self.headers.setdefault("Content-Type", "application/json")
        self.method = method
        self.query_params: dict = {}
        self.path_params: dict = {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("ROADSTEAD_API_KEYS", "ROADSTEAD_API_KEYS_FILE", "ROADSTEAD_ACL",
                "LLM_PROXY_ACL", "ROADSTEAD_ADMIN_NETS", "ROADSTEAD_REQUIRE_API_KEY",
                "ROADSTEAD_BEARER_PLACEHOLDERS", "ROADSTEAD_LEGACY_SUBMIT"):
        monkeypatch.delenv(var, raising=False)


def _resolver(*, placeholders=("not-needed",), keys=True) -> IdentityResolver:
    """The resolver these tests ask, with an address registration to fall to."""
    acl = IPIdentityMap()
    acl.register("203.0.113.10", "lan-generic", LLMPriority.P3_INGESTION)
    registry = KeyRegistry()
    if keys:
        registry.register(secret="real-key", agent_id="enrolled", key_id="prod")
    return IdentityResolver(
        acl, registry, placeholders=BearerPlaceholders(placeholders))


# ---------------------------------------------------------------------------
# 1 · What a declared placeholder does, and what it does not
# ---------------------------------------------------------------------------

def test_the_fixture_is_not_vacuous():
    """🚨 The counterweight, first.

    With an operator key configured and NOTHING declared, the literal is an
    unregistered credential and rule 1 refuses it. That is the production
    failure, reproduced — and without this assertion every test below would
    still pass on a build where the whole feature was deleted, because §1.5 rule
    2 would be doing the work instead.
    """
    res = _resolver(placeholders=()).resolve(_Req(host="203.0.113.10",
                                                  headers=_bearer("not-needed")))
    assert not res.ok
    assert res.denial.status == 401 and res.denial.code == "invalid_api_key"


def test_a_declared_placeholder_is_identified_by_address():
    res = _resolver().resolve(_Req(host="203.0.113.10",
                                   headers=_bearer("not-needed")))
    assert res.ok
    assert res.principal.agent_id == "lan-generic"
    assert res.principal.source == "ip"
    assert res.principal.authenticated is False


def test_it_is_the_same_answer_as_sending_no_header_at_all():
    """The promise, stated as an equality rather than as two similar asserts.

    'Read as though the header were absent' is the whole contract, and the way
    it goes wrong is a principal that is *nearly* the address's — a band filled
    in, a key_id retained — which no single-field assertion would catch.
    """
    resolver = _resolver()
    with_header = resolver.resolve(_Req(host="203.0.113.10",
                                        headers=_bearer("not-needed")))
    without = resolver.resolve(_Req(host="203.0.113.10"))
    assert with_header.principal == without.principal


def test_an_unregistered_key_that_is_not_declared_still_401s():
    """Rule 1 is narrowed by this knob, not repealed."""
    res = _resolver().resolve(_Req(host="203.0.113.10", headers=_bearer("nope")))
    assert not res.ok and res.denial.status == 401
    assert "not ignored in favour of the source address" in res.denial.message.lower()


def test_a_registered_key_still_beats_the_address_when_the_list_is_set():
    res = _resolver().resolve(_Req(host="203.0.113.10", headers=_bearer("real-key")))
    assert res.ok
    assert res.principal.agent_id == "enrolled"
    assert res.principal.source == "api_key"


@pytest.mark.parametrize("presented", ["NOT-NEEDED", "Not-Needed",
                                       "not-needed-x", "xnot-needed"])
def test_matching_is_exact_and_case_sensitive(presented):
    """🚨 Nothing is inferred from shape.

    A case-insensitive or prefix match would make values an operator never wrote
    into credentials-that-are-not — a hole they cannot see in their own
    configuration.
    """
    res = _resolver().resolve(_Req(host="203.0.113.10", headers=_bearer(presented)))
    assert not res.ok and res.denial.status == 401


def test_surrounding_whitespace_is_not_a_different_credential():
    """The one thing that is NOT a case-sensitivity exception.

    ``presented_key`` strips the token before anything compares it, and the
    declaration side strips too — so ``Bearer not-needed `` is the same
    credential a caller meant to send, not a near-miss to be refused. Pinned
    because the two halves must strip together: if either stopped, a header a
    proxy or a config file padded would 401 for a reason nobody could see.
    """
    res = _resolver().resolve(
        _Req(host="203.0.113.10", headers={"Authorization": "Bearer not-needed "}))
    assert res.ok and res.principal.agent_id == "lan-generic"


def test_basic_is_untouched_even_when_its_password_is_a_placeholder():
    """🚨 Deliberately NOT covered, and the reason is the CSRF gate.

    Basic is the management UI's channel: a browser attaches a cached Basic
    credential to any request to this origin on its own, and
    ``X-Roadstead-Request: 1`` is the one signal that tells the UI's own
    ``fetch()`` from a forged cross-site submission riding it. Reading a
    placeholder *password* as 'no credential' would move that request onto a
    path where the signal no longer applies. So a Basic password that happens to
    be a declared placeholder is an ordinary unregistered credential.
    """
    import base64

    blob = base64.b64encode(b"ui:not-needed").decode()
    res = _resolver().resolve(
        _Req(host="203.0.113.10", headers={"Authorization": f"Basic {blob}"}))
    assert not res.ok and res.denial.status == 401


def test_an_x_api_key_behind_a_placeholder_still_authenticates():
    """'As if the header were absent' is meant literally.

    A caller that also set ``X-API-Key`` presented a credential deliberately.
    Dropping it here would be precisely the silent demotion the load-time
    collision check exists to prevent, arrived at from the other direction.
    """
    res = _resolver().resolve(_Req(
        host="203.0.113.10",
        headers={**_bearer("not-needed"), "X-API-Key": "real-key"}))
    assert res.ok and res.principal.agent_id == "enrolled"


def test_a_placeholder_does_not_satisfy_require_api_key():
    """Nothing was presented, so a deployment that requires a key still says so."""
    acl = IPIdentityMap()
    acl.register("203.0.113.10", "lan-generic", LLMPriority.P3_INGESTION)
    keys = KeyRegistry()
    keys.register(secret="real-key", agent_id="enrolled")
    resolver = IdentityResolver(acl, keys, require_key=True,
                                placeholders=BearerPlaceholders(["not-needed"]))
    res = resolver.resolve(_Req(host="203.0.113.10", headers=_bearer("not-needed")))
    assert not res.ok and res.denial.status == 401
    assert "an API key is required" in res.denial.message


# ---------------------------------------------------------------------------
# 2 · Loading the list
# ---------------------------------------------------------------------------

def test_the_default_is_empty_which_is_off(monkeypatch):
    monkeypatch.delenv("ROADSTEAD_BEARER_PLACEHOLDERS", raising=False)
    assert not BearerPlaceholders.from_env()
    assert BearerPlaceholders.from_env().values == ()


def test_empty_and_whitespace_entries_are_ignored(monkeypatch):
    monkeypatch.setenv("ROADSTEAD_BEARER_PLACEHOLDERS",
                       " not-needed , ,, EMPTY ,not-needed")
    placeholders = BearerPlaceholders.from_env()
    assert placeholders.values == ("not-needed", "EMPTY")
    assert placeholders.declares("not-needed") and placeholders.declares("EMPTY")
    assert not placeholders.declares("")


def test_a_non_empty_list_warns_and_names_the_removal_condition(monkeypatch, caplog):
    """A shim that logs nothing at startup is a shim nobody remembers is
    load-bearing — the warning has to say what makes it removable."""
    monkeypatch.setenv("ROADSTEAD_BEARER_PLACEHOLDERS", "not-needed")
    with caplog.at_level("WARNING", logger="roadstead.identity"):
        BearerPlaceholders.from_env()
    assert "not-needed" in caplog.text
    assert "real keys" in caplog.text
    assert "placeholder_bearers" in caplog.text


def test_a_placeholder_that_is_a_registered_key_refuses_to_start():
    """🚨 A refusal, where this package usually reports and carries on.

    The colliding key would still be presented and still be accepted — as a
    weaker, address-derived identity with a different fair share and no admin.
    That is a working credential silently demoted, which is the failure rule 1
    exists to prevent; a configuration that cannot be served safely is one to
    refuse. The message names the key so an operator knows which of the two to
    remove.
    """
    keys = KeyRegistry()
    keys.register(secret="not-needed", agent_id="chatter", key_id="chat-prod")
    with pytest.raises(ValueError) as exc:
        IdentityResolver(IPIdentityMap(), keys,
                         placeholders=BearerPlaceholders(["not-needed"]))
    assert "chat-prod" in str(exc.value)
    assert "ROADSTEAD_BEARER_PLACEHOLDERS" in str(exc.value)


def test_the_collision_check_is_re_runnable_against_a_later_registry():
    """What ``ProxyState`` calls after the management overlay is applied — the
    overlay enrols keys after the resolver exists, and one it restores must not
    be a value this list would demote."""
    resolver = _resolver()
    resolver.assert_placeholders_are_not_keys()          # clean, as constructed
    resolver.keys.register(secret="not-needed", agent_id="late", key_id="late")
    with pytest.raises(ValueError):
        resolver.assert_placeholders_are_not_keys()


def test_a_proxy_service_refuses_to_start_on_a_collision(monkeypatch, tmp_path):
    """The same refusal where an operator meets it: process start."""
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "not-needed=chatter")
    monkeypatch.setenv("ROADSTEAD_BEARER_PLACEHOLDERS", "not-needed")
    with pytest.raises(ValueError):
        ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))


# ---------------------------------------------------------------------------
# 3 · The inventory — a counter and one line a day
# ---------------------------------------------------------------------------

def test_one_line_per_placeholder_and_address_per_day(caplog):
    """Per DAY because a proxy that restarts nightly would otherwise report the
    same inventory every morning; per (value, address) because the inventory is
    a list of callers to migrate and a fleet-wide line names none of them."""
    resolver = _resolver()
    with caplog.at_level("INFO", logger="roadstead.identity"):
        for _ in range(3):
            resolver.resolve(_Req(host="203.0.113.10", headers=_bearer("not-needed")))
        resolver.resolve(_Req(host="203.0.113.11", headers=_bearer("not-needed")))

    lines = [r for r in caplog.records
             if "treated as no credential" in r.getMessage()]
    assert len(lines) == 2, "one line per (placeholder, address) per UTC day"
    assert lines[0].getMessage() == (
        "placeholder bearer not-needed from 203.0.113.10 treated as no "
        "credential (ROADSTEAD_BEARER_PLACEHOLDERS)")
    # 🚨 The unregistered address is still counted. It resolves to no identity
    # and is refused, but it IS a caller still sending the placeholder, which is
    # what the inventory is a list of.
    assert resolver.placeholders.tally["count"] == 4
    assert resolver.placeholders.tally["by_address"] == {
        "203.0.113.10": 3, "203.0.113.11": 1}


@pytest.mark.asyncio
async def test_the_counter_is_published_on_v1_status(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADSTEAD_BEARER_PLACEHOLDERS", "not-needed")
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "real-key=enrolled")
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    svc._state.identity.resolve(_Req(host="127.0.0.1", headers=_bearer("not-needed")))
    body = json.loads((await svc.handle_status(_Req(method="GET"))).body)
    assert body["reliability"]["placeholder_bearers"] == {
        "count": 1, "by_address": {"127.0.0.1": 1}}


@pytest.mark.asyncio
async def test_the_counter_stays_empty_where_the_knob_is_unset(tmp_path):
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    body = json.loads((await svc.handle_status(_Req(method="GET"))).body)
    assert body["reliability"]["placeholder_bearers"] == {
        "count": 0, "by_address": {}}


# ---------------------------------------------------------------------------
# 4 · Every door that resolves identity
# ---------------------------------------------------------------------------

def _ok_backend(svc):
    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = call


async def _enqueued(svc, coro_factory):
    """Run one request and return the QueuedRequest it produced."""
    seen: list = []
    orig = sched.QueuedRequest.create.__func__
    sched.QueuedRequest.create = classmethod(
        lambda cls, **kw: seen.append(orig(cls, **kw)) or seen[-1])
    try:
        resp = await asyncio.wait_for(coro_factory(), timeout=10.0)
    finally:
        sched.QueuedRequest.create = classmethod(orig)
    return resp, (seen[-1] if seen else None)


@pytest.fixture
async def svc(tmp_path, monkeypatch):
    """A live service with an operator key configured — so rule 1 is in play —
    and ``not-needed`` declared."""
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "real-key=enrolled")
    monkeypatch.setenv("ROADSTEAD_BEARER_PLACEHOLDERS", "not-needed")
    monkeypatch.setenv("ROADSTEAD_LEGACY_SUBMIT", "1")
    s = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db"),
                                 admin_store_path=str(tmp_path / "o.json")))
    _ok_backend(s)

    async def healthy(ep):
        return True

    s._backend.probe_health = healthy
    await s.startup()
    try:
        yield s
    finally:
        await s.shutdown()


_CHAT = {"model": "chat", "messages": [{"role": "user", "content": "hi"}],
         "max_tokens": 8}


@pytest.mark.asyncio
async def test_the_openai_door_serves_a_placeholder_and_refuses_a_bad_key(svc):
    """🚨 The production call, both ways round. This is the door that was down.

    The refusal half rides along because it is the same header on the same door
    a character apart: if the shim ever widened into 'a bearer we do not know is
    fine', this is where it would show.
    """
    ok, queued = await _enqueued(svc, lambda: svc.handle_openai_chat(
        dict(_CHAT), _Req(headers=_bearer("not-needed"))))
    assert ok.status_code == 200
    assert queued.agent_id == "internal"          # the loopback ACL registration

    bad = await svc.handle_openai_chat(dict(_CHAT), _Req(headers=_bearer("nope")))
    assert bad.status_code == 401
    assert json.loads(bad.body)["error"]["code"] == "invalid_api_key"


@pytest.mark.asyncio
async def test_the_counter_counts_REQUESTS_not_resolutions(svc):
    """🚨 Found live, not in the suite: one ``curl`` reported ``count: 2``.

    The OpenAI doors and ``/rs/v1/chat`` resolve identity TWICE by design — the
    door resolves, and ``Lifecycle.handle_submit`` resolves again so it is safe
    on its own rather than safe by virtue of who calls it. A counter
    incremented inside ``resolve`` therefore counts *resolutions*: 2 per request
    on the hot doors, 1 on ``/rs/v1/plan``, and an operator reading it as
    traffic is off by a factor that varies with the door. Plausible, stable,
    and answering a different question than the one it gates.

    Pinned at the DOOR rather than at the resolver, because at the resolver the
    two calls are indistinguishable and the bug is invisible.
    """
    await _enqueued(svc, lambda: svc.handle_openai_chat(
        dict(_CHAT), _Req(headers=_bearer("not-needed"))))
    tally = svc._state.placeholder_bearers
    assert tally == {"count": 1, "by_address": {"127.0.0.1": 1}}, (
        f"one request was counted {tally['count']} times — the counter is "
        f"measuring identity resolutions, not callers")


@pytest.mark.asyncio
async def test_the_embeddings_door_serves_a_placeholder(svc):
    resp = await svc.handle_openai_embeddings(
        {"model": "embed", "input": "hi"}, _Req(headers=_bearer("not-needed")))
    assert resp.status_code != 401


@pytest.mark.asyncio
async def test_the_enriched_door_serves_a_placeholder(svc):
    body = {"payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8},
            "intent": "chat"}
    resp, queued = await _enqueued(
        svc, lambda: svc.handle_rs_chat(body, _Req(headers=_bearer("not-needed"))))
    assert resp.status_code == 200
    assert queued.agent_id == "internal"


@pytest.mark.asyncio
async def test_the_legacy_submit_door_serves_a_placeholder(svc):
    """Under its flag — the door a migrating fleet is most likely to be on when
    it also has no keys yet, which is the whole population this shim is for."""
    body = {"agent_id": "legacy-caller", "endpoint": "chat", "call_site": "t",
            "payload_type": "chat_completion",
            "payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8}}
    resp, queued = await _enqueued(
        svc, lambda: svc.handle_legacy_submit(body, _Req(headers=_bearer("not-needed"))))
    assert resp.status_code == 200
    # §1.9.2's identity exception applies, exactly as it does with no header:
    # an internal-net caller names itself and the placeholder changed nothing.
    assert queued.agent_id == "legacy-caller"


@pytest.mark.asyncio
async def test_an_admin_route_refuses_a_placeholder_and_serves_the_real_key(svc):
    """🚨 403, not 200. An address never grants admin, and a placeholder yields
    an address identity — so this is the pre-existing rule holding, and the test
    exists because 'identified' reads like 'allowed' at a glance."""
    headers = enrol_admin(svc)
    refused = await svc.handle_admin_flags(
        _Req(headers=_bearer("not-needed"), body={"context_gate_enforce": True}))
    assert refused.status_code == 403
    assert svc._state.flags.get("context_gate_enforce") is False

    allowed = await svc.handle_admin_flags(
        _Req(headers={**headers, "X-Roadstead-Request": "1"},
             body={"context_gate_enforce": True}))
    assert allowed.status_code == 200
