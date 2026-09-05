"""``POST /v1/submit`` — the flag-gated legacy door (``roadstead/legacy.py``).

The door exists to reproduce a contract, so these tests are transcriptions of
that contract rather than assertions about how the code reaches it. Every status
code, ``code`` value and load-bearing substring below is spelled out as a
literal: importing it from anywhere — the server's own constants, the fleet that
consumes it — would be the two-values-from-one-source tautology
``tests/wire_contract.py`` describes, and the whole point of a compatibility
door is that it agrees with something it cannot read.

Three groups, and the middle one is the one with a deliberate deviation in it:

  * **Registration** — OFF means the route does not exist.
  * **Identity** — §1.9.2's exception, and its two edges (a registered address,
    an authenticated key).
  * **Shapes** — the six-key success envelope, each admission gate's status +
    code + marker, and the priority coercion table.

Streaming, embeddings, rerank and a real 404 are in
``tests/e2e/test_legacy_submit.py``, over a socket, because framing and route
resolution are exactly the things an in-process call cannot check.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.backend import BackendError, BackendResponse, BackendTimeout
from roadstead.config import LLMPriority, ProxyConfig
from roadstead.legacy import ROUTE
from roadstead.routes import make_routes
from roadstead.service import ProxyService
from tests.wire_contract import CONTEXT_OVERFLOW_MARKER, carries_deferral_marker


class _Req:
    """A loopback caller presenting nothing — the internal-net case."""

    def __init__(self, host: str = "127.0.0.1", headers: dict | None = None) -> None:
        self.client = type("_C", (), {"host": host})()
        self.headers = headers or {}


def _body(**kw) -> dict:
    body = {
        "agent_id": "a-caller",
        "endpoint": "tier3",
        "priority": "P3_INGESTION",
        "call_site": "t.legacy",
        "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}],
                    "max_tokens": 8},
        "timeout_s": 5.0,
    }
    payload_extra = kw.pop("payload", None)
    if payload_extra is not None:
        body["payload"] = payload_extra
    body.update(kw)
    return body


def _ok_backend(svc: ProxyService, content: str = "y") -> None:
    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": content},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call


async def _submit(svc: ProxyService, body: dict, req: _Req):
    """Drive the legacy door, capturing the ``QueuedRequest`` it built.

    The capture is how the identity and priority assertions read what was
    ADMITTED rather than what came back — the envelope publishes neither, by
    design (docs/api.md §1.6), so a response-only assertion could not tell a
    correctly-attributed call from a wrongly-attributed one.
    """
    from roadstead import scheduler as _sched
    created: list = []
    orig = _sched.QueuedRequest.create.__func__
    _sched.QueuedRequest.create = classmethod(
        lambda cls, **kw: created.append(orig(cls, **kw)) or created[-1])
    try:
        resp = await asyncio.wait_for(
            svc.handle_legacy_submit(body, req), timeout=10.0)
    finally:
        _sched.QueuedRequest.create = classmethod(orig)
    return resp, created


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

def test_the_route_does_not_exist_unless_the_flag_is_set(monkeypatch):
    """🚨 OFF means ABSENT, not refusing.

    A door that 403s still tells a scanner it is there and still has to be
    reasoned about; a route that was never registered answers with the same 404
    every unknown path has answered with since Workstream C removed this one.
    """
    monkeypatch.delenv("ROADSTEAD_LEGACY_SUBMIT", raising=False)
    assert ROUTE not in {r.path for r in make_routes(None)}

    monkeypatch.setenv("ROADSTEAD_LEGACY_SUBMIT", "1")
    submit = [r for r in make_routes(None) if r.path == ROUTE]
    assert len(submit) == 1 and submit[0].methods == {"POST"}


@pytest.mark.parametrize("value,registered", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("", False), ("0", False), ("false", False), ("maybe", False),
])
def test_the_flag_reads_the_same_vocabulary_as_the_admin_ui_flag(
        monkeypatch, value, registered):
    """One spelling of "on" across the operator-facing flags. Two would be a
    flag that works in the compose file and not in the shell."""
    monkeypatch.setenv("ROADSTEAD_LEGACY_SUBMIT", value)
    assert (ROUTE in {r.path for r in make_routes(None)}) is registered


# --------------------------------------------------------------------------- #
# Identity — docs/api.md §1.9.2
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_an_internal_caller_is_the_agent_id_in_its_own_body():
    """The exception, stated plainly. This is the ONE thing about the legacy
    envelope that cannot be reproduced without a deviation from §1.5, and it is
    what every fleet caller's DRR share depends on."""
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        resp, created = await _submit(svc, _body(agent_id="chat-agent"), _Req())
        assert resp.status_code == 200
        assert created[-1].agent_id == "chat-agent"

        # …and with nothing declared, the registration fills it in, exactly as
        # it does on every other door.
        body = _body()
        del body["agent_id"]
        resp, created = await _submit(svc, body, _Req())
        assert created[-1].agent_id == "internal"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_registered_address_keeps_its_registration():
    """🚨 The exception is a NARROWING of the shared rule, not a widening.

    ``handle_submit``'s own rule lets any address-derived caller name itself —
    an address identifies a host, and several callers share one. The legacy door
    is stricter: an operator who wrote an ACL entry has said who that host is,
    and §1.9.2 publishes that as the contract. A door reproducing an old shape
    must not hand out more than the old shape did.
    """
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    svc._acl.register("198.51.100.7", "lan-worker", LLMPriority.P3_INGESTION)
    await svc.startup()
    try:
        resp, created = await _submit(
            svc, _body(agent_id="chat-agent"), _Req(host="198.51.100.7"))
        assert resp.status_code == 200
        assert created[-1].agent_id == "lan-worker"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_forwarded_address_never_reaches_the_exception(monkeypatch):
    """"Already on the box" is what the internal nets stand for, and a front
    proxy is precisely the thing that makes it untrue — the same reasoning
    ``acl.is_admin(trust_builtin_nets=False)`` applies to the admin grant."""
    monkeypatch.setenv("ROADSTEAD_TRUSTED_PROXIES", "127.0.0.0/8")
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    svc._acl.register("203.0.113.0/24", "edge-tenant", LLMPriority.P3_INGESTION)
    await svc.startup()
    try:
        resp, created = await _submit(
            svc, _body(agent_id="chat-agent"),
            _Req(headers={"x-forwarded-for": "203.0.113.9"}))
        assert resp.status_code == 200
        assert created[-1].agent_id == "edge-tenant"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_key_without_may_assert_ignores_the_body(monkeypatch):
    """A credential is the stronger statement, on this door as on every other.
    The flag re-opens a shape; it does not re-open the self-asserted
    ``agent_id`` this project closed on 2026-09-01."""
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "kk=billed-here:P4_HYGIENE")
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        resp, created = await _submit(
            svc, _body(agent_id="somebody-else"),
            _Req(headers={"authorization": "Bearer kk"}))
        assert resp.status_code == 200
        assert created[-1].agent_id == "billed-here"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_key_that_may_assert_the_name_is_honoured():
    """The other half: delegation still works here, through the one decision
    (``IdentityResolver.delegate``) that owns it. `may_assert` is a keys-FILE
    grant, so it is registered directly rather than through the env grammar."""
    svc = ProxyService(ProxyConfig())
    svc._identity.keys.register(secret="kk", agent_id="fleet", may_assert=["chat-agent"])
    _ok_backend(svc)
    await svc.startup()
    try:
        resp, created = await _submit(
            svc, _body(agent_id="chat-agent"),
            _Req(headers={"authorization": "Bearer kk"}))
        assert resp.status_code == 200
        assert created[-1].agent_id == "chat-agent"

        resp, _ = await _submit(
            svc, _body(agent_id="not-granted"),
            _Req(headers={"authorization": "Bearer kk"}))
        assert resp.status_code == 403
        assert json.loads(resp.body)["code"] == "access_denied"
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# The success envelope — docs/api.md §1.9.3
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_the_ok_envelope_is_exactly_six_keys():
    """🚨 The key SET, not a subset. A caller of this door validates it, which
    makes an additive field here a breaking change — the opposite of the rule on
    an error envelope, and the reason the two are serialized separately."""
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc, content="hello")
    await svc.startup()
    try:
        resp, _ = await _submit(svc, _body(), _Req())
        body = json.loads(resp.body)
        assert resp.status_code == 200
        assert set(body) == {"status", "request_id", "queue_wait_ms",
                             "backend_latency_ms", "estimated_cost_ss",
                             "response"}
        assert body["status"] == "ok"
        assert body["request_id"].startswith("req_")
        assert body["response"]["choices"][0]["message"]["content"] == "hello"
        # None of the enriched wire's blocks may leak onto this one.
        assert "attribution" not in body and "timing" not in body
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_cache_hit_adds_cache_hit_and_zeroes_the_timings():
    svc = ProxyService(ProxyConfig())
    svc._cache.put("k", {"choices": [{"message": {"content": "cached"}}]})
    svc._cache.cache_key = lambda endpoint, payload: "k"
    resp, _ = await _submit(svc, _body(), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 200
    assert body["cache_hit"] is True
    assert body["status"] == "ok"
    assert body["queue_wait_ms"] == 0 and body["backend_latency_ms"] == 0
    assert body["response"]["choices"][0]["message"]["content"] == "cached"


# --------------------------------------------------------------------------- #
# The error envelopes — docs/api.md §1.9.4, §2.1, §2.2
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_draining_is_503_draining_and_deferrable():
    svc = ProxyService(ProxyConfig())
    svc._draining.set()
    resp = await svc.handle_legacy_submit(_body(), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["status"] == "error" and body["code"] == "draining"
    assert body["error"] == "proxy draining for shutdown — backpressure"
    assert carries_deferral_marker(body["error"])


@pytest.mark.asyncio
async def test_unknown_endpoint_is_404_and_deliberately_not_deferrable():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"unknown_endpoint_enforce": True})
    resp = await svc.handle_legacy_submit(_body(endpoint="typo-role"), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 404
    assert body["status"] == "error" and body["code"] == "unknown_endpoint"
    assert body["error"].startswith("unknown endpoint 'typo-role' — no such model/role")
    # A typo is deterministic: a defer-loop on one never converges.
    assert not carries_deferral_marker(body["error"])


@pytest.mark.asyncio
async def test_malformed_messages_is_400_invalid_messages():
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_legacy_submit(
        _body(payload={"messages": ["hi", "there"]}), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 400
    assert body["code"] == "invalid_messages"
    assert body["error"] == (
        "invalid request: 'messages' must be a list of {role, content} objects")
    assert body["request_id"].startswith("req_")


@pytest.mark.asyncio
async def test_invalid_grammar_is_422_invalid_grammar():
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_legacy_submit(
        _body(payload={"messages": [{"role": "user", "content": "x"}],
                       "grammar": "not a grammar at all"}), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 422
    assert body["status"] == "error" and body["code"] == "invalid_grammar"
    assert not carries_deferral_marker(body.get("detail", ""), body.get("error", ""))


@pytest.mark.asyncio
async def test_context_overflow_is_422_and_carries_the_verbatim_marker():
    """The marker is load-bearing at both ends — a client-side chunker matches
    it to decide whether to re-chunk. Do not reword it (§2.2)."""
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"context_gate_enforce": True})
    resp = await svc.handle_legacy_submit(
        _body(endpoint="chat",
              payload={"messages": [{"role": "user", "content": "x" * 1_200_000}],
                       "max_tokens": 100}), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 422
    assert body["code"] == "context_overflow"
    assert CONTEXT_OVERFLOW_MARKER in body["error"]
    assert body["error"].endswith(
        "— chunk the input or route to a larger-context endpoint")


@pytest.mark.asyncio
async def test_circuit_open_is_503_with_retry_after():
    svc = ProxyService(ProxyConfig())
    svc._endpoint_health["tier3"]["healthy"] = False
    resp = await svc.handle_legacy_submit(
        _body(priority="P1_TURN_SUPPORT"), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "circuit_open"
    # 🚨 `startswith`, because a failover REFUSAL appends to this message and
    # travels in its own `degraded_refusal` field. The published prefix — the
    # part a caller's classifier matches — must not move; what follows it is
    # triage for an operator and may grow.
    assert body["error"].startswith("backend tier3 unavailable (circuit open)")
    assert resp.headers["retry-after"]
    assert carries_deferral_marker(body["error"])


@pytest.mark.asyncio
async def test_a_paused_endpoint_reads_as_draining():
    svc = ProxyService(ProxyConfig())
    svc._endpoint_health["tier3"]["healthy"] = True
    svc._paused_endpoints.add("tier3")
    resp = await svc.handle_legacy_submit(
        _body(priority="P1_TURN_SUPPORT"), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "draining"
    assert body["error"].startswith(
        "backend tier3 paused for maintenance (drain) — backpressure")
    assert resp.headers["retry-after"]


@pytest.mark.asyncio
async def test_load_shed_is_429_backpressure_with_retry_after():
    svc = ProxyService(ProxyConfig())
    svc._shed_depth = 0  # any queued depth sheds
    resp = await svc.handle_legacy_submit(_body(), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 429
    assert body["code"] == "backpressure"
    assert body["error"] == "backpressure: tier3 background queue saturated"
    assert resp.headers["retry-after"]
    assert carries_deferral_marker(body["error"])


@pytest.mark.asyncio
async def test_a_backend_failure_is_502_backend_error():
    svc = ProxyService(ProxyConfig())

    async def failing_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        raise BackendError(400, "backend rejected the payload")

    svc._backend.call = failing_call
    await svc.startup()
    try:
        resp, _ = await _submit(svc, _body(), _Req())
        body = json.loads(resp.body)
        assert resp.status_code == 502
        assert body["status"] == "error" and body["code"] == "backend_error"
        assert "backend rejected the payload" in body["error"]
        # Additive on an ERROR envelope, where no caller pins the key set —
        # §2.2's permanent-vs-transient fact, which the old door could not say.
        assert body["backend_status"] == 400
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_deadline_that_fires_is_504_proxy_timeout():
    """🚨 ``error`` is the bare ``"timeout"`` the old door published, and
    ``status`` is the one field added to it — a caller reading
    ``.get("status") != "ok"`` gets the same answer either way, so supplying it
    can only make more of them agree with the rest of the taxonomy."""
    svc = ProxyService(ProxyConfig())

    async def never_returns(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        await asyncio.sleep(1.2)
        raise BackendTimeout("late")

    svc._backend.call = never_returns
    await svc.startup()
    try:
        resp, _ = await _submit(svc, _body(timeout_s=0.3), _Req())
        body = json.loads(resp.body)
        assert resp.status_code == 504
        assert body == {"status": "error", "request_id": body["request_id"],
                        "error": "timeout", "code": "proxy_timeout"}
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_non_object_body_is_a_400_and_not_a_500():
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_legacy_submit(["not", "an", "envelope"], _Req())
    assert resp.status_code == 400
    assert json.loads(resp.body)["code"] == "invalid_request_error"


# --------------------------------------------------------------------------- #
# The envelope's own fields
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_the_priority_coercion_table_and_that_none_of_it_500s():
    """docs/api.md §1.9.2's table, one service, one dispatch each.

    🚨 The malformed rows are the point: a caller's ``priority`` reaching the
    proxy as a 500 turns a cosmetic mistake into a failed LLM call, and the old
    door soft-defaulted every one of them.
    """
    table = [
        ("P0_REALTIME", LLMPriority.P0_REALTIME),
        ("p2_post_turn", LLMPriority.P2_POST_TURN),
        ("interactive", LLMPriority.P1_TURN_SUPPORT),
        ("foreground", LLMPriority.P2_POST_TURN),
        ("background", LLMPriority.P3_INGESTION),
        ("P3_BACKGROUND", LLMPriority.P3_INGESTION),
        (4, LLMPriority.P4_HYGIENE),
        (99, LLMPriority.P4_HYGIENE),          # clamped, not rejected
        (-7, LLMPriority.P0_REALTIME),         # clamped the other way
        (None, LLMPriority.P1_TURN_SUPPORT),   # declared nothing
        ("nonsense", LLMPriority.P1_TURN_SUPPORT),
        (True, LLMPriority.P1_TURN_SUPPORT),   # a JSON bool is not a priority
    ]
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        for declared, expected in table:
            resp, created = await _submit(
                svc, _body(priority=declared), _Req())
            assert resp.status_code == 200, (declared, resp.status_code)
            assert created[-1].priority is expected, declared
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_a_caller_deadline_wins_and_a_malformed_one_defaults():
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        _, created = await _submit(svc, _body(timeout_s=42.0), _Req())
        assert created[-1].timeout_s == 42.0

        body = _body()
        del body["timeout_s"]
        _, created = await _submit(svc, body, _Req())
        computed = created[-1].timeout_s
        assert computed > 0 and computed != 42.0

        _, created = await _submit(svc, _body(timeout_s="soon"), _Req())
        assert created[-1].timeout_s == computed
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_the_door_does_not_forward_fields_it_never_published():
    """🚨 The envelope is copied field by field, not forwarded whole.

    ``Lifecycle.handle_submit`` reads four keys this door never published, and
    forwarding the caller's dict would make the compatibility door quietly WIDER
    than the contract it exists to reproduce.
    """
    svc = ProxyService(ProxyConfig())
    _ok_backend(svc)
    await svc.startup()
    try:
        _, created = await _submit(
            svc, _body(allow_spill=True, allow_degrade=True,
                       requested="something-else"), _Req())
        req = created[-1]
        assert req.allow_spill is None and req.allow_degrade is None
        assert req.requested == "tier3"   # the endpoint, not the caller's word
    finally:
        await svc.shutdown()


# --------------------------------------------------------------------------- #
# Deprecation
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_each_caller_is_named_once_a_day_and_counted_every_time(caplog):
    """The migration inventory, taken two ways.

    Per CALLER because the inventory is a list of callers and a fleet-wide line
    names none of them; per DAY because a proxy that restarts nightly would
    otherwise report the same list every morning, and one that runs for a month
    would report it once and then look migrated.
    """
    svc = ProxyService(ProxyConfig())
    svc._draining.set()  # refuse fast; the notice is taken before admission
    with caplog.at_level("WARNING", logger="roadstead.legacy"):
        for _ in range(3):
            await svc.handle_legacy_submit(_body(agent_id="chat-agent"), _Req())
        await svc.handle_legacy_submit(_body(agent_id="forum-agent"), _Req())

    lines = [r.getMessage() for r in caplog.records
             if "legacy /v1/submit used by" in r.getMessage()]
    assert len(lines) == 2, lines
    assert lines[0] == ("legacy /v1/submit used by agent_id=chat-agent "
                        "(call_site=t.legacy) — migrate to /rs/v1/chat "
                        "(docs/api.md §1.9)")
    assert "agent_id=forum-agent" in lines[1]

    # The counter is per CALL, not per notice — one notice a day would make the
    # inventory look like two callers making one request each.
    assert svc._state.legacy_submits == {
        "count": 4, "callers": {"chat-agent": 3, "forum-agent": 1}}
    status = json.loads((await svc.handle_status(_Req())).body)
    assert status["reliability"]["legacy_submits"]["callers"]["chat-agent"] == 3


@pytest.mark.asyncio
async def test_the_caller_map_is_bounded_because_the_name_is_caller_asserted():
    """🚨 A cap is not a bound, and this is the case where that matters: the
    ``agent_id`` is what the caller says it is, so an unbounded map keyed on it
    is a caller-controlled allocation on a door with no credential."""
    from roadstead.legacy import _MAX_TRACKED_CALLERS

    svc = ProxyService(ProxyConfig())
    svc._draining.set()
    for i in range(_MAX_TRACKED_CALLERS + 50):
        await svc.handle_legacy_submit(_body(agent_id=f"caller-{i}"), _Req())

    tally = svc._state.legacy_submits
    assert tally["count"] == _MAX_TRACKED_CALLERS + 50
    assert len(tally["callers"]) == _MAX_TRACKED_CALLERS
    assert len(svc._legacy._notified) <= _MAX_TRACKED_CALLERS
