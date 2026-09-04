"""Whose default band applies — one question, four steps, one answer. 🚨

`agents.yaml`'s `default_priority` was parsed, allowlisted, editable through
`PATCH /rs/v1/admin/quotas`, and **reported by the management plane as the
caller's `declared_priority`** — while the request path read only the
credential's band. Two sources for one question, with the operator-facing
surface naming the one that was not in force, and their defaults did not even
agree (`P1_TURN_SUPPORT` in the config dataclass, `P3_INGESTION` in the identity
grammar). An operator setting `default_priority: P4_HYGIENE` got nothing, and the
dashboard told them it had worked.

It was invisible because every door **pre-filled** `priority` into the submit
body from `principal.priority`, so by the time `handle_submit` looked, the
identity's own default was indistinguishable from a band the caller had asked
for. The fix removes that duplication: a door passes only what the CALLER
declared, and `handle_submit` owns the precedence.

    1. what this REQUEST declared        `priority`, or `interactive`
    2. what the CREDENTIAL declared      `agent_id:P1_TURN_SUPPORT`
    3. what the AGENT's config declares  agents.yaml `default_priority`
    4. the built-in default

🚨 Step 3 is below step 2 because a credential is the stronger statement — but it
has to EXIST, because since delegation one key can act as many `agent_id`s and a
single band on that key cannot say "interactive for `chat-agent`, background for
`forum-agent`". Per-agent config can, and it is where the weights that go with those
bands already live.
"""
from __future__ import annotations

import asyncio

import pytest

from roadstead.backend import BackendResponse
from roadstead.config import LLMPriority, ProxyConfig, AgentQuotaConfig
from roadstead.service import ProxyService
from roadstead import scheduler as sched


class _Req:
    def __init__(self, host="127.0.0.1", headers=None):
        class _C:
            pass
        _C.host = host
        self.client = _C()
        self.headers = dict(headers) if headers else {}
        # 🚨 identity.py's CSRF gate (`admin_denial._csrf_denial`) requires
        # `Content-Type: application/json` on every mutating admin request —
        # this double always reports POST (below), even for the GET-only
        # routes some of these tests exercise, so it must carry the header
        # unconditionally rather than only when a test remembers to.
        self.headers.setdefault("Content-Type", "application/json")
        self.method = "POST"
        self.query_params: dict = {}


def _ok(svc):
    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1,
            finish_reason="stop")
    svc._backend.call = call


async def _band(svc, coro_factory):
    """Run one request and return the band it was actually enqueued at."""
    seen: list = []
    orig = sched.QueuedRequest.create.__func__
    sched.QueuedRequest.create = classmethod(
        lambda cls, **kw: seen.append(orig(cls, **kw)) or seen[-1])
    try:
        await asyncio.wait_for(coro_factory(), timeout=10.0)
    finally:
        sched.QueuedRequest.create = classmethod(orig)
    assert seen, "no request was enqueued"
    return seen[-1].priority


@pytest.fixture
async def svc(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADSTEAD_ACL", "127.0.0.1=quiet-batcher")
    s = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    _ok(s)
    async def healthy(ep):
        return True
    s._backend.probe_health = healthy
    await s.startup()
    # Wired after startup, exactly as `build_app` supplies it.
    s._config.agents["quiet-batcher"] = AgentQuotaConfig(
        agent_id="quiet-batcher", default_priority=LLMPriority.P4_HYGIENE)
    try:
        yield s
    finally:
        await s.shutdown()


_CHAT = {"model": "chat", "messages": [{"role": "user", "content": "hi"}],
         "max_tokens": 8}


async def test_the_agents_config_default_reaches_the_openai_door(svc):
    """🚨 The gap. This door reads no `priority` from the caller and the ACL
    entry names no band, so the agent's configured default is the only thing
    left that has an opinion — and for its whole life it was ignored."""
    band = await _band(svc, lambda: svc.handle_openai_chat(dict(_CHAT), _Req()))
    assert band is LLMPriority.P4_HYGIENE, (
        f"agents.yaml declares default_priority=P4_HYGIENE and the request ran "
        f"as {band.name} — the config field is not in force")


async def test_a_credential_that_names_a_band_still_wins(svc, monkeypatch):
    """Step 2 above step 3: a credential is the stronger statement, and an
    operator who scoped a key to a band meant it."""
    svc._state.identity.keys.register(
        secret="kb", agent_id="quiet-batcher",
        priority=LLMPriority.P1_TURN_SUPPORT, priority_declared=True,
        key_id="kb", source="runtime")
    band = await _band(svc, lambda: svc.handle_openai_chat(
        dict(_CHAT), _Req(headers={"Authorization": "Bearer kb"})))
    assert band is LLMPriority.P1_TURN_SUPPORT


async def test_a_credential_that_names_NO_band_defers_to_the_agent(svc):
    """🚨 The distinction the `priority_declared` flag exists for. This key and
    the one above resolve to the same `priority` value by default; only the flag
    separates 'the operator wrote P3' from 'the operator wrote nothing'."""
    svc._state.identity.keys.register(
        secret="kn", agent_id="quiet-batcher", key_id="kn", source="runtime")
    band = await _band(svc, lambda: svc.handle_openai_chat(
        dict(_CHAT), _Req(headers={"Authorization": "Bearer kn"})))
    assert band is LLMPriority.P4_HYGIENE, (
        "a credential that named no band shadowed the agent's configured one — "
        "which is indistinguishable from the field being dead again")


async def test_a_per_call_priority_still_beats_everything(svc):
    """Step 1. On the enriched door, where a caller can declare one."""
    body = {"payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8},
            "intent": "chat", "priority": "P0_REALTIME"}
    band = await _band(svc, lambda: svc.handle_rs_chat(body, _Req()))
    assert band is LLMPriority.P0_REALTIME


async def test_interactive_false_is_a_declaration_and_the_config_does_not_override(svc):
    """`interactive` is the enriched spelling of the same declaration, so it is
    step 1 too — not a hint the agent's config may overrule."""
    body = {"payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8},
            "intent": "chat", "interactive": True}
    band = await _band(svc, lambda: svc.handle_rs_chat(body, _Req()))
    assert band is LLMPriority.P1_TURN_SUPPORT


async def test_no_door_pre_fills_the_band_into_the_submit_body():
    """🚨 The guard on the CAUSE, not the symptom.

    Every door pre-filled `priority` from `principal.priority`, which is what
    made the identity's default indistinguishable from a caller's declaration
    and the agent config unreachable. A door that starts doing it again breaks
    the precedence silently — the tests above would still pass for any caller
    whose credential happens to agree with its config.

    Widened 2026-09-04 (§1.7): the same shape reappeared one level down, not as
    a dict literal but as `LLMPriority.coerce(..., default=<identity band>)` —
    `/rs/v1/plan` resolved only steps 1-2 that way, so an agent with a
    configured `default_priority` and no per-credential band got a plan priced
    at the wrong band and dispatched at the right one. The fix is
    `Lifecycle.resolve_declared_priority`, the one place all four steps are
    decided; a door calling `LLMPriority.coerce` itself with an
    identity-derived default is resolving the precedence AGAIN, badly.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "roadstead"
    offenders: list[str] = []
    for name in ("http_handlers.py", "enriched.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if not (isinstance(key, ast.Constant) and key.value == "priority"):
                        continue
                    src = ast.dump(value)
                    if "principal" in src or "default_priority" in src:
                        offenders.append(f"{name}:{key.lineno}")
            # A door calling `LLMPriority.coerce(...)` directly with a
            # `default=` derived from the identity/principal — the precedence
            # belongs to `resolve_declared_priority` alone; a second caller
            # computing "the identity's band" as a fallback is the same bug in
            # a call argument instead of a dict literal.
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "coerce"
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "LLMPriority"):
                for kw in node.keywords:
                    if kw.arg != "default":
                        continue
                    src = ast.dump(kw.value)
                    if "principal" in src or "default_priority" in src:
                        offenders.append(f"{name}:{node.lineno}")
    assert not offenders, (
        "a door is resolving the priority band's identity fallback itself "
        f"instead of going through Lifecycle.resolve_declared_priority: "
        f"{offenders} — pass only what the CALLER declared and let the shared "
        "resolver apply the credential/agents.yaml/default steps, or a plan "
        "and the call it precedes can disagree about the band again")


# --------------------------------------------------------------------------- #
# The OPERATOR surface must not state a band the caller does not run in
# --------------------------------------------------------------------------- #

async def test_the_admin_view_shows_which_key_overrides_the_agents_band(svc):
    """🚨 `declared_priority` reports the AGENT's configured band — step 3 — and
    a credential that names its own (step 2) beats it. Many keys to one
    `agent_id` is a documented shape, so "the band in force for this caller" has
    no single value, and a scalar claiming otherwise would be this plane
    reporting something untrue about itself.

    So the agent's default stays where it is and every KEY says what it does.
    """
    import json
    from tests.admin_key import ADMIN_HEADERS, enrol_admin

    reg = svc._state.identity.keys
    reg.register(secret="pins", agent_id="quiet-batcher", key_id="pins",
                 priority=LLMPriority.P1_TURN_SUPPORT, priority_declared=True,
                 source="runtime")
    reg.register(secret="defers", agent_id="quiet-batcher", key_id="defers",
                 source="runtime")
    enrol_admin(svc)

    resp = await svc.handle_admin_callers(_Req(headers=dict(ADMIN_HEADERS)))
    row = next(r for r in json.loads(resp.body)["callers"]
               if r["agent_id"] == "quiet-batcher")
    by_id = {k["key_id"]: k for k in row["identities"]["keys"]}

    # The one that pins a band is marked, with the band it pins.
    assert by_id["pins"]["overrides_agent_default"] is True
    assert by_id["pins"]["priority"] == "P1_TURN_SUPPORT"
    # 🚨 The one that names none reports `priority: None`, NOT the grammar's
    # default. "Declared P3_INGESTION" and "declared nothing" resolve to the
    # same value and mean opposite things; reporting the value for both is how
    # the operator surface would go back to being wrong.
    assert by_id["defers"]["overrides_agent_default"] is False
    assert by_id["defers"]["priority"] is None

    # And the agent's own configured band is still reported, unchanged.
    assert row["spend"]["declared_priority"] == "P4_HYGIENE"


async def test_a_delegated_call_records_the_credential_that_asserted_it(svc):
    """🚨 `agent_id` and the credential were the same thing by construction
    until delegation landed. With that guarantee gone, "which key ran up chat-agent's
    bill?" needs an answer, and the admin audit trail records admin ACTIONS
    rather than dispatch.

    NULL means "the credential IS the agent_id" — the pre-delegation invariant,
    still true of every undelegated row — so a value is always meaningful.
    """
    svc._state.identity.keys.register(
        secret="dk", agent_id="originfleet", key_id="dk",
        may_assert=["quiet-batcher"], source="runtime")
    body = {"payload": {"messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8},
            "intent": "chat", "agent_id": "quiet-batcher"}
    await _band(svc, lambda: svc.handle_rs_chat(
        body, _Req(headers={"Authorization": "Bearer dk"})))
    svc._queue_db.flush()
    row = svc._queue_db._conn.execute(
        "SELECT agent_id, key_id FROM proxy_completions ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert row == ("quiet-batcher", "dk"), (
        f"the delegated row does not name the credential that asserted it: {row}")


async def test_an_undelegated_call_records_no_key_id(svc):
    """The column stays NULL on the ordinary path, so a value in it always
    means a delegation happened rather than being populated on everything and
    read on nothing."""
    svc._state.identity.keys.register(
        secret="plain", agent_id="quiet-batcher", key_id="plain", source="runtime")
    await _band(svc, lambda: svc.handle_openai_chat(
        dict(_CHAT), _Req(headers={"Authorization": "Bearer plain"})))
    svc._queue_db.flush()
    row = svc._queue_db._conn.execute(
        "SELECT agent_id, key_id FROM proxy_completions ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    assert row == ("quiet-batcher", None), row
