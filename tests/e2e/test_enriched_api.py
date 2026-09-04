"""``/rs/v1/*`` end to end — the enriched north face (roadmap Workstream C).

Driven over the real ASGI front door against a real fake backend socket, so what
is asserted is what a caller would receive on the wire, not what the handler
intended to send.

Three things here are doctrine rather than behaviour, and each has its own test
because each has a plausible-looking change that would break it silently:

  * **Resolution is not substitution.** An intent that lands on an endpoint the
    caller never named is not a substitution and must not be reported as one, or
    ``substituted: true`` fires on every intent-routed call and means nothing.
  * **No priority, no band, no queue position** in the response — ``docs/api.md``
    §1.6, a caller cannot observe its own spend demotion.
  * **The per-request substitution flags NARROW and never widen.**
"""
from __future__ import annotations

import json


from roadstead.enriched import ENRICHMENT_HEADERS
from roadstead.intent import BUILTIN_PROFILES


def _body(**kw) -> dict:
    body = {
        "payload": {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
    }
    payload_extra = kw.pop("payload", None)
    if payload_extra:
        body["payload"].update(payload_extra)
    body.update(kw)
    return body


# ---------------------------------------------------------------------------
# GET /rs/v1/models
# ---------------------------------------------------------------------------

async def test_models_publishes_capability_live_state_and_price(proxy):
    resp = await proxy.client.get("/rs/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    rows = {r["endpoint"]: r for r in body["models"]}

    # Every endpoint in the catalog, routed or not — a `planned` remote endpoint
    # is part of what the fleet IS even though nothing can be dispatched to it.
    assert {"tier1", "tier2", "tier3", "embed", "rerank"} <= set(rows)
    assert rows["spill-chat"]["routed"] is False
    # 🚨 An unrouted endpoint is never reported healthy: nothing polls it, and a
    # green light on a name nobody can dispatch to is worse than no light.
    assert rows["spill-chat"]["healthy"] is False

    t3 = rows["tier3"]
    assert "vision" in t3["capabilities"] and "reasoning" in t3["capabilities"]
    assert t3["kind"] == "chat" and t3["context"] > 0
    assert t3["max_slots"] > 0 and t3["free_slots"] <= t3["max_slots"]
    # 🚨 The two kinds of money are distinguished by a bool on the price, not by
    # which field the number is in — a consumer that adds them has the bug
    # spend.py exists to prevent.
    assert t3["price"]["real"] is False
    # Never measured reads as null, not 0.0: a client plotting 0 ms would draw
    # an infinitely fast model out of an absence of evidence.
    assert t3["typical_ms"] is None
    # The names a caller may pin with, so it can use the one its config holds.
    assert "reasoner" in t3["aliases"]


async def test_models_publishes_the_intent_vocabulary(proxy):
    """A deployment may extend the profile table, so a caller cannot guess it —
    and an "unknown intent" error is a poor place to learn one."""
    body = (await proxy.client.get("/rs/v1/models")).json()
    by_name = {i["name"]: i for i in body["intents"]}
    # Layered, not replaced: the example catalog adds `bulk` and overrides
    # nothing, so every built-in is still on the wire beside it. A file that
    # defined one profile silently emptying the other nine is the failure this
    # pins — it would break every caller coding against the published table.
    assert set(by_name) > set(BUILTIN_PROFILES)
    assert all(i["summary"] for i in body["intents"])
    # `source` is the disclosure that makes an override visible to the caller.
    assert by_name["reasoning"]["source"] == "builtin"
    assert by_name["bulk"]["source"] == "models.yaml"
    assert by_name["bulk"]["prefer"] == "capacity"


# ---------------------------------------------------------------------------
# POST /rs/v1/plan
# ---------------------------------------------------------------------------

async def test_plan_answers_where_how_long_and_what_it_costs(proxy):
    resp = await proxy.client.post("/rs/v1/plan", json={
        "intent": "reasoning", "est_in": 4000, "est_out": 500})
    assert resp.status_code == 200
    plan = resp.json()
    assert plan["endpoint"] == "tier3"          # the only reasoning endpoint
    assert plan["requested"] == "reasoning"
    assert plan["timing"]["recommended_deadline_s"] > 0
    assert plan["cost"]["estimated_usd"] >= 0
    assert plan["model"]["endpoint"] == "tier3"
    # 🚨 Two substitution questions, never collapsed into one field: "may a
    # worse model answer if this is DOWN" and "may we pay somebody else if this
    # is FULL" have different triggers and different failure modes.
    assert set(plan["substitution"]) == {"degrade", "spill"}


async def test_plan_and_chat_agree_on_where_the_request_would_go(proxy):
    """🚨 The planner runs the SAME resolver the router runs, over the same
    facts. A planner that approximated the router would be a second answer to
    the question the router is about to answer differently, and a caller would
    have no way to tell which one lied."""
    plan = (await proxy.client.post(
        "/rs/v1/plan", json={"intent": "vision"})).json()
    result = (await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="vision"))).json()
    assert plan["endpoint"] == result["attribution"]["endpoint"]


async def test_plan_does_not_dispatch(proxy):
    before = len(proxy.controller.requests)
    await proxy.client.post("/rs/v1/plan", json={"intent": "chat"})
    assert len(proxy.controller.requests) == before, (
        "plan reached the backend — it exists precisely so a caller can ask "
        "before committing")
    assert proxy.total_in_flight() == 0


# ---------------------------------------------------------------------------
# POST /rs/v1/chat
# ---------------------------------------------------------------------------

async def test_an_intent_routes_and_discloses_without_claiming_substitution(proxy):
    """🚨 Resolution is not substitution.

    The caller said "reasoning" and got tier3, an endpoint it never named. That
    is Roadstead doing the job it was asked to do, and reporting it as a
    substitution would make the flag fire on every intent-routed call.
    """
    resp = await proxy.client.post("/rs/v1/chat", json=_body(intent="reasoning"))
    assert resp.status_code == 200
    attrib = resp.json()["attribution"]
    assert attrib["requested"] == "reasoning"
    assert attrib["resolved"] == "tier3" == attrib["endpoint"]
    assert attrib["substituted"] is False
    assert attrib["substitution"] is None
    assert attrib["provider"] and attrib["engine"]


async def test_a_pin_is_honoured_and_echoed_in_the_callers_own_spelling(proxy):
    resp = await proxy.client.post("/rs/v1/chat", json=_body(model="reasoner"))
    assert resp.status_code == 200
    attrib = resp.json()["attribution"]
    assert attrib["requested"] == "reasoner"     # the alias, as sent
    assert attrib["endpoint"] == "tier3"         # the class that served


async def test_a_pin_that_cannot_meet_a_requirement_is_refused_not_served(proxy):
    """🚨 The refusal, not a fallback. tier2 has no vision; tier3 does. Serving
    this from tier3 would be a silent substitution the caller could not detect —
    the same shape as the finish_reason repair that became a silencer."""
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(model="tier2", requires=["vision"]))
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "unknown_endpoint"
    assert "vision" in body["error"]
    assert [c["endpoint"] for c in body["considered"]] == ["tier2"]
    assert proxy.total_in_flight() == 0


async def test_an_unsatisfiable_intent_names_the_near_misses_only(proxy):
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", min_context=99_000_000))
    assert resp.status_code == 404
    considered = {c["endpoint"] for c in resp.json()["considered"]}
    # Chat endpoints that were the right shape and missed on context — not
    # every embedder in the fleet, which fails every chat request forever.
    assert "embed" not in considered and "rerank" not in considered
    assert "tier2" in considered


async def test_the_five_blocks_are_all_present_and_timing_is_real(proxy):
    resp = await proxy.client.post("/rs/v1/chat", json=_body(intent="chat"))
    body = resp.json()
    # 🚨 Equality, not a subset: a block added here is a deliberate decision
    # about what the enriched envelope publishes. `identity` joined on
    # 2026-09-02 (§1.7.3) — it is the disclosure that stops an IGNORED
    # delegation from being invisible.
    assert set(body) == {"status", "request_id", "response",
                         "attribution", "identity", "timing", "usage"}
    # Nothing declared, so the block is the resolved identity alone — no
    # `honoured`, which would be True on every ordinary call and mean nothing.
    assert body["identity"] == {"agent_id": "internal"}
    timing = body["timing"]
    assert timing["deadline_source"] == "computed"   # no deadline_s supplied
    assert timing["total_ms"] >= timing["backend_latency_ms"] >= 0
    assert timing["ttft_ms"] is None                 # nothing to time, not zero
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["slot_seconds"] >= 0
    # The backend's body stays NESTED — a caller must never have to tell our
    # fields from the model's.
    assert body["response"]["object"] == "chat.completion"


async def test_a_caller_supplied_deadline_is_labelled_as_the_callers(proxy):
    """The label is a behaviour difference, not decoration: a deadline we chose
    is a soft budget the streaming path may extend, one the caller chose is a
    hard wall."""
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", deadline_s=45.0))
    timing = resp.json()["timing"]
    assert timing["deadline_source"] == "caller"
    assert timing["deadline_s"] == 45.0


async def test_the_envelope_never_leaks_the_callers_band(proxy):
    """🚨 docs/api.md §1.6 — a caller cannot observe its own spend demotion.

    Publishing the effective priority would turn a threshold that "never
    rejects" into one every client could detect and branch on, which is a
    rejection with extra steps.
    """
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", priority="P4_HYGIENE"))
    flat = json.dumps(resp.json())
    for leaked in ("priority", "band", "queue_position", "demoted",
                   "effective_priority", "P4_HYGIENE"):
        assert f'"{leaked}"' not in flat and f'"{leaked}"' not in flat, leaked
    assert "P4_HYGIENE" not in flat


async def test_interactive_is_a_first_class_declaration(proxy):
    """`interactive` is the enriched spelling of what `priority` has always
    encoded and never named. Both are accepted; `priority` wins because it says
    strictly more."""
    for body in (_body(intent="chat", interactive=True),
                 _body(intent="chat", interactive=False),
                 _body(intent="chat", interactive=True, priority="P3_INGESTION")):
        resp = await proxy.client.post("/rs/v1/chat", json=body)
        assert resp.status_code == 200, resp.text[:200]
    assert proxy.total_in_flight() == 0


async def test_a_malformed_declaration_is_a_deterministic_400(proxy):
    for body in (_body(),                                   # nothing declared
                 _body(intent="reasonning"),
                 _body(intent="chat", prefer="quality"),
                 _body(intent="chat", substitution="no")):
        resp = await proxy.client.post("/rs/v1/chat", json=body)
        assert resp.status_code == 400, (body, resp.text[:200])
        assert resp.json()["code"] == "invalid_request_error"
    # And a payload that is not an object at all.
    resp = await proxy.client.post(
        "/rs/v1/chat", json={"intent": "chat", "payload": "hi"})
    assert resp.status_code == 400
    assert proxy.total_in_flight() == 0


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

async def _frames(proxy, body) -> list[dict]:
    out: list[dict] = []
    async with proxy.client.stream("POST", "/rs/v1/chat", json=body) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            line = line.strip()
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: "):]))
    return out


async def test_the_stream_names_what_will_serve_before_the_first_token(proxy):
    """The one thing a streaming caller cannot learn from a header, because the
    headers are on the wire before admission's answer could change."""
    frames = await _frames(proxy, _body(intent="reasoning",
                                        payload={"stream": True}))
    assert frames[0]["type"] == "accepted"
    assert frames[0]["attribution"]["endpoint"] == "tier3"
    assert frames[0]["timing"]["deadline_s"] > 0

    done = frames[-1]
    assert done["type"] == "done"
    # 🚨 Attribution again on `done`, and THAT one is authoritative: failover
    # and spill both move a request after `accepted` is already sent.
    assert done["attribution"]["endpoint"] == "tier3"
    assert done["usage"]["output_tokens"] >= 0
    assert done["timing"]["ttft_ms"] is not None
    assert "[DONE]" not in json.dumps(frames)


# ---------------------------------------------------------------------------
# The OpenAI door keeps its shape, and gets headers instead
# ---------------------------------------------------------------------------

async def test_the_openai_body_gains_nothing_and_the_headers_carry_it(proxy):
    """🚨 A client validating against OpenAI's schema must not break because it
    pointed at Roadstead. "We only added fields" is not a defence: strict
    validators reject unknown keys."""
    resp = await proxy.client.post("/v1/chat/completions", json={
        "model": "tier3", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) <= {"id", "object", "created", "model", "choices",
                         "usage", "system_fingerprint", "service_tier"}
    assert not any(k.lower().startswith("roadstead") for k in body)

    assert resp.headers[ENRICHMENT_HEADERS["endpoint"]] == "tier3"
    assert resp.headers[ENRICHMENT_HEADERS["request_id"]].startswith("req_")
    assert float(resp.headers[ENRICHMENT_HEADERS["deadline_s"]]) > 0
    assert resp.headers[ENRICHMENT_HEADERS["deadline_source"]] == "computed"


async def test_the_openai_stream_carries_headers_too(proxy):
    async with proxy.client.stream("POST", "/v1/chat/completions", json={
            "model": "tier2", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8, "stream": True}) as resp:
        assert resp.headers[ENRICHMENT_HEADERS["endpoint"]] == "tier2"
        async for _ in resp.aiter_lines():
            pass
    assert proxy.total_in_flight() == 0


# ---------------------------------------------------------------------------
# The removed door
# ---------------------------------------------------------------------------

async def test_v1_submit_is_gone(proxy):
    """A recorded breaking change (CHANGELOG.md). Pinned so it cannot come back
    by accident — a route that quietly reappears is a second contract nobody
    decided to keep."""
    resp = await proxy.client.post("/v1/submit", json={"payload": {}})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delegated identity — a key acting as another caller (2026-09-02)
# ---------------------------------------------------------------------------

def _delegating_key(proxy, *, may_assert):
    """Enrol one key on the live registry and return its bearer header."""
    proxy.svc._state.identity.keys.register(
        secret="deleg-secret", agent_id="originfleet", key_id="c2",
        may_assert=may_assert, source="runtime")
    return {"Authorization": "Bearer deleg-secret"}


async def test_a_granted_agent_id_becomes_the_fair_share_key(proxy):
    """🚨 The whole point, checked on the WIRE and against the durable record
    rather than against the handler's intent.

    The `agent_id` is what DRR budgets, quota, spend and the completion row are
    all keyed on. A delegation that reached the response but not the record
    would look correct from outside and bill the wrong caller forever.
    """
    headers = _delegating_key(proxy, may_assert=["chat-agent", "knowledge_store"])
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", agent_id="chat-agent"), headers=headers)
    assert resp.status_code == 200
    body = resp.json()

    proxy.svc._queue_db.flush()
    row = proxy.svc._queue_db._conn.execute(
        "SELECT agent_id FROM proxy_completions WHERE request_id=?",
        (body["request_id"],)).fetchone()
    assert row is not None, "no completion row for the request just served"
    assert row[0] == "chat-agent", (
        f"the delegated agent_id did not reach the durable record: {row[0]!r} — "
        f"the call was billed to the credential rather than to the caller")


async def test_an_ungranted_agent_id_is_refused_rather_than_rebilled(proxy):
    """🚨 403, not a silent fall-back to the credential's own identity."""
    headers = _delegating_key(proxy, may_assert=["chat-agent"])
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", agent_id="knowledge_store"),
        headers=headers)
    assert resp.status_code == 403
    body = resp.json()
    assert body["code"] == "access_denied"
    assert "chat-agent" in body["error"]          # the grant is named


async def test_a_key_without_a_grant_still_ignores_the_body(proxy):
    """§1.5 rule 3, unchanged for every key that predates the feature."""
    proxy.svc._state.identity.keys.register(
        secret="plain-secret", agent_id="bridge-agent", key_id="h1", source="runtime")
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", agent_id="chat-agent"),
        headers={"Authorization": "Bearer plain-secret"})
    assert resp.status_code == 200
    proxy.svc._queue_db.flush()
    row = proxy.svc._queue_db._conn.execute(
        "SELECT agent_id FROM proxy_completions WHERE request_id=?",
        (resp.json()["request_id"],)).fetchone()
    assert row[0] == "bridge-agent", (
        "a key with no may_assert grant took the body's word for who it is")


async def test_plan_and_the_call_after_it_agree_about_who_is_asking(proxy):
    """§1.7 promises a plan and the call that follows it agree. `/rs/v1/plan`
    reports the caller's degrade/spill opt-in, which is read off the AGENT
    config — so a plan resolved as the credential would answer for a different
    caller than the one about to dispatch."""
    headers = _delegating_key(proxy, may_assert=["chat-agent"])
    resp = await proxy.client.post(
        "/rs/v1/plan", json=_body(intent="chat", agent_id="not-granted"),
        headers=headers)
    assert resp.status_code == 403, (
        "/rs/v1/plan accepted an identity /rs/v1/chat would refuse — a plan "
        "that answers for a caller the next call cannot be is worse than none")


async def test_an_ignored_delegation_is_visible_on_the_wire(proxy):
    """🚨 End to end, because the door overwrites `agent_id` with the resolved
    identity before `handle_submit` ever sees it — so the caller's own word has
    to be carried separately, and a unit test of the block cannot prove it was.

    That was a real bug in the first cut: the block reported `declared` as the
    RESOLVED name, so `honoured` was True in exactly the case it exists to flag.
    """
    proxy.svc._state.identity.keys.register(
        secret="nogrant", agent_id="bridge-agent", key_id="ng", source="runtime")
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", agent_id="chat-agent"),
        headers={"Authorization": "Bearer nogrant"})
    assert resp.status_code == 200
    identity = resp.json()["identity"]
    assert identity == {"agent_id": "bridge-agent", "declared": "chat-agent",
                        "honoured": False}, identity


async def test_a_granted_delegation_reports_honoured(proxy):
    headers = _delegating_key(proxy, may_assert=["chat-agent"])
    resp = await proxy.client.post(
        "/rs/v1/chat", json=_body(intent="chat", agent_id="chat-agent"), headers=headers)
    assert resp.json()["identity"] == {
        "agent_id": "chat-agent", "declared": "chat-agent", "honoured": True}


async def test_the_streaming_done_frame_carries_the_same_block(proxy):
    """One caller handling both wires must not have to handle two shapes."""
    proxy.svc._state.identity.keys.register(
        secret="ng2", agent_id="bridge-agent", key_id="ng2", source="runtime")
    frames = []
    async with proxy.client.stream(
            "POST", "/rs/v1/chat",
            json=_body(intent="chat", agent_id="chat-agent", payload={"stream": True}),
            headers={"Authorization": "Bearer ng2"}) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    done = [f for f in frames if f.get("type") == "done"]
    assert done, "no done frame"
    assert done[0]["identity"] == {"agent_id": "bridge-agent", "declared": "chat-agent",
                                  "honoured": False}
