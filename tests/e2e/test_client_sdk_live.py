"""The client SDK driven against the real proxy, over the real front door.

The contract half — the SDK's hand transcription of ``docs/api.md``, and the
proof that it imports without the server — is ``tests/test_client_sdk.py``. This
half checks the typed views against BYTES: everything below goes out through
``roadstead.client``, through Starlette, into ``ProxyService``, and comes back
the same way. A fixture somebody wrote to match the models would prove nothing.
"""
from __future__ import annotations

import httpx
import pytest

from roadstead.client import AsyncRoadsteadClient, UnroutableError


@pytest.fixture
async def sdk(proxy):
    """The real SDK over the real ASGI proxy. `base_url` is nominal — the
    transport is the app — so what is exercised is the SDK's own request
    building, envelope parsing and error typing, not httpx's networking."""
    transport = httpx.ASGITransport(app=proxy.app, raise_app_exceptions=False,
                                    client=("127.0.0.1", 41999))
    http = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                             timeout=30.0)
    client = AsyncRoadsteadClient("http://proxy", client=http)
    try:
        yield client
    finally:
        await http.aclose()


async def test_models_and_intents_come_back_typed(sdk):
    models = await sdk.models()
    by_name = {m.endpoint: m for m in models}
    t3 = by_name["tier3"]
    assert "vision" in t3.capabilities and t3.kind == "chat"
    assert t3.context > 0 and t3.routed and t3.free_slots >= 0
    assert t3.typical_ms is None            # never measured, not fast
    assert t3.price.real is False           # local: a saving, not an invoice
    assert "reasoner" in t3.aliases
    assert by_name["spill-chat"].routed is False

    names = {i["name"] for i in await sdk.intents()}
    assert "reasoning" in names and "fast-chat" in names


async def test_plan_then_chat_through_the_sdk(sdk):
    plan = await sdk.plan(intent="reasoning", est_in=2000, est_out=200)
    assert plan.endpoint == "tier3"
    assert plan.recommended_deadline_s > 0
    assert plan.model.endpoint == "tier3"
    assert isinstance(plan.may_degrade, bool) and isinstance(plan.may_spill, bool)

    result = await sdk.chat(
        intent="reasoning",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16)
    assert result.content
    assert result.attribution.endpoint == plan.endpoint
    assert result.attribution.substituted is False
    assert result.attribution.avoided_usd >= 0
    # 🚨 Exactly one of the two is ever non-zero, and a consumer must never add
    # them. Local capacity is a saving; a remote provider's price is an invoice.
    assert result.attribution.spent_usd == 0.0
    assert result.timing.deadline_source == "computed"
    assert result.timing.ttft_ms is None
    assert result.usage.input_tokens > 0


async def test_a_pin_and_a_supplied_deadline_go_through(sdk):
    result = await sdk.chat(model="reasoner", deadline_s=42.0,
                            messages=[{"role": "user", "content": "hi"}])
    assert result.attribution.requested == "reasoner"
    assert result.attribution.endpoint == "tier3"
    assert result.timing.deadline_source == "caller"
    assert result.timing.deadline_s == 42.0


async def test_an_unroutable_request_raises_the_typed_error(sdk):
    with pytest.raises(UnroutableError) as exc:
        await sdk.chat(model="tier2", requires=["vision"],
                       messages=[{"role": "user", "content": "hi"}])
    err = exc.value
    assert err.code == "unknown_endpoint"
    assert err.status == 404
    assert err.deferrable is False          # deterministic; retrying is futile
    assert [c["endpoint"] for c in err.considered] == ["tier2"]


async def test_streaming_yields_accepted_chunks_and_done(sdk):
    frames = [f async for f in sdk.stream(
        intent="chat", messages=[{"role": "user", "content": "hi"}],
        max_tokens=16)]
    assert frames[0]["type"] == "accepted"
    assert frames[-1]["type"] == "done"
    assert any(f["type"] == "chunk" for f in frames)


async def test_text_stream_reduces_to_the_deltas(sdk):
    text = "".join([d async for d in sdk.text_stream(
        intent="chat", messages=[{"role": "user", "content": "hello"}],
        max_tokens=16)])
    assert "hello" in text


async def test_the_narrowing_flags_ride_the_envelope(sdk):
    """A caller declining spill for one prompt. It cannot be observed in the
    response by design (declining produces the same local service), so what is
    pinned here is that the SDK builds the envelope the server reads — the
    server-side effect is `test_substitution_narrowing.py`."""
    from roadstead.client._client import _envelope

    body = _envelope(messages=[{"role": "user", "content": "secret"}],
                     intent="chat", allow_spill=False)
    assert body["substitution"] == {"spill": False}
    assert "degrade" not in body["substitution"]   # undeclared stays undeclared
    result = await sdk.chat(intent="chat", allow_spill=False,
                            messages=[{"role": "user", "content": "secret"}])
    assert result.attribution.substitution == ""


async def test_the_sdk_can_speak_the_negative_constraint(sdk):
    """`exclude` is a routing declaration, so it belongs OUTSIDE `payload` with
    the other ones — inside, it would reach a backend as an unknown field.

    And it is a constraint, not a hint: a name this fleet does not serve is
    refused by the server rather than dropped, so the SDK that can send one must
    also surface that refusal as `UnroutableError` rather than as a bare 404.
    """
    from roadstead.client._client import _envelope

    body = _envelope(messages=[{"role": "user", "content": "hi"}],
                     intent="chat", exclude=["tier1"])
    assert body["exclude"] == ["tier1"]
    assert "exclude" not in body["payload"]

    result = await sdk.chat(intent="chat", exclude=["tier1"],
                            messages=[{"role": "user", "content": "hi"}])
    assert result.attribution.endpoint != "tier1"

    with pytest.raises(UnroutableError):
        await sdk.chat(intent="chat", exclude=["no-such-endpoint"],
                       messages=[{"role": "user", "content": "hi"}])


async def test_unknown_response_fields_stay_reachable():
    """A proxy newer than the SDK must be usable, not lossy. A client that
    discards what it does not recognise turns every server-side addition into an
    invisible loss whose cause nobody can distinguish from it not being sent."""
    from roadstead.client import CallResult

    r = CallResult({"status": "ok", "request_id": "req_1",
                    "attribution": {"endpoint": "tier3", "future_field": 7},
                    "brand_new_block": {"x": 1}})
    assert r.attribution.endpoint == "tier3"
    assert r.attribution.raw["future_field"] == 7
    assert r.raw["brand_new_block"] == {"x": 1}


# --------------------------------------------------------------------------- #
# The other two payload types
# --------------------------------------------------------------------------- #

async def test_embed_reaches_the_embedder_and_the_body_comes_back_whole(sdk):
    """🚨 The claim being checked is that `response` is UNTOUCHED.

    `CallResult.response` is documented as the backend's own body, and the whole
    argument for routing embeddings through `/rs/v1/chat` rather than
    `/v1/embeddings` rests on it: the OpenAI door translates a hybrid reply into
    `{object, data, usage}` and drops the sparse and colbert halves, because
    OpenAI's schema has nowhere to put them. So this asserts the bytes, not that
    a call succeeded — a translation that quietly happened here would look
    identical to success.
    """
    r = await sdk.embed(texts=["a chunk", "another chunk"])
    assert r.attribution.endpoint == "embed"
    # The fake speaks the OpenAI-shaped dialect, so what proves "untouched" is
    # that the backend's own envelope survives rather than being re-wrapped:
    # `object`/`data`/`model`/`usage` are the FAKE's keys, not ours.
    assert r.response["object"] == "list"
    assert len(r.response["data"]) == 2
    assert r.response["data"][0]["embedding"]
    assert "usage" in r.response and "model" in r.response


async def test_embed_sends_both_dialects_and_the_declaration_defaults(sdk):
    """Both `texts` and `input`, for `handle_openai_embeddings`' reason: a shim
    reads `texts` and ignores the rest, an OpenAI-shaped server reads `input` and
    SIZES ITS REPLY from it — so sending only `texts` returns one vector for an
    N-input request, a well-formed list of the wrong length.

    And `intent` is supplied only when the caller declared nothing: §1.7.1
    requires a declaration, but a caller who pinned must keep their pin.
    """
    from roadstead.client._client import _envelope, _routing

    body = _envelope(
        payload_type="embedding", payload={"texts": ["a"], "input": ["a"]},
        **_routing({}, method="embed", default_kind="embed"))
    assert body["payload"]["texts"] == body["payload"]["input"] == ["a"]
    assert body["intent"] == "embed"
    # 🚨 A pin keeps its pin and gains NO intent — but it does gain the `kind`,
    # without which it resolves against `kind='chat'` and 404s at an embedder.
    assert _routing({"model": "embed"}, method="embed",
                    default_kind="embed") == {"model": "embed", "kind": "embed"}
    # An explicit kind is never overwritten.
    assert _routing({"kind": "chat"}, method="embed",
                    default_kind="embed")["kind"] == "chat"

    # And the pin path really resolves, which is what the unit assertions above
    # only make plausible.
    assert (await sdk.embed(texts=["a"], model="embed")
            ).attribution.endpoint == "embed"


async def test_rerank_has_a_route_again(sdk):
    """🚨 Rerank had NO route from anywhere between the removal of `/v1/submit`
    and the SDK learning to send `payload_type`. There is deliberately no
    `/v1/rerank` door: OpenAI has no rerank shape to be compatible with."""
    r = await sdk.rerank(query="why is the queue deep?",
                         documents=["about queues", "about cheese", "about DRR"])
    assert r.attribution.endpoint == "rerank"
    results = r.response["results"]
    assert len(results) == 3
    assert all("relevance_score" in row for row in results)
    # Descending by construction in the fake; what matters here is that the
    # scores arrived at all, which is the thing the missing route cost.
    assert results[0]["relevance_score"] > results[-1]["relevance_score"]


async def test_the_raw_passthrough_sends_a_payload_type_verbatim(sdk):
    """`call()` is the escape hatch, so it must NOT validate against this SDK's
    transcribed literal — a proxy newer than the SDK stays usable. What it does
    refuse is a payload type with no routing declaration."""
    # 🚨 `kind` is explicit here: `call()` supplies none, deliberately — it
    # cannot know one for a payload type it has never heard of — and the
    # resolver's default is `chat`.
    r = await sdk.call(payload_type="embedding", model="embed", kind="embed",
                       payload={"texts": ["x"], "input": ["x"]})
    assert r.response["object"] == "list"

    with pytest.raises(TypeError, match="payload_type"):
        await sdk.embed(texts=["x"], payload_type="rerank")


async def test_the_correlation_fields_reach_the_durable_record(proxy, sdk):
    """`caller_id` and `request_id` were read by the server and sent by nobody.
    Losing `caller_id` flattens `/v1/fleet/top-callers` — a missing MEASUREMENT,
    which reports itself to no one, so it is checked against the record."""
    r = await sdk.chat(intent="chat", caller_id="kv4.recall",
                       request_id="caller-own-id-1",
                       messages=[{"role": "user", "content": "hi"}],
                       max_tokens=8)
    # 🚨 The caller's own id BECOMES the request id (`QueuedRequest.create`
    # mints one only when none was sent), which is what makes a caller's log
    # line joinable to our completion row — and is exactly what was lost while
    # the SDK sent no `request_id` at all.
    assert r.request_id == "caller-own-id-1"
    # The completion row is written by the single background writer, so the
    # flush is not a convenience — without it this reads an empty table and
    # passes or fails on timing.
    proxy.svc._queue_db.flush()
    row = proxy.svc._queue_db._conn.execute(
        "SELECT caller_id, session_id FROM proxy_completions WHERE request_id=?",
        (r.request_id,)).fetchone()
    assert row is not None, "no completion row for the request the SDK just made"
    assert row[0] == "kv4.recall", (
        f"the caller_id the SDK sent did not reach the durable record: {row[0]!r}")
