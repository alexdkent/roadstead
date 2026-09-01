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
    from roadstead.client._client import _chat_body

    body = _chat_body(
        messages=[{"role": "user", "content": "secret"}], intent="chat",
        model="", requires=None, kind="", min_context=0, prefer="",
        priority=None, interactive=None, deadline_s=None,
        allow_degrade=None, allow_spill=False, call_site="", session_id=None,
        turn_id=None, stream=False, payload=None, extra_payload=None)
    assert body["substitution"] == {"spill": False}
    assert "degrade" not in body["substitution"]   # undeclared stays undeclared
    result = await sdk.chat(intent="chat", allow_spill=False,
                            messages=[{"role": "user", "content": "secret"}])
    assert result.attribution.substitution == ""


async def test_unknown_response_fields_stay_reachable():
    """A proxy newer than the SDK must be usable, not lossy. A client that
    discards what it does not recognise turns every server-side addition into an
    invisible loss whose cause nobody can distinguish from it not being sent."""
    from roadstead.client import ChatResult

    r = ChatResult({"status": "ok", "request_id": "req_1",
                    "attribution": {"endpoint": "tier3", "future_field": 7},
                    "brand_new_block": {"x": 1}})
    assert r.attribution.endpoint == "tier3"
    assert r.attribution.raw["future_field"] == 7
    assert r.raw["brand_new_block"] == {"x": 1}
