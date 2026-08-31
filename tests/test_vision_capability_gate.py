"""The proxy's vision-capability gate — `capabilities.vision` is load-bearing now.

Until 2026-08-20 `models.yaml`'s `capabilities.vision` was pure documentation:
nothing in `llmproxy/` read it. So when the tier2 split re-pointed a shared role
constant onto a box with no mmproj, discord's image and avatar describe went to a
text-only backend for a day. The backend was loud (HTTP 500 "image input is not
supported") but the caller swallowed it into "", which every vision helper reads
as "couldn't read the image" — no error, no alert, no red test.

Ledger: `a-role-rename-carried-vision-to-a-text-only-box`.

The gate counts (and can optionally refuse) an image sent to an endpoint that
cannot see. These tests pin three things, and the middle one is the one that
would have caught the original defect:
  1. the image DETECTOR sees both wire shapes the proxy accepts;
  2. the declaration actually REACHES the submit path, checked against the REAL
     models.yaml — a mirrored field that silently stays False would make the
     whole gate a no-op that still reports green;
  3. the gate counts, and enforces only when armed.
"""

from __future__ import annotations

import pytest

from roadstead import model_catalog
from roadstead.config import ProxyConfig
from roadstead.lifecycle import _carries_image


# --------------------------------------------------------------- 1. detector

def test_detector_sees_an_image_inside_the_SUBMIT_ENVELOPE():
    """🚨 THE REGRESSION TEST FOR THE GATE'S OWN FIRST BUG. Every real caller
    arrives wrapped — `{"agent_id", "endpoint", "payload": {...}}` — with the
    chat request under ``payload``. A detector reading only top-level
    ``messages`` passes a flat-body unit test and then never fires in
    production, which is exactly what happened on 2026-08-20: the live counter
    sat at {} through a confirmed backend 500."""
    assert _carries_image({
        "agent_id": "a", "endpoint": "e",
        "payload": {"model": "e", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
        ]}]},
    })


def test_detector_sees_the_openai_image_shape():
    assert _carries_image({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
    ]}]})


def test_detector_sees_the_anthropic_image_shape():
    """Two callers on this fleet send Anthropic typed blocks and the proxy
    rewrites them in backend.py. A detector that knew only the OpenAI shape
    would be blind to exactly those callers."""
    assert _carries_image({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/jpeg", "data": "AAAA"}},
    ]}]})


@pytest.mark.parametrize("body", [
    {},
    {"messages": []},
    {"messages": [{"role": "user", "content": "just text"}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
    {"messages": "malformed"},
    {"messages": [None, 7, {"role": "user"}]},
])
def test_detector_is_false_and_never_raises_on_non_image_bodies(body):
    """A telemetry gate must never be able to reject a valid request, so the
    detector returns False on anything unexpected rather than raising."""
    assert _carries_image(body) is False


# ------------------------------------------- 2. the declaration REACHES config

def test_vision_declaration_is_mirrored_from_the_real_catalog():
    """🚨 THE LOAD-BEARING TEST. `capabilities.vision` in models.yaml must
    survive into `EndpointConfig.vision`, or the submit-path gate reads False
    for everything and silently never fires — a guard that cannot fail.

    Checked against the REAL catalog, not a fixture, because the defect this
    guards was precisely a real-catalog fact that no fixture modelled."""
    catalog = model_catalog.load_catalog()
    endpoints = ProxyConfig().endpoints

    declared = {name for name in endpoints
                if (e := catalog.entry(name)) is not None
                and e.capabilities.get("vision")}
    assert declared, (
        "no endpoint in the live catalog declares capabilities.vision — either "
        "models.yaml lost the field or this test is looking in the wrong place; "
        "either way the gate below is vacuous")

    for name in declared:
        assert endpoints[name].vision is True, (
            f"{name!r} declares capabilities.vision in models.yaml but its "
            "EndpointConfig.vision is False — the mirror in "
            "model_catalog is broken, and the submit-path gate would warn on "
            "every legitimate image call to this endpoint")


def test_a_vision_and_a_text_only_endpoint_both_exist_so_the_gate_discriminates():
    """Anti-vacuity. If every endpoint were vision-capable the gate could never
    fire and this suite would pass while testing nothing."""
    endpoints = ProxyConfig().endpoints
    seeing = {n for n, e in endpoints.items() if e.vision}
    blind = {n for n, e in endpoints.items() if not e.vision}
    assert seeing, "no vision-capable endpoint — the gate would fire on everything"
    assert blind, "no text-only endpoint — the gate could never fire"


def test_the_chat_lane_is_text_only_and_the_analyst_can_see():
    """The specific topology the 2026-08-19 regression violated. If these two
    ever collapse onto one capability, re-read the discord vision routing in
    `agents/discord/_common.py` before changing this test."""
    endpoints = ProxyConfig().endpoints
    if "tier2-chat" in endpoints and "creative" in endpoints:
        assert endpoints["tier2-chat"].vision is False, (
            "tier2-chat became vision-capable — if jetty grew an mmproj that is "
            "good news, but discord's VISION_MODEL_ROLE split exists because it "
            "had none; re-read that decision rather than just editing this line")
        assert endpoints["creative"].vision is True, (
            "the analyst endpoint lost its vision capability — every vision "
            "caller on the fleet routes here")


# ------------------------------------------------------------ 3. gate counting

_IMAGE_MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "describe this"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
]}]


def _image_body(endpoint: str) -> dict:
    """The REAL submit envelope, not a hand-written flat body.

    🚨 This shape is the whole point. Both front doors —
    `http_handlers.handle_openai_chat` and `framework/llm_proxy_client` — wrap
    the chat request under ``payload``. The first version of these tests put
    ``messages`` at the top level, passed, and certified a gate that fired on
    nothing in production. Build the fixture from the wire, not from memory.
    """
    return {
        "agent_id": "test.vision_gate",
        "caller_id": "test.vision_gate",
        "call_site": "test.vision_gate",
        "endpoint": endpoint,
        "priority": 2,
        "payload_type": "chat_completion",
        "payload": {"model": endpoint, "messages": _IMAGE_MESSAGES},
    }


@pytest.mark.asyncio
async def test_image_to_a_blind_endpoint_is_counted_and_refused_when_armed():
    from roadstead.service import ProxyService

    svc = ProxyService(ProxyConfig())
    blind = next(n for n, e in svc._state.config.endpoints.items() if not e.vision)
    svc._state.flags.set_many({"vision_capability_enforce": True})

    body = _image_body(blind)
    # The tally is keyed by the RESOLVED endpoint, not by the `model` string the
    # caller sent — a role name and its endpoint_class are two different maps
    # (§11 lesson 6). Resolve it the same way the gate does.
    resolved = svc._lifecycle.resolve_endpoint(body)

    resp = await svc._lifecycle.handle_submit(body, None, openai=False)

    assert resp.status_code == 400
    tally = svc._state.vision_capability_violations.get(resolved)
    assert tally is not None and tally["count"] == 1, (
        svc._state.vision_capability_violations)
    assert tally["callers"]["test.vision_gate"] == 1


def test_a_text_only_request_to_a_blind_endpoint_is_not_counted():
    """The gate must key on IMAGE CONTENT, not on the endpoint. Text calls to
    non-vision endpoints are the overwhelming majority of fleet traffic;
    counting those would bury the real signal instantly.

    Asserted on the detector rather than through a full submit, because a
    text-only request is (correctly) NOT short-circuited — it would go on to
    enqueue against a scheduler this test never started."""
    body = _image_body("anything")
    body["payload"]["messages"] = [{"role": "user", "content": "plain text only"}]
    assert _carries_image(body) is False


@pytest.mark.asyncio
async def test_the_gate_is_shadow_by_default():
    """Unarmed, it must COUNT without changing behaviour — an enforcement that
    arrives with the telemetry would take a working caller down the moment a
    models.yaml stanza forgot to declare vision."""
    from roadstead.service import ProxyService

    svc = ProxyService(ProxyConfig())
    blind = next(n for n, e in svc._state.config.endpoints.items() if not e.vision)
    assert not svc._state.flags.get("vision_capability_enforce"), (
        "vision_capability_enforce ships ARMED — that is a behaviour change, "
        "not a telemetry addition; see the field comment in config.py")
