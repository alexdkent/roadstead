"""M2 — structured error taxonomy: machine-readable ``code`` on every proxy
error envelope, with the LOAD-BEARING deferrable substrings preserved.

Each error path asserts two things:
  1. the ``code`` field (additive, machine-readable),
  2. that the message carries — or pointedly does not carry — the marker
     substrings that decide whether a caller defers or surfaces the error.

**Why this is not a tautology.** Deferral classification does not live in the
proxy: callers match SUBSTRINGS of the error message, historically in the
host's ``framework/nexus_errors.py``. The original version of this file
imported that classifier and used it as an oracle, which is exactly why it
could not run standalone. Importing Roadstead's own copy of the rule and
asserting it agrees with itself would prove nothing — two values compared from
one source (``tests/_pending/README.md``).

So the markers live in ``tests/wire_contract.py`` as literals transcribed from
the shared boundary object, ``docs/api.md`` §2.2 — read that module's docstring
for the full reasoning. This file pins the SERVER side to them; the monorepo's
integration test pins the CLIENT side to the same literals. Drift on either side
then fails on that side, which is the whole point.

The one assertion genuinely left to the monorepo is marked inline: that a 502
envelope becomes deferrable via the ``LLM proxy error 502`` prefix, because the
client constructs that prefix — the proxy never emits it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from roadstead.backend import BackendError, BackendResponse
from roadstead.config import ProxyConfig
from roadstead.service import ProxyService
from tests.wire_contract import carries_deferral_marker

class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


def _body(endpoint="tier3", **payload_extra):
    return {
        "agent_id": "a", "endpoint": endpoint, "priority": "P3_INGESTION",
        "call_site": "t", "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "x"}], **payload_extra},
        "timeout_s": 5.0,
    }



@pytest.mark.asyncio
async def test_draining_is_deferrable_with_code():
    svc = ProxyService(ProxyConfig())
    svc._draining.set()
    resp = await svc.handle_submit(_body(), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "draining"
    assert carries_deferral_marker(body["error"])  # "backpressure"


@pytest.mark.asyncio
async def test_circuit_open_and_paused_are_deferrable_with_codes():
    svc = ProxyService(ProxyConfig())
    # Auto-circuit trip.
    svc._endpoint_health["tier3"]["healthy"] = False
    resp = await svc.handle_submit({**_body(), "priority": "P1_TURN_SUPPORT"}, _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "circuit_open"
    assert carries_deferral_marker(body["error"])  # "circuit open"

    # Operator drain reads as draining, still deferrable.
    svc._endpoint_health["tier3"]["healthy"] = True
    svc._paused_endpoints.add("tier3")
    resp = await svc.handle_submit({**_body(), "priority": "P1_TURN_SUPPORT"}, _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 503
    assert body["code"] == "draining"
    assert carries_deferral_marker(body["error"])  # "backpressure"


@pytest.mark.asyncio
async def test_backpressure_shed_is_deferrable_with_code():
    svc = ProxyService(ProxyConfig())
    svc._shed_depth = 0  # any queued depth sheds
    resp = await svc.handle_submit(_body(), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 429
    assert body["code"] == "backpressure"
    assert resp.headers.get("retry-after")
    assert carries_deferral_marker(body["error"])


@pytest.mark.asyncio
async def test_invalid_grammar_is_not_deferrable_and_coded():
    svc = ProxyService(ProxyConfig())
    resp = await svc.handle_submit(
        _body(grammar="not a grammar at all"), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 422
    assert body["code"] == "invalid_grammar"
    # Deterministic request error: the message must NOT carry a deferral
    # marker (a defer-loop on a static grammar never converges).
    assert not carries_deferral_marker(
        body.get("detail", ""), body.get("error", ""))


@pytest.mark.asyncio
async def test_unknown_endpoint_enforce_404_not_deferrable():
    svc = ProxyService(ProxyConfig())
    svc._flags.set_many({"unknown_endpoint_enforce": True})
    resp = await svc.handle_submit(_body(endpoint="qwen-composr-typo"), _Req())
    body = json.loads(resp.body)
    assert resp.status_code == 404
    assert body["code"] == "unknown_endpoint"
    assert not carries_deferral_marker(body["error"])


@pytest.mark.asyncio
async def test_backend_error_envelope_coded_and_deferrable_via_status():
    svc = ProxyService(ProxyConfig())

    async def failing_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        raise BackendError(400, "backend rejected the payload")

    svc._backend.call = failing_call
    await svc.startup()
    try:
        resp = await asyncio.wait_for(svc.handle_submit(_body(), _Req()), timeout=10.0)
        body = json.loads(resp.body)
        assert resp.status_code == 502
        assert body["status"] == "error"
        assert body["code"] == "backend_error"
        # LEFT TO THE MONOREPO: a 502 envelope becomes deferrable via the
        # client-constructed "LLM proxy error 502: ..." prefix. The proxy never
        # emits that prefix, so there is nothing here to assert it against —
        # asserting it from this side would only restate the client's own rule.
        # What IS ours is the status and the code, above.
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_proxy_timeout_coded():
    svc = ProxyService(ProxyConfig())

    async def never_returns(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        # Slow enough to outlive the caller's 0.3s deadline, short enough that
        # shutdown's drain isn't held to its 30s straggler deadline.
        await asyncio.sleep(1.2)
        from roadstead.backend import BackendTimeout
        raise BackendTimeout("late")

    svc._backend.call = never_returns
    await svc.startup()
    try:
        resp = await asyncio.wait_for(
            svc.handle_submit({**_body(), "timeout_s": 0.3}, _Req()), timeout=10.0)
        body = json.loads(resp.body)
        assert resp.status_code == 504
        assert body["code"] == "proxy_timeout"
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_success_envelope_unchanged_no_code_key():
    """The ok envelope is a hard client contract — taxonomy must be additive
    on ERRORS only."""
    svc = ProxyService(ProxyConfig())

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")

    svc._backend.call = ok_call
    await svc.startup()
    try:
        resp = await asyncio.wait_for(svc.handle_submit(_body(), _Req()), timeout=10.0)
        body = json.loads(resp.body)
        assert resp.status_code == 200
        assert body["status"] == "ok"
        assert "code" not in body
        assert set(body) == {"status", "request_id", "response",
                             "attribution", "timing", "usage"}
        # 🚨 docs/api.md §1.6: a caller cannot observe its own spend demotion in
        # a response, so the enriched envelope publishes no band, no priority
        # and no queue position. Any of them would make a threshold that "never
        # rejects" into one every client could detect and branch on.
        leaked = {"priority", "band", "queue_position", "demoted", "effective_priority"}
        flat = json.dumps(body)
        for key in leaked:
            assert f'"{key}"' not in flat, (
                f"the enriched envelope leaks {key!r} — see docs/api.md §1.6")
    finally:
        await svc.shutdown()
