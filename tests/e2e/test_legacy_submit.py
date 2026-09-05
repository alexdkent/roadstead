"""``POST /v1/submit`` end to end — over the real ASGI door and a real socket.

What can only be checked here: **route resolution** (a flag that gates a route
is a fact about the route table, and the only honest test of "OFF means absent"
is the 404 a caller actually receives) and **SSE framing** (an in-process call
returns a generator; a caller reads bytes).

The backend is the repo's programmable fake on an ephemeral socket, so the
embedding and rerank assertions below are about a body that really crossed a
wire — which is the whole claim for those two payload types: nothing translates
them, in either direction.
"""
from __future__ import annotations

import json

import httpx
import pytest

from roadstead.__main__ import build_app
from roadstead.config import ProxyConfig


@pytest.fixture(autouse=True)
def _legacy_door_open(monkeypatch):
    """The flag, set before ``build_app`` runs.

    🚨 Autouse and function-scoped so it is set up before the ``proxy`` fixture
    it has to precede — the route table is built once, at ``build_app``, and a
    flag flipped afterwards would change nothing while looking like it had.
    ``test_the_route_is_absent_when_the_flag_is_unset`` builds its own app for
    exactly that reason.
    """
    monkeypatch.setenv("ROADSTEAD_LEGACY_SUBMIT", "1")


def _body(**kw) -> dict:
    body = {
        "agent_id": "e2e-caller",
        "endpoint": "chat",
        "priority": "P1_TURN_SUPPORT",
        "call_site": "e2e.legacy",
        "payload_type": "chat_completion",
        "payload": {"messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16},
        "timeout_s": 20.0,
    }
    payload = kw.pop("payload", None)
    if payload is not None:
        body["payload"] = payload
    body.update(kw)
    return body


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #

async def test_the_route_is_absent_when_the_flag_is_unset(monkeypatch, tmp_path):
    """🚨 The same 404 an unknown path gets — body and all.

    A door that answered 405, or 404 with a message about itself, would still be
    telling a caller it exists. Nothing dispatches on this path, so the app is
    built without a backend or a startup: what is under test is the route table.
    """
    monkeypatch.delenv("ROADSTEAD_LEGACY_SUBMIT", raising=False)
    app = build_app(ProxyConfig(queue_db_path=str(tmp_path / "queue.db")))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False,
                                    client=("127.0.0.1", 41999))
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://proxy") as client:
        gone = await client.post("/v1/submit", json=_body())
        unknown = await client.post("/v1/no-such-route", json={})
    assert gone.status_code == 404
    assert (gone.status_code, gone.text) == (unknown.status_code, unknown.text)


async def test_the_route_answers_when_the_flag_is_set(proxy):
    """The counterweight — without it the test above passes on a broken door."""
    resp = await proxy.client.post("/v1/submit", json=_body())
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #

async def test_the_sync_envelope_is_the_published_six_keys(proxy):
    resp = await proxy.client.post("/v1/submit", json=_body())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"status", "request_id", "queue_wait_ms",
                         "backend_latency_ms", "estimated_cost_ss", "response"}
    assert body["status"] == "ok"
    # `response` is the BACKEND's own chat.completion, unwrapped by nothing.
    assert body["response"]["choices"][0]["message"]["content"]
    assert body["response"]["object"] == "chat.completion"
    assert body["backend_latency_ms"] >= 0


# --------------------------------------------------------------------------- #
# Streaming — docs/api.md §1.9.3
# --------------------------------------------------------------------------- #

async def _frames(proxy, body: dict) -> tuple[list[dict], httpx.Headers]:
    frames: list[dict] = []
    async with proxy.client.stream("POST", "/v1/submit", json=body) as resp:
        assert resp.status_code == 200
        headers = resp.headers
        async for line in resp.aiter_lines():
            line = line.strip()
            if line.startswith("data: "):
                frames.append(json.loads(line[len("data: "):]))
    return frames, headers


async def test_the_stream_is_queued_admitted_chunks_done(proxy):
    """The frame sequence, in order, with no ``[DONE]`` sentinel.

    🚨 ``chunk.data`` is a STRING carrying the backend's raw chunk JSON, not a
    parsed object. Every consumer of this door does its own ``json.loads`` on
    it; handing them an object would break each one at the same line.
    """
    body = _body()
    body["payload"]["stream"] = True
    frames, headers = await _frames(proxy, body)

    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"

    assert frames[0]["type"] == "queued"
    assert frames[0]["request_id"].startswith("req_")
    assert set(frames[0]) == {"type", "request_id"}

    assert frames[1]["type"] == "admitted"
    assert frames[1]["queue_wait_ms"] >= 0

    chunks = [f for f in frames if f["type"] == "chunk"]
    assert chunks, frames
    for chunk in chunks:
        assert isinstance(chunk["data"], str)
        assert json.loads(chunk["data"])["object"] == "chat.completion.chunk"

    done = frames[-1]
    assert done["type"] == "done"
    assert done["queue_wait_ms"] >= 0
    assert done["backend_latency_ms"] >= 0
    assert done["ttft_ms"] >= 0
    assert set(done["usage"]) == {"prompt_tokens", "completion_tokens"}
    # The enriched wire's reshaped `done` must not leak onto this one.
    assert "attribution" not in done and "timing" not in done

    # The old door never emitted the OpenAI sentinel; an internal-envelope
    # consumer would try to json.loads it.
    assert not any(f.get("type") == "[DONE]" for f in frames)


async def test_a_stream_that_fails_ends_in_an_error_frame(proxy):
    """The terminal frame is ``error`` and the opening one is still ``queued`` —
    a caller that has already read the opening frame must not be left waiting on
    a ``done`` that is never coming."""
    body = _body()
    body["payload"]["stream"] = True
    proxy.controller.set_fault("http_500", 0.0)
    frames, _ = await _frames(proxy, body)
    assert frames[0]["type"] == "queued"
    assert frames[-1]["type"] == "error"
    assert isinstance(frames[-1]["error"], str) and frames[-1]["error"]


# --------------------------------------------------------------------------- #
# Embeddings and rerank — verbatim, in both directions
# --------------------------------------------------------------------------- #

async def test_an_embedding_response_is_the_backend_body_verbatim(proxy):
    """🚨 No OpenAI translation on this door. ``/v1/embeddings`` reshapes a
    backend body into OpenAI's `data[].embedding` list; this one does not, and a
    caller that has been reading the shim's own dialect for a year would break
    on the day it started."""
    resp = await proxy.client.post("/v1/submit", json=_body(
        endpoint="embed", payload_type="embedding",
        payload={"texts": ["one", "two"], "return_dense": True}))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    served = body["response"]
    # Exactly what the fake's /embed shim returned — keys and all.
    assert set(served) == {"object", "data", "model", "usage"}
    assert served["data"][0]["object"] == "embedding"
    assert isinstance(served["data"][0]["embedding"], list)


async def test_a_rerank_response_is_the_backend_body_verbatim(proxy):
    resp = await proxy.client.post("/v1/submit", json=_body(
        endpoint="rerank", payload_type="rerank",
        payload={"model": "reranker", "query": "q",
                 "documents": ["a", "b", "c"]}))
    assert resp.status_code == 200
    served = resp.json()["response"]
    assert set(served) == {"results", "usage"}
    assert [r["index"] for r in served["results"]] == [0, 1, 2]
    assert served["results"][0]["relevance_score"] == 1.0


# --------------------------------------------------------------------------- #
# The deprecation notice, over the wire
# --------------------------------------------------------------------------- #

async def test_the_status_counter_names_the_callers_still_on_this_door(proxy):
    """Removing the door again is gated on this being empty, not on a date."""
    await proxy.client.post("/v1/submit", json=_body(agent_id="chat-agent"))
    await proxy.client.post("/v1/submit", json=_body(agent_id="chat-agent"))
    await proxy.client.post("/v1/submit", json=_body(agent_id="forum-agent"))

    status = (await proxy.client.get("/v1/status")).json()
    tally = status["reliability"]["legacy_submits"]
    assert tally["count"] == 3
    assert tally["callers"] == {"chat-agent": 2, "forum-agent": 1}
