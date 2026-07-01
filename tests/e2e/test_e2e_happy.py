"""Happy-path E2E: the real proxy, driven over its front door, talking to the
fake backend over a real socket. Asserts response correctness, streaming shape,
embeddings, and the slot-accounting invariant (in-flight returns to baseline).

These are the baseline the adversarial/fuzz phases build on: if the harness
can't serve a clean request end-to-end, nothing else it reports is trustworthy.
"""

from __future__ import annotations

import json

import pytest


async def test_chat_sync_roundtrips(proxy):
    assert proxy.total_in_flight() == 0
    resp = await proxy.chat("hello world")
    assert resp.status_code == 200
    body = resp.json()
    # bare OpenAI chat.completion shape
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "echo: hello world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 12
    # slot accounting: no leak
    assert proxy.total_in_flight() == 0
    # the fake actually received the dispatched call
    assert any(r.path == "/v1/chat/completions" for r in proxy.controller.requests)


async def test_chat_stream_frames(proxy):
    frames = await proxy.stream_frames("one two three")
    assert frames, "expected SSE frames"
    assert frames[-1] == "[DONE]"
    # reassemble the streamed content from chat.completion.chunk deltas
    content = ""
    saw_stop = False
    for f in frames[:-1]:
        obj = json.loads(f)
        choices = obj.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content += delta.get("content") or ""
            if choices[0].get("finish_reason") == "stop":
                saw_stop = True
    assert saw_stop
    assert "echo: one two three" in content
    assert proxy.total_in_flight() == 0


async def test_vllm_shape_chat(proxy):
    # `thinker` is the vLLM class (role llama-thinker) → the proxy applies the
    # vLLM normalization (grammar relocation, enable_thinking) before dispatch;
    # the fake serves it identically. Proves both engine shapes are exercised.
    resp = await proxy.chat("vllm path", model="thinker")
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "echo: vllm path"
    assert proxy.total_in_flight() == 0


async def test_embeddings_roundtrip(proxy):
    resp = await proxy.client.post(
        "/v1/embeddings",
        json={"model": "bge-m3", "input": ["alpha", "beta"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    assert len(body["data"][0]["embedding"]) == 8
    assert proxy.total_in_flight() == 0


async def test_unknown_model_is_404_not_500(proxy):
    # north-face: an unknown model must be a clean typed 404, never a burned
    # slot + late 502.
    resp = await proxy.chat("hi", model="no-such-model-xyz")
    assert resp.status_code == 404
    assert proxy.total_in_flight() == 0


async def test_repeated_calls_no_slot_leak(proxy):
    for i in range(5):
        resp = await proxy.chat(f"call {i}")
        assert resp.status_code == 200
    assert proxy.total_in_flight() == 0
