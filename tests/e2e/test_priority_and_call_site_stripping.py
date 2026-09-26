"""A non-int `priority` body field reaches vLLM's own int-typed field -> 400 ->
502, reproduced live 2026-09-26: `POST /v1/chat/completions` with
`{"model": "tier3", "priority": "P2_POST_TURN", ...}`. `/v1/chat/completions`
forwards everything it does not itself read (docs/api.md §1.1), and a caller
carrying Roadstead's OWN `priority` spelling — a fleet band, never an int —
into an OpenAI-shaped body reached vLLM's genuine, int-typed scheduling
`priority` field and failed its `int_parsing` validation.

`tests/test_payload_normalization.py` proves the STRIP at the
`prepare_chat_payload` unit; this proves the REQUEST actually succeeds
end-to-end and that the fake backend is the thing that receives the corrected
body — the two levels `test_openai_door_fields.py` and the unit suite
cannot see together.
"""

from __future__ import annotations


async def test_a_string_priority_is_stripped_and_the_request_succeeds(proxy):
    # "tier3" is the vLLM class (see test_e2e_happy.py::test_vllm_shape_chat) —
    # the one engine whose OpenAI-compatible server actually has a typed
    # `priority` field to collide with.
    resp = await proxy.chat("hi", model="tier3",
                            extra={"priority": "P2_POST_TURN"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "echo: hi"

    sent = proxy.controller.requests[-1].body
    assert "priority" not in sent, (
        f"Roadstead's own priority spelling reached the backend body: {sent}")
    assert proxy.total_in_flight() == 0


async def test_call_site_is_stripped_and_the_request_succeeds(proxy):
    resp = await proxy.chat("hi", model="tier3",
                            extra={"call_site": "chat-agent.chat"})
    assert resp.status_code == 200, resp.text

    sent = proxy.controller.requests[-1].body
    assert "call_site" not in sent


async def test_a_genuine_int_priority_reaches_the_backend_unchanged(proxy):
    """The negative control: vLLM's own scheduling priority is a real
    backend feature and must pass through untouched, not just "not crash"."""
    resp = await proxy.chat("hi", model="tier3", extra={"priority": 5})
    assert resp.status_code == 200, resp.text

    sent = proxy.controller.requests[-1].body
    assert sent.get("priority") == 5, (
        f"a caller's genuine int priority must reach the backend unchanged, "
        f"got {sent.get('priority')!r}")


async def test_a_non_vllm_backend_strips_the_same_two_fields(proxy):
    """`chat` (the default e2e class) is llama.cpp — the strip is not vLLM-
    specific, so it must fire there too, even though llama.cpp never had this
    particular 400 in the first place."""
    resp = await proxy.chat("hi", model="chat",
                            extra={"priority": "P2_POST_TURN",
                                   "call_site": "x.y"})
    assert resp.status_code == 200, resp.text

    sent = proxy.controller.requests[-1].body
    assert "priority" not in sent
    assert "call_site" not in sent
