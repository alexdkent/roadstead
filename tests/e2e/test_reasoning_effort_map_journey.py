"""The "no accidental MAX" reasoning-effort guard, through the real front
door: a caller's `reasoning_effort` reaches the fake backend NORMALIZED
against the endpoint's declared `policy.reasoning_effort_map`, and the
remap is visible on `/v1/status`.

The unit suite (`tests/test_reasoning_effort_map.py`) proves the correction
itself; this proves it is actually WIRED into the submit path and that the
operator-facing counter is reachable.
"""

from __future__ import annotations


def _arm(proxy, effort_map: dict, ep: str = "tier3") -> None:
    """Declare the map on ONE endpoint's live config — the same shortcut
    `test_goodput_collapse_journey.py` uses for its own policy block, pinned
    by `test_reasoning_effort_map.py::test_declared_map_reaches_endpoint_config`
    so this cannot diverge from the real `models.yaml` path unnoticed."""
    proxy.svc._config.endpoints[ep].reasoning_effort_map = dict(effort_map)


async def test_a_typo_never_reaches_the_backend(proxy):
    _arm(proxy, {"low": "low", "high": "high"})
    resp = await proxy.chat("hi", model="tier3",
                            extra={"reasoning_effort": "mediumm"})
    assert resp.status_code == 200, resp.text

    sent = proxy.controller.requests[-1].body
    assert "reasoning_effort" not in sent, (
        f"an undeclared word must never reach the backend: {sent}")

    status = (await proxy.client.get("/v1/status")).json()
    remaps = status["reliability"]["reasoning_effort_remaps"]
    row = remaps["tier3|mediumm->"]
    assert row["to"] is None
    assert row["count"] >= 1


async def test_medium_is_translated_to_the_templates_own_word(proxy):
    _arm(proxy, {"medium": "high"})
    resp = await proxy.chat("hi", model="tier3",
                            extra={"reasoning_effort": "medium"})
    assert resp.status_code == 200, resp.text

    sent = proxy.controller.requests[-1].body
    assert sent.get("reasoning_effort") == "high"

    status = (await proxy.client.get("/v1/status")).json()
    row = status["reliability"]["reasoning_effort_remaps"]["tier3|medium->high"]
    assert row["from"] == "medium" and row["to"] == "high"


async def test_an_undeclared_endpoint_forwards_the_callers_word_unchanged(proxy):
    """`chat` (llama.cpp, the default e2e class) declares no map at all — the
    guard must not fire anywhere it wasn't asked to."""
    resp = await proxy.chat("hi", model="chat",
                            extra={"reasoning_effort": "whatever-i-like"})
    assert resp.status_code == 200, resp.text

    sent = proxy.controller.requests[-1].body
    assert sent.get("reasoning_effort") == "whatever-i-like"
