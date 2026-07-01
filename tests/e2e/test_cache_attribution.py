"""Phase 2a — per-caller prefix-cache attribution: builder happy-path + parity.

Covers the capture→persist→rollup→surface chain over the REAL in-process proxy +
fake backend:
  - the pure ``extract_cached_tokens`` parser (present / top-level / absent),
  - sync AND streaming capture of vLLM ``usage.prompt_tokens_details.cached_tokens``,
  - the per-caller rollup math (hit rate = cached/attributable),
  - graceful n/a when the backend reports no counter (llama.cpp shape),
  - the fleet de-blend (NULL rows never drag the number to 0 — the ~4.8% artifact),
  - the ``/v1/fleet/cache-attribution`` endpoint,
  - the per-endpoint ``cache_hit_rate`` on ``/v1/status`` after a stats cycle.

The adversarial (hostile-shape) matrix lives in the independent track's file; this
is the builder's own happy + parity coverage only.
"""
from __future__ import annotations

import pytest

from originfleet.llmproxy.backend import extract_cached_tokens


# --------------------------------------------------------------------------- #
# Unit — the pure parser (both seams meet here).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("usage,expected", [
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 6}}, 6),
    ({"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}}, 0),
    ({"prompt_tokens": 12, "cached_tokens": 8}, 8),            # top-level fallback
    ({"prompt_tokens": 12}, None),                            # llama.cpp: absent
    ({}, None),
    ({"prompt_tokens_details": {}}, None),                    # present dict, no key
    ({"prompt_tokens_details": {"cached_tokens": None}}, None),
    ({"prompt_tokens_details": {"cached_tokens": "5"}}, None),  # string → reject
    ({"prompt_tokens_details": {"cached_tokens": True}}, None),  # bool → reject
    ({"prompt_tokens_details": {"cached_tokens": -3}}, None),  # negative → reject
    ({"prompt_tokens_details": "notadict"}, None),
    ("notadict", None),
    (None, None),
])
def test_extract_cached_tokens(usage, expected):
    assert extract_cached_tokens(usage) == expected


def test_extract_cached_tokens_prefers_details_over_toplevel():
    # details is the canonical (vLLM) location; if both appear, trust details.
    usage = {"prompt_tokens": 12, "cached_tokens": 99,
             "prompt_tokens_details": {"cached_tokens": 6}}
    assert extract_cached_tokens(usage) == 6


# --------------------------------------------------------------------------- #
# E2E — capture + rollup over the real proxy.
# --------------------------------------------------------------------------- #

def _attr(proxy):
    # completions are written by the async writer thread — flush it before the
    # read or the rollup races ahead of the queued INSERTs.
    proxy.svc._queue_db.flush(timeout=5.0)
    return proxy.svc._queue_db.cache_attribution(window_s=3600)


async def test_sync_capture_rollup_deblend_and_surfaces(proxy):
    """One proxy spin exercising the whole SYNC chain (kept lean for the
    tollgate budget): capture → per-caller rollup math → the NULL/0 distinction
    → the fleet de-blend → the /v1/fleet/cache-attribution endpoint → the
    per-endpoint cache_hit_rate on /v1/status.

    Traffic: 2 attributed vLLM chats on 'chat' (cached 6/12) + 4 NULL
    (llama.cpp shape) chats on 'companion'."""
    proxy.controller.cached_tokens = 6
    for _ in range(2):
        assert (await proxy.chat("hi", model="chat")).status_code == 200
    proxy.controller.cached_tokens = None  # llama.cpp: no counter
    for _ in range(4):
        assert (await proxy.chat("hi", model="companion")).status_code == 200

    attr = _attr(proxy)
    chat_ep = next(e for e in attr["by_endpoint"] if e["endpoint"] == "chat")
    assert chat_ep["attributed_calls"] == 2
    assert chat_ep["unattributed_calls"] == 0
    assert chat_ep["cached_tokens"] == 12             # 2 × 6
    assert chat_ep["attributable_input_tokens"] == 24  # 2 × 12
    assert chat_ep["hit_rate"] == 0.5
    # NULL (llama.cpp) rows: n/a, NOT a 0% hit — counted as unattributed.
    comp = next(e for e in attr["by_endpoint"] if e["endpoint"] == "companion")
    assert comp["attributed_calls"] == 0
    assert comp["unattributed_calls"] == 4
    assert comp["hit_rate"] is None
    # fleet de-blend: reflects ONLY the attributable population (0.5), NOT
    # 12/24 diluted by the 4 NULL rows → a non-reporting backend can't drag it.
    assert attr["fleet"]["hit_rate"] == 0.5, "NULL rows must not enter the ratio"
    assert attr["fleet"]["attributed_calls"] == 2
    assert attr["fleet"]["unattributed_calls"] == 4

    # endpoint surface: shape + the attributed rate.
    resp = await proxy.client.get("/v1/fleet/cache-attribution?window=1h")
    assert resp.status_code == 200
    j = resp.json()
    assert set(("window_s", "by_call_site", "by_endpoint", "fleet")) <= set(j)
    assert any(r["endpoint"] == "chat" and r["hit_rate"] == 0.5
               for r in j["by_endpoint"])
    for row in j["by_call_site"]:
        assert set(("call_site", "endpoint", "calls", "attributed_calls",
                    "unattributed_calls", "cached_tokens",
                    "attributable_input_tokens", "hit_rate")) <= set(row)

    # /v1/status per-endpoint cache_hit_rate after one cache-stats cycle.
    await proxy.svc._health.compute_cache_stats()
    snap = (await proxy.client.get("/v1/status")).json()["endpoints"]
    assert snap["chat"].get("cache_hit_rate") == 0.5
    # an endpoint with no ATTRIBUTABLE calls must not carry the field.
    assert "cache_hit_rate" not in snap.get("companion", {})


async def test_stream_cached_tokens_captured(proxy):
    """The streaming usage frame carries cached_tokens too → captured on the
    stream path (not just sync)."""
    proxy.controller.cached_tokens = 3   # 3/12 = 0.25
    frames = await proxy.stream_frames("a b c", model="chat")
    assert frames and any("[DONE]" in f or "stop" in f for f in frames)
    chat_ep = next(e for e in _attr(proxy)["by_endpoint"] if e["endpoint"] == "chat")
    assert chat_ep["attributed_calls"] == 1
    assert chat_ep["cached_tokens"] == 3
    assert chat_ep["hit_rate"] == 0.25
