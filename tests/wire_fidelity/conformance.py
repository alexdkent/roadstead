"""The south-face wire contract, as executable assertions.

`docs/api.md` §4 says in prose what Roadstead requires *of a backend*. This says
it in code, against a live HTTP endpoint — and the same assertions run twice:

  * against `roadstead.testing` on every suite run (fast, default path);
  * against a **real** `llama-server` when one is reachable (`-m wire_fidelity`).

That pairing is the point. The fake backend is not just a convenience — it is a
**claim about how real engines behave**, and every test that trusts it inherits
that claim. Running one set of assertions against both is where the claim gets
audited, so the day an engine changes its `/props` shape the failure lands here
instead of in production capacity discovery.

Every assertion below traces to a line in `roadstead/health.py` or
`roadstead/backend.py` that actually reads the field. Nothing is asserted
because it looks like part of the protocol; if Roadstead does not read it, it is
not in here.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Liveness — backend.probe_health
# ---------------------------------------------------------------------------

def check_health(base_url: str) -> None:
    r = httpx.get(f"{base_url}/health", timeout=TIMEOUT)
    assert r.status_code == 200, f"/health returned {r.status_code}: {r.text[:200]}"


# ---------------------------------------------------------------------------
# /v1/models — backend.probe_models, probe_vllm_capacity, probe_model_fingerprint
# ---------------------------------------------------------------------------

def check_models(base_url: str, *, expect_max_model_len: bool) -> dict[str, Any]:
    """Returns `data[0]`. `expect_max_model_len` only for vLLM — llama.cpp does
    not publish it, which is why vLLM concurrency stays config-seeded."""
    r = httpx.get(f"{base_url}/v1/models", timeout=TIMEOUT)
    assert r.status_code == 200, f"/v1/models returned {r.status_code}"
    body = r.json()
    data = body.get("data")
    assert isinstance(data, list) and data, f"/v1/models has no data[]: {body}"
    entry = data[0]
    assert isinstance(entry, dict), f"data[0] is not an object: {entry!r}"

    # probe_models reads exactly this, and returns None on anything else.
    model_id = entry.get("id")
    assert isinstance(model_id, str) and model_id, (
        f"data[0].id must be a non-empty string (probe_models returns None "
        f"otherwise, and the endpoint keeps its pre-discovery wire id): {entry!r}")

    if expect_max_model_len:
        mlen = entry.get("max_model_len")
        assert isinstance(mlen, int) and mlen > 0, (
            f"vLLM must publish data[0].max_model_len as a positive int — it is "
            f"the ONLY capacity fact vLLM exposes, and probe_vllm_capacity "
            f"returns None without it: {entry!r}")
    return entry


# ---------------------------------------------------------------------------
# /props — llama.cpp capacity discovery (health.apply_discovered_capacity)
# ---------------------------------------------------------------------------

def check_props_shape(base_url: str) -> dict[str, Any]:
    r = httpx.get(f"{base_url}/props", timeout=TIMEOUT)
    assert r.status_code == 200, f"/props returned {r.status_code}: {r.text[:200]}"
    props = r.json()
    assert isinstance(props, dict), f"/props is not an object: {props!r}"
    return props


def discovered_slots(props: dict[str, Any]) -> int:
    """Reproduce health.apply_discovered_capacity's slot resolution, in order.

    Kept as a copy rather than importing the private logic, deliberately: this
    file is a statement of the WIRE contract, and importing the reader would
    make it agree with itself by construction.
    """
    gen = props.get("default_generation_settings") or {}
    n = gen.get("n_parallel")
    if n is None:
        n = props.get("total_slots")
    if n is None:
        slots = props.get("slots")
        if isinstance(slots, list):
            n = len(slots)
    assert isinstance(n, int) and n >= 1, (
        f"no slot count discoverable from /props via n_parallel, total_slots "
        f"or len(slots) — capacity discovery silently keeps the CONFIGURED "
        f"value, which is the thing runtime discovery exists to replace: "
        f"{json.dumps(props)[:400]}")
    return n


def discovered_context_per_slot(props: dict[str, Any]) -> tuple[int, str]:
    """Returns (per-slot context, which field it came from).

    🚨 The units differ between the two fields, and getting it wrong quartered
    every multi-slot endpoint's context once already:

      * `default_generation_settings.n_ctx` is ALREADY per-slot (confirmed live:
        `--ctx-size 131072 --parallel 4` reports 32768 there, not 131072);
      * top-level `props["n_ctx"]` is treated as an aggregate and DIVIDED by the
        slot count.
    """
    gen = props.get("default_generation_settings") or {}
    gen_n_ctx = gen.get("n_ctx")
    if gen_n_ctx:
        assert isinstance(gen_n_ctx, int) and gen_n_ctx > 0, (
            f"default_generation_settings.n_ctx must be a positive int: {gen_n_ctx!r}")
        return gen_n_ctx, "default_generation_settings.n_ctx"
    top = props.get("n_ctx")
    assert isinstance(top, int) and top > 0, (
        "neither default_generation_settings.n_ctx nor a top-level n_ctx is "
        "usable — the context gate would run on a stale configured value")
    return top // discovered_slots(props), "n_ctx (aggregate, divided)"


# ---------------------------------------------------------------------------
# Streaming — the terminal-chunk rule (correction.py; docs/ledger.md)
# ---------------------------------------------------------------------------

TERMINAL_ALONE = "finish_reason on its own chunk with an empty delta"
TERMINAL_WITH_CONTENT = "finish_reason on the same chunk as content"


def check_stream_terminal_shape(base_url: str, model: str) -> str:
    """Which of the two known terminal-chunk shapes does this backend emit?

    A backend that ends a stream with neither a `finish_reason` nor a `[DONE]`
    is a REAL truncation and nothing may be synthesized — the distinction the
    repair layer must never collapse (`docs/ledger.md`: a client burned a second
    model call on 47 turns that were already complete).

    Returns the shape name, so the caller can record it rather than this file
    deciding which one is "right": both are handled, a third would not be.
    """
    frames: list[str] = []
    with httpx.stream(
        "POST", f"{base_url}/v1/chat/completions",
        json={"model": model, "stream": True, "max_tokens": 16,
              "messages": [{"role": "user", "content": "Count to three."}]},
        timeout=TIMEOUT,
    ) as resp:
        assert resp.status_code == 200, f"stream returned {resp.status_code}"
        for line in resp.iter_lines():
            line = line.strip()
            if line.startswith("data: "):
                frames.append(line[len("data: "):])

    assert frames, "the stream produced no data: frames at all"
    assert frames[-1] == "[DONE]", (
        f"stream did not end with the [DONE] sentinel (last frame: "
        f"{frames[-1][:120]!r}). Without it a missing finish_reason is a REAL "
        f"truncation and the proxy must NOT synthesize a terminal chunk.")

    chunks = [json.loads(f) for f in frames[:-1]]
    finishing = [
        c for c in chunks
        if (c.get("choices") or [{}])[0].get("finish_reason") is not None
    ]
    assert len(finishing) == 1, (
        f"expected exactly one chunk carrying finish_reason, found "
        f"{len(finishing)} — the proxy's terminal-chunk repair assumes one")

    delta = (finishing[0].get("choices") or [{}])[0].get("delta") or {}
    shape = TERMINAL_ALONE if not delta.get("content") else TERMINAL_WITH_CONTENT
    return shape


def check_sync_completion(base_url: str, model: str) -> dict[str, Any]:
    """The non-streamed shape: choices[0].message.content plus finish_reason."""
    r = httpx.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "max_tokens": 16,
              "messages": [{"role": "user", "content": "Say hello."}]},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200, f"completion returned {r.status_code}: {r.text[:200]}"
    body = r.json()
    choices = body.get("choices")
    assert isinstance(choices, list) and choices, f"no choices[]: {body}"
    msg = choices[0].get("message") or {}
    assert isinstance(msg.get("content"), str), (
        f"choices[0].message.content must be a string: {choices[0]!r}")
    assert choices[0].get("finish_reason"), (
        "choices[0].finish_reason is missing — a completion with no completion "
        "signal is exactly the failure the correction layer exists to catch")
    return body
