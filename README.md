# Roadstead

> A **roadstead** is the sheltered anchorage outside a harbour where vessels wait for a berth to
> free up.

Capacity-aware admission control for self-hosted LLM inference fleets — several llama.cpp servers
and a vLLM tensor-parallel pair across heterogeneous hardware, **without Kubernetes**.
OpenAI-compatible on the front, model-authoritative on the back.

> ⚠️ **Pre-release, and private.** Extracted from a production monorepo on 2026-08-31 and still
> stabilising. It is an independent project rather than a replacement for its origin — expect it to
> become a superset, and expect occasional deliberate breaks (`docs/compatibility.md`). This
> repository must not be made public until the scrub in `docs/corpus_and_scrub_plan.md` is complete.

## Why it exists

Most LLM gateways treat a backend as an opaque endpoint with a health bit. They load-balance across
it, retry on failure, and take a timeout as a number the caller supplies. That works when the
backend is an elastic cloud API.

It does not work when the backend is a llama.cpp server with exactly four slots and a fixed
per-slot context, sitting on a GPU you own. There, the interesting question is not *which* backend
to send to — it is **whether to send at all right now, whose request goes first, and how long it is
reasonable to wait.**

Roadstead answers that question by measuring rather than assuming:

- **It discovers capacity.** It probes llama.cpp `/props` for real slot counts and per-slot context
  size, and gates admission against what is actually there.
- **It fair-shares in slot-seconds.** Deficit round-robin across callers within three strict
  priority bands, where the unit of fairness is *backend occupancy time* — EWMA-calibrated per
  endpoint, charged up front and retroactively corrected. Request counts and token counts both
  mis-price a backend with fixed slots; occupancy time does not.
- **It computes deadlines instead of accepting them.** Callers declare priority and interactivity;
  Roadstead derives the timeout from a latency distribution it learned from its own traffic,
  conditioned on endpoint, tier and request size.
- **It repairs what comes back.** Grammar validation and repair, a JSON-schema retry backstop,
  degeneration detection, truncation integrity, and repair of malformed SSE — including the missing
  terminal `finish_reason` chunk that no client notices until it burns a second model call.

A survey of ~25 open-source gateways (`docs/evaluation.md`) found none that does all of this
outside Kubernetes, and none at all that does the last two.

## Status

| Phase | |
|---|---|
| 0 · Sever host-application imports | ✅ done |
| 1 · Standalone repo, namespace, packaging | ✅ done |
| 2 · Standalone test harness | ✅ done |
| 3 · Soak + hardening | 🔨 next |
| 4 · Scrub, then publish | gated on the scrub |

Parity against the origin copy, and the cutover it existed to make safe, were **removed from the
plan on 2026-08-31** when the project stopped being tied to its origin. A parity gate on a
deliberate superset fails on every improvement.

Plan and current state: **`docs/handoff.md`**.

## Quick start

```sh
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

The suite needs **no fleet, no network and no inference backend** — it runs against
`roadstead.testing`, described below.

## A programmable backend, in the box

`roadstead.testing` ships the fake backend the suite runs on, because for a gateway whose thesis is
capacity-aware admission control, *a backend that lies about its capacity on demand* is a capability
rather than test scaffolding — and it is not something you can ask a real GPU for.

```python
from roadstead.testing import FakeBackend, FakeBackendServer, FAULT_CAPACITY_DESYNC

server = FakeBackendServer(FakeBackend(engine="vllm")).start()
server.controller.set_fault(FAULT_CAPACITY_DESYNC, 2)   # accepts 2, 503s the rest,
                                                        # while /props claims otherwise
```

It is a real Starlette app under real uvicorn on a real socket — so real `httpx` and real SSE framing
are exercised, not a mock transport. It speaks both llama.cpp and vLLM wire shapes across
`/v1/chat/completions`, `/embed`, `/rerank`, `/props`, `/v1/models`, `/metrics` and `/health`, and
serves twenty-odd south-face pathologies on command: truncated and invalid JSON, empty completions,
degenerate repetition, schema violations, phantom and truncated tool calls, partial and interleaved
SSE frames, TTFT and inter-token stalls, mid-stream resets, and capacity desync.

It is also the executable form of §4 of `docs/api.md` — what Roadstead requires *of a backend*.

## Documentation

| | |
|---|---|
| `CLAUDE.md` | Orientation, the concurrency invariant, and the engine-behaviour findings that explain why the code is shaped the way it is. **Read before changing anything.** |
| `docs/api.md` | The four API surfaces: north face, error contract, admin/control plane, and what Roadstead requires *of a backend*. |
| `docs/handoff.md` | The extraction plan, current state, and what to do next. |
| `docs/evaluation.md` | Why this exists rather than adopting something else — the field survey and decision record. |
| `docs/corpus_and_scrub_plan.md` | What must be scrubbed before this can go public, and why the working tree is not enough. |
| `docs/ledger.md` | Defects that came back, with the guard that now prevents each. |

## Licence

Apache-2.0.
