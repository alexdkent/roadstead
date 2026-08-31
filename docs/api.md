# Roadstead API specification

**Status:** v0.1 — the load-bearing contracts, verified against source on 2026-08-31. Two areas are
marked **INCOMPLETE** below and need filling before anyone builds against this.

This document is the **shared boundary object** between Roadstead and any host application. During
the dual-track period (see `handoff.md`) it is a *forward* contract: it records what must not drift
on either side so that a future cutover stays possible. It is not describing a live integration —
the monorepo currently runs its own in-situ copy and does not call this package.

> 🚨 **The most important thing in this document is §2.** The proxy's error *codes* are not the
> contract that callers actually depend on — the human-readable **marker substrings** are, because
> the client-side classifier matches on them. Changing an error message can therefore break a caller
> even when the `code` is unchanged.

---

## 1. North face — the client-facing surface

Routes are registered in one place: `routes.make_routes()` (`routes.py:138-183`). There is no legacy
duplicate table.

### 1.1 Request fields beyond the OpenAI API

A caller may send these alongside standard OpenAI fields. Each is optional.

| Field | Type | Effect |
|---|---|---|
| `priority` | enum | **The only tier declaration.** Drives band selection (interactive / foreground / background), DRR accounting and the timeout ceiling. |
| `timeout_s` | float | Caller's own deadline. Omitted → the smart default applies (a computed recommendation, see §1.2). |
| `thinking` | bool \| int | Opt into reasoning. `true` grants a flat headroom on `max_tokens`; an int gives an explicit, usually smaller, headroom — the form an interactive caller wants. |
| `grammar` | string | GBNF. Validated and repaired pre-enqueue; invalid grammar fails loud with 422. |
| `agent_id` | string | Caller identity — **the DRR fair-share key.** Absent → the caller is not individually accounted. |
| `call_site` | string | Sub-identity for attribution and cache analytics. |
| `session_id`, `turn_id`, `caller_id`, `request_id` | string | Correlation only. |

Headers: **`X-Timeout-S`** (alternative to the body field) and **`X-Request-ID`** (forwarded).

🚨 **There is no caller-sendable `tier` field.** `priority` is the only tier declaration — a
frequent wrong assumption.

### 1.2 Timeouts are computed, not accepted

Callers declare *intent*; Roadstead owns the number. When a caller supplies no `timeout_s`, the
smart default applies a recommendation derived from the learned latency distribution, conditioned on
`(endpoint, tier, input-size bucket, output-size bucket)`, as `max(p99 × margin, floor)` under
load-aware and context-length-aware multipliers and clamped to a per-class ceiling.

A supplied `timeout_s` is honoured but is **extend-only** on the client side: never lowered below the
caller's value, raised toward the recommendation when the model/tier/size needs more.

Deadlines are **soft** for streaming: the streaming path may extend a deadline while the backend is
demonstrably still making decode progress, rather than killing work that is advancing.

---

## 2. Error contract 🚨

### 2.1 Codes

Every error envelope carries a machine-readable `code`. **Fourteen exist** (a common under-count is
eight):

`backpressure` · `circuit_open` · `draining` · `unknown_endpoint` · `invalid_grammar` ·
`proxy_timeout` · `backend_error` · `context_overflow` · `access_denied` · `invalid_messages` ·
`invalid_request_error` · `vision_not_supported` · `on_demand_unavailable` · `structured_invalid_json`

### 2.2 The marker substrings are the real contract

**The deferrable-vs-non-deferrable classification does not live in the proxy.** Callers classify by
matching **substrings of the error message** — historically in the host's
`framework/nexus_errors.py`. Matched markers include **`circuit open`** and **`backpressure`**.

Two consequences, both easy to get wrong:

1. **Rewording an error message is a breaking API change**, even when the `code` is untouched.
2. Any client library Roadstead ships should classify on `code`, not on prose — but it must keep
   emitting the legacy substrings until every existing caller has migrated.

The **context-overflow marker is verbatim**:

```
exceeds the available context size
```

It is load-bearing in two places — the proxy emits it, and a client-side helper matches it to decide
whether to chunk. Do not reword it.

### 2.3 Broad shape

Deterministic 4xx are non-deferrable (the caller must change something). 429 and 5xx are deferrable
(retry or defer is correct). The authoritative mapping is the client-side matcher above.

---

## 3. Admin / control plane

**ACL: loopback and docker-internal only** (plus an explicit allow-list). These are not exposed to
the LAN inference door.

| Route | Purpose |
|---|---|
| `POST /v1/admin/endpoints/{ep}/pause` | Drain an endpoint: background defers, interactive fast-fails, the poller stops probing. Auto-opens an annotated PLANNED maintenance window. |
| `POST /v1/admin/endpoints/{ep}/resume` | Re-probe, **re-discover capacity**, drain the deferred queue, close the window. |
| `GET`/`POST /v1/admin/flags` | Read/flip runtime flags; persisted to JSON, survives restart. |
| `GET`/`POST /v1/admin/maintenance` | List, or backdate a closed window for a restart done without draining. |

Open surfaces: `GET /v1/status` (per-endpoint health, capacity, reliability counters),
`GET /v1/timeouts`, `GET /metrics` (Prometheus), `GET /health`, `GET /readyz` (fails closed on
readiness-critical endpoints), and the `/v1/fleet/*` analytics family.

🚨 **`/v1/status` and `/metrics` have external consumers** — in the origin fleet a gateway, a
ground-truth verifier and a web UI all read them. Treat their top-level key names as public API.

**INCOMPLETE:** the nested response schemas for the `/v1/fleet/*` analytics routes (`fleet_activity`,
`savings_summary`, `usage_rollup`) were not chased to column level. Fill before publishing.

---

## 4. South face — what Roadstead requires *of a backend*

**This is the portability contract, and it did not exist before this document.** It is what an
inference engine must expose for Roadstead to drive it.

### 4.1 Dispatch

| Purpose | Method + path on the backend |
|---|---|
| Chat | `POST /v1/chat/completions` |
| Embeddings | `POST /embed` — 🚨 **not** `/v1/embeddings` |
| Rerank | `POST /rerank` |

Embed and rerank are FastAPI-shaped shims, not OpenAI-compatible, and have no OpenAI route.

### 4.2 Streaming

Standard OpenAI SSE. Two backend behaviours are load-bearing:

- **A terminal `[DONE]`** is treated as the backend *asserting completeness*. It is the sole
  discriminator that licenses synthesising a missing `finish_reason` chunk. A stream that ends
  without `[DONE]` and without `finish_reason` is treated as a **real truncation** and is not
  repaired.
- `finish_reason` conventionally rides **alone** on a final chunk whose `delta` is `{}`.

### 4.3 Capacity discovery — asymmetric by engine

| | llama.cpp | vLLM |
|---|---|---|
| Probe | `GET /props` | `GET /v1/models` |
| Slot count | ✅ `n_parallel` / `total_slots` / `len(slots)` | ❌ **not exposed** |
| Per-slot context | ✅ `n_ctx` (already per-slot in current builds) | via `max_model_len` |
| Consequence | discovered at runtime | **concurrency stays config-seeded**, with a drift alert |

🚨 A vLLM-shaped backend cannot have its concurrency discovered. This is a property of the engine,
not a gap in Roadstead. A known consequence is configured-vs-actual slot drift.

### 4.4 Optional signals and what degrades without them

| Signal | Absent → |
|---|---|
| `usage.prompt_tokens_details.cached_tokens` | per-request cache attribution falls back to an endpoint-level rate |
| `GET /metrics` (see prefixes below) | no prefix-cache visibility; treated as *n/a*, never as 0% |
| decode-progress counters | streaming deadlines cannot be extended on progress; the soft deadline becomes hard |

**Load-bearing metric name prefixes** (matched by prefix):
`vllm:prefix_cache_hits_total` · `vllm:prefix_cache_queries_total` · `vllm:prompt_tokens_total` ·
`vllm:generation_tokens_total` · `llamacpp:prompt_tokens_total` · `llamacpp:n_decode_total`

### 4.5 Health

A liveness endpoint, plus the discovery probes above. Backends that are FastAPI shims with no
`/props` or `/v1/models` must be declared `skip_discovery` or they generate continuous 404 noise.

---

## 5. The integration seam (host applications)

`roadstead.hooks` is the only place the package reports outward. A host may register a degradation
sink matching:

```python
def sink(*, component: str, reason: str, impact: str, **fields) -> None: ...
```

Default is a WARNING log in a grep-able shape. Everything else in the package reports through it.

---

## 6. Verification status

**Verified against source 2026-08-31:** the route table location, the non-standard request fields,
the fourteen error codes, the deferrability mechanism and its marker substrings, the
context-overflow marker verbatim, backend dispatch paths, the capacity-discovery asymmetry, and the
metric-name prefixes.

**Not verified / INCOMPLETE:**
1. Nested response schemas for `/v1/fleet/*` analytics (§3).
2. Delegator-signature drift between `service.py` and `http_handlers.py` — the audit relied on a
   stated identical-signature contract without opening `service.py`. Spot-check before publishing.
