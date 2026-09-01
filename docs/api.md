# Roadstead API specification

**Status:** v0.1 — the load-bearing contracts, verified against source on 2026-08-31. The two areas
previously marked INCOMPLETE are now filled (§3.1 and §5), and both are pinned by tests that read
this document back, so it fails the suite rather than rotting.

This document is Roadstead's **public contract** — the surface `docs/compatibility.md` marks 🔒
stable. Everything else in the package is internal and may change without notice; changing anything
here is a breaking change and needs a `CHANGELOG.md` entry, **even when the behaviour is unchanged**
(§2.2 explains why rewording an error message counts).

🚨 **It is executable, not decorative.** `tests/test_wire_contract.py`,
`tests/test_fleet_analytics_schema.py`, `tests/test_timeout_floor_contract.py` and
`tests/test_keepalive_invariant.py` read this file back and fail when the code and the document
disagree. You cannot quietly drift from it.

It was written as a *forward* contract for a cutover to the origin monorepo. That cutover was
removed from the plan on 2026-08-31; the document outlived its original purpose because a gateway
needs a published contract regardless of who is on the other end.

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

### 1.3 The keepalive ordering invariant 🚨

Both ends pool HTTP connections, and **the ordering between their idle timeouts is load-bearing.**
Whichever side expires an idle socket first is the side that closes it cleanly; if the *server* wins
the race, a client POST can land on a socket the server has already closed and fail with
`RemoteProtocolError("Server disconnected without sending a response.")` — a transport error for a
request that was never attempted.

| | value | where |
|---|---|---|
| Client `keepalive_expiry` | **4.5s** | the caller's HTTP pool (host's `_CLIENT_KEEPALIVE_EXPIRY_S`) |
| Server `timeout_keep_alive` | **30s** | `PROXY_SERVER_KEEPALIVE_S`, env `ROADSTEAD_PROXY_SERVER_KEEPALIVE_S` |

**Required: client < server, with at least 5s of margin** — enough to cover clock skew and RTT
jitter, not merely a positive difference. The client must always retire idle connections first, so
the proxy never yanks a socket a caller is about to reuse.

Raised from uvicorn's 5s default on 2026-07-06 precisely because 5s *coincided* with the client's
own expiry, making either side equally likely to close first under a concurrent burst.

⚠️ `PROXY_SERVER_KEEPALIVE_S` is env-overridable, so this invariant can be broken from a deployment
config without touching code. `tests/test_keepalive_invariant.py` pins the server side.

---

### 1.4 `GET /v1/timeout-advice` — and the floor a client must not mirror wrong

Query: `model` (**required**; role names and aliases are normalized, so `reasoner` and `tier3` are the
same endpoint), `priority` (name or number, default `P1_TURN_SUPPORT`), `est_in`, `est_out`.
Unknown `model` or `priority` → **400**, and the `model` error lists the known endpoints.

| field | type | meaning |
|---|---|---|
| `model` | string | The **normalized** endpoint, not the name you asked with. |
| `priority` | string | The resolved band name. |
| `est_in`, `est_out` | int | Echo of the query. |
| `min_ms`, `median_ms`, `p95_ms` | float | The measured latency distribution for the resolved cell. |
| `recommended_ms` | float | The recommendation, after load and size uplift. |
| `recommended_timeout_s` | int | `ceil(recommended_ms / 1000)`. **This is the number to use.** |
| `sample_count` | int | Samples behind it. `0` means the floor answered. |
| `source` | string | Which fallback level produced it — see below. |
| `surge`, `size_stretch`, `ceiling_s` | float | The uplift factors and the resolved ceiling. Observability; present only when the uplift succeeded. |

**`source` names the fallback level**, narrowest first: `cell` (this exact
priority × input-size × output-size cell) → `tier_out` → `tier` → `endpoint` → **`floor`**.

#### The floor, and why a client should ask rather than mirror

Every endpoint class has a **floor**: `recommended_timeout_s` is never below it, however thin the
evidence. A thin sample must not produce a dangerously low deadline.

🚨 **`source == "floor"` means `recommended_timeout_s` *is* the floor for that class.** That is the
supported way to learn a floor over the wire — query a cold cell and read the number. A client that
instead keeps its own copy of the floor table has taken on a **mirror it must keep in sync by hand**,
and a copy that drifts makes the client's honour-a-sub-floor-deadline decision disagree with what the
server enforces.

Two values are contract for any client that does mirror:

| | value | |
|---|---|---|
| Fallback floor, unknown endpoint class | **60.0s** | `_DEFAULT_FLOOR_S` |
| Ceiling, interactive (P0/P1/P2) | **600.0s** | |
| Ceiling, background (P3/P4) | **1800.0s** | |

The per-class floors are **not** listed here on purpose: they are deployment data, seeded from
`models.yaml` `timeout_floor_s`, and differ per fleet. Ask the endpoint.

The resolved ceiling is always lifted to at least the class floor, so a ceiling can never strangle a
call below the deadline the model already guarantees.

`tests/test_timeout_floor_contract.py` pins this side of it.

---

### 1.5 Identity — who a caller is 🚨

**A caller's identity is its `agent_id`, and that string is the DRR fair-share key, the quota holder
and the budget holder.** It is established in one of three ways, in strict precedence:

| | how | strength |
|---|---|---|
| 1 | **API key** — `Authorization: Bearer <key>` (what an OpenAI client already sends) or `X-API-Key: <key>` | authenticated |
| 2 | **Source address** — an operator registration in `ROADSTEAD_ACL` | a weak second factor: it identifies a *host*, and several callers may share one |
| 3 | **`agent_id` in the body** — `/v1/submit` only | a claim, honoured only where nothing stronger contradicts it |

A key carries its own default `priority`, an optional `min_timeout_s` deadline floor, and an
optional `admin` scope, so all four facts travel with the caller rather than with the machine it
runs on.

Three rules, each of which is a decision rather than an implementation detail:

1. 🚨 **A presented key that does not resolve is a `401 invalid_api_key`. It never falls back to the
   source address.** A wrong or revoked credential must not silently become a *different, weaker*
   identity that still works — from the outside that is indistinguishable from the credential being
   fine. Remove the header to be identified by address instead; the error message says so.
2. 🚨 **When no keys are configured, a presented key is ignored entirely** and the address decides.
   Every OpenAI client sends an `Authorization` header whether or not anybody meant it to, so
   treating one as significant before an operator has configured any key would refuse the existing
   world over a credential nobody chose.
3. 🚨 **A key overrides a body-declared `agent_id`; an address only fills in one the body omitted.**
   A verified credential is a stronger statement about who is calling than anything in the body.

**Every door is gated**, `/v1/submit` included. Out of the box, loopback and docker-internal
addresses resolve to the identity `internal` and everything else is refused — default-deny, and the
local-first case needs no configuration at all. `ROADSTEAD_REQUIRE_API_KEY=1` additionally refuses
any request that presents no key.

### 1.6 Spend, spill, and what a threshold does 🚨

The identity above is also the **quota holder and the budget holder**, which is what makes the
following expressible at all.

**One admission decision, three outcomes.** For each request, at the moment it reaches the head of
its band, Roadstead decides exactly one of:

| outcome | when |
|---|---|
| **dispatch** | a local slot is free — tried first, for every caller, unconditionally |
| **spill** | local is full, the caller is opted in (`spill_ok`), inside its cap, and the endpoint declares a `spill_to` whose target is healthy and fits the request |
| **defer** | anything else — the request stays queued and is served locally when a slot frees |

Spill is **overflow, not a fallback tier**: it is never the first answer, so local capacity is never
bypassed while it has room, and it never chains (a spilled request does not spill again).

**Two kinds of money, never summed.** A local endpoint is priced at what renting the same class of
model *would* have cost — a saving, reported as `avoided_usd`. A remote provider's published price
is an invoice, reported as `spent_usd`. Only the second counts against a threshold; a threshold that
counted the first would throttle a caller for using capacity that is free.

🚨 **Thresholds DEGRADE. They never reject.** Crossing `daily_spend_usd` costs a caller exactly two
things: **one priority band** (floored at the lowest, however far over it is) and **access to paid
spill**. It never costs local capacity, and **no error code exists for it** — the absence from §2.1
is the contract, not an omission. Two reasons: admission control is about capacity rather than
billing, and a misconfigured quota must not be able to take a caller offline. A runaway caller stays
bounded by what is free and by DRR fairness.

A caller cannot observe its own demotion in a response; it is reported to the operator through the
degradation seam (§5) once per caller per day, and shown on `/v1/status` under `spend`.


## 2. Error contract 🚨

### 2.1 Codes

Every error envelope carries a machine-readable `code`. **Fifteen exist** (a common under-count is
eight):

`backpressure` · `circuit_open` · `draining` · `unknown_endpoint` · `invalid_grammar` ·
`proxy_timeout` · `backend_error` · `context_overflow` · `access_denied` · `invalid_api_key` ·
`invalid_messages` · `invalid_request_error` · `vision_not_supported` · `on_demand_unavailable` ·
`structured_invalid_json`

`access_denied` (403) and `invalid_api_key` (401) are **not interchangeable** and a client should not
collapse them: the first says *this source is not enrolled*, the second says *this credential is
wrong*, and they send an operator to different files. On the OpenAI door `invalid_api_key` rides
OpenAI's own envelope for a rejected credential — `type: invalid_request_error`, `code:
invalid_api_key` — while `access_denied` keeps `type: access_denied`, which is what that door has
always emitted.

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

**Admin is narrower than inference, deliberately.** An operator who enrols a subnet for inference has
said nothing about who may pause a backend fleet-wide. Two ways in: an **API key with the `admin`
scope** (from anywhere), or a **source address** in the admin nets — loopback and docker-internal by
default, extended by `ROADSTEAD_ADMIN_NETS`.

🚨 **An authenticated non-admin identity is refused even from a host in the admin nets.** Once a
caller says who it is, its privileges are that identity's; inheriting the host's would mean a scoped
key could only ever widen access and never narrow it, which makes it worthless on the machine it
runs on.

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

### 3.1 `/v1/fleet/*` analytics — response schemas

Chased to column level 2026-08-31. Every field below is pinned by
`tests/test_fleet_analytics_schema.py`, which drives the real producers against a seeded
`queue.db` and reads **this section** back — so an added, renamed or dropped field fails the suite
rather than silently breaking a dashboard.

All three run **off the event loop** via `asyncio.to_thread` (§5c): a heavy `GROUP BY` over the
whole-fleet completions table must never stall scheduling under a hot dashboard.

🚨 **These read `proxy_completions`, so every window is bounded by `completions_retention_s`**
(default 30 days). "Total" means *total retained*, not total ever.

#### `GET /v1/fleet/activity` → `fleet_activity(window_s, bin_s)`

Query: `window` (default `24h`, clamped to 24h), `bin` (defaults from the window).

| field | type | meaning |
|---|---|---|
| `window_s` | int | Echo of the resolved window. |
| `bin_s` | int | Echo of the resolved bin width. |
| `now` | float | Server wall-clock at computation. ⚠️ **Absent when the DB is unopened** — see the note below. |
| `calls` | list | One entry per time bin, ascending by `ts`. |
| `by_endpoint_1h` | list | Per-endpoint breakdown over the last hour, by descending `n`, **capped at 16 rows**. |

`calls[]`:

| field | type | meaning |
|---|---|---|
| `ts` | int | Bin start, epoch seconds (floor of `completed_at` to `bin_s`). |
| `n` | int | Completions in the bin. |
| `fails` | int | Of those, `status != 'ok'`. |
| `tokens_in` | int | Summed `input_tokens`, nulls as 0. |
| `tokens_out` | int | Summed `output_tokens`, nulls as 0. |
| `p95` | float | p95 of `duration_s * 1000`, **milliseconds**, 1dp. `0.0` when the bin has no latencies — not null. |

`by_endpoint_1h[]`:

| field | type | meaning |
|---|---|---|
| `endpoint` | string | Endpoint class as persisted. |
| `n` | int | Completions in the last hour. |
| `fails` | int | Of those, `status != 'ok'`. |
| `p95` | float | Milliseconds, 1dp. |

#### `GET /v1/fleet/savings` → `savings_summary(today_start)`

Query: `since` (epoch seconds; digits only, else the local midnight is used).

**Cloud-equivalent cost avoided by running locally** — money *not spent*, not money spent. The proxy
only sees local traffic.

| field | type | meaning |
|---|---|---|
| `today_usd` | float | Summed across endpoints, 2dp. |
| `total_usd` | float | All retained completions, 2dp. |
| `today_tokens_in` | int | |
| `today_tokens_out` | int | |
| `total_tokens_in` | int | |
| `total_tokens_out` | int | |
| `today_start` | int | The boundary actually used — echo it rather than recomputing midnight client-side. |
| `by_endpoint` | list | Descending by `total_usd`. |

`by_endpoint[]`:

| field | type | meaning |
|---|---|---|
| `endpoint` | string | |
| `today_usd` | float | 4dp — the per-row precision is finer than the 2dp totals. |
| `total_usd` | float | 4dp. |
| `tokens_in` | int | **Total** retained, not today. |
| `tokens_out` | int | **Total** retained, not today. |

#### `GET /v1/usage` → `usage_rollup(dimension, hours)`

Query: `by` ∈ `agent` \| `call_site` \| `endpoint` \| `provider` (anything else → `agent`);
`hours` (default 24, **capped at 168**).

The HTTP envelope wraps the rows:

| field | type | meaning |
|---|---|---|
| `dimension` | string | The resolved `by` value. |
| `hours` | float | The resolved, capped window. |
| `rows` | list | Descending by `requests`. |

`rows[]`:

| field | type | meaning |
|---|---|---|
| `key` | string | The dimension value. Falls back to **`"—"` (em dash)** on a NULL — *not* `null` and *not* `""`. Defensive only: `agent_id`, `endpoint` and `call_site` are all `NOT NULL` in `proxy_completions`, so nothing written through `persist_complete` / `persist_external_call` can reach it. It becomes load-bearing the day a migration makes one of them nullable. |
| `requests` | int | |
| `ok` | int | |
| `errors` | int | `requests - ok`. |
| `tokens_in` | int | |
| `tokens_out` | int | |
| `cost_usd` | float | 4dp. Cloud-equivalent **avoided**. Grouped by `(dim, endpoint)` and summed up, so a caller spanning endpoints is charged each endpoint's own rate. |
| `p50_ms` | float | 1dp, over `duration_s * 1000 + queue_wait_ms` — **includes queue wait**, unlike `fleet_activity`'s `p95`. |
| `p95_ms` | float | 1dp, same basis. |

#### ⚠️ The unopened-DB shapes are narrower

With no DB connection each producer returns an early-out that is **not** the full shape:

| producer | degraded shape |
|---|---|
| `fleet_activity` | `window_s`, `bin_s`, `calls`, `by_endpoint_1h` — **no `now`** |
| `savings_summary` | `today_usd`, `total_usd`, `by_endpoint` — **no token totals, no `today_start`** |
| `usage_rollup` | `[]` |

A consumer that assumes `now` or `today_start` is always present will `KeyError` rather than degrade.
Documented because it is easy to hit in a test double and never in production.

---

## 4. South face — what Roadstead requires *of a backend*

**This is the portability contract, and it did not exist before this document.** It is what an
inference engine must expose for Roadstead to drive it.

**A *provider* is the adapter that meets one kind of backend on its terms** (`roadstead/providers/`).
Three exist: `llama.cpp`, `vllm`, and `openrouter`. What follows is what each requires and publishes;
a provider's `ProviderDescriptor` is the machine-readable form of the same thing, and the code
branches on that rather than on an engine name.

### 4.1 Dispatch

| Purpose | Method + path on the backend |
|---|---|
| Chat | `POST /v1/chat/completions` |
| Embeddings | `POST /embed` — 🚨 **not** `/v1/embeddings` |
| Rerank | `POST /rerank` |

Embed and rerank are FastAPI-shaped shims, not OpenAI-compatible, and have no OpenAI route.

**Remote providers differ in where the routes are and who may call them.** An endpoint may declare a
`base_url` (superseding `host`/`port`, which cannot express a scheme or a base path), and its
provider supplies its own auth headers — OpenRouter's chat route is `POST {base_url}/chat/completions`
with a bearer token, the key read at request time from the environment variable named by
`api_key_env`. Roadstead never holds a key in config.

🚨 **A provider that cannot honour a request refuses it**, and the call fails as a `400` naming the
provider. This applies to a caller **constraint** it cannot enforce (a GBNF grammar sent to
OpenRouter) and to a payload type it has no route for. It does *not* apply to engine **hints** the
proxy itself added (`id_slot`, `chat_template_kwargs`, `thinking_token_budget`), which are dropped
silently because no caller asked for them. The distinction matters because a dropped constraint is
invisible: the caller cannot tell unconstrained output from a model that answered badly.

### 4.2 Streaming

Standard OpenAI SSE. Two backend behaviours are load-bearing:

- **A terminal `[DONE]`** is treated as the backend *asserting completeness*. It is the sole
  discriminator that licenses synthesising a missing `finish_reason` chunk. A stream that ends
  without `[DONE]` and without `finish_reason` is treated as a **real truncation** and is not
  repaired.
- `finish_reason` conventionally rides **alone** on a final chunk whose `delta` is `{}`.

### 4.3 Capacity discovery — asymmetric by provider

| | llama.cpp | vLLM | OpenRouter |
|---|---|---|---|
| Probe | `GET /props` | `GET /v1/models` | `GET {base_url}/models` |
| Slot count | ✅ `n_parallel` / `total_slots` / `len(slots)` | ❌ **not exposed** | ❌ **none exists** |
| Per-request context | ✅ `n_ctx` (already per-slot in current builds) | via `max_model_len` | via `context_length` |
| Served model | ✅ one, named on `/v1/models` | ✅ one (alias; weights via `root`) | ❌ a catalogue — the model is config |
| Token costs | ❌ | ❌ | ✅ `pricing` (per token, in dollars, as **strings**) |
| Consequence | discovered at runtime | **concurrency stays config-seeded**, with a drift alert | **concurrency is a policy cap we choose** |

🚨 A vLLM-shaped backend cannot have its concurrency discovered. This is a property of the engine,
not a gap in Roadstead. A known consequence is configured-vs-actual slot drift.

🚨 **Remote capacity is not local capacity.** A remote provider has no occupancy to publish — what it
sells is money and rate limit — so nothing reports slots and the endpoint's concurrency stays a
deliberate cap on *our own* traffic. Slot-seconds remain the unit of fairness because *local* slots
are the scarce thing.

**Published prices are read on every discovery pass** and feed the spend accounting in §1.6. Three
things about that column that are easy to get wrong: the values are **strings** (a price like
`0.0000005` is where a JSON float starts losing digits), they are **per single token** (scaled to
per-million on the way in), and **a price of zero is a real price**. A free model on a remote
provider is still a remote model — reading its zero as *unpublished* would push it onto the imputed
avoided-cost table and book a *saving* for a call made over the internet. An operator-declared price
in the catalog beats a published one, because somebody who wrote a number down knows something the
catalogue does not.

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

A remote provider's discovery probe doubles as its liveness signal, and a credential it cannot
resolve reads as *cannot tell* — so a misconfigured remote endpoint fails its health probe and stops
being dispatched to, rather than failing live traffic one call at a time.

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
the error codes, the deferrability mechanism and its marker substrings, the context-overflow marker
verbatim, backend dispatch paths, the capacity-discovery asymmetry, and the metric-name prefixes.

**Updated 2026-09-01:** §1.5 (identity and API keys) is new, `invalid_api_key` joins §2.1, and §3
restates the admin gate now that a key can carry the scope — Workstream B in `docs/roadmap.md`.
§1.6 (spend, spill and the degrading threshold) and the price row in §4.3 are Workstream D. 🚨 §1.6
adds **no error code**: that a spend threshold cannot produce one is the contract.

**Previously INCOMPLETE — both closed 2026-08-31:**
1. ~~Nested response schemas for `/v1/fleet/*` analytics.~~ Chased to column level in §3.1 and
   pinned by `tests/test_fleet_analytics_schema.py`, which reads this document back.
2. ~~Delegator-signature drift between `service.py` and `http_handlers.py`.~~ Audited by AST:
   **30 delegators, 30 identical signatures, zero drift** — the stated contract holds. Now
   continuously enforced by `tests/test_delegator_signatures.py` rather than re-asserted by hand.
