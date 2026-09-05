# Roadstead API specification

**Status:** v0.1 — the load-bearing contracts, verified against source on 2026-08-31. The two areas
previously marked INCOMPLETE are now filled (§3.6 and §5), and both are pinned by tests that read
this document back, so it fails the suite rather than rotting.

This document is Roadstead's **public contract** — the surface `docs/compatibility.md` marks 🔒
stable. Everything else in the package is internal and may change without notice; changing anything
here is a breaking change and needs a `CHANGELOG.md` entry, **even when the behaviour is unchanged**
(§2.2 explains why rewording an error message counts).

🚨 **It is executable, not decorative.** `tests/test_wire_contract.py`,
`tests/test_fleet_analytics_schema.py`, `tests/test_timeout_floor_contract.py`,
`tests/test_keepalive_invariant.py`, `tests/test_spend.py` and `tests/test_client_sdk.py` read this
file back and fail when the code and the document disagree. You cannot quietly drift from it.

It was written as a *forward* contract for a cutover to the origin monorepo. That cutover was
removed from the plan on 2026-08-31; the document outlived its original purpose because a gateway
needs a published contract regardless of who is on the other end.

> 🚨 **The most important thing in this document is §2.** The proxy's error *codes* are not the
> contract that callers actually depend on — the human-readable **marker substrings** are, because
> the client-side classifier matches on them. Changing an error message can therefore break a caller
> even when the `code` is unchanged.

---

## 1. North face — the client-facing surface

Routes are registered in one place: `routes.make_routes()`. There is no legacy duplicate table.

**There are two north faces.** `/v1/*` is OpenAI-compatible and strictly so; `/rs/v1/*` is the
enriched Roadstead API (§1.7), with its own version because the other one is versioned by OpenAI.
§1.1–§1.4 below describe the OpenAI door.

### 1.1 Request fields beyond the OpenAI API — one table per DOOR 🚨

**Rewritten 2026-09-02, and the rewrite is the point.** This section used to be a single table, and
it documented the **union of two doors** as though it were one. `caller_id` and `request_id` really
are read — on `/rs/v1/chat` (§1.7.2) — and sat here, on a door where nothing reads them, beside
`grammar` and `thinking`, which do work here. Almost-right is worse than plainly wrong: a migrator
sending `agent_id` to `/v1/chat/completions` got no error and no effect, and never learned that the
field they wanted for fair-share attribution was on the other door.

The **Read by** column is a pin, not a footnote: `tests/test_openai_door_fields.py` checks every row
against what the named module actually reads, and fails when a row claims a field is read that no
longer is — or when a handler grows a body field this table does not list.

#### `POST /v1/chat/completions`

A caller may send these alongside standard OpenAI fields. Each is optional.

| Field | Type | Effect | Read by |
|---|---|---|---|
| `model` | string | The endpoint: a class, role or alias, normalized. Validated **before** enqueue — an unknown one is a `404 model_not_found` rather than a wasted slot and a late 502. | `http_handlers.handle_openai_chat` |
| `timeout_s` | float | Caller's own deadline. Omitted → the smart default applies (a computed recommendation, see §1.2). Popped from the body, so it never reaches the backend. | `http_handlers.handle_openai_chat` |
| `thinking` | bool \| int | Opt into reasoning. `true` grants a flat headroom on `max_tokens`; an int gives an explicit, usually smaller, headroom — the form an interactive caller wants. | `correction.py` |
| `grammar` | string | GBNF. Validated and repaired pre-enqueue; invalid grammar fails loud with 422. | `correction.py` |

Header: **`X-Timeout-S`**, read by the same handler as an alternative to the body field.

⚠️ **`X-Request-ID` is not read on the way in.** It is a header Roadstead *emits* southbound,
carrying the request id Roadstead itself minted (§4.1) — this document used to list it here as
"forwarded", which reads as "send one and we will carry it". Nothing consults an inbound one. The
enriched door's `request_id` body field (§1.7.2) is the supported way to correlate a caller's own id
with the durable record.

##### What this door does NOT read 🚨

Everything else in the body is **left in the payload and forwarded toward the backend**. For the
names below that is not what a caller carrying habits from `/v1/submit` expects, so each is stated
rather than omitted:

| Field | What actually happens here |
|---|---|
| `agent_id` | **Ignored.** Identity is the API key, or the source address — §1.5. A body could claim any `agent_id`, including one with a better DRR weight. |
| `priority` | **Ignored, and that is now a decision rather than a gap** (2026-09-02). The band is *configured* per caller instead — see "A static band per caller" below. A per-call body field was measured against real traffic and rejected. |
| `call_site` | **Ignored.** Set to `<agent_id>.openai_compat`, so this door's traffic is distinguishable in attribution. |
| `caller_id` | **Ignored.** Set to the `agent_id`. Read on `/rs/v1/chat`, which is where a caller with sub-identities should be. |
| `session_id`, `turn_id`, `request_id` | **Ignored** — and, unlike the four above, not replaced either: they travel to the backend inside the payload, where a strict engine may reject the unknown key. Read on `/rs/v1/chat`. |

**The header spellings are ignored too**, and they are listed because a fleet arriving from a proxy
that read identity out of headers will try them before it reads this table:

| Header | What actually happens here |
|---|---|
| `X-Agent-Id` | **Ignored.** Nothing reads it — not this handler, and nothing downstream. Identity is the API key or the source address (§1.5), exactly as for the body field above. |
| `X-Call-Site` | **Ignored.** `call_site` is set to `<agent_id>.openai_compat` here and `<agent_id>.openai_compat_embed` on `/v1/embeddings`, whatever you send. |

`X-Timeout-S` is the only request header either OpenAI door reads. The `X-Roadstead-*` names in §1.8
are ones Roadstead *emits*.

🚨 **There is no caller-sendable `tier` field**, on either door — a frequent wrong assumption.
`priority` is the tier declaration, and it is read on `/rs/v1/chat` (§1.7.2) only.

##### Why this door will not grow a `priority` field 🚨

Measured on a week of real traffic through the proxy this one replaces: **196k requests, 3.6% of
them through an OpenAI-shaped door.** Every one of that 3.6% was an off-box caller reached by a
third-party OpenAI client — a CLI, a bridge, a shell tool. Those are precisely the callers that
*cannot* set a non-standard body field: they use an OpenAI SDK, which is why they are on this door
at all. The other 96.4% declared `priority` explicitly, and every one of them lands on `/rs/v1/chat`
after the migration in §1.9, where it is read.

So the field would be added for a population that is empty by construction, at the cost of the one
promise `/v1/*` makes — that a strict OpenAI validator can point here without breaking. What those
callers actually need is a **standing** band, not a per-call one, and a key already carries one.

⚠️ If a per-call band is ever genuinely wanted here, the precedent is `X-Timeout-S`: a **header**,
like the deadline it parallels, never a body field. The body is OpenAI's; the headers are ours
(§1.8).

##### A static band per caller — where to configure it 🚨

A caller that cannot send a field can still be *given* a band, in configuration, and there are two
places to write one. **Whose default applies is one question with four steps, narrowest first:**

| # | source | where |
|---|---|---|
| 1 | what this **request** declared | `priority` / `interactive` — `/rs/v1/chat` only |
| 2 | what the **credential** declared | `ROADSTEAD_API_KEYS`, a keys file, or `ROADSTEAD_ACL`: `agent_id:P1_TURN_SUPPORT` (§1.5) |
| 3 | what the **agent's config** declares | `agents.yaml` → `default_priority`, keyed on `agent_id` |
| 4 | the built-in default | `P3_INGESTION` |

Step 2 sits above step 3 because a credential is the stronger statement, and an operator who scoped
a key to a band meant it. 🚨 **A credential that names NO band falls through to step 3** rather than
pinning the caller to the grammar's default — the two are different statements and the registry
records which one was made.

🚨 **Step 3 did nothing at all until 2026-09-02.** `default_priority` was parsed, allowlisted,
editable through `PATCH /rs/v1/admin/quotas` and reported by §3.4 as the caller's
`declared_priority` — while the request path read only the credential's band. The operator-facing
surface named the source that was not in force, and the two defaults did not even agree
(`P1_TURN_SUPPORT` in the config, `P3_INGESTION` in the identity grammar). It was invisible because
every door pre-filled `priority` into the internal submit body from the resolved identity, so the
identity's own default arrived indistinguishable from a band the caller had asked for.

**Which to reach for.** Step 2 when the band is a property of the *credential* — one process, one
key, one job. Step 3 when it is a property of the *caller* — and since delegation (§1.5) one key can
act as many `agent_id`s, a single band on that key cannot say "interactive for `chat-agent`, background for
`forum-agent`". `agents.yaml` can, and it is already where the DRR weights that go with those bands live.

#### `POST /v1/embeddings`

| Field | Type | Effect | Read by |
|---|---|---|---|
| `input` | string \| [string] | **Required.** The text to embed. Normalized to a list. | `http_handlers.handle_openai_embeddings` |
| `encoding_format` | string | `float` (default) or `base64`. Anything else is a `400`. | `http_handlers.handle_openai_embeddings` |
| `model` | string | **Echoed in the response and nothing else** — it does not select an endpoint. This door routes to the `embed` class unconditionally, because OpenAI's embeddings shape gives a caller no way to express a choice the gateway would honour. | `http_handlers.handle_openai_embeddings` |

🚨 **This door is LOSSY, on purpose, and must stay that way.** It is a translator in both
directions — OpenAI `{"input": …}` in, OpenAI `{object, data, usage}` out — over a hybrid embedder
that answers `{dense, sparse, colbert}`. OpenAI's schema has nowhere to put the sparse and colbert
halves, so they are dropped. "Fixing" that here would put a non-OpenAI shape behind an OpenAI URL,
which is the one thing `/v1/*` promises not to do.

**The lossless path is `POST /rs/v1/chat` with `payload_type: "embedding"`** (§1.7.2), where the
backend's body is returned whole under `response`. `roadstead.client`'s `embed()` sends exactly that.

#### `POST /rs/v1/chat` and `POST /rs/v1/plan`

The enriched door's fields are §1.7.1 (what you want) and §1.7.2 (how urgently, and the payload).
`priority`, `caller_id`, `request_id`, `session_id`, `turn_id` and `call_site` are all read there.

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

#### `GET /v1/timeouts` → `timeouts_report(hours)`

**Which calls gave up instead of finishing, and at which bound.** Open (§3 lists it among the open
surfaces): it names endpoints, tiers and callers, never a payload. Its companion
`/v1/timeout-advice/shadow-report` below asks the other half of the question — not *who gave up*, but
*would the recommendation above have made it worse*. Transcribed from the producers 2026-09-04.

Query: `hours` (default 24, clamped to **0.1–168**).

| field | type | meaning |
|---|---|---|
| `hours` | float | Echo of the resolved, clamped window. |
| `total` | int | Timeout events in the window, every layer. |
| `premature` | int | Of those, the ones that fired **below** the deadline the timeout model recommended — someone's deadline severing a call the model always expected to take longer. |
| `premature_unplanned` | int | `premature` minus the events that fell inside an operator maintenance window. |
| `premature_foreground_unplanned` | int | …and of those, the P0–P2 subset. Background work (P3/P4) defers and retries silently by design, so this is the number that answers "is the fleet under pressure a user can feel?". |
| `planned` | int | Events inside a maintenance window. 🚨 `premature` and `planned` are **independent flags that overlap** — a drained backend can also sever a call below its recommendation, and that event is counted in both. Do not subtract one from the other. |
| `by_abort_reason` | object | `{reason: count}` fleet-wide, over the `stream` layer's four reasons. |
| `rows` | list | One entry per `(endpoint, priority, layer)` cell, descending by `count`. |
| `stream_extensions` | object | Process-lifetime counters read straight off the loop — **not** windowed, and reset by a restart. |
| `maintenance_windows` | list | The windows `planned` was computed against: `{id, endpoint, started_at, ended_at, reason, operator, source}`. `ended_at` is `null` while a window is still open; `endpoint` is `*` for one covering every class. |

`rows[]`:

| field | type | meaning |
|---|---|---|
| `endpoint` | string | Endpoint class as persisted. |
| `priority` | int | The band, numeric. |
| `layer` | string | Where the call gave up — the vocabulary below. |
| `count` | int | Events in the cell. |
| `premature` | int | Of those, below the recommendation. |
| `planned` | int | Of those, inside a maintenance window. |
| `elapsed_s_p50` | float | Seconds waited before giving up, 2dp. |
| `elapsed_s_p95` | float | 2dp, same basis. |
| `avg_in_flight` | float | Mean in-flight count on the endpoint at the moment each event fired, 1dp — the load context, and the reason a timeout report is readable at all. |
| `avg_queued` | float | Mean queue depth at the same moments, 1dp. |
| `recommended_ms_p50` | float | Median recommendation for the cell, 1dp — the number `premature` is measured against. |
| `avg_context_used_pct` | float \| null | Mean context occupancy, 1dp. `null` when no event in the cell recorded one. |
| `top_callers` | object | `{caller_id: count}`, the three largest. |
| `abort_reasons` | object | `{reason: count}` for this cell alone. |

`stream_extensions`:

| field | type | meaning |
|---|---|---|
| `count` | int | Streams whose deadline was extended because the backend was still emitting tokens. |
| `total_s` | float | Slot-seconds spent on those extensions, 1dp. Without it a progress-governed deadline is an invisible capacity sink — you would see the kills it prevents and never what they cost. |
| `hard_cap_aborts` | int | Streams cut at the hard cap regardless of progress. |
| `progress_extensions` | int | Gap deadlines extended on proven backend progress. Read it beside `by_abort_reason.stall`: this rising while stalls fall is the mechanism working, both rising is a backend genuinely in trouble. |

**The `layer` vocabulary is four values**, and they name failures with different owners:

| `layer` | the call gave up |
|---|---|
| `admission` | waiting to be admitted — it never reached a backend. Capacity. |
| `client_wait` | admitted, then out of time on the deadline in force for it. |
| `backend` | the backend failed or its transport deadline fired. |
| `stream` | mid-stream, and `by_abort_reason` says which bound: `ttft` (accepted, never emitted a first token), `stall` (was emitting, then stopped), `hard_cap` (still healthy, cut for capacity) or `caller_deadline`. The first two are the substrate dying under a caller that did nothing wrong; the last two are Roadstead deciding to stop. |

⚠️ **With no DB the envelope is narrower**, the same trap §3.6 documents for the fleet analytics: the
producer early-outs to `{total, premature, rows}`, so the response is `hours`, those three and
`stream_extensions` — **no `by_abort_reason`, no `planned`, no `premature_unplanned`, no
`premature_foreground_unplanned`, no `maintenance_windows`**.

#### `GET /v1/timeout-advice/shadow-report` → `timeout_shadow_report(hours)`

**Would the recommendation have held?** For every completed call the proxy logs what §1.4 would have
advised and whether that number would have fired. This summarises the log per `(endpoint, priority)`,
so the headroom a data-driven deadline reclaims can be read *before* anything is switched onto it.

Query: `hours` (default 24, clamped to **0.1–168**).

| field | type | meaning |
|---|---|---|
| `hours` | float | Echo of the resolved, clamped window. |
| `report` | list | One row per `(endpoint, priority)`, ascending by both. `[]` with no DB. |

`report[]` — these seven are always present:

| field | type | meaning |
|---|---|---|
| `endpoint` | string | Endpoint class as persisted. |
| `priority` | int | The band, numeric. |
| `samples` | int | Shadow rows behind the cell. 🚨 Completions **only** — see the survivorship note below. |
| `actual_timeouts` | int | Calls in the same cell and window that really did time out. |
| `would_timeout` | int | Of `samples`, how many the recommendation would have severed. |
| `would_timeout_rate` | float | `would_timeout / samples`, 4dp. `0.0` when `samples` is 0. |
| `observed_timeout_rate` | float | `actual_timeouts / (samples + actual_timeouts)`, 4dp — the real rate, with the censored samples in the denominator. |

🚨 **`would_timeout_rate` is computed over survivors and reads ≈0 while calls are actually timing
out.** The shadow log records `status == ok` completions, so a call that really timed out is never in
`samples` — it is in `actual_timeouts`. The two censored-sample fields are the 2026-06-06 correction
for exactly this; quoting `would_timeout_rate` on its own as "the timeout rate" is the mistake they
exist to prevent.

**The remaining seven fields are conditional** — present only when the cell has at least one shadow
row. A cell holding nothing but real timeouts arrives with the counts above and none of these, so
read them with a `.get()`:

| field | type | meaning |
|---|---|---|
| `recommended_ms_p50` | float | Median recommendation for the cell, 1dp. |
| `recommended_ms_p95` | float | 1dp, same basis. |
| `actual_total_ms_p50` | float | What the calls really took end to end, 1dp. |
| `actual_total_ms_p95` | float | 1dp. Compare against `recommended_ms_p95`: a recommendation below it is one that would sever the tail. |
| `headroom_vs_applied_ms_p50` | float | `applied_timeout_ms - recommended_ms`, 1dp — the wall-clock the recommendation hands back versus the deadline in force today. **Negative means the recommendation is longer**, not that headroom was lost. |
| `headroom_vs_applied_ms_p95` | float | 1dp, same basis. |
| `sources` | object | `{source: count}` over §1.4's fallback levels, so a cell whose advice came from the `floor` is visible as one rather than reading like measured evidence. |

---

### 1.5 Identity — who a caller is 🚨

**A caller's identity is its `agent_id`, and that string is the DRR fair-share key, the quota holder
and the budget holder.** It is established in one of three ways, in strict precedence:

| | how | strength |
|---|---|---|
| 1 | **API key** — `Authorization: Bearer <key>` (what an OpenAI client already sends), `Authorization: Basic <base64(anything:key)>` (what a *browser* can send — the key is the **password** half and the username is ignored), or `X-API-Key: <key>` | authenticated |
| 2 | **Source address** — an operator registration in `ROADSTEAD_ACL` | a weak second factor: it identifies a *host*, and several callers may share one |
| 3 | **`agent_id` in the body** | **Only a name the credential was granted** (`may_assert`, below). It was unchecked on `/v1/submit`, which was removed in Workstream C (§1.9); it came back on `/rs/v1/*` on 2026-09-02 as a *delegation* rather than a claim. 🚨 The flag-gated legacy door restores the unchecked reading for an internal-net caller and for nobody else — §1.9.2, the one deliberate departure from this section. A caller still cannot name its own fair-share key — it can only pick from the ones its operator wrote down. |

A key carries its own default `priority`, an optional `min_timeout_s` deadline floor, and an
optional `admin` scope, so all four facts travel with the caller rather than with the machine it
runs on. Rows 1 and 2 share one grammar —
`agent_id[:priority][:min_timeout_s][:admin][:readonly]`, segments recognised by **shape** so their
order does not matter — because an operator writing `ROADSTEAD_ACL` and `ROADSTEAD_API_KEYS` in one
compose file should not have to learn two spellings of the same facts. `:readonly` **narrows** the
admin scope to reads (§3.3) and grants nothing on its own; written without `:admin` it is warned
about rather than silently ignored.

Three rules, each of which is a decision rather than an implementation detail:

1. 🚨 **A presented key that does not resolve is a `401 invalid_api_key`. It never falls back to the
   source address.** A wrong or revoked credential must not silently become a *different, weaker*
   identity that still works — from the outside that is indistinguishable from the credential being
   fine. Remove the header to be identified by address instead; the error message says so.
2. 🚨 **When no keys are configured, a presented key is ignored entirely** and the address decides.
   Every OpenAI client sends an `Authorization` header whether or not anybody meant it to, so
   treating one as significant before an operator has configured any key would refuse the existing
   world over a credential nobody chose.
3. 🚨 **A key overrides a body-declared `agent_id` unless it was granted the name; an address only
   fills in one the body omitted.**
   A verified credential is a stronger statement about who is calling than anything in the body, so
   the body may never *claim* an identity. What changed on 2026-09-02 is that a credential may
   **delegate** one: `may_assert` lists the `agent_id`s this key is permitted to act as, and a
   declared name inside that list is honoured while one outside it is a **403** — never a quiet
   fall-back to the credential's own identity, which would put the work on one caller's bill and
   the record on another's with nothing anywhere to say so.

   🚨 **A key with no `may_assert` ignores a declared `agent_id` exactly as before, and does not
   refuse it.** The asymmetry is deliberate and turns on whether an operator opted in: one who wrote
   `may_assert` asked for the field to mean something, so a bad value there earns a sentence; one
   who wrote none has a caller sending a field carried over from `/v1/submit`, and 403-ing that
   would break every such caller on upgrade over a claim that was already inert.

   **Why the rule needed an exception at all.** It assumed the security boundary and the fairness
   boundary are the same object, and on a real fleet they are not: callers are often sibling
   processes in one container sharing a filesystem and a uid — *one* trust domain — while DRR
   fair-share, quotas and spend all need to tell them apart. Issuing one key per fair-share identity
   puts N secrets where one boundary is; issuing one and collapsing the identities destroys what the
   weights exist for. The allowlist keeps the property that made this rule right — a caller cannot
   claim an identity nobody gave it — while letting one credential carry many fair shares.

   🚨 **Delegation moves the fair-share key and grants no policy.** The band, deadline floor, admin
   scope and `key_id` all stay the credential's. A key that could hand itself a different policy by
   naming another agent would be the self-asserted `agent_id` bug restored rather than fenced.
4. 🚨 **A forwarded address is believed only from a trusted proxy, and the caller is the rightmost
   hop that is not one.** `X-Forwarded-For` is a caller-supplied string. It is read only when the
   peer is listed in **`ROADSTEAD_TRUSTED_PROXIES`** (a comma-separated list of addresses or CIDRs,
   **empty by default**, so the peer address decides until an operator opts in). Rule 4 is rule 3
   one level down: honouring the header unconditionally would let any caller assert any source
   address, and taking its *leftmost* element — the intuitive reading — would do the same, because
   the leftmost is exactly the part the caller wrote before any proxy appended what it observed.

#### What a trusted proxy changes

Put a reverse proxy or TLS terminator in front of Roadstead without configuring one and **every
caller collapses into the proxy's address**: the address layer becomes a single identity, so
`ROADSTEAD_ACL` stops distinguishing anybody, and if the proxy sits in the admin nets — loopback and
docker-internal are there by default, and a sidecar usually is one of them — the control plane is
granted to everyone who can reach the proxy.

Configuring `ROADSTEAD_TRUSTED_PROXIES` fixes that, and changes one other thing on purpose:

🚨 **A forwarded address does not inherit the built-in admin nets.** Loopback and docker-internal are
auto-granted admin because reaching them meant already being on the machine; a front proxy is
precisely what makes that untrue. A forwarded request is admin only via an **`admin` API key** (the
recommended path — it works from anywhere and is revocable) or an address the operator named in
`ROADSTEAD_ADMIN_NETS`. A trusted proxy that forwards *no* header is treated as forwarded too, so a
front proxy that lost its configuration does not become an administrator.

Two chain shapes fail closed rather than back to the peer, because resolving to the peer would hand
the proxy's identity — and its grants — to whoever sent the header: an **unparseable hop** where a
caller's address should be, and a **chain longer than 32 hops**. Both resolve to `unknown`, which is
not an address, matches no registration, and is therefore refused.

`GET /rs/v1/admin/config` reports what is trusted and what that changed (§3.5).

#### Basic carries a key, not a password 🚨

A browser cannot attach a bearer token to a navigation, and `EventSource` cannot set a request
header at all — so the management UI (§3.7) needs a scheme the browser itself carries. That scheme
is Basic, and **the credential inside it is an ordinary API key**. Minting a password to go with it
would be a second kind of credential with its own store, its own rotation and its own revocation,
running in parallel with a registry that already does all three; `identity.py` exists to prevent
exactly that. The username is ignored rather than checked against `key_id`, because the label is
public and requiring it buys nothing an attacker holding the key does not already have.

Rules 1 and 2 hold through the wrapper: a Basic password that does not resolve is a **401** and never
falls back to the address, and with no keys configured a Basic header is ignored entirely. A header
that cannot be base64-decoded is **no credential at all** rather than a failed one — the address
decides, exactly as with no header — because a header we cannot parse was probably never meant as
ours.

⚠️ **Changed 2026-09-01.** `Authorization: Basic` was previously ignored as "somebody else's auth".
A deployment that has keys configured *and* callers presenting an unrelated Basic header will now see
those callers refused with a 401 rather than identified by address. Recorded in `CHANGELOG.md`.

#### A placeholder bearer, for an SDK that refuses an empty key 🚨

**`ROADSTEAD_BEARER_PLACEHOLDERS`** is a comma-separated list of literal bearer values — **empty by
default, which is off** — that this proxy reads as *no credential presented*. A request carrying one
is identified by its **source address**, exactly as if it had sent no `Authorization` header at all.

**Why it exists.** An OpenAI SDK will not construct a client with an empty `api_key`, so a fleet
that authorises by address has to send *something*, and what it sends is a literal that means
nothing: `not-needed`, `EMPTY`, `sk-no-key-required`. Rule 2 already covers that while no keys are
configured. The moment an operator configures one, rule 1 takes over and every such caller is
refused — correctly, and on 2026-09-04 fatally: two fleet callers sending
`Authorization: Bearer not-needed` took an interactive chat path down for eleven minutes and the
cutover was rolled back. This is the narrow, declared way across.

| | |
|---|---|
| **matching** | an **exact, case-sensitive** match of the presented token against a declared value. `Not-Needed` is not `not-needed`. Nothing is inferred from shape |
| **anything else** | unchanged: a registered key authenticates, an unregistered one is a **401** and still never falls back to the address |
| **where** | every door that resolves identity — `/v1/chat/completions`, `/v1/embeddings`, `/rs/v1/*`, the flag-gated `/v1/submit` (§1.9) and the admin routes |
| **scheme** | **`Bearer` only.** `Basic` is untouched — see below |
| **what it grants** | what an address grants, and no more: no `may_assert`, no admin, no band and no deadline floor. On an admin route the request proceeds as address-identified and is then refused by the admin gate, exactly as a request with no header is |
| **with `ROADSTEAD_REQUIRE_API_KEY=1`** | still a **401**. Nothing was presented, so the requirement is not met |

🚨 **It is not applied to `Basic`.** Basic is the management UI's channel and the CSRF gate keys off
it (§3.7): a browser attaches a cached Basic credential to any request to this origin on its own,
and `X-Roadstead-Request: 1` is the one signal that tells the UI's own `fetch()` from a forged
cross-site submission riding it. Reading a placeholder *password* as "no credential" would move that
request onto a path where the signal no longer applies. A Basic password that is a declared
placeholder is therefore an ordinary unregistered credential: **401**, per rule 1.

🚨 **A placeholder that is also a registered key's plaintext refuses to start**, with a message
naming the `key_id`. That key would still be presented and still be accepted — as a weaker,
address-derived identity with a different fair share — which is a working credential silently
demoted, the failure rule 1 exists to prevent. Elsewhere this package reports a configuration
mistake and carries on (§3.5); this one it will not serve. Checked against the environment, the keys
file and the management overlay at boot; a key enrolled at runtime through §3.3 is checked on the
next start.

**It is a shim, and it comes out.** A non-empty list logs a WARNING at startup naming the removal
condition, `GET /v1/status` → `reliability.placeholder_bearers` reports `{count, by_address}` since
boot — **requests**, counted once each even on the doors that resolve identity twice — and an INFO
line — `placeholder bearer <value> from <addr> treated as no credential
(ROADSTEAD_BEARER_PLACEHOLDERS)` — names each (placeholder, address) pair once per UTC day. When
that stays empty across a representative window the callers hold real keys and the variable can be
deleted.

**Every door is gated**, the three `/rs/v1` routes included. Out of the box, loopback and docker-internal
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

🚨 **Thresholds DEGRADE. They never reject.** Crossing one costs a caller exactly two things: **one
priority band** (floored at the lowest, however far over it is) and **access to paid spill**. It never
costs local capacity, and **no error code exists for it** — the absence from §2.1 is the contract, not
an omission. Two reasons: admission control is about capacity rather than billing, and a misconfigured
quota must not be able to take a caller offline.

**There are two thresholds, and they have the same consequence on purpose.**

| | crossing it means | why it exists |
|---|---|---|
| `daily_spend_usd` | one band, no paid spill | real money left the machine |
| `requests_per_minute` | one band, no paid spill | **the abuse control DRR is not** |

DRR is fairness *under contention*: a caller alone on a quiet fleet is unthrottled by design, which is
correct for fairness and exactly why it does not bound a runaway. `requests_per_minute` closes that
gap without becoming a rate *limit* — a 429 would be a fourth spelling of "no" and would put a
misconfigured threshold in a position to take a caller offline, which is the same argument that keeps
a spend cap from refusing anything. A caller going too fast ends up behind every caller behaving
itself, and that is enough.

🚨 **Degradations do not STACK.** A caller over both thresholds drops **one** band, not two. Two
independent one-band penalties would mean that adding a second threshold silently doubled the first
one's, and the unbounded-penalty argument above applies with more force to two of them than to one.

Both are keyed on the **`agent_id`**, like every other quota: the budget holder is the caller, not the
key, and a per-key threshold whose penalty landed on the caller's band would punish a team for one
credential's behaviour anyway. The answer to a single credential misbehaving is to **revoke** it,
which is instant (§3.3).

A caller cannot observe its own demotion in a response; each crossing is reported to the operator
through the degradation seam (§5) once per caller per day — **separately**, because a rate problem
must not look like a billing one — and shown on `/v1/status` and `GET /rs/v1/admin/callers`.

### 1.7 The enriched API — `/rs/v1/*` 🚨

**Two north faces, and they are versioned separately.** `/v1/*` is
OpenAI-compatible and is versioned by OpenAI; `/rs/v1/*` is Roadstead's own and is
versioned by us. Pinning them together would mean either following somebody
else's version number or publishing a `/v2/chat/completions` that is not
OpenAI's v2.

🚨 **Enrichment never appears inside an OpenAI-shaped body.** A client that
validates against OpenAI's schema must not break because it pointed at
Roadstead, and "we only *added* fields" is not a defence: strict validators
reject unknown keys, and lenient ones hand the extra key to a caller that then
depends on it from a server that is not us. Everything the OpenAI door can carry
rides in response headers (§1.8); everything else lives here.

| Route | Purpose |
|---|---|
| `GET /rs/v1/models` | What can serve me, what can it do, what is it like *now*, what does it cost. |
| `POST /rs/v1/plan` | Where would this go, how long should I allow, what would it cost — **without dispatching**. |
| `POST /rs/v1/chat` | The enriched call — a chat completion, an embedding or a rerank, selected by `payload_type` (§1.7.2). |

Every route is gated exactly like the OpenAI doors (§1.5). `/rs/v1/models` is a
map of the fleet — slot counts, occupancy, health and prices — and an unenrolled
caller has no more business reading that than dispatching to it.

🚨 **The `price` block has ONE shape on every route that carries it** — the five
fields tabled in §1.7.3, on a `/rs/v1/models` row, in a `/rs/v1/plan` estimate
and inside `attribution.cost`. Until 2026-09-01 the models row carried three of
the five, silently: a client with one typed view for the price read `source` and
`detail` as empty for every endpoint, which is the field that says whether a
price was published by the provider or imputed by us. Two spellings of the same
object is the gap `docs/ledger.md` is mostly made of.

#### 1.7.1 Declaring what you want: intent, or a pin

A request must declare at least one of `intent`, `model`, `requires` or
`exclude`, and may combine them — a pin and a profile together (`model: tier3`,
`intent: vision`) reads as "this endpoint, and it had better be able to see":

| Field | Type | Meaning |
|---|---|---|
| `intent` | string | A capability **profile** — `fast-chat`, `chat`, `reasoning`, `vision`, `tools`, `structured`, `long-context`, `embed`, `rerank` ship built in, and a deployment may add or override them. Roadstead owns the choice of model, provider and moment. |
| `model` | string | A **pin**: an endpoint class, role or alias. A constraint on routing, not a different API. |
| `exclude` | [string] | Endpoints this request must **not** use — classes, roles or aliases, resolved exactly as `model` is. A negative constraint, for a caller working around one bad model. |
| `requires` | [string] | Capabilities that must be declared on whatever serves: `vision`, `reasoning`, `streaming`, `tool_calling`, `structured_output`. Composes on top of the profile's own. |
| `kind` | string | `chat` \| `embed` \| `rerank`. Defaults from the profile, else `chat`. |
| `min_context` | int | Minimum context window, tokens. |
| `prefer` | string | How to order candidates: `balanced` (default), `latency`, `capacity`, `context`, `cost`. An explicit value beats the profile's. |

`GET /rs/v1/models` publishes the profile table under `intents`, and that is the
only correct place to read it. Nine profiles ship built in; a deployment adds or
overrides them with an `intents:` section in its catalog, **layered over** the
built-ins rather than replacing them, so a file that defines one profile has said
nothing about the other nine. Each published profile carries a `source` of
`builtin` or `models.yaml`: a fleet may redefine `reasoning` to mean its own
thing, and a caller reading this document for that word needs to be able to see
that it no longer applies.

🚨 **A profile is expressed in declared capabilities and can never name an
endpoint.** There is no config field that can. A profile naming endpoints would
be a second routing table to keep in step with the catalog, and it would break on
every fleet whose classes are spelled differently from ours. `exclude` is not an
exception to this: it is one caller's words about one request — exactly as `model`
already is — rather than shared, published vocabulary. The line is between config
and request, not between positive and negative.

🚨 **An `exclude` naming an endpoint this fleet does not have is a
`404 unknown_endpoint`, not a warning.** The tempting reading is that such an
exclusion is satisfied trivially, since the endpoint it forbids is absent. That
assumes the one thing the proxy cannot check: a name that resolves to nothing is
either "not in this fleet" or "in this fleet, under a spelling you got wrong",
and from here those are identical. Serving the second sends the request to
precisely the endpoint the exclusion existed to avoid and reports success — the
same shape as a repair that becomes a silencer. A caller that must spell an
endpoint correctly to demand it does not get to misspell one to avoid it. Naming
the same endpoint in `model` and `exclude` is a `400`: a contradiction wholly
visible in the request, refused where the caller can see it rather than resolved
to an empty candidate set that reads as a fault in the fleet.

🚨 **There is deliberately no `quality` preference.** A gateway cannot measure
model quality, and a key spelled `quality` that resolved to "the one with the
biggest context" would be read by a caller as a promise about answers. Every
name above is something the proxy actually observes.

Two resolution rules are doctrine:

1. 🚨 **A real-cost endpoint sorts last under every preference.** Local capacity
   is the design center and remote capacity is *overflow*. If an intent could
   prefer a remote endpoint because it happened to be faster or emptier, traffic
   would leave the machine on the ordinary path rather than only when local
   capacity said no, and the operator would find out on an invoice. A remote
   endpoint is reachable through an intent only when **no local candidate
   satisfies the requirements at all** — which is what "overflow" means.
2. 🚨 **An endpoint with no latency samples is treated as slow, not fast.**
   `typical_ms` is reported as `null` when unmeasured. Ranking it first would be
   preferring a backend *because* there is no evidence about it.

A declaration nothing can satisfy is a **`404 unknown_endpoint`** carrying
`considered` — the near-misses and why each failed. It is never served from
whatever was nearest: a pin at a text-only endpoint with `requires: ["vision"]`
is a contradiction, and answering it from the vision model next door would be a
substitution the caller could not detect.

#### 1.7.2 Prioritisation and the deadline

| Field | Type | Effect |
|---|---|---|
| `priority` | enum \| int | The band, as on the OpenAI door. Wins over `interactive`, because it says strictly more. |
| `interactive` | bool | The first-class spelling of what `priority` has always encoded and never named: whether somebody is waiting. `true` → `P1_TURN_SUPPORT`, `false` → `P3_INGESTION`. |
| `deadline_s` | float | The enriched spelling of `timeout_s`. **Usually omit it** — §1.2. |
| `substitution` | object | `{"degrade": bool, "spill": bool}`. See §1.7.4. |
| `call_site`, `session_id`, `turn_id`, `caller_id`, `request_id` | string | Attribution and correlation. |
| `payload` | object | **The model request itself** — messages, `max_tokens`, `tools`, `response_format`, `stream`. |
| `agent_id` | string | **Act as this caller.** Honoured only when the credential's `may_assert` grants the name (§1.5 rule 3); refused `403` when it has a grant that does not, ignored when it has none. |
| `payload_type` | string | What SHAPE `payload` is: `chat_completion` (the default) \| `embedding` \| `rerank`. It selects the route on the backend — the chat route, the embedder's `/embed`, the reranker's `/rerank`. |

🚨 **One agent, two kinds of work: three different knobs, and only the third needs a grant.** A
caller that does interactive chat *and* background summarisation is asking one of three questions,
and reaching for `agent_id` when it wanted `priority` costs an operator a credential grant for
nothing:

| what differs | the knob | needs a grant? |
|---|---|---|
| **When it should run** — someone is waiting vs. nobody is | `priority`, or `interactive` | No. Per call, always available. |
| **How it shows up in the numbers** — `chat-agent.chat` vs `chat-agent.summarise` | `call_site`, `caller_id` | No. Free-form, per call. |
| **Whose budget it spends** — its own DRR balance, quota and spend cap | `agent_id` | **Yes** — §1.5's `may_assert`. |

🚨 **The third is a real question, not a redundant spelling of the first**, and the reason is that
the **DRR balance is one per `agent_id`, shared across bands**. Priority orders the work — a P1 turn
is dequeued ahead of a P3 storm every time — but both spend the *same* balance. So a background
storm can drain the balance its own interactive turns spend from, and those turns then meet DRR
fairness empty-handed and yield to other callers in their band. If that matters for a given agent,
split the identity (`chat-agent` and `chat-agent-summarise`, both on the key's allowlist, each with its own weight)
and the two stop competing for one balance. If it does not, one `agent_id` and a per-call `priority`
is the simpler and correct answer.

🚨 **`payload_type` is not `kind`, and neither defaults from the other.** `kind` (§1.7.1) is a
ROUTING declaration — what sort of endpoint may serve this — and an intent profile sets it.
`payload_type` is a declaration about the BODY. `kind: "embed"` alone routes to an embedder and then
posts a chat-shaped request to it; `payload_type: "embedding"` alone asks a chat model to answer an
embedding body. Both questions are real, so both fields exist.

🚨 **This is rerank's only route.** `/v1/submit` carried `payload_type` and was removed (§1.9),
which left rerank with no door at all until the SDK learned to send this field on 2026-09-02. There
is deliberately no `/v1/rerank`: OpenAI has no rerank shape to be compatible with, so such a route
would be Roadstead's own API wearing somebody else's version number. `POST /rs/v1/chat` is where
Roadstead's own API lives, and it is named for the route rather than for chat.

🚨 **The split between the envelope and `payload` is the contract.** Routing
declarations outside, the model request inside. That is what lets `deadline_s`
exist at all without being forwarded to a backend that would reject the unknown
field — the failure mode `/v1/submit` had to pop `timeout_s` out of the body to
avoid.

#### 1.7.3 The response

Five blocks, plus the backend's own body **nested** under `response` so a caller
never has to tell Roadstead's fields from the model's.

🚨 **`identity` says who the call was BILLED to** — `{agent_id, declared?, honoured?}` — and the
second half is why it exists. A credential with no delegation grant *ignores* a declared `agent_id`
(§1.5 rule 3) rather than refusing it, which is right for upgrade compatibility and would be a
**silencer** if it were also invisible: the caller asks to be `chat-agent`, the work is billed to the
credential, and both outcomes are a 200 with otherwise identical bodies. `agent_id` is always
present; `declared` and `honoured` appear only when the caller declared something, on the same rule
that keeps `substituted` from firing on every intent-routed call and meaning nothing.

It leaks no band, queue position or demotion (§1.6): the value is either the caller's own word or
the name on the credential it presented, and neither is news to the caller.

```json
{
  "status": "ok",
  "request_id": "req_...",
  "response": { "...the backend's OpenAI-shaped body, untouched..." },
  "corrections": [],
  "attribution": {
    "requested": "reasoning", "resolved": "tier3", "endpoint": "tier3",
    "substituted": false, "substitution": null,
    "provider": "large-box", "engine": "vllm", "model": "...",
    "cost": {"spent_usd": 0.0, "avoided_usd": 0.0031,
             "price": {"input_usd_per_mtok": 0.8, "output_usd_per_mtok": 2.4,
                       "real": false, "source": "imputed",
                       "detail": "usage_rates class tier3"}}
  },
  "timing": {
    "queue_wait_ms": 3.1, "backend_latency_ms": 2210.0, "ttft_ms": null,
    "total_ms": 2213.1, "deadline_s": 180.0,
    "deadline_source": "computed", "predicted_ms": 2400.0
  },
  "usage": {"input_tokens": 812, "output_tokens": 410, "slot_seconds": 1.02}
}
```

| field | meaning |
|---|---|
| `attribution.requested` | The caller's **own words** — the intent profile, or the pin as written (an alias stays the alias). |
| `attribution.resolved` | The endpoint resolution chose, before any substitution. |
| `attribution.endpoint` | The endpoint that **actually served**. |
| `attribution.substituted` | True exactly when `resolved != endpoint`. |
| `attribution.substitution` | `failover` \| `spill` \| `null`. |
| `attribution.model` | The model the backend is serving, as discovery found it — not the class, and not what was asked for. |
| `attribution.cost.spent_usd` | **An invoice.** Non-zero only for a real (remote) price. |
| `attribution.cost.avoided_usd` | **A saving.** What renting the same class of model would have cost. |
| `attribution.cost.price.real` | 🚨 `true` = an invoice, `false` = a cost avoided. The one bit that says which kind of money the two fields above are. |
| `attribution.cost.price.source` | `provider` (the backend published it) \| `config` (an operator declared it) \| `imputed` (`usage_rates.py`'s avoided-cost model). The first two are real money. **Where the number came from, not how recent it is** — a published price and an imputed one can be equal and still mean different things. |
| `attribution.cost.price.detail` | Free-text provenance for a readout. Never parse it. |
| `timing.ttft_ms` | `null` for a non-streaming call — there is no first token to time, and `0` would read as an instantaneous one. |
| `timing.deadline_source` | `caller` when you supplied `deadline_s`, `computed` when Roadstead chose it. A behaviour difference, not a label: a computed deadline is a soft budget the streaming path may extend while tokens are still arriving; a supplied one is a hard wall. |
| `timing.predicted_ms` | What the timeout model expected. `null` when the evidence is too thin. |
| `usage.slot_seconds` | Backend occupancy — **the unit of DRR fairness**, and the number that explains scheduling in a way token counts cannot. |
| `corrections` | The SAME token vocabulary as `X-Roadstead-Corrected` (§1.8), as a list — `[]` when nothing was rewritten. Always present, never omitted, so a caller need not distinguish "clean response" from "envelope predates the field". |

🚨 **`spent_usd` and `avoided_usd` are never summed.** Both are USD and nothing
else distinguishes them; exactly one is non-zero per call, and which one is a
property of the price rather than of the endpoint. §1.6.

🚨 **The response carries no priority, no band and no queue position.** §1.6: a
caller cannot observe its own spend demotion. Publishing the effective priority
would turn a threshold that "never rejects" into one every client could detect
and branch on, which is a rejection with extra steps. The absence is the
contract, not an omission — `tests/e2e/test_enriched_api.py` fails if one
appears.

**Errors** keep `code` and `error` at the top level, with the §2.1 spellings
verbatim: the enriched API is a new shape, not a new taxonomy, and §2.2's marker
substrings live in `error`.

#### 1.7.4 Substitution: opt-in, narrowable, always disclosed

Two independent permissions, granted per identity in the agents config:

| | question | granted by | triggered by |
|---|---|---|---|
| **degrade** | this backend is DOWN — may a smaller model answer? | `degrade_ok` | health |
| **spill** | this backend is FULL — may we pay somebody else to answer now? | `spill_ok` | occupancy |

🚨 **Neither defaults from the other**, and one endpoint can want both. A caller
whose work may be answered by a smaller *local* model may still be one whose
prompts must never leave; the reverse is just as common.

A request may send `substitution: {"degrade": false, "spill": false}`.

🚨 **This NARROWS and never widens.** Both gates take the AND with the
operator's opt-in, so `true` grants nothing on its own. A request that could
grant itself either would let a caller award itself a permission its operator
withheld — and for spill the consequence is money spent on somebody's behalf
without their say-so. The useful direction is the other one: one confidential
prompt on an identity that is otherwise happy to spill. Declining is a **defer**,
not an error — the request keeps its place and is served locally when a slot
frees.

🚨 **Resolution is not substitution.** An intent that lands on an endpoint the
caller never named is Roadstead doing the job it was asked to do; nothing was
promised and nothing was swapped. Only a *later* move — failover or spill — is
reported as a substitution. Conflating them would make `substituted: true` fire
on every intent-routed call and therefore mean nothing.

#### 1.7.5 Streaming

`payload.stream: true`. SSE, with typed frames:

| frame | when |
|---|---|
| `accepted` | Once, first. Carries `attribution` and `timing` for the endpoint **admission chose**. |
| `admitted` | A scheduling marker; ignorable. |
| `chunk` | `data` is the backend's raw OpenAI `chat.completion.chunk` JSON, byte-identical. |
| `done` | Once, last. `attribution`, `timing`, `usage`. |
| `error` | Terminal instead of `done`; carries `code` and `error`. |

🚨 **The `done` frame's attribution is the authoritative one.** Failover and
spill both move a request *after* `accepted` is on the wire, so a caller that
trusted the opening frame would be told the endpoint we intended rather than the
one that answered. The enriched stream never emits `[DONE]`.

---

### 1.8 Enrichment on the OpenAI door 🚨

The OpenAI body stays byte-identical (§1.7). What can be said in headers is:

| header | meaning |
|---|---|
| `X-Roadstead-Request-Id` | Correlates with `/v1/timeouts`, the completion row and the logs. |
| `X-Roadstead-Endpoint` | The endpoint admission chose. |
| `X-Roadstead-Deadline-S` | The deadline actually applied. |
| `X-Roadstead-Deadline-Source` | `caller` \| `computed` — see §1.7.3. |
| `X-Roadstead-Corrected` | Comma-separated tokens naming what the correction layer rewrote (or silently could not fix) before this response was served — `json_object_stripped`, `schema_repaired`, `schema_retried`, `schema_unrecoverable`, `degenerate_unrecovered`, `toolcall_truncated`. Absent when nothing fired. On a STREAMING response this can only ever carry `json_object_stripped` — the rest are decided after the backend has answered, past the point headers go on the wire (see the note below). |

🚨 **`toolcall_truncated` is also an envelope `code` (§2.1).** It is the one token
above that names a call which FAILED: the rule that detects it sets
`code = toolcall_truncated` on the envelope and the header token is derived from
that code, so the two always appear together. The other five annotate a response
that was still served. A client classifying on `error.code` therefore sees this
one whether or not it reads headers.

🚨 **Only what is known before the body starts.** A streaming response's headers
are on the wire before the first token, so a later failover or spill cannot be
reflected in them. That is a real limit of the header channel and it is why the
enriched API exists: a caller that must know what actually served has to ask on
`/rs/v1`. Advertising a substitution here that a stream might contradict would
be worse than advertising nothing.

A response carrying none of these did not come from Roadstead. Treat their
absence as "not ours", never as a default.

---

### 1.9 Migrating off `/v1/submit`

**`POST /v1/submit` was removed** (Workstream C; `CHANGELOG.md`). It carried no
intent vocabulary, no attribution and no timing, and each of those would have had
to be bolted onto a shape never designed to hold them. `/rs/v1/chat` is its
replacement, and the map is mechanical.

🚨 **It can be re-opened, unchanged, behind `ROADSTEAD_LEGACY_SUBMIT`** — see
§1.9.1. That is a migration window with a removal condition, not a reprieve: the
map below is still the destination, and the flag exists so a fleet can cross it
one caller at a time instead of all at once.

| `/v1/submit` | `/rs/v1/chat` |
|---|---|
| `agent_id` | **A delegation, not a claim.** Identity is the API key or the source address (§1.5); the body's `agent_id` is honoured only where the key's `may_assert` grants that name, and refused otherwise. On `/v1/submit` it was unchecked — a caller could name any `agent_id`, including one with a better DRR weight — which is why that door went. A fleet migrating many callers behind one credential grants them here rather than issuing one key each. |
| `endpoint` | `model` (a pin) — or drop it and send `intent`. |
| `payload` | `payload`, unchanged. |
| `timeout_s` / `X-Timeout-S` | `deadline_s` — and usually: omit it, §1.2. |
| `priority` | `priority`, or `interactive: true`/`false`. |
| `call_site`, `session_id`, `turn_id`, `caller_id`, `request_id` | Unchanged. |
| `payload_type` | Unchanged (`chat_completion` \| `embedding` \| `rerank`) — see §1.7.2. 🚨 The proxy has always read it here; nothing SENT it between the removal of `/v1/submit` and 2026-09-02, which is why rerank had no route for that window. |
| response `queue_wait_ms`, `backend_latency_ms` | `timing.*` |
| response `estimated_cost_ss` | `usage.slot_seconds` |
| response `response`, `status`, `request_id`, `code`, `error` | Unchanged. |
| response `degraded`, `degraded_from` | `attribution.substituted` / `.substitution` / `.resolved` — which now also cover spill. |
| stream frame `queued` | `accepted`, and it carries attribution. |
| stream frame `done` fields | `timing`, `usage`, `attribution`. |

`roadstead.client` (§7) speaks this API and is the shortest path across.

#### 1.9.1 …and the door itself, back behind a flag 🚨

**`POST /v1/submit` can be re-opened, unchanged, by setting
`ROADSTEAD_LEGACY_SUBMIT`.** Removing it was the right decision and this does not
reverse it; what the removal did not allow for was a fleet with a dozen callers
already speaking the old envelope and no window in which to move them all at
once. So the door comes back **byte-compatible with what it published**, and it
comes back **off**:

| | |
|---|---|
| Flag | `ROADSTEAD_LEGACY_SUBMIT` — `1` \| `true` \| `yes` \| `on` |
| Default | **OFF, and OFF means the route does not exist.** A request gets the same `404` any unknown path gets — not a `405`, not a `404` that mentions itself. Same posture as `ROADSTEAD_ADMIN_UI` (§3.7). |
| Where it lives | `roadstead/legacy.py`. It translates into `Lifecycle.handle_submit` like the other doors do and adds no admission path of its own: `wire` picks the response shape and nothing else. |
| Removal | Gated on **the inventory being empty**, not on a date. `GET /v1/status` → `reliability.legacy_submits` (`{count, callers{agent_id: n}}`) is that inventory; a WARNING names each caller once per UTC day beside it. |

**Request envelope.** Exactly these fields; anything else in the body is ignored,
including the four `/rs/v1/chat` reads that this door never published
(`declared_agent_id`, `requested`, `allow_degrade`, `allow_spill`).

| Field | Type | Effect |
|---|---|---|
| `agent_id` | string | The fair-share key — subject to §1.9.2. Absent → the resolved identity's own. |
| `endpoint` | string | The routing endpoint (a role or alias). Default `chat`. 🚨 Overridden by `payload.model` when that names a *different known* endpoint and `payload_type` is `chat_completion`; the reconcile is logged. |
| `priority` | string \| int \| null | Coerced, never rejected — see the table below. `null`/absent means *declared nothing*, and §1.5's four-step precedence resolves the band. |
| `call_site` | string | Attribution. Default `unknown`. |
| `payload_type` | string | `chat_completion` \| `embedding` \| `rerank`. Unvalidated, as before: an unknown value takes the chat route. |
| `payload` | object | The model request. `payload.stream` selects SSE. |
| `timeout_s` | number | The caller's deadline, and it wins. Absent (or `null`, or malformed — which logs and defaults) → the computed default, per-identity floor included (§1.2, §1.5). |
| `session_id`, `turn_id`, `caller_id`, `request_id` | string \| null | Carried to the durable record. `request_id` is minted when absent. |

Priority coercion — the same table `LLMPriority.coerce` has always applied, restated because a
caller of this door depends on a malformed value *defaulting* rather than failing its LLM call:

| sent | resolved |
|---|---|
| a member name, any case (`P0_REALTIME`, `p2_post_turn`) | that member |
| `interactive` \| `foreground` \| `background` | `P1_TURN_SUPPORT` \| `P2_POST_TURN` \| `P3_INGESTION` |
| `realtime`, `turn`, `turn_support`, `post_turn`, `ingestion`, `hygiene` | the named member |
| any `P<n>_<suffix>` (e.g. `P3_BACKGROUND`) | by its numeric prefix, with a WARNING |
| an int, in or out of range | the member, clamped to `0…4` |
| `null` or absent | *declared nothing* → §1.5's precedence |
| anything else, including a JSON `true` | `P1_TURN_SUPPORT`, with a WARNING. **Never a 500.** |

#### 1.9.2 The identity exception 🚨

**On this door, and only under the flag, a caller inside the built-in internal
nets is identified by the `agent_id` in its own body, unchecked.** That is the
one place Roadstead deliberately departs from §1.5, it is written down here
because it cannot be reproduced quietly, and its edges are narrow:

| caller | identified by |
|---|---|
| loopback / docker-internal (`127.0.0.0/8`, `172.16.0.0/12`, `::1`), on a connection the transport itself vouches for | **the body's `agent_id`**, unchecked — the exception |
| a **registered** address (`ROADSTEAD_ACL`) | its registration. An operator who wrote the entry said who that host is. |
| an address that arrived via `X-Forwarded-For` | its registration, never the body. "Already on the box" is what the internal nets stand for, and a front proxy is exactly what makes that untrue — the same reasoning `acl.is_admin(trust_builtin_nets=False)` applies to the admin grant. |
| a caller presenting an **API key** | the key, with `may_assert` deciding a declared name (§1.5 rule 3). Unchanged. |

It grants **a name and nothing else**: no band, no deadline floor, no quota, and
🚨 **an address still never grants admin** (§3). Note that this makes the legacy
door *stricter* than `/rs/v1/chat` for a registered address, where §1.5 rule 3
lets the body fill in an identity the registration did not claim — a door
reproducing an old contract must not widen it on the way past.

#### 1.9.3 Response shapes

**Sync success** — HTTP 200, **exactly six keys**:

```json
{"status":"ok","request_id":"req_…","queue_wait_ms":0.0,"backend_latency_ms":0.0,
 "estimated_cost_ss":0.0,"response": <the backend body, verbatim>}
```

🚨 The key **set** is the contract, and callers validate it — so an additive field
here is a breaking change, which is why the enriched envelope's `attribution`,
`identity`, `timing`, `usage` and `corrections` blocks are absent rather than
merely unused. Two documented additions: `"cache_hit": true` on a cache hit (with
the timing fields at zero), and `degraded` / `degraded_from` when a failover
served a smaller model.

For `payload_type: "embedding"` and `"rerank"`, `response` is the backend body
**verbatim** — no OpenAI translation, unlike `POST /v1/embeddings` (§1.1). A
caller reading the embedding shim's own dialect keeps reading it.

**Streaming** (`payload.stream` true) — `text/event-stream`, framed `data: {json}\n\n`,
headers `Cache-Control: no-cache` and `X-Accel-Buffering: no`, and **no `[DONE]`
sentinel**. (The four `X-Roadstead-*` enrichment headers of §1.8 ride along too;
they are additive and no legacy consumer reads them.)

| # | frame |
|---|---|
| 1 | `{"type":"queued","request_id":"req_…"}` — *not* the enriched wire's `accepted`, which carries attribution a legacy consumer would drop |
| 2 | `{"type":"admitted","queue_wait_ms":<float>}` |
| 3… | `{"type":"chunk","data":"<raw backend chunk JSON>"}` — 🚨 `data` is a **string**, not an object. Every consumer of this door parses it itself. |
| last | `{"type":"done","queue_wait_ms":…,"backend_latency_ms":…,"ttft_ms":…,"usage":{"prompt_tokens":…,"completion_tokens":…}}` (+ `degraded`/`degraded_from`) |
| or | `{"type":"error","error":"<string>"}` on any failure, including the consumer-side deadline (`"timeout"`) |

#### 1.9.4 Errors

The §2.1 codes and the §2.2 marker substrings, in the envelope this door
published: `{"status":"error","request_id":…,"error":…,"code":…}`.

| status | `code` | when |
|---|---|---|
| 503 + `Retry-After` | `draining` | draining for shutdown, or the endpoint is paused for maintenance |
| 404 | `unknown_endpoint` | `unknown_endpoint_enforce` is on and the endpoint is not routable |
| 400 | `vision_not_supported` | `vision_capability_enforce` is on and the payload carries an image |
| 400 | `invalid_messages` | `payload.messages` is not a list of objects |
| 422 | `invalid_grammar` | the GBNF grammar does not parse (`detail` carries why) |
| 422 | `context_overflow` | `context_gate_enforce` is on and the request does not fit |
| 503 | `on_demand_unavailable` | an on-demand backend could not be loaded |
| 503 + `Retry-After` | `circuit_open` | the backend is unhealthy (`degraded_refusal` appended when a failover was refused) |
| 429 + `Retry-After` | `backpressure` | a non-interactive band's queue is saturated |
| 502 | `backend_error` | the backend failed after dispatch |
| 504 | `proxy_timeout` | the deadline fired |
| 400 / 401 / 403 / 413 | `invalid_request_error` / `invalid_api_key` / `access_denied` | §1.5, §1.10 |

Two deliberate differences from what this door published before it was removed,
both additive and both stated so nobody has to diff a response to find them:

* the **504** body is `{"status":"error","request_id":…,"error":"timeout","code":"proxy_timeout"}`.
  `error` is still the bare `"timeout"`; `status` is new. The old body omitted it, every caller
  reads `.get("status") != "ok"`, and supplying it can only make more of them agree with the rest of
  the taxonomy.
* a **502** carries `backend_status` when the failure came from a backend (§2.2). No caller pins the
  key set of an error envelope, and the alternative is this door advising a retry the proxy itself
  declined to make.

### 1.10 Request and response size caps 🚨

Two independent bounds, checked at opposite ends of a call:

| | value | env | where |
|---|---|---|---|
| Inbound request body | **16 MiB** | `ROADSTEAD_MAX_REQUEST_BYTES` | `__main__.RequestSizeLimitMiddleware` |
| Streamed backend response | **64 MiB** | `ROADSTEAD_MAX_RESPONSE_BYTES` | `backend.BackendClientPool.stream` |

**The request cap is enforced at the ASGI layer, before anything calls `request.json()`.** Every door
— the OpenAI-shaped doors, the enriched `/rs/v1/*` doors and the admin plane alike — parses its body
with no code of its own bounding how large it may be. `Content-Length` is honoured when a caller
declares it honestly; a caller that declares short and keeps streaming, or declares nothing at all
(chunked), is caught by counting bytes as they actually arrive. Refused with `413` and `code:
invalid_request_error` (§2.1 — no new code minted; not deferrable either way). 16 MiB is sized
for a legitimate vision chat payload: a caller inlines its images as base64 (~1.33x the raw bytes),
and a handful of them in one turn must clear this without the cap being wide enough to let an
unbounded body tie up a queue slot before anything has validated it.

**The response cap is a running count on the streaming loop**, checked line-by-line as
`aiter_lines()` yields rather than after the whole response is already buffered — a wedged or
adversarial backend that never stops talking (no `[DONE]`, no natural end) would otherwise grow the
proxy's own memory to match it. Aborted as a `BackendError(502)`; not classified transient
(`Correction.is_transient_backend_error` only special-cases `BackendUnavailable` and an "empty
completion" detail), because the same request would very likely reproduce the same oversized
response and the proxy's own defer/retry loop must not spend a slot retrying it.

Neither cap touches the inference doors' Bearer-keyed admission logic, and neither is configurable
per caller — both are fleet-wide, ASGI/transport-level bounds.


## 2. Error contract 🚨

### 2.1 Codes

Every error envelope carries a machine-readable `code`. **Seventeen exist** (a common under-count is
eight):

`backpressure` · `circuit_open` · `draining` · `unknown_endpoint` · `invalid_grammar` ·
`proxy_timeout` · `backend_error` · `context_overflow` · `access_denied` · `invalid_api_key` ·
`invalid_messages` · `invalid_request_error` · `vision_not_supported` · `on_demand_unavailable` ·
`structured_invalid_json` · `schema_invalid` · `toolcall_truncated`

🚨 **The last two were emitted for months before this list named them, and one of them cost real
work.** The correction layer mints both, and `lifecycle.py` forwards whatever a correction rule
attached — `body["code"] = result.get("code") or "backend_error"` — so neither ever appeared in a
handler that names a code, and the list stayed at fifteen while the wire carried seventeen. No check
could catch it: every one of them compared this document to the SDK's transcription of it, and two
transcriptions of one document agree perfectly about a code neither has heard of.
`toolcall_truncated` is the expensive half — a client classifies on the code and treats an unknown
one as non-deferrable, so a truncated tool call, which §2.2 says to retry with a larger budget, was
discarded rather than retried. `tests/test_error_codes_published.py` now walks the package for every
`code` it assigns and fails when one reaches the wire without a row below.

| code | status | emitted when | deferrable | what the client should do |
|---|---|---|---|---|
| `schema_invalid` | `502` | A structured response parsed as JSON but violated the declared schema, and the proxy's own repair **and** its one bounded retry-with-the-error-fed-back both failed (`correction.py`, `_schema_retry` — "never retries more than once"). The body is dropped rather than served, and never cached. | **No** | **Change something.** The model has now failed this schema twice with the validation error in front of it; a third identical attempt is a third billed call for the same answer. Simplify the schema, or send the request to a model that can hold it. |
| `toolcall_truncated` | `502` | A tool call whose `function.arguments` were cut mid-JSON while the backend labelled the response `finish_reason=tool_calls` — the same fault as a `finish_reason=length` truncation, wearing the wrong label. Emitted only for backends whose `ProviderDescriptor` declares `mislabels_truncated_tool_calls`; llama.cpp labels it `length` correctly and never reaches this rule. | **Yes** | **Raise the output budget and retry** — exactly as §2.2's `truncated structured output` row says, because it is that fault. The answer did not fit; it was not refused. |

`schema_invalid` is non-deferrable for the reason §2.2 gives `backend_error` below: **the proxy
already declined to retry**, and a caller that retries on its behalf loops against a schema the model
cannot satisfy while the proxy watches. Its published sibling `structured_invalid_json` — same
family, fewer attempts already spent — is non-deferrable on the same grounds. (Two comments in
`correction.py` and one e2e docstring still call this path "deferrable". They describe the *legacy*
classifier, which read the `LLM proxy error 502` prefix a client builds from the status and which
§2.2 has superseded with the code; the same supersession already narrowed `backend_error`.)

🚨 **`invalid_request_error` now also covers `413` (§1.10's request-size cap).** No new code was
minted for it — the same reasoning as §3's "mints no error code of its own": a caller that already
treats this code as non-deferrable (it carries none of §2.2's markers) handles the new status for
free, and the same body is the same size on retry either way.

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

Two more are emitted and matched, and both mean *retry* rather than *give up* — a client that treats
either as terminal throws away a call the next attempt would have completed:

| marker | emitted when | what the client must do |
|---|---|---|
| `truncated structured output` | A structured request (a grammar, a `response_format` schema, structured outputs) came back with `finish_reason=length`, so the body is almost certainly unparseable JSON — `lifecycle.py` fails it loud with a deferrable error rather than record garbage, and never caches it. `correction.py` emits the same marker for the other spelling of the same fault: a tool call whose `arguments` were cut mid-JSON while `finish_reason` claimed `tool_calls`. That variant carries the code `toolcall_truncated` (§2.1, deferrable) and additionally reports `toolcall_truncated` in `X-Roadstead-Corrected` (§1.8). | **Raise the output budget and retry** — a larger `max_tokens`, or re-chunk the input so the answer fits under the one in force. Retrying the identical request unchanged truncates identically. |
| `returned empty completion` | A backend answered `200` with no content at all (`backend.py`). The proxy retries this itself first, and arms a `min_tokens` re-dispatch to break a position-0-EOS degeneration — it matches this very substring to decide to do so, which makes the marker load-bearing *inside* the proxy as well as at the caller. A caller only sees it once those attempts have been spent. | **Retry.** It surfaces as a `502` carrying `backend_status: 502`, so the rule below keeps it deferrable — but a caller matching prose must not read "empty" as "the model had nothing to say". |

Free-form truncation is *not* in this set: a `finish_reason=length` on an unstructured request is a
short answer, not a broken one, and it is returned normally.

Nor is `schema_invalid`: its message says *schema-invalid structured output*, which carries **no
marker in this table** and is meant to carry none. A response that violated the schema is not a
response that was cut off, and giving it the truncation marker would tell a prose-matching caller to
raise `max_tokens` for a fault more output cannot fix. The absence is the contract, as in §1.6.

Two consequences, both easy to get wrong:

1. **Rewording an error message is a breaking API change**, even when the `code` is untouched.
2. Any client library Roadstead ships should classify on `code`, not on prose — but it must keep
   emitting the legacy substrings until every existing caller has migrated.

🚨 **A permanent backend failure carries `backend_status` and is NOT deferrable.** `backend_error`
covers a transient 502 and a permanent 400 alike, so the code alone cannot say whether retrying is
worth anything — and the proxy already knows, because it declined to retry internally
(`is_transient_backend_error`: *"a real 4xx / other-5xx is deterministic → surface"*). Keeping that
to itself meant the proxy gave up on a permanent failure and advised the caller to retry it.

The envelope therefore carries `backend_status`, the status the **backend** returned, whenever the
failure came from one. A `4xx` other than `408` and `429` is permanent; everything else, and an
absent field, stays deferrable — so a client pointed at an older proxy behaves exactly as before.
This narrows `backend_error` only, never `backpressure`.

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
said nothing about who may pause a backend fleet-wide.

🚨 **BOTH a network gate and a credential, as of 2026-09-01 — and this changed.** Until then an
address could BE an admin: a request from loopback presenting nothing at all could pause a backend,
re-weight a caller's quota or flip a runtime flag, and the audit trail recorded the change with
`key_id: null, source: "ip"`. That default belongs to the inference door, where "already on the box"
is a fair proxy for "allowed" on a local-first proxy; the admin plane and then the UI were added on
the same port and inherited it without the question being re-asked. **An address is a gate now and
never a grant.**

- **Reach** — `ROADSTEAD_ADMIN_NETS` names who may reach the plane at all. Loopback is always in the
  set and cannot be configured out (it is where the bootstrap key below is usable). Docker-internal
  is in the set by default so a containerised deployment reaches its own plane; **naming any net
  drops it**, because a `/12` is a weak gate and an operator who has named their own nets has said
  what they want. 🚨 The **built-in** half is withdrawn from any request that arrived through a
  trusted proxy (§1.5 rule 4); what `ROADSTEAD_ADMIN_NETS` names is not.
- **Credential** — an API key with the `admin` scope, presented as `X-API-Key`, `Authorization:
  Bearer`, or HTTP Basic with the key in the **password** half (which is what the UI uses).

The network is checked **first**, so a refusal from off-net says so and no credential answers it —
and a blocked address never learns whether the key it presented was valid.

🚨 **`ROADSTEAD_ADMIN_NETS` is a BREAKING reinterpretation, not a new variable.** It used to grant
admin to the addresses it named; it now says which addresses may reach the plane, with a credential
still required. An `:admin` flag on a `ROADSTEAD_ACL` address entry likewise no longer grants the
scope. Both are recorded in `CHANGELOG.md`.

**Bootstrap: never open, never locked out.** If no *operator* key is configured, the process mints a
random admin key at startup and logs it. It is not persisted — a new one is minted each boot until
`ROADSTEAD_API_KEYS` is set — and 🚨 it does **not** make the registry "configured" for §1.5 rule 2,
so it cannot flip the inference door's identity regime. Counting it there would 401 every OpenAI
client that sends a placeholder `Authorization` header, because the proxy generated a key for its own
dashboard.

🚨 **An authenticated non-admin identity is refused even from a host in the admin nets.** Once a
caller says who it is, its privileges are that identity's; inheriting the host's would mean a scoped
key could only ever widen access and never narrow it, which makes it worthless on the machine it runs
on. Since 2026-09-01 no host confers the scope in the first place, so this holds trivially — it is
kept stated because the property it protects is the one a future "convenience" default would break.

🚨 **A resolved, admin-scoped WRITE still clears one more gate: CSRF (`IdentityResolver.admin_denial` /
`_csrf_denial`), as of 2026-09-04.** A network gate and a credential are not enough on their own,
because HTTP Basic — the scheme the management UI needs, §3.7 — is exactly the kind of credential a
BROWSER attaches to a request on its own, cookie or not: a cross-site `<form enctype="text/plain">`
POST is a CORS *simple request* (no preflight), the browser attaches the victim's cached Basic
credential to it automatically, and a `text/plain` body can still be syntactically valid JSON. Three
checks, on every mutating (non-GET/HEAD/OPTIONS) route on both `/rs/v1/admin/*` and the legacy
`/v1/admin/*` spellings, in this ONE place rather than per handler — and, because the decision is
taken on the METHOD rather than from a route list, on every other surface that funnels through the
same gate too, `POST /v1/calls/log` (§3.11) being the one that is not an `admin/` path:

1. **`Content-Type: application/json` is required** (media type; parameters such as `charset` are
   fine) — otherwise `415`. A cross-site form cannot send that media type without a preflight.
2. **`Sec-Fetch-Site: cross-site` is refused** with `403`. Sent by every modern browser on a
   cross-origin request and absent from curl/an SDK — its absence is allowed, only that one value
   is refused.
3. **A Basic-authenticated write additionally requires `X-Roadstead-Request: 1`.** A browser attaches
   a cached Basic credential automatically but never adds an arbitrary header on its own initiative,
   so this is what tells the management UI's own `fetch()` apart from a forged cross-site submission
   riding the same credential. A Bearer/`X-API-Key` caller is never auto-attached by a browser in the
   first place and does not need it — the SDK path is unaffected.

The inference doors (`/v1/chat/completions`, `/v1/embeddings`, `/rs/v1/chat`, `/rs/v1/plan`) do **not**
gain this gate: they take Bearer keys, never a browser-cached credential, so CSRF does not apply there
and adding the check would risk refusing an OpenAI SDK client that sends
`Content-Type: application/json; charset=utf-8`.

**The management plane lives at `/rs/v1/admin/*`.** `/v1` is versioned by OpenAI (§1.7), and this is
the surface most likely to need its own second version — it grows with the product rather than with
somebody else's published standard. The four control routes that predate it keep their `/v1/admin/*`
spelling *and* gain the new one: same handler, same gate, both paths served, so no existing consumer
breaks and an operator has one prefix rather than two.

🚨 **§3 mints no error code of its own.** The same call as §1.6 and §1.7, for the third time. Every
refusal here is one of §2.1's existing codes — `invalid_api_key` (401), `access_denied` (403),
`invalid_request_error` (400/404/409/**415**, the last added for the CSRF content-type check above).
In particular a control action that could not be **persisted** is not an error: see §3.2.

| Route | Purpose |
|---|---|
| `GET /rs/v1/admin/config` | The configuration sources, and everything written that is **not in force** (§3.5). |
| `GET /rs/v1/admin/keys` | The key registry, redacted. Never a key, never a digest. |
| `POST /rs/v1/admin/keys` | Enrol a credential. Returns the secret **once** (§3.3). |
| `DELETE /rs/v1/admin/keys/{key_id}` | Revoke one credential, whatever declared it. |
| `POST /rs/v1/admin/keys/{key_id}/rotate` | Issue a successor and retire this one, as **one** action (§3.3). Returns the new secret once. |
| `GET /rs/v1/admin/callers` | Per caller: identities, quota (declared vs in force), DRR budget, spend, live occupancy. |
| `PATCH /rs/v1/admin/callers/{agent_id}` | Edit one caller's quota (§3.4). Partial; absent fields untouched. |
| `GET /rs/v1/admin/providers` | Providers and endpoints: declared vs discovered capacity, credential presence, prices, health. Includes endpoints the catalog declares that nothing is serving. |
| `POST /rs/v1/admin/providers/{provider}/credential` | Supply the value for the provider's `api_key_env` (§3.9). Write-only, **never persisted**. |
| `POST /rs/v1/admin/endpoints/{ep}/status` | Bring a declared endpoint into service or take it out — `active` \| `planned` (§3.9). |
| `GET /rs/v1/admin/providers/{provider}/models` | The models this provider could serve, priced — for choosing one. Refused for an engine that serves a single model and names it. |
| `PUT`/`PATCH`/`DELETE /rs/v1/admin/providers/{provider}` | Create, edit or delete a provider (§3.10). |
| `PUT`/`PATCH`/`DELETE /rs/v1/admin/endpoints/{ep}` | Create, edit or delete an endpoint (§3.10). |
| `GET /rs/v1/admin/audit` | Who changed what, and when (§3.8). Reports its own bound and durability. |
| `POST /rs/v1/admin/endpoints/{ep}/pause` · `POST /v1/admin/endpoints/{ep}/pause` | Drain an endpoint: background defers, interactive fast-fails, the poller stops probing. Auto-opens an annotated PLANNED maintenance window. |
| `POST /rs/v1/admin/endpoints/{ep}/resume` · `POST /v1/admin/endpoints/{ep}/resume` | Re-probe, **re-discover capacity**, drain the deferred queue, close the window. |
| `GET`/`POST /rs/v1/admin/flags` · `GET`/`POST /v1/admin/flags` | Read/flip runtime flags; persisted to JSON, survives restart. |
| `GET`/`POST /rs/v1/admin/maintenance` · `GET`/`POST /v1/admin/maintenance` | List, or backdate a closed window for a restart done without draining. |
| `GET /rs/v1/admin/ui` | The operator UI (§3.7). **Only when `ROADSTEAD_ADMIN_UI` is set** — otherwise the route does not exist. Refuses with 401 + `WWW-Authenticate: Basic`. |
| `GET /rs/v1/admin/stream` · `GET /v1/stream` | Live `call.completed` + `metrics` SSE. Admin-gated since 2026-09-01. The alias exists because `EventSource` cannot set a header (§3.7). |
| `POST /v1/calls/log` | Ingest a call the proxy did not schedule — audio, imagegen, OCR (§3.11). Admin-gated by the same gate, on a path that is not an `admin/` one; refuses anything the proxy records natively. |

Open surfaces: `GET /v1/status` (per-endpoint health, capacity, reliability counters),
`GET /v1/timeouts` and `GET /v1/timeout-advice/shadow-report` (§1.4), `GET /v1/recent` (§3.11),
`GET /v1/inflight`, `GET /v1/history`, `GET /v1/series`, `GET /v1/metrics/cost-model` and
`GET /v1/timeouts/stalls` (§3.12), `GET /metrics` (Prometheus), `GET /health`, `GET /readyz` (fails
closed on readiness-critical endpoints), and the `/v1/fleet/*` analytics family together with
`GET /v1/usage` (§3.6; `/v1/fleet/top-callers` and `/v1/fleet/cache-attribution` are in §3.12,
`/v1/fleet/cache-stats` in §3.11).

The three `/rs/v1` inference routes are **not** open — they are gated like the inference doors
(§1.7). `GET /rs/v1/models` in particular is a map of the fleet: slot counts, live occupancy, health
and prices. An unenrolled caller has no more business reading that than dispatching to it.

🚨 **`/v1/status` and `/metrics` have external consumers** — in the origin fleet a gateway, a
ground-truth verifier and a web UI all read them. Treat their top-level key names as public API.

**`GET /metrics` — Prometheus text exposition (`text/plain; version=0.0.4`), open, no credential.**
In-memory reads only (the scheduler, the 300s rolling window, since-boot counters), so a scrape
costs nothing and a busy proxy is not slowed by being watched. It **never raises**: a render failure
logs and returns an empty body rather than a 500, because a monitoring endpoint that can take the
process down with it is worse than one that goes quiet. Layer-split timeout counts and
would-timeout% are deliberately **not** here — they need SQL over the event tables and live on
`GET /v1/timeouts`.

Every series it can emit is listed below, and 🔒 **the names are contract** (`docs/compatibility.md`).
A dashboard or alert rule may rely on this being the whole set — `tests/test_metrics_names.py` reads
this table back and fails if the builder emits a name that is not here, or if a name here stops being
emitted. `_total` on a `gauge` row is a legacy suffix on a since-boot tally that can be reset by a
restart; it is kept rather than corrected because the name is the contract.

| Series | Type | Labels | What it is |
|---|---|---|---|
| `roadstead_endpoint_slots_total` | gauge | `endpoint` | Configured max concurrent slots. |
| `roadstead_endpoint_inflight` | gauge | `endpoint` | Requests dispatched and in flight. |
| `roadstead_endpoint_queued` | gauge | `endpoint` | Requests waiting in queue. |
| `roadstead_endpoint_queued_by_band` | gauge | `endpoint`, `band` | Queued requests split by priority band. |
| `roadstead_queue_wait_ms` | gauge | `endpoint`, `quantile` | Queue-wait latency (ms). `quantile` is `0.5` or `0.95`; absent for an endpoint with no samples in the window. |
| `roadstead_backend_latency_ms` | gauge | `endpoint`, `quantile` | Backend latency (ms), same quantiles and same absence rule. |
| `roadstead_recent_timeouts_5m` | gauge | `endpoint` | Timeouts in the last 5 min. |
| `roadstead_recent_requests_5m` | gauge | `endpoint` | Requests in the last 5 min. |
| `roadstead_slot_seconds_5m` | gauge | `endpoint` | Slot-seconds consumed in 5 min. |
| `roadstead_endpoint_utilization_pct` | gauge | `endpoint` | 5-min slot utilization %. Emitted only where `max_slots > 0`. |
| `roadstead_endpoint_healthy` | gauge | `endpoint` | 1 if the endpoint is healthy (not paused), else 0. |
| `roadstead_structured_empty_rate` | gauge | `endpoint` | Fraction of structured responses carrying no answer (30m window). 🚨 **Emitted only once the endpoint clears the sample floor** — an unevaluated endpoint must not publish a 0/0 that reads as "healthy", so absence here is not zero. |
| `roadstead_structured_samples_30m` | gauge | `endpoint` | Structured responses in that window; same sample-floor rule. |
| `roadstead_empty_completion_total` | gauge | `endpoint` | Empty (position-0-EOS) completions, counted each time the fail-loud gate trips. |
| `roadstead_truncations_total` | gauge | `endpoint`, `caller`, `structured` | Output-cap (`finish_reason=length`) hits, split `structured="true"`/`"false"` so a rule can alert on structured truncations alone — a climbing structured series is a caller's `max_tokens` set too low. |
| `roadstead_dispatched_total` | counter | — | Since-boot scheduler dispatches. |
| `roadstead_completed_total` | counter | — | Since-boot scheduler completions. |
| `roadstead_timeouts_total` | counter | — | Since-boot scheduler timeouts. |
| `roadstead_scheduler_alive` | gauge | — | 1 if the scheduler loop ticked recently. |
| `roadstead_poller_alive` | gauge | — | 1 if the capacity poller iterated recently. |
| `roadstead_writer_thread_alive` | gauge | — | 1 if the SQLite writer thread is alive. |
| `roadstead_alerts_active` | gauge | `severity` | Standing alert conditions by severity — `CRITICAL`, `ERROR`, `WARNING`, `INFO`, all four always emitted. |

The three liveness gauges exist because a dead poller, scheduler or DB-writer used to be visible
only on `/health` and in the log — nothing a TSDB rule could fire on.

**Log markers are not contract.** The greppable `ROADSTEAD_*` markers (`ROADSTEAD_TRUNCATION`,
`ROADSTEAD_STRUCTURED_EMPTY`, `ROADSTEAD_FAILOVER_ENTER`, …) are 🔓 internal — grep them, but pin an
alert to a metric or to a `/v1/status` field, not to log wording that may be reworded.

### 3.2 What a control action promises 🚨

Every mutating route answers with the change it made **plus** `persisted` and, when that is false, a
`reason`.

🚨 **A change that cannot be persisted still takes effect.** The runtime store is unwritable in an
embedded deployment, on a read-only disk, and in a test. Refusing a revocation on those grounds is a
correctness argument answered, in the moment, by a breach — so the mutation applies in memory and the
response says it will not survive a restart. That is a fact an operator can act on; a 503 is not.

🚨 **A runtime edit never rewrites your config file.** `models.yaml`, `agents.yaml` and a keys file
stay exactly as they were written, comments included. Runtime changes go to a separate JSON overlay
(`ROADSTEAD_ADMIN_STORE`, default `<data dir>/admin_overlay.json`) that is **layered over** those
files at startup: enrolments and quota overrides applied on top, revocations applied last. So what
you wrote and what the API changed remain two separately readable things, which is what makes "who
changed this" answerable at all.

### 3.3 Keys 🚨

**A generated secret is returned exactly once**, by the `POST` that creates it, and cannot be
recovered: the registry holds only the SHA-256 digest. Losing it costs a revoke-and-enrol, which is
the correct price — a surface that could re-read a key is a key store.

**A plaintext secret is never accepted.** There is no `key` field; a secret in a request body lands
in an access log, a proxy buffer and a shell history. To migrate an existing credential, send its
`key_sha256`.

**The first enrolment changes the identity regime for every caller**, and the response says so: until
then a presented key is ignored and the address decides (§1.5 rule 2); from then on an unrecognised
key is a 401.

🚨 **Revoking the LAST key changes it back, and the response says that too.** An empty registry is
not in play at all, so a presented key — including the one just revoked — is ignored again and the
source address decides. Nothing is escalated (the credential confers nothing either way), but
*revoked means refused* stops being true for a caller whose address is enrolled. Rule 2 is not
narrowed to hide this: it exists so a deployment with no keys is not broken by the placeholder
`Authorization` header every OpenAI client sends, and a registry that stayed in play once populated
would 401 exactly the deployment that has just emptied it on purpose.

Disclosures arrive in a **`warnings` array**, not a string: two of them can be true of one action —
revoking an env-declared key that is also the last one is both — and a single field means the second
silently overwrites the first.

**Revocation is never refused on provenance grounds.** A key declared in `ROADSTEAD_API_KEYS` or a
keys file can be revoked at runtime and the revocation survives a restart — but this plane cannot
edit an environment, so the response warns that the declaration will outlive the reason it is dead.

🚨 **`admin_readonly` NARROWS `admin`, and can never widen anything.** A credential with
`admin: true, admin_readonly: true` reaches every `GET` on the management plane and is refused, with
a 403 that says why, on every method that is not one. It is refused at enrolment when `admin` is not
set beside it — on a non-admin key the field changes nothing, and an operator who wrote it believes
they have issued a safer credential than they have. The same narrowing is spelled `:readonly` in the
shared identity grammar (§1.5), so `ROADSTEAD_API_KEYS` and `ROADSTEAD_ACL` express it too.

The read/write split is taken from the **HTTP method** — `GET`, `HEAD` and `OPTIONS` read, everything
else mutates — and not from a list of write routes. A route list is a second thing to keep in step
with the table above, and when it falls behind, the failure is silent and in the widening direction.

🚨 **An address may be narrowed the same way, and a narrowing beats an overlapping grant.**
`127.0.0.1=ops:admin:readonly` in `ROADSTEAD_ACL` is read-only even though loopback is a built-in
admin net: if the widest overlapping grant won, that line would silently be a full grant, since every
operator writing it is on loopback. A narrowing another grant can cancel is not a narrowing.

🚨 **`GET /rs/v1/admin/callers` reports each key's band, not one number for the caller.** A
credential may name its own band, which beats the agent's configured `default_priority` (§1.1), and
many keys to one `agent_id` is a supported shape — so "the band in force for this caller" has no
single value. `identities.keys` therefore carries an object per key: `key_id`, the `priority` it
declares (**`null` when it declares none**, which is a different statement from declaring
`P3_INGESTION` and resolves differently), `overrides_agent_default`, and its `may_assert` grant. The
agent's own configured band stays reported as `declared_priority` beside its `effective_priority`.

🚨 **`may_assert` is the one editable field on this plane that WIDENS.** Every other quota knob moves
a share, a band or a cap, and none can express a rejection (§1.6's write boundary); this one names
`agent_id`s a credential may bill. It is safe on the same argument `substitution` is: the operator
grants, the caller can only spend inside the grant, and a name outside it is a 403 rather than a
silent re-bill. A key listing its **own** `agent_id` is refused rather than trimmed — it reads as
though the list is exhaustive, and an operator who believes that will later wonder why the key still
works with the entry removed.

`GET /rs/v1/admin/keys` publishes it. It is a list of fair-share names, not a credential, and "which
callers may this key bill" is answerable nowhere else. A **rotation carries the grant to the
successor**, unlike the expiry: the grant is a policy the operator made about this identity, and a
rotation that silently dropped it would break every delegated caller at the moment the key changed.

#### Lifecycle: expiry, rotation, binding

**A key expires or it does not.** `expires_at` (an absolute date in a keys file, `expires_in_s` — a
**duration** — over the API) makes a credential stop working at an instant, without anybody having to
remember to revoke it. A key with no expiry never expires, which is what every key was before
2026-09-01, so an existing registry behaves exactly as it did.

🚨 **An expired key gets its own sentence, and the same code.** Both an expired and an unknown key are
`401 invalid_api_key` and neither falls back to the address — a caller can act on no distinction, so
§2.1 mints no second code. What differs is the message, because the *operator* reading the log can:
"not registered" sends them to check whether they pasted the right string, and "expired at T" sends
them to issue a successor. A restart never extends a key: the store holds the absolute instant, not
the duration it was created from.

**`POST /rs/v1/admin/keys/{key_id}/rotate` is one action because the manual version is two calls in
an order that matters — and both orders are wrong.** Enrol-then-revoke leaves a window where the
successor is live and the caller does not have it; revoke-then-enrol leaves one where nothing works.
The successor **inherits** the predecessor's policy (agent_id, priority, deadline floor, admin scope,
binding): a rotation is a new secret for the same identity, and changing policy in the same call
would make one request do two things, of which the unreviewed one is the dangerous one.

🚨 **`overlap_s` defaults to 0 — the predecessor is revoked immediately, and the response says so in
a `warnings` entry.** The other default is tempting and wrong: the usual reason to rotate is that the
old credential should stop working, and a rotation that silently left it alive is the one an operator
believes they have completed. With an overlap the predecessor is given an **expiry** rather than a
tombstone, so it keeps working for the deployment window and then stops on its own — and it survives
a restart, because a timer would not. If the successor cannot be registered the predecessor is
**untouched**: a rotation that revoked the old key and then failed to mint the new one is an outage.

🚨 **An overlap may shorten a predecessor's life and never lengthens it.** `overlap_s: 86400` on a
key the operator gave two hours retires it at the two hours it already had, and says so — a rotation
quietly extending a credential is the same widening refused everywhere else here.

🚨 **A successor inherits the policy but not the expiry, and the response says so.** Inheriting an
absolute instant would mint a successor that expired at the predecessor's moment — possibly seconds
later — and the original *duration* is not recoverable, because `created_at` on an env- or
file-declared key is process start rather than enrolment. So the successor is permanent unless
`expires_in_s` is given, and a rotation of a key that *had* an expiry warns that it now has none.
Silently weakening a control the operator deliberately set is the one thing this plane must not do.

**`bind` is an ADDITIONAL constraint on a key, never a way for one to widen what an address grants.**
A bound key is refused when presented from outside its CIDRs, and is otherwise exactly the credential
it always was — it does not *become* an address identity inside the binding. An unparseable entry is
a 400 at enrolment and matches nothing at resolution: a narrowing that failed open would be worse
than no narrowing at all.

🚨 **A binding is only as trustworthy as `ROADSTEAD_TRUSTED_PROXIES`, and `GET /rs/v1/admin/keys`
says which case you are in.** It is checked against the *resolved* address (§1.5 rule 4), so with no
trusted proxy configured that is the TCP peer and a caller cannot spoof it; with one configured it is
whatever the front proxy reported. The same `bind: ["10.0.0.0/8"]` is therefore a network-level fact
on a direct deployment and a statement about what a proxy vouches for behind one. The `binding` block
in the read view reports `checked_against` rather than leaving an operator to infer it — the §3.5 rule
applied to a security control.

🚨 **Keys are flat, and that is the answer to team-level quotas rather than a gap in it.** The quota
holder, the DRR fair share and the spend cap are all keyed on `agent_id`, never on the key — so
several keys naming one `agent_id` already give a team one budget with per-key revocation and per-key
priority. `GET /rs/v1/admin/keys` groups by `agent_id` so the structure is visible rather than
inferable.

### 3.4 Caller quotas 🚨

`PATCH /rs/v1/admin/callers/{agent_id}` accepts exactly the fields an `agents.yaml` stanza accepts:
`weight`, `max_balance_ss`, `default_priority`, `degrade_ok`, `spill_ok`, `daily_spend_usd`,
`requests_per_minute`. An unknown field is a **400 that names the known set** — never a silent drop,
on the surface whose purpose is to expose silent drops. For both thresholds `null` is *no threshold*
and `0` is a real one (*no paid spend at all*; *this caller should not be sending*), and they are one
keystroke apart.

🚨 **An edit changes policy, never history.** A new weight moves the replenish rate and leaves the
deficit already run; lowering a cap charges nobody retroactively. Clearing a balance on a config edit
would hand a fresh allowance to precisely the caller being reweighted because it consumes too much.

🚨 **§1.6 is inherited here, not re-implemented.** There is no field with which to express a
rejection — every knob changes a share, a band or a threshold, and crossing a threshold still costs
one priority band and paid spill and nothing else. A `blocked` field, or a `max_requests` that
*refused* the request after it, would break that without touching a line of admission code, which is
why the editable set is closed and pinned.

That is the line `requests_per_minute` had to stay on the right side of, and the distinction is the
**consequence**, not the unit. A field counting requests is fine; a field that answers a request with
"no" is not. `requests_per_minute` reaches exactly the same two levers a spend cap reaches, mints no
code, and is unreachable from any path that can refuse — which is why it could be added to a set whose
whole property is that nothing in it can express a rejection.

### 3.5 The gap — what you wrote that is not in force 🚨

`GET /rs/v1/admin/config` reports the configuration **sources** (catalog, agents, keys, ACL, flags,
the runtime store and whether it is writable) and a list of **notices**: every knob an operator wrote
that no code reads.

Three loaders accept a fixed set of keys and drop the rest — `models.yaml`'s `policy:` block,
`agents.yaml`'s stanzas, and a keys-file entry. Each has silently ignored a real knob at least once,
and the cost is always the same: a dropped `spill_ok` is indistinguishable from a caller who never
opted in. The keys are still dropped (a typo must not stop a fleet booting) and still logged — but
they are now **retained** and readable here, because the operator who reads a boot log and the
operator who asks why a knob does nothing are usually not the same person, a week apart.

An empty `notices` list is the healthy state. It is not the same as "no config was loaded": the
`sources` block says which files were read.

`sources` also carries **`trusted_proxies`** — what is trusted, and the flag
`builtin_admin_nets_apply_to_forwarded: false` — beside **`admin_nets`**, split into `builtin` and
`operator`. Which of those two an admin grant came from decides whether it survives a proxy being
put in front (§1.5 rule 4), and an operator whose only admin path was "curl from the box" should
read that here rather than meet it as a 403.

#### 3.9 Bringing an endpoint into service

Two writes, and they are deliberately the *small* pair. 🚨 **Neither creates anything.** Both resolve
their subject through `models.yaml` and 404 what is not declared there — adding a provider or an
endpoint that the catalog does not contain is a file edit and a restart, because an endpoint is a
routing-table entry that discovery, health and the DRR denominator all key on. Promoting one that is
already written down is a far smaller claim: the stanza has been parsed and validated already, and
the only thing that changes is membership of the routing table.

**`POST /rs/v1/admin/providers/{provider}/credential`** — body `{"value": "..."}`.

Sets the environment variable the provider's `api_key_env` names. It takes effect on the next
request, because a provider reads the variable at call time rather than at startup.

- 🚨 **Write-only.** No surface reads it back — not the value, not a prefix, not a digest, not a
  length. §3.5's rule is unchanged; this adds a way in, not a way out.
- 🚨 **Never persisted, and the response says so** rather than leaving it to be discovered at the
  next restart. The admin overlay holds key *digests* and has never held a secret; an outbound
  provider key in a JSON file on disk is a different posture. The durable path is the environment.
- A provider with no `api_key_env` is a 400, not an invented variable name.
- The audit record names the **variable**, never the value.

**`POST /rs/v1/admin/endpoints/{ep}/status`** — body `{"status": "active"|"planned"}`.

- 🚨 **A promotion whose credential does not resolve is REFUSED.** `models.yaml` says in its own
  words what `planned` is for: a deployment that has not set the key "should not have an endpoint in
  its routing table that cannot serve". Promoting without it would put exactly that into the routing
  table, from the surface whose purpose is reporting the gap. The ordering follows from the refusal
  rather than being imposed: credential first, then promote.
- 🚨 **A demotion with work in flight is REFUSED.** Removal is the dangerous direction — the request
  path reads the endpoint's config after dispatch, so pulling it out from under live work is a null
  dereference. Pause already drains, so the safe order exists; pause, then demote.
- `on_demand` and `retired` are catalog-only. One changes how health probes, the other records that
  a name is gone; neither is a routing decision to make from here.
- **The status persists** (the overlay's `endpoints` section) and is re-applied at startup through
  the same function, so a promoted endpoint comes back configured identically.
- 🚨 **But the startup replay applies the SAME credential rule**, because the credential does not
  persist and the status does. Without that, a restart would be the one moment a routed endpoint
  that cannot serve appears — the exact condition the refusal above exists to prevent, arriving
  through the back door. A promotion whose variable is unset stays in the overlay, logs a warning
  naming the variable, and takes effect the moment it is set and promoted again. It is not lost and
  it is not silently in force.

#### 3.10 Writing the catalog

**`PUT`** replaces a runtime stanza, **`PATCH`** merges into it, **`DELETE`** removes the name.
Bodies use `models.yaml`'s own field names, because they are `models.yaml` stanzas.

🚨 **`models.yaml` is never written.** Runtime stanzas live in the admin overlay and are layered over
the file, exactly as an endpoint status override always was. Your comments, your formatting and your
hand-authored intent survive a bad save, and every response says the file is unchanged. The
Configuration view shows declared beside in-force so you can see which is which.

🚨 **The overlay contributes catalog STANZAS, not a second model of an endpoint.** A runtime fragment
is merged into the file's raw dict *before coercion*, so it is parsed, defaulted and validated by
exactly the code that parses a file-authored stanza. There is one catalog format.

- A stanza for a name the file also declares is a **partial, merged over** it.
- A stanza for a name the file lacks **stands alone** — that is creation.
- A tombstone removes the name. Applied after the file, so a later statement wins.

🚨 **Validation is by construction:** the candidate catalog is built and the write is refused if it
complains. The file loader treats the same complaint as non-fatal (a typo must not stop a fleet
booting); from a request it is a `400`, because there is an operator who can fix it now.

**Refusals:** deleting an endpoint with work in flight (pause first — that drains); deleting a
provider endpoints still name (they are listed); an unknown field (never a silent drop); a field
called `api_key` (a stanza names `api_key_env`, a *variable*, and never a key).

**Not supported: renaming.** A rename is a delete plus a create, and the DRR budget, the spend ledger
and the timeout model are all keyed on the endpoint name. Carrying that history silently to a new
name, or silently dropping it, are both wrong; the operator should choose.

The same principle shapes the other two read views. `GET /rs/v1/admin/providers` reports each
endpoint's **declared** capacity (the `models.yaml` seed) beside what is **in force** (what discovery
left), plus whether the engine publishes it at all — so "discovery agreed" and "discovery never ran"
are distinguishable, which they are not on `/v1/status`. `GET /rs/v1/admin/callers` reports quota as
three blocks — `in_force`, `declared` (what the file set) and `runtime` (what this API changed) — so
an override reads as a difference rather than a label.

### 3.7 The management UI 🚨

`GET /rs/v1/admin/ui` — one static HTML page, vanilla JS, **no bundler and no external references at
all**. It ships in the wheel, so it is public surface: the `Content-Security-Policy` it is served
with is `default-src 'none'` with `connect-src 'self'` and no host permitted anywhere, and a test
sweeps the file for a `<script src>`, a webfont or an `@import`.

**The route does not exist unless `ROADSTEAD_ADMIN_UI` is set.** Off means absent, not refusing —
the same posture as `ROADSTEAD_TRUSTED_PROXIES` and `ROADSTEAD_REQUIRE_API_KEY`, because a
capability that widens what is reachable is an operator's decision and an HTML door is reachable by
things that would never send an API request on purpose.

**It refuses with a 401 and `WWW-Authenticate: Basic`, where the plane behind it answers 403.** That
deviation is deliberate and applies to the door only. The 401/403 split is right for an API client,
which reads the two differently; it is a dead end for a browser, where a 403 produces no password box
and an operator arriving with no credential — which is everyone, the first time — has no way to
answer the refusal. The challenge is identical whether or not any key is configured, so it discloses
nothing about which identity regime (§1.5) is in play.

🚨 **No cookie is minted — that does NOT make CSRF unreachable, and this line used to claim it did.**
The browser attaches the credential to any request to this origin on its own, cookie or not, which is
what makes `GET /v1/stream`'s alias at **`/rs/v1/admin/stream`** work at all: `EventSource` cannot set
a header, so a page reaches an authenticated stream only through credentials the browser attaches by
directory, and that is the directory the page was challenged in. Same handler, same gate — and the
same mechanism that makes a cross-site `<form enctype="text/plain">` POST able to carry that
credential too, as a CORS *simple request* with no preflight and a body that can still be valid JSON.
**The real protection is the CSRF gate described above** (`IdentityResolver.admin_denial` /
`_csrf_denial`, applied to every mutating route on both prefixes): a strict
`Content-Type: application/json`, a refusal on `Sec-Fetch-Site: cross-site`, and — for a
Basic-authenticated write specifically — the `X-Roadstead-Request: 1` header this page's own
`fetch()` sends and a forged submission cannot. Recorded in `CHANGELOG.md`.

⚠️ **`GET /v1/stream` is admin-gated as of 2026-09-01** and was not before. A frame there names the
caller, endpoint, tokens and timing of every call the fleet serves — the live form of
`/rs/v1/admin/callers`, which has been gated since it existed. Recorded in `CHANGELOG.md`.

### 3.8 The audit trail 🚨

`GET /rs/v1/admin/audit`. The fifth reporting seam, and the one the other four cannot be: they all
report a **state**, and a state cannot say who put it there. §3.5 shows a quota that is not what the
file says; only this says which credential moved it, and when.

**Every mutating admin route records, including the four that predate this plane.** Flags,
pause, resume and maintenance do not touch the admin overlay and would otherwise record nothing — and
a trail covering only the routes that happen to persist would be worse than none, because an operator
reading it assumes completeness. *"Who paused `tier2`"* is exactly the question it would silently
fail to answer. `tests/test_admin_audit.py` drives every mutating admin route in the table above and
fails if one records nothing.

**A record names the credential, never carries one.** `actor` is `{key_id, agent_id, source,
address}` — the same rule that governs every other readout here: not the key, not the digest.
Both the key label and the address are recorded **always**, even though one is usually redundant: a
record showing an address and no `key_id` means an address-derived admin made the change, which is a
meaningful and slightly alarming thing to find, and collapsing them into one actor string would hide
which factor actually authorized it.

🚨 **It reports its own limits as data, not as prose you have to know to look for.** `persisted` says
whether the trail survives a restart — **in-memory is the default**, because no admin store is
configured by default — and `dropped` counts the records the bound (`capacity`, 500) has discarded.
It is an operator-facing change trail, **not a security log of record**: it cannot outlive its own
bound, and a trail that presented itself as complete while being neither durable nor unbounded would
be the `finish_reason` repair again — a thing that looks like an answer and silences the question.

**A record is written on the loop and reaches disk off it**, riding the same `to_thread` write as the
change it describes (docs/internals.md: *mutate on the loop, persist off it*). One consequence follows and is
not treated as a bug: when the store is unwritable the trail applies and does not survive, exactly
like the change it records. A trail that refused to record an action the plane had already taken
would make the log *less* truthful, not more.

It is a `GET`, so a **read-only** admin scope (§3.3) reaches it — which is the point. The operator who
cannot change anything is often exactly the one auditing what changed.

### 3.6 `/v1/fleet/*` analytics — response schemas

Chased to column level 2026-08-31. Every field in the three schemas below is pinned by
`tests/test_fleet_analytics_schema.py`, which drives the real producers against a seeded
`queue.db` and reads **this section** back — so an added, renamed or dropped field fails the suite
rather than silently breaking a dashboard. §3.12 pins seven more the same way. §3.11 documents the
three call-metrics routes that ship beside these and are **not** pinned that way.

All three run **off the event loop** via `asyncio.to_thread` (§5c): a heavy `GROUP BY` over the
whole-fleet completions table must never stall scheduling under a hot dashboard.

🚨 **These read `proxy_completions`, so every window is bounded by `completions_retention_s`**
(default 30 days). "Total" means *total retained*, not total ever.

#### `GET /v1/fleet/activity` → `fleet_activity(window_s, bin_s)`

Query: `window` (default `24h`, floor 60s, **capped at 7 days** — the shared clamp's default
cap, not 24h), `bin` (defaults from the window).

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

### 3.11 The rest of the call-metrics plane — `/v1/recent`, `/v1/calls/log`, `/v1/fleet/cache-stats`

Three routes that ship beside the family above and were never written down: the completion feed, the
non-LLM ingest, and the prefix-cache observability read. Transcribed from the producers 2026-09-04.

⚠️ **Unlike §3.6, nothing reads this section back.** `tests/test_fleet_analytics_schema.py` pins the
three schemas above in both directions; these three are a hand transcription, so the date is a
freshness marker rather than a guarantee.

#### `GET /v1/recent` → `recent_requests(limit)`

The completion feed — the most recent calls the proxy scheduled, newest first. Open, like the rest of
the analytics family; it names callers and endpoints, never a payload.

Query: `limit` (default 50, **capped at 200**; anything unparseable falls back to the default rather
than erroring).

| field | type | meaning |
|---|---|---|
| `requests` | list | Descending by `completed_at`. `[]` with no DB — the only degraded shape here. |

`requests[]`:

| field | type | meaning |
|---|---|---|
| `request_id` | string | The id Roadstead minted (§4.1), the one `X-Roadstead-Request-Id` carries. |
| `agent_id` | string | The resolved caller identity (§1.5), not anything a body claimed. |
| `endpoint` | string | Endpoint class as persisted. |
| `call_site` | string | The finer-grained attribution key — `<agent_id>.openai_compat` for OpenAI-door traffic (§1.1). |
| `priority` | int | The band, numeric. |
| `input_tokens` | int | |
| `output_tokens` | int | |
| `duration_s` | float | Backend time, 2dp. Excludes queue wait. |
| `queue_wait_ms` | float | Time spent waiting for a slot, 1dp. |
| `status` | string | `ok`, `error`, `truncated`, … as persisted — not an HTTP status. |
| `completed_at` | string \| null | **ISO-8601 UTC**, not the epoch float the other analytics rows carry. `null` when the row has no completion time. |

🚨 **`completed_at` is the one timestamp in this document that is a string.** Every other analytics
route returns epoch seconds. A consumer that pools rows across §3.6 and this route has to convert one
of them.

#### `POST /v1/calls/log` → `persist_external_call(...)`

**Ingest for a call the proxy did not schedule** — audio, image generation, OCR, translation. It
exists so one store answers "what did the fleet spend?", rather than the proxy's own traffic living
here and everything else living somewhere a dashboard has to join against.

🚨 **Admin-gated**, and one of the two admin-gated routes that are not an `admin/` path — the other
is `GET /v1/stream` (§3.7), gated since 2026-09-01. The gate is `deny_non_admin`, so the network
gate, the `admin` scope and — because this is a mutation — the three CSRF checks of §3 all apply. In
practice a pusher must send `Content-Type: application/json` (else `415`) and, if it authenticates
with HTTP Basic, `X-Roadstead-Request: 1`.

Body — every field optional except the endpoint, which may arrive under any of three names:

| field | type | meaning |
|---|---|---|
| `endpoint` \| `provider` \| `unit` | string | **Required**, first non-empty of the three wins. Absent or blank → `400`. |
| `kind` | string | What sort of call this was; default `external`. Refused when it names proxy-native LLM work — see below. |
| `agent` \| `agent_id` | string | The caller to attribute it to; default `unknown`. |
| `call_site` | string | Finer-grained attribution; defaults to `kind`. |
| `caller_id` | string | Sub-identity, recorded as given. |
| `request_id` | string | Default `ext-<uuid4 hex>`, which is also what the response echoes. |
| `input_tokens`, `output_tokens` | int | Default 0. Malformed values fall back to 0 rather than erroring. |
| `latency_ms` | float | Default 0. |
| `duration_s` | float | Default `latency_ms / 1000` — send either, not both. |
| `status` | string | Default `ok`, or `error` when `success` is present and false. |
| `success` | bool | Only consulted when `status` is absent. |

Success is `200 {"ok": true, "request_id": "…"}`, and the record is fanned out on the SSE stream
(§3's `/v1/stream`) as a `call.completed` frame with `priority: "P2_POST_TURN"` — a pushed call has
no band of its own, so one is asserted rather than left null.

🚨 **A `409` means the proxy already recorded this call itself.** Two guards, because the same
double-count arrived under both spellings: a `kind` of `chat`, `embed`, `rerank` or `llm` is refused
outright, and so is any `endpoint` that normalizes to a class the proxy serves — the 2026-06-11
rerank double-count arrived with `kind: "external"` and a rerank endpoint, and only the second guard
would have caught it. Push what the proxy cannot see, never what it can.

The other refusals are `400` (unparseable JSON, or no endpoint) and `500 {"error": "ingest failed"}`
— the ingest is best-effort and a failed write is logged rather than raised. ⚠️ These three bodies
carry `error` alone, **no `code`**: §2.1's codes are minted by the gate in front of this handler
(`401`, `403`, `415`), not by the handler itself.

#### `GET /v1/fleet/cache-stats` → `build_fleet_payload(...)`

**Prefix-cache observability**: what share of prompt tokens the backends served out of their KV
prefix cache, and which call_sites are throwing that reuse away by varying their leading block.
Computed from the periodic snapshots the proxy takes, not from a live scrape.

Query: `window` (default `7d`, floor 60s, **capped at 30 days**). ⚠️ Until 2026-09-05 the cap was
7 days: the handler passed `30 * 86400` positionally, which is `_clamp_window`'s DEFAULT slot — the
fallback for an unparseable string — leaving the keyword cap at a week, so `?window=30d` returned
seven days of snapshots and echoed `window_s: 604800` while doing it.
`tests/test_analytics_window_clamp.py` pins the window the producer is handed, since the payload
cannot show the difference.

| field | type | meaning |
|---|---|---|
| `models` | list | One row per routed chat endpoint, ascending by class. |
| `offenders` | list | Misaligned call_sites from each endpoint's latest screen, ROI-descending, **capped at 25**. |
| `rollup` | object | Fleet totals — see below. |
| `trend` | object | `{endpoint: [{t, rate}]}`, the windowed hit-rate between each consecutive pair of snapshots. `t` is the later snapshot's epoch seconds. An endpoint with fewer than two counter-bearing snapshots is absent. |
| `drift` | list | `{call_site, endpoint, from, to}` — call_sites whose front-loaded-prefix share collapsed below its own trailing baseline, i.e. someone edited a prompt and broke the cacheable leading block. Shared with the periodic drift alarm, one implementation. |
| `snapshots` | int | How many snapshots the window held. `0` makes every rate above `null` — read it before reading anything else. |

`models[]`:

| field | type | meaning |
|---|---|---|
| `endpoint` | string | Endpoint class. |
| `label` | string | The class plus every alias that resolves to it, `/`-joined, so a row reads as the names callers actually use. |
| `engine` | string | The backend engine, or `?` when the catalog does not say. |
| `actual_hit_rate` | float \| null | Lifetime rate from the latest snapshot's cumulative counters, 4dp. **`null` unless the engine publishes prefix-cache counters** — that is a missing instrument, never a 0% hit rate. |
| `window_hit_rate` | float \| null | The rate across the window: Δhits / Δqueries between the first and last counter-bearing snapshots, 4dp. This is the one to quote — `actual_hit_rate` is cumulative since the backend booted. |
| `screen_misaligned` | int | Call_sites in the latest screen whose verdict is `misaligned`. |
| `screen_total` | int | Call_sites in that screen at all. The denominator `screen_misaligned` needs. |

`offenders[]` carries the screen row plus its `endpoint`: `{call_site, reqs, avg_tok, lcp_pct, jacc,
verdict, wasted_tokens, roi, endpoint}`. `lcp_pct` (front-loaded common prefix, char-level) and
`jacc` (word-set overlap) are **proxies** for block-level KV reuse, not measurements of it — a
`misaligned` verdict says two requests share most of their words while sharing little of their
opening, which is the shape a cache cannot exploit.

`rollup`:

| field | type | meaning |
|---|---|---|
| `wasted_cacheable_tokens` | int | Σ `wasted_tokens × reqs` over **every** misaligned screen row, including the ones the 25-row `offenders` cap drops — the size of the prize, not a measured loss. |
| `captured_pct` | float \| null | Fleet windowed hit rate, 4dp. `null` when nothing in the window carried counters. |
| `est_prefill_s_saved` | float | Prefill seconds the cache hits avoided, 1dp, at a fixed tokens-per-second constant. An estimate, and named one. |

### 3.12 The rest of the analytics plane — live, historical and forensic reads

🚨 **The whole analytics family is OPEN to any admitted caller — `/v1/recent`, `/v1/inflight`,
`/v1/history`, `/v1/series`, `/v1/usage`, `/v1/fleet/*`, `/v1/timeouts/*`, `/v1/metrics/cost-model` —
while `GET /v1/stream`, which carries the same facts, is admin-gated (§3.7). The split is
deliberate, and it is about the bound rather than the sensitivity.** Everything here is a POLLED
aggregate over a window the caller asks for and the server caps: 30 days of cache snapshots, 168
hours of usage, 200 rows of feed, one run's stalls. One request buys one bounded answer, and what it
contains is identities and arithmetic — agent, call_site, endpoint, token counts, timings — never a
payload, a prompt or a completion. The stream is those same frames with the bound removed: a
subscriber holds the connection open and receives every call the fleet serves for as long as it
stays connected, which is a different capability from reading a dashboard. An operator who wants the
family closed as well will not find a flag here: today the remedy is the deployment's rather than
the process's — put the proxy behind a reverse proxy (README, *Behind a reverse proxy*) or make it
reachable only from the admin nets, and gate it there. That is the shape `SECURITY.md` already asks
for around the admin plane, and it is why an open analytics route is a documented decision to
disagree with rather than a finding to report.

🚨 **Two fields are exempt, because the argument above does not cover them.** "Identities and
arithmetic" is accurate for the whole family except `GET /v1/status`'s
`reliability.admin_ips_seen` (the source addresses seen on admin routes) and
`reliability.placeholder_bearers.by_address` (the addresses of callers presenting a credential that
means nothing). Those are ADDRESSING, and specifically the addressing of the privileged callers and
the weakly-credentialled ones — a target list and a shortlist of who to try first, from one
unauthenticated GET. Since 2026-09-05 they degrade for any caller who could not reach the admin
plane from where they are standing:

| field | admin-net caller | anyone else |
|---|---|---|
| `admin_ips_seen` | `{route: [addresses]}` | `{route: count}` |
| `placeholder_bearers` | `{count, by_address}` | `{count, by_address_count}` |

The route stays open and still answers — this is a redaction of two fields, not a narrowing of the
family. The counts are what the fields are read for: an ACL-tightening go/no-go asks *has anything
but me touched an admin route*, and the placeholder inventory asks *is it empty yet*. Both are
answered without naming anyone. The predicate is the admin plane's own NETWORK gate
(`acl.may_reach_admin`), not a credential check, so the operator reading status on the box still
sees the addresses — and a FORWARDED address never counts as on-box, so the reverse-proxy collapse
in the README cannot hand a stranger the list by arriving through a sidecar.
Pinned by `tests/test_status_address_disclosure.py`.

Seven routes the code shipped and this document did not: the live board, the history and series
charts, the calibration state, the two fleet rollups that are not in §3.6, and the per-caller stall
lookup. Transcribed from the handlers 2026-09-05, and — unlike §3.11 — **pinned in both directions**
by `tests/test_fleet_analytics_schema.py`, which drives these handlers against a seeded `queue.db`
and reads the tables below back. The tables describe the **HTTP response**, not the producer's
return: where a handler wraps or adds something (`/v1/history`'s envelope, `/v1/inflight`'s `ts`,
`hit_rate_source` on the attribution rollups) the row is here and the test drives the handler.

#### `GET /v1/inflight` → `inflight_snapshot(now)`

**The live board** — every request currently executing, plus per-endpoint occupancy. A pure
in-memory read off the scheduler: no DB, no query params, and nothing here survives a restart. The
proxy is the dispatch authority, so this is the authoritative answer to "what is flowing right now";
the same snapshot is pushed as the SSE `inflight` frame for sub-second feel.

| field | type | meaning |
|---|---|---|
| `requests` | list | Everything dispatched and not yet complete, **descending by `elapsed_s`** — the longest-running call first, which is the one an operator is looking for. |
| `per_endpoint` | object | `{endpoint: occupancy}` for **every configured endpoint**, including the idle ones. |
| `ts` | float | Server wall-clock, epoch seconds, stamped by the handler. The snapshot's own arithmetic is monotonic; this is the only wall-clock value in it. |

`requests[]`:

| field | type | meaning |
|---|---|---|
| `request_id` | string | The id Roadstead minted (§4.1). |
| `endpoint` | string | The endpoint CLASS the request was routed to (a role resolves to one, §1.1). |
| `served_model` | string \| null | The model id that class actually serves, `null` when the catalog does not say. |
| `backend` | string \| null | `host:port` for a local engine (`192.0.2.13:8000`), the base URL for a remote provider, `null` when the address is genuinely unknown. |
| `agent` | string | The resolved caller identity (§1.5). Named `agent` here and `agent_id` on `/v1/recent` (§3.11) — same value, two spellings. |
| `call_site` | string | The finer-grained attribution key. |
| `priority` | string | 🚨 The band's **enum NAME** (`P1_TURN_SUPPORT`), where `/v1/recent` returns the same band as an **int**. A consumer joining the two converts one of them. |
| `band` | string | The scheduling band, lower-cased — `interactive` \| `foreground` \| `background`. |
| `input_tokens` | int | The ESTIMATE admission used, not a measurement — the real count is not known until the backend answers. |
| `elapsed_s` | float | Since dispatch, 2dp. Excludes queue wait: a queued request is not in this list at all. |
| `estimated_remaining_s` | float | From the cost model, 2dp. An estimate, and it does not shrink to zero reliably — read it as a hint about which call is long, not as a countdown. |

`per_endpoint{}` — one object per endpoint class:

| field | type | meaning |
|---|---|---|
| `max_slots` | int | Concurrency the endpoint is configured for; `0` for an endpoint the catalog declares and nothing serves. |
| `in_flight` | int | Dispatched, not yet complete. |
| `queued` | int | Waiting for a slot, all bands. |
| `queue_by_band` | object | The same depth split three ways: `{interactive, foreground, background}`. Always all three keys, zeros included. |

#### `GET /v1/history` → `history_buckets(hours, bucket_minutes)`

**Time-bucketed history**, the chart behind the dashboard's activity view. Same corpus as §3.6's
`fleet_activity`, cut differently: buckets carry a per-endpoint AND a per-agent breakdown, and the
timestamps are ISO strings rather than epoch bin starts.

Query: `hours` (default 4, **capped at 168**), `bucket_minutes` (default 5, clamped to **1–60**).
Both fall back to the default on anything unparseable rather than erroring.

| field | type | meaning |
|---|---|---|
| `buckets` | list | Ascending by time, **only the buckets that have rows** — an idle stretch is absent, not a zero-filled bucket. `[]` with no DB. |

`buckets[]`:

| field | type | meaning |
|---|---|---|
| `start` | string | **ISO-8601 UTC**. The window's own start plus `n × bucket_minutes`, so the boundaries move with the request rather than sitting on the clock. |
| `end` | string | ISO-8601 UTC, `start + bucket_minutes`. |
| `per_endpoint` | object | `{endpoint: {...}}` for the endpoints active in this bucket. |
| `per_agent` | object | `{agent_id: {...}}` for the agents active in this bucket. |

`per_endpoint{}`:

| field | type | meaning |
|---|---|---|
| `requests` | int | |
| `ok` | int | `status = 'ok'`. |
| `errors` | int | Everything else. |
| `avg_duration_s` | float | 2dp. ⚠️ An AVERAGE of the SQL averages when a bucket spans several groups — good enough for a sparkline, not a statistic to quote. |
| `total_in_tokens` | int | |
| `total_out_tokens` | int | |

`per_agent{}`:

| field | type | meaning |
|---|---|---|
| `requests` | int | |
| `slot_seconds` | float | Summed `duration_s`, 1dp — how much backend time this agent occupied in the bucket. The fairness number; `requests` alone hides one caller holding a slot for a minute. |

#### `GET /v1/series` → `endpoint_series(endpoint, window_s, bin_s)`

**One endpoint's own chart** — the detail-modal counterpart to `fleet_activity`, with latency
percentiles the fleet view does not carry.

Query: `endpoint` (**required** — absent or empty is a `400` with `{"error": "endpoint query param
required"}`, the one refusal in §3.12); `window` (default `24h`, floor 60s, capped at 7 days); `bin`
(defaults from the window: 60s under an hour, 300s under six, 600s under a day, else 3600s). An
unknown endpoint is not an error — it is an empty series, because a class that has served nothing and
a class that does not exist are the same absence of data.

| field | type | meaning |
|---|---|---|
| `endpoint` | string | The **normalized** class, so an alias comes back as what it resolved to. |
| `window_s` | int | Echo of the resolved window. |
| `bin_s` | int | Echo of the resolved bin width. |
| `now` | float | Server wall-clock at computation. ⚠️ **Absent when the DB is unopened** — as in §3.6. |
| `calls_series` | list | Ascending by `ts`, only the bins that have rows. |

`calls_series[]`:

| field | type | meaning |
|---|---|---|
| `ts` | int | Bin start, epoch seconds. |
| `n` | int | Completions in the bin. |
| `fail_pct` | float | Percentage of `n` whose `status != 'ok'`, 1dp — a PERCENTAGE where §3.6's `calls[]` carries a `fails` COUNT. |
| `tokens_in` | int | |
| `tokens_out` | int | |
| `p50` | float \| null | Milliseconds, 1dp. `null` — not `0.0` — when the bin carried no latencies, which is the opposite convention to `fleet_activity`'s `p95`. |
| `p95` | float \| null | Milliseconds, 1dp. |
| `p99` | float \| null | Milliseconds, 1dp. |
| `avg_in_toks` | int \| null | Mean `input_tokens`, rounded. `null` when every row in the bin had none. |

#### `GET /v1/metrics/cost-model` → `cost_model.snapshot()`

**What the deadline arithmetic is calibrated to.** The proxy computes timeouts rather than accepting
them (§1.2), and this is the state that computation reads: per endpoint, the prefill constant and
the decode curve, both EWMA-calibrated from real completions. Read it when a recommended timeout
looks wrong — the answer is usually a `samples` count of 0.

No query params. **The response is an object keyed by endpoint class**; the table describes one
value. It is `{}` before startup registers the endpoints, and an endpoint absent from it has no
model rather than a zeroed one.

| field | type | meaning |
|---|---|---|
| `endpoint` | string | The class, repeated inside its own entry. |
| `max_slots` | int | Concurrency the curve below is indexed against. |
| `prefill_k` | float | Seconds per input token, 6dp. |
| `decode_tps` | list | Tokens/second at occupancy 1..`max_slots`, 1dp, **1-indexed by position** — `decode_tps[0]` is the single-caller rate. Empty when capacity discovery has cleared it. |
| `prefill_ewma` | object | The tracker behind `prefill_k` — see below. |
| `per_call_site` | object | `{call_site: {p50_output_tokens, samples}}` — how long this caller's answers usually are, which is what sizes its deadline. Absent call_sites have never completed a call here. |

`prefill_ewma` (the shape every EWMA tracker in the snapshot uses):

| field | type | meaning |
|---|---|---|
| `value` | float | Current EWMA, 6dp. |
| `stddev` | float | 6dp. |
| `p95` | float | 6dp. |
| `samples` | int | 🚨 **0 means the number beside it is a DEFAULT, not a measurement.** Read this first. |

`per_call_site{}`:

| field | type | meaning |
|---|---|---|
| `p50_output_tokens` | int | The EWMA of observed output length, truncated to int. |
| `samples` | int | How many completions it is built from. |

#### `GET /v1/fleet/top-callers` → `top_callers(window_s, per_endpoint)`

**Who is using each endpoint** — the attribution view, ranked per endpoint rather than fleet-wide, so
a busy endpoint cannot bury a small one's top caller.

Query: `window` (default `1h`, floor 60s, capped at 7 days); `per_endpoint` (default 5, clamped to
**1–20**).

| field | type | meaning |
|---|---|---|
| `window_s` | int | Echo of the resolved window. |
| `now` | float | Server wall-clock. ⚠️ **Absent when the DB is unopened.** |
| `providers` | object | `{endpoint: [caller, …]}`, each list descending by `n` and truncated to `per_endpoint`. Endpoints with no traffic in the window are absent, not empty. |

`providers[][]`:

| field | type | meaning |
|---|---|---|
| `agent` | string | The caller. Falls back to `"—"` (em dash) on a NULL, the same defensive-only fallback §3.6 documents for `usage_rollup`. |
| `n` | int | Completions in the window. |
| `tokens_in` | int | |
| `tokens_out` | int | |

#### `GET /v1/fleet/cache-attribution` → `cache_attribution(window_s, limit)`

**Per-caller MEASURED prefix-cache hit rate** — the Tier-2 complement to §3.11's
`/v1/fleet/cache-stats`, which reports the backends' own counters. Hit rate here is
`sum(cached_tokens) / sum(input_tokens)` over **only** the rows a backend actually attributed; rows
it did not are counted as `unattributed_calls` and excluded from the ratio, so a backend that reports
nothing can never masquerade as a 0% hit rate. Reads `kind = 'chat'` only — nothing else has a prefix
cache to hit.

Query: `window` (default `1h`, floor 60s, capped at 7 days); `limit` on `by_call_site` (default 40,
clamped to **1–200**).

| field | type | meaning |
|---|---|---|
| `window_s` | int | Echo of the resolved window. |
| `now` | float \| null | Server wall-clock, **`null` when the DB is unopened** — present-but-null rather than absent, unlike the routes above. |
| `by_call_site` | list | Per `(call_site, endpoint)`, descending by `calls`, capped at `limit`. |
| `by_endpoint` | list | Per endpoint, descending by `calls`. |
| `fleet` | object \| null | One row over everything in the window. `null` only with no DB. |

`by_call_site[]`:

| field | type | meaning |
|---|---|---|
| `call_site` | string | |
| `endpoint` | string | |
| `calls` | int | Chat completions in the window. |
| `attributed_calls` | int | Of those, the ones carrying a `cached_tokens` count. |
| `unattributed_calls` | int | `calls - attributed_calls`. **Not** zeros — a backend that does not report. |
| `cached_tokens` | int | Summed, over attributed rows only. |
| `attributable_input_tokens` | int | Summed `input_tokens` over the SAME rows, so the ratio's denominator matches its numerator's population. |
| `hit_rate` | float \| null | 4dp. `null` when nothing in the group was attributable — n/a, never 0%. |

`by_endpoint[]` — the same fields without `call_site`, plus one the handler adds:

| field | type | meaning |
|---|---|---|
| `endpoint` | string | |
| `calls` | int | |
| `attributed_calls` | int | |
| `unattributed_calls` | int | |
| `cached_tokens` | int | |
| `attributable_input_tokens` | int | |
| `hit_rate` | float \| null | 4dp — **overwritten by the backend's own windowed rate when this endpoint had nothing attributable of its own**, and left strictly alone when it did. Real per-request measurement always wins. |
| `hit_rate_source` | string | 🚨 **Present only on an overlaid row** (`backend_prefix_cache_metrics`). Its absence means the number beside it was summed from per-request counts. A consumer must treat it as optional; `tests/test_cache_attribution_overlay.py` pins both halves. |

`fleet` — identical to a `by_endpoint` row with `scope` in place of `endpoint`:

| field | type | meaning |
|---|---|---|
| `scope` | string | Always `"fleet"`. |
| `calls` | int | |
| `attributed_calls` | int | |
| `unattributed_calls` | int | |
| `cached_tokens` | int | |
| `attributable_input_tokens` | int | |
| `hit_rate` | float \| null | 4dp. Overlaid — as a query-weighted mean of the backends' rates — only when NOTHING in the window was attributable. |
| `hit_rate_source` | string | Optional, exactly as above. |

#### `GET /v1/timeouts/stalls` → `stall_aborts(caller, since, until)`

**Did the backend stall under THIS run?** The forensic counterpart to `/v1/timeouts` (§1.4), which
aggregates and answers "is the fleet under pressure?". A consumer that died needs the other question
— an aggregate has already thrown away which caller and which second — and the two failures it
separates, an upstream substrate stall and its own hang, are indistinguishable from the outside.

Query: `caller` (**a PREFIX**, matched `caller%`, because caller ids carry a per-run suffix; `_` and
`%` are escaped, so `dsh_write` cannot match `dshXwrite`); `since` / `until` (epoch seconds,
**inclusive both ends** — a stall recorded on the exact terminalisation second counts). `since`
defaults to 0 and `until` to now.

🚨 **An unknown caller and an empty window are `count: 0`, never an error** — the consumer is meant to
read "no evidence" and "cannot tell" the same way. An **empty `caller` is also `count: 0`**, and
deliberately: a blank prefix would `LIKE`-match every caller in the window and hand back somebody
else's stall as evidence about your own run.

| field | type | meaning |
|---|---|---|
| `caller` | string | Echo of the prefix, stripped. |
| `since` | float | Echo, epoch seconds. |
| `until` | float | Echo. |
| `count` | int | `len(rows)`. |
| `rows` | list | Ascending by `occurred_at`. |

`rows[]`:

| field | type | meaning |
|---|---|---|
| `request_id` | string | |
| `occurred_at` | float | Epoch seconds — when the stream was killed, which is the instant the consumer observed as its own failure. |
| `endpoint` | string | |
| `abort_reason` | string | Only the backend-stall reasons; a capacity decision of ours is not in this table. |
| `caller_id` | string | The full id, suffix included — what the `caller` prefix matched. |
| `agent_id` | string | |
| `call_site` | string | |
| `elapsed_s` | float | How long the call had been alive when it was abandoned. |
| `layer` | string | Where it was abandoned — `stream`, `transport`, … |

#### ⚠️ The unopened-DB shapes are narrower here too

The same trap §3.6 documents, in four more places. With no DB connection:

| producer | degraded shape |
|---|---|
| `history_buckets` | `[]` → the response is `{"buckets": []}`, the only one whose shape is unchanged |
| `endpoint_series` | `endpoint`, `window_s`, `bin_s`, `calls_series` — **no `now`** |
| `top_callers` | `window_s`, `providers` — **no `now`** |
| `cache_attribution` | every key, but `now` and `fleet` are **null** |
| `stall_aborts` | `[]` → `count: 0`, envelope intact |

`/v1/inflight` and `/v1/metrics/cost-model` read no DB at all and are unaffected — though the cost
model is `{}` until startup registers the endpoints, which is the same "empty is not zero" reading.

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

## 7. The client SDK — `roadstead.client`

Ships inside the package, the same way `roadstead.testing` does, and is what
another project installs to speak §1.7.

```python
from roadstead.client import AsyncRoadsteadClient

async with AsyncRoadsteadClient("http://proxy:42161", api_key=KEY) as rs:
    plan = await rs.plan(intent="reasoning", est_in=8_000)
    result = await rs.chat(intent="reasoning",
                           messages=[{"role": "user", "content": "..."}])
    print(result.content, result.attribution.endpoint,
          result.timing.queue_wait_ms)
```

`AsyncRoadsteadClient` is the implementation; `RoadsteadClient` is a blocking
wrapper that owns a private event loop on a worker thread, so it composes with
sync code and refuses to run inside a loop rather than deadlocking one.

**All three payload types, one route.** `chat()`, `embed()` and `rerank()` differ only by the
`payload_type` they put on the envelope (§1.7.2); `call(payload_type=…, payload=…)` sends any other
one a newer proxy may accept, unchecked, so the SDK does not gate a deployment on a literal
transcribed here. All four return a `CallResult` — named for the route, not for chat, since
2026-09-02 — whose `.response` is the backend's body untouched, so a hybrid embedder's
`{dense, sparse, colbert}` and a reranker's `{results: […]}` arrive whole. 🚨 `embed()` is
therefore the **lossless** embedding path and `/v1/embeddings` is deliberately not (§1.1).
`ChatResult` remains as an alias.

**Three things it does that a hand-rolled `httpx.post` would not:**

🚨 **It classifies errors on the `code`.** `RoadsteadError.deferrable` reads
§2.1's machine-readable code first and falls back to §2.2's legacy marker
substrings only for a proxy older than the SDK. That is the migration §2.2 asks
for — off prose-matching, without dropping it while callers are still on it.

🚨 **It sets the client keepalive.** §1.3's ordering invariant is the client's
side to get right, and the SDK sets it rather than leaving it to whoever
configures the pool.

🚨 **It keeps unknown fields.** Every typed view exposes `.raw`, so a proxy
newer than the SDK is usable rather than lossy.

**It imports nothing from the Roadstead server**, by AST-enforced rule
(`tests/test_client_sdk.py`). Two reasons: a consumer sending an HTTP request
should not be installing Starlette, uvicorn, PyYAML and jsonschema; and a client
that read the server's own constants would agree with it *by construction* and
could never catch a drift. The contract lives in `roadstead/client/_wire.py` as
literals transcribed from **this document**, which the same test reads back —
the two-ended pin `tests/wire_contract.py` uses from the server side.

`enrichment_from(response.headers)` reads §1.8 off an OpenAI-door response, for
callers not yet ready to move.

---

## 6. Verification status

**Verified against source 2026-08-31:** the route table location, the non-standard request fields,
the error codes, the deferrability mechanism and its marker substrings, the context-overflow marker
verbatim, backend dispatch paths, the capacity-discovery asymmetry, and the metric-name prefixes.

**Updated 2026-09-01:** §1.5 (identity and API keys) is new, `invalid_api_key` joins §2.1, and §3
restates the admin gate now that a key can carry the scope — Workstream B in `docs/roadmap.md`.
§1.6 (spend, spill and the degrading threshold) and the price row in §4.3 are Workstream D. 🚨 §1.6
adds **no error code**: that a spend threshold cannot produce one is the contract.

**Updated 2026-09-01, Workstream C:** §1.7 (the enriched API at `/rs/v1/*`), §1.8 (enrichment
headers on the OpenAI door), §1.9 (the `/v1/submit` migration map) and §7 (the client SDK) are new.
🚨 §1.7 adds **no error code either** — an unresolvable intent is the existing `unknown_endpoint`,
because a caller already classifies that and a second spelling of "nothing here can serve you" would
buy nobody anything. `POST /v1/submit` is **removed**; §1.9 is the map and `CHANGELOG.md` carries the
reason. Read back by `tests/test_client_sdk.py` and `tests/e2e/test_enriched_api.py`.

**Updated 2026-09-02:** §1.1 is now **one table per door**, because the single table it replaced
documented the union of two of them — see `CHANGELOG.md` for the full account. `payload_type` joins
§1.7.2, which makes `/rs/v1/chat` the route for embeddings and reranks as well as chat completions
and gives rerank a door again for the first time since `/v1/submit` was removed. §7 records that the
SDK speaks all three, and that `ChatResult` is now `CallResult`. Pinned by
`tests/test_openai_door_fields.py` (each §1.1 table against what the named module reads, both
directions) and `tests/test_enriched_envelope_coverage.py` (every field the enriched door reads is
one the SDK can send).

**Updated 2026-09-04 — the routes and fields the code shipped and the document did not.** Four
routes gained a section: `GET /v1/timeouts` and `GET /v1/timeout-advice/shadow-report` in §1.4,
beside the advice they report on; `GET /v1/recent`, `POST /v1/calls/log` and
`GET /v1/fleet/cache-stats` in the new §3.11, beside the analytics family in §3.6. §2.2 gains the two
deferral markers it was missing — `truncated structured output` and `returned empty completion`,
both of which mean *retry* — and §1.1 states that `X-Agent-Id` and `X-Call-Site` are ignored, since a
migrating fleet reaches for the header spelling before it reads the body table. 🚨 §3.11 is a hand
transcription: unlike §3.6 no test reads it back, and it says so in place.

**Previously INCOMPLETE — both closed 2026-08-31:**
1. ~~Nested response schemas for `/v1/fleet/*` analytics.~~ Chased to column level in §3.6 and
   pinned by `tests/test_fleet_analytics_schema.py`, which reads this document back.
2. ~~Delegator-signature drift between `service.py` and `http_handlers.py`.~~ Audited by AST:
   **30 delegators, 30 identical signatures, zero drift** — the stated contract holds. Now
   continuously enforced by `tests/test_delegator_signatures.py` rather than re-asserted by hand.
