# Configuration reference

Every knob Roadstead has is an **environment variable**. There is no config file for behaviour —
`models.yaml` describes the fleet and `agents.yaml` describes the callers, and both are *data*,
pointed at by variables listed here. The command line carries four flags and nothing else.

This document is the complete list, generated against the source and **pinned by
`tests/test_configuration_doc.py`**, which walks `roadstead/` in both directions: a variable the
code reads and this table omits fails the suite, and so does a variable named here that nothing
reads. That matters more than it sounds — the defect this whole area exists for was a documented
variable name that had silently stopped being read (`CHANGELOG` 2026-09-02), and a reference nobody
checks is exactly how that happens again.

---

## How to read the table

**The 🔒 column.** `docs/compatibility.md` marks one thing stable: the contract in `docs/api.md`.
A variable is 🔒 here **iff `docs/api.md` names it** — that document is what the policy pins, and
twelve of these variables are part of it because a caller's behaviour depends on them (what
authenticates, what is refused, which routes exist at all). Everything else is 🔓 **internal**: it
works, it is supported, and it may be renamed or removed in a release with a `CHANGELOG` line and
no deprecation period. Do not build tooling that greps for a 🔓 name.

**Booleans have exactly two spellings**, and which one a variable uses follows from its default:

| family | default | `1` `true` `yes` `on` | `0` `false` `no` `off` `""` | anything else |
|---|---|---|---|---|
| **opt-in** (default OFF) | off | **on** | off | off |
| **kill-switch** (default ON) | on | on | **off** | on |

The asymmetry is deliberate: an opt-in must not be turned on by a typo, and a kill-switch must not
be turned off by one. Values are stripped and lower-cased. 🚨 One variable does not follow either
family — `ROADSTEAD_PROXY_THINKING_CANARY` tests `!= "0"`, so `false` turns it **on**; the row says
so.

**Numbers are clamped and never fatal.** Every numeric variable is parsed inside a `try`, and an
unparseable value falls back to the default with no error — the proxy starts. The clamp column is
the floor the value is held at; a value below it is raised, not refused.

**Paths are absolute.** A relative path is resolved against the process's working directory, which
in the shipped image is `/app` and in a systemd unit is whatever the unit says. Set absolute ones.

---

## The command line

```
roadstead [--host HOST] [--port PORT] [--log-level LEVEL] [--data-dir DIR]
```

| flag | environment equivalent | notes |
|---|---|---|
| `--host` | `ROADSTEAD_HOST` | bind address |
| `--port` | `ROADSTEAD_PORT` | listening port |
| `--log-level` | `ROADSTEAD_LOG_LEVEL` | root logger level; `httpx` is pinned to WARNING regardless |
| `--data-dir` | `ROADSTEAD_DATA_DIR` | root for all durable state |

**A flag beats its variable, which beats the default.** `--data-dir` is published back into the
environment before the app is built, so the four paths derived from the data dir (`ROADSTEAD_QUEUE_DB`,
`ROADSTEAD_RUNTIME_FLAGS`, `ROADSTEAD_ADMIN_STORE`, `ROADSTEAD_LOG_DIR`) follow it — and each of
those can still be set individually to override the flag for its own file.

🚨 **There is no bare `PORT`, `HOST` or `LOG_LEVEL`.** Every variable answers to the `ROADSTEAD_`
prefix, including these three. A PaaS that injects a bare `PORT` is not configuring this process.

🚨 **`roadstead test …` is reachable only as `python -m roadstead test …`.** The `test` subcommand
tree is dispatched from the module's `__main__` guard, which the console script does not run.

### The default data directory

Resolution order, highest first:

1. `--data-dir DIR`
2. `ROADSTEAD_DATA_DIR`
3. `$XDG_STATE_HOME/roadstead`
4. `~/.local/state/roadstead`

The XDG basedir spec's *state* directory is the right home for this: data that must persist between
restarts, is not a cache and is not configuration, which is `queue.db` exactly. **Changed 2026-09-05
from `/tmp/agents/llmproxy`** — a durable record on storage a reboot deletes. The shipped image sets
`ROADSTEAD_DATA_DIR=/var/lib/roadstead` and is unaffected either way.

Two consequences worth knowing before you upgrade:

* 🚨 **A deployment that relied on the old default will start with an empty database.** Its DRR
  balances, the day's spend and the endpoint drain state are still in `/tmp/agents/llmproxy` until
  something clears `/tmp`; move the directory, or set `ROADSTEAD_DATA_DIR` to the old path.
* 🚨 **With no `HOME` and no passwd entry for the uid** — a distroless image running as an arbitrary
  `runAsUser` — `~` cannot be resolved. Rather than silently write a relative path under the working
  directory, the process warns and falls back to `/tmp/roadstead`, which then trips the
  ephemeral-storage warning as well. Set `ROADSTEAD_DATA_DIR` in that deployment.

---

## 1. Identity, access and the admin plane

Nothing here has a default that grants anything. A proxy with none of it set has no credentials, no
registered addresses and no admin nets beyond the built-in loopback/docker grant — which is the
supported local-first deployment, not an unfinished one.

| variable | 🔒 | type / default | what it does |
|---|---|---|---|
| `ROADSTEAD_API_KEYS` | 🔒 | comma-separated `<secret>=<agent_id>[:priority][:min_timeout_s][:admin]`; empty | Inline API keys. For the one-key container case where a file is ceremony. A presented key authenticates and **overrides** address-based identity. |
| `ROADSTEAD_API_KEYS_FILE` | 🔓 | path; empty | The documented path for anything real — it can carry digests rather than secrets. Read **in addition to** `ROADSTEAD_API_KEYS` when both are set. A path that does not exist logs a warning and loads no keys. |
| `ROADSTEAD_ACL` | 🔒 | comma-separated `<ip-or-cidr>=<agent_id>[:priority][:min_timeout_s][:admin][:readonly]`; empty | Address→identity registrations. Segments are recognised by shape, so their order does not matter. `:admin` also adds the address to the admin nets; `:readonly` narrows that grant to reads. The pre-rename `LLM_PROXY_ACL` is still read (see *Retired spellings*). |
| `ROADSTEAD_ADMIN_NETS` | 🔒 | comma-separated CIDRs; empty | Hosts that need the control plane but no inference identity. Naming any net here **replaces** the built-in loopback/docker admin grant. |
| `ROADSTEAD_REQUIRE_API_KEY` | 🔒 | opt-in; **off** | Refuse any request presenting no credential. The right setting for a proxy reachable from anything wider than one host; wrong for the laptop case, so it is a decision, not a default. Set with no keys configured and every request is refused — startup warns. |
| `ROADSTEAD_TRUSTED_PROXIES` | 🔒 | comma-separated CIDRs; empty | Networks whose `X-Forwarded-For` is believed. 🚨 Until this is set, a forwarded address is ignored entirely — and once it is set, the built-in loopback/docker admin grant no longer applies to a *forwarded* request, because "already on the box" stops being true with a front proxy in the way. |
| `ROADSTEAD_ADMIN_STORE` | 🔒 | path; `<data-dir>/admin_overlay.json` | Where the management plane persists runtime key enrolments, revocations and quota overrides. Unset **and** underivable means changes apply immediately and are lost on restart; `GET /rs/v1/admin/config` reports that in words. |

## 2. Doors and compatibility shims

Three switches that change **which routes exist**. All default OFF, and OFF means the route is not
registered — a request gets the same 404 as any unknown path, rather than a 403 that tells a
scanner the door is there.

| variable | 🔒 | type / default | what it does |
|---|---|---|---|
| `ROADSTEAD_ADMIN_UI` | 🔒 | opt-in; **off** | Registers `GET /rs/v1/admin/ui`, the single-file operator page. An HTML door on a proxy is reachable by things that would never send an API request on purpose, so opening it is an operator's decision. |
| `ROADSTEAD_LEGACY_SUBMIT` | 🔒 | opt-in; **off** | Restores `POST /v1/submit`, the envelope removed in Workstream C, byte-for-byte as `docs/api.md` §1.9.1 published it. A compatibility surface with an expiry date: it exists so a fleet can move its callers one at a time. While on, it emits one deprecation WARNING per `agent_id` per UTC day and a counter on `/v1/status` — that counter is the inventory removal is gated on. |
| `ROADSTEAD_BEARER_PLACEHOLDERS` | 🔒 | comma-separated literals; empty | Bearer values to treat as *no credential at all*, for SDKs that refuse to send an empty `api_key`. A request presenting one is identified by its source address. Warns loudly at startup naming every value, and `/v1/status` → `reliability.placeholder_bearers` names who still needs it. |

## 3. Catalog, durable state and backends

| variable | 🔒 | type / default | what it does |
|---|---|---|---|
| `ROADSTEAD_MODELS_YAML` | 🔓 | path; the shipped `roadstead/models.yaml` | THE routing table: endpoints, providers, aliases, capabilities. The shipped file is an **example with a real schema** — invented hosts at RFC 5737 documentation addresses. Point this at your own. |
| `ROADSTEAD_AGENTS_CONFIG` | 🔓 | path; the shipped `roadstead/agents.yaml` | Per-caller DRR quota overrides (`weight`, `max_balance_ss`, `default_priority`, `degrade_ok`, `spill_ok`, `daily_spend_usd`, `requests_per_minute`). Missing file → empty → every caller gets `AgentQuotaConfig` defaults. 🚨 A key the parser does not know is **silently ignored**. |
| `ROADSTEAD_ON_DEMAND_DISPATCHER_URL` | 🔓 | URL; empty | The GPU-slot dispatcher an `on_demand` endpoint leases from before its model is loaded. A host-side service Roadstead neither owns nor ships, so there is deliberately **no default address**. Unset, an `on_demand` endpoint fails `ensure_loaded` as unreachable — the same clean deferrable error as a dispatcher that is down. |
| `ROADSTEAD_DATA_DIR` | 🔓 | path; `$XDG_STATE_HOME/roadstead`, else `~/.local/state/roadstead` | Root for everything durable — see *The default data directory* below. Also `--data-dir`. The shipped image sets it to `/var/lib/roadstead` and declares that a `VOLUME`; `tests/test_env_var_naming.py` pins both. |
| `ROADSTEAD_QUEUE_DB` | 🔓 | path; `<data-dir>/queue.db` | The durable record — DRR balances, completion rows, the day's spend, endpoint drain state. This is what the bounded SIGTERM drain and the published stop-grace exist to flush; lose it and every restart resets all four. |
| `ROADSTEAD_RUNTIME_FLAGS` | 🔓 | path; `<data-dir>/runtime_flags.json` | Runtime-mutable feature flags, flipped through `POST /v1/admin/flags`. Not env-gated, deliberately: a flag you can flip without a restart is a different thing from a variable. |
| `ROADSTEAD_LOG_DIR` | 🔓 | path; `<data-dir>/logs` | 🚨 **Under** the data dir, not beside it. The deployment contract is that an operator provides ONE persistent path and the application keeps everything it owns inside it; rooted elsewhere, moving the data dir moved the database and quietly left the request log on the container's ephemeral layer. |
| `ROADSTEAD_REQUEST_LOG` | 🔓 | path; `<log-dir>/llmproxy_requests.jsonl` | The per-request JSONL record. |
| `ROADSTEAD_REQUEST_LOG_MAX_BYTES` | 🔓 | int; `DEFAULT_REQUEST_LOG_MAX_BYTES` | Rotation size for that log. Rotation lives in the application because the file is the application's — an external rotation of a handle held open in append mode silently writes to the unlinked inode. |
| `ROADSTEAD_REQUEST_LOG_BACKUPS` | 🔓 | int; `DEFAULT_REQUEST_LOG_BACKUPS` | How many rotated generations to keep. |
| `ROADSTEAD_PAYLOAD_RETENTION_S` | 🔓 | float seconds; `172800` (48h) | How long `queue.db` keeps request payloads before the persistence cleaner drops them. |
| `ROADSTEAD_COMPLETIONS_RETENTION_S` | 🔓 | float seconds; `2592000` (30d) | How long completion rows survive. These are what the spend and savings figures are computed from, so shortening this shortens the analytics window. |
| `ROADSTEAD_WAL_CHECKPOINT_INTERVAL_S` | 🔓 | float seconds; `300` | Cadence of the WAL checkpoint. |
| `ROADSTEAD_INCR_VACUUM_INTERVAL_S` | 🔓 | float seconds; `600` | Cadence of the incremental vacuum. |
| `ROADSTEAD_INCR_VACUUM_PAGES` | 🔓 | int pages; `4000` | Pages reclaimed per incremental vacuum — the bound on how long one costs. |
| `ROADSTEAD_STARTUP_VACUUM_FREELIST_BYTES` | 🔓 | int bytes; `209715200` (200 MiB) | Freelist size above which startup does a full vacuum. High enough that a normal boot never pays for one. |

## 4. Correction and cooldown

Everything in this group is an **incident knob**. The defaults are the measured-correct settings and
none of them is a tuning surface: each one either turns off a repair the proxy performs on a bad
response, or turns on a repair still being shadow-measured. The pattern that recurs is
detect-then-act — a guard ships in a `_SHADOW` mode that logs and counts without touching the
response, and is promoted once the false-positive rate is known.

🚨 **A `_SHADOW` variable does not disable its guard.** It demotes it to detect-only. Turning a
guard off is the guard's own variable.

| variable | 🔒 | type / default | what it does |
|---|---|---|---|
| `ROADSTEAD_PROXY_THINKING` | 🔓 | kill-switch; **on** | Fleet-wide disable for the reasoning path. The per-request `thinking: true` is the real gate — nothing reasons unless a caller asks — so this exists only to shut the path down during an incident. |
| `ROADSTEAD_PROXY_THINKING_BUDGET` | 🔓 | int tokens; `8000`, clamped ≥ 0 | Reasoning headroom **added** to a thinking request's `max_tokens`. Reasoning is generated output and counts against the cap, so too small a budget truncates mid-thought (`finish=length`). Generous by directive: prefer slowness to cutoff, then tune **down** while watching the truncation metric. |
| `ROADSTEAD_PROXY_THINKING_CANARY` | 🔓 | ⚠️ `!= "0"`; **on** | The startup capability probe for the reasoning path. 🚨 The only variable here that is not one of the two boolean families: **only the literal `0` turns it off** — `false`, `no` and `off` all leave it ON. Set it to `0` for a backend that must not receive a probe request. |
| `ROADSTEAD_PROXY_FORCED_REASONING_BUDGET` | 🔓 | int tokens; `1536`, clamped ≥ 0 | The same headroom for an endpoint whose model **always** emits a reasoning trace (`capabilities.reasoning: true`), which the caller cannot switch off. Deliberately much smaller than the budget above: those turns reason a few hundred tokens, and inflating a small `max_tokens` into a large one is its own failure. |
| `ROADSTEAD_PROXY_STRUCTURED_VALIDITY` | 🔓 | kill-switch; **on** | Validate a structured response against the schema the request declared, and fail it loudly (`structured_invalid_json`, marker `LLMPROXY_STRUCTURED_INVALID`) rather than returning something that parses but does not conform. |
| `ROADSTEAD_PROXY_SCHEMA_BACKSTOP` | 🔓 | opt-in; **off** | Repair a response that missed its schema, instead of only reporting it. |
| `ROADSTEAD_PROXY_SCHEMA_BACKSTOP_SHADOW` | 🔓 | opt-in; **off** | Run that backstop in detect-only mode — count what it *would* have repaired. Only meaningful with the backstop enabled. |
| `ROADSTEAD_PROXY_DEGENERATION_GUARD` | 🔓 | kill-switch; **on** | Catch a degenerate response — one long n-gram repeated many times — and re-dispatch with an anti-repetition penalty. A 200-with-garbage is invisible to both the transient-error and the grammar checks; this is the layer that sees it. |
| `ROADSTEAD_PROXY_DEGENERATION_SHADOW` | 🔓 | opt-in; **off** | Demote that guard to detect-and-count, no re-dispatch. The measurement that confirms it never flags legitimate repetition — a chorus, a list. |
| `ROADSTEAD_PROXY_UNIFORM_CORRECTION` | 🔓 | opt-in; **off** | Route the **streaming** and enriched `/rs/v1` responses through the same correction layer the synchronous path already uses: tool-call stream sanitising on both doors, plus truncation and degeneration *detection* over a stream's reassembled content. Off is byte-identical to the historical behaviour. |
| `ROADSTEAD_PROXY_SHADOW_EGRESS` | 🔓 | kill-switch; **on** | Check every grammar-bearing structured response for conformance in shadow: log and count the silent grammar-drops (a backend that dropped the grammar and ran free-form) without ever mutating the response. Zero caller risk, which is why it is on — the point is the per-call-site baseline. |
| `ROADSTEAD_PROXY_ENDPOINT_COOLDOWN` | 🔓 | opt-in; **off** | Cool an endpoint that is failing **intermittently** — the case the 3-consecutive circuit breaker never trips on. |
| `ROADSTEAD_PROXY_ENDPOINT_COOLDOWN_SHADOW` | 🔓 | opt-in; **off** | Detect-only for the same, before it is allowed to take an endpoint out. |
| `ROADSTEAD_PROXY_COOLDOWN_ALLOWED_FAILS` | 🔓 | int; `4`, clamped ≥ 1 | Backend-fault failures inside the window that trip a cooldown. Above the consecutive-failure circuit on purpose, so the two catch different shapes. |
| `ROADSTEAD_PROXY_COOLDOWN_WINDOW_S` | 🔓 | float seconds; `60`, clamped ≥ 1 | The sliding window those failures are counted over. |
| `ROADSTEAD_PROXY_COOLDOWN_DURATION_S` | 🔓 | float seconds; `30`, clamped ≥ 1 | How long a tripped endpoint stays cooled before it auto-recovers. |
| `ROADSTEAD_PROXY_MAX_SLOTS_RECONCILE` | 🔓 | kill-switch; **on** | Reconcile the discovered slot count against what the backend's `/props` reports. Observability only — it never changes routing or admission. |

## 5. Limits and the listening socket

| variable | 🔒 | type / default | what it does |
|---|---|---|---|
| `ROADSTEAD_HOST` | 🔓 | address; `0.0.0.0` | Bind address. Also `--host`. |
| `ROADSTEAD_PORT` | 🔓 | int; `42161` | Listening port. Also `--port`. |
| `ROADSTEAD_LOG_LEVEL` | 🔓 | uvicorn level name; `info` | Also `--log-level`. An unrecognised name falls back to INFO. |
| `ROADSTEAD_MAX_REQUEST_BYTES` | 🔒 | int bytes; `16777216` (16 MiB), clamped ≥ 1 | Cap on an inbound body, enforced at the ASGI layer **before** any handler reads it, so an oversized body never reaches a queue slot. A lied-short `Content-Length` is caught by counting bytes as they arrive. Over it → `413`, `code: invalid_request_error`. Sized for a vision payload with several inlined base64 images. |
| `ROADSTEAD_MAX_RESPONSE_BYTES` | 🔒 | int bytes; `67108864` (64 MiB), clamped ≥ 1 | Running cap on a streamed backend response, checked chunk by chunk, so a wedged or adversarial backend cannot grow this process's memory without bound. |
| `ROADSTEAD_PROXY_SERVER_KEEPALIVE_S` | 🔒 | int seconds; `30` | Idle keepalive close. 🚨 **Must stay above the client's `keepalive_expiry`** so the client always retires an idle socket first — at uvicorn's default 5s the two coincided and a POST reusing a server-closed socket raised `RemoteProtocolError`. Pinned by `tests/test_keepalive_invariant.py` against the published constant in `docs/api.md`. |

## 6. Observability

| variable | 🔒 | type / default | what it does |
|---|---|---|---|
| `ROADSTEAD_PROXY_CACHE_DRIFT_ALARM` | 🔓 | kill-switch; **on** | Each cache-stats cycle, flag call_sites whose front-loaded-prefix share collapsed against their own trailing baseline — the signature of a prompt edit that broke a cacheable leading block. Raises a `CACHE_DRIFT_ALERT` log marker and an `llmproxy_cache_drift` security event, deduplicated. Never changes routing, admission or output. |
| `ROADSTEAD_PROXY_INFLIGHT_INTERVAL_S` | 🔓 | float seconds; `1.5`, clamped ≥ 0.25 | Cadence of the SSE `inflight` reconcile frame behind the live board. The instant dispatch/complete events do the real work; this refreshes elapsed/queue/occupancy and recovers a missed event. Gated on connected clients — free when nobody is watching. |

---

## Retired spellings

Variables were renamed twice: `COLLECTIVE_*` → `ROADSTEAD_*` (2026-08-31) and, for fourteen the
entry point had missed, `LLM_PROXY_*` → `ROADSTEAD_*` (2026-09-02).

* **`COLLECTIVE_*` is not honoured.** Two spellings for one switch is how they end up disagreeing.
  A `COLLECTIVE_`-prefixed variable still set in the environment gets a WARNING at startup naming
  its replacement, because the dangerous half of a rename is not the flag that stops working — it
  is the flag that stops working *in silence*.
* **`LLM_PROXY_*` still works, and warns.** Every variable is read through a helper that tries the
  `ROADSTEAD_` spelling first and falls back, logging the rename. The new spelling always wins when
  both are set, so a migration can be done without deleting anything first.
* **`LLM_PROXY_ACL` is the one legacy name kept without a plan to remove it**, because it
  configures access: a proxy that silently stops recognising its callers on upgrade fails closed in
  the most confusing way available.

## Removed variables

Removed, not renamed — there is no new spelling. A removed name still set in the environment is
reported at startup by the same warning path the retired prefixes use, because a variable that
means *nothing* is the quietest failure of the three: nothing looks wrong, the setting simply does
not happen.

| variable | removed | what to do instead |
|---|---|---|
| `ROADSTEAD_HOT_ROOT` | 2026-09-05 | It only ever composed the default data directory, which is now the XDG state directory. Set `ROADSTEAD_DATA_DIR` to the full path. |

## Names that look like configuration and are not

Three `ROADSTEAD_`-prefixed strings appear in the source and are **not** environment variables the
proxy reads. They are listed so a reader who greps the tree is not left guessing, and so the
bidirectional test above has somewhere honest to put them.

| name | what it actually is |
|---|---|
| `ROADSTEAD_SPILL` | A **log marker**. `service.py` emits it on the line recording one request moved to remote capacity because local was full. Grep for it; do not set it. |
| `ROADSTEAD_TIMEOUT_ADVICE_CAP_S` | A **client-side** variable, named in a `constants.py` comment only, to record that the server's smart-default deadline cap (1800s) is deliberately the same number the client uses. Nothing in this package reads it. |
| `ROADSTEAD_AGENT_NAME` | **Exported, never read.** The entry point sets it (`setdefault`, to `llmproxy`) for the benefit of a surrounding process tree. No module in this package consults it, so setting it changes nothing here. |
