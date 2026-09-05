# Changelog

Notable changes to Roadstead. Breaking changes get their own section with a reason, per
`docs/compatibility.md` — the stable surface is the wire contract in `docs/api.md`; everything else
is internal and changes without an entry.

Pre-1.0: breaks are permitted, but each one is a recorded decision rather than a surprise.

## Unreleased

### Added — the legacy `/v1/submit` door, flag-gated

`POST /v1/submit` was removed in Workstream C, deliberately and recorded. That decision stands and
this does not reverse it — what it did not allow for was a fleet with a dozen callers already
speaking the old envelope and no window in which to move them all at once. Setting
**`ROADSTEAD_LEGACY_SUBMIT`** re-opens the door byte-compatible with what it published
(`docs/api.md` §1.9.1–§1.9.4): the same envelope, the same six-key success shape, the same error
codes and marker substrings, the same `queued`/`admitted`/`chunk`/`done` SSE framing with `data` as
a **string** and no `[DONE]`, and embedding/rerank bodies passed through **verbatim**.

**Default OFF, and OFF means the route does not exist** — a request gets the same `404` any unknown
path gets, the posture `ROADSTEAD_ADMIN_UI` established. It is not a second hot path either: the
door translates into `Lifecycle.handle_submit` like the other two north faces and `wire` picks the
response bytes, so nothing about admission, correction or accounting varies with it.

🚨 **It carries one deliberate departure from §1.5, valid only on this route and only under the
flag:** a caller inside the built-in internal nets is identified by the `agent_id` in its own body,
unchecked — which is what that door always did and what every fleet caller's DRR share depends on.
The edges are narrow and are the point: a **registered** address keeps its registration (making this
door *stricter* than `/rs/v1/chat`, where §1.5 rule 3 lets the body fill in an omitted identity), a
**forwarded** address never reaches the exception at all, a **key** keeps key semantics through
`may_assert`, and an address still never grants admin.

Two additions to the old shapes, both stated rather than left to be discovered by diffing: the `504`
body gains `"status":"error"` (the old one omitted it; every caller reads `.get("status") != "ok"`),
and a `502` carries `backend_status` when the failure came from a backend (§2.2 — no caller pins the
key set of an *error* envelope, and the alternative is this door advising a retry the proxy itself
declined to make).

**Removal is gated on the inventory being empty, not on a date.** `GET /v1/status` →
`reliability.legacy_submits` reports `{count, callers{agent_id: n}}` since boot, and a WARNING names
each caller once per UTC day beside it — `legacy /v1/submit used by agent_id=<x> (call_site=<y>) —
migrate to /rs/v1/chat (docs/api.md §1.9)`. When `callers` stays empty across a representative
window, delete `roadstead/legacy.py`, its two tests, the flag branch in `routes.make_routes` and the
four `WIRE_LEGACY` branches in `lifecycle.py`.

### Fixed — an on-demand endpoint's in-flight lease could never idle-release

`OnDemandManager.ensure_loaded` counts a request in-flight before `handle_submit` has decided
whether it is admitted. Three paths after it used to `return`/reroute a request without a
matching `request_done()`: a paused-endpoint or circuit-open refusal, a load-shed `429`, and a
failover reroute away from an on-demand SOURCE (`Failover.apply` repoints `req.endpoint` in
place, and `record_completion` only ever releases whatever `req.endpoint` names at completion).
All three are deferrable, so a retrying client re-incremented the count and refreshed the idle
clock on every retry — the dispatcher GPU lease never idle-released, bounded only by the
20-minute `_STUCK_INFLIGHT_S` force-release. Fixed by releasing at each of the three points
(`lifecycle.py`); new `tests/test_on_demand.py`.

### Fixed — `/rs/v1/plan` could predict a different priority band than `/rs/v1/chat` dispatches at

`handle_rs_plan` resolved only request → credential, skipping the agent's `agents.yaml
default_priority` step `handle_submit` already ran — so a plan's `timing` (and interactive/
long-form ceiling) could be computed at the wrong band for a caller whose credential declares
none. The four-step precedence is now one shared method, `Lifecycle.resolve_declared_priority`,
called by both routes.

### Added — correction-layer rewrites are now disclosed per call

The schema backstop's in-memory repair/retry and `apply_json_object_guard`'s stripped
`response_format` used to be visible only as fleet-wide `/v1/status` counters — a caller had no
way to know its own response had been rewritten. `X-Roadstead-Corrected` (OpenAI door, §1.8) and
`corrections` (enriched envelope, §1.7.3) now name which of `json_object_stripped`,
`schema_repaired`, `schema_retried`, `schema_unrecoverable`, `degenerate_unrecovered` or
`toolcall_truncated` applied to this response; absent/empty when nothing did.

### Changed — CSRF hardening on the mutating admin plane 🚨 BREAKING

A browser attaches a cached `Authorization: Basic` credential to any request to this origin
on its own, cookie or not — including a cross-site `<form enctype="text/plain">` POST, a CORS
*simple request* that skips the preflight a real cross-site JSON POST would need, whose body
can still be syntactically valid JSON. Every mutating (non-GET/HEAD/OPTIONS) route on the admin
plane — both `/rs/v1/admin/*` and the legacy `/v1/admin/*` spellings, the four control routes
that predate the management plane included — now requires `Content-Type: application/json`
(`415` otherwise) and refuses a request labelled `Sec-Fetch-Site: cross-site` (`403`); a
Basic-authenticated write additionally requires `X-Roadstead-Request: 1` (`403` without it). One
gate — `identity.IdentityResolver.admin_denial` / `_csrf_denial` — reached by every mutating
admin handler already on both prefixes, so nothing had to change per handler.

`docs/api.md` §3.7 previously claimed "no cookie is minted, so CSRF is never reachable on these
mutating routes"; that was wrong, and the section (plus a new §3 paragraph) now says what the
real protection is. `ui/index.html`'s one `fetch()` wrapper (`api()`) sends both headers on
every mutating call automatically, and its module docstring's claim is corrected to match. A
caller scripting the admin plane directly (curl, an ops script) needs to add
`Content-Type: application/json` to every mutating request, and `X-Roadstead-Request: 1` if it
authenticates with Basic. Not applied to the inference doors (`/v1/chat/completions`,
`/v1/embeddings`, `/rs/v1/chat`, `/rs/v1/plan`): they take Bearer keys, never a browser-cached
credential, so CSRF does not apply there and the check would risk refusing an OpenAI SDK
client's `Content-Type: application/json; charset=utf-8`. New `code`: `invalid_request_error`
gains status `415` (§2.1 — §3 still mints no code of its own).

### Added — a request body cap and a streamed-response cap

Neither door had an upper bound on how much it would read. `ROADSTEAD_MAX_REQUEST_BYTES`
(default 16 MiB) refuses an oversized inbound request body with `413`/`invalid_request_error`
(§2.1 — no new code minted, it just gains a status) at the ASGI layer, before `request.json()`
ever runs on it; sized for a legitimate multi-image vision chat payload (`docs/api.md` §1.10).
`ROADSTEAD_MAX_RESPONSE_BYTES` (default 64 MiB) is a running cap on a streamed backend
response, checked line-by-line as `aiter_lines()` yields rather than after the whole thing is
already buffered — a wedged or adversarial backend that never stops talking is aborted as a
`BackendError(502)` rather than buffered without bound (`backend.BackendClientPool.stream`).

### Fixed — the admin overlay was world-readable

`admin_overlay.json` (runtime key digests, quota overrides) is now written mode `0600`. It
always used an atomic temp-file-plus-`os.replace`; only the permission bit was missing.

### Fixed — public-readiness pass: quick-start port, packaging, container hardening

`README.md` and the `roadstead.client` docstrings pointed at port `42100`, a stale number left
behind by the extraction — the code's default has been `42161` throughout. A guard,
`tests/test_docs_port_consistency.py`, now reads the default from `roadstead.config.ProxyConfig`
and fails if a tracked doc or module drifts from it again.

CI now builds the wheel and installs it into a fresh venv as a separate `package` job (import,
`--help`, and package-data presence for `ui/index.html`/`models.yaml`/`agents.yaml`), and runs the
scrub sweep (`tests/test_scrub_sweep.py`) by name so a future `-k`/deselect change cannot quietly
stop running it; checkout now uses `fetch-depth: 0`. `pyproject.toml` gained `build` under the `dev`
extra for this — a build-time dependency, not a runtime one; `tests/test_admin_ui.py`'s runtime-deps
allowlist is unchanged.

`Dockerfile` gained a `HEALTHCHECK` against `/health` (liveness, not `/readyz`'s routing-oriented
readiness — see the Dockerfile comment) using stdlib `urllib`, since the base image carries no
`curl`. The image stays root by default: the one real deployment bind-mounts a root-owned ZFS host
path, and flipping the default user would not make that safer, only silently unwritable at the next
rebuild — non-root there needs the deployment's own `chown`/`chmod` cooperation, tracked rather than
forced by this change.

`roadstead.__version__` now reads the installed distribution's version via `importlib.metadata`,
falling back to `"0.0.0+unknown"` off a bare checkout. Added `SECURITY.md` and `CONTRIBUTING.md`, and
`Homepage`/`Issues`/`Changelog` under `[project.urls]`.

A private Proxmox container id (`CTnnn`) in `docs/roadmap.md` was replaced with "the deployment
sandbox" — `docs/corpus_and_scrub_plan.md` §S1 now flags that the human identifier-sweep half of the
scrub (S5/S7) needs re-running over everything committed after 2026-09-01, since this finding postdates
both passes.

### Added — the durable record names the credential behind a delegated call

`proxy_completions` gains a nullable `key_id`, written **only when a delegation was exercised** —
NULL means the credential IS the `agent_id`, which was an invariant of that table until `may_assert`
landed and is still true of every undelegated row. So a value is always meaningful rather than a
column populated on everything and read on nothing. "Which key ran up `chat-agent`'s bill?" had no answer
otherwise: the admin audit trail records admin *actions*, never dispatch. Migrated through the
existing `_add_missing_columns` path.

### Changed — `GET /rs/v1/admin/callers` reports a band per KEY

**Breaking for a reader of that response:** `identities.keys` was a list of key-id strings and is now
a list of objects — `{key_id, priority, overrides_agent_default, may_assert}`. The bundled admin UI
is updated in the same change.

*Why:* `declared_priority` reports the agent's configured band, and a credential that names its own
beats it. Many keys to one `agent_id` is a supported shape, so **the band in force for a caller has
no single value** — and a scalar field claiming otherwise is the operator plane stating something
untrue about itself, on the surface whose stated purpose is `declared` beside `in_force`. Measured
before the change: agent config `P4_HYGIENE`, credential `P1_TURN_SUPPORT`, request ran
`P1_TURN_SUPPORT`, plane reported `P4_HYGIENE`.

🚨 A key that declares no band reports `priority: null`, not the grammar's default. "Declared
`P3_INGESTION`" and "declared nothing" resolve to the same value and mean opposite things; reporting
the value for both is how this surface would go straight back to being wrong.

### Added — the enriched response says who the call was BILLED to (`identity`)

A fifth block on `/rs/v1/chat`'s envelope and on the streaming `done` frame:
`{agent_id, declared?, honoured?}` — `docs/api.md` §1.7.3.

🚨 **It exists because "ignored" must not also mean "invisible".** A credential with no delegation
grant IGNORES a body-declared `agent_id` rather than refusing it (§1.5 rule 3) — right, because
refusing would break every caller carrying a `/v1/submit` habit over a claim that was already inert.
But without disclosure the caller asks to be `chat-agent`, the work is billed to the credential, and both
outcomes are a 200 with otherwise identical bodies: the `finish_reason` silencer in another costume.
The resolved `agent_id` is always reported; `declared`/`honoured` appear only when the caller
declared something, on the same rule that keeps `substituted` from firing on every intent-routed call
and meaning nothing. No band, queue position or demotion (§1.6) — the value is either the caller's
own word or the name on the credential it presented.

`roadstead.client` exposes it as `CallResult.identity`. Two bugs found while building it, both by
tests rather than by reading: the door overwrites `agent_id` with the resolved identity before
`handle_submit` sees it, so the caller's raw word has to travel separately (`declared_agent_id`); and
the first cut read it with `or`, which cannot tell a door that set `""` from one that set nothing —
so `declared` echoed the resolved identity and `honoured` was `true` on every call, the field firing
on everything and meaning nothing. The same truthiness trap the priority block is compared against
`None` to avoid.

The golden behaviour baseline was regenerated; the diff is the new block and nothing else, which is
the check that it is additive on the wire.

### Added — a key may act as callers its operator granted (`may_assert`)

Landed 2026-09-02. A key entry gains `may_assert`, a list of `agent_id`s the credential is permitted
to act as; a request declaring one of them in `agent_id` is billed to that caller, one outside a
non-empty grant is a **403**, and a key with no grant ignores the field exactly as before.

🚨 **This partially reverses §1.5 rule 3, deliberately, and the reason is measurement.** Rule 3 —
"a key overrides a body-declared `agent_id`" — assumed the security boundary and the fairness
boundary are the same object. On the fleet this proxy is replacing they are not: **38 distinct
`agent_id`s across 197k requests/week, most of them sibling processes inside one container** sharing
a filesystem and a uid. One trust domain, 38 fair shares. Forcing those to be one thing goes wrong
in both directions — one key per `agent_id` puts 38 secrets where one boundary is, which is
labelling wearing authentication's clothes; one key with the identities collapsed destroys what the
DRR weights exist for, and the largest caller alone is 47% of all traffic.

An allowlist is the third answer and it keeps the property that made rule 3 right: **a caller still
cannot claim an identity nobody gave it.** The operator writes the names, the caller picks among
them, and anything else is refused rather than quietly re-billed to the credential — which would put
the work on one caller's bill and the record on another's, with nothing anywhere to say so.

- **Delegation moves the fair-share key and grants no policy.** Band, deadline floor, admin scope
  and `key_id` stay the credential's. A key that could hand itself a different policy by naming
  another agent would be the self-asserted `agent_id` bug restored rather than fenced.
- **The decision lives in `IdentityResolver.delegate` and nowhere else**, guarded by AST — a second
  place deciding what a credential permits is the `_remote_ip` shape, and this one decides who is
  billed. `lifecycle.handle_submit` is where the fair-share key is finally settled; it re-resolves
  from the request rather than trusting what a door put in the body, so it stays safe on its own.
  🚨 The first cut delegated in the door only and `lifecycle` silently undid it — caught by an
  end-to-end test that reads the **completion row**, not the response.
- **`may_assert` is the one editable field on the management plane that WIDENS** rather than narrows.
  Recorded in §3.3 as the exception to §1.6's write boundary rather than left for a reader to notice.
  A rotation carries the grant to its successor (unlike the expiry, which is an absolute instant).
- **An unreadable grant fails CLOSED** — a widening that failed open would be strictly worse than none.

`docs/api.md` §1.7.2 also gains the table this raised: **one agent doing two kinds of work has three
knobs, and only the third needs a grant** — `priority` for when it runs, `call_site`/`caller_id` for
how it is labelled, `agent_id` for whose budget it spends. The third is not a redundant spelling of
the first, because the **DRR balance is one per `agent_id`, shared across bands**: priority orders
the work, but a background storm still drains the balance its own interactive turns spend from.

### Fixed — `agents.yaml`'s `default_priority` did nothing, and the dashboard said it did

Landed 2026-09-02, found while answering "can a caller be given a static band in configuration?"
The answer was yes — and the obvious place to write one was dead.

`default_priority` was parsed by `load_agent_configs`, in the `agents.yaml` allowlist, editable
through `PATCH /rs/v1/admin/quotas`, and **reported by the management plane as the caller's
`declared_priority`** beside its `effective_priority`. The request path never read it. So an
operator setting `default_priority: P4_HYGIENE` got nothing and was told by the dashboard that it had
worked — a gap between two sources for one question, with the operator-facing surface naming the one
not in force. Their defaults did not even agree: `P1_TURN_SUPPORT` in the config dataclass,
`P3_INGESTION` in the identity grammar.

🚨 **The cause was duplication, not a missing lookup.** Every door pre-filled `priority` into the
internal submit body from `principal.priority`, so by the time `handle_submit` looked, the
*identity's default* was indistinguishable from a band the *caller* had asked for — and nothing
below could tell they were different. The doors now pass only what the caller declared
(`enriched._priority_for` returns `None` for "nothing declared"; the OpenAI doors send no `priority`
at all, which is also what §1.1 says they read), and `handle_submit` owns the one precedence:

    1. the REQUEST's `priority`/`interactive`   2. the CREDENTIAL's declared band
    3. the AGENT's configured `default_priority`   4. the built-in default

Step 2 above step 3 because a credential is the stronger statement. Step 3 exists at all because
**since delegation one key can act as many `agent_id`s**, and a single band on that key cannot say
"interactive for `chat-agent`, background for `forum-agent`" — per-agent config can, and it is where the DRR
weights that go with those bands already live.

`Principal.priority_declared` is new and is what makes step 2-vs-3 decidable: a credential that wrote
`P3_INGESTION` and one that wrote nothing produced identical values, so the fall-through could never
fire. `parse_identity_spec` returns it as a sixth element and both registries carry it.

Guarded by `tests/test_priority_precedence.py`, including an AST guard on the **cause** — a door that
starts pre-filling the band again breaks the precedence silently, and the behavioural tests would
still pass for any caller whose credential happens to agree with its config.

### Changed — `/v1/chat/completions` will not grow a `priority` field

Settled 2026-09-02, against a week of real traffic rather than by argument: **196k requests, 3.6%
through an OpenAI-shaped door**, and every one of that 3.6% an off-box caller using a third-party
OpenAI client — precisely the callers that cannot set a non-standard body field, which is why they
are on that door. The other 96.4% declare `priority` explicitly and land on `/rs/v1/chat`, where it
is read. The field would be added for a population that is empty by construction, at the cost of the
one promise `/v1/*` makes. What those callers need is a **standing** band, and a key already carries
one (`agent_id:P1_TURN_SUPPORT`). §1.1 records the reasoning, and that if a per-call band is ever
genuinely wanted there the precedent is a **header**, like `X-Timeout-S`, never a body field.


### Added — the SDK can express all three payload types, and rerank has a route again

Landed 2026-09-02. `roadstead.client` gained `embed()`, `rerank()` and `call()` on both the async and
blocking clients, and now sends **`payload_type`** on the enriched envelope.

🚨 **The server has read `payload_type` since before this repository existed** (`enriched.py`,
defaulting to `chat_completion`) and the SDK sent it never. So the SDK could make chat calls only,
and — since `POST /v1/submit` was removed with Workstream C — **rerank had no route from anywhere at
all**. Not a broken feature: an unreachable one, with nothing on either side to report it.

There is deliberately **no new `/v1/rerank` door**. OpenAI has no rerank shape to be compatible with,
so such a route would be Roadstead's own API wearing somebody else's version number. `/rs/v1/chat` is
the enriched door, not the chat door; it is named for the route.

`payload_type` is **not** `kind` and neither defaults from the other: `kind` declares what sort of
endpoint may serve the request, `payload_type` declares what shape the body is. The typed methods
supply the `kind` their payload type implies (an `embed()` call is *for* an embedder) but supply an
`intent` only when the caller declared nothing — filling one in beside a caller's own pin would be
the SDK making a routing declaration on their behalf.

🚨 **`embed()` is the lossless embedding path and `/v1/embeddings` is deliberately not.** The OpenAI
door translates a hybrid embedder's `{dense, sparse, colbert}` into OpenAI's `{object, data, usage}`,
which has nowhere to put the sparse and colbert halves, and drops them. That stays. `CallResult.response`
is the backend's body untouched, so the enriched path carries all three whole. Documented as a split
in `docs/api.md` §1.1 rather than left for a caller to discover by comparing two vector counts.

### Fixed — the SDK stopped dropping `caller_id` and `request_id`

Both are read by `enriched.py` and neither was ever sent. Losing `caller_id` silently collapsed every
SDK caller into one call site, so `/v1/fleet/top-callers` and `/v1/fleet/cache-attribution` went flat
— a missing **measurement**, which reports itself to nobody. Losing `request_id` meant a caller's own
correlation id never reached the durable record, so their log line and our completion row could not
be joined.

Guarded by `tests/test_enriched_envelope_coverage.py`, which is the test that would have caught all
three losses on the day the SDK was written: every field the enriched door reads off the request body
must be one `_envelope` can put on the wire. 🚨 Its SDK half **calls** the builder rather than reading
it by AST — the AST version passed against a simulated regression that wrapped every assignment in
`if False:`, because AST does not care whether a line can run.

### Changed — `ChatResult` is now `CallResult`

**Not breaking:** `ChatResult` remains as an alias and existing code needs no edit.

*Why:* `/rs/v1/chat` carries chat completions, embeddings and reranks. A result class named for one
of the three tells an embedding caller they are holding the wrong object.

### Changed — `docs/api.md` §1.1 is one table per DOOR

**Breaking for a reader, which is the only way this section could break.** It documented the **union
of two doors** as though it were one: `priority`, `agent_id`, `call_site`, `session_id`, `turn_id`,
`caller_id` and `request_id` were listed as accepted body fields on `POST /v1/chat/completions`, and
`handle_openai_chat` reads exactly two things off that body — `model` and `timeout_s`. The identity
fields are overwritten from the resolved principal; the rest travel to the backend inside the payload.

🚨 **It was almost right, which is worse than plainly wrong.** `caller_id` and `request_id` genuinely
are read — on `/rs/v1/chat` — and `grammar` and `thinking` genuinely do work on the OpenAI door,
because `correction.py` reads them off the payload downstream. A migrator sending `agent_id` here got
no error, no effect, and no hint that the field they wanted was on the other door. Rows were split
per door and annotated with a **Read by** column, not deleted.

Two smaller corrections in the same pass: `X-Request-ID` is **not** read inbound (it is a header
Roadstead emits southbound; the section said "forwarded", which reads as an invitation to send one),
and `/v1/embeddings`' `model` field is echoed in the response and selects nothing.

`tests/test_openai_door_fields.py` pins each table against what the named module actually reads, in
both directions — a row claiming a reader that does not read it, and a handler growing a body field
the table does not list. Observed going red for both, and for a third: a field §1.1 sends a migrator
to `/rs/v1/chat` for that `enriched.py` stops reading.

`roadstead/agents.yaml` lost its claim that identity can arrive "on `/v1/submit`, from the body". That
door was removed precisely because a body could claim any `agent_id`, including one with a better
DRR weight.


### Fixed — the request log grew without bound, and could fail a live request

Landed 2026-09-02. `RequestLogger` opened its file in append mode and never rotated, while calling
itself the authoritative per-request record — 10-20 MB/day at a measured ~27,700 requests/day,
forever.

🚨 **It was nearly handed to the infrastructure as a logrotate job, which would have been the wrong
boundary twice.** The file is the application's, so bounding it is the application's job; and the
handle is held open in append mode, so a rename-and-create rotation leaves the proxy writing to the
unlinked inode — new file empty, disk still filling, everything looking configured. `copytruncate` is
the only external rotation that works on it, and requiring an operator to know that is a trap.

Now size-bounded in the application: `request_log_max_bytes` (64 MB) x `request_log_backups` (7),
capping the set at ~512 MB — deliberately the same order as `completions_retention_s`, since the two
are views of the same traffic. `max_bytes=0` opts out for a deployment driving rotation itself;
`backups=0` truncates rather than meaning "unbounded", because that reading is already spelled
`max_bytes=0`.

🚨 **A logging failure can no longer fail a request.** `log()` runs on the scheduler loop inside
`lifecycle`'s completion path, so an unwritable log used to raise straight into live traffic — a 500
for a caller whose request had actually succeeded. It now fails open and discloses once, the doctrine
`spend.py` already follows. The first version caught only `OSError` and the guard caught the hole: a
**closed** handle raises `ValueError`, and a closed handle is exactly what a vanished mount looks
like from inside.

### Changed — the request log moved under `ROADSTEAD_DATA_DIR`

**Breaking for anyone reading the old path.** It was rooted at `$ROADSTEAD_HOT_ROOT/logs`; it is now
`<data_dir>/logs`, with `ROADSTEAD_LOG_DIR` to override.

*Why:* the deployment contract asks an operator for **one** persistent path and undertakes that
everything the application owns lives inside it. With the log rooted elsewhere, setting
`ROADSTEAD_DATA_DIR=/var/lib/roadstead` moved the queue DB onto the mount and silently left the
request log on the container's ephemeral layer — a half-kept promise, which is the version of this
that gets discovered late.

### Verified — the published 108s stop-grace holds under load, not just at concurrency 1

`tools/sigterm_drain_probe.py` gained **S4**: 24 concurrent dispatches all outlasting the drain. Its
three existing scenarios each parked exactly one request, and `tests/test_shutdown_budget.py` pins
only the arithmetic — so the number an operator is told to configure had never been observed under
the load whose failure it prevents.

**78.24s at concurrency 24 against 78.21s at concurrency 1.** The drain does not scale with load: the
cost is the two serial timeouts, not per-request work. `RECOMMENDED_STOP_GRACE_S = 108` holds with
~30s margin. All 24 callers received the correct `draining` 503, and `proxy_completions` recorded 4
— the number that actually reached the backend — rather than inventing rows for queued work. Full
write-up in `docs/ledger.md`.


### Fixed — the durable event log defaulted to `/tmp`, and in a container it was lost on every rebuild

Landed 2026-09-02. Found by running the proxy against real backends for the
first time rather than by any test.

`queue.db` is the durable record: DRR budgets, the day's completion rows that
`startup` replays so a caller's spend survives a deploy, endpoint drain state,
timeout-model samples. Its default path is `/tmp/agents/llmproxy/queue.db`.
Measured in the real container the same day: the file sat on the **ephemeral
writable layer**, while the mounted volume held only the admin overlay. So every
`docker compose up --build` silently reset all of it.

🚨 **The whole shutdown apparatus was protecting a file the next rebuild
deleted.** SIGTERM runs a bounded drain specifically to flush those rows;
`SHUTDOWN_DEADLINE_S` and `RECOMMENDED_STOP_GRACE_S` are computed and published
so a container stop-grace of 108s does not truncate the flush. All of that care,
on to disk that does not survive a recreate.

Two changes, deliberately different in kind:

- **The image** now sets `ROADSTEAD_DATA_DIR=/var/lib/roadstead` and declares it
  as a `VOLUME`. That is what makes it right for everyone who runs the artifact.
  🚨 The `VOLUME` is not a substitute for mounting a real one — it moves the
  default from *certainly lost* to *not silently lost*.
- **The default itself is unchanged, and disclosed instead.** `/tmp` is correct
  for a developer running the module directly, and relocating an existing
  deployment's state on upgrade is a worse failure than the one being fixed. So
  startup now logs a `WARNING` naming the path and what is lost with it. Same
  doctrine as every other disclosure here: the default was defensible, its
  silence was not.

### Fixed — `ROADSTEAD_QUEUE_DB` and thirteen siblings were silently ignored

Landed 2026-09-02, found while fixing the above — by setting
`ROADSTEAD_QUEUE_DB` and watching it do nothing.

The rename to the `ROADSTEAD_` prefix skipped `__main__.py`. Fourteen variables
stayed under `LLM_PROXY_` **only** — among them `DATA_DIR`, `QUEUE_DB`, `HOST`,
`PORT`, `LOG_LEVEL` and every retention/vacuum knob. Nothing failed, because the
legacy spellings still worked; what broke was the **documented** spelling, which
was a silent no-op. That is the `models.yaml` allowlist failure again: an
unknown key dropped in silence, costing whoever spelled it the way the docs say.

`acl.py` had already solved this correctly for its own variable — read both,
warn on the legacy one, document which wins. That pattern is now shared
(`config.env_with_legacy_prefix`) and applied everywhere. **Backward compatible:
every `LLM_PROXY_*` spelling still works and now says it is deprecated.**

🚨 The guard immediately caught one the sweep had missed: `management.py`'s
`sources` view — the operator-facing "where does config come from" diagnostic —
was reporting `LLM_PROXY_AGENTS_CONFIG` as the variable to set, while every
sibling reported `ROADSTEAD_*`. It sent operators to the one spelling that did
not work under the documented prefix.

`tests/test_env_var_naming.py` is the guard: an AST sweep for any direct legacy
read outside the helper, the helper's precedence, and the Dockerfile's data dir.
All of it watched going red first — including a fourth `/private/tmp` case added
after the disclosure proved silent on macOS, which is the platform where the
`/tmp` default is exercised most.


### Scrub — a real host name shipped in source, and is gone from the tree (S7, half-closed)

Landed 2026-09-01. `docs/corpus_and_scrub_plan.md` § S7 has the full account.

S6 claimed host names were pseudonymised through the whole history. They were not. A fresh check
found real, currently-resolving names surviving in history, and one of them — the GPU head node — was
in **eleven tracked files, four of them shipping source** (`roadstead/health.py`, `config.py`,
`lifecycle.py`, `correction.py`), next to a port that turned out to belong to a **live production
inference server**. So what shipped in the wheel was a working address, not just a name.

It is now spelled `anvil` (`anvil2` for its paired worker) throughout the tracked tree. The
measurements those comments record are untouched — same rule as `models.yaml`: the measurement is
real, the machine it names is not.

🚨 **Only half of S7 is closed.** The tracked tree is clean; the **history is not**, and that still
blocks going public. The two halves were split deliberately because their costs differ by orders of
magnitude — the tracked half is ordinary edits and stops the name shipping *today*, while the history
half needs a second `filter-repo` pass that changes every SHA again. Leaving them coupled is why
neither had happened.

🚨 **One entry in this file is now knowingly inaccurate.** The `ANVIL_DISPATCHER_URL` →
`ROADSTEAD_ON_DEMAND_DISPATCHER_URL` note recorded the old variable under its real spelling, and that
spelling carried the host name. It now reads `ANVIL_DISPATCHER_URL`, naming a variable that never
existed. Accepted for the reason S6 rewrote commit messages: a scrub that spares the record leaks
through the record.

### Added — the wire contract, confirmed against a real vLLM for the first time

Landed 2026-09-01. `tests/wire_fidelity/test_real_vllm.py`, opt-in on
`ROADSTEAD_WIRE_FIDELITY_VLLM_URL` and deselected by default like the rest of the marker. Full
write-up in `docs/ledger.md`.

Nothing in this project had ever run against a real vLLM. Every vLLM claim in `providers/vllm.py` is
an assertion about an **absence** — `publishes_slot_count=False`, `publishes_slot_context=False` —
and an absence is the one thing a programmable fake can never confirm: it withholds what it was told
to withhold and agrees with the descriptor by construction. **All seven assertions pass against a
real engine**, so the config-seeded-concurrency decision now has evidence on both sides.

🚨 **The terminal-chunk rule holds on vLLM too.** `finish_reason` rides alone on a chunk with an
empty `delta`. That rule was established on llama.cpp and `correction.py` has applied it to *every*
backend ever since — until now the second engine was being repaired against a rule measured on the
first.

🚨 **The near miss worth knowing:** `/metrics` publishes `vllm:cache_config_info` with a
`kv_cache_max_concurrency` label — a plausible-looking float that is **not** the admission cap. It is
`kv_cache_size_tokens / max_model_len`, i.e. how many *full-context* requests fit, and on a
long-context server it reads below 2 while the engine fields many more. Seeding `max_slots` from it
would cap a busy endpoint at one, wearing the authority of a discovered fact.
`check_no_published_concurrency` fails loudly if `max_num_seqs` ever appears — the one development
that would make vLLM slot discovery real, and an alarm to act on rather than route around.

### Fixed — two wire-fidelity tests could only ever be aimed at their own compose file

Landed 2026-09-01, found by pointing them at a production llama.cpp (build `b1-6d05498`, 6 slots,
262144 per-slot context) rather than the 0.5B model in `compose.yaml`.

`test_top_level_n_ctx_is_absent_not_an_aggregate` and
`test_slot_count_comes_from_total_slots_not_n_parallel` compared against `COMPOSE_PARALLEL` /
`EXPECTED_PER_SLOT` **without** the `_launched_by_compose()` gate their siblings use. Aimed at any
other real engine they failed on the *launch config* rather than the wire shape — `262144 != 2048`
and `6 != 4`, both correct values for that server. That defeats the purpose: this directory is meant
to be an alarm you can aim at a production backend.

Each is now split. 🚨 **The number is a property of the launch; the resolution order is a property of
the engine** — and only the second is what these tests are about. Every b5350 finding survived the
build gap, and `default_generation_settings.n_ctx` being **per-slot** is now corroborated
independently by `/slots`, where each of the six entries reports the same value.


### Fixed — a permanent backend failure is no longer advised as retryable

Landed 2026-09-01, found in live OpenRouter validation. An endpoint pinned to a model that does not
exist returned `code: backend_error, deferrable: true` for a permanent `400` — so a caller following
the SDK's own advice would retry forever against a misconfiguration.

🚨 **The proxy already knew.** `correction.is_transient_backend_error` says in its docstring that
"a real 4xx / other-5xx is deterministic → surface", and declines to retry it internally. It then
handed the caller a code every client classifies as retryable — the proxy giving up on a permanent
failure and simultaneously advising a retry it had just refused to make itself. One judgement, made
twice, differently.

The error envelope now carries **`backend_status`**, the status the backend returned, whenever the
failure came from one. `RoadsteadError.deferrable` returns `False` for `backend_error` with a
permanent status — a `4xx` other than `408` (it timed out) and `429` (rate limited), which are the
two that say *later* rather than *never*.

**Backward compatible by construction:** an absent `backend_status` (an older proxy) classifies
exactly as before, and only `backend_error` is narrowed — never `backpressure`, whose entire purpose
is to say *try again shortly*. The golden behaviour baseline moved by exactly one field across
seventeen cases.

### Fixed — 🚨 SECURITY: a credential pasted into `api_key_env` was stored, persisted and echoed back

Landed 2026-09-01, after it happened. An operator pasted a live OpenRouter key into the provider
form's `api_key_env` box — reasonably, because it was the only key-shaped field on the form and the
form offered nowhere else to put a key. That field takes the **name** of an environment variable. The
plane accepted the key as a name, wrote it to the admin overlay on disk in plaintext, recorded it in
the audit trail, and echoed it back from `GET /rs/v1/admin/providers`.

Every one of those is something §3.5's "a management surface never emits a credential" rule exists to
prevent. **The rule was right and nothing enforced its precondition.**

Two fixes, because the first is necessary and not sufficient:

- **The value must be a valid POSIX identifier.** Not a heuristic: `api_key_env` names an environment
  variable, and no common credential format is a legal identifier — they all carry `-`, `.` or `/`.
  The refusal does **not echo** what was sent, because a 400 quoting the value puts the secret in the
  response body, the access log and the browser history. 🚨 It does not catch `ghp_…` or `sk_live_…`,
  which are legal identifiers, and a test asserts that out loud so nobody mistakes this for a
  credential detector.
- **The form gives the key its own home:** a separate write-only password field that posts to
  `.../credential` after the provider is saved, and is cleared either way. The structural cause was a
  form with one key-shaped box and no right answer.

**If you have pasted a key into that field, treat it as compromised and rotate it** — it was written
to disk and to the audit trail, both of which outlive the process.

### Added — the provider form asks for the fields the engine actually uses

Landed 2026-09-01. Selecting `openrouter` asked for `host` and `port`, which that engine ignores
entirely, and did not mark `api_key_env` as required, which it refuses to serve without. The form
now follows the engine's **descriptor**, not its name:

- `ProviderDescriptor` gains `addressed_by_base_url`, `requires_credential` and `default_base_url`.
  🚨 Each has a second reader beyond the form, which is this dataclass's standing rule — the
  management plane now **refuses** a provider stanza that gives the wrong address kind or omits a
  credential the engine requires, instead of letting it fail as `ProviderMisconfigured` on the first
  real request. A config gap surfacing as a runtime fault is what this plane exists to prevent.
  🚨 `addressed_by_base_url` is deliberately not the same question as `kind`: "remote" is about whose
  capacity it is and who bills for it, this is about what an address looks like. They coincide today,
  and conflating them is how a local engine behind a gateway becomes unconfigurable.
- `GET /rs/v1/admin/providers` publishes an `engines` block — every engine this build registered,
  with its descriptor. The UI's engine list and per-engine fields are **data**, so a fourth provider
  gets a correct form without the page being edited. A list in the page would be the fifth place an
  engine has to be added and the one nobody remembers.

Selecting `llama.cpp` shows host and port; selecting `openrouter` shows a base URL prefilled with the
service's own address and a required `api_key_env` — the NAME of a variable, never a key.

### Added — the catalog is writable at runtime (roadmap J2)

Landed 2026-09-01. `PUT`/`PATCH`/`DELETE` on `/rs/v1/admin/providers/{p}` and
`/rs/v1/admin/endpoints/{e}`: create a backend that is in no file, edit any field of one that is,
delete either. A provider and an endpoint invented through the API served a real request seconds
later, and `models.yaml` never saw any of it.

🚨 **The overlay contributes catalog STANZAS, not a second model of an endpoint.** A runtime fragment
is merged into the file's raw dict *before coercion*, so a UI-created endpoint is parsed, defaulted
and validated by exactly the code that parses a file-authored one. Bodies use `models.yaml`'s own
field names because they *are* `models.yaml` stanzas. A partial merges over the file's, a new name
stands alone, `null` is a tombstone — the same "a later statement wins" rule that puts revoke after
enrol.

🚨 **`models.yaml` is never written**, and every response says so. Comments, formatting and
hand-authored intent survive a bad save.

🚨 **Validation is by construction** — the candidate catalog *and its endpoint kwargs* are built, and
the write is refused if either complains. Building only the catalog was not enough: the `policy:`
allowlist check lives in `build_endpoint_kwargs`, so a mistyped policy key was accepted and the
complaint arrived after the write had been agreed.

**File format:** the overlay's `endpoints` section became `catalog.endpoints`. An existing file is
**migrated on load**, not dropped — otherwise an upgrade silently takes every promoted endpoint out
of service.

### Fixed — three defects found by running the new writes

- **`RuntimeError: dictionary changed size during iteration`** in the capacity poller. Creating an
  endpoint mutated `config.endpoints` while the poller was iterating it and awaiting between items.
  The routing table is now **replaced, not mutated** — an in-flight iteration finishes over the table
  it started with, and a reconcile is one atomic change rather than a sequence a reader can observe
  halfway through. That is also the alternative to auditing 27 call sites for an `await` and being
  wrong about one.
- **An edit was accepted and did nothing.** Reconcile leaves existing entries alone so discovery's
  corrections survive, so nothing rebuilt an endpoint whose stanza had changed. Callers now name what
  they touched.
- 🚨 **Startup deleted endpoints configured in code.** The first reconcile imposed the whole catalog,
  and the shipped catalog declares `spill-chat` as `planned` — so a routed one added directly to
  `config.endpoints` by an embedding caller was removed at startup, silently. Reconcile now touches
  only the names its caller says changed. Caught by an existing spill test going red.

### Fixed — the management plane reported admin-plane reach it was not using

Landed 2026-09-01. `GET /rs/v1/admin/config` reported `admin_nets.builtin` from
`acl.builtin_admin_nets()` — the *identity-grant* net list, a different question — so after an
operator named their own nets it went on advertising the docker-internal default that naming them had
**dropped**. `acl.reach_nets()` is the right source and its docstring says it exists for the
management plane; only the startup log was calling it. So the log told the truth and the view did
not, which is the two-sources-that-agree-most-of-the-time failure this plane exists to expose. The
view now reports `in_force` and `docker_default_dropped`.

### Added — the management plane can bring a declared endpoint into service (roadmap J1)

Landed 2026-09-01. The Providers tab was the configuration surface and carried **no controls at
all**. Two writes now:

- **`POST /rs/v1/admin/providers/{provider}/credential`** — supply the value for the provider's
  `api_key_env`. 🚨 Write-only (no surface reads it back — not the value, not a prefix, not a
  digest, not a length) and 🚨 **never persisted**, which the response says up front rather than
  leaving it to be found at the next restart. The overlay holds key digests and has never held a
  secret. It takes effect immediately because a provider reads the variable at call time.
- **`POST /rs/v1/admin/endpoints/{ep}/status`** — `active` | `planned`.

🚨 **Neither creates anything.** Both resolve their subject through `models.yaml` and 404 what is
not declared there. Promoting an endpoint that is already written down changes exactly one thing —
membership of the routing table — and is a far smaller claim than hot-adding one, which remains a
file edit and a restart.

Two refusals carry the design. **A promotion whose credential does not resolve is refused**, because
`models.yaml` says in its own words that a deployment without the key "should not have an endpoint
in its routing table that cannot serve". **A demotion with work in flight is refused**, because the
request path reads the endpoint's config after dispatch; pause already drains. And 🚨 **the startup
replay applies the same credential rule** — the status persists and the credential deliberately does
not, so a restart would otherwise be the one moment a routed-but-unusable endpoint appears.

### Fixed — a promoted endpoint was routable but reported as unrouted

Landed 2026-09-01. `enriched.facts()` computed `routed` from the catalog entry while its own comment
said "an endpoint present in the routing table is ROUTED whatever the catalog says". The branch that
disagreed was unreachable — `config.endpoints` is built from `catalog.routed()` at startup, so the
two could not differ — until J1 made them differ deliberately. The symptom was a promoted endpoint
reported unrouted on `GET /rs/v1/models`, and since `intent.py` filters on that field, a pin to it
404'd while the management plane reported it live. `config.endpoints` is the routing table.

### Changed — the operator UI reads like an instrument, not a wall of text

Landed 2026-09-01. Same one file, same zero dependencies, same CSP with no external reference.

- **The doctrine moved behind a disclosure.** Every section printed a paragraph of reasoning as
  permanent body copy, above the data an operator opened the page for. All 19 are now `<details>` —
  every word kept, one click away, no JavaScript, keyboard-accessible. Alerts got the same treatment:
  their four-line RECONCILE instructions were the largest text block on the page.
- **Capacity is drawn as berths.** A roadstead is the anchorage where vessels wait for a berth, the
  unit of fairness here is time spent occupying one, and the page rendered that as the string
  `"0 / 4"`. One cell per slot, filled when occupied.
- **A palette from the subject** — harbour water at night, chart-paper ink, and navigation lights for
  state (starboard green serving, amber waiting, port red refused) — replacing near-black with a
  borrowed blue accent.
- **The 1500px cap is gone**, which was wasting a third of a wide monitor on a page of dense tables.

### Fixed — the management plane hid every endpoint that was not in force

Landed 2026-09-01. `GET /rs/v1/admin/providers` reported only endpoints in the **routing table**,
because it iterated `state.config.endpoints` — which `model_catalog` builds from `catalog.routed()`.
A `planned` or `retired` stanza was filtered out one layer below the view, and nothing said so.

That is the plane's own question failing on its plainest case. `management.py` opens by saying the
surface exists to answer *what did you write that is not in force?*, and a `planned` endpoint is
exactly that: written down, parsed, validated, deliberately not serving. An operator could not see
that the example catalog's `spill-chat` existed, nor that the only thing between it and service was
an unset `$OPENROUTER_API_KEY`.

Unrouted endpoints are now reported with `routed: false`, the declared capacity, a `not_in_force`
block naming the reason and the credential variable, and 🚨 **every in-force field as `null`, never
zero** — `slots: 0` would say the backend was asked and answered nothing, which is the one confusion
this view exists to remove. `discoverable` is `null` for the same reason: `false` is a claim about an
engine that was consulted. The UI's endpoint table renders the status and the missing credential.

### Fixed — the blocking client leaked connections and hung after `close()`

Landed 2026-09-01. `roadstead.client.RoadsteadClient` — exported, documented, and until now never
constructed by a single test. The async class had two files of coverage; its blocking wrapper had
none, and the wrapper is where the hard part lives: a private event loop on a worker thread with
async generators pumped across it.

- **Abandoning a stream leaked the connection.** `break`-ing out of `for frame in
  client.stream(...)` left the async generator to be finalized by the GC on a thread with no running
  loop, so the `async with` around the HTTP response never unwound and the connection was never
  released. It announced itself only as `Exception ignored in: <async_generator ...>`, which nothing
  reads. Streams are now unwound on the loop that owns them, and `close()` drains the ones it handed
  out before stopping that loop.
- **Any call after `close()` blocked forever**, waiting on a future a stopped loop will never
  resolve. It now refuses with a sentence, which is what the module docstring already promised.
- `close()` is idempotent, and the "you are inside a running loop" refusal no longer trails a
  `RuntimeWarning: coroutine '_make_async' was never awaited`.

No wire change; a caller who never abandoned a stream and never used a closed client is unaffected.

### Changed — the enriched `price` block has ONE shape on every route

Landed 2026-09-01. **Additive.** `GET /rs/v1/models` was emitting three of the price block's five
fields, where a chat envelope and a plan estimate emit all five. A client with one typed view for
"a price" — which is what the shipped SDK has — read `source` and `detail` as empty for every
endpoint. `source` is the field that separates a price the provider **published** from one Roadstead
**imputed**, which is the same measured-or-guessed distinction the management plane reports for slot
counts, and the one a caller choosing where to send work most needs.

Found by a new guard that extracts every field `roadstead/client/_models.py` reads and walks each
one against a response a real server produced — the client-side twin of the UI's `pick()` guard, and
for the same reason: a renamed or never-sent field reads as `""` forever and nothing raises. The
block's five fields are now published in `docs/api.md` §1.7.3 instead of being elided as
`{"...": "..."}`, which is why the drift was invisible.

### Changed — 🚨 BREAKING and security-relevant: an address no longer grants admin

Landed 2026-09-01. **The admin plane now requires BOTH a reachable address and an `admin`-scoped
credential.** Until this change a request from **loopback presenting nothing at all** could pause a
backend, re-weight a caller's quota, ingest call logs or flip a runtime flag. The audit trail
recorded such a change as `key_id: null, source: "ip"` — the system stating in its own log that
nobody had authenticated, and permitting the write anyway.

**How it got there matters more than the fix.** It was not a bug anybody wrote. The auto-grant
belongs to the INFERENCE door, where "already on the box" is a fair proxy for "allowed" on a
local-first proxy, and it was correct there. The admin plane, and later the management UI, were added
**on the same port** and inherited it — a default written for one door silently governing a different
one, with the UI's elaborate reasoning about CSRF, credential kinds and the 401/403 split sitting on
top of a door that was already open on localhost.

**What breaks:**

- **`ROADSTEAD_ADMIN_NETS` is reinterpreted, not replaced.** It named addresses that WERE admins; it
  now names addresses that MAY REACH the plane, with a credential still required. A deployment
  relying on it for unauthenticated admin will start getting 403s and needs a key.
- **`:admin` on a `ROADSTEAD_ACL` address entry no longer grants the scope**, for the same reason.
- **Loopback is no longer admin.** Scripts driving `/v1/admin/*` from the local host must present a
  key.
- **A key alone is no longer enough from anywhere.** The network gate applies to credentials too, so
  an admin key used from off-net needs its network named in `ROADSTEAD_ADMIN_NETS`.

**What replaces it:**

- **Reach**: loopback always, docker-internal by default so a container reaches its own plane, and
  whatever `ROADSTEAD_ADMIN_NETS` names — naming any net **drops the docker default**, because a
  `/12` is a weak gate. The network is checked first, so an off-net refusal says it is about the
  network and no credential answers it.
- **Credential**: an `admin`-scoped key, as `X-API-Key`, `Bearer`, or HTTP Basic in the **password**
  half (what the UI uses — unchanged).
- **A bootstrap key, so the plane is never open and never unreachable.** With no *operator* key
  configured, startup mints a random admin key and logs it; it is not persisted and a new one is
  minted each boot until `ROADSTEAD_API_KEYS` is set. 🚨 It is **invisible to
  `KeyRegistry.configured`**, so it cannot flip §1.5 rule 2 on the inference door — counting a
  self-minted key there would 401 every OpenAI client that sends a placeholder `Authorization`
  header, which is the exact failure rule 2 exists to prevent.

**Roughly sixty tests failed at once when the grant was removed, and none of them meant to assert
it.** They were testing what the plane does; that they also asserted it needs no credential was
invisible until it wasn't. They are authenticated centrally through `tests/admin_key.py` rather than
one at a time, and the gate itself is pinned separately in `tests/test_admin_gate.py` — a helper that
authenticates the plane's tests must never also be the thing that tests the gate. Two of the failures
were doctrine tests asserting the old behaviour in as many words, including one whose docstring
called the unauthenticated audit record *"a meaningful — and slightly alarming — thing for an
operator to find"*. It was.

### Changed — BREAKING for anyone holding a clone or a SHA: the history was rewritten (scrub S6)

Landed 2026-09-01. The second and final `git filter-repo` pass, the last item in
`docs/corpus_and_scrub_plan.md`. **Every commit SHA in this repository changed.** A clone from
before this date shares no ancestor with `main` and cannot be fast-forwarded; re-clone rather than
pull. Any SHA cited in an external document, a branch name, or a bookmark is dead.

**It took two passes, not one.** The first rewrote blobs and commit messages. The second was a
`--mailmap` run over commit *metadata*: author and committer identity is neither blob content nor a
commit message, so it survived the first pass entirely. All 322 commits now carry one identity at a
public address. 🚨 **That pass found a second private hostname on 103 commits that nothing had
flagged** — because a sweep over tracked files cannot see the author line, and every check to that
point had been a sweep over tracked files.

This is recorded here rather than passed over as housekeeping because it is the most broadly
breaking change the project has made — it breaks something for every holder of a copy, which no API
break does — and because two things about it were wrong in the plan that specified it:

- 🚨 **`--replace-text` rewrites blobs and nothing else.** The documented one-liner would have left
  all 321 commit messages untouched, and messages were the *denser* surface: 282 of the 321 commits
  were authored inside the origin monorepo and describe its hosts, its container IDs and its sibling
  projects far more freely than the code ever did. `--replace-message` takes the same rules file.
- 🚨 **Bare-word rules would have corrupted content.** `Delta` also occurs as `Gated DeltaNet`, a
  model architecture, and `Chase` as "Chased to column level". Every one of the 42 rules is a
  multi-word key or a distinctive stem, authored against an inventory of the actual occurrences.

Host and sibling-project names became pseudonyms throughout — the names you will now read are
`nexus`, `nasbox`, `boxa`, `tideway`, `beacon` and `sidekick` — and the fleet's `/24` became
`10.0.0.x`. 🚨 **The mapping itself is not recorded anywhere in this repository, deliberately.**
Writing `old`→`new` in a changelog would restore every name the pass removed and hand a reader the
key to reverse the rest; a scrub that documents its own substitutions has not scrubbed anything. **The measurement comments are still records** — `CLAUDE.md` says so, and now
also says the box names within them are pseudonyms, because a reader who goes looking for the
machine should be told there isn't one. Model and endpoint-class vocabulary was out of scope and is
unchanged.

Also fixed in passing: **a real airline booking reference was still in the working tree.** S4
replaced the traveller, the airline and the airports around it and carried the booking reference
over verbatim — and a booking reference plus a surname retrieves a booking. It reads `QQ7X2R` now,
in the tree and throughout the history.

### Fixed — today's spend survives a restart (Workstream D)

Landed 2026-09-01. `SpendLedger` is in-memory and its day bucket reset to zero on every boot.

🚨 **This was a correctness bug, not the acceptable simplification the roadmap had called it.** The
old wording — "right for what it governs, whether a caller is degraded *now*" — quietly assumed *now*
was the same length as the window the threshold reads. It is not: `daily_spend_usd` is a **day** and
the process was measuring an **uptime**. A deploy at noon handed every caller its whole allowance a
second time, so the more a fleet ships the less its spend cap means — worst precisely on the
deployment where somebody set the cap deliberately. DRR balances have survived a restart since Phase
3.4; this is the same argument about the other per-caller quantity, missed when `spend.py` landed.

`startup` now replays today's rows from `proxy_completions` through the ledger's existing `charge`,
via a new `day_spend_rollup(since)` aggregated **in SQL** — one row per caller-endpoint pair rather
than one per request, on the existing `idx_pc_completed` index.

- 🚨 **The rollup groups by endpoint as well as by caller.** Collapsing it re-prices a caller's whole
  day at one arbitrary endpoint from the group, and for a caller spanning a cheap and an expensive
  model that is the entire number.
- **`charge` gained `requests=`** rather than growing a second `recover()` method, so there is ONE
  implementation of the pricing and day-rollover rules. A second copy is what `cost_model.context_fit`
  was created to undo, and that copy had already gone silently dead.
- **Re-priced at TODAY's prices**, not each call's price at the time. Right for a threshold that
  answers "is this caller degraded now", wrong for a bill — and anything billable reads
  `proxy_completions`, which keeps the **tokens**.
- 🚨 **The seed fails OPEN, loudly.** Refusing to boot because we cannot prove a caller crossed a
  threshold whose entire consequence is one priority band would let a spend cap take the proxy down,
  which is the same argument that stops it taking a *caller* down.
- `spend.day_start` is derived from `day_bucket` so the query that reloads a day and the threshold
  that reads one cannot disagree about where a day begins — they would drift apart at exactly one
  instant a day.

**Where cost truth lives, settled by construction.** `proxy_completions` stores **tokens, never
dollars**, so pricing lives in one place, the ledger is a derived view rather than a second record to
reconcile, and a provider's invoice can only ever disagree with us about *price* rather than about
*usage*. Reconciliation against a real invoice still cannot be built here honestly — it needs a real
billed account, and a reconciler written against an invented invoice would agree with itself and
prove nothing.

`tests/test_spend_durability.py`. Eight mutations, every guard observed going red. One first
SURVIVED and was a real weakness: the price fixture charged every endpoint the same rate, so dropping
the endpoint from the `GROUP BY` changed nothing — a caller spanning two prices is what makes that
grouping observable.


### Removed — BREAKING: `normalize_endpoint` no longer rewrites a `nexus-` prefix

Landed 2026-09-01, found by re-running the straggler sweep (`docs/corpus_and_scrub_plan.md` S5).
Recorded here per `docs/compatibility.md`: this is internal behaviour rather than the wire contract,
but it changes how a *name a caller sends* resolves, which is as close to the contract as internal
gets.

`normalize_endpoint` stripped a `nexus-` prefix and mapped a bare `nexus` to `chat` — **one private
fleet's host naming, hardcoded since the first commit (`1becf53`, 2026-05-27) and shipped to
everyone.** It was a
scrub finding and a design defect at once, and the second is the reason it is removed rather than
renamed:

- It was a **second aliasing mechanism** beside `models.yaml`'s `aliases:`, which is exactly what
  this codebase refuses everywhere else — and the refusal has a name here, since the alias
  duplicate/shadow notice added in the same release cannot see this one at all.
- It could not be configured, overridden or disabled, and it silently rewrote **any** endpoint whose
  name happened to begin with those six characters. An operator with `nexus-a` and `a` had a pin
  at the first silently reaching the second.

**Migration**, if you actually want that mapping: put it where every other name lives —
`aliases: [nexus]` on the endpoint, which is declared, reported and collision-checked. The whole
suite passed unchanged with the branch removed, which is how long it had been dead weight.

Two smaller findings from the same sweep: three arbitrary test addresses that merely *looked* like
the private subnet (a sweep cannot tell, so each cost a human adjudication per re-run and S6 would
have rewritten them through the history for nothing), and a sibling private project's name used as a
shipped `ROADSTEAD_ACL` example.

🚨 **The topology half of S5 now runs on every commit** (`tests/test_scrub_sweep.py`) — a sweep that
lives in a shell command in a document is one somebody has to remember. The identifier half stays a
human pass **and the test says so**, because there is no pattern to key on for a name, which is
exactly how S4's finding survived the first sweep. Four mutations, all red.


### Fixed — a cancelled straggler's caller gets the envelope, not a raw 500

Landed 2026-09-01. `docs/ledger.md` carried this as "Open, and ours to fix"; it was parked as
"belongs upstream" while the origin monorepo was authoritative for behaviour, and that rule was
retired on 2026-08-31.

**Root cause.** `asyncio.CancelledError` derives from `BaseException`, not `Exception`. At
`timeout_graceful_shutdown` uvicorn calls `task.cancel()` on every in-flight handler, and the
resulting exception sails past *both* the `exception_handlers` backstop in `build_app` and
Starlette's own `ServerErrorMiddleware` — neither can see a `BaseException`. The caller of a
straggler got `500 Internal Server Error`, plain text, no `code`, no marker, for a condition that is
retryable and entirely ours.

`ShutdownEnvelopeMiddleware` closes it. A **middleware**, not another `exception_handlers` entry: a
handler there could never be reached, because Starlette dispatches them from inside an
`except Exception`. User middleware sits outside the router and inside `ServerErrorMiddleware`, which
is the outermost place a route's `BaseException` is still catchable.

- **No new code.** `draining` + 503 + the literal `backpressure` marker is exactly what `lifecycle`
  already emits for work *refused* while draining, and it has to be: the caller's situation is
  identical. §2.2 classifies deferrable on the marker substring, so dropping the word would make a
  shutdown read as a hard failure to every client that has not migrated to classifying on `code`.
- 🚨 **The cancellation is always re-raised.** Swallowing it breaks the asyncio contract and leaves
  uvicorn waiting on a task that has declined to die — turning the bounded shutdown into the
  unbounded one `SHUTDOWN_DEADLINE_S` exists to prevent. This adds a response; it does not decline
  to stop. A failure to deliver the envelope is swallowed rather than the cancellation.
- 🚨 **A cancelled STREAM gets an error frame and NO `[DONE]`.** `[DONE]` is an assertion of
  completeness; emitting one over a truncated answer is the exact collapse `correction.py`'s
  `finish_reason` repair refuses to make.
- A cancelled non-stream whose headers are already out has its body **closed**, not left hanging.

Measured with `tools/sigterm_drain_probe.py` S3, before → after: `500 Internal Server Error` at
48.77s → `503 {"code": "draining", …}` at 48.76s. Exit time (78.2s) and what is persisted are both
unchanged. `tests/test_shutdown_envelope.py`. Nine mutations, every guard observed going red — one
first failed with a bare `TypeError` and asserts first now.


### Fixed — an alias claimed twice routed silently, and the soak was measuring itself

Landed 2026-09-01. Two unrelated defects, both found by doing the thing rather than reading it: one
by the first pass of the origin-ledger transplant, one by running `tools/soak.py` for 25 minutes.

**An alias claimed by two endpoints resolved in silence.** Transplanted from the origin monorepo's
`models-yaml-alias-collision` (2026-07-30) — and it turned out not to be a transplant. The origin
**raised** at module import on a duplicate alias, which took a whole gateway down and crash-looped it
to FATAL over one duplicated line. This repo builds the same map with `setdefault`, so the second
claim is discarded in file order, in silence: the operator's name reaches an endpoint they did not
intend, on every request, forever. **The lesson survived the extraction and the guard did not.**
🚨 Neither behaviour was right — refusing to load contradicts this repo's own rule that a typo must
not stop a fleet booting. So it loads, resolution is unchanged, and both cases report through
`hooks.config_notice`: `duplicate` for two endpoints claiming one alias, `shadowed` for an alias that
another endpoint's class or role wins (that ordering is deliberate and stays, and it still killed an
alias somebody wrote). Each names the endpoint the name actually reaches in an **`in_force`** field —
the management plane's own vocabulary, and read by `GET /rs/v1/admin/config`. A self-alias is
deliberately not reported: it maps to itself, it is common, and a notice that fires on every boot is
one nobody reads. `tests/test_alias_collision.py`, `docs/ledger.md`. Six mutations, all red — one
first SURVIVED because the test read the winner out of the prose sentence, which interpolates the
same value the emptied field held.

**🚨 `tools/soak.py` was measuring its own test double.** A 25-minute run climbed
**118MB → 1867MB (+72.8 MB/min, linear, no plateau)** — precisely the shape that tool's docstring
calls unhealthy. The growth was `FakeBackend.requests`: an unbounded list holding a body dict and a
header dict per request, for the life of the process. Two defects in one:

- `roadstead.testing` is **shipped public surface**, so that is a real leak for anyone who installs
  it and drives sustained load through it.
- The soak exists to detect leaks by RSS slope, and its headline number — the field it explicitly
  tells you is "the number to look at" — was dominated by the harness rather than the proxy.

The recorder is now a bounded deque (`REQUEST_LOG_CAPACITY`, exported, 1000) that **reports what it
drops** (`requests_seen`, `requests_dropped`) rather than truncating in silence — the same answer the
audit trail gives. Slope after the fix: **+8.2 MB/min and flattening**, the remainder being
`RollingMetrics` filling its own 300s window toward steady state. `tests/test_fake_backend_recorder.py`.
Six mutations plus a no-op control; one first SURVIVED because the ordering test drove a local helper
instead of the app's real record path, so it was asserting the file's own model of the recorder
against itself.

**Also fixed on the way past:** `RateLedger.prune` was never called — see the entry below.


### Fixed — the rate ledger grew without bound, and nothing was watching it

Landed 2026-09-01. `RateLedger.prune` shipped with Workstream I carrying a docstring that said
"Called from the maintenance tick". **Nothing called it**, in the package or anywhere else.

- 🚨 **Both dicts behind the rate threshold are keyed by `agent_id`, which is a caller-supplied
  string on the address path** (`docs/api.md` §1.5 — an address only fills in an `agent_id` the body
  omitted, so a body may declare its own). They grew one entry per distinct name ever seen, without
  bound, under a caller's control. `rate_demotion_noted` is the second one and the one a reader
  forgets exists; forgetting a caller now forgets that we noted its demotion, which is also correct
  on its own terms — it had no traffic in the window, so if it returns and crosses again the operator
  should hear about it again.
- 🚨 **It is pruned on WALL time, not the poller's monotonic clock**, and the call sits one line
  below `timeout_model.prune(mono)`, which takes exactly that. `RateLedger.record` stamps
  `time.time()`, so a monotonic `now` would compare a process uptime against epoch timestamps: the
  cutoff lands decades before every sample, nothing is ever stale, and the call returns 0 forever
  while looking wired. A leak fixed by a call that does nothing is worse than the leak, because the
  call is evidence it was handled. Pinned by a test that asserts the two clocks are far apart, so it
  cannot pass by coincidence.
- **The ledger is now armed in `tests/loop_affinity.py`.** It was not armed at all — a whole
  workstream's state outside the guard, while the soak reported a full method count. It has two
  loop-side writers now (the submit path and the poller tick), which is precisely the pair that would
  interleave if either ever moved off the loop.
- **The threshold this feeds cannot reject**, which is why its own memory must not be a denial
  surface either — a rate control that degrades is deliberately not a defence against a runaway
  caller, and its bookkeeping becoming one inverts that.
- `tests/test_rate_prune.py`. Six mutations, every guard observed going red by assertion. One first
  SURVIVED and was a real weakness: the live-caller test asserted `prune_rate_state() == 0`, so it
  never reached the `if dropped:` branch and proved only that a prune which does nothing changes
  nothing. Another failed with a bare `KeyError` and was rewritten to assert first.


### Added — the intent vocabulary in config, and the negative constraint (Workstream C)

Landed 2026-09-01. Both of C's open items, and they turned out to be one question — *who owns the
words a caller may use* — with one line running through it: **config vs request**, not positive vs
negative.

**`models.yaml` grows an `intents:` section.** Nine profiles still ship built in; a deployment's own
are **layered over** them rather than replacing them, because a file that defines one profile has
said nothing about the other nine, and a whole-table swap would silently empty a vocabulary
`GET /rs/v1/models` publishes and callers code against. Overriding by name is how a fleet says its
`reasoning` means something particular, and it is **disclosed**: every published profile carries
`source: builtin | models.yaml`, since a caller reading our documentation for a word this fleet
redefined has no other way to notice.

🚨 **A profile still cannot name an endpoint — and now there is a config parser that must not learn
how.** A profile naming endpoints would be a second routing table to keep in step with `endpoints:`,
and it would break on every fleet spelled differently from the example's. Guarded from both ends: the
allowlist (`_PROFILE_FIELDS`) and `Profile`'s own fields, so the rule does not rest on a door in a
wall with a second door.

**An unusable stanza is REFUSED, not offered.** An unknown capability, kind or preference makes a
profile that matches nothing on every request — and the caller reads "no endpoint satisfies
requires=[…]", a sentence about the fleet, for a fault in a config file. Refused, they get "unknown
intent", which points at the vocabulary, where the fault is. Either way the operator gets a
`hooks.config_notice`, readable at `GET /rs/v1/admin/config`. A refused *override* does not leave the
built-in standing under the operator's spelling — they would be reading their own summary in the file
while callers got ours on the wire, which is the declared-vs-in-force gap the management plane exists
to close, created by us.

**`exclude` — the negative constraint, and why it is expressible.** A caller working around one bad
model wants "anything but this", and the alternative is enumerating every endpoint it *would* take,
which is a routing table in the caller. It fits because the vocabulary it needs is not the profile
vocabulary: a profile is shared, published, operator-written and stays in capabilities; an intent is
one caller's words about one call, where `pin` already names an endpoint. `exclude` says the same
kind of thing in the other direction and adds no table. It is normalized through the same callable as
`pin` (an exclusion compared literally would not match an alias, and would route to exactly the
endpoint it was written to avoid), it is a complete declaration on its own, and it is reported ahead
of any property of the endpoint — a caller who excluded something also unrouted needs the reason they
can act on.

🚨 **An `exclude` naming an endpoint this fleet does not have is a `404 unknown_endpoint`, not a
warning.** This was built as a disclosure first and changed, because the disclosure argument assumes
the one thing the proxy cannot check. Such a name is either "not in this fleet" or "in this fleet,
under a spelling you got wrong", and from inside those are identical bytes; serving the second sends
the request to precisely the endpoint the exclusion existed to avoid and reports success. Same shape
as the `finish_reason` repair that became a silencer, and as the GBNF grammar OpenRouter refuses
rather than drops. It is also what `pin` already does: a caller that must spell an endpoint correctly
to demand it does not get to misspell one to avoid it. Naming the same endpoint in `model` and
`exclude` is a `400` — a contradiction wholly visible in the request, refused where the caller can
see it rather than resolved to an empty candidate set that reads as a fault in the fleet.

**One bug, found by running it.** The refusal echoed the **normalized** name: `normalize` lower-cases
and strips, so a caller who wrote `"tierX"` was told `'tierx'` — a word it never sent, in the one
message whose whole job is helping it find a typo. It is the reason `pin_as_written` exists, missed in
the other direction, and the suite was green throughout because the refusal fired and named something
plausible. `Intent` now keeps the *pairing* rather than two sets, and matches on the normalized form
while reporting the written one.

**No new error code.** An unresolvable exclusion is the existing `unknown_endpoint`; a contradiction
is the existing `invalid_request_error`. `docs/api.md` §1.7.1. The shipped example gains one additive
profile (`bulk`) so the config path is exercised on every boot rather than documented and unrun.
`tests/test_intent_config.py`. Sixteen mutations, every guard observed going red — one first failed
by raising from `intent.py` rather than by asserting, which is half a guard, and was rewritten.


### Changed — two cleanups on the hot path and the client boundary

Landed 2026-09-01. Both were diagnosed and left open; neither changes a contract.

- **`/rs/v1/chat` no longer walks the latency ladder once per endpoint per request.**
  `EnrichedApi.facts()` runs on every request to resolve one intent, and it called
  `timeout_model.advise` once per endpoint to fill a single field — N ladder walks, on the hot path,
  for a number that is a median over thousands of samples and cannot meaningfully move between two
  requests a millisecond apart. Memoised for 1s: **14,000 `advise` calls became 7** over 2,000
  `facts()` calls, and the call itself went from 80µs to 47µs.
  🚨 `typical_ms` is **not** dropped and **not** made optional, which is why this is a memo rather
  than a flag: `prefer=latency` and `prefer=balanced` rank on it, and an endpoint with no samples
  sorts as SLOW — so omitting it would silently re-rank every intent-routed request rather than lose a
  display field. The memo is keyed by endpoint as well as time, so an endpoint that appears later is
  priced rather than served from a map that never saw it.
- **The client SDK is now driven over a real socket.** `tests/e2e/test_client_sdk_live.py` uses
  `ASGITransport` and says so — it exercises request building and envelope parsing, not httpx's
  networking — which left the SDK's own transport configuration asserted as a comparison between two
  constants. `tests/e2e/test_client_sdk_socket.py` runs the real proxy under real uvicorn on a real
  ephemeral socket, points a client the SDK built **itself** at it, and observes: a real round trip,
  real SSE framing over the wire, connection reuse inside the keepalive window, and 🚨 **the idle
  socket actually being retired** after `CLIENT_KEEPALIVE_EXPIRY_S` (4.5s) — §1.3's ordering
  invariant watched rather than computed. It reaches into httpx's pool deliberately, and asserts the
  internals exist rather than skipping when they do not, so an httpx upgrade fails loudly instead of
  the guard going quietly blind.
- Eight mutations, every guard observed going red by assertion. One first SURVIVED and was a real
  weakness: the memo-expiry test advanced its fake clock by *the TTL*, so a mutation setting the TTL
  to a billion seconds moved the goalposts with it.

### Fixed — the shutdown drain now has a ceiling, and it is published once

Landed 2026-09-01. SIGTERM was already the correct signal and that stayed settled by experiment
(`docs/ledger.md`); the drain semantics are unchanged and `tools/sigterm_drain_probe.py` still
measures **78.2s** for the same scenario. What was wrong is that **nothing enforced a ceiling**.

- 🚨 **The tail after the drain was unbounded.** The 30s drain was bounded; everything after it was
  not — and `OnDemandManager.close` makes a **network call per held GPU lease**, so a dispatcher that
  stopped answering hung the lifespan shutdown indefinitely, past any stop-grace an operator had set,
  at which point the container SIGKILLs a process that has flushed nothing. Lease release and pool
  close now run **concurrently under one bound**, and a timeout there costs a lease its TTL and never
  the flush that follows.
- **Every phase is named and bounded**, and `SHUTDOWN_DEADLINE_S` is their **sum** rather than a
  literal: drain 30s, straggler unwind 3s, teardown 5s, queue flush+join 10s → a 48s ceiling on the
  lifespan shutdown. 🚨 There is deliberately **no outer `wait_for`** around `shutdown()` — it would
  cancel `queue_db.close()` mid-flush, which is the SIGKILL failure the drain exists to avoid, and a
  half-written completion row is worse than a slow exit. The ceiling is true by arithmetic and the
  arithmetic is pinned.
- **`RECOMMENDED_STOP_GRACE_S` is computed in one place** and read by the uvicorn argument, the
  `Dockerfile` label, the Dockerfile prose, a startup log line and the probe. It used to be three
  numbers and a comment.
- ⚠️ **The recommended container stop-grace went UP, from 90s to 108s** — and that is the fix rather
  than a regression. 90 came from a *measured* worst case (~78s) taken while the tail had no bound at
  all, so it was an observation rather than a ceiling and a wedged dispatcher would have blown
  through it. 108 is the first number that is a ceiling. **Update `stop_grace_period` /
  `--stop-timeout` accordingly**; below it, `docker stop` SIGKILLs the proxy with zero budgets and
  zero completion rows persisted, and the 10s default works fine while the proxy is quiet — which is
  how it gets tested and why it first fails under load.
- **`timeout_graceful_shutdown` is no longer derived from `_DRAIN_DEADLINE_S`.** The old
  `_DRAIN_DEADLINE_S + 18` carried a comment claiming uvicorn's budget must exceed the drain "or it
  hard-kills the process mid-flush", which assumes a nesting that does not hold — uvicorn hands over
  to the lifespan shutdown. Its **value is unchanged**, so nothing about the measured behaviour moves;
  what changed is that it is its own knob for its own job.
- `tests/test_shutdown_budget.py`. Ten mutations, every guard observed going red by assertion.

### Added — key lifecycle, and the abuse control DRR is not (Workstream I)

Landed 2026-09-01. What was left of Workstream B, and the oldest unclosed thing in the repo:
`created_at` was written, surfaced in two views, and **read for no decision**.

**Expiry.** `expires_at` (an absolute date in a keys file) / `expires_in_s` (a duration over the API).
A key with no expiry never expires, so an existing registry behaves exactly as it did. An expired key
is a `401 invalid_api_key` with **its own sentence and the same code**: a caller can act on no
distinction between "expired" and "unknown", so §2.1 mints nothing new, but the operator reading the
log can — one means check what you pasted, the other means issue a successor. A restart never extends
a key; the store holds the absolute instant, not the duration it came from.

**Rotation.** `POST /rs/v1/admin/keys/{key_id}/rotate`, one action, because the manual version is two
calls in an order that matters and **both orders are wrong** — enrol-then-revoke leaves the successor
live and unknown to the caller, revoke-then-enrol leaves nothing working. The successor inherits the
predecessor's policy. `overlap_s` defaults to **0** (revoke now, and the response says so); with an
overlap the predecessor gets an **expiry** rather than a tombstone, so it survives a restart a timer
would not. If the successor cannot be minted the predecessor is **untouched**.

**Per-key address binding.** `bind` is an ADDITIONAL constraint, never a way for a key to widen what
an address grants: a bound key is refused outside its CIDRs and is otherwise exactly the credential it
always was. An unparseable entry is a 400 at enrolment and matches nothing at resolution — a narrowing
that failed open would be worse than none. 🚨 It is checked against the *resolved* address, so a
binding is only as trustworthy as `ROADSTEAD_TRUSTED_PROXIES`, and `GET /rs/v1/admin/keys` reports
`checked_against` rather than leaving an operator to infer which case they are in.

**`requests_per_minute` — and no 429.** DRR is fairness *under contention*: a caller alone on a quiet
fleet is unthrottled by design, which is correct for fairness and exactly why it does not bound a
runaway. Crossing this costs a caller exactly what crossing `daily_spend_usd` costs it — **one
priority band and paid spill, never local capacity** — and mints no code. A rate *limit* would be a
fourth spelling of "no" and would put a mistyped threshold in a position to take a caller offline.
`roadstead/rate.py` is the **sixth** pure-computation module; CLAUDE.md's layout was updated because
`tests/test_pure_modules.py` fails until it is.

🚨 **Degradations do not STACK.** A caller over both thresholds drops **one** band, not two — enforced
in `ProxyState.effective_priority`, which consults at most one standing, so a third threshold cannot
compose by accident. Each crossing reports through the degradation seam **separately**, because one
means "look at the bill" and the other means "look for a loop". `effective_priority` is split from
`spend_demote` so a management *read* fires no notice.

**Three more found by running it, none of which a test would have reached.**

1. The rotate response was assembled from the **predecessor's** row, so a successor registered with
   the wrong policy would still have been reported as inheriting it — the response was not evidence.
   It now reads back from the registry, and the test asserts against the registry too.
2. A successor inherits the policy but **not the expiry**, and came out permanent — a silent
   weakening of a control the operator deliberately set. The default is right (inheriting an absolute
   instant would mint a successor that expired seconds later) so it is **disclosed**, not changed.
3. 🚨 `overlap_s` **extended** a predecessor past its own expiry. A day-long overlap on a two-hour key
   pushed its expiry a day out — a rotation quietly lengthening a credential, which is the widening
   this repo refuses everywhere else. Truncated to the earlier instant, and said out loud.

`docs/api.md` §1.5, §1.6 (a two-threshold table, and the non-stacking rule), §3.3, §3.4, one row in
§3's route table. `tests/test_key_lifecycle.py`, `tests/test_rate_threshold.py`. Twenty-six mutations,
every guard observed going red by assertion.

### Added — a read-only admin scope, and the audit trail (Workstream H)

Landed 2026-09-01. Two changes to what an admin identity *is*, landed together because they touch the
same three files and answer halves of one question: the read views report a **state**, and a state
cannot say who put it there; and an operator who wanted somebody to be able to *look* had to give
them the ability to change everything.

**`admin_readonly` — a narrowing that can never widen.**

- A credential with `admin: true, admin_readonly: true` reaches every `GET` on the management plane
  and is refused, with a 403 that says why, on everything else. On a key without `admin` it is a
  **400 at enrolment**, not a silent no-op: an operator who wrote it believes they issued a safer
  credential than they have.
- **The read/write split comes from the HTTP method**, not from a list of write routes — a route list
  is a second thing to keep in step with `routes.py`, and when it falls behind the failure is silent
  and widening. It lives in the shared gate, so the four control routes that predate the management
  plane inherit it without a line of their own.
- **`identity.py` decides, everything else renders.** `IdentityResolver.admin_denial` owns which of
  the three refusals applies; `deny_non_admin` renders it. An AST guard fails if `may_admin_write` or
  `admin_readonly` is read anywhere but `identity.py` — a second place deciding what a scope permits
  is the shape of the `_remote_ip` that had to be removed from `management.py`.
- Spelled `:readonly` in the shared identity grammar, so `ROADSTEAD_API_KEYS` and `ROADSTEAD_ACL`
  express it too. 🚨 **A narrowing beats an overlapping grant**: `127.0.0.1=ops:admin:readonly` is
  read-only even though loopback is a built-in admin net. If the widest grant won, that line would
  silently be a full grant for every operator who wrote it.

**`GET /rs/v1/admin/audit` — who changed what, and when.**

- **Every mutating admin route records**, including flags, pause, resume and maintenance, which touch
  no overlay state and would otherwise record nothing. A trail covering only some of them is worse
  than none, because a reader assumes completeness. The completeness guard is driven from
  `routes.py`, not from a list: all **11** mutating admin routes, both spellings.
- **A record names the credential and never carries one** — `key_id`, never the key or the digest.
  Both the key label and the address are recorded always: a record with an address and no `key_id`
  means an address-derived admin made the change, which is meaningful and slightly alarming, and
  collapsing them into one actor string would hide which factor authorized it.
- **It reports its own limits as data**: `persisted` (in-memory is the default), `dropped`,
  `capacity`. It is an operator-facing change trail, not a security log of record — and says so,
  rather than presenting itself as complete while being neither durable nor unbounded.
- Written on the loop, persisted off it. The overlay lock moved onto the overlay, because two modules
  write that file now and a lock owned by one of them serialises half the writers.

**Three bugs, and two of them were found by RUNNING the page.** The suite was green for all three.

1. A control-route record only reached disk if a *later* overlay write happened — so the most recent
   entry, the one an operator looks for after an incident, was the one a restart lost. A drain is
   taken in a hurry, often right before the restart that would drop the record of it.
2. The Fleet view's Pause/Resume was the one write control not wrapped in the scope guard. It
   rendered, it worked, and a read-only operator would have learned their scope from a 403 *after*
   clicking. There is now a test that finds every write control from the HTTP method it sends and
   requires it guarded.
3. 🚨 `render()` fired before `primeChrome()` resolved, so the **first paint had no scope** and the
   guard defaulted to allowed — every write control live for a read-only operator until the 30-second
   heartbeat. The first paint is the one somebody clicks.

`docs/api.md` §1.5, §3.3 and a new §3.8; two rows in §3's route table.
`tests/test_admin_audit.py`. Twenty mutations, every guard observed going red by assertion — no
timeouts, no collection errors.

### Fixed — one context-fit predicate, and the dead fourth copy of it

Landed 2026-09-01. "Does this request fit in this much context?" was written out by hand at four
sites — the admission gate (`lifecycle.handle_submit`), the failover gate (`failover.plan`), the
spill gate (`scheduler._admit`) and the WAL-recovery shadow tally (`service`). It is now
`cost_model.context_fit`, called by all four.

- **🚨 The recovery tally never fired.** It read `req.est_input_tokens or 0`, but that field is
  cached by `scheduler.enqueue` and `recover_queued` builds its `QueuedRequest` straight from the
  WAL row — the recovery path has not reached the scheduler yet, so the estimate was always **0**
  and the tally could only fire when `max_tokens` *alone* exceeded the ceiling. The audit that added
  it (2026-07-02) did so precisely because recovered oversized requests were invisible to the
  `context_gate_enforce` flip evidence; they stayed invisible. Measured on the corpus this file's
  differential harness uses: the dead copy disagreed with the shared predicate on **27 of 153**
  cases. `service.py` had also carried an unused `estimate_input_tokens` import ever since — the
  residue of a fix that meant to call it.
- **Nothing else changes behaviour.** The three live gates were byte-identical in effect and are
  verified so rather than asserted so: a differential harness ran the pre-split implementations
  transcribed verbatim from `7563fa6` (2026-09-01, *"Give the operator a face"*) against the
  shared one over **2295** combinations of payload
  shape, payload type, ceiling and endpoint name, comparing the boolean *and* the refusal-message
  bytes. Zero mismatches.
- **🚨 The consequences stay different, which is the whole risk of this refactor.** Admission is
  shadow-or-422 behind `context_gate_enforce`; failover refuses unconditionally (there the
  alternative to refusing is a guaranteed backend 400, not a request that probably works); spill
  defers; recovery counts and never rejects. `context_fit` returns the answer and its arithmetic and
  takes no action at all — folding the consequence in would have armed a flag nobody flipped, and
  that is pinned by a test.
- **`CONTEXT_OVERFLOW_MARKER` is defined once per side of the client boundary.** It is wire contract
  (`docs/api.md` §2.2) and was previously typed out at two call sites. An AST constant sweep — not a
  substring grep — now allows exactly one spelling in the server and one in `roadstead.client`,
  which must keep its own because it imports nothing from the server; the document pins them
  together. A second AST guard fails if any module re-derives a ceiling comparison from
  `estimate_input_tokens` instead of calling `context_fit`.
- `tests/test_context_fit.py`. Twelve mutations, every guard observed going red **by assertion** —
  no timeouts, no collection errors.

### Added — the management UI (Workstream G)

Landed 2026-09-01. `GET /rs/v1/admin/ui` — the roadmap's "manage and monitor Roadstead as a
standalone product" line. **Off by default**: unset `ROADSTEAD_ADMIN_UI` and the route does not
exist, which is absence rather than refusal, the same posture as `ROADSTEAD_TRUSTED_PROXIES`.

- **One static HTML file, vanilla JS, no bundler and no external references at all.** The
  dependency list is still six packages and a test asserts it. The file ships in the wheel, so it is
  public surface — the same argument as `roadstead.testing` — and the CSP it is served with
  (`default-src 'none'`, `connect-src 'self'`, no host anywhere) forbids an external reference
  outright rather than trusting a reviewer to notice one.
- **It shows the GAP, not the config.** Declared beside in force wherever the two can disagree:
  the catalog's slot seed against what discovery left (with whether the engine publishes it at all),
  quota `in_force`/`declared`/`runtime`, declared band against effective band — which is the only
  place a spend threshold is visible, since crossing one mints no error code. A pair collapses to
  one value when they agree, so a highlight means something.
- **🚨 Auth is HTTP Basic and the credential is an API key.** A browser cannot attach a bearer token
  to a navigation and `EventSource` cannot set a header at all, so a browser-facing surface needs a
  scheme the browser carries. Minting a *password* to go in it would be a second credential kind
  with its own store, rotation and revocation, parallel to a registry that already does all three.
  So the key goes in the password half and the username is ignored. **No cookie is minted, so CSRF
  never becomes reachable** on the mutating routes — the concern that has bound this since the
  roadmap was written is answered by not creating it.
- **The door refuses 401 where the plane answers 403**, with `WWW-Authenticate: Basic`. The
  401/403 split is right for an API client and a dead end for a browser: a 403 produces no password
  box, so an operator arriving with no credential — everyone, the first time — has no way to answer
  the refusal. The challenge is identical whether or not keys are configured, so it discloses
  nothing about which §1.5 regime is in play.
- **🚨 Every field the page reads is pinned against a real response.** There is no compiler, no
  schema and no types here: rename a field on the server and one cell renders "—" forever while the
  page looks healthy. So every read goes through `pick(obj, "a.b.c")` — a rule with a test, not a
  style — and the guard walks every extracted path against responses a real service produced. It
  found a live bug on its first run (the feed read `agent_id`/`total_ms` off a frame publishing
  `agent`/`duration_s`), and rendering the page in a browser found two more the tests could not see:
  a one-level `flat()` stringifying nested nodes as `[object HTMLSpanElement]`, and boolean
  attributes rendered empty so `[data-same="true"]` never matched and no pair ever collapsed.
- **The asset read goes off-loop** and is cached after the first hit. A dashboard on a fast cadence
  is exactly the load that finds a concurrency violation.
- `docs/api.md` §1.5, §3 and a new §3.7. Sixteen mutations, every guard observed going red.

### Changed — `GET /v1/stream` is admin-gated 🚨 BREAKING

It was not, and a frame there names the caller, the endpoint, the tokens and the timing of **every
call the fleet serves** — the live form of `/rs/v1/admin/callers`, which has been gated since it
existed. `handle_maintenance_list` carries a note from the previous round of this tightening ("was
unauthenticated — tightened with the rest"); the stream was missed because it reads as plumbing
rather than as a view. A consumer polling it from a host outside the admin nets, with no admin key,
now gets a 403. It is also aliased at `/rs/v1/admin/stream` — same handler, same gate — because
`EventSource` cannot set a request header, so the UI can only reach it through credentials the
browser attaches by directory.

### Changed — `Authorization: Basic` is now one of our credential forms 🚨 BREAKING

Previously ignored as "somebody else's auth". The key rides in the **password** half. A deployment
that has keys configured *and* callers presenting an unrelated Basic header will now see those
callers refused with a 401 rather than quietly identified by address — which is §1.5 rule 1 working,
but it is a behaviour change and it is recorded as one. A header that cannot be base64-decoded is
**no credential at all** rather than a failed one, so it still falls through to the address: a header
we cannot parse was probably never meant as ours. Unrecognised schemes are still ignored.

### Fixed — trusted proxies, and the admin grant behind one 🚨

Landed 2026-09-01. **Not breaking**: `ROADSTEAD_TRUSTED_PROXIES` is empty by default, so a
deployment that sets nothing behaves exactly as before, byte for byte. The header is not "validated
and rejected" by default — it is never consulted.

- **`identity.remote_ip` read `request.client.host` and nothing else**, so anything in front of
  Roadstead — a TLS terminator, an ingress, a sidecar — collapsed every caller into the proxy's
  address. Two consequences, and the second is a security bug: `ROADSTEAD_ACL` silently stopped
  distinguishing anybody, and if the proxy's address fell in the admin nets (**loopback and
  docker-internal are there by default**, and a sidecar usually is one of them) the control plane
  was granted to everyone who could reach the proxy. Latent only because nothing told anybody to
  deploy that way — and Workstream G's UI is what makes a front proxy normal.
- **`ROADSTEAD_TRUSTED_PROXIES`** (addresses or CIDRs, comma-separated, empty by default) is the
  opt-in. `X-Forwarded-For` is read only from a listed peer, and **the caller is the rightmost hop
  that is not itself a trusted proxy**. Not the leftmost: that element is whatever the caller wrote
  before a proxy appended what it observed, so reading it re-introduces the spoof — the
  self-asserted `agent_id` bug in its third costume. The walk is used rather than counting hops
  because the trusted set is CIDRs, whose *width* is not its depth.
- **🚨 A forwarded address does not inherit the BUILT-IN admin nets.** The whole justification for
  auto-granting admin to loopback and docker-internal is that reaching them meant already being on
  the box, and a front proxy is exactly what makes that untrue. `ROADSTEAD_ADMIN_NETS` and an
  `admin` API key are unaffected — the operator's explicit statements stand, the inherited default
  does not. Safe to do unconditionally *because* it is gated on trusted-proxy configuration, which
  is empty until somebody opts in, so no deployment can lose a grant it has today. A trusted proxy
  that forwards **no** header is marked forwarded too: a front proxy that lost its config must not
  become an administrator.
- **Two chain shapes fail closed rather than back to the peer** — an unparseable hop standing where
  a caller's address should be, and a chain longer than 32 hops. Resolving to the peer there would
  hand the proxy's identity and its grants to anyone who typed junk into a header. Both become
  `unknown`, which is not an address, matches no registration and is refused.
- **`identity.py` is now the only place an address is resolved**, with an **AST** guard rather than
  a substring one (`management.py` had its own `_remote_ip`; a second was a second answer to a
  question with one owner, on the surface where getting it wrong grants the control plane). A
  substring sweep for `client.host` is satisfied by prose — `getattr(getattr(request, "client",
  None), "host", "")` is the same bug and walks straight past it.
- **The management plane shows it**: `GET /rs/v1/admin/config` reports `trusted_proxies` beside
  `admin_nets` split into `builtin` and `operator`, and an unparseable `ROADSTEAD_TRUSTED_PROXIES`
  entry becomes a `hooks.config_notice` — a typo removes trust rather than granting it, which is the
  safe direction and for that reason a completely silent one.
- `docs/api.md` §1.5 rule **4**, §3 and §3.5. `tests/test_trusted_proxies.py`; thirteen mutations,
  every guard observed going red — one of which **survived** the first pass and was real (the
  spelling-sensitive sweep above), and one of which errored at collection instead of asserting and
  had to be rewritten before it proved anything.

### Added — the management plane (Workstream E)

Landed 2026-09-01. **Nothing here is breaking, deliberately**: the plane is additive, the four
pre-existing `/v1/admin/*` control routes keep working unchanged, and §3 mints **no new error code**
— the third time that call has been made (§1.6, §1.7).

- **`/rs/v1/admin/*` — the management plane** (`roadstead/management.py`, `docs/api.md` §3). Six new
  routes: `config`, `keys` (GET/POST), `keys/{key_id}` (DELETE), `callers` (GET),
  `callers/{agent_id}` (PATCH), `providers`. *Why not `/v1/admin`:* `/v1` is versioned by OpenAI, and
  management is the surface most likely to need its own second version. The four legacy control
  routes are served at **both** spellings — same handler, same gate — so no consumer breaks and an
  operator has one prefix rather than two. A test pins the pair, because an alias that stops aliasing
  is a control surface that works on one spelling and 404s on the other.
- **Keys are enrolled and revoked at runtime.** Workstream B made a key the caller's identity and
  left "create one" meaning *edit a file and restart*. A generated secret is returned **once** and
  never stored (the registry keeps the digest); a plaintext secret is never *accepted*, because a
  secret in a request body lands in an access log. Revocation is never refused on provenance grounds
  — a key declared in the environment can be killed now, and the response says the declaration will
  outlive the reason it is dead.
- **🚨 Revoking the LAST key is disclosed, because it changes the identity regime back.** An empty
  registry is not in play (§1.5 rule 2), so a presented key — including the one just revoked — is
  ignored again and the address decides. Nothing is escalated; what stops being true is *revoked
  means refused*. Found by the e2e journey rather than reasoned about, and fixed by **disclosing**
  it: narrowing rule 2 would 401 exactly the deployment that has just emptied its registry on
  purpose. Disclosures arrive in a `warnings` **array**, because two can be true of one action.
- **Caller quotas are editable at runtime** — `weight`, `max_balance_ss`, `default_priority`,
  `degrade_ok`, `spill_ok`, `daily_spend_usd`, the same set `agents.yaml` accepts, pinned against it
  *and* against the dataclass. A weight change reaches the **live** DRR budget (or the edit would
  apply only to callers the proxy has never seen — "it did nothing" for exactly the busy caller it
  was aimed at) and moves the **rate, never the balance**.
- **🚨 §1.6 is inherited at the write boundary, not re-implemented there.** No field can express a
  rejection, so the plane cannot mint a policy admission refuses to honour. An unknown field is a
  400 naming the known set — never a silent drop, on the surface whose purpose is exposing them.
- **A runtime edit never rewrites your config file.** Changes go to a JSON overlay
  (`ROADSTEAD_ADMIN_STORE`) layered over the files at startup: enrolments and overrides on top,
  revocations last. Comments survive, the operator's editor is not raced, and "who changed this" is
  answerable. **A change that cannot be persisted still takes effect and says so** (`persisted:
  false` plus a reason) — refusing a revocation because a disk is read-only is a correctness
  argument answered, in the moment, by a breach.
- **`hooks.config_notice` — a fourth reporting seam, and the first that reports *in*.** Three loaders
  drop unknown keys (`models.yaml` `policy:`, `agents.yaml`, a keys-file entry) and each has cost
  something, because a dropped `spill_ok` is indistinguishable from a caller who never opted in. The
  keys are still dropped and still logged, but now **retained** and readable at
  `GET /rs/v1/admin/config` — the operator who reads a boot log and the one who asks why a knob does
  nothing are not the same person, a week apart.
- **The read views report the GAP, not the config.** `providers` shows declared capacity (the
  catalog seed) beside what is in force and whether the engine publishes it at all, so "discovery
  agreed" and "discovery never ran" stop being indistinguishable. `callers` splits quota into
  `in_force` / `declared` / `runtime`, so an override reads as a difference rather than a label.
- **🚨 Keys stay FLAT, and that answers the roadmap's open multi-tenancy question rather than
  deferring it.** The quota holder, the DRR share and the spend cap are keyed on `agent_id`, never on
  the key — so several keys naming one `agent_id` already give a team one budget with per-key
  revocation and per-key priority. The view groups by it to make that visible.
- **New supporting surface:** `IPIdentityMap.entries()` (registrations only — the built-in internal
  nets are deliberately absent, or an operator would think removing a line closes a door that is
  built in), `Scheduler.agent_snapshot`, `BudgetManager.reweight`, `KeyRegistry.revoke` and key
  provenance. `tests/loop_affinity.py` arms the three newly-mutable objects and grew dotted-path
  resolution to reach the registry behind the identity resolver.

### Breaking

Landed 2026-09-01 with **Workstream C** (the enriched API) and **Workstream F** (hardening). Two of
these touch the wire contract in `docs/api.md`; the rest are internal or additive.

- **🔒 `POST /v1/submit` is REMOVED.** The enriched API at `/rs/v1/*` replaces it —
  `docs/api.md` §1.7, with a field-by-field migration map in §1.9. *Why removed rather than
  deprecated:* the envelope carried no intent vocabulary, no attribution and no timing, and each of
  those would have had to be bolted onto a shape never designed to hold them — three additive
  changes to a contract we had already decided to retire, and a second surface to keep correct
  meanwhile. It also let the body declare `agent_id`, which Workstream B had already had to gate;
  the enriched door takes identity only from the credential or the address, so the fair-share key
  can no longer be named by the request at all. *Migration:* `POST /rs/v1/chat`, `endpoint` →
  `model` (a pin) or `intent` (let Roadstead choose), `timeout_s` → `deadline_s` (and usually: omit
  it), `agent_id` → drop it and present a key. `roadstead.client` speaks the new API and is the
  shortest path across. The OpenAI doors are unaffected.
- **🔒 A new north face: `/rs/v1/*`, versioned separately from `/v1/*`.** Three routes —
  `GET /rs/v1/models`, `POST /rs/v1/plan`, `POST /rs/v1/chat`. *Why its own version:* `/v1/*` is
  versioned by OpenAI. Sharing the number would mean either following somebody else's release
  cadence or publishing a `/v2/chat/completions` that is not OpenAI's v2.
- **🔒 The OpenAI door gains four `X-Roadstead-*` response headers** (§1.8) and **no body fields**.
  *Why headers:* a client validating against OpenAI's schema must not break because it pointed at
  Roadstead, and "we only added fields" is not a defence — strict validators reject unknown keys.
  Additive and ignorable; listed because §1.8 is now contract.
- **`docs/api.md` §1.7 adds NO new error code.** An intent nothing can satisfy is the existing
  `unknown_endpoint`. A caller already classifies that, and a second spelling of "nothing here can
  serve you" would buy nobody anything. The absence is deliberate, as it is for §1.6.
- **`failover.py`'s degrade refusal no longer cites `llmproxy/agents.yaml`.** A dead monorepo path
  in operator-visible error text, carried since the extraction; it now names the knob
  (`degrade_ok`) rather than a file that does not exist here. An error-message reword is a breaking
  change under §2.2 even when the `code` is untouched — this one carries no deferrability marker, so
  no classifier can be affected, but it is recorded because the rule is the rule.

### Added

- **`roadstead.client` — a client SDK, shipped in the package**, the way `roadstead.testing` is.
  `AsyncRoadsteadClient` (real) plus a blocking `RoadsteadClient` that owns a private loop on a
  worker thread. Typed views over the enriched envelope, and typed errors whose `deferrable`
  property **classifies on the `code`** with §2.2's legacy marker substrings kept only as a
  fallback — which is the migration §2.2 asked a shipped client library to make.
  🚨 **It imports nothing from the server**, enforced by AST: a consumer sending an HTTP request
  should not be installing Starlette, uvicorn, PyYAML and jsonschema, and a client reading the
  server's own constants would agree with it *by construction* and could never catch a drift. Its
  contract literals are transcribed from `docs/api.md` and checked against it — the same two-ended
  pin `tests/wire_contract.py` uses from the other side.
- **Model abstraction.** A caller declares an `intent` (a capability profile) and Roadstead owns the
  choice; a `model` is a pin and is treated as a constraint on routing. `roadstead/intent.py` is a
  fifth pure-computation module. Two ranking rules are doctrine: a real-cost endpoint sorts last
  under every preference (or "intent" becomes a back door around the spill doctrine), and an
  endpoint with no latency samples sorts as slow (or the resolver prefers the backend it knows least
  about *because* it knows least about it).
- **Per-request substitution narrowing.** `substitution: {"degrade": false, "spill": false}`.
  🚨 Narrows only: both gates take the AND with the operator's opt-in, so `true` grants nothing.
  Declining is a defer, never an error.
- **`EndpointConfig.kind` and `EndpointConfig.capabilities`**, mirrored whole from the catalog
  stanza. This makes `tool_calling` and `structured_output` load-bearing for the first time rather
  than documentation — the vision-ledger lesson applied to the rest of the block. A capability
  outside the known vocabulary now warns at load instead of silently matching nothing.
- **`GET /v1/models` gains `kind`**, and sorts on it rather than on a literal `{"embed", "rerank"}`
  set. That set was correct for exactly one catalog — the example one — so any deployment whose
  embedder was named something else silently got an embedder as `data[0]`, which is the default a
  client with no configured model picks up.
- **The concurrency invariant is guarded** (Workstream F). `tests/loop_affinity.py` arms it,
  `tests/e2e/test_soak.py` runs sustained overlapping load and asserts thread affinity, slot
  conservation, DRR budget conservation, ledger conservation and that every request was answered —
  and includes a test that mutates budget state from a second thread on purpose, so the guard is
  observed going red from inside the suite forever. `tools/soak.py` is the unbounded version.
- **The "pure computation, no I/O" claim is guarded** (`tests/test_pure_modules.py`). `CLAUDE.md`
  calls those five modules the crown jewels and said "keep them that way"; nothing checked it. What
  purity buys is that every scheduling, costing, deadline, spend and routing decision is testable
  against a fleet that does not exist — and the first `httpx` import into one would end that while
  the suite stayed green.

### Breaking — earlier

Landed 2026-09-01 with Workstream B (identity and API keys), which also clears scrub items **S1** and
**S3**. One of these touches the wire contract in `docs/api.md`; the rest are deployment surface.

- **`/v1/submit` now requires an identity.** It had none: the OpenAI doors were ACL-gated and this
  one was not, on the same port, so any caller could reach it unenrolled *and* claim any `agent_id`
  it liked — including one with a better DRR weight. The fair-share key was self-asserted. It is now
  gated exactly like the OpenAI doors. *Why:* a scheduler whose unit of fairness is caller identity
  cannot let callers choose their own. Migration: present an API key, or enrol the source address in
  `ROADSTEAD_ACL`. Loopback and docker-internal callers are unaffected.
- **A presented API key overrides a body-declared `agent_id`.** An address still only fills in an
  `agent_id` the body omitted. *Why:* a verified credential is a stronger statement about who is
  calling than anything in the body; letting the body win would launder a claim past the credential.
- **The ACL ships no registrations, and no addresses at all.** It carried a private fleet's LAN —
  ten hosts by address, role and deadline floor (scrub item **S1**). A fresh install now allows
  loopback and docker-internal and refuses everything else. *Why:* a default that happens to match
  somebody's LAN hands an identity, a DRR share and a deadline floor to whatever answers at an
  address we guessed. Migration: `ROADSTEAD_ACL=<ip-or-subnet>=<agent_id>[:priority][:min_timeout_s][:admin]`,
  comma-separated. `LLM_PROXY_ACL` is still read — the one legacy env name kept, because a proxy
  that silently stops recognising its callers on upgrade fails closed in the most confusing way
  available.
- **🔒 A new error code, `invalid_api_key` (401)** — `docs/api.md` §2.1, so this one is the wire
  contract. It is deliberately NOT `access_denied`: 403 says *this source is not enrolled*, 401 says
  *this credential is wrong*, and collapsing them sends an operator to the wrong file. A presented
  key that does not resolve never falls back to the address, which is the reason a separate code was
  needed at all.
- **A caller that declares no `priority` now takes its identity's default**, rather than always
  `P1_TURN_SUPPORT`. *Why:* the OpenAI doors already did this; `/v1/submit` ignoring the same
  registration meant one caller landed in two different bands depending on which door it used.
- **`agents.yaml` is a generic example**, on the caller archetypes in `docs/roadmap.md`
  (`chat-assistant`, `coding-assistant`, `summarizer`, `extractor`, `hygiene`). It shipped one
  fleet's agent roster with weekly request volumes and infra paths — a fourth private inventory,
  found the way `usage_rates.py` was found during S2, by walking past it. The same arrangement as
  `models.yaml`: the example IS the default, so it boots a fresh install and cannot rot. The DRR
  sizing reasoning and the whole `degrade_ok` doctrine are kept; only the vocabulary changed. The
  built-in simulation scenarios (`roadstead.simulation`) were renamed to match — their measured
  shapes are untouched.
- **`ANVIL_DISPATCHER_URL` → `ROADSTEAD_ON_DEMAND_DISPATCHER_URL`, and it has no default.** The
  old default was one deployment's dispatcher address (scrub item **S3**). *Why:* the GPU-slot
  dispatcher is a host-side service Roadstead does not own or ship, so a baked-in URL could only
  ever be somebody else's. An `on_demand` endpoint with this unset fails `ensure_loaded` as
  unreachable — the same clean deferrable error as a dispatcher that is genuinely down.

All four below landed together on 2026-08-31 with the catalog redesign, and none touches the wire contract
in `docs/api.md` — the OpenAI surface is unaffected.

- **`models.yaml` is a new schema: `providers:` + `endpoints:`.** Connection and engine on one side,
  capacity and policy on the other; each endpoint names its provider. The old flat `models:` block,
  `proxy_endpoint:`, `endpoint_class:`, `fallback:` and `backend_engine:`-on-the-endpoint are gone.
  *Why:* a remote provider fronts many models behind one base URL and one credential, and the flat
  shape had nowhere to say that once. Migration: move `host`/`port` into a `providers:` entry, key
  each endpoint by its class, rename `fallback:` → `failover_to:`.
- **The shipped catalog is now a generic example** on RFC 5737 addresses, with classes `tier1`,
  `tier2`, `tier3`, `embed`, `rerank`. It is the default, so a fresh install boots. *Why:* the data
  was one private fleet's hardware inventory (scrub item S2); the schema is the contract, the data
  never was. Real deployments set `ROADSTEAD_MODELS_YAML`.
- **Every environment variable was renamed `COLLECTIVE_*` → `ROADSTEAD_*`** (24 of them). *Why:* the
  last monorepo fingerprint in the runtime surface. The old names are **not** honoured — two
  spellings for one switch is how they come to disagree — but a `COLLECTIVE_*` variable that is
  still set is reported at startup by name, with its replacement, because a flag that stops working
  in silence is the failure that matters.
- **`model_catalog`'s API changed with the schema.** `ModelEntry` → `ProviderEntry` +
  `EndpointEntry`; `Catalog.proxy_endpoints()` → `Catalog.routed()`. Seven `build_*` helpers that
  only served monorepo consumers were deleted (`build_telemetry_units`, `build_dispatcher_entries`,
  `build_port_to_role`, `build_host_ports`, `build_role_aliases`, `build_context_windows`,
  `build_valid_providers`), along with the `ModelEntry` fields that fed them.

**Planned:** the `/v1/submit` envelope will be superseded by the enriched Roadstead API
(`docs/roadmap.md`). The OpenAI-compatible surface is unaffected and stays strictly compatible.

### Added

- **`roadstead/spend.py` — money, and the admission decision that spends it** (roadmap Workstream
  D). A fourth pure-computation module beside the scheduler, the cost model and the timeout model.
  - **Admission is ONE decision with THREE outcomes.** `Scheduler._admit` returns
    `DISPATCH` / `SPILL` / `DEFER`. Local capacity is tried first for every caller — nothing about
    money appears above that test — so an over-cap caller, a caller with no `spill_ok` and a caller
    nobody configured all reach the same local dispatch. Spill is considered only once local has
    said no, which is what makes remote capacity *overflow* rather than a parallel system with its
    own fairness, and it never chains.
  - **Two kinds of money, kept in fields that are never summed.** `usage_rates.py` prices a local
    endpoint at what renting the same class of model would have cost — a saving (`avoided_usd`). A
    remote provider's published price is an invoice (`spent_usd`). Only the second counts against a
    threshold: one that counted the first would throttle a caller for using capacity that is free
    and already paid for.
  - **`publishes_token_costs` has a reader.** OpenRouter's catalogue prices arrive on the same
    discovery pass that reads the context ceiling. They are strings, per single token, scaled to
    per-million on the way in — and 🚨 **a published price of zero is a real price**: reading it as
    unpublished would push a free remote model onto the imputed table and book a *saving* for a call
    made over the internet. An operator-declared `policy.input_usd_per_mtok` beats a published one.
  - **Thresholds degrade and never reject.** Crossing `daily_spend_usd` costs a caller one priority
    band (floored at the lowest) and access to paid spill. It never costs local capacity, and 🚨 **no
    error code exists for it** — `docs/api.md` §1.6 says so where `tests/test_spend.py` reads it
    back and fails if a spend-shaped code ever joins §2.1.
  - **New config:** `spill_to` and `policy.input_usd_per_mtok` / `output_usd_per_mtok` on an
    endpoint; `spill_ok` and `daily_spend_usd` on an agent. `spill_to` resolves only to a routed
    endpoint, the same rule `failover_to` follows, which is what makes the shipped example's
    `tier3 -> spill-reasoning` inert until somebody sets `$OPENROUTER_API_KEY` and flips that
    endpoint to `active`.
  - **`/v1/status` grows a `spend` block** — per-caller totals, the price book, spill counters, and
    who is over their cap. The two money columns are reported separately there too.

- **`roadstead/identity.py` — API keys as the caller identity** (roadmap Workstream B). A key
  resolves to a `Principal` carrying the `agent_id` (the DRR fair-share key, quota holder and budget
  holder), a default priority, an optional `min_timeout_s` deadline floor and an optional `admin`
  scope — so all four travel with the caller rather than with the machine it runs on.
  - **Keys are held as SHA-256 digests**, which is both the storage form and the lookup key: the
    plaintext never outlives the load. A config entry may give `key:` (hashed here) or `key_sha256:`
    (the documented form — a key in a config file is a key in a git history). Configure via
    `ROADSTEAD_API_KEYS` for the one-key container case or `ROADSTEAD_API_KEYS_FILE` for anything
    real; `ROADSTEAD_REQUIRE_API_KEY=1` refuses a request that presents none.
  - **One identity-spec grammar for both registries** — `agent_id[:priority][:min_timeout_s][:admin]`,
    where segments are recognised by shape rather than position, so an operator configuring
    `ROADSTEAD_ACL` and `ROADSTEAD_API_KEYS` in the same file learns one spelling of the same four
    facts. Backwards-compatible with the two forms the ACL already accepted.
  - **The address ACL is demoted to a second factor**, and `IdentityResolver` is the one place that
    knows the precedence — the two OpenAI doors, `/v1/submit`, the five admin gates and the deadline
    floor all read the answer off the principal, so a third factor lands in one file.
  - **An interactive identity floored above its own ceiling is reported at load** through
    `hooks.degradation`. That shape was live for a day in the origin fleet — a caller promoted from
    the background band kept the background 1800s floor against a 600s interactive ceiling — and it
    used to be pinned by a test that asserted about specific fleet hosts. It is now a guard that
    fires for anybody's registration, in either registry.

- **`roadstead/providers/` — the provider interface** (roadmap Workstream A, the foundation the rest
  of the roadmap depends on). The llama.cpp/vLLM branching that was inline in `backend.py` is now
  two adapters behind one interface: `prepare_chat_payload`, `path_for`, `discover_capacity` /
  `parse_capacity`, and a `ProviderDescriptor` that states what a backend publishes, what it
  requires of a request, and what it gets wrong. `backend.py` keeps the transport — pools,
  deadlines, error taxonomy, SSE relay — and providers borrow its probes rather than opening
  sockets of their own.
  - The descriptor makes the capacity asymmetry a declaration rather than a comment: llama.cpp
    publishes real slots and per-slot context, vLLM publishes only a context ceiling and keeps
    `--max-num-seqs` off the API, so its concurrency stays config-seeded.
  - Four call sites that asked `backend_engine == "vllm"` now read the capability they actually
    meant (prefix-cache counters, truncated-tool-call mislabelling, switchable reasoning, which
    discovery probe to run), and `tests/test_provider_interface.py` fails if a new engine-name
    comparison appears outside the config plumbing.
  - Behaviour-preserving, and checked rather than asserted: 563,200 payloads and 105 discovery
    bodies compared old-vs-new, zero differences, side effects included.

- **An OpenRouter provider — the first backend Roadstead does not own.** `EndpointConfig` grew
  `base_url` (superseding `host`/`port`, which cannot express a scheme or a base path) and
  `api_key_env` (the NAME of the environment variable holding the key, never the key), and the
  connection pool is keyed on the URL. Auth is the provider's business, not the transport's.
  - **A provider that cannot honour a constraint now refuses.** OpenRouter cannot enforce a GBNF
    grammar, so the request fails with a reason instead of silently returning free-form text a
    caller could not distinguish from a model answering badly. Engine *hints* (`id_slot`,
    `chat_template_kwargs`, `thinking_token_budget`) are still dropped in silence — they are ours,
    not the caller's.
  - **`BackendClientPool.probe_json`** — a generic probe, so a provider owns its route and its
    parsing. New providers use it rather than growing another `probe_<engine>_<thing>`.
  - Remote capacity is **not** modelled as local capacity: nothing reports slots, and a remote
    endpoint keeps a config-seeded concurrency cap. Spill under one admission decision is
    Workstream D.
- **`roadstead/models.yaml` is a worked example that cannot rot** — it is both the shipped default
  and what the suite runs against, so a schema change that breaks it fails the build rather than a
  README snippet quietly going stale.
- **`roadstead.testing` speaks a remote wire shape** — `FakeBackend(engine="openrouter")` serves its
  routes off a base path, 401s without a bearer token, and publishes a two-entry catalogue with
  `context_length` and per-token pricing. `FakeBackendServer.base_url` is what an endpoint points at.

- **The catalog is provider-shaped** (`models.yaml`): `providers:` declares how to reach a backend
  and how to speak to it; `endpoints:` declares a routable unit of capacity with its policy. One
  local provider hosts one endpoint; one remote provider hosts many. `status: planned` documents an
  endpoint's shape without routing to it, which is how the example can show a remote provider's 1:N
  arrangement without putting an endpoint nobody has a credential for into the routing table.

- **A stated mission** (`docs/roadmap.md`, and the README lead): Roadstead is a **local-first LLM
  scheduler** that stands between many kinds of caller and many kinds of model and absorbs the
  mismatch, so neither side has to model the other. Positioning note: *scheduler*, not
  "orchestrator" — that word means agent/chain frameworks in this field, and Roadstead runs no
  workflows.
- **`docs/roadmap.md`** — what Roadstead is being built into: modular providers (llama.cpp and vLLM
  local, OpenRouter and others remote), remote capacity as spill under a single admission decision,
  an enriched API beside the OpenAI one, caller-intent model abstraction, API-key identity, and
  cost/token thresholds that degrade rather than reject.
- **`roadstead.testing`** — the programmable fake backend is now shipped API, not test scaffolding.
  A real ASGI app on a real socket, both engine wire shapes, and ~20 south-face pathologies on
  demand including `capacity_desync`. Was `tests/fake_backend.py`.
- **`docs/api.md` §1.3 and §1.4** — the keepalive ordering invariant and the `/v1/timeout-advice`
  contract, neither previously documented. §1.4 exists because a client had to *mirror* a floor
  table by hand, the endpoint it should have read it from being unspecified.
- **`docs/api.md` §3.1** — `/v1/fleet/*` response schemas to column level.
- **`Dockerfile`**, carrying `org.roadstead.required-stop-grace-period-seconds=90` as a label, so
  the shutdown requirement is discoverable from the image rather than only from a document.
- **`tests/wire_fidelity/`** — one south-face contract, run against both the fake backend and a real
  `llama-server`.
- **`tools/sigterm_drain_probe.py`**, **`tools/docker_stop_probe/`** — the shutdown measurements,
  kept re-runnable rather than merely cited.

### Fixed

- **`ProxyConfig` no longer shares its `EndpointConfig` objects** with the module-global
  `DEFAULT_ENDPOINTS`. It was a shallow dict copy around the same mutable objects, and those are
  mutated at runtime — capacity discovery writes `max_slots`/`context_per_slot`, the poller writes
  `served_model_id` — so one service's discovery reached into another's config. Invisible in
  production, where there is one; in the suite one test's mutation silently governed every test
  after it.
- **`GET /v1/models` advertised a name derived from a catalog API that had been renamed**, and would
  have raised on every call. It had no test in this repo; it does now. It advertises the endpoint
  CLASS — the name a client pins and we agree to keep answering to — never the `role`.
- **`probe_prefix_cache` is stubbed in the unit suite.** It is called from the poller every
  cache-stats tick, and was only *appearing* harmless because the old catalog's addresses were on a
  LAN that answered or refused quickly. Against unroutable documentation addresses every tick paid a
  full 5s connect timeout on the event loop and the poller stopped cycling.

### Changed

- **The structured-output corpus keeps every fixture and loses the vocabulary around them** (scrub
  item **S4**). `tests/corpus/schemas.py`'s four structured cases and five chat-loop cases stay —
  each pins a property no other one does, and they came from prompts that actually ran, which is
  what an invented fixture can never be. What went: case names that were a private deployment's
  agent names, `source=` fields that were `file:line` pointers into a monorepo that resolves nowhere
  here, and `"model": "orchestrator-reasoner"` (both that deployment's vocabulary and a word
  `CLAUDE.md` rejects — the chat cases now name a catalog endpoint class). No schema, grammar,
  output shape or message ordering moved.
- **🚨 And the personal identifiers inside those fixtures, which the scrub plan had ruled out.** Its
  headline finding was that no replay corpus of real traffic came across — true, and checked — from
  which it concluded there was no personal data in the repo. That does not follow: the synthesized
  example prompts were written around whatever was to hand, which included a real full name, a
  household member, a home town, a named local dental practice, and fabricated notices attributed to
  a real utility and a real bank. All fictional now, and the module says so at the top. The
  straggler sweep in `docs/corpus_and_scrub_plan.md` grew a second pattern, because a grep for the
  thing you imported cannot find the thing somebody typed.
- **`usage_rates.py` is anchored to model CLASSES, not to one fleet's models.** It was the second
  hardware inventory in the tree — model names, cutover narratives, host-prefixed unit names. A
  remote endpoint maps to `None` (unmetered) rather than to a rate: pricing spill from an
  avoided-cost table would credit the fleet with saving money it is in fact spending.
- `build_app` uses `lifespan=` instead of `on_startup=`/`on_shutdown=`, lifting the load-bearing
  `starlette<1.0` pin. Verified on 0.52.1 and 1.6.0.
- `roadstead.testing`'s `/props` now defaults to the **verified** narrow llama.cpp shape rather than
  a superset no real engine emits. The old shape is `props_profile="legacy"`. `/v1/models` is now
  per-engine. See `docs/ledger.md` — the superset was hiding the fact that the `total_slots`
  fallback, which real capacity discovery entirely depends on, was never exercised by any test.
- `roadstead.testing`'s usage sentinels gained public names (`USAGE_DEFAULT`, `OMIT_USAGE`); the
  underscored originals remain as aliases.

### Removed

- **`docs/handoff.md`** — retired to `docs/history.md` as a closed record. The extraction handoff is
  complete; the forward-looking half became `docs/roadmap.md`.
- **Golden-oracle parity against the origin monorepo, and the cutover it existed to make safe.**
  Roadstead is an independent project heading for a superset, and a parity gate on a superset fails
  on every improvement. `docs/compatibility.md` replaces it.
