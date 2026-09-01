# Roadstead — Claude Code guide

> A **roadstead** is the sheltered anchorage outside a harbour where vessels wait for a berth to
> free up. That is not a metaphor for the admission queue — it *is* the queue, and specifically one
> that exists because capacity is finite and currently occupied.

**What this is:** a **local-first LLM scheduler** — it stands between many kinds of caller and many
kinds of model and absorbs the mismatch, so neither side has to model the other. Today that means
several llama.cpp servers and a vLLM pair across heterogeneous hardware, without Kubernetes;
OpenAI-compatible on the front, model-authoritative on the back. Mission in `docs/roadmap.md`.

Call it a *scheduler*, not an "orchestrator" — that word means agent/chain frameworks here, and
Roadstead runs no workflows. "Gateway" is fine in technical prose.

**Where it is going — read `docs/roadmap.md` before planning anything.** Roadstead is an independent
project with two goals: a high-quality local capability (primary) and an open-source project others
can use (secondary). It is growing a provider abstraction (llama.cpp + vLLM local, OpenRouter and
others remote, with remote capacity as *spill* under one admission decision), an enriched API
alongside the OpenAI one, caller-intent model abstraction, API-key identity, and cost/token
governance whose thresholds degrade rather than reject. Much of what follows describes the code as
it is *today*, which is still shaped by one private fleet.

**The thesis in one sentence:** the proxy empirically models its backends' capacity and latency
behaviour, and uses that model for both admission control and deadline setting.

Four things make it different from every gateway surveyed in `docs/evaluation.md`:

1. **DRR fair-share denominated in slot-seconds** — deficit round-robin across *callers* within
   three strict priority bands, where the unit of fairness is *backend occupancy time*, EWMA-
   calibrated and retroactively corrected. Most gateways fair-share over request count or tokens.
2. **Runtime capacity discovery** — it probes llama.cpp `/props` for real slot counts and per-slot
   context, rather than being told its capacity.
3. **Intent-declared timeouts** — callers declare priority and interactivity; the proxy computes the
   number from a learned latency distribution.
4. **A response-correction layer** — grammar repair, schema backstop, degeneration re-dispatch,
   truncation integrity, malformed-SSE repair.

---

## Provenance — and why the authority rule is GONE

This code was extracted on 2026-08-31 from a private monorepo (`OriginFleet`), where it still runs
in production as `originfleet.llmproxy`. The history here is the real thing — 282 commits at
extraction, going back to `9b11729` (2026-05-27, *"centralized LLM scheduler proxy — DRR scheduling,
priority bands"*), extracted with `git filter-repo` rather than copied, so `git log`/`git blame` on
any line still reaches its original rationale. **Use that.** It is the best documentation this project has.

🚨 **For a few hours on 2026-08-31 this file said the monorepo copy was AUTHORITATIVE FOR BEHAVIOUR
and told you not to fix behaviour here. That rule is retired.** If you are reading a cached summary,
an old branch, or a stale sibling document that still says it, ignore it.

**Roadstead is an independent project.** It is not a 1:1 replacement for the in-situ copy and is not
trying to become one — it will rapidly become a superset, and it may occasionally break exact
backward compatibility on purpose. There is no parity gate, no cutover to plan, and no re-sync
obligation in either direction.

What that changes, concretely:

- **Fix behaviour here.** Bugs get fixed in this repo. No round-trip.
- **But look before you fix.** The reason the old rule existed is still half-true: a surprising
  amount of this code is a deliberate workaround for measured engine behaviour, not an oversight.
  The findings below and `docs/ledger.md` exist so you can tell the difference. If a thing looks
  wrong and has no recorded reason, it is probably wrong — fix it.
- **The monorepo is evidence, not authority.** It serves real traffic; this repo serves a fake
  backend. When it reports something (a defect rate over 5,460 real responses, a live `/props`
  shape), that is data we cannot generate here and it is worth having. It carries no obligation and
  gates nothing.
- **Breaking changes are allowed, and must be recorded.** See `docs/compatibility.md`: the wire
  contract in `docs/api.md` is the stable surface, everything else is internal, and every break goes
  in `CHANGELOG.md` with its reason. "Occasionally, deliberately, written down" — not "freely".

**What does NOT apply here.** This repo has no access to and no dependency on: the fleet's agents,
`ship.sh`, `cexec`, the `.claude/skills/` layer, the monorepo's regression ledger, or any fleet host.
If a comment or docstring references one of those, it is a leftover from extraction — the reference
is dead, but the *reasoning* it points at is usually still valid. Don't delete the reasoning.

---

## 🚨 The concurrency invariant — now guarded, and still the most dangerous thing here

**Guarded as of 2026-09-01** (Workstream F), after being unguarded for the whole of this repo's
life. `tests/loop_affinity.py` arms it — it wraps the mutating methods of every single-loop object
and records the thread — and `tests/e2e/test_soak.py` runs sustained overlapping load with it armed,
asserting thread affinity, slot conservation, DRR budget conservation, ledger conservation, and that
every request was answered. One test in that file mutates DRR budget state from a second thread **on
purpose**, so the guard is observed going red from inside the suite forever rather than once.
`tools/soak.py` is the unbounded version, for what accumulates over minutes.

🚨 **A guard is not a licence.** The rules below are unchanged, and the soak catches a violation only
on a path the load actually reaches — which is most of them, but not the one you are about to add.

- **Single event loop. No locks** on in-memory scheduler / budget / cache state. That is only safe
  because there is exactly one thread mutating it.
- **Exactly ONE sanctioned background writer thread** — the `queue.py` DB writer, which owns its
  write connection.
- The loop thread uses its **own read connection**. Heavy dashboard aggregations run **off-loop via
  `asyncio.to_thread`**, and each pool thread gets its **own read-only connection** (thread-local;
  WAL permits concurrent readers) so a slow aggregation never blocks the loop.
- **Never add a `workers=` parameter, a thread pool that WRITES, or a second thread that touches
  scheduler or budget state.** Connections are never shared across threads.
- 🚨 **Workstream E added a new KIND of writer: a request handler.** Everything else that mutates
  single-loop state is written by the scheduler loop itself; `management.py` mutates the key
  registry, `config.agents` and the DRR budgets from an admin request. The rule it follows is
  **mutate on the loop, persist off it** — the file write goes through `asyncio.to_thread`, the
  mutation does not. That is deliberately stricter than `flags.py`, which runs its whole `set_many`
  off-loop: a flag dict is written in a blue moon, whereas a registry write racing `resolve()` on
  every request is the exact interleaving this invariant forbids. `loop_affinity.py` arms all three
  objects (and grew dotted-path resolution to reach the registry behind the identity resolver).

Shut down with **SIGTERM**, which runs a bounded drain (≤30s) that finishes in-flight work, flushes
the write queue, and persists DRR budgets. The monorepo's knowledge layer contradicted itself here;
**settled by experiment on 2026-08-31** (`tools/sigterm_drain_probe.py`, full result in
`docs/ledger.md`). Both halves were right about different things: SIGTERM is correct — the drain
persists the DRR budget row and the completion row even for a straggler it cancels, which is exactly
what SIGKILL loses — *and* it really can hang, for longer than the code's own comment implies.

🚨 **The two shutdown budgets are serial, not nested.** uvicorn's `timeout_graceful_shutdown` bounds
the in-flight HTTP *connections*; only when it expires does uvicorn send `lifespan.shutdown`, and
only then does `ProxyService.shutdown`'s drain begin. uvicorn never bounds the lifespan shutdown at
all, so the worst case is their **sum**.

🚨 **Every phase is bounded and the ceiling is PUBLISHED, as of 2026-09-01.** It was not before: the
drain was bounded and the tail after it was not, and `OnDemandManager.close` makes a network call per
held GPU lease — so a wedged dispatcher hung shutdown indefinitely. `service.py` names each phase
(drain 30s, straggler unwind 3s, lease-release + pool-close 5s concurrently, queue flush+join 10s),
sums them into `SHUTDOWN_DEADLINE_S`, and computes `RECOMMENDED_STOP_GRACE_S` from that plus
uvicorn's own budget plus margin. 🚨 **There is deliberately no outer `wait_for` around `shutdown()`**
— it would cancel `queue_db.close()` mid-flush, which is the SIGKILL failure the drain exists to
avoid — so the ceiling is true by arithmetic and `tests/test_shutdown_budget.py` pins the arithmetic.

🚨 **Any container stop-grace-period must be ≥108s** (it was 90 until 2026-09-01, and it went **up**
because 90 was a *measurement* of a fast-but-unbounded tail rather than a bound). Measured in a real
container (`tools/docker_stop_probe/`): at `docker stop`'s **default 10s** with work in flight, the
proxy is **SIGKILLed with zero budget and zero completion rows persisted** — the shutdown handler
never runs at all. The default *works while the proxy is quiet*, which is how it will be tested and
why it would first fail under load. The number is computed in one place and the `Dockerfile` label,
the prose and the runtime log all read it.

---

## Engine-behaviour findings — why the code is shaped this way

Every item below was measured against real backends. Each one looks like a wart until you know the
reason. **Do not simplify these away.**

**Capacity discovery is asymmetric, deliberately.** llama.cpp `/props` yields real
`n_parallel`/`total_slots` and per-slot `n_ctx`. vLLM exposes only `max_model_len`, so vLLM
concurrency stays **config-seeded** with a drift alert. This is a property of the engines, not an
oversight — and it is now *declared*, in each provider's `ProviderDescriptor`
(`publishes_slot_count` / `publishes_slot_context` / `publishes_context_ceiling`) rather than left
for a reader to infer from a branch.

**`finish_reason` rides alone on a terminal SSE chunk whose `delta` is `{}`.** It carries no text, so
losing it is invisible if you only inspect content. The proxy synthesizes a missing terminal chunk —
but **ONLY when the backend sent its own `[DONE]`**, i.e. asserted completeness and merely failed to
label it. No `[DONE]` and no `finish_reason` is a REAL truncation: nothing is synthesized. 🚨 **Never
collapse those two cases** — the repair would become a silencer. A downstream client once burned a
second model call on 47 turns "continuing" answers that were already complete.

**The prompt-cache reuse floor is caused by `-ub`, not by a missing flag.** `--cache-reuse` is
*architecturally refused* by llama.cpp on two classes of backend — multimodal (`mctx != nullptr`) and
non-shiftable attention memory — with two different loader messages. Adding the flag does nothing.
Lowering `-ub` is a real trade (measured −27% prefill), not a free win.

**A flat transport read-timeout silently capped every caller deadline at 600s** and surfaced as
`BackendError(502)` — so a deadline the caller set looked like a broken backend. The transport
timeout now tracks the request deadline so `asyncio.wait_for` is always the layer that fires and the
failure is typed as a TIMEOUT.

**vLLM 400s the entire request if `thinking_token_budget` is sent without `--reasoning-config`.**
`--reasoning-parser` alone is NOT enough. The budget is therefore a *declaration of a launch flag*,
not a tuning knob.

**The thinking switch is spelled differently per model family** (`thinking` vs `enable_thinking`) and
is declared in the catalog, never hardcoded. Getting it wrong makes thinking a silent no-op.

**Excessive reasoning is a symptom of an over-constrained prompt, not a slow model.** Given room, the
same call used *fewer* tokens and less than half the wall clock than when capped. A too-small budget
costs quality AND latency. Above a hard ceiling no `max_tokens` fixes it — the fix is the prompt.

**The vLLM `qwen3_xml` parallel-tool-call defects are streaming-delta artifacts ONLY** — 5,460
non-streaming responses over 7 days, zero defects. The stream sanitizer is the complete fix. **Do not
add a response-path sanitizer without new evidence.**

**Degeneration re-dispatch is deliberately SKIPPED when the original hit its output cap**
(`finish_reason=length`) — a capped loop just re-caps.

**Structured-output reason-injection was built, measured, and REMOVED.** A blind cross-model judge
preferred the baseline in 35/58 discordant pairs and conformance fell below gate. Don't rebuild it
without new, backend-correct A/B evidence.

---

## Layout

```
roadstead/          the package (37 modules + providers/ + client/)
  scheduler.py      DRR + priority bands + admission        — pure computation, no I/O
  cost_model.py     slot-second cost, EWMA-calibrated       — pure computation, no I/O
  timeout_model.py  learned latency → recommended deadline  — pure computation, no I/O
  spend.py          prices, per-caller spend, thresholds    — pure computation, no I/O
  rate.py           per-caller request rate, same shape     — pure computation, no I/O
  intent.py         a declared capability → an endpoint     — pure computation, no I/O
  correction.py     the output-integrity layer
  lifecycle.py      admission → dispatch → streaming → timeout recording
  enriched.py       north face TWO: /rs/v1 (see below)
  management.py     north face THREE: /rs/v1/admin — the operator's plane
  ui/index.html     the operator's FACE — one static file, no toolchain (see below)
  http_handlers.py  north face ONE: the OpenAI doors, admin, analytics
  health.py         capacity discovery, circuit breaker, drain
  queue.py          durable event log + THE single writer thread
  identity.py       who is calling: API keys first, acl.py as second factor
  backend.py        south face TRANSPORT: pools, deadlines, error taxonomy, SSE relay
  providers/        south face ENGINES: llama.cpp | vllm | openrouter (see below)
  model_catalog.py  reads models.yaml — providers + endpoints (see below)
  hooks.py          the integration seam (see below)
  client/           SHIPPED client SDK for /rs/v1 — imports NOTHING from the server
  testing/          SHIPPED test doubles — the programmable fake backend
  __main__.py       entrypoint
tests/              the suite + corpus/ + loop_affinity.py (arms the invariant above)
tools/              off-default-path experiments (real processes, real signals, soak)
docs/               specs, plan, evaluation, ledger
```

The six `pure computation, no I/O` modules are the crown jewels and the easiest to test — keep
them that way.

**`models.yaml` has two sections, and the split is load-bearing.** `providers:` is *how to reach a
backend and how to speak to it* — engine, address, credential. `endpoints:` is *a routable unit of
capacity with policy* — slots, context, floors, capabilities, failover — each naming its provider.
A local provider hosts ONE endpoint (a llama.cpp or vLLM server serves one model); a remote provider
hosts MANY, and declaring its base URL and key once is why the sections are separate.

🚨 **The shipped catalog is an EXAMPLE** — `tier1`/`tier2`/`tier3`/`embed`/`rerank` plus two
`planned` remote endpoints, on RFC 5737 documentation addresses. It is what the suite runs against,
so it is a worked example that cannot rot. Point `ROADSTEAD_MODELS_YAML` at your own file.
**Comments throughout this package cite measurements taken on a real fleet under ITS names**
(`gemma`, `creative`, `tier2-chat`, `llama-thinker`, specific model and box names). Those are
records of what was measured — do not "fix" them to match the example, and do not read them as
references to classes that exist here.

**`providers/` is where engine differences live, and nowhere else.** A provider owns the two things
backends genuinely disagree about: what a request must look like to be accepted
(`prepare_chat_payload`, `path_for`) and what the backend will tell us about itself
(`discover_capacity`, and the `ProviderDescriptor`). Everything shared — connection pools, the
deadline, the error taxonomy, the SSE relay — stays in `backend.py`, and providers borrow its
probe methods rather than opening sockets of their own (which is also what keeps the unit suite off
the network: it stubs `probe_*` by name on the pool).

**Three providers today: `llama.cpp`, `vllm`, `openrouter`.** The last one is the reason the interface
exists — it is reached at a `base_url` with a base path instead of `host:port`, it needs a credential
(`api_key_env` names the environment variable; **never the key itself**), it fronts a *catalogue* so
there is no served model to discover, and it publishes prices instead of occupancy. 🚨 **Remote
capacity is not local capacity.** A remote endpoint still carries a config-seeded concurrency cap —
a policy knob we choose, not a discovered capacity — because slot-seconds are the unit of *local*
fairness. Making remote capacity an outcome of the same admission decision (dispatch / spill / defer)
is Workstream D.

🚨 **A provider that cannot honour a CONSTRAINT refuses; one that cannot use a HINT drops it.**
OpenRouter cannot enforce a GBNF grammar, so `prepare_chat_payload` raises `UnsupportedRequest` and
the call fails with a reason. Dropping the grammar would hand the caller free-form text it could not
distinguish from a model answering badly — the same shape as the `finish_reason` repair that became a
silencer. An engine hint (`id_slot`, `chat_template_kwargs`, `thinking_token_budget`) is ours, not the
caller's, and is dropped in silence.

🚨 **Branch on a descriptor capability, never on an engine name.** `backend_engine == "vllm"` used
to appear at four sites and each one meant something narrower — "publishes prefix-cache counters",
"mislabels a truncated tool call", "reasoning can be switched off", "has no `/props`". A third
engine would have had to be added at every site by hand, and a missed one fails silently in one
direction. `tests/test_provider_interface.py` fails if such a comparison reappears outside
`config.py` / `model_catalog.py`, where the engine name is a config value rather than a decision.
Providers are **stateless singletons** shared across every endpoint on the single loop — per-request
state on one is a data race no test here would catch. An unknown engine string resolves to
llama.cpp, deliberately: that is what `!= "vllm"` always did.

🚨 **TWO north faces, and enrichment never crosses between them.** `/v1/*` is OpenAI-compatible and
strictly so; `/rs/v1/*` (`enriched.py`) is Roadstead's own, with its own version because the other
one is versioned by OpenAI. A client that validates against OpenAI's schema must not break because
it pointed here, and *"we only added fields"* is not a defence — strict validators reject unknown
keys. So the OpenAI body is byte-identical and its enrichment rides in four `X-Roadstead-*` response
headers, which carry only what admission had already settled: a streaming response's headers are on
the wire before a failover the enriched `done` frame can still report. **`POST /v1/submit` is gone**
(`docs/api.md` §1.9 maps it field by field).

🚨 **Resolution is not substitution, and `intent.py` is where that line is drawn.** A caller declares
a capability and Roadstead owns the choice of model; a `model` is a *pin* and is a constraint on
routing, refused rather than quietly served from next door when it cannot be met. An intent landing
on an endpoint the caller never named is the job being done — reporting it as a substitution would
make `substituted: true` fire on every intent-routed call and mean nothing. Only a *later* move,
failover or spill, is disclosed as one. Two ranking rules are doctrine and each has a test observed
going red: **a real-cost endpoint sorts last under every preference** (otherwise "intent" is a back
door around the whole spill doctrine and traffic leaves the machine on the ordinary path), and **an
endpoint with no latency samples sorts as SLOW** (the obvious ascending sort prefers the backend we
know least about *because* we know least about it). Profiles are expressed in declared capabilities,
never endpoint names — one that named classes would be a third routing table to keep in step with
`models.yaml`.

🚨 **The intent VOCABULARY is configurable; the routing table is not.** `models.yaml` has an
`intents:` section, layered over the nine built-ins rather than replacing them — a file defining one
profile has said nothing about the other nine — and an override is disclosed through each published
profile's `source` (`builtin` | `models.yaml`), because a caller reading our docs for a word this
fleet redefined has no other way to notice. 🚨 **A profile still cannot name an endpoint**, and now
there is a config parser that must not learn how: guarded from both ends, the allowlist
(`model_catalog._PROFILE_FIELDS`) and `Profile`'s own fields. An **unusable** stanza — unknown
capability, kind or preference — is REFUSED rather than offered, because a profile that matches
nothing sends the caller "no endpoint satisfies requires=[…]", a sentence about the fleet for a fault
in a config file; refused, they get "unknown intent", which points at the vocabulary. A refused
*override* does not leave the built-in standing under the operator's spelling.

🚨 **`exclude` is the caller's negative constraint, and the line it respects is CONFIG vs REQUEST —
not positive vs negative.** A profile is shared, published, operator-written vocabulary and stays in
capabilities; an intent is one caller's words about one call, where `pin` already names an endpoint.
So `exclude` names endpoints and adds no table. 🚨 **An `exclude` naming an endpoint this fleet does
not have is a 404, never a warning.** "Satisfied trivially, the endpoint it forbids is absent"
assumes the thing we cannot check: a name resolving to nothing is either "not here" or "here, under a
spelling you got wrong", identical from inside — and serving the second routes to precisely the
endpoint the exclusion existed to avoid, reporting success. Never collapse "we checked" with "we
could not tell". `model` and `exclude` naming the same endpoint is a 400.

🚨 **A request may DECLINE a substitution; it may never grant itself one.** `substitution: {degrade,
spill}` narrows what the operator granted, and both gates take the **AND** — an `or` there would let
a caller award itself a permission its operator withheld, which is the self-asserted `agent_id` bug
in a different costume, and for spill the consequence is money. Declining is a DEFER, not an error:
the request keeps its place and is served locally.

🚨 **The management plane answers ONE question: what did you write that is not in force?**
`management.py` (`/rs/v1/admin/*`) is the operator's face, and it is a diagnostic rather than a
readout — a view that echoed `models.yaml` back would be a worse `cat`. Every expensive failure in
`docs/ledger.md` lives in a gap between two sources that agree most of the time, so the views report
the **declared** value beside the one **in force**: the catalog's slot seed beside what discovery
left (and whether the engine publishes it at all, so "discovery agreed" and "discovery never ran"
stop being indistinguishable), a caller's quota as `in_force`/`declared`/`runtime`, and
`hooks.config_notice` — the fourth reporting seam and the first that reports *in* — retaining every
knob the three allowlist parsers dropped. It is on `/rs/v1/admin`, not `/v1/admin`, because `/v1` is
versioned by OpenAI; the four control routes that predate it are served at **both** spellings, same
handler, same gate.

🚨 **A management surface never emits a credential, and a control action that cannot be persisted
still takes effect.** Not the key, not the digest (a digest is a working credential to anyone who can
compute one), never the value behind an `api_key_env` — only its name and whether it resolved. And a
runtime edit **never rewrites the operator's config file**: changes go to a JSON overlay layered over
the files at startup (enrol, then revoke — a tombstone is a later statement than the enrolment it
follows). When the overlay is unwritable the change applies in memory and the response says
`persisted: false` with a reason, because refusing a revocation over a read-only disk is a
correctness argument answered, in the moment, by a breach. Same reason revocation is never refused on
*provenance* grounds — an env-declared key can be killed now, and the response says the declaration
will outlive the reason it is dead. 🚨 **Revoking the LAST key changes the identity regime back**
(§1.5 rule 2: an empty registry is not in play, so the revoked key is *ignored* rather than refused),
and that is disclosed rather than fixed — narrowing rule 2 would 401 exactly the deployment that just
emptied its registry on purpose. Disclosures ride in a `warnings` **array**: two can be true at once,
and a single field means the second silently overwrites the first.

🚨 **The write boundary INHERITS §1.6 rather than re-implementing it, and keys stay FLAT.** No
editable field can express a rejection — every quota knob changes a share, a band or a cap — so the
plane cannot mint a policy the admission path refuses to honour; an unknown field is a 400 that names
the known set, never a silent drop, on the surface whose whole purpose is exposing silent drops. A
quota edit reaches the **live** DRR budget (or it applies only to callers the proxy has never seen,
which reads as "the edit did nothing" for exactly the busy caller it was aimed at) and moves the
**rate, not the balance**. And keys are flat because the budget holder is the `agent_id`, not the
key: many keys → one `agent_id` is already team-level quota inheritance, which is what the roadmap's
"multi-tenancy depth" question was asking for.

🚨 **`admin_readonly` NARROWS `admin`, and a narrowing is never cancellable by a wider grant.**
A credential with `admin: true, admin_readonly: true` reaches every `GET` on the management plane and
is refused a 403 that says why on everything else; on a key without `admin` it is a **400 at
enrolment**, because an operator who wrote it believes they issued a safer credential than they have.
🚨 **The read/write split comes from the HTTP METHOD**, in the one shared gate — a list of write
routes is a second thing to keep in step with `routes.py`, and when it falls behind the failure is
silent and *widening*. All 11 mutating admin routes inherit it, the four control routes that predate
the plane included. It is spelled `:readonly` in the shared identity grammar, so `ROADSTEAD_API_KEYS`
and `ROADSTEAD_ACL` express it too — and `127.0.0.1=ops:admin:readonly` is read-only **even though
loopback is a built-in admin net**. If the widest grant won there, that line would silently be a full
grant for every operator who wrote it. `identity.py` decides and everything else renders:
`IdentityResolver.admin_denial` owns which of the three refusals applies, and an AST guard fails if
`may_admin_write` or `admin_readonly` is read anywhere else — a second place deciding what a scope
permits is the `_remote_ip` shape again.

🚨 **`GET /rs/v1/admin/audit` says who changed what, and EVERY mutating admin route records.**
Flags, pause, resume and maintenance touch no overlay state and would otherwise record nothing; a
trail covering only some of them is worse than none, because a reader assumes completeness. The
completeness guard is driven from `routes.py`, not from a list. **A record names the credential and
never carries one** — `key_id`, never the key or the digest — and it records the key label *and* the
address always: a record with an address and no `key_id` means an address-derived admin made the
change, which is meaningful and slightly alarming, and collapsing the two into one actor string would
hide which factor authorized it. It **reports its own limits as data** (`persisted` — in-memory is
the default — `dropped`, `capacity`), because it is an operator-facing change trail rather than a
security log of record, and presenting itself as complete while being neither durable nor unbounded
is the gap this repo keeps paying for. Written on the loop, persisted off it.

🚨 **The management UI is ONE static file, and the credential is still a key.** `roadstead/ui/index.html`
(`GET /rs/v1/admin/ui`, registered only when `ROADSTEAD_ADMIN_UI` is set — off means the route does
not exist) is vanilla JS with no bundler and no external reference of any kind; it ships in the wheel,
so it is public surface on the same argument as `roadstead.testing`, and the CSP forbids an external
reference rather than trusting a reviewer to spot one. Auth is **HTTP Basic carrying an API key in
the PASSWORD half** — a browser cannot attach a bearer token to a navigation and `EventSource` cannot
set a header at all, and minting a password would be a second credential kind with its own store,
rotation and revocation beside a registry that already does all three. No cookie is minted, so CSRF
never becomes reachable. The door refuses **401 + `WWW-Authenticate`** where the plane answers 403,
because a 403 gives a browser no way to answer it. 🚨 **Every field the page reads goes through
`pick(obj, "a.b.c")`** so the paths are extractable, and `tests/test_admin_ui.py` walks each one
against a real response — a UI has no compiler and no schema, so a renamed field renders "—" forever
in one cell while the page looks healthy. 163 paths and 14 `guarded(...)` write controls as of
2026-09-01.

🚨 **There is deliberately NO browser-driven test, and the usual reason for that is wrong.** The
"dependency list is six" rule is about the **runtime** list (the test reads `project.dependencies`);
a browser driver is a *dev* dependency and would violate none of it. What rules one out is the
suite's promise above — install and run, **no network** — which a downloaded browser binary breaks
for every contributor, to cover one page. The accepted gap is DOM-level faults in the render helpers,
and the compensating discipline is: **when rendering the page finds something, leave behind a guard a
source read can make.** `docs/roadmap.md` under G carries the decision and what would reopen it.

🚨 **`roadstead.client` imports nothing from the server, and that is a rule with a test.** Two
reasons, and the second is the one that would be lost silently: a consumer sending an HTTP request
should not be installing Starlette, uvicorn, PyYAML and jsonschema; and a client that read the
server's own constants would agree with it *by construction* and could never catch a drift. Its
contract literals are transcribed from `docs/api.md` and checked against the document — the same
two-ended pin `tests/wire_contract.py` uses from the emitting side. It also classifies errors on the
`code` with §2.2's marker substrings as a fallback, which is the migration §2.2 asked for.

🚨 **Identity is a credential first and an address second, and the precedence is doctrine.**
`identity.py` resolves every request to a `Principal` — `agent_id` (the DRR fair-share key, quota
holder, budget holder), a default priority, an optional deadline floor, an optional admin scope —
and it is the ONLY place that knows the order. Three rules, each of which looks arbitrary until it
bites, all in `docs/api.md` §1.5:

- **A presented key that does not resolve is a 401 and never falls back to the address.** Falling
  back means a wrong or revoked credential silently becomes a *different, weaker* identity that
  still works — the same shape as the `finish_reason` repair that became a silencer.
- **With no keys configured the registry is not in play**, so a presented key is ignored and the
  address decides. Every OpenAI client sends an `Authorization` header whether anybody meant it to
  or not; treating one as significant before an operator configured any key would 401 the world.
- **A key overrides a body-declared `agent_id`; an address only fills in one the body omitted.** And
  an authenticated non-admin identity does NOT inherit its host's admin privileges — a scoped key
  that can only widen access and never narrow it is worthless on the machine it runs on.

- **A forwarded address is believed only from a trusted proxy, and the caller is the RIGHTMOST hop
  that is not one.** `ROADSTEAD_TRUSTED_PROXIES` is empty by default, so `X-Forwarded-For` is not
  consulted at all until an operator opts in. The leftmost element — the intuitive reading — is
  precisely the part the caller wrote before any proxy appended what it observed. 🚨 And a forwarded
  address does **not** inherit the BUILT-IN admin nets: loopback and docker-internal are auto-granted
  admin because reaching them meant already being on the box, and a front proxy is exactly what makes
  that untrue. `ROADSTEAD_ADMIN_NETS` and an `admin` key are unaffected — what the operator said
  stands, what was inherited does not. An unparseable hop or a chain over 32 long resolves to
  `unknown` (not an address, matches nothing, refused) rather than back to the peer, because falling
  back to the peer hands the proxy's grants to whoever sent the header.

`identity.py` is the ONLY place an address is resolved, and that is an **AST** guard, not a substring
one — `management.py` had grown its own `_remote_ip`, and a sweep for `client.host` is walked past by
`getattr(getattr(request, "client", None), "host", "")`, which is the same bug.

🚨 **A key has a LIFE, a SUCCESSOR and a PLACE, and every one of the three may only narrow.**
`created_at` was written, surfaced in two views and read for no decision until 2026-09-01.

- **Expiry** — `expires_at` (an absolute instant in a keys file) / `expires_in_s` (a duration over
  the API). No expiry means never expires, so an existing registry behaves exactly as it did. An
  expired key is a `401 invalid_api_key`: **its own sentence, the same code**, because a caller can
  act on no distinction between "expired" and "unknown" while the operator reading the log can — one
  means check what you pasted, the other means issue a successor. The store holds the absolute
  instant, never the duration it came from, so a restart cannot extend a key.
- **Rotation** — `POST /rs/v1/admin/keys/{key_id}/rotate`, ONE action, because the manual version is
  two calls in an order that matters and **both orders are wrong**: enrol-then-revoke leaves the
  successor live and unknown to the caller, revoke-then-enrol leaves nothing working. The successor
  inherits the predecessor's policy but **not its expiry** (inheriting an absolute instant would mint
  a successor that expired seconds later) — right, and therefore *disclosed* rather than changed.
  `overlap_s` defaults to 0; with an overlap the predecessor gets an **expiry** rather than a
  tombstone, so it survives a restart a timer would not. 🚨 An overlap may **shorten** a
  predecessor's life and never lengthen it — a day-long overlap on a two-hour key pushed its expiry a
  day out, which is a rotation quietly widening a credential. If the successor cannot be minted the
  predecessor is untouched.
- **Binding** — per-key `bind` CIDRs are an ADDITIONAL constraint, never a way for a key to widen
  what an address grants. An unparseable entry is a 400 at enrolment and matches nothing at
  resolution: a narrowing that failed open would be worse than none. 🚨 It is checked against the
  *resolved* address, so a binding is only as trustworthy as `ROADSTEAD_TRUSTED_PROXIES` — and
  `GET /rs/v1/admin/keys` reports `checked_against` rather than leaving an operator to infer it.

`/v1/submit` is gated like the OpenAI doors as of 2026-09-01. It was not, on the same port, which
meant the fair-share key was self-asserted by anyone who used that door.

🚨 **Money: two kinds of it, and adding them together is the bug `spend.py` exists to prevent.**
`usage_rates.py` prices a LOCAL endpoint at what renting the same class of model would have cost —
a saving, never a bill. A remote provider's published price is an invoice. Both are USD per million
tokens and nothing in the type system separates them, so a `TokenPrice` carries which kind it is and
`SpendLedger` keeps `spent_usd` and `avoided_usd` in fields that are never summed. **A threshold
reads only the spent one** — one that counted avoided cost would throttle a caller for using
capacity that is free and already paid for, which is local-first inverted.

🚨 **Thresholds DEGRADE, they never reject, and there is NO ERROR CODE for one.** Crossing
`daily_spend_usd` costs a caller exactly two things: one priority band (floored at the lowest,
however far over it is) and access to paid spill. It never costs local capacity. Admission control is
about *capacity*, not billing, and a misconfigured quota must not be able to take a caller offline —
a runaway caller is already bounded by what is free and by DRR fairness. `docs/api.md` §1.6, and
`tests/test_spend.py` fails if a spend-shaped code ever appears in §2.1.

🚨 **There are TWO thresholds now, and degradations do NOT STACK.** `requests_per_minute` joins
`daily_spend_usd` and costs a caller exactly what that one costs it — **one priority band and paid
spill, never local capacity** — and mints no code either. DRR is fairness *under contention*, so a
caller alone on a quiet fleet is unthrottled by design; that is correct for fairness and exactly why
it does not bound a runaway, which is the gap this fills. A rate *limit* would be a fourth spelling
of "no" and would put a mistyped threshold in a position to take a caller offline. 🚨 A caller over
**both** thresholds drops **one** band, not two — `ProxyState.effective_priority` consults at most
one standing by construction, so a third threshold cannot compose by accident. Each crossing reports
through the degradation seam **separately**, because one means "look at the bill" and the other means
"look for a loop". `effective_priority` is split from `spend_demote` so a management *read* fires no
notice. `roadstead/rate.py` is the **sixth** pure-computation module.

🚨 **Admission is ONE decision with THREE outcomes** — `Scheduler._admit` returns
`DISPATCH` / `SPILL` / `DEFER`. **Local capacity is tried first, for everybody**: no test involving
money appears above that line, so an over-cap caller, an un-opted-in caller and a caller nobody
configured all reach the same local dispatch. Spill is considered only once local has said no, which
is what makes remote capacity *overflow* rather than a parallel system with its own fairness. It
never chains — a spilled request does not spill again, or two endpoints pointing at each other would
hand one request back and forth a hop per tick forever.

🚨 **"Does this fit in the context?" is asked in ONE place — `cost_model.context_fit`.** It was
written out by hand at four sites, and the fourth (the WAL-recovery shadow tally) had diverged and
was **dead**: it read `req.est_input_tokens or 0`, but that field is cached by `scheduler.enqueue`
and `recover_queued` builds its `QueuedRequest` straight from the WAL row, so the estimate was always
0 and the tally could only fire when `max_tokens` *alone* exceeded the ceiling. It disagreed with the
live predicate on 27 of 153 corpus cases. 🚨 **`context_fit` returns the answer and its arithmetic
and takes NO action** — the consequences stay deliberately different (admission is shadow-or-422
behind `context_gate_enforce`; failover refuses unconditionally, because there the alternative is a
guaranteed backend 400; spill defers; recovery counts and never rejects), and folding the consequence
in would have armed a flag nobody flipped. A test pins that. An AST guard fails if any module
re-derives a ceiling comparison from `estimate_input_tokens` instead of calling it, and a second
constant sweep allows exactly one spelling of `CONTEXT_OVERFLOW_MARKER` per side of the client
boundary.

🚨 **`spill_to` is not `failover_to`, and `spill_ok` is not `degrade_ok`.** Failover asks "this
backend is DOWN, may a smaller model answer" — a quality judgement, answered by whether the output is
retractable. Spill asks "this backend is BUSY, may we pay somebody else to answer now" — a
confidentiality-and-money judgement. The triggers differ (health vs occupancy), the failure modes
differ (a worse answer somebody can react to vs an invoice and a prompt on a third party's server),
and one endpoint can want both. Neither flag defaults from the other.

**`hooks.py` is the integration seam.** A host application can register a degradation sink; the
default is a WARNING log. Everything else in the package reports through it. Keep this module
dependency-free — an import of anything outside the stdlib re-creates the coupling it exists to
remove.

---

## Working here

- **Python 3.11** (the production container runs 3.11.15).
- Dependencies are deliberately few: `httpx`, `starlette`, `uvicorn`, `PyYAML`, `jsonschema`,
  `json_repair`.
- `pip install -e '.[dev]'` then `pytest`. No fleet, no network, no backends required — the suite
  runs entirely against `tests/fake_backend.py`.
- **`roadstead.testing` is the most reused asset in the repo, and it ships.** A Starlette app under
  real uvicorn on a real socket (so real `httpx` and real SSE framing are exercised, not a
  `MockTransport`), emulating both engine shapes and a library of south-face pathologies selectable
  per-request via an `X-Fault` header — including `capacity_desync`, which accepts N concurrent and
  503s beyond while `/props` lies about capacity. It was `tests/fake_backend.py` until 2026-08-31;
  promoted because for a gateway whose thesis is capacity-aware admission, a backend that lies about
  its capacity on demand is a capability, not furniture.

  🚨 **It is public surface now.** A new import in it is a new import for everyone who installs
  Roadstead — keep it to Starlette/uvicorn/httpx, which are already dependencies. Nothing in the
  library core imports it, so a production deployment never pays for it. `__all__` and the fault
  library are pinned against each other by `tests/test_testing_module_is_public.py`, which also
  proves the import works from outside the repo.

- **`roadstead.client` is the second shipped sub-package**, and its boundary is tighter: **httpx and
  the stdlib, and nothing from the server**, enforced by AST in `tests/test_client_sdk.py`. It is
  what another project installs to speak `/rs/v1`, and it must be importable in an environment that
  has none of the server's dependencies — a subprocess test blocks Starlette, uvicorn, PyYAML,
  jsonschema and json_repair and requires it to import anyway.

## 🚨 Before this repo goes public

It is **private** and must stay private until the scrub in `docs/corpus_and_scrub_plan.md` is done.

**S2 is done** (2026-08-31): `models.yaml` is a generic example on RFC 5737 addresses, and the
schema redesign that had to happen anyway went in with it. `usage_rates.py` lost its fleet model
anchors in the same pass.

**S1, S1b and S3 are done** (2026-09-01), with Workstream B — and that ordering was necessary, not
convenient: the ACL seeds could only be removed *in favour of* something, and the something is an
API key. The working tree carries no private address. It also turned up `agents.yaml` as a fourth
inventory nobody had listed (one fleet's agent roster, weekly volumes, infra paths — and the package
DEFAULT), now a generic example on the roadmap's caller archetypes.

**S4 is done** (2026-09-01). `tests/corpus/schemas.py` keeps all nine fixtures — each pins a
property no other one does — and lost every word of the vocabulary around them. It also turned up
the thing the scrub plan had explicitly ruled out: **personal identifiers**, in synthesized prompts
somebody wrote around whatever was to hand. No bulk corpus came across, which is what was checked;
that is not the same as no personal data, and the sweep in S5 now runs both patterns.

🚨 **What remains under fleet names is COMMENTS RECORDING MEASUREMENTS, and they stay.** Same rule as
`models.yaml`: those are records of what was measured, not references to anything that exists here.

🚨 **And scrubbing the working tree is not enough — it is in the history**, across all 312 commits
(282 of them extracted), which means another `git filter-repo` pass (S6, last, because it invalidates
every SHA). Read that plan before changing visibility.
