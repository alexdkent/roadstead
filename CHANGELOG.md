# Changelog

Notable changes to Roadstead. Breaking changes get their own section with a reason, per
`docs/compatibility.md` — the stable surface is the wire contract in `docs/api.md`; everything else
is internal and changes without an entry.

Pre-1.0: breaks are permitted, but each one is a recorded decision rather than a surprise.

## Unreleased

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
