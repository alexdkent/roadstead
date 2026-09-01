# Changelog

Notable changes to Roadstead. Breaking changes get their own section with a reason, per
`docs/compatibility.md` — the stable surface is the wire contract in `docs/api.md`; everything else
is internal and changes without an entry.

Pre-1.0: breaks are permitted, but each one is a recorded decision rather than a surprise.

## Unreleased

### Breaking

All four landed together on 2026-08-31 with the catalog redesign, and none touches the wire contract
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
