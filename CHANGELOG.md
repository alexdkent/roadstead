# Changelog

Notable changes to Roadstead. Breaking changes get their own section with a reason, per
`docs/compatibility.md` — the stable surface is the wire contract in `docs/api.md`; everything else
is internal and changes without an entry.

Pre-1.0: breaks are permitted, but each one is a recorded decision rather than a surprise.

## Unreleased

### Breaking

*None yet.* The project became free to break compatibility on 2026-08-31 and has not used it.

**Planned:** the `/v1/submit` envelope will be superseded by the enriched Roadstead API
(`docs/roadmap.md`). The OpenAI-compatible surface is unaffected and stays strictly compatible.

### Added

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

### Changed

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
