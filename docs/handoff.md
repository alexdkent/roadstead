# Handoff — where this stands and what to do next

**Written 2026-08-31**, at the moment of extraction, for whoever picks this up next — most likely a
fresh Claude Code instance with none of the origin project's context.

Read `CLAUDE.md` first. It carries the concurrency invariant and the engine-behaviour findings, and
those are the two things that will bite you if you skip them.

---

## Where this came from

Roadstead was extracted from a private monorepo (`OriginFleet`) where it runs in production as
`originfleet.llmproxy`, fronting all LLM traffic for a six-machine inference fleet. Before
extraction it was evaluated against ~25 open-source LLM gateways to decide whether to abandon it and
adopt one of them. The answer was no — nothing in the field satisfies the requirement, and two of
its capabilities have no counterpart anywhere surveyed. That decision record is `evaluation.md`, and
it is worth reading because it is also the clearest statement of what this project is *for*.

**The origin copy remains live and authoritative for behaviour.** See the authority rule in
`CLAUDE.md`. Structural work here is this repo's own; behavioural change is not.

---

## What was done on 2026-08-31

**Phase 0 — the sever.** Four imports of the host application's framework were removed from the
library core, routed through a new `hooks.py` integration seam and a vendored `metrics.py`. 29 of
30 modules now import nothing outside the package; `__main__.py` holds the only host imports and
both are soft.

**Phase 1 — mostly done, and it works.**

- Extracted with `git filter-repo`, **not** copied: 282 commits preserved, reaching back to
  `9b11729` (2026-05-27, *"centralized LLM scheduler proxy — DRR scheduling, priority bands"*).
  `git blame` still finds original rationale.
- Namespace rewritten `originfleet.llmproxy` → `roadstead` across 96 files, with test root-anchor
  depths (`parents[N]`) recomputed per file.
- `pyproject.toml`, Apache-2.0 `LICENSE`, CI, and this documentation set added.
- **The suite is green standalone: 1064 passed, 1 skipped**, in a clean venv with no `originfleet`
  on the path and no fleet, network or backend required.

### One real finding from the standalone build — since resolved

`starlette` *was* pinned **`<1.0`**, and the bound was load-bearing: Starlette 1.0 removed the
`on_startup`/`on_shutdown` constructor arguments this app used, so installing unpinned produced
**142 identical collection errors**. Migrating `build_app` to `lifespan=` lifted the pin the same
day — see "What is left → Immediate → 2" below.

---

## What is left

### Immediate

1. **Empty `tests/_pending/`** — 14 quarantined files, each needing a small understood change. Its
   README explains each and why deleting them would be wrong. This is the highest-value next task:
   6 are contract tests whose *server-side half* Roadstead genuinely needs, and 8 are fleet-coupled
   doctrine tests that mostly belong back in the monorepo.
2. ~~**Migrate off `on_startup`/`on_shutdown` to `lifespan=`**, then drop the starlette upper
   bound.~~ ✅ **done 2026-08-31.** `build_app` now takes `lifespan=`; the `<1.0` bound is gone.
   Verified green on both the pinned 0.52.1 and the unpinned 1.6.0 — 1066 passed, 1 skipped each.
   `tests/test_lifespan_wiring.py` is new: nothing previously exercised the lifespan protocol at all
   (every other test calls `svc.startup()`/`svc.shutdown()` directly and drives the app through
   `httpx.ASGITransport`, which skips it), so the wiring could have been unhooked with the whole
   suite still green. It was mutation-checked — remove `lifespan=` and both new tests fail.
3. **Finish `api.md`** — two sections are marked INCOMPLETE: the nested `/v1/fleet/*` analytics
   schemas, and a spot-check for delegator-signature drift between `service.py` and
   `http_handlers.py`.

### Phase 2 — the harness

Backends are ~80% covered and callers ~50%. The main work is **promotion and packaging, not
invention**:

- **Promote `tests/fake_backend.py` to a first-class module.** For a gateway whose thesis is
  capacity-aware admission, *"here is a programmable backend that lies about its capacity on
  demand"* is a product feature, not test scaffolding. It already emulates both engine shapes and a
  per-request fault library including `capacity_desync`.
- **Add a real-engine wire-fidelity test** — a compose file running one small real `llama-server`,
  purely to catch the day an engine changes its `/props` shape. Slow, few tests, off the default
  path.
- **Client-contract tests** for the error envelope, the deferrability substrings and the
  timeout-floor mirror — currently only asserted from the host's side, which is the tautology trap
  (two values compared from one source prove nothing).

### Phase 3 — parity, then the deferred phases

Golden-oracle parity against the origin copy, then soak. Cutover (Phase 4) and publication
(Phase 5) are explicitly deferred and gated separately — publishing is gated on churn settling, not
on cutover.

---

## Things that will mislead you

- **Comments referencing `ship.sh`, `cexec`, skills, agents or fleet hosts** are extraction
  leftovers. The reference is dead; **the reasoning usually is not.** Don't delete the reasoning.
- **`models.yaml` is real fleet data**, not an example. See `corpus_and_scrub_plan.md` — it must
  become an example before this goes public.
- **`creative` is a retired endpoint name kept deliberately** in three load-bearing places
  (persisted rows, the pre-discovery wire id, and a routable alias). Renaming it is a data
  migration, not a rename.
- **Several "obvious simplifications" are load-bearing workarounds.** `CLAUDE.md` lists them with
  the measurement behind each. The SSE terminal-chunk rule is the one most likely to look like dead
  code and most costly to remove.
- **The suite passing does not prove a capability works.** Every schema and generation config needs
  one real call at the size you will actually send before you trust it.

## Open questions inherited

1. ~~**SIGTERM vs SIGKILL for shutdown**~~ ✅ **resolved by experiment, 2026-08-31.** SIGTERM is
   correct — the drain persists DRR budgets and completions even for a straggler it cancels, which
   SIGKILL loses. But it is slower than the code implies: uvicorn's `timeout_graceful_shutdown` and
   the app's `_DRAIN_DEADLINE_S` are **serial, not nested** (uvicorn bounds the in-flight
   *connections*, then sends `lifespan.shutdown`, and never bounds the lifespan shutdown at all), so
   the worst case is their sum — **78.25s measured** against a 48s budget that reads as if it covered
   everything. **A container stop-grace-period must be ≥90s.** Re-runnable:
   `tools/sigterm_drain_probe.py`; full write-up in `ledger.md`.

   One thing it turned up is **behavioural, so it belongs upstream, not here**: the caller of a
   cancelled straggler gets a raw `500 Internal Server Error` at the 48s mark rather than the proxy's
   clean JSON error envelope, because uvicorn cancels the handler task and bypasses the
   `exception_handlers` backstop. Also worth reconsidering upstream: the comment deriving
   `timeout_graceful_shutdown` from `_DRAIN_DEADLINE_S` reasons from a nesting that does not exist.
2. **Catalog placement** — Roadstead currently owns `models.yaml` and its reader. The host keeps its
   own copy; there is deliberately **zero build-time coupling** between them during the dual-track
   period. Revisit only at cutover.
