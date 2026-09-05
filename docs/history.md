# History — the extraction, and what it left behind

**This is a closed record, not a plan.** It was written on 2026-08-31 at the moment of extraction
for whoever picked the project up next. That handoff is complete: Roadstead is its own project with
no dependencies or expectations outside it. The forward-looking half moved to **`roadmap.md`**;
what remains here is the part still worth being able to look up.

Kept because several things reference it and because the reasoning behind some odd-looking code is
recorded nowhere else: where the code came from, what the extraction did and cost, and the findings
that came out of standalone-ing it. `ledger.md` and a few test docstrings cite this file.

> Statements below describe the state of things in August 2026. Two governance rules quoted in
> passing — that the origin repo was authoritative for behavior, and that a cutover to it was
> planned — were both retired on 2026-08-31. See `docs/internals.md` and `compatibility.md`.

---

## Where this came from

Roadstead was extracted from a private monorepo, where it ran in production fronting all LLM
traffic for a multi-machine inference fleet. Before extraction it was evaluated against ~25
open-source LLM gateways to decide whether to abandon it and adopt one of them instead. The answer
was no — nothing in the field satisfies the requirement, and two of its capabilities have no
counterpart anywhere surveyed. That decision record is `evaluation.md`, and it is worth reading
because it is also the clearest statement of what this project is for.

**Roadstead is no longer tied to that origin (2026-08-31).** It is not a 1:1 replacement and is not
trying to become one — it will become a superset and may occasionally break exact backward
compatibility on purpose. Behavioral fixes land here, with no round-trip back to the origin repo.
See `docs/internals.md` and `compatibility.md`.

---

## What was done on 2026-08-31

**Phase 0 — the sever.** Four imports of the host application's framework were removed from the
library core, routed through a new `hooks.py` integration seam and a vendored `metrics.py`. All but
one module now import nothing outside the package; `__main__.py` holds the only host imports and
both are soft.

**Phase 1 — done.**

- Extracted with `git filter-repo`, not copied: 282 commits preserved, reaching back to the first
  commit on 2026-05-27 ("centralized LLM scheduler proxy — DRR scheduling, priority bands").
  `git blame` still finds the original rationale.
- Namespace rewritten to the `roadstead` package across 96 files, with test root-anchor depths
  recomputed per file.
- `pyproject.toml`, Apache-2.0 `LICENSE`, CI, and this documentation set added.
- The suite is green standalone: 1064 passed, 1 skipped, in a clean virtualenv with no host package
  on the path and no fleet, network or backend required.
- A batch of tests that depended on host-specific paths or fixtures did not make the trip; the
  invariants worth keeping were re-derived against fixtures that live in this repo instead (see
  `tests/wire_contract.py` and `tests/corpus/grammars/` below).

### One real finding from the standalone build — since resolved

`starlette` was pinned `<1.0`, and the bound was load-bearing: Starlette 1.0 removed the
`on_startup`/`on_shutdown` constructor arguments this app used, so installing unpinned produced 142
identical collection errors. Migrating `build_app` to `lifespan=` lifted the pin the same day — see
"What is left → Immediate → 2" below.

---

## What is left

### Immediate

1. ~~**Empty the test quarantine directory**~~ ✅ **done 2026-08-31.** The directory and its README
   are gone. Suite 1064 → 1196 passed at that point (1234 after item 3). Most of the quarantined
   tests were not coupled to the host application at all — they were coupled to a *path* that
   assumed the old monorepo checkout, reaching into the host's tree for files that are now this
   repo's own. Two new shared assets came out of untangling that: `tests/wire_contract.py` (the
   marker substrings and keepalive figures as literals transcribed from `docs/api.md`, so client and
   server can each be pinned without either asserting its own rule back at itself, with
   `tests/test_wire_contract.py` reading the doc back so the transcription cannot rot silently) and
   `tests/corpus/grammars/` (six vendored GBNF fixtures covering the shapes real callers send,
   replacing a scan of a private system's agent tree, asserted by name rather than count so a
   swapped fixture cannot keep the count up while dropping a shape). A handful of tests were tied
   tightly enough to the origin system's internals or its private catalog data that they were left
   behind rather than ported.
2. ~~**Migrate off `on_startup`/`on_shutdown` to `lifespan=`**, then drop the starlette upper
   bound.~~ ✅ **done 2026-08-31.** `build_app` now takes `lifespan=`; the `<1.0` bound is gone.
   Verified green on both the pinned 0.52.1 and the unpinned 1.6.0 — 1066 passed, 1 skipped each.
   `tests/test_lifespan_wiring.py` is new: nothing previously exercised the lifespan protocol at all
   (every other test calls `svc.startup()`/`svc.shutdown()` directly and drives the app through
   `httpx.ASGITransport`, which skips it), so the wiring could have been unhooked with the whole
   suite still green. It was mutation-checked — remove `lifespan=` and both new tests fail.
3. ~~**Finish `api.md`** — two incomplete sections.~~ ✅ **done 2026-08-31.** Both closed, and
   neither by prose alone:

   - **`/v1/fleet/*` analytics schemas** chased to column level as §3.1, and pinned by
     `tests/test_fleet_analytics_schema.py`, which drives the real producers against a seeded
     `queue.db` and reads §3.1 back, comparing table-by-table in both directions. An added, renamed
     or dropped field now fails the suite instead of silently breaking a dashboard. It was
     mutation-tested — and the first version was caught being weaker than it claimed: pooling the
     fields of every table under a heading let a deleted `calls[].p95` pass because
     `by_endpoint_1h[]` happened to document a field of the same name. Compare shape to shape.
   - **Delegator-signature drift** audited by AST: 30 delegators, 30 identical signatures, zero
     drift — the stated contract holds. Now enforced by `tests/test_delegator_signatures.py` rather
     than re-asserted by hand.

   Two things the audit turned up and the doc now records: `usage_rollup`'s p50/p95 include queue
   wait while `fleet_activity`'s `p95` does not (so the two are not comparable, which nothing said);
   and all three producers return a narrower shape when the database is unopened — no `now`, no
   `today_start` — so a consumer that assumes those keys `KeyError`s rather than degrading.

### Phase 2 — the harness ✅ **done 2026-08-31**

All three items. Suite 1064 at extraction → 1267 passed, 1 skipped, 6 deselected.

- ~~**Promote the fake backend to a first-class module.**~~ → **`roadstead.testing`**. It ships.
  `tests/test_testing_module_is_public.py` proves it imports from the installed distribution in a
  subprocess with the repo off `sys.path`, and pins `__all__` against the module and `ALL_FAULTS`
  against the `FAULT_*` constants. Internal sentinels gained public spellings
  (`USAGE_DEFAULT`/`OMIT_USAGE`) — a shipped module should not make callers import underscored
  names — with the old names kept as aliases.
- ~~**Add a real-engine wire-fidelity test.**~~ → **`tests/wire_fidelity/`**. Built around a
  sharper framing than "a second suite": the fake backend is a claim about how real engines behave,
  so `conformance.py` states the south-face contract once and both the fake (every run) and a real
  `llama-server` (`-m wire_fidelity`) are held to it. The real-engine half has never been executed —
  no Docker on the machine it was written on. See its README; the image tag and env names are
  unverified.
- ~~**Client-contract tests.**~~ Error envelope and deferrability substrings landed in Phase 1 with
  `tests/wire_contract.py`. The timeout-floor mirror is now `tests/test_timeout_floor_contract.py`,
  and closing it needed `docs/api.md` §1.4, which did not exist: the endpoint the client mirrors a
  floor from was undocumented, which is precisely why mirroring was the only option. §1.4 now
  publishes the fallback floor and both ceilings, and documents that `source == "floor"` means
  `recommended_timeout_s` **is** the floor — so a client can ask instead of mirroring.

Two findings recorded rather than fixed (see "Open questions" below): the fake's `/props` `n_ctx`
fidelity gap, and the fact that per-class floors are deliberately absent from §1.4 as deployment
data, with a test that fails if anyone tabulates them into a document headed for publication.

### Phases 3+ — moved to `roadmap.md`

Everything forward-looking left this file on 2026-08-31, when the project's scope widened well past
being a standalone copy of the original component. Soak testing and scrubbing example data both
survive as workstreams there.

For the record, the plan as it stood at extraction had five phases ending in a cutover back to the
origin repo. Phases 0-2 were completed; parity and cutover were deleted rather than done — the
project stopped being a 1:1 replacement, and a parity gate on a deliberate superset fails on every
improvement.

---

## Things that will mislead you

- **Comments referencing the origin repo's internal tooling** are extraction leftovers. The
  reference is dead; the reasoning usually is not. Don't delete the reasoning when you find it.
- **`models.yaml` was real fleet data**, and is an example now (2026-08-31). The schema is
  Roadstead's contract; the fleet in the shipped file is invented, on RFC 5737 addresses. A comment
  or a test that reads as though the catalog described somebody's real machines predates that.
- **`creative` is a retired endpoint name kept deliberately** in three load-bearing places
  (persisted rows, the pre-discovery wire id, and a routable alias). Renaming it is a data
  migration, not a rename.
- **Several "obvious simplifications" are load-bearing workarounds.** `docs/internals.md` lists
  them with the measurement behind each. The SSE terminal-chunk rule is the one most likely to look
  like dead code and most costly to remove.
- **The suite passing does not prove a capability works.** Every schema and generation config needs
  one real call at the size you will actually send before you trust it.

## Open questions inherited at extraction

1. ~~**SIGTERM vs SIGKILL for shutdown**~~ ✅ **resolved by experiment, 2026-08-31.** SIGTERM is
   correct — the drain persists DRR budgets and completions even for a straggler it cancels, which
   SIGKILL loses. But it is slower than the code implies: uvicorn's `timeout_graceful_shutdown` and
   the app's `_DRAIN_DEADLINE_S` are serial, not nested (uvicorn bounds the in-flight connections,
   then sends `lifespan.shutdown`, and never bounds the lifespan shutdown at all), so the worst case
   is their sum — 78.25s measured against a 48s budget that reads as if it covered everything. A
   container stop-grace-period must be at least 90s. Re-runnable: `tools/sigterm_drain_probe.py`;
   full write-up in `ledger.md`. (Superseded — now 108s, see `docs/internals.md` /
   `tests/test_shutdown_budget.py`.)

   Two things it turned up are now ours to fix rather than parked as belonging upstream: the caller
   of a cancelled straggler gets a raw 500 Internal Server Error at the 48s mark rather than the
   proxy's clean JSON error envelope (uvicorn cancels the handler task, bypassing the
   `exception_handlers` backstop), and a comment deriving `timeout_graceful_shutdown` from
   `_DRAIN_DEADLINE_S` reasons from a nesting that does not exist.
2. ~~**The fake backend's `/props` `n_ctx` units**~~ ✅ **resolved 2026-08-31**, against a real
   `llama-server` (b5350) — and not the way either candidate answer expected. There is no
   top-level `n_ctx` on a current build, so `health.py`'s divide-by-slots fallback is dead code
   rather than wrong. The sharper finding sat next to it: the fake also published
   `default_generation_settings.n_parallel`, which a real engine does not, and `health.py` prefers
   it over `total_slots` — so the fallback real discovery entirely depends on was never exercised by
   any test. `roadstead.testing` now defaults to the verified narrow shape. Full write-up in
   `ledger.md`; measurements in `tests/wire_fidelity/README.md`.
3. **Catalog placement** — Roadstead owns `models.yaml` and its reader. There was deliberately zero
   build-time coupling to the origin repo's copy during the dual-track period, and with the dual
   track gone the two are simply separate files in separate projects. The open question is no
   longer where the catalog lives but what ships: the schema is Roadstead's contract, the data is a
   private fleet's. Shipping an example catalog is the answer, and it landed 2026-08-31 —
   `roadstead/models.yaml` is that example, and the suite runs against it.
4. **The `n_parallel` preference order** — new 2026-08-31, and newly actionable. `health.py`
   prefers `default_generation_settings.n_parallel` over `total_slots`, but a current llama.cpp
   publishes only the latter. The preference is harmless (the fallback fires) but it is backwards
   relative to measured reality, and it hid a coverage hole for months. Worth reordering — with a
   test that fails if a build ever publishes both and they disagree.
