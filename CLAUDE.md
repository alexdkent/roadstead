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
in production as `originfleet.llmproxy`. The history here is the real thing — 282 commits going back
to `9b11729` (2026-05-27, *"centralized LLM scheduler proxy — DRR scheduling, priority bands"*),
extracted with `git filter-repo` rather than copied, so `git log`/`git blame` on any line still
reaches its original rationale. **Use that.** It is the best documentation this project has.

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

## 🚨 The concurrency invariant — NOT guarded by any test

This is the single most dangerous thing to get wrong, and **nothing in the suite will catch you.**

- **Single event loop. No locks** on in-memory scheduler / budget / cache state. That is only safe
  because there is exactly one thread mutating it.
- **Exactly ONE sanctioned background writer thread** — the `queue.py` DB writer, which owns its
  write connection.
- The loop thread uses its **own read connection**. Heavy dashboard aggregations run **off-loop via
  `asyncio.to_thread`**, and each pool thread gets its **own read-only connection** (thread-local;
  WAL permits concurrent readers) so a slow aggregation never blocks the loop.
- **Never add a `workers=` parameter, a thread pool that WRITES, or a second thread that touches
  scheduler or budget state.** Connections are never shared across threads.

Shut down with **SIGTERM**, which runs a bounded drain (≤30s) that finishes in-flight work, flushes
the write queue, and persists DRR budgets. The monorepo's knowledge layer contradicted itself here;
**settled by experiment on 2026-08-31** (`tools/sigterm_drain_probe.py`, full result in
`docs/ledger.md`). Both halves were right about different things: SIGTERM is correct — the drain
persists the DRR budget row and the completion row even for a straggler it cancels, which is exactly
what SIGKILL loses — *and* it really can hang, for longer than the code's own comment implies.

🚨 **The two shutdown budgets are serial, not nested.** uvicorn's `timeout_graceful_shutdown` bounds
the in-flight HTTP *connections*; only when it expires does uvicorn send `lifespan.shutdown`, and
only then does `ProxyService.shutdown`'s `_DRAIN_DEADLINE_S` drain begin. uvicorn never bounds the
lifespan shutdown at all. Worst case is their **sum** — 78.25s measured, against a 48s budget that
reads as though it covers everything.

🚨 **Any container stop-grace-period must be ≥90s.** Measured in a real container
(`tools/docker_stop_probe/`): at `docker stop`'s **default 10s** with work in flight, the proxy is
**SIGKILLed with zero budget and zero completion rows persisted** — the shutdown handler never runs
at all. With `-t 90` the same case exits cleanly at 78.31s with both persisted. The default *works
while the proxy is quiet*, which is how it will be tested and why it would first fail under load.

---

## Engine-behaviour findings — why the code is shaped this way

Every item below was measured against real backends. Each one looks like a wart until you know the
reason. **Do not simplify these away.**

**Capacity discovery is asymmetric, deliberately.** llama.cpp `/props` yields real
`n_parallel`/`total_slots` and per-slot `n_ctx`. vLLM exposes only `max_model_len`, so vLLM
concurrency stays **config-seeded** with a drift alert. This is a property of the engines, not an
oversight.

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
roadstead/          the package (34 modules)
  scheduler.py      DRR + priority bands + admission        — pure computation, no I/O
  cost_model.py     slot-second cost, EWMA-calibrated       — pure computation, no I/O
  timeout_model.py  learned latency → recommended deadline  — pure computation, no I/O
  correction.py     the output-integrity layer
  lifecycle.py      admission → dispatch → streaming → timeout recording
  health.py         capacity discovery, circuit breaker, drain
  queue.py          durable event log + THE single writer thread
  backend.py        south face: the HTTP client to inference engines
  model_catalog.py  reads models.yaml — the naming/capability authority
  hooks.py          the integration seam (see below)
  testing/          SHIPPED test doubles — the programmable fake backend
  __main__.py       entrypoint
tests/              the suite + corpus/ (GBNF fixtures, north-face, schemas)
tools/              off-default-path experiments (real processes, real signals)
docs/               specs, plan, evaluation, ledger
```

The three `pure computation, no I/O` modules are the crown jewels and the easiest to test — keep
them that way.

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

## 🚨 Before this repo goes public

It is **private** and must stay private until the scrub in `docs/corpus_and_scrub_plan.md` is done.
The tree and its 282 commits of history contain private LAN topology (`10.0.0.x`), real host names,
and a `models.yaml` describing actual hardware. **Scrubbing the working tree is not enough — it is in
the history**, which means another `git filter-repo` pass. Read that plan before changing visibility.
