# Roadstead — Claude Code guide

> A **roadstead** is the sheltered anchorage outside a harbour where vessels wait for a berth to
> free up. That is not a metaphor for the admission queue — it *is* the queue, and specifically one
> that exists because capacity is finite and currently occupied.

**What this is:** a capacity-aware admission-controlling gateway for self-hosted LLM inference
fleets — several llama.cpp servers and a vLLM tensor-parallel pair across heterogeneous hardware,
without Kubernetes. OpenAI-compatible on the front, model-authoritative on the back.

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

## 🚨 Provenance and the authority rule — READ THIS FIRST

This code was extracted on 2026-08-31 from a live monorepo (`OriginFleet`, private), where it runs
in production as the Python package `originfleet.llmproxy` (on disk:
`originfleet/originfleet/llmproxy/`). **That in-situ copy is still live, still under development,
and is AUTHORITATIVE FOR BEHAVIOUR.**

The history here is the real thing — 282 commits going back to `9b11729` (2026-05-27, *"centralized
LLM scheduler proxy — DRR scheduling, priority bands"*), extracted with `git filter-repo` rather
than copied, so `git log`/`git blame` on any line still reaches its original rationale.

Until the extraction plan's Phase 3 exit criterion is met:

- **Behavioural fixes land in the monorepo first**, then come here on a deliberate re-sync.
- **Do not "fix" a behaviour here unilaterally.** If something looks wrong, it is more likely a
  deliberate workaround whose reason is recorded below or in `docs/ledger.md` than a bug.
- Structural work — packaging, namespace, harness, CI, docs — is this repo's own and does not need
  to go through the monorepo.

Divergence is the main risk this project carries. Keep re-syncs deliberate and reviewed.

**What does NOT apply here.** This repo has no access to and no dependency on: the fleet's agents,
`ship.sh`, `cexec`, the `.claude/skills/` layer, the regression ledger, or any fleet host. If a
comment or docstring references one of those, it is a leftover from extraction — the reference is
dead, but the *reasoning* it points at is usually still valid. Don't delete the reasoning.

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
the write queue, and persists DRR budgets. ⚠️ The monorepo's knowledge layer contradicts itself on
this point (one source says SIGTERM hangs behind slow in-flight requests and SIGKILL is correct).
**This is an open question — resolve it by experiment before containerising**, because `docker stop`
sends SIGTERM then hard-kills after 10s, which would silently lose the drain.

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
  __main__.py       entrypoint
tests/              the suite (~107 files) + fake_backend.py + corpus/
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
- **`tests/fake_backend.py` is the most reused asset in the repo.** A Starlette app under real
  uvicorn on a real socket (so real `httpx` and real SSE framing are exercised, not a
  `MockTransport`), emulating both engine shapes and a library of south-face pathologies selectable
  per-request via an `X-Fault` header — including `capacity_desync`, which accepts N concurrent and
  503s beyond while `/props` lies about capacity.

## 🚨 Before this repo goes public

It is **private** and must stay private until the scrub in `docs/corpus_and_scrub_plan.md` is done.
The tree and its 282 commits of history contain private LAN topology (`10.0.0.x`), real host names,
and a `models.yaml` describing actual hardware. **Scrubbing the working tree is not enough — it is in
the history**, which means another `git filter-repo` pass. Read that plan before changing visibility.
