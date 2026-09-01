# Roadmap

**Adopted 2026-08-31.** Replaces the extraction plan in `history.md`, which is now closed.

Roadstead is its own project, with no dependencies or expectations outside it. Nothing here is
scoped by what a downstream consumer currently expects — we optimise for the function this gateway
provides. A new API spec is a *deliverable* to whoever wants it, not a constraint on the design.

---

## What we are building

**Primary: a high-quality capability, run locally.** This is the thing that has to be excellent. It
is judged by whether it serves real traffic well — not by feature count.

**Secondary: an open-source project other people can use.** Real, and it shapes decisions — generic
configuration, no private topology, a published contract, a credible getting-started path. But when
the two conflict, the local capability wins.

The origin was a proxy for one fleet. The thesis generalises: *the interesting question is not which
backend to send to, but whether to send at all right now, whose request goes first, how long it is
reasonable to wait, and — new — what it costs and who pays.*

---

## The shape it is heading for

### North face — two APIs, one of them enriched

**OpenAI-compatible**, and strictly so. It stays a drop-in target for existing clients. 🚨
Enrichment never appears as extra fields inside an OpenAI-shaped body: a client that validates the
schema must not break. Enriched data rides in response headers, or the caller uses the other API.

**The enriched Roadstead API**, which supersedes today's `/v1/submit` envelope rather than extending
it. It is designed for what a caller actually needs from a capacity-aware gateway and cannot get
from OpenAI's shape:

- **enriched model information** — what is available, what it can do (context, vision, tools,
  reasoning), what it costs, and what it is *currently* like (live latency distribution, queue
  depth, whether it is healthy);
- **timing data** — the recommended deadline for this call before you make it, and afterwards what
  actually happened: queue wait, TTFT, decode, total, against what was predicted;
- **prioritisation** — priority and interactivity as first-class declarations, with the deadline
  computed rather than supplied (already true; now stated in the API rather than buried);
- **attribution** — which model actually served, on which provider, at what cost, and whether that
  differed from what was asked for.

### Model abstraction — the key feature

**Intent preferred, concrete names honoured.** A caller declares a capability
(`reasoning`, `fast-chat`, `vision`, `embed`) and Roadstead owns the choice of model, provider and
moment. A caller who must pin a specific model may, and that is treated as a constraint on routing
rather than a different API.

**Substitution is opt-in and always disclosed.** Today's `degrade_ok` generalises: a caller opts
into being served by an alternative when its first choice is unavailable, over capacity, or over
budget. Whatever happens, the response says what actually served it. A caller that did not opt in is
never silently given something else.

### South face — modular providers

A provider interface, with **llama.cpp and vLLM local** (llama.cpp keeps its `/props` slot discovery
— it is the only backend that publishes true slot counts and per-slot context, and that measurement
is what makes admission control accurate rather than assumed), plus **OpenRouter** and other remote
providers behind the same interface.

Providers differ in what they can *tell* us, and the interface must make that explicit rather than
average it away — a provider descriptor declares whether it publishes occupancy, per-slot context,
token costs, or nothing at all. Capacity discovery is already asymmetric by necessity
(`CLAUDE.md`); the abstraction should formalise that asymmetry, not hide it.

### Remote capacity is overflow, not a parallel universe

**One admission decision, three outcomes: dispatch locally, spill to a remote provider, or defer.**

This keeps *"should I send at all right now"* as the central question and makes remote capacity the
answer to local scarcity rather than a second system with its own rules. Slot-seconds stay the unit
of local fairness because local slots are the scarce thing; remote spill is governed by cost and by
whether the caller is allowed to spend.

### Authentication and identity

**API keys.** A key is the caller's identity — and therefore the DRR fair-share key, the quota
holder, and the budget holder. The IP-based ACL is demoted to an optional second factor, which also
clears scrub item **S1** (no addresses shipped in code).

### Token management, thresholds, costing

Costing becomes real, not notional: today `usage_rates.py` computes cloud-equivalent cost *avoided*,
because everything was local. With remote providers, some spend is actual money.

**Thresholds degrade; they never reject.** Crossing a budget costs a caller its priority band and
its access to paid remote spill. It never costs it access to local capacity. Two reasons: admission
control should be about *capacity*, not billing, and a misconfigured quota must not be able to take
a caller offline. A runaway caller is bounded by what is free and by DRR fairness, which is what DRR
is for.

### A management interface

Eventually: manage and monitor Roadstead as a standalone product — keys, quotas, budgets, providers,
backends, and the live picture of what the fleet is doing. Deferred, but two constraints bind from
now: it must not violate the concurrency invariant (`CLAUDE.md` — heavy reads go off-loop), and it
must not drag a frontend toolchain into a package whose dependency list is deliberately short.

---

## Workstreams

Not phases. Several can run concurrently; the dependencies between them are what matters.

### A · Provider abstraction — *the foundation*

Extract a provider interface out of `backend.py` (54KB, with llama.cpp/vLLM branching inline).
Formalise the capability descriptor. Nothing else on this list is buildable first: OpenRouter needs
it, spill needs it, per-provider costing needs it, and enriched model information is largely a
readout of it.

### B · Identity and API-key auth

Keys → identity → fair-share key → quota/budget holder. Demote the IP ACL. **Clears scrub S1.**
Blocks meaningful per-caller costing.

### C · The enriched API

The new north face, plus model abstraction (capability aliases, honoured pins, disclosed
substitution). Depends on **A** for model/provider information to be real rather than a config
readout. This is where the eventual API spec deliverable comes from.

### D · Spill, token management and costing

Admission returns local/spill/defer. Per-caller token and cost accounting. Degrading thresholds.
Needs **A** (remote providers) and **B** (who is being metered).

### E · Management interface

Read first, control second. After **B** and **D** exist to be managed.

### Cross-cutting · Soak and hardening

Carried over, and the case for it is unchanged: the concurrency invariant — single loop, no locks,
one writer thread — is the most dangerous thing in the codebase and **nothing in the suite guards
it**. Sustained concurrent load is the only thing that surfaces a violation. It also has to grow to
cover the new surfaces, especially anything that touches remote providers over the network.

### Cross-cutting · The scrub

`corpus_and_scrub_plan.md`. Gates the open-source goal, and **S2 (`models.yaml` → an example) is now
on the critical path for a second reason**: the config schema has to change anyway to describe
providers, capabilities, costs and aliases. Redesigning the catalog and genericising it are the same
piece of work, and doing them separately means doing them twice.

⚠️ The history rewrite stays last — it invalidates every SHA.

---

## Consequences to plan for

**The `/v1/submit` envelope is going away.** A planned breaking change under
`compatibility.md`: deliberate, and recorded in `CHANGELOG.md` when it lands. The OpenAI surface is
unaffected.

**`models.yaml` grows a provider dimension** and stops being a description of one fleet's hardware.

**`docs/api.md` is executable** — four test files read it back and fail when code and document
disagree. Every change above lands with its contract, or the suite says so.

**Not everything from the origin generalises.** The retired `creative` endpoint name, the
fleet-specific roles, the `degrade_ok` spelling: these are one deployment's vocabulary. Keep the
mechanism, drop the vocabulary — but check `git log` first, because several of them are load-bearing
in ways the name does not suggest (`history.md`, "Things that will mislead you").

---

## Still open

- **Multi-tenancy depth.** Are keys flat, or do they nest (team → key) for quota and budget
  inheritance? Flat is enough for the primary goal and probably not for the secondary one.
- **Where cost truth lives.** Provider-reported spend vs. our own token accounting; they will
  disagree, and one of them has to be authoritative for threshold decisions.
- **Streaming through a remote provider** under a computed deadline — the soft-deadline extension
  logic assumes decode progress is observable, which is provider-dependent.
- **Whether the timeout model can learn per-provider** for remote backends whose latency we do not
  control, or whether those get a different deadline strategy entirely.
