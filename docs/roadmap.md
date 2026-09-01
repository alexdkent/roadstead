# Roadmap

**Adopted 2026-08-31.** Replaces the extraction plan in `history.md`, which is now closed.

Roadstead is its own project, with no dependencies or expectations outside it. Nothing here is
scoped by what a downstream consumer currently expects — we optimise for the function this gateway
provides. A new API spec is a *deliverable* to whoever wants it, not a constraint on the design.

---

## The mission

**Roadstead is a local-first LLM scheduler. It stands between many kinds of caller and many kinds of
model, and absorbs the mismatch so that neither side has to model the other.**

That sentence is doing specific work, so each half is worth unpacking.

**Diverse callers.** Low-latency chat turns, coding assistants, summarizers, extractors, batch
ingestion. They differ in *urgency* more than in what they ask for: an interactive turn and an
overnight summarizer may want the identical model and cannot wait the same amount of time. That is
why deadline tolerance — not model choice — is the axis fair-sharing has to run along, and why the
unit of fairness is occupancy *time*.

**Diverse backends.** Small to medium to cloud. Thinking and non-thinking. Four fixed slots or
elastic. Grammar-conforming and grammar-broken. That last one is the part the field mostly pretends
away: backends are not merely faster or slower, they are unreliable in *specific, characterizable*
ways — a model that drops `finish_reason`, a vLLM that 400s the whole request over a missing launch
flag, a structured endpoint that returns `{}` for 31 hours while looking perfectly healthy.
Absorbing that is not a side feature; `correction.py` is the largest module in the package.

**Local-first**, precisely: local capacity is the default *and the design center*, cloud is explicit
overflow rather than the fallback that quietly becomes the norm. It runs with no account and no
internet, and nothing leaves the machine unless someone opts into spilling it. Not local-only, and
emphatically not cloud-first-with-local-bolted-on.

**Neither side models the other.** Callers declare intent, not models, and never need to know that
one endpoint has four slots of 32K each or that another needs a specific flag to reason. Backends
are met on their own terms, warts included.

🚨 **Not an "LLM orchestrator."** In this field that word means agent and chain frameworks;
Roadstead runs no workflows and chains no calls. It is a *scheduler* in the operating-system sense —
what runs where, when, and for whom, under contention. Use "scheduler" in positioning and "gateway"
in technical prose where it is literally accurate. Getting this wrong attracts the wrong audience
and loses the right one.

### The two goals

**Primary: a high-quality capability, run locally.** This is the thing that has to be excellent. It
is judged by whether it serves real traffic well — not by feature count.

**Secondary: an open-source project other people can use.** Real, and it shapes decisions — generic
configuration, no private topology, a published contract, a credible getting-started path. But when
the two conflict, the local capability wins.

### Why anyone else would want it

**Cloud gateways assume elastic capacity, so admission control is uninteresting to them. Local tools
assume one user, so fairness is uninteresting to them.** The intersection — capacity that is finite
*and* many callers competing for it — is unoccupied, and everything distinctive in this codebase
falls out of standing in it: slot-second fairness, runtime capacity discovery, computed deadlines,
and a correction layer for backends that misbehave under load.

Because the capacity is finite and yours, the interesting question stops being *which backend* and
becomes: **whether to send at all right now, whose turn it is, how long is reasonable to wait, what
it costs — and what to do when the answer comes back malformed.**

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
clears scrub item **S1** (no addresses shipped in code). **Landed 2026-09-01** — see Workstream B.

### Token management, thresholds, costing

Costing becomes real, not notional: today `usage_rates.py` computes cloud-equivalent cost *avoided*,
because everything was local. With remote providers, some spend is actual money. **Landed
2026-09-01** — see Workstream D.

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

Extract a provider interface out of `backend.py` (llama.cpp/vLLM branching inline). Formalise the
capability descriptor. Nothing else on this list is buildable first: OpenRouter needs it, spill needs
it, per-provider costing needs it, and enriched model information is largely a readout of it.

**Landed 2026-08-31 — the local half.** `roadstead/providers/` is the interface, with llama.cpp and
vLLM behind it: `prepare_chat_payload`, `path_for`, `discover_capacity` / `parse_capacity`, and a
`ProviderDescriptor` declaring what a backend publishes, what it requires of a request, and what it
gets wrong. `backend.py` keeps the transport. The four sites that asked `backend_engine == "vllm"`
now read the capability they meant, and a test fails if a new engine-name comparison appears.
Behaviour-preserving, checked by differential comparison against the pre-split code rather than
asserted.

**Landed 2026-08-31 — the remote half.** `providers/openrouter.py`, the first backend Roadstead does
not own, and with it the assumptions the local engines let us keep:

- **`base_url` + `api_key_env` on `EndpointConfig`**, and a transport keyed on the URL rather than
  building `http://host:port` itself. The key is read from the environment by the *name* the config
  declares — a key in a config file is a key in a git history.
- **Auth is the provider's business.** `Provider.request_headers` supplies it; the transport never
  learns about credentials. A missing key refuses before a socket is used, and makes the endpoint
  fail its health probe rather than 502-ing live traffic one call at a time.
- **`ProviderError`**: a provider that cannot honour a *constraint* refuses (OpenRouter and GBNF);
  one that cannot use a *hint* drops it (`id_slot`, `chat_template_kwargs`). Dropping a constraint
  silently is the failure this codebase treats as worst.
- **`probe_json`** — a generic probe, so a provider owns its route and its parsing while the
  transport keeps owning the pool and the deadline. New providers use it instead of growing another
  `probe_<engine>_<thing>`.
- **`roadstead.testing` speaks the remote shape** (`engine="openrouter"`): base path, a 401 without a
  bearer token, and a two-entry catalogue with `context_length` and per-token pricing. The suite
  exercises the remote path on a real socket, with no network.

**Still open in A:**

**Landed 2026-08-31 — the catalog.** `models.yaml` is now `providers:` + `endpoints:`: connection and
engine on one side, capacity and policy on the other, each endpoint naming its provider. A local
provider hosts one endpoint; a remote one hosts many, which is what makes the split earn its keep.
Shipped as a generic example on RFC 5737 addresses — the same piece of work as scrub item **S2**,
done together as planned rather than twice.
- **Per-provider costing** is descriptor-shaped (`publishes_token_costs`, and OpenRouter's catalogue
  carries the prices) but has no reader; it lands with **D**.
- **A second remote provider** would be the real test of the abstraction. One of each is enough to
  find the `host:port` assumption; it is not enough to know which of OpenRouter's shapes are
  *OpenRouter's* and which are *remote's*.

### B · Identity and API-key auth

**Landed 2026-09-01.** Keys → identity → fair-share key → quota/budget holder; the address ACL is
demoted to a second factor. `identity.py` is the one place that knows the precedence, and everything
downstream — the two OpenAI doors, `/v1/submit`, the admin gates, the deadline floor — reads the
answer off a `Principal` rather than asking about an address.

- **`/v1/submit` is gated.** It was not, on the same port as the ACL-gated OpenAI doors, so the DRR
  fair-share key was self-asserted by anyone who used it. A key now overrides a body-declared
  `agent_id`; an address only fills in one the body omitted.
- **Three doctrine rules**, in `docs/api.md` §1.5 and enforced by `tests/test_identity_keys.py`: a
  presented key that does not resolve is a 401 and never falls back to the address; with no keys
  configured a presented key is ignored entirely (every OpenAI client sends one whether anybody
  meant it to or not); an authenticated non-admin identity does not inherit its host's admin
  privileges.
- **Clears scrub S1**, as planned — and S3 with it, plus a fourth private inventory the plan had not
  listed (`agents.yaml`, the same way S2 turned up `usage_rates.py`).

**Still open in B:** keys are flat — there is no team → key nesting for quota and budget
inheritance, which is the "multi-tenancy depth" question below. Nothing reads a key on the
management side yet either: there is no enrolment surface, so a key is created by editing config and
restarting. That belongs with **E**.

### C · The enriched API

The new north face, plus model abstraction (capability aliases, honoured pins, disclosed
substitution). Depends on **A** for model/provider information to be real rather than a config
readout. This is where the eventual API spec deliverable comes from.

### D · Spill, token management and costing

**Landed 2026-09-01.** `spend.py` is a fourth pure-computation module: prices, per-caller accounting,
and the threshold. `Scheduler._admit` returns `DISPATCH` / `SPILL` / `DEFER` — one decision, three
outcomes, with local capacity tried first for every caller before anything about money is consulted.

- **Two kinds of money, never summed.** A local endpoint is priced at what renting the same class of
  model would have cost — a saving. A remote provider's published price is an invoice. A
  `TokenPrice` carries which it is; only the invoice counts against a threshold. A threshold on
  avoided cost would throttle a caller for using capacity that is free, which is local-first
  inverted.
- **Prices are read from the provider.** `publishes_token_costs` finally has a reader:
  OpenRouter's catalogue prices arrive on the same discovery pass that reads the context ceiling. An
  operator-declared price in the catalog beats a published one — somebody who wrote a number down
  knows something the catalogue does not.
- **Thresholds degrade.** Crossing `daily_spend_usd` costs one priority band and paid spill, and
  nothing else. **No error code exists for it**, and `docs/api.md` §1.6 says so where a test can read
  it back.
- **Spill is overflow, not a fallback tier.** `spill_to` is a separate field from `failover_to` and
  `spill_ok` a separate flag from `degrade_ok`, because "may a worse model answer" and "may this
  leave the machine at our expense" are different questions and neither implies the other. Spill
  never chains.

**Still open in D:** *Where cost truth lives* (below) is untouched — the ledger is our own token
accounting, and nothing reconciles it against what a provider actually invoices. The live ledger is
also in-memory and resets on restart, which is right for what it governs (whether a caller is
degraded *now*) and wrong for anything an operator would want to bill on; the durable record stays in
`queue.db`. Nothing yet spills on *cost* — the decision is capacity-triggered, so a cheaper remote
endpoint is never preferred to an expensive local one, deliberately, but a deployment with several
remote providers will eventually want to choose between them.

### E · Management interface

Read first, control second. After **B** and **D** exist to be managed.

### Cross-cutting · Soak and hardening

Carried over, and the case for it is unchanged: the concurrency invariant — single loop, no locks,
one writer thread — is the most dangerous thing in the codebase and **nothing in the suite guards
it**. Sustained concurrent load is the only thing that surfaces a violation. It also has to grow to
cover the new surfaces, especially anything that touches remote providers over the network.

### Cross-cutting · The scrub

`corpus_and_scrub_plan.md`. Gates the open-source goal. **S2 is done** (2026-08-31), together with
the catalog redesign it shared its work with — and it turned up a second inventory nobody had listed,
`usage_rates.py`, which is now anchored to model classes rather than to one fleet's models.

**S1 and S3 are done** (2026-09-01), with Workstream B, which is what made S1 possible: removing the
address seeds needed an identity mechanism to remove them *in favour of*. The working tree is clean
of private topology. It also turned up a fourth inventory nobody had listed — `agents.yaml` shipped
one fleet's agent roster with weekly volumes and infra paths — which is now a generic example on the
caller archetypes named above, exactly as `models.yaml` is.

**S4 is done** (2026-09-01): the structured-output corpus keeps every fixture and loses the
vocabulary around them — and it corrected this plan's own headline finding, which had ruled personal
data out of the repo on the strength of no bulk corpus having come across. The synthesized prompts
were written around real identifiers. The straggler sweep now looks for both.

**Only the history rewrite (S6) is left**, and it is the expensive one.

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
