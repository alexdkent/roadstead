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

**The enriched Roadstead API** at `/rs/v1`, which superseded the `/v1/submit` envelope rather than
extending it (**landed 2026-09-01** — Workstream C; the old door is removed). It is designed for what
a caller actually needs from a capacity-aware gateway and cannot get from OpenAI's shape:

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
rather than a different API. **Landed 2026-09-01** — see Workstream C.

**Substitution is opt-in and always disclosed.** `degrade_ok` and `spill_ok` are the operator's two
grants; a request may narrow either and never widen one. Whatever happens, the response says what
actually served it. A caller that did not opt in is never silently given something else — and
choosing an endpoint for an intent is *not* substitution, because nothing was promised.

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

Manage and monitor Roadstead as a standalone product — keys, quotas, budgets, providers, backends,
and the live picture of what the fleet is doing. **The HTTP half landed 2026-09-01** (`/rs/v1/admin`
— Workstream E); a UI has not, and the two constraints still bind: it must not violate the
concurrency invariant (`CLAUDE.md` — heavy reads go off-loop, mutations stay on it), and it must not
drag a frontend toolchain into a package whose dependency list is deliberately short.

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

**Also landed 2026-09-01 — trusted proxies.** `identity.remote_ip` read the peer address and nothing
else, so anything in front of Roadstead collapsed every caller into one identity and, where that
address fell in the default admin nets (loopback, docker-internal — a sidecar is usually one), handed
the control plane to everyone who could reach the proxy. `ROADSTEAD_TRUSTED_PROXIES` is the opt-in;
the caller is the rightmost hop that is not itself trusted; and a forwarded address stops inheriting
the *built-in* admin grant, because "it arrived on loopback" means nothing once a front door exists.
Latent until now, and Workstream G is what makes a front proxy normal — so it went first, alone.

**~~Still open in B~~ — closed 2026-09-01 by E.** Enrolment and revocation are now runtime
operations on `/rs/v1/admin/keys`, so a key is no longer created by editing config and restarting.
Keys stay flat, and that turned out to be the answer rather than a gap: the budget holder is the
`agent_id`, not the key, so many keys → one `agent_id` is already the team-level inheritance the
"multi-tenancy depth" question was asking for.

### C · The enriched API

**Landed 2026-09-01.** `/rs/v1/*` — its own version prefix, because the other north face is
versioned by OpenAI and pinning them together would mean either following somebody else's number or
publishing a `/v2/chat/completions` that is not OpenAI's v2. Three routes: `models` (what can serve
me, what can it do, what is it like now, what does it cost), `plan` (where would this go, how long
should I allow — **without dispatching**), `chat` (do it, and tell me what actually happened).

- **`intent.py` is a fifth pure-computation module.** It is handed an immutable snapshot of every
  candidate endpoint and returns which should serve and why every other could not; every lookup
  happens in `enriched.py`, which assembles the facts. That separation is what makes routing
  testable against a fleet that does not exist.
- **Profiles are expressed in DECLARED CAPABILITIES, never endpoint names.** A profile table that
  named classes would be a third routing table to keep in step with `models.yaml`, and it would
  break on every fleet whose classes are not spelled like the example's. Which finally made
  `tool_calling` and `structured_output` load-bearing rather than documentation — the same lesson as
  the vision ledger entry.
- **Two ranking rules are doctrine.** A real-cost endpoint sorts last under *every* preference, or
  "intent" becomes a back door around the whole spill doctrine and traffic leaves the machine on the
  ordinary path. An endpoint with no latency samples sorts as slow, or the resolver prefers the
  backend it knows least about *because* it knows least about it.
- **Resolution is not substitution.** An intent landing somewhere the caller never named is the job
  being done; only a later failover or spill is disclosed as a substitution. Conflating them would
  make `substituted: true` fire on every intent-routed call and mean nothing.
- **Substitution narrows, never widens.** A request may decline `degrade` or `spill`; it can never
  grant itself either, because the operator grants and the caller may only refuse. Declining is a
  defer, not an error.
- **Enrichment never enters an OpenAI body.** The OpenAI door gains four `X-Roadstead-*` response
  headers and nothing else — and those carry only what admission had settled, because a streaming
  response's headers precede a failover the enriched `done` frame can still report.
- **`POST /v1/submit` is removed**, not deprecated. `docs/api.md` §1.9 is the field-by-field map;
  `CHANGELOG.md` carries the reason.
- **`roadstead.client` ships with it** — the SDK the enriched API exists to be consumed by, and the
  migration `docs/api.md` §2.2 asked for: it classifies errors on the `code` and keeps the legacy
  marker substrings only as a fallback. It imports nothing from the server, by AST-enforced rule, so
  a consumer sending an HTTP request does not install Starlette — and so the contract literals it
  holds are checked against `docs/api.md` rather than against the server, which would agree with
  itself.

**Still open in C:** the profile table is built in only — `models.yaml` has no `intents:` section
yet, so a deployment cannot add or override a profile without editing the package. Intent resolution
also cannot yet express a *negative* constraint ("anything but this endpoint"), which is what a
caller working around one bad model actually wants.

### D · Spill, token management and costing

**Landed 2026-09-01.** `spend.py` is the fourth pure-computation module: prices, per-caller accounting,
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

**Landed 2026-09-01.** `/rs/v1/admin/*` — read first, control second, and it went last because there
was nothing worth managing until keys, quotas, budgets and providers all existed. The UI on top of it
is **G**, below.

**The thesis.** An operator's questions are not the caller's questions one level up. A caller asks
*what can serve me, how long, what does it cost*; an operator asks something the package could not
answer at all: **"what did I write that is not in force?"** Every expensive failure in
`docs/ledger.md` lives in a gap between two sources that agree most of the time — a dropped `policy:`
key, a declared slot count discovery overwrote (or did not), a fingerprint the backend stopped
matching, an `api_key_env` nobody exported. So the views report the declared value beside the one in
force, and a surface that merely echoed `models.yaml` back would be a worse `cat`.

- **On `/rs/v1/admin/*`, not `/v1/admin/*`.** `/v1` is versioned by OpenAI and management is the
  surface most likely to need its own second version. The four control routes that predate this are
  served at **both** spellings — same handler, same gate — so no consumer breaks and an operator has
  one prefix rather than two.
- **`hooks.config_notice` is a fourth reporting seam and the first that reports IN.** The three
  allowlist parsers still drop unknown keys and still log, but a notice is now *retained* and read
  back by `GET /rs/v1/admin/config`. A startup WARNING is read by whoever booted the process; "why
  does this knob do nothing" is asked by somebody else, a week later.
- **Enrolment and revocation without a restart** — the half of **B** that was left open. A generated
  secret is returned once and never stored; a plaintext one is never accepted. Revocation is never
  refused on provenance grounds, because "that key came from the environment, use a different tool"
  is a correctness argument answered, in the moment, by a breach.
- **🚨 Revoking the LAST key changes the identity regime back, and is disclosed.** An empty registry
  is not in play (§1.5 rule 2), so the revoked key is *ignored* rather than refused and the address
  decides. Found by the e2e journey, not by reasoning; fixed by disclosing rather than by narrowing
  rule 2, which would 401 exactly the deployment that just emptied its registry deliberately.
- **Quota edits reach the LIVE budget**, and move the rate rather than the balance — clearing a
  deficit on a config edit would hand a fresh allowance to precisely the caller being reweighted
  because it consumes too much. **§1.6 is inherited at the write boundary**: no field can express a
  rejection, so the plane cannot mint a policy admission refuses to honour.
- **A runtime edit never rewrites a config file.** A JSON overlay is layered over the files at
  startup — enrolments and overrides on top, revocations last. Comments survive, the operator's
  editor is not raced, and "who changed this" stays answerable. **An unpersistable change still takes
  effect and says so**, which is why §3 mints no error code for it.
- Twenty mutations, every guard observed going red — two of which failed the first pass and were
  real: the doc pin was satisfied by a route named in *prose* rather than in the table, and one
  mutation was a no-op that had to be rewritten before it proved anything.

### G · The management UI

**Landed 2026-09-01.** `GET /rs/v1/admin/ui` — the face on E's plane, and the roadmap's "manage and
monitor as a standalone product" line. Both binding constraints held: **one static HTML file** with
vanilla JS, no bundler and no external reference of any kind (the dependency list is still six, and a
test asserts it), and the only server-side work is an asset read that goes **off-loop**.

- **Off by default.** Unset `ROADSTEAD_ADMIN_UI` and the route does not EXIST — absence rather than
  refusal, the same posture as `ROADSTEAD_TRUSTED_PROXIES` and `ROADSTEAD_REQUIRE_API_KEY`.
- **🚨 The CSRF question is answered by not creating it.** Of the three options the workstream had —
  a session cookie minted from a key, HTTP Basic, or a separate UI credential — the third is what
  `identity.py` exists to prevent, and the first makes a cookie authenticate mutating routes, which
  would be the first real CSRF surface in this codebase. Basic was chosen, **carrying an API key in
  the password half**: a browser can attach it to a navigation (a bearer token cannot be) and
  `EventSource` can attach it to a stream (a header cannot be set at all), while minting a *password*
  would have been a second credential kind with its own store, rotation and revocation beside a
  registry that already does all three. No cookie is minted, so no CSRF defence is needed.
- **The door refuses 401 + `WWW-Authenticate` where the plane answers 403.** The split is right for
  an API client and a dead end for a browser, which shows no password box for a 403.
- **🚨 Every field the page reads is pinned against a real response.** A UI has no compiler, no schema
  and no types: rename a field and one cell renders "—" forever while the page looks healthy. Reads
  go through `pick(obj, "a.b.c")` so the paths are extractable and each is walked against a response
  a real service produced. It found a live bug on the first run — and *rendering the page in a
  browser* found two more that no test could have: a one-level `flat()` stringifying nested nodes,
  and boolean attributes rendered empty so no declared-vs-in-force pair ever collapsed.
- **`GET /v1/stream` is admin-gated**, which it was not. A frame there names the caller, endpoint,
  tokens and timing of every call served — the live form of `/rs/v1/admin/callers`. Aliased at
  `/rs/v1/admin/stream` for the reason above.
- Sixteen mutations, every guard observed going red.

**Still open in G.** `admin` is one scope, so browsing the fleet and revoking a key are the same
privilege — a read-only scope is the obvious next split, and it changes what a key *means*, which is
why it did not land with the UI. There is also no audit trail: the plane shows what changed, not who
changed it or when (open in E too), and the UI makes that gap easier to reach. Nothing here is
covered by a browser-driven test; the guards are contract pins plus a rendered walkthrough by hand.

**Still open in E.** Providers and endpoints are **read-only**: adding a backend
is still a `models.yaml` edit and a restart, deliberately, because an endpoint is a routing-table
entry that discovery, health and the DRR denominator all key on, and hot-adding one is a much larger
question than hot-adding a key. Nothing here is audited: an operator can see what changed but not
*who* changed it or *when*, and the overlay is the obvious place for that.

### F · Hardening — the concurrency invariant, guarded

Promoted from a cross-cutting note to a named workstream on 2026-09-01, because "cross-cutting"
turned out to mean "nobody's", and it had been carried unchanged through four workstreams while the
state it protects grew by two modules.

**The case.** `CLAUDE.md` opens by naming the single most dangerous thing in this repo — single
loop, no locks on scheduler / budget / cache / spend state — and, until now, said in the same
breath that nothing in the suite guarded it. A violation does not raise. It interleaves, and the
symptom is a DRR budget that drifts or a request served twice, weeks later and nowhere near the
commit that caused it. Sustained concurrent load is the only thing that surfaces one.

**Landed 2026-09-01 — the core.** Every guard below was observed going red by mutating the code it
watches, which for a guard is not a nicety: an assertion about a failure that has never happened is
indistinguishable from one that cannot fire.

- **`tests/loop_affinity.py`** arms the invariant: it wraps the mutating methods of every
  single-loop object and records `threading.get_ident()` per call, so "one thread mutates this" is
  a checkable claim rather than a convention. Deliberately not clever — no `settrace`, no import
  hooks, which would make a soak measure the instrumentation. It **records rather than raising in
  place**, because a raise inside a mutation would unwind into one of the hot path's fail-open
  guards and be swallowed, which is how a guard reads green while detecting nothing.
- **`tests/e2e/test_soak.py`** runs sustained overlapping load — both north faces, both response
  modes, three bands, intents and pins — and asserts five things a race would break: thread
  affinity, slot conservation, DRR budget conservation, spend-ledger conservation, and that every
  request got *an* answer. It also asserts the workload actually overlapped, because a concurrency
  test that runs no concurrency passes trivially.
- **The guard is observed going red from inside the suite, permanently.** One test does the
  forbidden thing on purpose — mutates DRR budget state from a second thread — and requires the
  recorder to say so. Without it, a recorder that silently stopped recording would leave the soak
  green forever, and every other assertion in the file would be worth nothing.
- **`tools/soak.py`** is the unbounded version — minutes of load, growth and WAL behaviour — off
  the default path because it is an experiment rather than an assertion.
- **`tests/test_pure_modules.py`** closes a second unguarded claim found on the way past.
  `CLAUDE.md` calls the pure-computation modules the crown jewels and says "keep them that way";
  nothing checked it. What purity buys is that every scheduling, costing, deadline, spend and
  routing decision is testable against a fleet that does not exist — and the first `httpx` import
  into one would take that away permanently while the suite stayed green. Same class of failure as
  the vision capability nothing read: a property true when written, with no mechanism to notice the
  commit that ends it.

**Still open in F.** The soak runs against `roadstead.testing`, so it exercises concurrency and not
*duration*: nothing yet watches memory growth, WAL size or connection-pool behaviour over hours,
and nothing covers a **remote provider over a real network**, where the failure modes are latency
variance and partial responses rather than contention. The shutdown budgets (`CLAUDE.md`: 78.25s
measured against a 48s budget that reads as though it covers everything) are measured and
documented but still not *bounded* — uvicorn never bounds the lifespan shutdown at all.

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

**~~The `/v1/submit` envelope is going away.~~ Gone, 2026-09-01**, with Workstream C. Recorded in
`CHANGELOG.md`, mapped field by field in `docs/api.md` §1.9. The OpenAI surface is unaffected, as
planned.

**`models.yaml` grows a provider dimension** and stops being a description of one fleet's hardware.

**`docs/api.md` is executable** — six test files read it back and fail when code and document
disagree. Every change above lands with its contract, or the suite says so.

**Not everything from the origin generalises.** The retired `creative` endpoint name, the
fleet-specific roles, the `degrade_ok` spelling: these are one deployment's vocabulary. Keep the
mechanism, drop the vocabulary — but check `git log` first, because several of them are load-bearing
in ways the name does not suggest (`history.md`, "Things that will mislead you").

---

## Still open

- **Where the intent vocabulary lives.** Profiles are built into `intent.py` today. A deployment
  whose fleet has a capability the built-ins do not name has to edit the package — which is the
  `models.yaml` argument again, one layer up.
- ~~**Multi-tenancy depth.**~~ **Settled 2026-09-01 with E: keys stay flat, because the shape that
  was wanted already exists.** The quota holder, the DRR fair share and the spend cap are keyed on
  the `agent_id`, never on the key — so several keys naming one `agent_id` give a team one budget
  with per-key revocation and per-key priority, which is what nesting was for. `GET
  /rs/v1/admin/keys` groups by `agent_id` so the structure is visible rather than inferable. A real
  team → key hierarchy would only start to earn its keep with per-team *aggregate* caps distinct from
  the per-caller ones, and nothing yet needs that.
- **Where cost truth lives.** Provider-reported spend vs. our own token accounting; they will
  disagree, and one of them has to be authoritative for threshold decisions.
- **Streaming through a remote provider** under a computed deadline — the soft-deadline extension
  logic assumes decode progress is observable, which is provider-dependent.
- **Whether the timeout model can learn per-provider** for remote backends whose latency we do not
  control, or whether those get a different deadline strategy entirely.
