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
— Workstream E) and **the UI with it** (Workstream G), with both constraints held rather than
relaxed: it does not violate the concurrency invariant (`CLAUDE.md` — heavy reads go off-loop,
mutations stay on it, and the only server-side work is an asset read that goes off-loop), and it
drags no frontend toolchain into a package whose dependency list is deliberately short — one static
file, vanilla JS, no bundler, and the dependency list is still six. **H** then split the scope it
needed (`admin_readonly`) and added the trail it made everyone want (`/rs/v1/admin/audit`).

---

## Workstreams

Not phases. Several can run concurrently; the dependencies between them are what matters.

### A · Provider abstraction — *the foundation*

**Landed 2026-08-31** (`321bd82`, *"Extract the provider interface out of backend.py"*). The case
for it, which every workstream after this one drew on:
extract a provider interface out of `backend.py` (llama.cpp/vLLM branching inline) and formalise the
capability descriptor. Nothing else on this list was buildable first — OpenRouter needed it, spill
needed it, per-provider costing needed it, and enriched model information is largely a readout of it.

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

**Landed 2026-08-31 — the catalog.** `models.yaml` is now `providers:` + `endpoints:`: connection and
engine on one side, capacity and policy on the other, each endpoint naming its provider. A local
provider hosts one endpoint; a remote one hosts many, which is what makes the split earn its keep.
Shipped as a generic example on RFC 5737 addresses — the same piece of work as scrub item **S2**,
done together as planned rather than twice.

**~~Per-provider costing has no reader.~~ Closed 2026-09-01 by D.** `publishes_token_costs` was
descriptor-shaped and unread; OpenRouter's catalogue prices now arrive on the same discovery pass
that reads the context ceiling.

**Still open in A:** a **second remote provider** would be the real test of the abstraction. One of
each is enough to find the `host:port` assumption; it is not enough to know which of OpenRouter's
shapes are *OpenRouter's* and which are *remote's*.

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

**~~Still open in C.~~ Closed 2026-09-01** — both halves, and they turned out to be one question
(*who owns the words a caller may use*) with one line running through it.

- **`models.yaml` grows an `intents:` section**, layered over the built-ins rather than replacing
  them: a file that defines one profile has said nothing about the other nine, and reading it as a
  whole-table swap would silently empty a vocabulary `GET /rs/v1/models` publishes and callers code
  against. Overriding by name is how a fleet says its `reasoning` means something particular, and it
  is **disclosed** — every published profile carries `source: builtin | models.yaml`, because a
  caller reading our documentation for a word this fleet redefined has no other way to notice.
- 🚨 **A profile still cannot name an endpoint, and now there is a config parser that must not learn
  how.** There is no field for it, guarded from both ends — the allowlist and `Profile`'s own fields.
- **An unusable stanza is refused rather than offered.** A profile requiring a capability nothing can
  declare would match nothing on every request, and the caller would read "no endpoint satisfies
  requires=[…]" — a sentence about the fleet, for a fault in a config file. Refused, they get
  "unknown intent", which points at the vocabulary. A refused *override* does not leave the built-in
  standing under the operator's spelling, which would be a declared-vs-in-force gap we created.
- **`exclude` is the negative constraint**, and it is expressible after all — because the vocabulary
  it needs is not the profile vocabulary. The line is between **config and request**, not between
  positive and negative: a profile is shared, published, operator-written vocabulary and must stay in
  capabilities; an intent is one caller's words about one call, and `pin` already names an endpoint
  there. `exclude` says the same kind of thing in the other direction and adds no table.
- 🚨 **An `exclude` naming an endpoint this fleet does not have is a 404, not a warning.** The
  tempting reading — that it is satisfied trivially, since the endpoint it forbids is absent —
  assumes the one thing the proxy cannot check. A name resolving to nothing is either "not here" or
  "here, under a spelling you got wrong", and from inside they are the same bytes; serving the second
  sends the request to precisely the endpoint the exclusion existed to avoid and reports success.
  Same shape as the `finish_reason` repair that became a silencer. `model` and `exclude` naming the
  same endpoint is a 400.
- `tests/test_intent_config.py`. Sixteen mutations, every guard observed going red by assertion —
  one first failed by raising an `IntentError` from `intent.py` rather than by asserting, which is
  half a guard, and was rewritten. **Running it found the bug the suite could not**: the refusal
  echoed the normalized name, so a caller who wrote `"tierX"` was told `'tierx'` — `pin_as_written`'s
  reason for existing, missed in the other direction.

### D · Spill, token management and costing

**Landed 2026-09-01.** `spend.py` joined the pure-computation modules — the fourth at the time, of
**six** today — with prices, per-caller accounting, and the threshold. `Scheduler._admit` returns `DISPATCH` / `SPILL` / `DEFER` — one decision, three
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

**~~The live ledger resets on restart.~~ Fixed 2026-09-01, and it was a correctness bug rather than
the acceptable simplification this used to call it.** The old wording — "right for what it governs,
whether a caller is degraded *now*" — quietly assumed *now* was the same length as the window the
threshold reads. It is not: `daily_spend_usd` is a **day** and the process was measuring an
**uptime**, so a deploy at noon handed every caller its whole allowance a second time and the more a
fleet ships the less its spend cap means. `startup` now replays today's rows out of
`proxy_completions`, exactly as DRR balances have been restored since Phase 3.4 — the same argument
about the other per-caller quantity, missed when `spend.py` landed.

Two choices inside that are worth knowing. It is **re-priced at today's prices**, not at each call's
price at the time: right for a threshold that answers "is this caller degraded now", wrong for a bill
— and anything billable reads `proxy_completions`, which keeps the **tokens**. And the seed **fails
open**, loudly: refusing to boot because we cannot prove a caller crossed a threshold whose whole
consequence is one priority band would let a spend cap take the proxy down, which is the same
argument that stops it taking a *caller* down.

**Still open in D — and the two halves need different things.**

- **Reconciliation against a provider's invoice cannot be built here honestly.** It needs a real
  account with real billing, and this repo serves a fake backend; a reconciler written against an
  invented invoice would agree with itself and prove nothing, which is the same failure as a client
  that read the server's own constants. What *can* be settled in advance is which side is
  authoritative, and it is now settled by construction: **`proxy_completions` keeps tokens, never
  money**, so re-pricing is always possible and the ledger is a derived view rather than a second
  record to reconcile. When a provider's numbers do arrive they disagree with ours about *price*, not
  about *usage*, which is a much smaller argument.
- **Nothing spills on cost, and that stays deliberate.** A cheaper remote endpoint must never be
  preferred to an expensive local one, or remote capacity stops being overflow — the same doctrine
  that makes a real-cost endpoint sort last under every intent preference. The real open question is
  narrower than "spill on cost": with **several** remote providers, which one does an overflow go to?
  That is ranking among remotes *after* the admission decision has already said SPILL, and it never
  touches the local-first rule.

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

**~~Still open in G: one scope, and no audit trail.~~ Both closed 2026-09-01 by H** — and the UI is
what made them urgent, exactly as predicted here.

**~~Still open in G: a browser-driven test.~~ ~~DECIDED 2026-09-01: no.~~ 🚨 That decision was
made twice on two different reasons and BOTH were wrong. Corrected 2026-09-01: nothing in this
repo's conventions forbids one. It is open on cost, which is a different question and a much
weaker one.**

**Reason one, retired: the dependency argument.** "It must not drag a frontend toolchain into a
package whose dependency list is six" is about the **runtime** list —
`tests/test_admin_ui.py::test_the_dependency_list_did_not_grow_a_frontend` reads
`project.dependencies` and asserts exactly those six, plus the absence of `package.json` /
`node_modules` / a bundler config. A test-time browser driver is a **dev** dependency and violates
none of it.

**Reason two, also retired: the suite's promise.** The replacement argument was `pip install -e
'.[dev]'` then `pytest` — *no fleet, no network, no backends* — broken by a browser binary fetched
at install time. 🚨 **It does not survive reading `pyproject.toml`, which is the same failure as
reason one: an argument from a document nobody re-read.** That file already carries
`addopts = "-m 'not wire_fidelity'"` and a registered `wire_fidelity` marker, for tests needing a
**real inference engine** — precisely the class of test that cannot run in the default environment.
The repo already answered this question and answered it with a marker.

So the promise constrains **the default `pytest` run and the `dev` extra**, not the repository. A
`browser` marker deselected by default, with the driver in its own extra rather than in `dev`,
leaves `pip install -e '.[dev]' && pytest` doing exactly what it does today: no browser fetched, no
browser test run, no new flake, offline still fine. The seven `wire_fidelity` tests are the proof by
construction — they sit in this repo right now under exactly that arrangement.

🚨 **And this correction was RUN rather than reasoned, because reasoning from the document is what
produced two wrong answers already.** A `browser`-marked test added temporarily on 2026-09-01, with
`addopts = "-m 'not wire_fidelity and not browser'"`: deselected by the default run, selected by
`-m browser`, and the existing `wire_fidelity` deselection unaffected (8 deselected, 7 + 1). Then
reverted. The mechanism does what the correction claims.

**What is actually left is a cost judgement, and it should be argued as one.** A browser test is
permitted; it is not obviously *worth it*. Against: a driver to keep current, a second CI lane
nobody runs locally, and a page whose every control is one request and one re-render. For: the gap
below is real, has produced two live bugs, and both were found by a human doing by hand what the
test would do every commit.

🚨 **The lesson generalises past this decision.** Two arguments in a row cited a constraint that the
file defining it does not contain, and both stood because they *sounded* like this codebase's
values. `docs/ledger.md` is full of this shape. Cite the mechanism and re-read it, or do not cite
it.

**What is already covered**, measured against the running page on 2026-09-01: **163** `pick()` paths,
each walked against a response a real service produced; **14** `guarded(...)` write controls, each
found from the HTTP method it sends rather than from a list; and the scope proven known before the
first paint.

**What is NOT covered, and is accepted as a known gap:** DOM-level faults in the render helpers — a
node stringified into a cell as `[object HTMLSpanElement]`, a boolean attribute rendered empty so no
declared-vs-in-force pair collapses. Both of those actually happened, and both were found by a human
rendering the page. **The compensating discipline is the one already in use:** when rendering finds
something, leave behind a guard a *source read* can make. That has worked twice (H).

**~~The gap~~ — closed by hand on 2026-09-01, without a driver and without a download.** All seven
tabs rendered against a running server, a write control clicked (the runtime-flag toggle: banner,
state flip, button relabel, and an audit row naming the credential), and the read-only view rendered
for the first time. Cost: zero bytes downloaded, because the browser was already installed. 🚨 **That
is the distinction the "browser test" argument kept collapsing** — validating the page ONCE needs no
driver at all; only a permanent CI lane does, and only that lane costs a browser binary.

**It found one real defect and the guard for it found two more.** As a read-only operator the
Maintenance form's three inputs accepted typing while their Record button was correctly disabled — a
form inviting an operator to fill in something they could never send. Not a security hole (the
button is dead and the plane 403s anyway); the same lie as a cell rendering "—" forever. The
compensating discipline applied as usual: `test_every_input_a_write_action_reads_is_scope_guarded_too`
keys on the real relationship — an input whose id is read inside a write action's body must be
`guarded(...)` — because the existing guard only sees elements carrying an `onclick`, which these do
not. It immediately found `#nk-agent` and `#rk-overlap` on a tab nobody had rendered read-only.

🚨 **And it corrected a wrong reading of my own.** The Record button *looked* enabled in a
screenshot and the DOM said `disabled: true`; the computed styles were identical to the flag
buttons. Checking the DOM rather than the pixels is what turned a false report into a real one.

**~~Still not covered: the live feed under real traffic.~~ CLOSED by hand on 2026-09-01.** The
obstacle was reaching an authenticated `EventSource` at all: it cannot carry a header, so it
authenticates from the browser's credential cache, which the fetch-injection technique used
previously bypasses. Credentials in the URL are refused by the driver and a plain navigation raises a
blocking Basic-auth modal. 🚨 **`XMLHttpRequest.open(method, url, false, user, password)` populates
that cache and `fetch` has no equivalent** — one XHR against the UI path from an unauthenticated
page on the same origin (`/health`), then navigate, and the page comes up with the stream
authenticated and no modal. Worth writing down: it is the only way in, and it is not obvious.

**Result: the feed works, and rendering it found nothing.** 2,566 calls through `roadstead.client`
from two enrolled callers across three intents, zero errors. The feed showed `live`, held its 60-row
cap, and every one of `feedRow`'s reads — `agent`, `endpoint`, `priority`, `duration_s`,
`queue_wait_ms`, `status` — rendered from real `call.completed` frames. All seven tabs were then
walked under sustained traffic: **no `[object HTMLSpanElement]`, no `undefined`, no `NaN`**, and every
`—` traced to a genuinely null field (`daily_spend_usd` and `requests_per_minute` on callers with no
threshold configured). Callers rendered live DRR balances, and `avoided_usd` populated with
`spent_usd` at zero — §1.6's two-kinds-of-money rule, correct on the page.

🚨 **The source guard was already telling the truth.** `feedRow` reads an SSE frame, not a REST
response, so it needed its own coverage — and it has it:
`test_admin_ui.py` pulls the `call.completed` payload out of the `sse.publish` call site by AST and
walks the feed's paths against that. Rendering confirmed the guard rather than correcting it, which
is the first time that has happened and is the outcome the discipline is supposed to produce.

**Weigh it against** the fact that every control is one request and one re-render, which is the
regime where a source-level guard can stand in for a rendered one. That regime ends if the page
grows client-side validation or state that survives a navigation.

### 🚨 DECIDED on cost, 2026-09-01: no permanent browser lane

Third time of asking, and the first time on the right question — the two earlier answers were
retracted because they argued permission, which was never in doubt.

**What the argument was missing was a marginal rate, and this session supplies one.** The case *for*
rests on two live bugs found by rendering. Both were found on the FIRST render of newly-written
code, and both left behind a source-level guard. The number that matters is not "rendering has found
bugs" but "rendering finds bugs the guards now miss" — and the page was rendered again on
2026-09-01, under sustained real traffic, across all seven tabs, with the feed live: **zero
DOM-level faults.** One render is a small sample, but it is the only evidence anyone has about the
marginal rate, and it is 0.

**Against, priced honestly.** A driver and a browser binary in their own extra — deselected by
default, so `pip install -e '.[dev]' && pytest` is untouched, exactly as `wire_fidelity` proves. The
real cost is not the download: it is a second lane nobody runs locally. `wire_fidelity` is the
precedent and its own README records the answer — executed **once**, on 2026-08-31. An opt-in lane
gets run when somebody remembers, and a guard that runs when somebody remembers is a guard whose
value is set by memory rather than by CI.

**So: no.** The discipline stays, and it gains the one piece it was missing — a written trigger,
because "re-render when something changes" with no statement of *what* is how the two earlier
retracted answers happened. 🚨 **Re-render by hand when: (a) a tab or view is added, (b) a render
helper (`el`, `pick`, `tag`, `fmt`, `guarded`) changes, or (c) the page grows client-side state that
survives a navigation** — (c) being the stated end of the regime where a source guard substitutes,
so it is also the trigger to reopen this decision entirely.

**The distinction that kept collapsing, stated once more because it is what makes this affordable:**
validating the page needs no driver and no download. Rendering it by hand has now been done twice,
cost zero bytes both times, and was productive both times — once finding two bugs, once establishing
that the guards left behind by the first are holding.

**~~Nothing here is audited.~~ Closed 2026-09-01 by H**, in the overlay, which is where this
predicted it would go.

**Still open in E.** Providers and endpoints are **read-only**: adding a backend
is still a `models.yaml` edit and a restart, deliberately, because an endpoint is a routing-table
entry that discovery, health and the DRR denominator all key on, and hot-adding one is a much larger
question than hot-adding a key. Leave it unless there is a reason — and write the reason down before
the code.

### J · A credential and an endpoint, from the operator's face

**J1 LANDED 2026-09-01.** The Providers tab was the configuration surface and had zero controls on
it; it now has two, and they are the two that turn "a dashboard" into "a way to bring a backend into
service". Validated by clicking, not by asserting: the refusal, the credential form, the promotion,
and discovery running against the real OpenRouter API and returning **published prices**
(`source: "provider"`, `real: true`) — which is the proof the promoted endpoint is genuinely wired
into discovery rather than merely recorded.

🚨 **Running it found a latent bug that could not previously fire.** `enriched.facts()` computed
`routed` from the CATALOG entry while its own comment said the opposite — "an endpoint present in
the routing table is ROUTED whatever the catalog says". The two could never disagree, because
`config.endpoints` is built from `cat.routed()` at startup, so the wrong branch was unreachable and
the comment went unchallenged for the life of the file. J1 makes them disagree on purpose, and the
result was a promoted endpoint reported unrouted on `/rs/v1/models` while the admin plane called it
live — and `intent.py` filters on that field, so a pin to it 404'd. `config.endpoints` IS the
routing table; there was never anything else to ask.

**Known edge, not fixed:** between promotion and the first discovery pass, a remote endpoint carries
the imputed avoided-cost price rather than its published one, so a call served in that window books
as `avoided_usd` rather than `spent_usd`. Self-corrects on the next poll and the window is seconds,
but it is the two-kinds-of-money inversion `spend.py` exists to prevent, and it is now reachable
where it was not before. Workstream D territory.

**Still open in J: J2**, and the design note below is unchanged by J1 — a promotion route does not
become a creation route by adding fields.

### J · A credential and an endpoint, from the operator's face

**Opened 2026-09-01, with the reason E asked for.** E parked provider and endpoint writes —
*"adding a backend is still a `models.yaml` edit and a restart, deliberately … Leave it unless there
is a reason — and write the reason down before the code."* The reason is now on the table: the
operator wants to add a remote provider's credential and bring its endpoints into service **from the
UI**, and to add local backends the same way. This section is the writing-down; it precedes the code
deliberately.

**What exists today.** `/rs/v1/admin/providers` is **GET only** — there is no write route for a
provider or an endpoint anywhere on the plane. The catalog states the sanctioned workflow in its own
comment on the `openrouter` stanza: *"Its endpoints are `planned` below … Flip one to `active` once
the key is in the environment."* Two steps, both a file-or-environment edit plus a restart.

🚨 **The two asks are NOT the same size, and conflating them is how this gets built wrong.**

#### J1 · Set a provider credential, and promote an endpoint that is already declared

Small, and it is what unblocks live OpenRouter.

**The credential is WRITE-ONLY, and the emit doctrine does not move.** `management.py` opens by
saying a management surface never emits a credential — not the key, not the digest, never the value
behind an `api_key_env`, only its name and whether it resolved. Accepting one is a different verb
from emitting one, and the plane already accepts credentials when it enrols a key. What is new is
that this is an **outbound** secret rather than an inbound identity.

🚨 **The provider reads `os.environ` at CALL time, not at startup** (`openrouter._api_key`), and
that single fact is what makes J1 small: setting the process environment takes effect on the very
next request, with no restart and no reload path to build.

**So: process environment, deliberately NOT persisted.** The overlay holds key *digests* and has
never held a secret, and writing an outbound provider key into a JSON file on disk is a change of
posture that deserves to be decided on its own merits rather than arriving as a side effect of a
convenience. Not persisting is also the honest shape and the repo already has it twice: the
bootstrap admin key is *"NOT saved and a new one is minted on every restart"*, and a control action
that cannot be persisted still takes effect and says `persisted: false` with a reason. The response
here says the same thing in advance — this is in force now and will not survive a restart, put it in
the environment to make it durable. The audit record names the **variable**, never the value.

**Promotion is not hot-adding.** `planned` → `active` changes one thing: membership of
`catalog.routed()`. The stanza already declares provider, model, slots, context, floors,
capabilities and failover — everything discovery, health and the DRR denominator key on is already
in the file and already parsed. That is a far smaller claim than inventing a routing-table entry at
runtime, and it is the whole of what E was protecting.

🚨 **A promotion whose credential does not resolve is REFUSED.** This is the load-bearing rule.
The catalog says exactly why `planned` exists: *"a deployment that has not set $OPENROUTER_API_KEY
should not have an endpoint in its routing table that cannot serve."* Promoting without the key
would place precisely that into the routing table — from the surface whose entire purpose is
reporting the gap between what was written and what is in force. The ordering falls out of the
refusal rather than being imposed: set the credential, then promote, and a promotion that cannot
verify says which of the two is missing.

**Demotion (`active` → `planned`) is the paired verb** and is never refused, on the same argument
revocation is never refused on provenance grounds: taking capacity out of service is the safe
direction, and an operator who wants a remote endpoint to stop costing money should not have to
argue with the plane about it.

#### J2 · Create a provider or an endpoint that is not in the file — **LANDED 2026-09-01**

Verified through the UI on the deployment sandbox against a real llama-server: an endpoint typed into a form served a
request seconds later, and the catalog file never mentioned it. Three panes replaced seven tabs at
the same time — Overview (what is going on, and the verbs you reach for while watching),
Configuration (what you set up), Audit (what happened). The old set was organised by which API fed
it, which is the system's structure rather than the reader's question, and it produced a tab called
"The gap" that needed a commit message to explain.

🚨 **Running it found what 1877 passing tests could not: the page did not parse.** Two missing `)`
in a render function, and every test was green — they read the source as text (`pick()` paths,
`guarded()` calls, the dependency list) and none asked whether a browser could run it. The page
showed "loading…" and a `SyntaxError`. `test_the_page_script_has_balanced_brackets` now scans the
script with string and comment literals stripped; it is not a parser and does not pretend to be one,
but unbalanced brackets is precisely what hand-editing nested `el(...)` calls produces, and it needs
no dependency — a check requiring `node` would skip where node is absent, and a guard that skips is
a guard that passes.



**Opened for real 2026-09-01**, on an explicit ask: the operator wants to add models and backends
from the UI, not from a file. Three panes — Overview, Configuration, Audit — with Configuration
owning models, providers, keys and networks. Decisions taken with the ask:

- **Full create / edit / delete.** `models.yaml` becomes a seed, not the only way in.
- **The overlay stays the ONLY writer.** `models.yaml` is never rewritten. It keeps its comments,
  its hand-authored intent, and its property of not being destroyable by a bad UI save. The cost —
  two sources to read before you know the truth — is paid by the Configuration pane showing
  declared-beside-in-force, which is what that surface already exists to do.
- **API keys, not passwords.** The no-second-credential-kind decision from G stands.

🚨 **THE DESIGN DECISION: the overlay contributes catalog STANZAS, not a parallel model of an
endpoint.** `load_catalog` reads YAML into a `raw` dict and then coerces it — `_coerce_provider`,
`_coerce_endpoint`, alias-collision detection, capability checking, `policy:` passthrough, the
`intents:` layer. The overlay is merged into **`raw`, before coercion**, so a UI-created endpoint is
parsed, defaulted and validated by exactly the code that parses a file-authored one.

The alternative — a second representation of "an endpoint the API made" — is the shape this repo
keeps paying for: two things that describe the same object and agree most of the time. It would need
its own validation, its own defaults, its own capability vocabulary and its own duplicate-alias rule,
and each of those is a place to drift. There is one catalog format; the overlay writes it too.

**Merge semantics, and they follow the doctrine already in this file:**

- A stanza for a name the file also declares is a **partial, merged over** the file's — so changing
  `slots` does not require restating `capabilities`. This subsumes J1's status override exactly:
  `{"status": "active"}` was already a partial stanza and needs no format change.
- A stanza for a name the file does not have **stands alone** — that is creation.
- `null` is a **tombstone** — deletion. Applied after the file, so a later statement wins, which is
  the same rule that puts revoke after enrol.

🚨 **Validation is by CONSTRUCTION, not by a second validator: build the catalog you would install
and refuse the write if it complains.** `hooks.config_notice` is the file loader's reporting seam and
is deliberately non-fatal there (a typo must not stop a fleet booting) — but the same complaint
arriving from a *request* must be a 400, because there is an operator on the other end who can fix it
now. Same check, two consequences, chosen by who is asking.

**What must still refuse:**

- **Deleting an endpoint with work in flight** — J1's rule, unchanged: the request path reads
  `config.endpoints.get(...)` after dispatch. Pause drains; pause first.
- **Deleting a provider that endpoints still name** — refused, naming them. The alternative is a
  routing table pointing at a connection that does not exist.
- **Activating an endpoint whose provider needs a credential that is not set** — J1's rule.
- **Renaming** — not supported. A rename is a delete plus a create, and the DRR budget, the spend
  ledger and the timeout model are all keyed on the endpoint name; silently carrying that history to
  a new name, or silently dropping it, are both wrong and the operator should choose.

**What is expected to work and is not new:** discovery against a backend nobody has probed (the
health poller re-reads `config.endpoints` every cycle), and the scheduler meeting an endpoint it has
never seen (`_queues` is created lazily). Both were verified on a real llama.cpp in a container on
2026-09-01 — a declared `slots: 2` was corrected to `4` by `/props`.

The real hot-add, and the one E was actually pointing at. Everything J1 leans on is absent: nothing
is parsed, nothing is validated, and a half-configured entry in the routing table is a live failure
rather than a rejected write. It needs the catalog's validation reachable from a request, a
discovery pass for a backend nobody has probed, the DRR denominator and the health poller both
picking up a member that did not exist a moment ago, and an answer for what happens to in-flight
work if it is removed again. **Not attempted alongside J1**, and J1 does not prejudge it — a
promotion route does not become a creation route by adding fields.

**What stays true across both.** Never emit a credential. `admin_readonly` narrows and the split
comes from the HTTP method, so these routes inherit the gate by being mutating. Every mutating admin
route records to the audit trail, and the completeness guard is driven from `routes.py`. Mutate on
the loop, persist off it. An allowlist parser gets the key **and** its guard **and**
`hooks.config_notice`.

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

**~~The shutdown budgets are measured but not bounded.~~ Closed 2026-09-01** — every phase is named
and bounded, `SHUTDOWN_DEADLINE_S` is their sum rather than a literal, and the recommended container
stop-grace went **up** (90 → 108s) because 90 had been an observation of a fast tail rather than a
ceiling. See the shutdown-drain entry in `docs/changelog-archive.md` for why there is deliberately no
outer `wait_for`.

**Duration answered 2026-09-01, and the answer took three runs.** The soak had never been run long.

- **`rate.RateLedger.windows` grew without bound** — found before any soak ran, by reading. `prune`
  existed, its docstring said the maintenance tick called it, and nothing did, so both dicts behind
  the rate threshold accumulated one entry per caller-supplied `agent_id` ever seen. Fixed on wall
  time rather than the poller's monotonic clock (the trap one line above it), and the ledger is now
  armed in `loop_affinity` — it had been outside the guard entirely.
- **A 25-minute run then reported a large leak — in the instrument.** RSS climbed
  118MB → 1867MB (+72.8 MB/min, linear, no plateau) and the growth was `FakeBackend.requests`, an
  unbounded list holding a body and a header dict per request. 🚨 `tools/soak.py` exists to detect
  leaks by RSS slope and its headline number was dominated by its own test double. The recorder is
  bounded now, and reports what it drops.
- **The proxy itself does not leak.** A 900s run at 462 req/s rose to 115MB while the 300s metrics
  window filled and then sat at **123MB for the remaining 600s** across 416,000 requests, every one
  answered, single-threaded throughout. WAL stayed at ~4.5MB, so checkpointing keeps up.
- **The tool now separates warm-up from leak.** `RollingMetrics` retains 300s, so RSS *cannot* be
  flat before then and any run shorter than that reports a positive slope that is steady state being
  reached. It reads the window off the live object and prints the post-warm-up slope as the leak
  number, or refuses to give one and says why. A leak detector whose own warm-up looks like a leak is
  the same class of fault as the recorder above.

**Still open in F.** `queue.db` grows ~1KB per request and nothing prunes it — 660MB over 697,000
requests. That is disk rather than memory and it is the durable record, so retention is a policy
question rather than a bug, but no deployment has been told to answer it. `AdminOverlay.audit` has
still never been soaked for duration (bounded at 500, and the store is rewritten whole on every
control action). And nothing covers a **remote provider over a real network**, where the failure
modes are latency variance and partial responses rather than contention.

### H · A read-only admin scope, and the audit trail

**Landed 2026-09-01.** Both halves of what G left open, together, because they touch the same three
files and answer halves of one question: the read views report a **state**, and a state cannot say
who put it there; and an operator who wanted somebody to be able to *look* had to give them the
ability to change everything.

- **`admin_readonly` narrows and can never widen.** `admin: true, admin_readonly: true` reaches every
  `GET` on the plane and is refused a 403 that says why on everything else; on a key without `admin`
  it is a **400 at enrolment**, not a silent no-op, because an operator who wrote it believes they
  issued a safer credential than they have.
- **The read/write split comes from the HTTP METHOD**, in the one shared gate — not from a list of
  write routes, which is a second thing to keep in step with `routes.py` and fails silently and
  *widening* when it falls behind. All 11 mutating admin routes inherit it, both spellings.
- **A narrowing beats an overlapping grant.** `127.0.0.1=ops:admin:readonly` is read-only even though
  loopback is a built-in admin net. If the widest grant won, that line would silently be a full grant
  for every operator who wrote it.
- **`identity.py` decides, everything else renders.** `IdentityResolver.admin_denial` owns which of
  the three refusals applies, under an AST guard — a second place deciding what a scope permits is
  the `_remote_ip` that had to be removed from `management.py`, in a new costume.
- **`GET /rs/v1/admin/audit`, and every mutating route records** — flags, pause, resume and
  maintenance included, which touch no overlay state and would otherwise record nothing. A trail
  covering only some of them is worse than none, because a reader assumes completeness; the guard is
  driven from `routes.py` rather than from a list.
- **A record names the credential and never carries one.** `key_id`, never the key or the digest.
  Label *and* address always: a record with an address and no `key_id` means an address-derived admin
  made the change, which is meaningful and slightly alarming, and one collapsed actor string would
  hide which factor authorized it.
- **It reports its own limits as data** — `persisted`, `dropped`, `capacity`. An operator-facing
  change trail, not a security log of record, and it says so rather than presenting itself as
  complete while being neither durable nor unbounded.
- **Two of the three bugs were found by RUNNING the page**, with the suite green for all three: a
  control-route record that only reached disk on a *later* overlay write (so the entry an operator
  looks for after an incident is the one a restart lost, and a drain is taken right before the
  restart that drops the record of it); the Fleet view's Pause/Resume being the one write control not
  scope-guarded; and `render()` racing `primeChrome()`, so the **first paint had no scope** and every
  write control was live for a read-only operator until the 30-second heartbeat. The first paint is
  the one somebody clicks. Two now have source-level guards, which is the pattern.

### I · Key lifecycle, and the abuse control DRR is not

**Landed 2026-09-01.** What was left of **B**, and the oldest unclosed thing in the repo: `created_at`
was written, surfaced in two views, and read for no decision.

- **Expiry** — `expires_at` / `expires_in_s`. No expiry means never expires, so an existing registry
  is unchanged. An expired key is a `401 invalid_api_key` with **its own sentence and the same
  code**: a caller can act on no distinction between "expired" and "unknown", the operator reading
  the log can. The store holds the absolute instant, so a restart never extends a key.
- **Rotation** — `POST /rs/v1/admin/keys/{key_id}/rotate`, one action, because the manual version is
  two calls in an order that matters and **both orders are wrong**. `overlap_s` defaults to 0; with
  an overlap the predecessor gets an expiry rather than a tombstone, so it survives a restart a timer
  would not, and 🚨 an overlap may only ever **shorten** a predecessor's life. If the successor cannot
  be minted the predecessor is untouched.
- **Per-key `bind` CIDRs** are an additional constraint and never a widening. Unparseable is a 400 at
  enrolment and matches nothing at resolution — a narrowing that failed open would be worse than
  none. Checked against the *resolved* address, so it is only as trustworthy as
  `ROADSTEAD_TRUSTED_PROXIES`, and `GET /rs/v1/admin/keys` reports `checked_against`.
- **`requests_per_minute`, and no 429.** DRR is fairness *under contention*, so a caller alone on a
  quiet fleet is unthrottled by design — correct for fairness, and exactly why it does not bound a
  runaway. Crossing this costs what crossing `daily_spend_usd` costs: one band and paid spill, never
  local capacity, and no code. A rate *limit* would be a fourth spelling of "no". `rate.py` is the
  **sixth** pure-computation module.
- 🚨 **Degradations do not stack.** Over both thresholds is **one** band, not two, enforced in
  `ProxyState.effective_priority`, which consults at most one standing — so a third threshold cannot
  compose by accident. Each crossing reports separately, because one means "look at the bill" and the
  other means "look for a loop".
- **Three more found by running it**, and the third is the interesting one: `overlap_s` *extended* a
  predecessor past its own expiry, which is a rotation quietly lengthening a credential. Truncated,
  and said out loud. The other two are disclosures rather than changes — a successor inherits policy
  but not expiry, and the rotate response now reads back from the registry rather than being
  assembled from the predecessor's row.

### Cross-cutting · The scrub

Tracked privately in the origin monorepo; the half a contributor needs is the scrub rule in
`CONTRIBUTING.md`. Gates the open-source goal. **S2 is done** (2026-08-31), together with
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

**S5 is done** (2026-09-01), and its topology half now runs on every commit as
`tests/test_scrub_sweep.py`. 🚨 **Two patterns, not one.** The topology grep could never have found a
person's name typed into an example prompt, and S4 found exactly that, so the identifier half is a
human pass — there is no fixed string to key on.

**S6 — the history rewrite — is done** (2026-09-01), over all 321 commits. Two corrections to what
this section used to say about it, both found by doing it:

- 🚨 **`--replace-text` is blobs only.** The plan's one-liner would have left all 321 commit
  messages untouched, and the messages are the *richer* surface — 282 of them were written inside
  the origin monorepo. `--replace-message` takes the same file.
- 🚨 **Every SHA in this repository changed**, this document's own `455e736` included. The citations
  were translated through filter-repo's `commit-map` and now carry a date and subject too, so a
  future rewrite cannot orphan them silently. Two SHAs in `docs/evaluation.md` turned out never to
  have been commits here at all — they cite the origin monorepo.

**Commit metadata took a second pass** the same day: `--replace-text` reaches blobs, `--replace-
message` reaches messages, and *neither* reaches author or committer identity. A `--mailmap` run
normalised all 322 commits onto one public address — and turned up a second private hostname nobody
had flagged, on 103 commits. 🚨 **A sweep over tracked files cannot see the author line**, which is
why that one survived every check until somebody looked directly at it.

⚠️ **What remains is a visibility decision, and it is not this plan's to make.**

---

## Consequences to plan for

**~~The `/v1/submit` envelope is going away.~~ Gone, 2026-09-01**, with Workstream C. Recorded in
`CHANGELOG.md`, mapped field by field in `docs/api.md` §1.9. The OpenAI surface is unaffected, as
planned.

**`models.yaml` grows a provider dimension** and stops being a description of one fleet's hardware.

**`docs/api.md` is executable** — a score of test files read it back and fail when code and document
disagree: §1.6's admission table, §2.1's codes (pinned from the server *and* the client side, and
again from `__main__`'s shutdown envelope), §3's route table (parsed from the table rows), §3.6's
analytics schemas, §1.5's trusted-proxy rule, §1.7.1's intent fields. Every change above lands with
its contract, or the suite says so. 🚨 The route-table sweep sets `ROADSTEAD_ADMIN_UI=1`, because two
routes only exist when it is — any future env-gated route needs the same treatment.

⚠️ Deliberately not a *number*: it was "six" for long enough to be wrong by a factor of three, and
was corrected to 19 on 2026-09-01 and outdated by two within the same session. The list of what is
pinned is the useful half and does not rot; a count in prose that nothing checks always does.

**Not everything from the origin generalises.** The retired `creative` endpoint name, the
fleet-specific roles, the `degrade_ok` spelling: these are one deployment's vocabulary. Keep the
mechanism, drop the vocabulary — but check `git log` first, because several of them are load-bearing
in ways the name does not suggest (`history.md`, "Things that will mislead you").

---

## Still open

- **Where the intent vocabulary lives.** Profiles are built into `intent.py` today. A deployment
  whose fleet has a capability the built-ins do not name has to edit the package — which is the
  `models.yaml` argument again, one layer up. The seam exists and says so: `BUILTIN_PROFILES` carries
  a note that `resolve_profile` and `parse_intent` both already take a table and nothing reads one
  out of `models.yaml`. 🚨 A config-supplied table must stay in **declared capabilities, never
  endpoint names**, or it becomes a second routing table to keep in step with `models.yaml`. Related,
  and possibly a different shape: intent cannot express a **negative** constraint ("anything but this
  endpoint"), which is what a caller working around one bad model wants.
- ~~**Multi-tenancy depth.**~~ **Settled 2026-09-01 with E: keys stay flat, because the shape that
  was wanted already exists.** The quota holder, the DRR fair share and the spend cap are keyed on
  the `agent_id`, never on the key — so several keys naming one `agent_id` give a team one budget
  with per-key revocation and per-key priority, which is what nesting was for. `GET
  /rs/v1/admin/keys` groups by `agent_id` so the structure is visible rather than inferable. A real
  team → key hierarchy would only start to earn its keep with per-team *aggregate* caps distinct from
  the per-caller ones, and nothing yet needs that.
- 🚨 **Intent routing never explores, so a fleet converges on ONE endpoint.** Found on 2026-09-01
  by running the thing: 2,566 calls from two callers across three intents (`fast-chat`, `chat`,
  `reasoning`) went to `tier3` and *only* `tier3`. `tier1` and `tier2` ended with
  `typical_ms: null` — never sampled, not once.

  The mechanism is two individually-correct rules composing into a ratchet, and neither is a bug:

  1. `_preference_key("latency")` is `(latency_key, -free_slots)`, and `latency_key` is `inf` for an
     endpoint with no samples — deliberately, so we never prefer a backend *because* we know nothing
     about it. **While every candidate is unmeasured the whole key collapses to `-free_slots`**, so
     `latency` silently ranks as `capacity` and the largest endpoint wins. `balanced` reaches the
     same answer by its own route.
  2. The winner is the only endpoint that then accumulates samples. Once it has enough to publish a
     `typical_ms`, its `latency_key` is finite and every rival is still `inf`, so it now wins
     *outright* — and the rivals can never acquire the evidence that would unseat them.

  The module docstring covers the cold-start tie ("everything ties, and the stable name tie-break
  decides; that is deterministic and honest") and stops at the first request. This is the second
  one. In a fleet with pinned traffic the smaller endpoints get sampled by other means and the
  ratchet never closes; on a fleet driven purely by intent it closes immediately and permanently.

  🚨 **Not obviously a defect, and deliberately not fixed here.** Every fix is a design decision:
  exploration (a bandit, which makes a pure resolver stateful and non-deterministic — `resolve` is
  pure, and `/rs/v1/plan` promises a plan and the call after it agree), or seeding `typical_ms` from
  the timeout model's priors, or admitting that `latency` with no evidence *is* `capacity` and
  saying so in the published profile table. The first is the one that changes the most. Write the
  reason down before the code.

- **Where cost truth lives.** Provider-reported spend vs. our own token accounting; they will
  disagree, and one of them has to be authoritative for threshold decisions.
- **Streaming through a remote provider** under a computed deadline — the soft-deadline extension
  logic assumes decode progress is observable, which is provider-dependent.
- **Whether the timeout model can learn per-provider** for remote backends whose latency we do not
  control, or whether those get a different deadline strategy entirely.
