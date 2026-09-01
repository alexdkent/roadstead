# Changelog

Notable changes to Roadstead. Breaking changes get their own section with a reason, per
`docs/compatibility.md` — the stable surface is the wire contract in `docs/api.md`; everything else
is internal and changes without an entry.

Pre-1.0: breaks are permitted, but each one is a recorded decision rather than a surprise.

## Unreleased

### Fixed — the blocking client leaked connections and hung after `close()`

Landed 2026-09-01. `roadstead.client.RoadsteadClient` — exported, documented, and until now never
constructed by a single test. The async class had two files of coverage; its blocking wrapper had
none, and the wrapper is where the hard part lives: a private event loop on a worker thread with
async generators pumped across it.

- **Abandoning a stream leaked the connection.** `break`-ing out of `for frame in
  client.stream(...)` left the async generator to be finalized by the GC on a thread with no running
  loop, so the `async with` around the HTTP response never unwound and the connection was never
  released. It announced itself only as `Exception ignored in: <async_generator ...>`, which nothing
  reads. Streams are now unwound on the loop that owns them, and `close()` drains the ones it handed
  out before stopping that loop.
- **Any call after `close()` blocked forever**, waiting on a future a stopped loop will never
  resolve. It now refuses with a sentence, which is what the module docstring already promised.
- `close()` is idempotent, and the "you are inside a running loop" refusal no longer trails a
  `RuntimeWarning: coroutine '_make_async' was never awaited`.

No wire change; a caller who never abandoned a stream and never used a closed client is unaffected.

### Changed — the enriched `price` block has ONE shape on every route

Landed 2026-09-01. **Additive.** `GET /rs/v1/models` was emitting three of the price block's five
fields, where a chat envelope and a plan estimate emit all five. A client with one typed view for
"a price" — which is what the shipped SDK has — read `source` and `detail` as empty for every
endpoint. `source` is the field that separates a price the provider **published** from one Roadstead
**imputed**, which is the same measured-or-guessed distinction the management plane reports for slot
counts, and the one a caller choosing where to send work most needs.

Found by a new guard that extracts every field `roadstead/client/_models.py` reads and walks each
one against a response a real server produced — the client-side twin of the UI's `pick()` guard, and
for the same reason: a renamed or never-sent field reads as `""` forever and nothing raises. The
block's five fields are now published in `docs/api.md` §1.7.3 instead of being elided as
`{"...": "..."}`, which is why the drift was invisible.

### Changed — 🚨 BREAKING and security-relevant: an address no longer grants admin

Landed 2026-09-01. **The admin plane now requires BOTH a reachable address and an `admin`-scoped
credential.** Until this change a request from **loopback presenting nothing at all** could pause a
backend, re-weight a caller's quota, ingest call logs or flip a runtime flag. The audit trail
recorded such a change as `key_id: null, source: "ip"` — the system stating in its own log that
nobody had authenticated, and permitting the write anyway.

**How it got there matters more than the fix.** It was not a bug anybody wrote. The auto-grant
belongs to the INFERENCE door, where "already on the box" is a fair proxy for "allowed" on a
local-first proxy, and it was correct there. The admin plane, and later the management UI, were added
**on the same port** and inherited it — a default written for one door silently governing a different
one, with the UI's elaborate reasoning about CSRF, credential kinds and the 401/403 split sitting on
top of a door that was already open on localhost.

**What breaks:**

- **`ROADSTEAD_ADMIN_NETS` is reinterpreted, not replaced.** It named addresses that WERE admins; it
  now names addresses that MAY REACH the plane, with a credential still required. A deployment
  relying on it for unauthenticated admin will start getting 403s and needs a key.
- **`:admin` on a `ROADSTEAD_ACL` address entry no longer grants the scope**, for the same reason.
- **Loopback is no longer admin.** Scripts driving `/v1/admin/*` from the local host must present a
  key.
- **A key alone is no longer enough from anywhere.** The network gate applies to credentials too, so
  an admin key used from off-net needs its network named in `ROADSTEAD_ADMIN_NETS`.

**What replaces it:**

- **Reach**: loopback always, docker-internal by default so a container reaches its own plane, and
  whatever `ROADSTEAD_ADMIN_NETS` names — naming any net **drops the docker default**, because a
  `/12` is a weak gate. The network is checked first, so an off-net refusal says it is about the
  network and no credential answers it.
- **Credential**: an `admin`-scoped key, as `X-API-Key`, `Bearer`, or HTTP Basic in the **password**
  half (what the UI uses — unchanged).
- **A bootstrap key, so the plane is never open and never unreachable.** With no *operator* key
  configured, startup mints a random admin key and logs it; it is not persisted and a new one is
  minted each boot until `ROADSTEAD_API_KEYS` is set. 🚨 It is **invisible to
  `KeyRegistry.configured`**, so it cannot flip §1.5 rule 2 on the inference door — counting a
  self-minted key there would 401 every OpenAI client that sends a placeholder `Authorization`
  header, which is the exact failure rule 2 exists to prevent.

**Roughly sixty tests failed at once when the grant was removed, and none of them meant to assert
it.** They were testing what the plane does; that they also asserted it needs no credential was
invisible until it wasn't. They are authenticated centrally through `tests/admin_key.py` rather than
one at a time, and the gate itself is pinned separately in `tests/test_admin_gate.py` — a helper that
authenticates the plane's tests must never also be the thing that tests the gate. Two of the failures
were doctrine tests asserting the old behaviour in as many words, including one whose docstring
called the unauthenticated audit record *"a meaningful — and slightly alarming — thing for an
operator to find"*. It was.

### Changed — BREAKING for anyone holding a clone or a SHA: the history was rewritten (scrub S6)

Landed 2026-09-01. The second and final `git filter-repo` pass, the last item in
`docs/corpus_and_scrub_plan.md`. **Every commit SHA in this repository changed.** A clone from
before this date shares no ancestor with `main` and cannot be fast-forwarded; re-clone rather than
pull. Any SHA cited in an external document, a branch name, or a bookmark is dead.

**It took two passes, not one.** The first rewrote blobs and commit messages. The second was a
`--mailmap` run over commit *metadata*: author and committer identity is neither blob content nor a
commit message, so it survived the first pass entirely. All 322 commits now carry one identity at a
public address. 🚨 **That pass found a second private hostname on 103 commits that nothing had
flagged** — because a sweep over tracked files cannot see the author line, and every check to that
point had been a sweep over tracked files.

This is recorded here rather than passed over as housekeeping because it is the most broadly
breaking change the project has made — it breaks something for every holder of a copy, which no API
break does — and because two things about it were wrong in the plan that specified it:

- 🚨 **`--replace-text` rewrites blobs and nothing else.** The documented one-liner would have left
  all 321 commit messages untouched, and messages were the *denser* surface: 282 of the 321 commits
  were authored inside the origin monorepo and describe its hosts, its container IDs and its sibling
  projects far more freely than the code ever did. `--replace-message` takes the same rules file.
- 🚨 **Bare-word rules would have corrupted content.** `Delta` also occurs as `Gated DeltaNet`, a
  model architecture, and `Chase` as "Chased to column level". Every one of the 42 rules is a
  multi-word key or a distinctive stem, authored against an inventory of the actual occurrences.

Host and sibling-project names became pseudonyms throughout — the names you will now read are
`nexus`, `nasbox`, `boxa`, `tideway`, `beacon` and `sidekick` — and the fleet's `/24` became
`10.0.0.x`. 🚨 **The mapping itself is not recorded anywhere in this repository, deliberately.**
Writing `old`→`new` in a changelog would restore every name the pass removed and hand a reader the
key to reverse the rest; a scrub that documents its own substitutions has not scrubbed anything. **The measurement comments are still records** — `CLAUDE.md` says so, and now
also says the box names within them are pseudonyms, because a reader who goes looking for the
machine should be told there isn't one. Model and endpoint-class vocabulary was out of scope and is
unchanged.

Also fixed in passing: **a real airline booking reference was still in the working tree.** S4
replaced the traveller, the airline and the airports around it and carried the booking reference
over verbatim — and a booking reference plus a surname retrieves a booking. It reads `QQ7X2R` now,
in the tree and throughout the history.

### Fixed — today's spend survives a restart (Workstream D)

Landed 2026-09-01. `SpendLedger` is in-memory and its day bucket reset to zero on every boot.

🚨 **This was a correctness bug, not the acceptable simplification the roadmap had called it.** The
old wording — "right for what it governs, whether a caller is degraded *now*" — quietly assumed *now*
was the same length as the window the threshold reads. It is not: `daily_spend_usd` is a **day** and
the process was measuring an **uptime**. A deploy at noon handed every caller its whole allowance a
second time, so the more a fleet ships the less its spend cap means — worst precisely on the
deployment where somebody set the cap deliberately. DRR balances have survived a restart since Phase
3.4; this is the same argument about the other per-caller quantity, missed when `spend.py` landed.

`startup` now replays today's rows from `proxy_completions` through the ledger's existing `charge`,
via a new `day_spend_rollup(since)` aggregated **in SQL** — one row per caller-endpoint pair rather
than one per request, on the existing `idx_pc_completed` index.

- 🚨 **The rollup groups by endpoint as well as by caller.** Collapsing it re-prices a caller's whole
  day at one arbitrary endpoint from the group, and for a caller spanning a cheap and an expensive
  model that is the entire number.
- **`charge` gained `requests=`** rather than growing a second `recover()` method, so there is ONE
  implementation of the pricing and day-rollover rules. A second copy is what `cost_model.context_fit`
  was created to undo, and that copy had already gone silently dead.
- **Re-priced at TODAY's prices**, not each call's price at the time. Right for a threshold that
  answers "is this caller degraded now", wrong for a bill — and anything billable reads
  `proxy_completions`, which keeps the **tokens**.
- 🚨 **The seed fails OPEN, loudly.** Refusing to boot because we cannot prove a caller crossed a
  threshold whose entire consequence is one priority band would let a spend cap take the proxy down,
  which is the same argument that stops it taking a *caller* down.
- `spend.day_start` is derived from `day_bucket` so the query that reloads a day and the threshold
  that reads one cannot disagree about where a day begins — they would drift apart at exactly one
  instant a day.

**Where cost truth lives, settled by construction.** `proxy_completions` stores **tokens, never
dollars**, so pricing lives in one place, the ledger is a derived view rather than a second record to
reconcile, and a provider's invoice can only ever disagree with us about *price* rather than about
*usage*. Reconciliation against a real invoice still cannot be built here honestly — it needs a real
billed account, and a reconciler written against an invented invoice would agree with itself and
prove nothing.

`tests/test_spend_durability.py`. Eight mutations, every guard observed going red. One first
SURVIVED and was a real weakness: the price fixture charged every endpoint the same rate, so dropping
the endpoint from the `GROUP BY` changed nothing — a caller spanning two prices is what makes that
grouping observable.


### Removed — BREAKING: `normalize_endpoint` no longer rewrites a `nexus-` prefix

Landed 2026-09-01, found by re-running the straggler sweep (`docs/corpus_and_scrub_plan.md` S5).
Recorded here per `docs/compatibility.md`: this is internal behaviour rather than the wire contract,
but it changes how a *name a caller sends* resolves, which is as close to the contract as internal
gets.

`normalize_endpoint` stripped a `nexus-` prefix and mapped a bare `nexus` to `chat` — **one private
fleet's host naming, hardcoded since the first commit (`1becf53`, 2026-05-27) and shipped to
everyone.** It was a
scrub finding and a design defect at once, and the second is the reason it is removed rather than
renamed:

- It was a **second aliasing mechanism** beside `models.yaml`'s `aliases:`, which is exactly what
  this codebase refuses everywhere else — and the refusal has a name here, since the alias
  duplicate/shadow notice added in the same release cannot see this one at all.
- It could not be configured, overridden or disabled, and it silently rewrote **any** endpoint whose
  name happened to begin with those six characters. An operator with `nexus-a` and `a` had a pin
  at the first silently reaching the second.

**Migration**, if you actually want that mapping: put it where every other name lives —
`aliases: [nexus]` on the endpoint, which is declared, reported and collision-checked. The whole
suite passed unchanged with the branch removed, which is how long it had been dead weight.

Two smaller findings from the same sweep: three arbitrary test addresses that merely *looked* like
the private subnet (a sweep cannot tell, so each cost a human adjudication per re-run and S6 would
have rewritten them through the history for nothing), and a sibling private project's name used as a
shipped `ROADSTEAD_ACL` example.

🚨 **The topology half of S5 now runs on every commit** (`tests/test_scrub_sweep.py`) — a sweep that
lives in a shell command in a document is one somebody has to remember. The identifier half stays a
human pass **and the test says so**, because there is no pattern to key on for a name, which is
exactly how S4's finding survived the first sweep. Four mutations, all red.


### Fixed — a cancelled straggler's caller gets the envelope, not a raw 500

Landed 2026-09-01. `docs/ledger.md` carried this as "Open, and ours to fix"; it was parked as
"belongs upstream" while the origin monorepo was authoritative for behaviour, and that rule was
retired on 2026-08-31.

**Root cause.** `asyncio.CancelledError` derives from `BaseException`, not `Exception`. At
`timeout_graceful_shutdown` uvicorn calls `task.cancel()` on every in-flight handler, and the
resulting exception sails past *both* the `exception_handlers` backstop in `build_app` and
Starlette's own `ServerErrorMiddleware` — neither can see a `BaseException`. The caller of a
straggler got `500 Internal Server Error`, plain text, no `code`, no marker, for a condition that is
retryable and entirely ours.

`ShutdownEnvelopeMiddleware` closes it. A **middleware**, not another `exception_handlers` entry: a
handler there could never be reached, because Starlette dispatches them from inside an
`except Exception`. User middleware sits outside the router and inside `ServerErrorMiddleware`, which
is the outermost place a route's `BaseException` is still catchable.

- **No new code.** `draining` + 503 + the literal `backpressure` marker is exactly what `lifecycle`
  already emits for work *refused* while draining, and it has to be: the caller's situation is
  identical. §2.2 classifies deferrable on the marker substring, so dropping the word would make a
  shutdown read as a hard failure to every client that has not migrated to classifying on `code`.
- 🚨 **The cancellation is always re-raised.** Swallowing it breaks the asyncio contract and leaves
  uvicorn waiting on a task that has declined to die — turning the bounded shutdown into the
  unbounded one `SHUTDOWN_DEADLINE_S` exists to prevent. This adds a response; it does not decline
  to stop. A failure to deliver the envelope is swallowed rather than the cancellation.
- 🚨 **A cancelled STREAM gets an error frame and NO `[DONE]`.** `[DONE]` is an assertion of
  completeness; emitting one over a truncated answer is the exact collapse `correction.py`'s
  `finish_reason` repair refuses to make.
- A cancelled non-stream whose headers are already out has its body **closed**, not left hanging.

Measured with `tools/sigterm_drain_probe.py` S3, before → after: `500 Internal Server Error` at
48.77s → `503 {"code": "draining", …}` at 48.76s. Exit time (78.2s) and what is persisted are both
unchanged. `tests/test_shutdown_envelope.py`. Nine mutations, every guard observed going red — one
first failed with a bare `TypeError` and asserts first now.


### Fixed — an alias claimed twice routed silently, and the soak was measuring itself

Landed 2026-09-01. Two unrelated defects, both found by doing the thing rather than reading it: one
by the first pass of the origin-ledger transplant, one by running `tools/soak.py` for 25 minutes.

**An alias claimed by two endpoints resolved in silence.** Transplanted from the origin monorepo's
`models-yaml-alias-collision` (2026-07-30) — and it turned out not to be a transplant. The origin
**raised** at module import on a duplicate alias, which took a whole gateway down and crash-looped it
to FATAL over one duplicated line. This repo builds the same map with `setdefault`, so the second
claim is discarded in file order, in silence: the operator's name reaches an endpoint they did not
intend, on every request, forever. **The lesson survived the extraction and the guard did not.**
🚨 Neither behaviour was right — refusing to load contradicts this repo's own rule that a typo must
not stop a fleet booting. So it loads, resolution is unchanged, and both cases report through
`hooks.config_notice`: `duplicate` for two endpoints claiming one alias, `shadowed` for an alias that
another endpoint's class or role wins (that ordering is deliberate and stays, and it still killed an
alias somebody wrote). Each names the endpoint the name actually reaches in an **`in_force`** field —
the management plane's own vocabulary, and read by `GET /rs/v1/admin/config`. A self-alias is
deliberately not reported: it maps to itself, it is common, and a notice that fires on every boot is
one nobody reads. `tests/test_alias_collision.py`, `docs/ledger.md`. Six mutations, all red — one
first SURVIVED because the test read the winner out of the prose sentence, which interpolates the
same value the emptied field held.

**🚨 `tools/soak.py` was measuring its own test double.** A 25-minute run climbed
**118MB → 1867MB (+72.8 MB/min, linear, no plateau)** — precisely the shape that tool's docstring
calls unhealthy. The growth was `FakeBackend.requests`: an unbounded list holding a body dict and a
header dict per request, for the life of the process. Two defects in one:

- `roadstead.testing` is **shipped public surface**, so that is a real leak for anyone who installs
  it and drives sustained load through it.
- The soak exists to detect leaks by RSS slope, and its headline number — the field it explicitly
  tells you is "the number to look at" — was dominated by the harness rather than the proxy.

The recorder is now a bounded deque (`REQUEST_LOG_CAPACITY`, exported, 1000) that **reports what it
drops** (`requests_seen`, `requests_dropped`) rather than truncating in silence — the same answer the
audit trail gives. Slope after the fix: **+8.2 MB/min and flattening**, the remainder being
`RollingMetrics` filling its own 300s window toward steady state. `tests/test_fake_backend_recorder.py`.
Six mutations plus a no-op control; one first SURVIVED because the ordering test drove a local helper
instead of the app's real record path, so it was asserting the file's own model of the recorder
against itself.

**Also fixed on the way past:** `RateLedger.prune` was never called — see the entry below.


### Fixed — the rate ledger grew without bound, and nothing was watching it

Landed 2026-09-01. `RateLedger.prune` shipped with Workstream I carrying a docstring that said
"Called from the maintenance tick". **Nothing called it**, in the package or anywhere else.

- 🚨 **Both dicts behind the rate threshold are keyed by `agent_id`, which is a caller-supplied
  string on the address path** (`docs/api.md` §1.5 — an address only fills in an `agent_id` the body
  omitted, so a body may declare its own). They grew one entry per distinct name ever seen, without
  bound, under a caller's control. `rate_demotion_noted` is the second one and the one a reader
  forgets exists; forgetting a caller now forgets that we noted its demotion, which is also correct
  on its own terms — it had no traffic in the window, so if it returns and crosses again the operator
  should hear about it again.
- 🚨 **It is pruned on WALL time, not the poller's monotonic clock**, and the call sits one line
  below `timeout_model.prune(mono)`, which takes exactly that. `RateLedger.record` stamps
  `time.time()`, so a monotonic `now` would compare a process uptime against epoch timestamps: the
  cutoff lands decades before every sample, nothing is ever stale, and the call returns 0 forever
  while looking wired. A leak fixed by a call that does nothing is worse than the leak, because the
  call is evidence it was handled. Pinned by a test that asserts the two clocks are far apart, so it
  cannot pass by coincidence.
- **The ledger is now armed in `tests/loop_affinity.py`.** It was not armed at all — a whole
  workstream's state outside the guard, while the soak reported a full method count. It has two
  loop-side writers now (the submit path and the poller tick), which is precisely the pair that would
  interleave if either ever moved off the loop.
- **The threshold this feeds cannot reject**, which is why its own memory must not be a denial
  surface either — a rate control that degrades is deliberately not a defence against a runaway
  caller, and its bookkeeping becoming one inverts that.
- `tests/test_rate_prune.py`. Six mutations, every guard observed going red by assertion. One first
  SURVIVED and was a real weakness: the live-caller test asserted `prune_rate_state() == 0`, so it
  never reached the `if dropped:` branch and proved only that a prune which does nothing changes
  nothing. Another failed with a bare `KeyError` and was rewritten to assert first.


### Added — the intent vocabulary in config, and the negative constraint (Workstream C)

Landed 2026-09-01. Both of C's open items, and they turned out to be one question — *who owns the
words a caller may use* — with one line running through it: **config vs request**, not positive vs
negative.

**`models.yaml` grows an `intents:` section.** Nine profiles still ship built in; a deployment's own
are **layered over** them rather than replacing them, because a file that defines one profile has
said nothing about the other nine, and a whole-table swap would silently empty a vocabulary
`GET /rs/v1/models` publishes and callers code against. Overriding by name is how a fleet says its
`reasoning` means something particular, and it is **disclosed**: every published profile carries
`source: builtin | models.yaml`, since a caller reading our documentation for a word this fleet
redefined has no other way to notice.

🚨 **A profile still cannot name an endpoint — and now there is a config parser that must not learn
how.** A profile naming endpoints would be a second routing table to keep in step with `endpoints:`,
and it would break on every fleet spelled differently from the example's. Guarded from both ends: the
allowlist (`_PROFILE_FIELDS`) and `Profile`'s own fields, so the rule does not rest on a door in a
wall with a second door.

**An unusable stanza is REFUSED, not offered.** An unknown capability, kind or preference makes a
profile that matches nothing on every request — and the caller reads "no endpoint satisfies
requires=[…]", a sentence about the fleet, for a fault in a config file. Refused, they get "unknown
intent", which points at the vocabulary, where the fault is. Either way the operator gets a
`hooks.config_notice`, readable at `GET /rs/v1/admin/config`. A refused *override* does not leave the
built-in standing under the operator's spelling — they would be reading their own summary in the file
while callers got ours on the wire, which is the declared-vs-in-force gap the management plane exists
to close, created by us.

**`exclude` — the negative constraint, and why it is expressible.** A caller working around one bad
model wants "anything but this", and the alternative is enumerating every endpoint it *would* take,
which is a routing table in the caller. It fits because the vocabulary it needs is not the profile
vocabulary: a profile is shared, published, operator-written and stays in capabilities; an intent is
one caller's words about one call, where `pin` already names an endpoint. `exclude` says the same
kind of thing in the other direction and adds no table. It is normalized through the same callable as
`pin` (an exclusion compared literally would not match an alias, and would route to exactly the
endpoint it was written to avoid), it is a complete declaration on its own, and it is reported ahead
of any property of the endpoint — a caller who excluded something also unrouted needs the reason they
can act on.

🚨 **An `exclude` naming an endpoint this fleet does not have is a `404 unknown_endpoint`, not a
warning.** This was built as a disclosure first and changed, because the disclosure argument assumes
the one thing the proxy cannot check. Such a name is either "not in this fleet" or "in this fleet,
under a spelling you got wrong", and from inside those are identical bytes; serving the second sends
the request to precisely the endpoint the exclusion existed to avoid and reports success. Same shape
as the `finish_reason` repair that became a silencer, and as the GBNF grammar OpenRouter refuses
rather than drops. It is also what `pin` already does: a caller that must spell an endpoint correctly
to demand it does not get to misspell one to avoid it. Naming the same endpoint in `model` and
`exclude` is a `400` — a contradiction wholly visible in the request, refused where the caller can
see it rather than resolved to an empty candidate set that reads as a fault in the fleet.

**One bug, found by running it.** The refusal echoed the **normalized** name: `normalize` lower-cases
and strips, so a caller who wrote `"tierX"` was told `'tierx'` — a word it never sent, in the one
message whose whole job is helping it find a typo. It is the reason `pin_as_written` exists, missed in
the other direction, and the suite was green throughout because the refusal fired and named something
plausible. `Intent` now keeps the *pairing* rather than two sets, and matches on the normalized form
while reporting the written one.

**No new error code.** An unresolvable exclusion is the existing `unknown_endpoint`; a contradiction
is the existing `invalid_request_error`. `docs/api.md` §1.7.1. The shipped example gains one additive
profile (`bulk`) so the config path is exercised on every boot rather than documented and unrun.
`tests/test_intent_config.py`. Sixteen mutations, every guard observed going red — one first failed
by raising from `intent.py` rather than by asserting, which is half a guard, and was rewritten.


### Changed — two cleanups on the hot path and the client boundary

Landed 2026-09-01. Both were diagnosed and left open; neither changes a contract.

- **`/rs/v1/chat` no longer walks the latency ladder once per endpoint per request.**
  `EnrichedApi.facts()` runs on every request to resolve one intent, and it called
  `timeout_model.advise` once per endpoint to fill a single field — N ladder walks, on the hot path,
  for a number that is a median over thousands of samples and cannot meaningfully move between two
  requests a millisecond apart. Memoised for 1s: **14,000 `advise` calls became 7** over 2,000
  `facts()` calls, and the call itself went from 80µs to 47µs.
  🚨 `typical_ms` is **not** dropped and **not** made optional, which is why this is a memo rather
  than a flag: `prefer=latency` and `prefer=balanced` rank on it, and an endpoint with no samples
  sorts as SLOW — so omitting it would silently re-rank every intent-routed request rather than lose a
  display field. The memo is keyed by endpoint as well as time, so an endpoint that appears later is
  priced rather than served from a map that never saw it.
- **The client SDK is now driven over a real socket.** `tests/e2e/test_client_sdk_live.py` uses
  `ASGITransport` and says so — it exercises request building and envelope parsing, not httpx's
  networking — which left the SDK's own transport configuration asserted as a comparison between two
  constants. `tests/e2e/test_client_sdk_socket.py` runs the real proxy under real uvicorn on a real
  ephemeral socket, points a client the SDK built **itself** at it, and observes: a real round trip,
  real SSE framing over the wire, connection reuse inside the keepalive window, and 🚨 **the idle
  socket actually being retired** after `CLIENT_KEEPALIVE_EXPIRY_S` (4.5s) — §1.3's ordering
  invariant watched rather than computed. It reaches into httpx's pool deliberately, and asserts the
  internals exist rather than skipping when they do not, so an httpx upgrade fails loudly instead of
  the guard going quietly blind.
- Eight mutations, every guard observed going red by assertion. One first SURVIVED and was a real
  weakness: the memo-expiry test advanced its fake clock by *the TTL*, so a mutation setting the TTL
  to a billion seconds moved the goalposts with it.

### Fixed — the shutdown drain now has a ceiling, and it is published once

Landed 2026-09-01. SIGTERM was already the correct signal and that stayed settled by experiment
(`docs/ledger.md`); the drain semantics are unchanged and `tools/sigterm_drain_probe.py` still
measures **78.2s** for the same scenario. What was wrong is that **nothing enforced a ceiling**.

- 🚨 **The tail after the drain was unbounded.** The 30s drain was bounded; everything after it was
  not — and `OnDemandManager.close` makes a **network call per held GPU lease**, so a dispatcher that
  stopped answering hung the lifespan shutdown indefinitely, past any stop-grace an operator had set,
  at which point the container SIGKILLs a process that has flushed nothing. Lease release and pool
  close now run **concurrently under one bound**, and a timeout there costs a lease its TTL and never
  the flush that follows.
- **Every phase is named and bounded**, and `SHUTDOWN_DEADLINE_S` is their **sum** rather than a
  literal: drain 30s, straggler unwind 3s, teardown 5s, queue flush+join 10s → a 48s ceiling on the
  lifespan shutdown. 🚨 There is deliberately **no outer `wait_for`** around `shutdown()` — it would
  cancel `queue_db.close()` mid-flush, which is the SIGKILL failure the drain exists to avoid, and a
  half-written completion row is worse than a slow exit. The ceiling is true by arithmetic and the
  arithmetic is pinned.
- **`RECOMMENDED_STOP_GRACE_S` is computed in one place** and read by the uvicorn argument, the
  `Dockerfile` label, the Dockerfile prose, a startup log line and the probe. It used to be three
  numbers and a comment.
- ⚠️ **The recommended container stop-grace went UP, from 90s to 108s** — and that is the fix rather
  than a regression. 90 came from a *measured* worst case (~78s) taken while the tail had no bound at
  all, so it was an observation rather than a ceiling and a wedged dispatcher would have blown
  through it. 108 is the first number that is a ceiling. **Update `stop_grace_period` /
  `--stop-timeout` accordingly**; below it, `docker stop` SIGKILLs the proxy with zero budgets and
  zero completion rows persisted, and the 10s default works fine while the proxy is quiet — which is
  how it gets tested and why it first fails under load.
- **`timeout_graceful_shutdown` is no longer derived from `_DRAIN_DEADLINE_S`.** The old
  `_DRAIN_DEADLINE_S + 18` carried a comment claiming uvicorn's budget must exceed the drain "or it
  hard-kills the process mid-flush", which assumes a nesting that does not hold — uvicorn hands over
  to the lifespan shutdown. Its **value is unchanged**, so nothing about the measured behaviour moves;
  what changed is that it is its own knob for its own job.
- `tests/test_shutdown_budget.py`. Ten mutations, every guard observed going red by assertion.

### Added — key lifecycle, and the abuse control DRR is not (Workstream I)

Landed 2026-09-01. What was left of Workstream B, and the oldest unclosed thing in the repo:
`created_at` was written, surfaced in two views, and **read for no decision**.

**Expiry.** `expires_at` (an absolute date in a keys file) / `expires_in_s` (a duration over the API).
A key with no expiry never expires, so an existing registry behaves exactly as it did. An expired key
is a `401 invalid_api_key` with **its own sentence and the same code**: a caller can act on no
distinction between "expired" and "unknown", so §2.1 mints nothing new, but the operator reading the
log can — one means check what you pasted, the other means issue a successor. A restart never extends
a key; the store holds the absolute instant, not the duration it came from.

**Rotation.** `POST /rs/v1/admin/keys/{key_id}/rotate`, one action, because the manual version is two
calls in an order that matters and **both orders are wrong** — enrol-then-revoke leaves the successor
live and unknown to the caller, revoke-then-enrol leaves nothing working. The successor inherits the
predecessor's policy. `overlap_s` defaults to **0** (revoke now, and the response says so); with an
overlap the predecessor gets an **expiry** rather than a tombstone, so it survives a restart a timer
would not. If the successor cannot be minted the predecessor is **untouched**.

**Per-key address binding.** `bind` is an ADDITIONAL constraint, never a way for a key to widen what
an address grants: a bound key is refused outside its CIDRs and is otherwise exactly the credential it
always was. An unparseable entry is a 400 at enrolment and matches nothing at resolution — a narrowing
that failed open would be worse than none. 🚨 It is checked against the *resolved* address, so a
binding is only as trustworthy as `ROADSTEAD_TRUSTED_PROXIES`, and `GET /rs/v1/admin/keys` reports
`checked_against` rather than leaving an operator to infer which case they are in.

**`requests_per_minute` — and no 429.** DRR is fairness *under contention*: a caller alone on a quiet
fleet is unthrottled by design, which is correct for fairness and exactly why it does not bound a
runaway. Crossing this costs a caller exactly what crossing `daily_spend_usd` costs it — **one
priority band and paid spill, never local capacity** — and mints no code. A rate *limit* would be a
fourth spelling of "no" and would put a mistyped threshold in a position to take a caller offline.
`roadstead/rate.py` is the **sixth** pure-computation module; CLAUDE.md's layout was updated because
`tests/test_pure_modules.py` fails until it is.

🚨 **Degradations do not STACK.** A caller over both thresholds drops **one** band, not two — enforced
in `ProxyState.effective_priority`, which consults at most one standing, so a third threshold cannot
compose by accident. Each crossing reports through the degradation seam **separately**, because one
means "look at the bill" and the other means "look for a loop". `effective_priority` is split from
`spend_demote` so a management *read* fires no notice.

**Three more found by running it, none of which a test would have reached.**

1. The rotate response was assembled from the **predecessor's** row, so a successor registered with
   the wrong policy would still have been reported as inheriting it — the response was not evidence.
   It now reads back from the registry, and the test asserts against the registry too.
2. A successor inherits the policy but **not the expiry**, and came out permanent — a silent
   weakening of a control the operator deliberately set. The default is right (inheriting an absolute
   instant would mint a successor that expired seconds later) so it is **disclosed**, not changed.
3. 🚨 `overlap_s` **extended** a predecessor past its own expiry. A day-long overlap on a two-hour key
   pushed its expiry a day out — a rotation quietly lengthening a credential, which is the widening
   this repo refuses everywhere else. Truncated to the earlier instant, and said out loud.

`docs/api.md` §1.5, §1.6 (a two-threshold table, and the non-stacking rule), §3.3, §3.4, one row in
§3's route table. `tests/test_key_lifecycle.py`, `tests/test_rate_threshold.py`. Twenty-six mutations,
every guard observed going red by assertion.

### Added — a read-only admin scope, and the audit trail (Workstream H)

Landed 2026-09-01. Two changes to what an admin identity *is*, landed together because they touch the
same three files and answer halves of one question: the read views report a **state**, and a state
cannot say who put it there; and an operator who wanted somebody to be able to *look* had to give
them the ability to change everything.

**`admin_readonly` — a narrowing that can never widen.**

- A credential with `admin: true, admin_readonly: true` reaches every `GET` on the management plane
  and is refused, with a 403 that says why, on everything else. On a key without `admin` it is a
  **400 at enrolment**, not a silent no-op: an operator who wrote it believes they issued a safer
  credential than they have.
- **The read/write split comes from the HTTP method**, not from a list of write routes — a route list
  is a second thing to keep in step with `routes.py`, and when it falls behind the failure is silent
  and widening. It lives in the shared gate, so the four control routes that predate the management
  plane inherit it without a line of their own.
- **`identity.py` decides, everything else renders.** `IdentityResolver.admin_denial` owns which of
  the three refusals applies; `deny_non_admin` renders it. An AST guard fails if `may_admin_write` or
  `admin_readonly` is read anywhere but `identity.py` — a second place deciding what a scope permits
  is the shape of the `_remote_ip` that had to be removed from `management.py`.
- Spelled `:readonly` in the shared identity grammar, so `ROADSTEAD_API_KEYS` and `ROADSTEAD_ACL`
  express it too. 🚨 **A narrowing beats an overlapping grant**: `127.0.0.1=ops:admin:readonly` is
  read-only even though loopback is a built-in admin net. If the widest grant won, that line would
  silently be a full grant for every operator who wrote it.

**`GET /rs/v1/admin/audit` — who changed what, and when.**

- **Every mutating admin route records**, including flags, pause, resume and maintenance, which touch
  no overlay state and would otherwise record nothing. A trail covering only some of them is worse
  than none, because a reader assumes completeness. The completeness guard is driven from
  `routes.py`, not from a list: all **11** mutating admin routes, both spellings.
- **A record names the credential and never carries one** — `key_id`, never the key or the digest.
  Both the key label and the address are recorded always: a record with an address and no `key_id`
  means an address-derived admin made the change, which is meaningful and slightly alarming, and
  collapsing them into one actor string would hide which factor authorized it.
- **It reports its own limits as data**: `persisted` (in-memory is the default), `dropped`,
  `capacity`. It is an operator-facing change trail, not a security log of record — and says so,
  rather than presenting itself as complete while being neither durable nor unbounded.
- Written on the loop, persisted off it. The overlay lock moved onto the overlay, because two modules
  write that file now and a lock owned by one of them serialises half the writers.

**Three bugs, and two of them were found by RUNNING the page.** The suite was green for all three.

1. A control-route record only reached disk if a *later* overlay write happened — so the most recent
   entry, the one an operator looks for after an incident, was the one a restart lost. A drain is
   taken in a hurry, often right before the restart that would drop the record of it.
2. The Fleet view's Pause/Resume was the one write control not wrapped in the scope guard. It
   rendered, it worked, and a read-only operator would have learned their scope from a 403 *after*
   clicking. There is now a test that finds every write control from the HTTP method it sends and
   requires it guarded.
3. 🚨 `render()` fired before `primeChrome()` resolved, so the **first paint had no scope** and the
   guard defaulted to allowed — every write control live for a read-only operator until the 30-second
   heartbeat. The first paint is the one somebody clicks.

`docs/api.md` §1.5, §3.3 and a new §3.8; two rows in §3's route table.
`tests/test_admin_audit.py`. Twenty mutations, every guard observed going red by assertion — no
timeouts, no collection errors.

### Fixed — one context-fit predicate, and the dead fourth copy of it

Landed 2026-09-01. "Does this request fit in this much context?" was written out by hand at four
sites — the admission gate (`lifecycle.handle_submit`), the failover gate (`failover.plan`), the
spill gate (`scheduler._admit`) and the WAL-recovery shadow tally (`service`). It is now
`cost_model.context_fit`, called by all four.

- **🚨 The recovery tally never fired.** It read `req.est_input_tokens or 0`, but that field is
  cached by `scheduler.enqueue` and `recover_queued` builds its `QueuedRequest` straight from the
  WAL row — the recovery path has not reached the scheduler yet, so the estimate was always **0**
  and the tally could only fire when `max_tokens` *alone* exceeded the ceiling. The audit that added
  it (2026-07-02) did so precisely because recovered oversized requests were invisible to the
  `context_gate_enforce` flip evidence; they stayed invisible. Measured on the corpus this file's
  differential harness uses: the dead copy disagreed with the shared predicate on **27 of 153**
  cases. `service.py` had also carried an unused `estimate_input_tokens` import ever since — the
  residue of a fix that meant to call it.
- **Nothing else changes behaviour.** The three live gates were byte-identical in effect and are
  verified so rather than asserted so: a differential harness ran the pre-split implementations
  transcribed verbatim from `7563fa6` (2026-09-01, *"Give the operator a face"*) against the
  shared one over **2295** combinations of payload
  shape, payload type, ceiling and endpoint name, comparing the boolean *and* the refusal-message
  bytes. Zero mismatches.
- **🚨 The consequences stay different, which is the whole risk of this refactor.** Admission is
  shadow-or-422 behind `context_gate_enforce`; failover refuses unconditionally (there the
  alternative to refusing is a guaranteed backend 400, not a request that probably works); spill
  defers; recovery counts and never rejects. `context_fit` returns the answer and its arithmetic and
  takes no action at all — folding the consequence in would have armed a flag nobody flipped, and
  that is pinned by a test.
- **`CONTEXT_OVERFLOW_MARKER` is defined once per side of the client boundary.** It is wire contract
  (`docs/api.md` §2.2) and was previously typed out at two call sites. An AST constant sweep — not a
  substring grep — now allows exactly one spelling in the server and one in `roadstead.client`,
  which must keep its own because it imports nothing from the server; the document pins them
  together. A second AST guard fails if any module re-derives a ceiling comparison from
  `estimate_input_tokens` instead of calling `context_fit`.
- `tests/test_context_fit.py`. Twelve mutations, every guard observed going red **by assertion** —
  no timeouts, no collection errors.

### Added — the management UI (Workstream G)

Landed 2026-09-01. `GET /rs/v1/admin/ui` — the roadmap's "manage and monitor Roadstead as a
standalone product" line. **Off by default**: unset `ROADSTEAD_ADMIN_UI` and the route does not
exist, which is absence rather than refusal, the same posture as `ROADSTEAD_TRUSTED_PROXIES`.

- **One static HTML file, vanilla JS, no bundler and no external references at all.** The
  dependency list is still six packages and a test asserts it. The file ships in the wheel, so it is
  public surface — the same argument as `roadstead.testing` — and the CSP it is served with
  (`default-src 'none'`, `connect-src 'self'`, no host anywhere) forbids an external reference
  outright rather than trusting a reviewer to notice one.
- **It shows the GAP, not the config.** Declared beside in force wherever the two can disagree:
  the catalog's slot seed against what discovery left (with whether the engine publishes it at all),
  quota `in_force`/`declared`/`runtime`, declared band against effective band — which is the only
  place a spend threshold is visible, since crossing one mints no error code. A pair collapses to
  one value when they agree, so a highlight means something.
- **🚨 Auth is HTTP Basic and the credential is an API key.** A browser cannot attach a bearer token
  to a navigation and `EventSource` cannot set a header at all, so a browser-facing surface needs a
  scheme the browser carries. Minting a *password* to go in it would be a second credential kind
  with its own store, rotation and revocation, parallel to a registry that already does all three.
  So the key goes in the password half and the username is ignored. **No cookie is minted, so CSRF
  never becomes reachable** on the mutating routes — the concern that has bound this since the
  roadmap was written is answered by not creating it.
- **The door refuses 401 where the plane answers 403**, with `WWW-Authenticate: Basic`. The
  401/403 split is right for an API client and a dead end for a browser: a 403 produces no password
  box, so an operator arriving with no credential — everyone, the first time — has no way to answer
  the refusal. The challenge is identical whether or not keys are configured, so it discloses
  nothing about which §1.5 regime is in play.
- **🚨 Every field the page reads is pinned against a real response.** There is no compiler, no
  schema and no types here: rename a field on the server and one cell renders "—" forever while the
  page looks healthy. So every read goes through `pick(obj, "a.b.c")` — a rule with a test, not a
  style — and the guard walks every extracted path against responses a real service produced. It
  found a live bug on its first run (the feed read `agent_id`/`total_ms` off a frame publishing
  `agent`/`duration_s`), and rendering the page in a browser found two more the tests could not see:
  a one-level `flat()` stringifying nested nodes as `[object HTMLSpanElement]`, and boolean
  attributes rendered empty so `[data-same="true"]` never matched and no pair ever collapsed.
- **The asset read goes off-loop** and is cached after the first hit. A dashboard on a fast cadence
  is exactly the load that finds a concurrency violation.
- `docs/api.md` §1.5, §3 and a new §3.7. Sixteen mutations, every guard observed going red.

### Changed — `GET /v1/stream` is admin-gated 🚨 BREAKING

It was not, and a frame there names the caller, the endpoint, the tokens and the timing of **every
call the fleet serves** — the live form of `/rs/v1/admin/callers`, which has been gated since it
existed. `handle_maintenance_list` carries a note from the previous round of this tightening ("was
unauthenticated — tightened with the rest"); the stream was missed because it reads as plumbing
rather than as a view. A consumer polling it from a host outside the admin nets, with no admin key,
now gets a 403. It is also aliased at `/rs/v1/admin/stream` — same handler, same gate — because
`EventSource` cannot set a request header, so the UI can only reach it through credentials the
browser attaches by directory.

### Changed — `Authorization: Basic` is now one of our credential forms 🚨 BREAKING

Previously ignored as "somebody else's auth". The key rides in the **password** half. A deployment
that has keys configured *and* callers presenting an unrelated Basic header will now see those
callers refused with a 401 rather than quietly identified by address — which is §1.5 rule 1 working,
but it is a behaviour change and it is recorded as one. A header that cannot be base64-decoded is
**no credential at all** rather than a failed one, so it still falls through to the address: a header
we cannot parse was probably never meant as ours. Unrecognised schemes are still ignored.

### Fixed — trusted proxies, and the admin grant behind one 🚨

Landed 2026-09-01. **Not breaking**: `ROADSTEAD_TRUSTED_PROXIES` is empty by default, so a
deployment that sets nothing behaves exactly as before, byte for byte. The header is not "validated
and rejected" by default — it is never consulted.

- **`identity.remote_ip` read `request.client.host` and nothing else**, so anything in front of
  Roadstead — a TLS terminator, an ingress, a sidecar — collapsed every caller into the proxy's
  address. Two consequences, and the second is a security bug: `ROADSTEAD_ACL` silently stopped
  distinguishing anybody, and if the proxy's address fell in the admin nets (**loopback and
  docker-internal are there by default**, and a sidecar usually is one of them) the control plane
  was granted to everyone who could reach the proxy. Latent only because nothing told anybody to
  deploy that way — and Workstream G's UI is what makes a front proxy normal.
- **`ROADSTEAD_TRUSTED_PROXIES`** (addresses or CIDRs, comma-separated, empty by default) is the
  opt-in. `X-Forwarded-For` is read only from a listed peer, and **the caller is the rightmost hop
  that is not itself a trusted proxy**. Not the leftmost: that element is whatever the caller wrote
  before a proxy appended what it observed, so reading it re-introduces the spoof — the
  self-asserted `agent_id` bug in its third costume. The walk is used rather than counting hops
  because the trusted set is CIDRs, whose *width* is not its depth.
- **🚨 A forwarded address does not inherit the BUILT-IN admin nets.** The whole justification for
  auto-granting admin to loopback and docker-internal is that reaching them meant already being on
  the box, and a front proxy is exactly what makes that untrue. `ROADSTEAD_ADMIN_NETS` and an
  `admin` API key are unaffected — the operator's explicit statements stand, the inherited default
  does not. Safe to do unconditionally *because* it is gated on trusted-proxy configuration, which
  is empty until somebody opts in, so no deployment can lose a grant it has today. A trusted proxy
  that forwards **no** header is marked forwarded too: a front proxy that lost its config must not
  become an administrator.
- **Two chain shapes fail closed rather than back to the peer** — an unparseable hop standing where
  a caller's address should be, and a chain longer than 32 hops. Resolving to the peer there would
  hand the proxy's identity and its grants to anyone who typed junk into a header. Both become
  `unknown`, which is not an address, matches no registration and is refused.
- **`identity.py` is now the only place an address is resolved**, with an **AST** guard rather than
  a substring one (`management.py` had its own `_remote_ip`; a second was a second answer to a
  question with one owner, on the surface where getting it wrong grants the control plane). A
  substring sweep for `client.host` is satisfied by prose — `getattr(getattr(request, "client",
  None), "host", "")` is the same bug and walks straight past it.
- **The management plane shows it**: `GET /rs/v1/admin/config` reports `trusted_proxies` beside
  `admin_nets` split into `builtin` and `operator`, and an unparseable `ROADSTEAD_TRUSTED_PROXIES`
  entry becomes a `hooks.config_notice` — a typo removes trust rather than granting it, which is the
  safe direction and for that reason a completely silent one.
- `docs/api.md` §1.5 rule **4**, §3 and §3.5. `tests/test_trusted_proxies.py`; thirteen mutations,
  every guard observed going red — one of which **survived** the first pass and was real (the
  spelling-sensitive sweep above), and one of which errored at collection instead of asserting and
  had to be rewritten before it proved anything.

### Added — the management plane (Workstream E)

Landed 2026-09-01. **Nothing here is breaking, deliberately**: the plane is additive, the four
pre-existing `/v1/admin/*` control routes keep working unchanged, and §3 mints **no new error code**
— the third time that call has been made (§1.6, §1.7).

- **`/rs/v1/admin/*` — the management plane** (`roadstead/management.py`, `docs/api.md` §3). Six new
  routes: `config`, `keys` (GET/POST), `keys/{key_id}` (DELETE), `callers` (GET),
  `callers/{agent_id}` (PATCH), `providers`. *Why not `/v1/admin`:* `/v1` is versioned by OpenAI, and
  management is the surface most likely to need its own second version. The four legacy control
  routes are served at **both** spellings — same handler, same gate — so no consumer breaks and an
  operator has one prefix rather than two. A test pins the pair, because an alias that stops aliasing
  is a control surface that works on one spelling and 404s on the other.
- **Keys are enrolled and revoked at runtime.** Workstream B made a key the caller's identity and
  left "create one" meaning *edit a file and restart*. A generated secret is returned **once** and
  never stored (the registry keeps the digest); a plaintext secret is never *accepted*, because a
  secret in a request body lands in an access log. Revocation is never refused on provenance grounds
  — a key declared in the environment can be killed now, and the response says the declaration will
  outlive the reason it is dead.
- **🚨 Revoking the LAST key is disclosed, because it changes the identity regime back.** An empty
  registry is not in play (§1.5 rule 2), so a presented key — including the one just revoked — is
  ignored again and the address decides. Nothing is escalated; what stops being true is *revoked
  means refused*. Found by the e2e journey rather than reasoned about, and fixed by **disclosing**
  it: narrowing rule 2 would 401 exactly the deployment that has just emptied its registry on
  purpose. Disclosures arrive in a `warnings` **array**, because two can be true of one action.
- **Caller quotas are editable at runtime** — `weight`, `max_balance_ss`, `default_priority`,
  `degrade_ok`, `spill_ok`, `daily_spend_usd`, the same set `agents.yaml` accepts, pinned against it
  *and* against the dataclass. A weight change reaches the **live** DRR budget (or the edit would
  apply only to callers the proxy has never seen — "it did nothing" for exactly the busy caller it
  was aimed at) and moves the **rate, never the balance**.
- **🚨 §1.6 is inherited at the write boundary, not re-implemented there.** No field can express a
  rejection, so the plane cannot mint a policy admission refuses to honour. An unknown field is a
  400 naming the known set — never a silent drop, on the surface whose purpose is exposing them.
- **A runtime edit never rewrites your config file.** Changes go to a JSON overlay
  (`ROADSTEAD_ADMIN_STORE`) layered over the files at startup: enrolments and overrides on top,
  revocations last. Comments survive, the operator's editor is not raced, and "who changed this" is
  answerable. **A change that cannot be persisted still takes effect and says so** (`persisted:
  false` plus a reason) — refusing a revocation because a disk is read-only is a correctness
  argument answered, in the moment, by a breach.
- **`hooks.config_notice` — a fourth reporting seam, and the first that reports *in*.** Three loaders
  drop unknown keys (`models.yaml` `policy:`, `agents.yaml`, a keys-file entry) and each has cost
  something, because a dropped `spill_ok` is indistinguishable from a caller who never opted in. The
  keys are still dropped and still logged, but now **retained** and readable at
  `GET /rs/v1/admin/config` — the operator who reads a boot log and the one who asks why a knob does
  nothing are not the same person, a week apart.
- **The read views report the GAP, not the config.** `providers` shows declared capacity (the
  catalog seed) beside what is in force and whether the engine publishes it at all, so "discovery
  agreed" and "discovery never ran" stop being indistinguishable. `callers` splits quota into
  `in_force` / `declared` / `runtime`, so an override reads as a difference rather than a label.
- **🚨 Keys stay FLAT, and that answers the roadmap's open multi-tenancy question rather than
  deferring it.** The quota holder, the DRR share and the spend cap are keyed on `agent_id`, never on
  the key — so several keys naming one `agent_id` already give a team one budget with per-key
  revocation and per-key priority. The view groups by it to make that visible.
- **New supporting surface:** `IPIdentityMap.entries()` (registrations only — the built-in internal
  nets are deliberately absent, or an operator would think removing a line closes a door that is
  built in), `Scheduler.agent_snapshot`, `BudgetManager.reweight`, `KeyRegistry.revoke` and key
  provenance. `tests/loop_affinity.py` arms the three newly-mutable objects and grew dotted-path
  resolution to reach the registry behind the identity resolver.

### Breaking

Landed 2026-09-01 with **Workstream C** (the enriched API) and **Workstream F** (hardening). Two of
these touch the wire contract in `docs/api.md`; the rest are internal or additive.

- **🔒 `POST /v1/submit` is REMOVED.** The enriched API at `/rs/v1/*` replaces it —
  `docs/api.md` §1.7, with a field-by-field migration map in §1.9. *Why removed rather than
  deprecated:* the envelope carried no intent vocabulary, no attribution and no timing, and each of
  those would have had to be bolted onto a shape never designed to hold them — three additive
  changes to a contract we had already decided to retire, and a second surface to keep correct
  meanwhile. It also let the body declare `agent_id`, which Workstream B had already had to gate;
  the enriched door takes identity only from the credential or the address, so the fair-share key
  can no longer be named by the request at all. *Migration:* `POST /rs/v1/chat`, `endpoint` →
  `model` (a pin) or `intent` (let Roadstead choose), `timeout_s` → `deadline_s` (and usually: omit
  it), `agent_id` → drop it and present a key. `roadstead.client` speaks the new API and is the
  shortest path across. The OpenAI doors are unaffected.
- **🔒 A new north face: `/rs/v1/*`, versioned separately from `/v1/*`.** Three routes —
  `GET /rs/v1/models`, `POST /rs/v1/plan`, `POST /rs/v1/chat`. *Why its own version:* `/v1/*` is
  versioned by OpenAI. Sharing the number would mean either following somebody else's release
  cadence or publishing a `/v2/chat/completions` that is not OpenAI's v2.
- **🔒 The OpenAI door gains four `X-Roadstead-*` response headers** (§1.8) and **no body fields**.
  *Why headers:* a client validating against OpenAI's schema must not break because it pointed at
  Roadstead, and "we only added fields" is not a defence — strict validators reject unknown keys.
  Additive and ignorable; listed because §1.8 is now contract.
- **`docs/api.md` §1.7 adds NO new error code.** An intent nothing can satisfy is the existing
  `unknown_endpoint`. A caller already classifies that, and a second spelling of "nothing here can
  serve you" would buy nobody anything. The absence is deliberate, as it is for §1.6.
- **`failover.py`'s degrade refusal no longer cites `llmproxy/agents.yaml`.** A dead monorepo path
  in operator-visible error text, carried since the extraction; it now names the knob
  (`degrade_ok`) rather than a file that does not exist here. An error-message reword is a breaking
  change under §2.2 even when the `code` is untouched — this one carries no deferrability marker, so
  no classifier can be affected, but it is recorded because the rule is the rule.

### Added

- **`roadstead.client` — a client SDK, shipped in the package**, the way `roadstead.testing` is.
  `AsyncRoadsteadClient` (real) plus a blocking `RoadsteadClient` that owns a private loop on a
  worker thread. Typed views over the enriched envelope, and typed errors whose `deferrable`
  property **classifies on the `code`** with §2.2's legacy marker substrings kept only as a
  fallback — which is the migration §2.2 asked a shipped client library to make.
  🚨 **It imports nothing from the server**, enforced by AST: a consumer sending an HTTP request
  should not be installing Starlette, uvicorn, PyYAML and jsonschema, and a client reading the
  server's own constants would agree with it *by construction* and could never catch a drift. Its
  contract literals are transcribed from `docs/api.md` and checked against it — the same two-ended
  pin `tests/wire_contract.py` uses from the other side.
- **Model abstraction.** A caller declares an `intent` (a capability profile) and Roadstead owns the
  choice; a `model` is a pin and is treated as a constraint on routing. `roadstead/intent.py` is a
  fifth pure-computation module. Two ranking rules are doctrine: a real-cost endpoint sorts last
  under every preference (or "intent" becomes a back door around the spill doctrine), and an
  endpoint with no latency samples sorts as slow (or the resolver prefers the backend it knows least
  about *because* it knows least about it).
- **Per-request substitution narrowing.** `substitution: {"degrade": false, "spill": false}`.
  🚨 Narrows only: both gates take the AND with the operator's opt-in, so `true` grants nothing.
  Declining is a defer, never an error.
- **`EndpointConfig.kind` and `EndpointConfig.capabilities`**, mirrored whole from the catalog
  stanza. This makes `tool_calling` and `structured_output` load-bearing for the first time rather
  than documentation — the vision-ledger lesson applied to the rest of the block. A capability
  outside the known vocabulary now warns at load instead of silently matching nothing.
- **`GET /v1/models` gains `kind`**, and sorts on it rather than on a literal `{"embed", "rerank"}`
  set. That set was correct for exactly one catalog — the example one — so any deployment whose
  embedder was named something else silently got an embedder as `data[0]`, which is the default a
  client with no configured model picks up.
- **The concurrency invariant is guarded** (Workstream F). `tests/loop_affinity.py` arms it,
  `tests/e2e/test_soak.py` runs sustained overlapping load and asserts thread affinity, slot
  conservation, DRR budget conservation, ledger conservation and that every request was answered —
  and includes a test that mutates budget state from a second thread on purpose, so the guard is
  observed going red from inside the suite forever. `tools/soak.py` is the unbounded version.
- **The "pure computation, no I/O" claim is guarded** (`tests/test_pure_modules.py`). `CLAUDE.md`
  calls those five modules the crown jewels and said "keep them that way"; nothing checked it. What
  purity buys is that every scheduling, costing, deadline, spend and routing decision is testable
  against a fleet that does not exist — and the first `httpx` import into one would end that while
  the suite stayed green.

### Breaking — earlier

Landed 2026-09-01 with Workstream B (identity and API keys), which also clears scrub items **S1** and
**S3**. One of these touches the wire contract in `docs/api.md`; the rest are deployment surface.

- **`/v1/submit` now requires an identity.** It had none: the OpenAI doors were ACL-gated and this
  one was not, on the same port, so any caller could reach it unenrolled *and* claim any `agent_id`
  it liked — including one with a better DRR weight. The fair-share key was self-asserted. It is now
  gated exactly like the OpenAI doors. *Why:* a scheduler whose unit of fairness is caller identity
  cannot let callers choose their own. Migration: present an API key, or enrol the source address in
  `ROADSTEAD_ACL`. Loopback and docker-internal callers are unaffected.
- **A presented API key overrides a body-declared `agent_id`.** An address still only fills in an
  `agent_id` the body omitted. *Why:* a verified credential is a stronger statement about who is
  calling than anything in the body; letting the body win would launder a claim past the credential.
- **The ACL ships no registrations, and no addresses at all.** It carried a private fleet's LAN —
  ten hosts by address, role and deadline floor (scrub item **S1**). A fresh install now allows
  loopback and docker-internal and refuses everything else. *Why:* a default that happens to match
  somebody's LAN hands an identity, a DRR share and a deadline floor to whatever answers at an
  address we guessed. Migration: `ROADSTEAD_ACL=<ip-or-subnet>=<agent_id>[:priority][:min_timeout_s][:admin]`,
  comma-separated. `LLM_PROXY_ACL` is still read — the one legacy env name kept, because a proxy
  that silently stops recognising its callers on upgrade fails closed in the most confusing way
  available.
- **🔒 A new error code, `invalid_api_key` (401)** — `docs/api.md` §2.1, so this one is the wire
  contract. It is deliberately NOT `access_denied`: 403 says *this source is not enrolled*, 401 says
  *this credential is wrong*, and collapsing them sends an operator to the wrong file. A presented
  key that does not resolve never falls back to the address, which is the reason a separate code was
  needed at all.
- **A caller that declares no `priority` now takes its identity's default**, rather than always
  `P1_TURN_SUPPORT`. *Why:* the OpenAI doors already did this; `/v1/submit` ignoring the same
  registration meant one caller landed in two different bands depending on which door it used.
- **`agents.yaml` is a generic example**, on the caller archetypes in `docs/roadmap.md`
  (`chat-assistant`, `coding-assistant`, `summarizer`, `extractor`, `hygiene`). It shipped one
  fleet's agent roster with weekly request volumes and infra paths — a fourth private inventory,
  found the way `usage_rates.py` was found during S2, by walking past it. The same arrangement as
  `models.yaml`: the example IS the default, so it boots a fresh install and cannot rot. The DRR
  sizing reasoning and the whole `degrade_ok` doctrine are kept; only the vocabulary changed. The
  built-in simulation scenarios (`roadstead.simulation`) were renamed to match — their measured
  shapes are untouched.
- **`ANVIL_DISPATCHER_URL` → `ROADSTEAD_ON_DEMAND_DISPATCHER_URL`, and it has no default.** The
  old default was one deployment's dispatcher address (scrub item **S3**). *Why:* the GPU-slot
  dispatcher is a host-side service Roadstead does not own or ship, so a baked-in URL could only
  ever be somebody else's. An `on_demand` endpoint with this unset fails `ensure_loaded` as
  unreachable — the same clean deferrable error as a dispatcher that is genuinely down.

All four below landed together on 2026-08-31 with the catalog redesign, and none touches the wire contract
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

- **`roadstead/spend.py` — money, and the admission decision that spends it** (roadmap Workstream
  D). A fourth pure-computation module beside the scheduler, the cost model and the timeout model.
  - **Admission is ONE decision with THREE outcomes.** `Scheduler._admit` returns
    `DISPATCH` / `SPILL` / `DEFER`. Local capacity is tried first for every caller — nothing about
    money appears above that test — so an over-cap caller, a caller with no `spill_ok` and a caller
    nobody configured all reach the same local dispatch. Spill is considered only once local has
    said no, which is what makes remote capacity *overflow* rather than a parallel system with its
    own fairness, and it never chains.
  - **Two kinds of money, kept in fields that are never summed.** `usage_rates.py` prices a local
    endpoint at what renting the same class of model would have cost — a saving (`avoided_usd`). A
    remote provider's published price is an invoice (`spent_usd`). Only the second counts against a
    threshold: one that counted the first would throttle a caller for using capacity that is free
    and already paid for.
  - **`publishes_token_costs` has a reader.** OpenRouter's catalogue prices arrive on the same
    discovery pass that reads the context ceiling. They are strings, per single token, scaled to
    per-million on the way in — and 🚨 **a published price of zero is a real price**: reading it as
    unpublished would push a free remote model onto the imputed table and book a *saving* for a call
    made over the internet. An operator-declared `policy.input_usd_per_mtok` beats a published one.
  - **Thresholds degrade and never reject.** Crossing `daily_spend_usd` costs a caller one priority
    band (floored at the lowest) and access to paid spill. It never costs local capacity, and 🚨 **no
    error code exists for it** — `docs/api.md` §1.6 says so where `tests/test_spend.py` reads it
    back and fails if a spend-shaped code ever joins §2.1.
  - **New config:** `spill_to` and `policy.input_usd_per_mtok` / `output_usd_per_mtok` on an
    endpoint; `spill_ok` and `daily_spend_usd` on an agent. `spill_to` resolves only to a routed
    endpoint, the same rule `failover_to` follows, which is what makes the shipped example's
    `tier3 -> spill-reasoning` inert until somebody sets `$OPENROUTER_API_KEY` and flips that
    endpoint to `active`.
  - **`/v1/status` grows a `spend` block** — per-caller totals, the price book, spill counters, and
    who is over their cap. The two money columns are reported separately there too.

- **`roadstead/identity.py` — API keys as the caller identity** (roadmap Workstream B). A key
  resolves to a `Principal` carrying the `agent_id` (the DRR fair-share key, quota holder and budget
  holder), a default priority, an optional `min_timeout_s` deadline floor and an optional `admin`
  scope — so all four travel with the caller rather than with the machine it runs on.
  - **Keys are held as SHA-256 digests**, which is both the storage form and the lookup key: the
    plaintext never outlives the load. A config entry may give `key:` (hashed here) or `key_sha256:`
    (the documented form — a key in a config file is a key in a git history). Configure via
    `ROADSTEAD_API_KEYS` for the one-key container case or `ROADSTEAD_API_KEYS_FILE` for anything
    real; `ROADSTEAD_REQUIRE_API_KEY=1` refuses a request that presents none.
  - **One identity-spec grammar for both registries** — `agent_id[:priority][:min_timeout_s][:admin]`,
    where segments are recognised by shape rather than position, so an operator configuring
    `ROADSTEAD_ACL` and `ROADSTEAD_API_KEYS` in the same file learns one spelling of the same four
    facts. Backwards-compatible with the two forms the ACL already accepted.
  - **The address ACL is demoted to a second factor**, and `IdentityResolver` is the one place that
    knows the precedence — the two OpenAI doors, `/v1/submit`, the five admin gates and the deadline
    floor all read the answer off the principal, so a third factor lands in one file.
  - **An interactive identity floored above its own ceiling is reported at load** through
    `hooks.degradation`. That shape was live for a day in the origin fleet — a caller promoted from
    the background band kept the background 1800s floor against a 600s interactive ceiling — and it
    used to be pinned by a test that asserted about specific fleet hosts. It is now a guard that
    fires for anybody's registration, in either registry.

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

- **The structured-output corpus keeps every fixture and loses the vocabulary around them** (scrub
  item **S4**). `tests/corpus/schemas.py`'s four structured cases and five chat-loop cases stay —
  each pins a property no other one does, and they came from prompts that actually ran, which is
  what an invented fixture can never be. What went: case names that were a private deployment's
  agent names, `source=` fields that were `file:line` pointers into a monorepo that resolves nowhere
  here, and `"model": "orchestrator-reasoner"` (both that deployment's vocabulary and a word
  `CLAUDE.md` rejects — the chat cases now name a catalog endpoint class). No schema, grammar,
  output shape or message ordering moved.
- **🚨 And the personal identifiers inside those fixtures, which the scrub plan had ruled out.** Its
  headline finding was that no replay corpus of real traffic came across — true, and checked — from
  which it concluded there was no personal data in the repo. That does not follow: the synthesized
  example prompts were written around whatever was to hand, which included a real full name, a
  household member, a home town, a named local dental practice, and fabricated notices attributed to
  a real utility and a real bank. All fictional now, and the module says so at the top. The
  straggler sweep in `docs/corpus_and_scrub_plan.md` grew a second pattern, because a grep for the
  thing you imported cannot find the thing somebody typed.
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
