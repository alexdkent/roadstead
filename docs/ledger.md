# Regression ledger

Defects that came back, or that were subtle enough to come back. Format: **symptom → root cause →
guard**. Add an entry *with* the fix, not after it.

> **Seeded, and the transplant has begun.** The origin monorepo keeps a large regression ledger
> (17,006 lines, 131 entries); only the entries about *this software* — rather than about a fleet
> deployment — belong here, plus what was learned during the extraction.
>
> **First sweep run 2026-09-01.** 131 entries split on their slug bullets and scored against
> proxy vocabulary; the dense candidates read by hand. Two useful results, and the second is the
> one worth repeating:
>
> - `models-yaml-alias-collision` **transplanted, and it was not a transplant** — the origin
>   *raised* on a duplicate alias and this repo resolved it *silently*. The lesson survived the
>   extraction and the guard did not. Entry below.
> - `thinking-was-a-no-op-and-reported-healthy` **deliberately not transplanted.** Its fix
>   (`Correction.fold_system_for_thinking`) was removed here on 2026-08-22 after all four cells
>   were re-measured live against a different model, and `correction.py` carries that reasoning at
>   the deletion site. 🚨 That is the "look before you fix" rule paying for itself: porting the
>   entry would have argued for restoring a workaround this repo had already retired on evidence.
>
> **Still open.** The three largest origin entries (4,000+ lines each) score high on proxy
> vocabulary incidentally and have not been read. Entries are multi-line prose, so a two-term regex
> on one line will miss them — split on the slug bullets, score whole entries, then read.

---

## An alias claimed twice routed silently to whichever endpoint the file listed first

**Symptom.** None here — which is why it is an entry. In the origin monorepo the same config had a
loud symptom: a gateway dead, the port refusing connections, a restart reporting only
`health-poll timeout (300s)`. `tier3` had been added to a second stanza while still present on the
first, `load_catalog()` raised `name collision` at **module import**, and the process crash-looped
to FATAL and needed its state cleared by hand.

**Root cause.** An alias resolves to exactly one endpoint. A move is a delete plus an add, and the
add lands before somebody remembers the delete.

**What the extraction changed, and why it is worse.** This repo builds the same map with
`by_name.setdefault(alias, e.name)`, so a second claim is not an error — it is discarded, in file
order, in silence. That trades a loud failure for an invisible one: the operator's name now reaches
an endpoint they did not intend, on every request, forever, with nothing anywhere saying so. The
same applies to an alias that collides with another endpoint's *class or role*, which the two later
registration loops overwrite — that ordering is deliberate and correct (a class must be reachable by
its own name) and it still silently kills an alias somebody wrote.

🚨 **Neither behaviour was right.** Refusing to load is not the answer either: this repo's own rule
elsewhere is that a typo must not stop a fleet booting — a dropped `policy:` key is reported, not
raised — and the origin's version of this took a whole gateway down at import over one duplicated
line.

**Guard.** `tests/test_alias_collision.py`. It loads, resolution is unchanged, and both cases report
through `hooks.config_notice` — `duplicate` for two endpoints claiming one alias, `shadowed` for an
alias another endpoint's class or role wins — each naming the endpoint the name actually reaches in
an `in_force` field, which is the management plane's own vocabulary and is read by
`GET /rs/v1/admin/config`. A self-alias is deliberately NOT reported: an alias equal to its own class
maps to itself, is common, and a notice that fires on every boot is a notice nobody reads. The
shipped example is asserted clean for the same reason.

## A fix for an invisible gap was itself invisible for two months

**Symptom.** None — which is the entry. An audit on 2026-07-02 found that WAL-recovered requests
bypassed `handle_submit` and so never contributed to the `context_gate_enforce` shadow evidence, and
added a tally on the recovery path. That tally counted **zero** for every oversized request it was
built for, and went on doing so until 2026-09-01.

**Root cause.** The recovery tally estimated from `req.est_input_tokens`. That field is *cached* by
`scheduler.enqueue`; `recover_queued` constructs a `QueuedRequest` directly from the WAL row and
never reaches the scheduler, so the field is its dataclass default of `0`. `est_in + est_out > limit`
then reduced to `max_tokens > limit`, which is true only for a degenerate request nobody sends. The
other three copies of the same predicate all called `estimate_input_tokens(payload)` and were
correct. `service.py` even imported `estimate_input_tokens` and never used it — the residue of a fix
that meant to call it and reached for the cached field instead.

Two things made it undetectable. A shadow counter has no user: nothing 422s, nothing is refused, and
"the evidence window is not accumulating" looks exactly like "no oversized requests were recovered",
which is the *expected* reading. And the predicate was written out four times, so there was no single
place where the divergence was visible as a difference.

**Guard.** One predicate — `cost_model.context_fit` — called by all four gates, with the per-site
differences (the denominator, and the consequence) kept deliberately outside it. `tests/test_context_fit.py` pins the recovered-request tally firing, and pins as a *fact* that
`recover_queued` does not populate `est_input_tokens`, so the day it starts, the reasoning is
revisited rather than silently invalidated. An AST guard fails if any module re-derives a ceiling
comparison from `estimate_input_tokens` rather than calling the shared predicate.

🚨 **The general shape, which is the reusable part:** a safety predicate written N times is N answers
to one question, and the copy that drifts will be the one whose failure mode is silence. Look for the
other instances of *"three places that must agree, kept in agreement by hand"* — and when one of them
feeds a decision nobody watches (a shadow counter, a flip-review window, a drift alert), weight it
higher, not lower.

---

## A stream's `finish_reason` rides alone on a chunk nobody misses

**Symptom.** A downstream client burned a second model call on **47 turns**, "continuing" answers
that were already complete.

**Root cause.** In OpenAI SSE, `finish_reason` arrives on a final chunk whose `delta` is `{}`. It
carries no text, so a backend that ends a stream without emitting it leaves the client with complete
text and no completion signal — invisible to anyone inspecting content. The client read that as a
mid-stream drop.

**Guard.** The proxy synthesizes the missing terminal chunk — but **only when the backend sent its
own `[DONE]`**, i.e. asserted completeness and merely failed to label it. No `[DONE]` and no
`finish_reason` is a real truncation: nothing is synthesized, it is logged, and the client's drop
handling stays correct. 🚨 **Never collapse those two cases** — the repair becomes a silencer.

A second repair splits a chunk carrying both content and `finish_reason` into two, matching what a
well-behaved backend would have sent.

---

## A flat transport read-timeout capped every caller deadline at 600s

**Symptom.** No non-streamed generation could exceed 600s whatever deadline it declared, and the
failure surfaced as `BackendError(502)` — so a deadline the *caller* set looked like a broken
*backend*.

**Root cause.** The pooled HTTP client was built with a flat `read=600.0`. The real deadline was
passed to `asyncio.wait_for`, but httpx stopped reading first, and `ReadTimeout` maps to an HTTP
error, not a timeout.

**Guard.** The transport timeout now tracks the request deadline (with margin, floored at the
historic value) so `wait_for` is always the layer that fires and the failure is typed as a TIMEOUT.
Measured before/after on the same call: 600.0s `status=error` → 770.3s `status=ok`.

---

## `--cache-reuse` is architecturally refused, so adding it guarantees nothing

**Symptom.** Two llama.cpp chat backends recomputed a fixed ~2,052-token prefix on every call, even
for a byte-identical repeated prompt. The leading hypothesis was a missing `--cache-reuse` flag.

**Root cause.** The flag was added and the loader **disabled it both times**, with two different
messages: multimodal contexts and non-shiftable attention memory each refuse it outright. The real
cause is `-ub`: the floor is exactly `-ub + 4`, proven by moving it.

**Guard.** Doctrine, not code: lowering `-ub` is a measured trade (−27% prefill), not a fix. Against
the real prompt-size distribution it is a net loss, so short-prompt reuse on those backends is
**structurally unavailable**, not a bug to chase.

---

## A green suite that ran nothing, twice, in one extraction

**Symptom.** During extraction, two verification steps reported success while proving nothing.

**Root cause.** (1) A test-run command piped through `| tail`, so the reported exit code was
`tail`'s, not `pytest`'s — a run with a real failure exited 0. (2) A structural check computed its
paths wrongly, found **zero** files, and printed "self-consistent" because its loop body never ran.

**Guard.** Never pipe a command whose exit status matters. Any check that iterates must refuse to
pass on an empty set — assert a minimum count before reporting success.

---

## SIGTERM's two shutdown budgets are serial, not nested

**Symptom.** The origin knowledge layer contradicted itself: one source said SIGTERM runs a clean
bounded drain, another said it hangs behind slow in-flight requests and SIGKILL is correct. Both
turn out to be describing the same behaviour from different ends.

**Root cause.** Measured with `tools/sigterm_drain_probe.py` (2026-08-31, three runs, ±0.05s):

| in-flight state | SIGTERM → exit | budgets + completions persisted |
|---|---|---|
| idle | **0.19s** | n/a |
| one dispatch finishing inside the drain | **7.67s** | yes |
| one dispatch outlasting the drain | **78.25s** | yes |

uvicorn's `timeout_graceful_shutdown` does **not** bound `ProxyService.shutdown`. It bounds the
in-flight HTTP *connections*; only once it expires does uvicorn send `lifespan.shutdown`, and at
*that* point the app's own `_DRAIN_DEADLINE_S` drain begins. uvicorn never bounds the lifespan
shutdown at all. The log timestamps show it exactly: `draining 1 in-flight dispatch(es) (≤30s)` is
emitted ~48s after the signal, and `drain deadline hit` exactly 30.000s after that.

So the worst case is the **sum**, `timeout_graceful_shutdown + _DRAIN_DEADLINE_S + close-tail`
(≈78s observed), not the larger of the two. The comment at `__main__.py` reasons the other way —
"uvicorn's budget MUST exceed drain + close-tail (+margin) or it hard-kills the process mid-flush" —
which assumes a nesting that does not hold. The margin is real but it is not doing what it says.

**Guard.** SIGTERM **is** the correct signal and the question is closed: the drain completes, and it
persists the DRR budget row and the completion row even for a straggler it had to cancel — exactly
what SIGKILL would lose. But the drain is slow enough to matter:

- 🚨 **A container stop-grace-period must be ≥108s.** *(Was ≥90s until 2026-09-01 — see "the tail
  had no ceiling" below for why the number went up rather than down.)* Measured, in a real
  container on a real Docker daemon (`tools/docker_stop_probe/`, 2026-08-31):

  | `docker stop` | in flight | outcome | persisted |
  |---|---|---|---|
  | `-t 10` (**the default**) | work outlasting the drain | **SIGKILL, exit 137** at 10.08s | **budgets 0, completions 0 — everything lost** |
  | `-t 10` (the default) | 5s of work | clean, 2.88s | budgets 1, completions 1 |
  | `-t 90` | work outlasting the drain | clean, **78.31s** | budgets 1, completions 1 |

  Two things worth staring at. The default **works when the proxy is quiet** — which is exactly how
  it will be tested, and exactly why the failure would first appear under load, in production, with
  no alarm. And in the killed case the log stops after uvicorn's `Shutting down`: SIGKILL landed
  while it was still waiting on the connection, so the app's shutdown handler **never ran at all** —
  no drain, no flush, no budget persistence.

  78.31s in a container against 78.25s bare-process: the two agree to 0.06s, so the cost is the
  drain itself and not container overhead. Set `stop_grace_period: 90s` (compose) or
  `--stop-timeout 90`. `Dockerfile` carries the requirement as a label so the image documents it.
- Re-run `tools/sigterm_drain_probe.py` if either budget changes — the two are independent knobs
  that look coupled.
- 🚨 **The tail had no ceiling, and 90 was a measurement rather than a bound (closed 2026-09-01).**
  The drain itself was bounded at 30s; everything after it was not. In particular
  `OnDemandManager.close` makes a **network call per held GPU lease**, so a dispatcher that stopped
  answering hung the lifespan shutdown for as long as it liked — past any stop-grace an operator had
  set, at which point the container SIGKILLs a process that has not flushed anything. The observed
  78.25s was a tail that happened to be fast.

  Every phase is bounded now — drain 30s, straggler unwind 3s, lease-release and pool-close 5s
  (concurrently), queue flush+join 10s — summing to a **48s ceiling on the lifespan shutdown**, which
  with uvicorn's own 48s and 12s of margin gives **108s**. It is computed once, in
  `service.RECOMMENDED_STOP_GRACE_S`, and `tests/test_shutdown_budget.py` fails if the Dockerfile
  label, the prose or the arithmetic disagree. So the recommendation went **up**: 108 is the first
  number that is a ceiling instead of an observation.

  Two things deliberately NOT done. There is no outer `wait_for` around `shutdown()` — it would
  cancel `queue_db.close()` mid-flush, which is the SIGKILL failure the drain exists to avoid, and a
  half-written completion row is worse than a slow exit. And uvicorn's `timeout_graceful_shutdown`
  was **not** retuned: it is now its own knob rather than `_DRAIN_DEADLINE_S + 18`, because that
  derivation encoded the nesting this entry disproves, but its VALUE is unchanged so the drain
  semantics settled by experiment stay settled.

- **~~Open, and ours to fix.~~ Fixed 2026-09-01.** The caller of the straggler received a raw
  `500 Internal Server Error` at the 48s mark, not the proxy's JSON envelope. **Root cause:**
  `asyncio.CancelledError` derives from `BaseException`, not `Exception`, so uvicorn's
  `task.cancel()` sailed past both the `exception_handlers` backstop in `build_app` and Starlette's
  own `ServerErrorMiddleware` — neither of which can see a `BaseException`. Parked as "belongs
  upstream" while the origin monorepo was authoritative; that rule was retired 2026-08-31.

  **Guard.** `ShutdownEnvelopeMiddleware`, and `tests/test_shutdown_envelope.py`. It is a
  *middleware* because no `exception_handlers` entry could ever be reached: Starlette dispatches
  those from inside an `except Exception`. User middleware sits outside the router and inside
  `ServerErrorMiddleware`, which is the outermost place a route's `BaseException` is still
  catchable. Measured with the probe, before and after:

  | | caller status | body | elapsed |
  |---|---|---|---|
  | before | `500` | `Internal Server Error` (plain text) | 48.77s |
  | after | `503` | `{"code": "draining", … "backpressure"}` | 48.76s |

  Three things are deliberate. 🚨 **The cancellation is always re-raised** — swallowing it would
  leave uvicorn waiting on a task that declined to die, converting the bounded shutdown into the
  unbounded one this whole entry is about. 🚨 **A cancelled STREAM gets an error frame and no
  `[DONE]`**: `[DONE]` is an assertion of completeness, and emitting one over a truncated answer is
  the exact collapse the `finish_reason` repair refuses to make. And it **mints no new code** —
  `draining` + 503 + the `backpressure` marker is precisely what `lifecycle` already emits for work
  *refused* while draining, because the caller's situation is identical: this instance is going
  away, the next one can serve you.

---

## The fake backend was more generous than any real engine

**Symptom.** None — and that is the point. Capacity discovery worked against the fake and would have
kept working after a "simplification" that broke it against every real backend.

**Root cause.** `roadstead.testing`'s `/props` published a superset of what llama.cpp actually
returns. Measured against a real `llama-server` (b5350) at `--ctx-size 8192 --parallel 4`
(`tests/wire_fidelity/`, 2026-08-31), the whole top level is:

    bos_token · build_info · chat_template · default_generation_settings
    eos_token · modalities · model_path · total_slots

So a current build publishes **no top-level `n_ctx`**, **no
`default_generation_settings.n_parallel`**, and **no `slots` list** — three fields the fake was
emitting. Consequences, in order of how much they matter:

1. **`health.py` prefers `default_generation_settings.n_parallel` and falls back to `total_slots`.
   Only the fallback exists on a real engine.** Against the old fake the preferred field was always
   present, so the path real discovery entirely depends on was never exercised. Deleting it as
   redundant would have left the suite green and broken capacity discovery in production.
2. The **top-level `n_ctx` question is settled, and neither candidate answer was right.** `health.py`
   divides it by the slot count (i.e. reads it as an aggregate) with a comment conceding it is
   "unconfirmed whether it's ever populated as an aggregate". It is not populated at all. The
   fallback is dead code against a current build, kept for older ones. The fake had been emitting it
   with the same value as the per-slot field, which cannot be right under either reading.
3. `default_generation_settings.n_ctx` reported **2048** at `--ctx-size 8192 --parallel 4` —
   **per-slot confirmed** on a build 611 versions newer than the one the original note was written
   against. The guard against re-dividing it holds.
4. `/v1/models` differs per engine more than the fake did: llama.cpp returns the **full GGUF path**
   as `id`, **no `root`**, **no `max_model_len`**, and a `meta` block
   (`n_params`/`n_vocab`/`n_ctx_train`/…). So `probe_model_fingerprint`'s `root` path is vLLM-only
   and its `meta` fallback is the one that carries llama.cpp — and the fake, emitting neither, had
   never exercised either.

Also confirmed: `finish_reason` really does ride alone on a terminal chunk with an empty delta.

**Guard.** `roadstead.testing` now defaults to the verified narrow shape and emits `/v1/models`
per-engine, so the fake is as stingy as the real thing — a fake being *more* generous than reality is
the dangerous direction, because it makes a passing test the reason a real path is never run. The old
superset survives as `props_profile="legacy"` for builds that do publish those fields.
`tests/wire_fidelity/` holds one contract that both the fake (every run) and a real engine (opt-in)
must satisfy, so the next divergence fails there.

---

## The extraction's own near-miss: a commit that imported files it did not contain

**Symptom.** The origin repo's `main` briefly held an `ImportError` — four edited modules were
committed while the two new modules they import were not.

**Root cause.** The ship harness resolves commit scope from `git diff`, which **cannot see untracked
files**. A brand-new module is therefore reported as *outside scope* rather than as *missing*. Every
gate passed because deployment ships file paths, not git objects — nothing in the validation chain
read git.

**Guard.** Open, in the origin project: `commit_gate` should fail, not warn, when an excluded file is
imported by an included one. Relevant here as a warning about any future tooling that reasons about
"what changed" from a diff.
