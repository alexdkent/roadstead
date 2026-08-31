# Regression ledger

Defects that came back, or that were subtle enough to come back. Format: **symptom → root cause →
guard**. Add an entry *with* the fix, not after it.

> **Seeded, not complete.** The origin monorepo keeps a large regression ledger; only the entries
> that are about *this software* (rather than about a fleet deployment) are reproduced here, plus
> what was learned during the extraction. **Transplanting the rest is an open task** — grep the
> origin ledger for llmproxy-relevant entries. Start with one distinctive word and widen; entries
> are multi-line prose, so a two-term regex on one line will miss them.

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

- 🚨 **A container stop-grace-period must be ≥90s.** `docker stop`'s default 10s window truncates
  the drain in every non-idle case; `stop_grace_period: 90s` (compose) or `--stop-timeout 90`.
  This is the item that was load-bearing and unresolved.
- Re-run `tools/sigterm_drain_probe.py` if either budget changes — the two are independent knobs
  that look coupled.
- **Open, and belongs in the monorepo** (behavioural, so not fixable here under the authority rule):
  the caller of the straggler receives a raw `500 Internal Server Error` at the 48s mark, not the
  proxy's clean JSON error envelope — uvicorn cancels the handler task, which bypasses the
  `exception_handlers` backstop in `build_app`.

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
