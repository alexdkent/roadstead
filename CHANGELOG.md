# Changelog

Notable changes to Roadstead. Breaking changes get their own section with a reason, per
`docs/compatibility.md` — the stable surface is the wire contract in `docs/api.md`; everything else
is internal and changes without an entry.

Pre-1.0: breaks are permitted, but each one is a recorded decision rather than a surprise.

Everything before 0.1.0 is in **`docs/changelog-archive.md`** — the sixty-five entries written while
the project was being extracted from a private monorepo. Those are the reasoning; this file is the
summary. The bullets below link in there where the long version is worth reading.

## Unreleased

### Added

- **A structured-truncation 502 (`truncated structured output`, §2.2) now carries `partial_content`
  — the text the backend actually generated before the cut.** Additive on every door that can emit
  the marker (legacy §1.9.4, enriched §1.7.3, and nested inside the OpenAI door's `error` object);
  the status, message and `code` are unchanged. Without it a caller's truncation-salvage path (repair
  a cut-off JSON array by dropping the partial trailing item) had nothing to repair — the 502 simply
  discarded the body. Measured 2026-09-24: a kv4 probe-generation call produced 52 complete, valid
  JSON items in 5000 tokens before truncating, and the caller received an empty string. Never sent on
  the degenerate-generation (repetition-loop) branch, which emits a distinct marker — that output is
  garbage by definition and not worth salvaging. Can be large (tens of KB for a long generation) and
  is never truncated on the way out.

- **An endpoint can declare the decode sampling to use while REASONING
  (`policy.thinking_temperature` / `policy.thinking_top_p`).** A reasoning model's failure mode at
  low temperature is not a worse answer, it is NO answer: an easy cyclic continuation out-competes a
  hard progress-making one, and that competition sharpens as temperature falls (arXiv 2512.12895).
  Measured on a DeepSeek-V4-Flash reasoner, one hard prompt, greedy decode — 4 of 6 draws cycled
  inside the reasoning channel, consumed the whole `max_tokens` and returned zero answer characters,
  worst case over an hour; at the vendor's published recipe the same prompt converged 3 of 3. That
  number belongs to the weights, so it belongs to the endpoint stanza rather than to a call site,
  which cannot know it. Do not expect a serving engine's generation-config setting to supply it: a
  checkpoint's own `generation_config.json` commonly carries the engine's neutral defaults, in which
  case every such setting decodes identically and the published recipe reaches the wire from
  nowhere. Applied at the provider seam so it covers every reasoning caller, fill-if-absent so a
  caller's own sampling wins, and inert unless declared. See `docs/api.md` §3.13b.
- **`GET /v1/status` reports `build_sha`** — the source commit the process was built from, from
  `ROADSTEAD_SOURCE_SHA`, or `unknown` when the deployment did not stamp one. A source SHA and not
  an image tag on purpose: a tag is customarily rewritten in place on each deploy, so it names a
  stream rather than a commit, and an operator debugging through the proxy cannot otherwise tell
  which code is answering without shell access to the host. `ROADSTEAD_SOURCE_SHA` joins the stable
  set — renaming it later would not fail a deploy, it would quietly make every deploy report
  `unknown`, which is the exact blindness the field exists to remove.

- **A streaming completion's `proxy_completions.response_json` now names which model served it.**
  Every non-streaming completion has always attached the backend's response body; a streaming one
  never did, because `execute_streaming`'s terminal `record_completion` calls passed no
  `response_body` at all — measured fleet-wide, 264/264 streaming rows had `response_json IS NULL`
  against 0/1332 non-streaming, making "which model answered?" unanswerable for every streaming
  caller. The captured object is deliberately tiny — `{"model": ..., "model_source": ...}`, never the
  streamed content — to avoid changing data volume or privacy posture for every streaming call on the
  fleet. `model` prefers what the BACKEND ITSELF echoed in a stream chunk's top-level `model` field
  (`model_source: "backend_echo"`, evidence of what actually ran) and falls back to the endpoint's
  declared `effective_model_id` only when no chunk carried one, e.g. a cancel/timeout/error before
  any chunk arrived (`model_source: "endpoint_config"`, the proxy's own belief, not a measurement).
  Fail-open, matching every other per-completion tally in `record_completion`: a capture failure
  loses the tag on that one row, never the completion or the stream itself.

  **Confirmed live 2026-09-17 (fleet build 2ed872b), not just assumed:** every SSE chunk of a real
  streaming request (19 of 19) carried the backend's own served-model id in its top-level `model`
  field, byte-identical to the id the non-streaming path already records for that same endpoint —
  and every configured chat endpoint checked the same way echoed its own distinct id. So in
  production this takes the `backend_echo` branch, not the `endpoint_config` fallback, which is
  what makes a downstream reader's gate on this field strong rather than a guess about engine
  behaviour.

- **A `response_format` json_schema is served as a FORCED TOOL CALL on a backend that declares no
  constrained decoding (`Correction.apply_forced_tool_schema` + `finalize_forced_tool_schema`).**
  Some builds ship without a grammar engine, and the honest ones REFUSE the field: a candidate
  backend evaluated 2026-09-13 answers both `{"type":"json_object"}` and a strict
  `{"type":"json_schema", …}` with an immediate HTTP 400 naming `json_schema` as not implemented by
  that build. The refusal is right — a schema it cannot enforce would return prose the caller then
  fails to parse — and it is also fatal to every caller that declares a schema, which on a fleet
  that took "always declare a schema" seriously is most of them.

  The capability is present on such a build; it is reached through the tool-argument path. Same
  model, same schema as a forced tool call (schema as the function's `parameters`, `tool_choice`
  naming the function): **29/29 fully schema-valid** on the hardest schema to hand — nested objects,
  `["string","null"]` unions, `maxLength` rails, objects inside an array's `items`,
  `additionalProperties: false` closed objects, `minItems == maxItems == n` exact counts (14/14 on
  the count checks), zero empty tool calls, integers typed as integers. On one incumbent mid-tier
  backend the forced tool call was also the faster path (4.7 s median vs 7.5 s, 8/8 either way) — but
  on a small classifier endpoint it was **2.2x SLOWER** (9.1 s vs 4.1 s, 8/8 either way). Which path
  is faster is a property of the BACKEND, not of the mechanism, so it is never a reason to translate
  an endpoint that can already do this itself.

  🚨 **The response half is the whole point.** A caller that sent `response_format` parses
  `choices[0].message.content`. Rewriting only the request hands every such call site `content: null`
  beside a `tool_calls` array it does not read — a loud 400 converted into a silent empty parse,
  which is strictly worse than the incompatibility. So the forced call's `arguments` become
  `content`, the synthesized tool is removed, and `finish_reason` becomes `stop` (a `length` finish
  is preserved: a truncation must stay visible). `tests/test_forced_tool_schema.py` drives a real
  `ProxyService` for both directions, because two units that agree can both be unreached.

  🚨 **Gated on the BACKEND'S OWN `/health`, not on the catalog.** Such a build publishes
  `"not_implemented": ["response_format", "text.format"]`; the capacity poller reads it every pass
  into `EndpointConfig.not_implemented` (`backend.probe_not_implemented` — one small GET per CHAT
  endpoint per pass, never on the request path) and the translation fires only on that positive
  statement. **Absence never fires**: a build publishing no such field (every incumbent engine), an
  unreachable `/health`, a non-200, a non-JSON body and a malformed value are one answer — "not the
  backend saying it lacks the feature". A failed read CLEARS the reading rather than leaving it
  standing, because it describes a BUILD and a build changes across a restart; that fails toward the
  caller getting the backend's own 400, which is loud and recoverable on the next poll.

  The first cut of this gate read the CATALOG instead — `tool_calling` declared, `structured_output`
  absent — and it was wrong in a way worth recording. Against a live fleet catalog that conjunction
  matched exactly ONE endpoint: a production classifier on the prompt-injection/PII path whose stanza
  merely omits `structured_output` while the backend implements it fine. It would have bought 2.2x
  latency there for zero correctness gain. **An omission in a catalog is indistinguishable from an
  incapacity**, so nothing that REWRITES a payload may key off one — now written next to the
  `capabilities:` legend in `models.yaml`, since intent RESOLUTION legitimately does read that block.

  A request carrying its own `tools`/`tool_choice` is left untranslated and takes the backend's 400,
  because forcing the schema function would suppress the caller's tool call. Streaming is declined
  outright for the same reason the response half exists: the body is already on the wire.

  What the removed `response_format` also bought is re-asserted rather than dropped — the
  truncation-integrity gate, the JSON parse floor and the declared schema every response-side guard
  validates against, now reached through `QueuedRequest.forced_tool_schema`. New disclosure token
  `forced_tool_schema` in `X-Roadstead-Corrected` / `corrections` (docs/api.md §1.8); no new error
  code (an unusable forced call is the published `toolcall_truncated`, which is what it is).

- **Endpoint-level goodput-collapse detection (`roadstead/goodput.py`), shipped DARK.** A backend
  whose engine wedges keeps answering `/health` — the HTTP server is fine, only the engine is not —
  so its slots fill with requests producing almost nothing, callers hit their deadlines, and the
  retries refill the slots. Measured on a real incident: the scheduler step sat at ~7 s for two days,
  226 timeout events, `healthy: true` throughout. The existing per-request stall watchdog worked and
  could not help: its conclusion is always "THIS request is stuck", never "this endpoint is sick".

  The signal is the backend's **own engine work counters**, sampled on the capacity poller: an
  occupancy gate plus up to three progress clauses (scheduler iterations, generation tokens per busy
  slot, prefill tokens), sustained over consecutive evaluations. Measured precision **1.000** (294
  firings, none outside a real timeout window) and 60.7% harm-minute recall, firing 2 minutes after
  the first timeout of the second day with 42 of that day's 43 events still to come.

  🚨 **One term is measurably load-bearing and the other three are not.** Ablation over the same
  4,264 minutes: removing the occupancy gate collapses precision to **0.425** (415 false positives);
  removing any single progress clause leaves it at 0.997-1.000. The triple conjunction is justified
  by MECHANISM — loop not turning / turning but emitting nothing / prefill-only — not by a measured
  precision gain. `tests/test_goodput.py` sabotages each clause separately, because a compound guard
  passes its own suite with a clause silently dropped.

  Enforcement is behind the `goodput_collapse_enforce` runtime flag, **default False** — so today it
  samples, evaluates, latches, counts, publishes and alerts, and changes nothing a caller can see.
  Arming it later is a `POST /v1/admin/flags` flip, not a code change. Recovery is bounded in both
  directions: consecutive healthy evaluations clear a trip, and an absolute maximum hold clears it
  regardless, including while blind — there is deliberately no path that stays tripped forever.

  🚨 **Thresholds ship as NOTHING.** The five numbers are one fleet's hardware measurements, and an
  endpoint with no `policy.goodput_*` declared in `models.yaml` **runs no detector at all** — it does
  not even pay the scrape. See `docs/api.md` §3.13 for the field reference and the worked example.

- **`abort_reason` is populated on every timeout layer.** It was `stream`-only, so five call sites
  defaulted to `NULL` — which is why **219 of one incident's 226 timeout events were unexplained**
  and `GET /v1/timeouts`' `by_abort_reason` rollup was blind to 97% of the population. Each site now
  names its own bound: `client_deadline`, `sse_consumer_deadline`, `admission_expiry`,
  `queue_deadline_exhausted`, `backend_transport_deadline`. None of them is added to
  `STALL_ABORT_REASONS` — every one is a bound we or the caller chose expiring, and counting them
  would let a capacity decision masquerade as a backend failure. `goodput_collapse` IS added, because
  a request refused at the door on measured evidence that the engine is broken is the substrate
  dying under a caller who did nothing wrong. Full table in `docs/api.md` §3.4.

- **Two engine counters on the `/metrics` scrape** (`backend.probe_progress_counters` now returns
  `iterations` and `running` beside `prompt` and `generation`). 🚨 `vllm:iteration_tokens_total_count`
  is matched on its FULL name: it is the `_count` of a histogram whose `_bucket`/`_sum`/`_created`
  siblings share the family prefix, and a prefix match sums them into a number that rises
  monotonically, graphs beautifully, and is garbage. The programmable fake backend lays that trap by
  default rather than on request.

  🚨 **Every absent counter is `None`, including `prompt` and `generation` — which were briefly
  coalesced to integer `0`, and that was a latching break of the detector's central invariant.** The
  prefill and generation clauses read exactly those two keys, so a counter the backend does not
  publish *satisfied* them: the verdict came back `COLLAPSED` with an empty `reason`, the breaker
  latched, and `roadstead_endpoint_goodput_unknown` read **0** — the metric whose only job is to make
  blindness visible reported nothing. Reproduced end to end on the supported llama.cpp 3-clause
  configuration against a body carrying only `llamacpp:requests_processing`. The same coalesce also
  swallowed every **malformed line**, since the parser drops what it cannot parse: a trailing space, a
  TAB separator, a `NaN`, or a body truncated mid-stream each became a zero — and a wedged engine is
  exactly when `/metrics` is slowest and a body most likely to arrive short, so that failure was
  *correlated with the condition being detected*. `lifecycle._stream_progress_probe`, the one caller
  that wants integers, coalesces at its own call site, where "absent counts as no progress" is the
  behaviour it has always had and deliberately wants.

  **Why the unit suite could not see it:** every absent-counter test used `iterations` or `running` —
  the two keys the scraper could already return as `None` — or stubbed the whole scrape to `None`.
  Nothing passed an absent `prompt` or `generation` into the monitor. The monitor was correct
  throughout; its producer was not. The regression tests now drive the real scraper into the real
  monitor, which is the only reading that could have caught it.

  **Three further exposition shapes now read as absence**, and the middle one is the same
  "easier-to-fire" class as the coalesce: `Inf`/`+Inf`/`1e400` (`int(float("Inf"))` raises
  `OverflowError`, which the per-line handler did not catch — it escaped and discarded the *whole*
  scrape, contradicting the function's own documented promise that a bad line drops only its own
  counter); **a repeated series with an identical label set** (summed, that inflated `running` 6 → 12,
  pushing occupancy past its gate *and* shrinking the per-request denominator at once — summing across
  *different* label sets remains correct and deliberate, because data parallelism publishes one series
  per engine); and a **negative** value (impossible for a token counter or a concurrency gauge, and
  its 0.0 rate satisfies every `<` clause). Absence is sticky per counter, so a body's meaning does
  not depend on the order the engine emitted its lines. Table in `docs/api.md` §4.4.

- **Two alert conditions**, on the existing `check_alerts` path so they are TSDB-visible for free:
  `endpoint_goodput_collapse` (CRITICAL when enforcing, WARNING in shadow) and
  `endpoint_goodput_blind` — a detector that is configured and cannot reach a verdict. The second is
  not decoration: refusing a verdict is the SAFE behaviour on a missing clause, which means a
  permanently blind endpoint otherwise looks exactly like a quiet healthy one. The pre-existing
  `endpoint_stalled` alert is untouched and still fires on its own signature; it keys on completed
  timeouts and requires `queued == 0`, which a collapse that fills every slot fails by construction.

### Breaking

- **`models.yaml` endpoint field `token_speed` renamed to `decode_tok_s`.** Nothing outside this
  repo has populated the field yet, so the blast radius is zero today. Two collisions forced the
  rename: (1) the fleet catalog this repo mirrors already uses `token_speed` for an unrelated
  `dict[str, Any]` shape served over the wire — a scalar in either catalog would raise `TypeError`
  in the other, and the two catalogs are kept equal by a reconciliation test; (2) worse, that
  fleet's own budgeting code documents `token_speed` as a **single-stream idle benchmark**, known to
  run ~2x optimistic versus real contended traffic — enforcing a floor on it would reproduce the
  exact "deadline the call physically cannot meet" bug the floor exists to prevent. `decode_tok_s`
  names the quantity the floor actually needs: a conservative rate for the slow tail of real
  traffic, not an idle number.

- **`POST /v1/chat/completions` now reads `session_id`/`turn_id` off the body, records them on the
  completion row, and pops them so they never reach the backend.** Before, §1.1 documented both as
  ignored-and-forwarded: unread by the proxy, left in the payload, and left for a strict backend to
  possibly reject as an unknown field. A caller who happened to send either now gets different wire
  behaviour — the field disappears from what the backend receives — which is why this is `### Breaking`
  and not `### Added` even though the change is additive in spirit. The reason: a downstream reader of
  `proxy_completions` needs to know WHICH JOB made a call, not just which model answered it, and the
  fleet's own agentic CLI stamps `session_id` on every request it makes — through this door, where it
  was silently dropped. `agent_id`/`caller_id`/`priority` are unchanged: they still come from nowhere
  but the resolved principal, never the body — a correlation id lets a caller tag its own request, not
  relabel who it is. Treated as untrusted input from an internet-facing door: non-string or longer than
  256 chars is silently dropped rather than causing a 400. `docs/api.md` §1.1 updated in the same
  commit (moved out of "What this door does NOT read" and into the main table).

### Fixed

- **A caller's own reasoning effort, sent the OpenAI way, drew a 400 on any endpoint declaring
  `policy.reasoning_effort`.** "A caller's own pin wins" was checked against
  `chat_template_kwargs` only, so a top-level `reasoning_effort` (plain OpenAI clients, agent
  harnesses) or `reasoning: {effort}` (the OpenRouter shape) got the declared default injected
  BESIDE it, and an engine that validates the pair refused every such request: `conflicting
  reasoning_effort: 'medium' at the top level and 'low' in chat_template_kwargs`. The caller's value
  now moves into `chat_template_kwargs` (the one channel a declaring endpoint is known to read) and
  the alias is removed, so the pin wins and arrives under one name; `"none"` sets the declared
  thinking switch off. A caller that sends two different values itself is left as sent. See
  `docs/api.md` §3.11.
- **`/v1/fleet/savings` reported a rolling 30-day window under the word "total".** `savings_summary`
  summed `proxy_completions` with no lower bound, and `cleanup_old_completions` prunes that table at
  `completions_retention_s` — so the figure a dashboard labels "saved total" stopped growing once the
  retention window filled. Measured against a live deployment: the endpoint returned `total_usd`
  554.31 and the sum of its last 31 daily buckets was 554.32. The total is now composed with a new
  never-pruned `proxy_savings_daily` table (endpoint x UTC day, one row each), finalised out of
  `proxy_completions` by the retention sweep before it deletes anything; existing databases are
  backfilled once from whatever completions they still hold, so the oldest bucket starts short by
  however much a previous prune already took. `today_*` is untouched — same local-midnight boundary,
  same code path.

  It stores TOKENS, never USD. The rates in `usage_rates.py` change (the thinker anchor was repriced
  on a model cutover) and the unpruned half of the figure is already priced at CURRENT rates, so
  freezing each day at whatever rate applied when it was finalised would mix pricing regimes inside
  one number and make the total impossible to recompute.

  🚨 **The finalisation and the DELETE are not a transaction, and cannot be made into one:** `_w`
  drops a write when its bounded queue is full — deliberately, so DB I/O never blocks the event loop.
  A dropped finalisation followed by a landed DELETE would destroy those tokens permanently, which
  for a lifetime counter is unrecoverable. So neither statement assumes the other ran. Finalisation
  is a MONOTONE upsert (`MAX` of stored and recomputed), which makes it idempotent and makes a
  partially-pruned bucket's low recompute a no-op rather than a clobber; the DELETE carries an
  `EXISTS` against the rollup, evaluated inside the statement on the writer thread, so a row whose
  bucket never reached the rollup survives the sweep and is pruned by the next one. Both properties
  are sabotage-verified in `tests/test_savings_lifetime.py`: reverting the upsert to a plain
  overwrite shrinks a straddling bucket 2M → 1M tokens, and dropping the `EXISTS` loses two rows'
  tokens the moment a finalisation write is dropped.

- **A reasoning cap on a tool-calling turn is now derived from the top of the allowance
  (`max_tokens` minus a fixed answer reserve) rather than from the ratio or the operator-declared
  absolute.** A cap that cuts reasoning SHORT on a tool turn corrupts the tool-call channel: the
  control tokens are emitted as garbled literal text in `content` and `finish_reason` degrades
  `tool_calls` → `stop`, so the caller gets a confident prose answer and no side effect. Measured
  dose-response against a turn whose natural reasoning is ~210 tokens: budget 64 → 0/4 tool calls,
  128 → 0/4, 256 → 3/4, 512 → 4/4, no budget → 4/4 — it is the severity of the cut, not "binding"
  as such. The previous release suppressed injection entirely on such turns, which restored the
  runaway the cap existed to bound: at `max_tokens=5000` with tools declared, reasoning consumed the
  whole allowance and the empty content channel returned **502** twice, while the no-tools control
  completed at 3,017 tokens. Both failures are the same quantity read from opposite ends — how much
  of the allowance is left for the answer — so the cap is now placed there. The ratio, the absolute
  and the ratio path's floor/ceiling stay off the tool path; each is a fraction-of-allowance or
  plain-generation number that can land in the failure zone (the 16,000 ceiling would cut an agentic
  turn whose worst observed reasoning block was 16,562). Where `max_tokens` is too small to place a
  cut above natural reasoning, nothing is injected and the residual runaway risk is accepted, since
  a suppressed tool call is silent and a 502 is not. Upstream vLLM #39697 and #44676 are both open.
  `docs/api.md` §3.11.

- **A structured `finish_reason=length` response that was actually a repetition LOOP (usually
  whitespace) was always classified as a benign truncation** — measured 26 times in 14 days on
  tier3, ~125 minutes of wasted decode. Two independent gaps let it through: the dispatch path for
  a truncated structured request `resolve_error`s and `return`s before `Correction.apply` (and its
  egress degeneration guard) ever runs, and `_is_degenerate_text`'s word-shingle detector floors at
  40 words, so a pure-whitespace body — zero words under `str.split()` — sailed past it. The
  downstream cost: the caller was told "truncated", and every truncation-recovery path in the fleet
  responds to that by re-asking with more tokens, pouring more decode into a loop that never
  terminates. `_is_degenerate_text` gained a second, character-level TAIL arm (`_DEGEN_TAIL_*`,
  calibrated against 12,899 real completions — a blank-ratio check and a distinct-24-gram check,
  since the measured shape is "legitimate prefix, then loop" and a whole-body ratio dilutes on the
  good prefix) and the structured-length dispatch path now runs it before calling the response a
  truncation. Ship dark behind the new `degenerate_length_enforce` runtime flag (`flags.py`,
  `POST /v1/admin/flags`): shadow counts + logs under the existing `degeneration_detected` /
  `degeneration_by_call_site` tally; enforce swaps the caller-visible error to a distinct
  "degenerate structured output" marker that deliberately does NOT contain the `"truncated
  structured output"` substring truncation-recovery callers match on.

- **`GET /v1/timeout-advice` could recommend a deadline below the physical decode time of the
  output being asked for** — a sparse/thin high-`est_out` bucket could never accumulate the
  `status=="ok"` samples that would have corrected it, because every call at that size timed out.
  `TimeoutModel.advise()` now floors `recommended` at `(est_out / decode_tok_s) * margin` for any
  endpoint with a declared `decode_tok_s` (new, optional `models.yaml` endpoint field — a fleet
  measurement, absent by default, so an endpoint nobody profiled is unaffected). The same floor is
  threaded into `effective_timeout_advice`'s ceiling resolution, so an INTERACTIVE call's tighter
  600s band cannot clip the deadline back down below what decode alone requires — closing the same
  bug on the caller-facing path, not just the raw advice.

## 0.1.1 — 2026-09-06

The first release anybody should install. Functionally 0.1.0 plus the documentation and hardening
commits that followed it; the version number moved because **0.1.0 was deleted from PyPI**, and a
deleted version can never be re-uploaded.

0.1.0 was published on 2026-09-05 and withdrawn about three and a half hours later. Its sdist
shipped `tests/`, and both artifacts carried internal names and backend ports from the private
monorepo this project was extracted from — material the extraction was supposed to have removed.
Deleting it (rather than yanking it, which only stops pip *selecting* a version and leaves it
downloadable by exact pin) reduces that exposure but does not reverse it: anything already
downloaded stays downloaded, and mirrors may retain copies. Treat every string in 0.1.0 as public.

This release is built from history that has been rewritten to remove those names.

## 0.1.0 — 2026-09-05

The first release, and the first version anybody outside the project can install.

Roadstead is an admission controller for self-hosted LLM inference: it sits in front of llama.cpp
and vLLM servers, speaks the OpenAI API on the way in, and decides — against capacity it measured
rather than assumed — whether to dispatch a request now, spill it to a remote provider, or defer it.
It fair-shares between callers in slot-seconds, computes each call's deadline from a latency
distribution learned from its own traffic, and repairs malformed output on the way back.

It ran for months as one operator's fleet proxy before being extracted into this repository on
2026-08-31. Everything below is what that extraction changed. The **Breaking** entries break against
that origin proxy and against the pre-release tree — there is no earlier release to break.

### Breaking

- **`POST /v1/submit` is removed.** Its replacement is the enriched API at `/rs/v1/*`, which carries
  the intent vocabulary, attribution and timing the old envelope had no room for; `docs/api.md` §1.9
  is a field-by-field migration map. A fleet that cannot move every caller at once can re-open the
  old door behind a flag — see **Added**.
- **Reaching the admin plane now takes a credential, not an address.** Being on a trusted network
  admits a caller; it no longer grants admin, and `GET /v1/stream` — an unbounded live feed of every
  caller's identities, sizes and timings — is admin-gated with the rest. A deployment that relied on
  the network boundary alone must enrol an admin key.
  ([archive](docs/changelog-archive.md#changed---breaking-and-security-relevant-an-address-no-longer-grants-admin))
- **Every `/metrics` series and every log marker is renamed off the origin project's name** —
  `llmproxy_<x>` → `roadstead_<x>` (22 series, same types, labels and HELP text) and
  `LLMPROXY_<X>` → `ROADSTEAD_<X>` (10 markers). Nothing is dual-emitted and there is no alias:
  **update dashboard panels, alert selectors and log-scrape patterns**, because a renamed metric
  empties a graph silently. On-disk names are unchanged.
- **The default data directory moved from `/tmp` to the XDG state directory**
  (`$XDG_STATE_HOME/roadstead`, else `~/.local/state/roadstead`). If you relied on the old default
  you will start with an empty database — move the directory, or set `ROADSTEAD_DATA_DIR` to the old
  path. The shipped image sets `/var/lib/roadstead` and is unaffected. `ROADSTEAD_HOT_ROOT` is gone.
- **The mutating admin plane has a CSRF gate.** A browser attaches a cached `Authorization: Basic`
  credential to a cross-site request on its own — and `Basic` is now one of Roadstead's own
  credential forms, being what the management UI attaches — so every non-GET admin route requires
  `Content-Type: application/json`, refuses `Sec-Fetch-Site: cross-site`, and, for a
  Basic-authenticated write, requires `X-Roadstead-Request: 1`. A script posting form-encoded bodies
  to the admin plane will now get a `415`.
  ([archive](docs/changelog-archive.md#changed--csrf-hardening-on-the-mutating-admin-plane--breaking))

### Added

- **The enriched API at `/rs/v1`**, alongside the strict OpenAI doors. A call gets back the deadline
  Roadstead computed, the priority band it was scheduled in, what it cost, which identity it was
  billed to and which model actually served it — what an operator of a contended fleet needs, and
  what an OpenAI-shaped response has nowhere to put. `POST /rs/v1/plan` answers the same questions
  without dispatching anything.
- **Callers declare intent rather than a model.** `models.yaml` grows an `intents:` section; nine
  profiles (`reasoning`, `fast-chat`, `vision`, …) ship built in and a deployment's own are layered
  over them, with every published profile disclosing whether it is `builtin` or local. Concrete
  model pins are still honoured, and substitution is always disclosed.
  ([archive](docs/changelog-archive.md#added--the-intent-vocabulary-in-config-and-the-negative-constraint-workstream-c))
- **API keys are the identity, the fair-share key, the quota holder and the spend cap.** Keys are
  enrolled, revoked and re-weighted through the admin plane, carry their own priority band, and a
  key may act as callers its operator explicitly granted it (`may_assert`), so one service can bill
  work to the caller it is doing that work for.
  ([archive](docs/changelog-archive.md#added--a-key-may-act-as-callers-its-operator-granted-may_assert))
- **`ROADSTEAD_LEGACY_SUBMIT` re-opens `/v1/submit`**, byte-compatible with what it published before
  removal, so a fleet with a dozen callers on the old envelope can cross one at a time. Off by
  default, warned about at startup, and it does not reverse the removal.
  ([archive](docs/changelog-archive.md#added--the-legacy-v1submit-door-flag-gated))
- **`ROADSTEAD_BEARER_PLACEHOLDERS`** — a list of literal bearer values read as *no credential
  presented*, for an SDK that refuses to construct a client with an empty `api_key`. Exact matching
  only, `Bearer` only, granting exactly what an address grants; a placeholder colliding with a real
  key refuses to start. It is a shim with a removal condition, and it reports its own use.
  ([archive](docs/changelog-archive.md#added--roadstead_bearer_placeholders-for-an-sdk-that-refuses-an-empty-key))
- **A response now says when it was rewritten.** `X-Roadstead-Corrected` on the OpenAI doors and
  `corrections` in the enriched envelope name which repair applied — a schema repaired, a retry, a
  stripped `response_format`, an unrecovered degeneration, a truncated tool call. This was
  previously visible only as a fleet-wide counter, so a caller could not tell.
  ([archive](docs/changelog-archive.md#added--correction-layer-rewrites-are-now-disclosed-per-call))
- **Size caps in both directions.** `ROADSTEAD_MAX_REQUEST_BYTES` (16 MiB) refuses an oversized body
  at the ASGI layer before anything parses it; `ROADSTEAD_MAX_RESPONSE_BYTES` (64 MiB) is checked
  line by line as a stream arrives, so a backend that never stops talking is aborted rather than
  buffered without bound.
  ([archive](docs/changelog-archive.md#added--a-request-body-cap-and-a-streamed-response-cap))
- **A management plane, and a browser UI for it** (`ROADSTEAD_ADMIN_UI`): the catalog is writable at
  runtime, a declared endpoint can be brought into service without a restart, and callers, keys,
  spend and live capacity read as an instrument rather than a wall of JSON.
  ([archive](docs/changelog-archive.md#added--the-management-plane-workstream-e))

### Changed

- **The client SDK's `ChatResult` is now `CallResult`** — it is returned by embeddings and rerank as
  well as chat, it can express all three payload types, and rerank has a route again. The enriched
  `price` block now has one shape on every route instead of three a client had to branch on.
- **Small surface:** `roadstead/py.typed` ships, so an installing project's type checker reads the
  annotations (PEP 561); `--data-dir` overrides `ROADSTEAD_DATA_DIR` explicitly;
  `roadstead --version` prints the installed distribution's version; and `ROADSTEAD_AGENT_NAME` is
  gone, having been exported with `setdefault` and read by nothing.

### Fixed

- **The request log is bounded.** It appended forever at 10–20 MB/day and could fail a live request
  when its write failed. Rotation is in the application (64 MB x 7), not left to an external
  logrotate that would have renamed a file this process holds open in append mode.
  ([archive](docs/changelog-archive.md#fixed--the-request-log-grew-without-bound-and-could-fail-a-live-request))
- **Durable state survives a restart, and a container rebuild.** The event log defaulted under
  `/tmp` and vanished on every rebuild; the day's spend was rebuilt from nothing and read as zero.
  ([archive](docs/changelog-archive.md#fixed--the-durable-event-log-defaulted-to-tmp-and-in-a-container-it-was-lost-on-every-rebuild))
- **`ROADSTEAD_QUEUE_DB` and thirteen sibling variables were read and then silently ignored** — a
  deployment that set them ran on the defaults with nothing to say so.
  ([archive](docs/changelog-archive.md#fixed--roadstead_queue_db-and-thirteen-siblings-were-silently-ignored))
- **SECURITY: a credential pasted into an endpoint's `api_key_env` field was stored, persisted to
  the runtime overlay and echoed back** by the management plane, and the overlay file itself was
  world-readable. Both are closed.
  ([archive](docs/changelog-archive.md#fixed---security-a-credential-pasted-into-api_key_env-was-stored-persisted-and-echoed-back))
- **Two analytics windows were handed to the wrong argument of the shared clamp.**
  `/v1/fleet/cache-stats` returned a week where its documented 30 days were asked for, and an
  unparseable `?window=` on `/v1/fleet/cache-attribution` ran a week-wide aggregate instead of an
  hour. Each response echoes the window it used, which cannot show the difference.
- **Six smaller ones:** the shutdown drain has a ceiling and publishes it once (108s, with the
  arithmetic); an on-demand endpoint's in-flight lease can idle-release; `/rs/v1/plan` predicts the
  band `/rs/v1/chat` will actually dispatch at; a cancelled straggler's caller gets an envelope
  rather than a raw 500; the blocking client no longer leaks connections after `close()`; and
  `roadstead test …` is reachable from the console script, not only from `python -m roadstead`.

### Documented

- **`docs/api.md` is the contract, and it is executable** — tests read it back in both directions.
  This release added the 21 surfaces the code shipped without documenting and §3.12 for seven
  analytics routes down to column level, and confirmed the wire contract against a real vLLM for the
  first time rather than against the fake backend that ships with the suite.
  ([archive](docs/changelog-archive.md#documented--21-things-the-code-shipped-and-docsapimd-did-not))
- **Seventeen error codes are published, not fifteen.** `schema_invalid` and `toolcall_truncated`
  reached the wire from the correction layer and appeared in no list, so a client library treated
  `toolcall_truncated` as non-deferrable and discarded truncated tool calls that a retry with a
  larger output budget would have completed. A new guard walks the source for every code that
  reaches the wire and fails when one has no row.
- **`docs/configuration.md`** — all 52 environment variables with type, default, clamp and meaning,
  plus the CLI flags, the boolean families and the retired spellings. Pinned in both directions by a
  test that recovers names from the environment *read* through an AST walk rather than grepping for
  a prefix.
- **The front door is written for a stranger:** what Roadstead is in the first two sentences, a
  "Run it" block whose every command was executed against this tree, `docs/compatibility.md` for
  what is stable and what is not, a Code of Conduct, an issue form that asks for the catalog stanza
  an admission bug cannot be reproduced without, and the scrub rule inlined into `CONTRIBUTING.md`.
