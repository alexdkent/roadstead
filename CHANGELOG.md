# Changelog

Notable changes to Roadstead. Breaking changes get their own section with a reason, per
`docs/compatibility.md` — the stable surface is the wire contract in `docs/api.md`; everything else
is internal and changes without an entry.

Pre-1.0: breaks are permitted, but each one is a recorded decision rather than a surprise.

Everything before 0.1.0 is in **`docs/changelog-archive.md`** — the sixty-five entries written while
the project was being extracted from a private monorepo. Those are the reasoning; this file is the
summary. The bullets below link in there where the long version is worth reading.

## Unreleased

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

### Fixed

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
