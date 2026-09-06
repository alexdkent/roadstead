# Roadstead

> A **roadstead** is the sheltered anchorage outside a harbor where vessels wait for a berth to
> free up.

**An LLM proxy and scheduler for backends you run yourself.** llama.cpp and vLLM on your own
hardware, with OpenRouter and other remote providers as explicit overflow. Callers reach it through
an OpenAI-compatible door. It decides what runs where and when, using priority bands, fair-share
between callers, and timeouts computed from latency it measured rather than a number the caller
guessed.

Two things make that hard, and they pull in opposite directions. Callers differ in *urgency*, not
just in what they ask for: an interactive turn and an overnight summarizer may want the same model
and cannot wait the same amount of time. Backends differ in capacity, capability, and failure mode:
four fixed slots or elastic, thinking or not, honors a grammar or silently drops it. Roadstead sits
in between so neither side has to know about the other.

**Local capacity is the default; cloud is explicit overflow**, never the fallback that quietly
becomes the norm. It runs with no account and no internet, and nothing leaves the machine unless
someone opts into spilling it.

> ⚠️ **Pre-1.0.** Extracted from a production monorepo on 2026-08-31 and still stabilizing. It is an
> independent project rather than a replacement for its origin: expect it to become a superset, and
> expect occasional deliberate breaks (`docs/compatibility.md`).

## Why it exists

Cloud gateways assume elastic capacity, so admission control is uninteresting to them. Local tools
assume one user, so fairness is uninteresting to them. Roadstead is the case neither serves:
capacity that is finite *and* several callers competing for it.

Most gateways treat a backend as an opaque endpoint with a health bit. They load-balance across it,
retry on failure, and take a timeout as a number the caller supplied. That is the right model for an
elastic cloud API.

It is the wrong model for a llama.cpp server with exactly four slots and a fixed per-slot context,
sitting on a GPU you own. The question there is not *which backend*. It is whether to send at all
right now, whose turn it is, how long is reasonable to wait, what it costs, and what to do when the
answer comes back malformed.

Roadstead answers by measuring rather than assuming:

- **It discovers capacity.** llama.cpp `/props` reports real slot counts and per-slot context size,
  and admission is gated against what is actually there. (vLLM cannot do this; see
  [Backends](#backends--what-roadstead-expects).)
- **It fair-shares in slot-seconds.** Deficit round-robin across callers within three strict
  priority bands, where the unit of fairness is backend occupancy time, EWMA-calibrated per
  endpoint, charged up front and corrected retroactively. Request counts and token counts both
  mis-price a backend with fixed slots. Occupancy time does not.
- **It computes deadlines instead of accepting them.** Callers declare priority and interactivity;
  the timeout comes from a latency distribution learned from real traffic, conditioned on endpoint,
  tier, and request size.
- **It checks what comes back.** Truncation, malformed JSON, dropped grammars, degenerate
  repetition, and broken SSE framing are caught at the proxy instead of at every caller. That half
  is large enough to have [its own section](#responses--what-it-checks-and-repairs).

A survey of ~25 open-source gateways (`docs/evaluation.md`) found none that does all of this outside
Kubernetes, and none at all that does the last two.

> **Not an "LLM orchestrator."** It does not chain calls or run multi-step workflows. No agents, no
> graphs. It is a scheduler in the operating-system sense: it decides what runs where, when, and for
> whom, under contention. It sits underneath an agent framework, not beside one.

## Status

Extraction is complete. The package is standalone, the suite runs against a shipped fake backend,
and the contract is published and executable.

**Where it is going** (`docs/roadmap.md`):

| | | |
|---|---|---|
| **North face** | OpenAI-compatible, strictly, plus the enriched Roadstead API at `/rs/v1` carrying live model information, computed deadlines, priority and attribution | ✅ |
| **Model abstraction** | Callers declare intent (`reasoning`, `fast-chat`, `vision`); Roadstead owns the choice. Concrete pins honored, substitution opt-in and always disclosed | ✅ |
| **South face** | Modular providers: llama.cpp and vLLM local, OpenRouter and others remote | ✅ |
| **Capacity** | One admission decision, three outcomes: dispatch locally, **spill** to a remote provider, or defer | ✅ |
| **Identity** | API keys as the fair-share, quota and budget key | ✅ |
| **Cost** | Token and spend accounting, with thresholds that **degrade rather than reject** | ✅ |
| **Hardening** | The concurrency invariant armed and asserted under sustained load | ✅ |
| **Operations** | A management interface for running it standalone | planned |

Parity against the origin copy was removed from the plan on 2026-08-31: a parity gate on a
deliberate superset fails on every improvement.

## Quick start

```sh
pip install roadstead
```

Python 3.11 or newer. That is the 0.1.1 release on PyPI. The source is on GitHub, and the
unreleased tip installs from it directly:

```sh
pip install git+https://github.com/alexdkent/roadstead
```

To work on it instead, take the source and its dev extras:

```sh
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

The suite needs **no fleet, no network and no inference backend**. It runs against
`roadstead.testing`, described below.

### Run it

The install puts a `roadstead` console script on your path. `python -m roadstead` is the same entry
point for anyone who would rather not rely on one.

```sh
ROADSTEAD_DATA_DIR=/var/lib/roadstead roadstead --port 42161
ROADSTEAD_DATA_DIR=/var/lib/roadstead python -m roadstead --port 42161
```

🚨 **Set `ROADSTEAD_DATA_DIR` or you will lose state you were told was durable.** It defaults to
`/tmp/agents/llmproxy`, which is right for a developer running the module for ten minutes and wrong
for anything else. DRR balances, the day's spend, and endpoint drain state live under it, and those
are exactly the rows the shutdown drain exists to flush. The process says so on startup when the
path looks ephemeral.

With no configuration at all it boots against the example catalog that ships in the package
(`roadstead/models.yaml`, invented backends on RFC 5737 addresses), mints a one-off bootstrap admin
key, and answers:

```sh
curl localhost:42161/readyz     # {"ready":true, ..., "scheduler_alive":true, "reason":"ok"}
curl localhost:42161/v1/models  # tier1, tier2, tier3, embed, rerank — from the catalog
```

Point `ROADSTEAD_MODELS_YAML` at your own catalog to route to backends that exist.

**In a container, the stop-grace period is load-bearing:**

```sh
docker run --stop-timeout 108 -p 42161:42161 -v roadstead-data:/var/lib/roadstead roadstead
```

SIGTERM starts a bounded drain that persists DRR budgets and completion rows. Uvicorn's connection
budget and the app's drain budget run in series rather than nested (48s, then 48s, plus a margin),
so 108 is a ceiling rather than a measurement. `docker stop` hard-kills after **10s** by default,
which truncates that flush in every non-idle case and loses the state silently. The default works
fine while the proxy is quiet, which is why it first fails under load. Compose spells the same thing
`stop_grace_period: 108s`. The number is computed in one place (`service.RECOMMENDED_STOP_GRACE_S`),
the image carries it as a label, and the process prints it on startup.

### Two doors

**OpenAI-compatible**, for anything that already speaks it:

```sh
curl localhost:42161/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"model": "tier2", "messages": [{"role": "user", "content": "hi"}]}'
```

**The enriched API**, for callers that want what OpenAI's shape cannot carry. Declare what you need
rather than which box, and be told what actually served you:

```python
from roadstead.client import AsyncRoadsteadClient

async with AsyncRoadsteadClient("http://localhost:42161", api_key=KEY) as rs:
    plan = await rs.plan(intent="reasoning", est_in=8_000)
    print(plan.endpoint, plan.recommended_deadline_s, plan.estimated_usd)

    result = await rs.chat(intent="reasoning",
                           messages=[{"role": "user", "content": "..."}])
    print(result.content)
    print(result.attribution.endpoint,      # what actually served
          result.attribution.substituted,   # and whether that differed
          result.timing.queue_wait_ms,      # how long it waited for a slot
          result.usage.slot_seconds)        # what it cost in the unit of fairness
```

The SDK ships in the package and imports nothing from the server: httpx and the stdlib only.

## Backends — what Roadstead expects

Three providers exist today (`roadstead/providers/`): `llama.cpp`, `vllm`, and `openrouter`. The
code branches on a provider's `ProviderDescriptor`, not on an engine name, so what each engine
publishes is declared rather than inferred. `docs/api.md` §4 is the full contract; this is what you
need to configure one correctly.

### Routes every local backend must serve

| Purpose | Method and path |
|---|---|
| Chat | `POST /v1/chat/completions` |
| Embeddings | `POST /embed` — 🚨 **not** `/v1/embeddings` |
| Rerank | `POST /rerank` |

Embed and rerank are FastAPI-shaped shims, not OpenAI-compatible, and have no OpenAI route.

### llama.cpp

**Concurrency is discovered, so you do not declare it.** `GET /props` reports
`n_parallel` / `total_slots` / `len(slots)` and a per-slot `n_ctx`, and admission is gated against
those numbers.

Things worth knowing before you tune it:

- **Truncation is labeled correctly.** `finish_reason=length` means what it says, which is why the
  tool-call rule below is vLLM-only.
- **`--cache-reuse` is architecturally refused on two classes of backend**, multimodal
  (`mctx != nullptr`) and non-shiftable attention memory, with two different loader messages. Adding
  the flag does nothing. The real lever on prefill reuse is lowering `-ub`, and that is a trade, not
  a free win: measured **−27% prefill**.
- Metrics prefixes read for prefix-cache visibility: `llamacpp:prompt_tokens_total`,
  `llamacpp:n_decode_total`.

### vLLM

🚨 **vLLM does not expose a slot count, so concurrency stays whatever you configured.** `GET
/v1/models` yields `max_model_len` and nothing about occupancy. `--max-num-seqs` is not readable
over the API. This is a property of the engine, not a gap in Roadstead, and it means **a wrong
number in your catalog stays wrong** rather than being corrected by discovery.

Declare it twice, on purpose:

```yaml
tier3:
  max_slots: 20                  # what Roadstead admits against
  documented_max_num_seqs: 20    # what the server was actually launched with
```

`max_slots` is the number admission uses. `documented_max_num_seqs` mirrors the real launch flag
from your serve script, and a shadow reconciler warns when the two disagree. Setting only the first
is how a backend ends up admitting 32 against a real cap of 20.

Three launch flags change what the proxy is allowed to send. Get them wrong and the failure is
quiet, or total:

- **`thinking_token_budget` 400s the entire request** unless the server was started with
  `--reasoning-config`. `--reasoning-parser` alone is **not** enough. Treat the budget as a
  declaration that a launch flag is present, not as a tuning knob.
- **`--structured-outputs-config '{"disable_any_whitespace": true}'` is load-bearing**: without it,
  structured output can run away emitting whitespace until `max_tokens`. With it, a bare
  `response_format: {"type": "json_object"}` becomes actively harmful, because `{}` is a legal,
  complete, zero-whitespace document and greedy decoding takes it. The proxy cannot introspect a
  launch flag, so mirror it on the endpoint as `disable_any_whitespace` and Roadstead will drop the
  bare form. Measured on one endpoint, same prompt, three shapes:

  | request | result |
  |---|---|
  | no `response_format` | 549 chars, valid JSON |
  | bare `json_object` | `{}` — 2 chars, `finish_reason=stop` |
  | strict `json_schema` | 241 chars, valid JSON |

- **The thinking switch is spelled differently per model family** (`thinking` vs `enable_thinking`).
  Declare it in the catalog; it is never hardcoded. The wrong spelling makes thinking a silent
  no-op.

Metrics prefixes: `vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total`,
`vllm:prompt_tokens_total`, `vllm:generation_tokens_total`.

### Remote providers

A remote endpoint declares a `base_url` (superseding `host`/`port`, which cannot express a scheme or
base path) and its provider supplies its own auth headers. The key is read at request time from the
environment variable named by `api_key_env`; Roadstead never holds a key in config.

Remote capacity is not local capacity. There is no occupancy to publish, so concurrency is a
deliberate cap on your own traffic rather than a discovered fact. Slot-seconds remain the unit of
fairness because local slots are the scarce thing.

Published prices are re-read on every discovery pass. Three things about them are easy to get wrong:
the values are **strings** (a price like `0.0000005` is where a JSON float starts losing digits),
they are **per single token**, and **zero is a real price**, not a missing one.

### Optional signals, and what degrades without them

| Signal | Absent → |
|---|---|
| `usage.prompt_tokens_details.cached_tokens` | per-request cache attribution falls back to an endpoint-level rate |
| `GET /metrics` | no prefix-cache visibility, reported as *n/a*, never as 0% |
| decode-progress counters | streaming deadlines cannot be extended on progress; the soft deadline becomes hard |

### Anything else

Any backend that does not serve `/props` or `/v1/models` — a FastAPI shim, an embedding or rerank
service, anything OpenAI-shaped that is neither of the two engines above — **must** be declared
`skip_discovery`, or discovery generates continuous 404 noise against it. Its concurrency and
context then come entirely from your catalog, with no drift detection at all.

## Responses — what it checks and repairs

A gateway that only forwards bytes pushes every backend pathology onto every caller. Most of what
follows was written after one of these reached production and cost something.

The governing rule: **a repair is only allowed where the backend asserted the response was
complete.** Everywhere else, a bad response fails loudly. A repair applied to a genuinely truncated
answer is not a fix, it is a silencer.

### Truncation

- **An output-cap hit never passes silently to any caller.** Every completion that ends
  `finish_reason=length` logs a stable `ROADSTEAD_TRUNCATION` marker and a per-(model, caller)
  tally, from the single completion choke point, for both streaming and non-streaming.
- **vLLM mislabels a `max_tokens` truncation mid-tool-call as `finish_reason="tool_calls"`.**
  llama.cpp gets this right. So on a vLLM non-streaming 200, a tool call whose `function.arguments`
  does not parse is treated as truncation and never as a repair target. This ordering is
  deliberate and load-bearing: if JSON repair ran first it would close the cut-off argument string
  into valid-but-fabricated JSON, and the caller would execute a silently wrong command. Instead it
  fails with a deferrable error shape so the caller retries.
- **A missing terminal SSE `finish_reason` chunk is synthesized only when the backend sent its own
  `[DONE]`** — that is, only when it asserted completeness and merely failed to label it. A stream
  that ends with neither is a real truncation and nothing is synthesized. 🚨 Collapsing those two
  cases turns the repair into a silencer. `finish_reason` conventionally rides alone on a final
  chunk whose `delta` is `{}`, so it carries no text and losing it is invisible if you only inspect
  content. One downstream client burned a second model call on 47 turns "continuing" answers that
  were already complete.

### JSON and structured output

- **A JSON-implying request never returns `ok` with content that fails `json.loads`.** This floor
  runs last, after every other repair layer has had its chance, so it fires only on what nothing
  recovered. It is parse-only; schema conformance belongs to the backstop, and grammar enforcement
  belongs to the backend. Kill switch: `ROADSTEAD_PROXY_STRUCTURED_VALIDITY`.
- **A JSON-schema retry backstop** and JSON repair for text that is nearly valid, both upstream of
  that floor.
- **The bare `json_object` guard**, described under vLLM above: on an endpoint declared
  `disable_any_whitespace`, a bare `json_object` request is rewritten rather than allowed to return
  `{}`.
- **Answerless-but-well-formed responses are detected and reported, never fixed.** `{}` is valid
  JSON with `finish_reason=stop` and no error, so it passes the degeneration check, the truncation
  check, the empty-completion check, and the validity floor above. It is caught as telemetry only:
  it never raises, never mutates the response, and deliberately never retries, because a retry would
  convert a silent failure into an expensive silent failure. This one went undetected for 31 hours
  and produced 3,100 pointless retries downstream before a human noticed.
- **Degenerate repetition is detected** by shingle analysis. Re-dispatch is deliberately skipped
  when the original hit its output cap, since a capped loop just re-caps.
- **Malformed streaming tool calls are sanitized in the stream.** The vLLM `qwen3_xml`
  parallel-tool-call defects are streaming-delta artifacts only: 5,460 non-streaming responses over
  7 days produced zero. The stream sanitizer is the complete fix, and there is deliberately no
  response-path equivalent.

### Grammars

GBNF grammars are validated and repaired before dispatch. **A provider that cannot honor a caller
constraint refuses the request** with a `400` naming the provider, rather than dropping it: a
grammar sent to a remote provider that has no such concept is a refusal, not a silent downgrade. The
caller cannot tell unconstrained output from a model that answered badly, so the constraint has to
fail loudly.

Engine *hints* the proxy itself added (`id_slot`, `chat_template_kwargs`, `thinking_token_budget`)
are dropped silently instead, because no caller asked for them.

## Who is calling — API keys

A caller's identity is the **DRR fair-share key**: the string fairness, quotas, and budgets are all
accounted against. An API key establishes it and carries that caller's policy with it (default
priority, an optional deadline floor, an optional admin scope), so the policy travels with the
caller rather than with the machine it happens to run on.

```sh
# one key, for the container case
ROADSTEAD_API_KEYS='sk-local-abc=coding-assistant:P1_TURN_SUPPORT'

# or a file, which can carry digests instead of secrets
ROADSTEAD_API_KEYS_FILE=/etc/roadstead/keys.yaml
```

```yaml
# keys.yaml — no example ships in the package, deliberately: a default key file
# is a default credential.
keys:
  - id: coding-assistant-laptop     # a public label, safe to log. Never the key.
    agent_id: coding-assistant      # the DRR fair-share key
    key_sha256: "8f4e…"             # `printf %s "$KEY" | shasum -a 256`
    priority: P1_TURN_SUPPORT       # the band when the caller declares none
    min_timeout_s: 600              # deadline floor, for a caller that sets none
  - id: ops
    agent_id: ops
    key: plaintext-is-allowed-too   # hashed at load; but a key in a file is a
    admin: true                     # key in a git history — prefer key_sha256
```

An OpenAI client needs nothing but its `api_key` set: the key rides the `Authorization: Bearer`
header it already sends.

🚨 **An SDK with no key still sends one.** Most OpenAI SDKs refuse to construct a client with an
empty `api_key`, so a fleet authorized by address ends up sending a literal that means nothing:
`not-needed`, `EMPTY`, `sk-no-key-required`. That is harmless until the first real key is
configured, at which point every one of those callers is refused at once. Declare the literals and
they are read as no credential at all:

```sh
ROADSTEAD_BEARER_PLACEHOLDERS='not-needed,EMPTY'   # empty by default, which is off
```

Matching is exact and case-sensitive, `Bearer` only. It grants exactly what the address grants and
no admin, and a value that is also a registered key's plaintext refuses to start rather than
silently demote that key. It is a migration shim: `GET /v1/status` →
`reliability.placeholder_bearers` names who still needs it. `docs/api.md` §1.5 has the rest.

**Out of the box there is no key and no configuration.** Loopback and docker-internal callers are
admitted as `internal`, everything else is refused. That is default-deny with the local-first case
free. To admit a host without issuing it a key, enroll its address:

```sh
ROADSTEAD_ACL='192.0.2.0/24=lan:P3_INGESTION,192.0.2.9=ingest:P4_HYGIENE:1800'
```

An address is deliberately the *weaker* factor, because it identifies a host and not a caller. Where
the two disagree the key wins, and a presented key that does not resolve is a 401 rather than a
quiet demotion to whatever the address would have given. `docs/api.md` §1.5 has the full precedence.

### Behind a reverse proxy

🚨 **If anything sits in front of Roadstead — a TLS terminator, an ingress, a sidecar — say so.**
Otherwise every caller arrives at the proxy's address, the whole address layer collapses into one
identity, and if that address is loopback or docker-internal (a sidecar usually is) it carries the
built-in admin grant with it.

```sh
ROADSTEAD_TRUSTED_PROXIES='127.0.0.1,172.18.0.0/16'   # empty by default
```

`X-Forwarded-For` is then read **only** on connections from those addresses, and the caller is taken
as the rightmost hop that is not itself a trusted proxy, never the leftmost, which is whatever the
caller wrote before any proxy appended what it saw.

Configuring this also **withdraws the built-in loopback/docker admin grant from forwarded
requests**, deliberately: "it arrived on loopback" stops meaning "somebody is already on the
machine" the moment a front door exists. Administer with an `admin` API key, which works from
anywhere and can be revoked, or name the address in `ROADSTEAD_ADMIN_NETS`. `GET
/rs/v1/admin/config` shows what is trusted and which admin nets are built in versus yours.

Two more consequences of leaving `ROADSTEAD_TRUSTED_PROXIES` empty behind a real proxy, both
worth naming because they compound rather than just collapsing identity:

- **The collapsed address doesn't just merge every caller — it lands them in the top priority
  band.** An internal/docker-local address resolves to `P1_TURN_SUPPORT`, the interactive lane
  (`roadstead/acl.py`). So every internet caller queues ahead of the operator's own background
  work, not merely alongside it.
- **The forwarded-address check that would normally stop this fails open in exactly this
  misconfiguration.** §1.9.2 of `docs/api.md` refuses the legacy self-declare exception to an
  address that arrived via `X-Forwarded-For` — but that check depends on Roadstead recognizing the
  request as forwarded at all, and it only does that for a peer in `ROADSTEAD_TRUSTED_PROXIES`
  (`roadstead/identity.py`). With the setting empty, every hop reads as unforwarded, so
  `may_self_declare` (`roadstead/legacy.py`) sees "internal, not forwarded" and grants the
  exception anyway. With `ROADSTEAD_LEGACY_SUBMIT` also enabled, any internet caller can then
  declare any `agent_id` in the request body and inherit that agent's DRR weight and priority.

Neither of these is a default-open hole — an unfronted Roadstead, or one with the setting
configured, doesn't have them. They're what "unset behind a real proxy" actually costs, on top of
the identity collapse above.

## Managing it — `/rs/v1/admin`

Keys, quotas, and budgets are runtime operations, so enrolling a caller does not mean editing a file
and restarting a proxy that is serving traffic.

```sh
# enroll a caller. The secret comes back ONCE and is never stored — only its digest is.
curl -sX POST localhost:42161/rs/v1/admin/keys \
     -d '{"agent_id": "coding-assistant", "priority": "P1_TURN_SUPPORT"}'

# adjust its share of the fleet, and cap what it may spend off-machine
curl -sX PATCH localhost:42161/rs/v1/admin/callers/coding-assistant \
     -d '{"weight": 3.0, "spill_ok": true, "daily_spend_usd": 5.0}'

# revoke, immediately, whatever declared it
curl -sX DELETE localhost:42161/rs/v1/admin/keys/coding-assistant-laptop
```

### The UI

```sh
ROADSTEAD_ADMIN_UI=1        # off by default: unset, the route does not exist
```

Then open `http://<host>/rs/v1/admin/ui`. It is one static HTML file with no bundler, no framework,
and no external references of any kind, the same judgment that keeps `roadstead.client` on
httpx-and-stdlib. It shows the thing the HTTP plane exists to show: wherever a value you declared
and the value actually in force can disagree, both appear side by side and a difference is made
loud.

Sign in with an **admin API key as the password** (the username is ignored). The browser is
challenged with HTTP Basic, so there is no login form, no session, and no cookie, which means CSRF
never becomes reachable on the mutating routes. Put it behind TLS, and see the trusted-proxy note
above, because a UI is the usual reason a reverse proxy appears in front of this.

**A runtime change never rewrites your config file.** It goes to a JSON overlay layered over
`models.yaml`, `agents.yaml`, and your keys file at startup, so your comments survive and what you
wrote stays separable from what the API changed.

The read side answers the question a config file cannot: **what did I write that is not in force?**

```sh
curl -s localhost:42161/rs/v1/admin/config     # sources, and every knob nothing reads
curl -s localhost:42161/rs/v1/admin/providers  # declared capacity vs what discovery found
curl -s localhost:42161/rs/v1/admin/callers    # quota in force, declared, and overridden
```

A dropped `policy:` key looks exactly like a knob that was never load-bearing, and a slot count that
discovery overwrote looks exactly like one it never probed. Those gaps are where the expensive
mistakes live, so the views report both numbers rather than the winning one. `docs/api.md` §3.

## Overflow, and what it costs

Local capacity is the design center; remote capacity is what happens when the local fleet is
**full**. For each request there is one decision with three outcomes: **dispatch** to a local slot,
**spill** to a remote provider, or **defer** and stay queued.

Local is tried first for every caller, unconditionally. Spill is considered only once local has said
no, so the fleet is never bypassed while it has room, and a deployment with no remote provider never
executes a line of it.

```yaml
# models.yaml — an endpoint says where its overflow goes
tier3:
  failover_to: tier2          # it is DOWN     -> a smaller local model may answer
  spill_to: spill-reasoning   # it is FULL     -> pay somebody else to answer now
```

```yaml
# agents.yaml — and a caller says whether it wants any of that
chat-assistant:
  degrade_ok: true            # may a WORSE model answer this?
  spill_ok: true              # may this leave the machine, at our expense?
  daily_spend_usd: 5.0
```

Those are two different questions and neither implies the other. A caller whose work degrades
happily may still be one whose prompts must never go to a third party.

**Two kinds of money, never added together.** A local call is priced at what renting the same class
of model *would* have cost. That is a saving, and it is reported separately from what you actually
spend. Only real spend counts against a cap.

🚨 **A cap degrades; it never rejects.** Crossing `daily_spend_usd` costs a caller one priority band
and its access to paid spill, and nothing else. It keeps full access to local capacity, and there is
no error code for it. Admission control here is about capacity, not billing, and a quota you typo'd
must not be able to take a caller offline.

## A programmable backend, in the box

`roadstead.testing` ships the fake backend the suite runs on. For a gateway whose thesis is
capacity-aware admission control, a backend that lies about its capacity on demand is a capability
rather than test scaffolding, and it is not something you can ask a real GPU for.

```python
from roadstead.testing import FakeBackend, FakeBackendServer, FAULT_CAPACITY_DESYNC

server = FakeBackendServer(FakeBackend(engine="vllm")).start()
server.controller.set_fault(FAULT_CAPACITY_DESYNC, 2)   # accepts 2, 503s the rest,
                                                        # while /props claims otherwise
```

It is a real Starlette app under real uvicorn on a real socket, so real `httpx` and real SSE framing
are exercised rather than a mock transport. It speaks both llama.cpp and vLLM wire shapes across
`/v1/chat/completions`, `/embed`, `/rerank`, `/props`, `/v1/models`, `/metrics`, and `/health`, and
serves twenty-odd south-face pathologies on command: truncated and invalid JSON, empty completions,
degenerate repetition, schema violations, phantom and truncated tool calls, partial and interleaved
SSE frames, TTFT and inter-token stalls, mid-stream resets, and capacity desync.

It is also the executable form of §4 of `docs/api.md`, the contract above.

## Known limitations

Three things worth knowing before you deploy it, each written up where it is being worked on:

- **Intent routing never explores, so a fleet converges on one endpoint.** Found by running it:
  2,566 calls across three intents all went to `tier3`, and the rivals ended with no latency samples
  at all. An unmeasured endpoint ranks `inf`, so the first winner is the only one that ever
  accumulates the evidence that could unseat it. A concrete pin still routes exactly where you said.
  [`docs/roadmap.md`](docs/roadmap.md)
- **Pre-1.0: breaking changes are allowed.** They are deliberate and recorded. The wire contract in
  `docs/api.md` is the stable surface, everything else may move, and none of it promises a
  deprecation *period*. Read the policy before you pin a version.
  [`docs/compatibility.md`](docs/compatibility.md)
- **The legacy `/v1/submit` door is a compatibility shim, off by default.** It exists so a fleet
  already speaking the old envelope can cross one caller at a time. `ROADSTEAD_LEGACY_SUBMIT` opens
  it; unset means the route does not exist. It also restores an unchecked body `agent_id` for
  internal-net callers, which is the one place Roadstead departs from its own identity rules. Its
  removal is gated on the inventory being empty, not on a date. [`docs/api.md`](docs/api.md) §1.9.1

## Documentation

| | |
|---|---|
| `docs/internals.md` | Orientation, the concurrency invariant, and the engine-behavior findings that explain why the code is shaped the way it is. **Read before changing anything.** |
| `docs/api.md` | The API surfaces: both north faces, the error contract, the admin/control plane, what Roadstead requires *of a backend*, and the client SDK. |
| `docs/configuration.md` | Every environment variable and catalog key, in one place. |
| `docs/roadmap.md` | **What is being built and why.** Start here for direction. |
| `docs/compatibility.md` | What is stable, what is not, and how to break something on purpose. |
| `docs/history.md` | Closed record of the extraction: where the code came from and what that cost. |
| `docs/evaluation.md` | Why this exists rather than adopting something else: the field survey and decision record. |
| `docs/ledger.md` | Defects that came back, with the guard that now prevents each. |

## License

Apache-2.0.
