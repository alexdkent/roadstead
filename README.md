# Roadstead

> A **roadstead** is the sheltered anchorage outside a harbour where vessels wait for a berth to
> free up.

**A local-first LLM scheduler.** It stands between many kinds of caller and many kinds of model, and
absorbs the mismatch so that neither side has to model the other.

Callers differ in *urgency*, not just in what they ask for — an interactive turn and an overnight
summarizer may want the same model and cannot wait the same amount of time. Backends differ in
capacity, capability and failure mode — four fixed slots or elastic, thinking or not, honours a
grammar or silently drops it. **Local capacity is the default and the design center; cloud is
explicit overflow**, never the fallback that quietly becomes the norm.

Local-first also means what you would hope: it runs with no account and no internet, and nothing
leaves the machine unless someone opts into spilling it.

> ⚠️ **Pre-release, and private.** Extracted from a production monorepo on 2026-08-31 and still
> stabilising. It is an independent project rather than a replacement for its origin — expect it to
> become a superset, and expect occasional deliberate breaks (`docs/compatibility.md`). This
> repository must not be made public until the scrub in `docs/corpus_and_scrub_plan.md` is complete.

## Why it exists

**Cloud gateways assume elastic capacity, so admission control is uninteresting to them. Local tools
assume one user, so fairness is uninteresting to them.** Roadstead is the case neither serves:
capacity that is finite *and* many callers competing for it.

Most LLM gateways treat a backend as an opaque endpoint with a health bit. They load-balance across
it, retry on failure, and take a timeout as a number the caller supplies. That works when the
backend is an elastic cloud API.

It does not work when the backend is a llama.cpp server with exactly four slots and a fixed
per-slot context, sitting on a GPU you own. There, the interesting question stops being *which
backend* and becomes: **whether to send at all right now, whose turn it is, how long is reasonable
to wait, what it costs — and what to do when the answer comes back malformed.**

Roadstead answers that question by measuring rather than assuming:

- **It discovers capacity.** It probes llama.cpp `/props` for real slot counts and per-slot context
  size, and gates admission against what is actually there.
- **It fair-shares in slot-seconds.** Deficit round-robin across callers within three strict
  priority bands, where the unit of fairness is *backend occupancy time* — EWMA-calibrated per
  endpoint, charged up front and retroactively corrected. Request counts and token counts both
  mis-price a backend with fixed slots; occupancy time does not.
- **It computes deadlines instead of accepting them.** Callers declare priority and interactivity;
  Roadstead derives the timeout from a latency distribution it learned from its own traffic,
  conditioned on endpoint, tier and request size.
- **It repairs what comes back.** Grammar validation and repair, a JSON-schema retry backstop,
  degeneration detection, truncation integrity, and repair of malformed SSE — including the missing
  terminal `finish_reason` chunk that no client notices until it burns a second model call.

A survey of ~25 open-source gateways (`docs/evaluation.md`) found none that does all of this
outside Kubernetes, and none at all that does the last two.

That survey was of a proxy for one private fleet. Roadstead generalises it: the same question, asked
across local *and* remote capacity, for callers who declare what they need rather than which model
to use. See `docs/roadmap.md`.

> **Not an "LLM orchestrator."** It does not chain calls or run multi-step workflows — no agents, no
> graphs. It is a *scheduler* in the operating-system sense: it decides what runs where, when, and
> for whom, under contention. It sits underneath an agent framework, not beside one.

## Status

Extraction is complete: the package is standalone, the suite runs against a shipped fake backend,
and the contract is published and executable. Roadstead is now its own project rather than a
standalone copy of the proxy it came from.

**Where it is going** (`docs/roadmap.md`):

| | | |
|---|---|---|
| **North face** | OpenAI-compatible, strictly — plus the enriched Roadstead API at `/rs/v1` carrying live model information, computed deadlines, priority and attribution | ✅ |
| **Model abstraction** | Callers declare intent (`reasoning`, `fast-chat`, `vision`); Roadstead owns the choice. Concrete pins honoured, substitution opt-in and always disclosed | ✅ |
| **South face** | Modular providers: llama.cpp and vLLM local, OpenRouter and others remote | ✅ |
| **Capacity** | One admission decision, three outcomes — dispatch locally, **spill** to a remote provider, or defer | ✅ |
| **Identity** | API keys as the fair-share, quota and budget key | ✅ |
| **Cost** | Token and spend accounting, with thresholds that **degrade rather than reject** | ✅ |
| **Hardening** | The concurrency invariant armed and asserted under sustained load | ✅ |
| **Operations** | A management interface for running it standalone | planned |

Parity against the origin copy, and the cutover it existed to make safe, were **removed from the
plan on 2026-08-31**: a parity gate on a deliberate superset fails on every improvement.

## Quick start

```sh
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

The suite needs **no fleet, no network and no inference backend** — it runs against
`roadstead.testing`, described below.

### Two doors

**OpenAI-compatible**, for anything that already speaks it:

```sh
curl localhost:42100/v1/chat/completions -H 'Authorization: Bearer $KEY' \
  -d '{"model": "tier2", "messages": [{"role": "user", "content": "hi"}]}'
```

**The enriched API**, for callers that want what OpenAI's shape cannot carry — declare what you need
rather than which box, and be told what actually served you:

```python
from roadstead.client import AsyncRoadsteadClient

async with AsyncRoadsteadClient("http://localhost:42100", api_key=KEY) as rs:
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

The SDK ships in the package and imports nothing from the server — httpx and the stdlib only.

## Who is calling — API keys

A caller's identity is the **DRR fair-share key**: the string fairness, quotas and budgets are all
accounted against. An API key establishes it, and carries that caller's policy with it — default
priority, an optional deadline floor, an optional admin scope — so it travels with the caller rather
than with the machine it happens to run on.

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

An OpenAI client needs nothing but its `api_key` set — the key rides the `Authorization: Bearer`
header it already sends.

**Out of the box there is no key and no configuration**: loopback and docker-internal callers are
admitted as `internal`, and everything else is refused. That is default-deny with the local-first
case free. To admit a host without issuing it a key, enrol its address:

```sh
ROADSTEAD_ACL='192.0.2.0/24=lan:P3_INGESTION,192.0.2.9=ingest:P4_HYGIENE:1800'
```

An address is deliberately the *weaker* factor — it identifies a host, not a caller. Where the two
disagree the key wins, and a presented key that does not resolve is a 401 rather than a quiet
demotion to whatever the address would have given. `docs/api.md` §1.5 has the full precedence and
the reasoning.

## Overflow, and what it costs

Local capacity is the design centre; remote capacity is what happens when the local fleet is **full**.
For each request Roadstead makes one decision with three outcomes — **dispatch** to a local slot,
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

Those are two different questions and neither implies the other: a caller whose work degrades happily
may still be one whose prompts must never go to a third party.

**Two kinds of money, and they are never added together.** A local call is priced at what renting the
same class of model *would* have cost — that is a saving, and it is reported separately from what you
actually spend. Only real spend counts against a cap.

🚨 **A cap degrades; it never rejects.** Crossing `daily_spend_usd` costs a caller one priority band
and its access to paid spill — and nothing else. It keeps full access to local capacity, and there is
no error code for it. Admission control here is about *capacity*, not billing, and a quota you typo'd
must not be able to take a caller offline.

## A programmable backend, in the box

`roadstead.testing` ships the fake backend the suite runs on, because for a gateway whose thesis is
capacity-aware admission control, *a backend that lies about its capacity on demand* is a capability
rather than test scaffolding — and it is not something you can ask a real GPU for.

```python
from roadstead.testing import FakeBackend, FakeBackendServer, FAULT_CAPACITY_DESYNC

server = FakeBackendServer(FakeBackend(engine="vllm")).start()
server.controller.set_fault(FAULT_CAPACITY_DESYNC, 2)   # accepts 2, 503s the rest,
                                                        # while /props claims otherwise
```

It is a real Starlette app under real uvicorn on a real socket — so real `httpx` and real SSE framing
are exercised, not a mock transport. It speaks both llama.cpp and vLLM wire shapes across
`/v1/chat/completions`, `/embed`, `/rerank`, `/props`, `/v1/models`, `/metrics` and `/health`, and
serves twenty-odd south-face pathologies on command: truncated and invalid JSON, empty completions,
degenerate repetition, schema violations, phantom and truncated tool calls, partial and interleaved
SSE frames, TTFT and inter-token stalls, mid-stream resets, and capacity desync.

It is also the executable form of §4 of `docs/api.md` — what Roadstead requires *of a backend*.

## Documentation

| | |
|---|---|
| `CLAUDE.md` | Orientation, the concurrency invariant, and the engine-behaviour findings that explain why the code is shaped the way it is. **Read before changing anything.** |
| `docs/api.md` | The API surfaces: both north faces, the error contract, the admin/control plane, what Roadstead requires *of a backend*, and the client SDK. |
| `docs/roadmap.md` | **What is being built and why.** Start here for direction. |
| `docs/compatibility.md` | What is stable, what is not, and how to break something on purpose. |
| `docs/history.md` | Closed record of the extraction — where the code came from and what that cost. |
| `docs/evaluation.md` | Why this exists rather than adopting something else — the field survey and decision record. |
| `docs/corpus_and_scrub_plan.md` | What must be scrubbed before this can go public, and why the working tree is not enough. |
| `docs/ledger.md` | Defects that came back, with the guard that now prevents each. |

## Licence

Apache-2.0.
