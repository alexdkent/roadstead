<!-- Provenance: copied verbatim from the origin monorepo's
     originfleet/docs/llmproxy_vs_open_source_evaluation_2026-08.md on 2026-08-31.
     It is a decision RECORD — paths and names inside refer to the origin project and are
     correct as history. Do not 'fix' them to Roadstead names. -->

<!-- FRESHNESS: 2026-08-31 — original evaluation. Landscape figures pulled from the GitHub API on 2026-08-31; re-verify project activity before relying on the dismissal list after ~2026-11. The llmproxy capability inventory is code-truth as of a72e2c3a7 + live runtime-flag state. -->

# llmproxy vs the open-source field — 2026-08-31

**Status:** current. Decision record. Evaluates whether to abandon `llmproxy` for an existing
open-source LLM gateway, extract it as a standalone open-sourceable project, or leave it as a
fleet-internal component. **Verdict: extract.** The execution plan is
`llmproxy_extraction_plan_2026-08.md`.

---

## 1. Verdict

Three outcomes were on the table. Two are refuted by evidence, not preference.

| Outcome | Verdict |
|---|---|
| **1. Abandon** — replace llmproxy with an existing OSS gateway | ❌ Ruled out — nothing to abandon *to* |
| **2. Extract** — split into a standalone project, publish it | ✅ **Recommended** |
| **3. Leave alone** — too interconnected to move | ❌ Ruled out — the premise is false |

**Outcome 1 fails because the replacement does not exist.** The gateway market has bifurcated, and
llmproxy sits on the fault line. Every project with genuine capacity-aware admission control
inherited it from the Kubernetes Gateway API Inference Extension, whose mechanism is scraping
*vLLM's Prometheus metrics* from *pods in an `InferencePool`*. Both halves are load-bearing:
llama.cpp speaks a different metric vocabulary, and systemd units on six machines are not pods.
Everything self-hosted and non-Kubernetes sits below the line and treats a backend as an opaque
endpoint with a health bit.

**Outcome 3 fails because llmproxy was already nearly free-standing.** It reached into the rest of
the package exactly four times, three of them lazy function-scope imports. Those four were severed
on 2026-08-31 (commits `6ca1c83f6` + `a36651c83`); 29 of 30 library modules now import nothing
outside the package. The thing genuinely welded into the fleet is the model catalog (~2,500 lines),
not the 18,600-line runtime.

**The qualification that shapes sequencing:** extraction will **not** reduce maintenance burden — it
raises it, since a public project adds issues, releases and API-stability obligations. And the code
is still moving fast (27,581 insertions against an 18,600-line subsystem in four months). Hence:
**extract structurally now, publish when churn settles.**

---

## 2. What llmproxy is

A Python asyncio proxy on `:42161`, the single front door for all fleet LLM traffic across six
machines — several llama.cpp servers and a vLLM tensor-parallel pair. OpenAI-compatible on the
front, model-authoritative on the back.

| Measure | Value |
|---|---|
| Source | 18,604 lines, 30 modules (32 after the sever) |
| Tests | 24,782 lines, 106 files — more test code than source |
| Churn | 257 commits in 4 months (~64/mo), 4% of all repo commits |
| Inbound coupling | 4 imports from originfleet (now 0 in the library core) |
| Secrets in source/config | none |

A source-level audit catalogued 47 distinct capabilities. Most are commodity. **Four are not**, and
they form one coherent idea: *the proxy empirically models its backends' capacity and latency
behaviour, and uses that model for both admission control and deadline setting.*

### 2.1 DRR fair-share denominated in slot-seconds

Three strict priority bands (interactive › foreground › background); deficit round-robin across
*agents* within each band; per-endpoint concurrency gating; a configurable background floor
guarantee so low-priority work cannot starve. Starvation rescue keys on head-of-queue wait, not
balance sign. `scheduler.py:188-233,542-614`, `agent_budget.py:14-61,163-224`.

The unit of fairness is **backend occupancy time**, EWMA-calibrated per endpoint, charged to the DRR
balance up front and retroactively corrected once actual duration is known (`cost_model.py`). Most
gateways fair-share over request count or tokens; occupancy time is the correct currency when
backends have fixed slot counts.

### 2.2 Runtime capacity discovery by probing the engine

The poller reads llama.cpp `/props` for real `n_parallel`/`total_slots` and per-slot `n_ctx`, feeding
admission, the context gate, DRR capacity and the timeout model (`health.py:761-809`).

🚨 **Asymmetric, and the asymmetry matters:** vLLM exposes only `max_model_len`, so vLLM concurrency
stays **config-seeded** with a `max_slots_drift` shadow alert (`health.py:810-826`). It is not
"discovers capacity from both".

### 2.3 Intent-declared timeouts from a learned latency model

Callers declare priority and interactivity; the proxy owns the number. It learns from its own
measured end-to-end latency (queue wait + inference), conditioned on `(endpoint, tier, input bucket,
output bucket)` across 8×5 buckets, returning `recommended = max(p99 × 1.5, floor)` under load-aware
(`surge_factor`) and context-length-aware (`size_stretch`) multipliers, clamped to per-class
ceilings. It bootstraps from `proxy_completions` at startup, so it survives restarts.
`timeout_model.py:346-646`.

The deadline is then **soft**: the streaming path can extend it by probing backend decode-progress
counters rather than killing work that is still advancing (`lifecycle.py:1422-1637`).

### 2.4 A response-correction layer

Enforced live: GBNF grammar validate-and-repair, JSON-schema backstop with bounded retry,
degeneration detection and re-dispatch, truncation integrity, thinking recovery, egress conformance
shadow-detection, and two SSE repairs — synthesising a missing terminal `finish_reason` chunk (only
when the backend sent its own `[DONE]`, i.e. asserted completeness) and splitting coalesced terminal
chunks.

Plus engine-quirk workarounds: vLLM `qwen3_xml` phantom tool-call sanitizer, Anthropic→OpenAI
image-block normalisation, strict-alternation fixes for Mistral templates, bare `json_object`
stripping on whitespace-banned backends.

> **Code-truth vs runtime-truth.** Several of these gates read as off-by-default in `flags.py`. They
> are shadow-first gates flipped on in production: `runtime_flags.json` carries
> `context_gate_enforce`, `inject_stream_usage`, `smart_default_timeout`,
> `unknown_endpoint_enforce` all true, and container env carries
> `ROADSTEAD_PROXY_{UNIFORM_CORRECTION,SCHEMA_BACKSTOP,ENDPOINT_COOLDOWN}=1`. The one genuinely
> inert gate is `vision_capability_enforce`.

---

## 3. The field

~25 projects surveyed. The decisive test is a six-way conjunction: run without Kubernetes, gate
admission on real backend capacity, discover that capacity at runtime, speak llama.cpp, speak vLLM,
fair-share between callers.

| Project | No K8s | Capacity admission | Runtime discovery | llama.cpp | vLLM | Caller fair-share |
|---|---|---|---|---|---|---|
| **llmproxy** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ DRR |
| Paddler | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| llm-d flow control | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| NVIDIA Dynamo | ✅ | ✅ | partial | dev flag | ✅ | hints only |
| Ray Serve LLM | ✅ | ✅ | ❌ static | ❌ | ✅ | ❌ |
| AIBrix standalone | ✅ | opt-in | ✅ | ❌ | ✅ | reduced |
| LiteLLM | ✅ | ❌ static ceiling | ❌ | opaque | opaque | beta |
| Olla | ✅ | ❌ by design | ❌ | ✅ | ✅ | ❌ |

**No project satisfies all six.**

**Olla is the most instructive data point.** It supports llama.cpp and vLLM first-class and documents
llama.cpp's `/slots` endpoint — it had every opportunity to build this. Its documented decision is
the opposite: pass the 503 through and tell the client to back off (*"Slot monitoring is not
available through Olla proxy endpoints"*). Its config reference has no concurrency, queue or capacity
setting at all. That is the category boundary, written down.

### 3.1 Two capabilities have no counterpart anywhere

- **The correction layer.** Not one surveyed project offers grammar validation-with-repair, a schema
  retry backstop, degeneration detection, truncation integrity or malformed-SSE repair as a gateway
  feature. The industry answer to structured output is constrained decoding *inside the engine*
  (xgrammar, outlines, lm-format-enforcer) — a materially different guarantee: it prevents malformed
  *tokens* at generation time and does nothing about a truncated stream, a degenerate repetition
  loop, a coalesced terminal chunk or a missing `finish_reason`.
- **Intent-declared timeouts.** Every product surveyed takes a timeout as a number. Nothing computes
  it from declared intent against a learned latency distribution.

### 3.2 The two near-misses, ruled out

**Paddler** (Rust, Apache-2.0, 1,665★, v4.1.0 2026-07-19) — slot-aware buffered admission with live
llama.cpp capacity, single non-K8s binary, even exposes GBNF. Out on three counts: **no vLLM at
all**; **no priority or fair-share** (FIFO buffer, no bands, no per-caller accounting); and as of v4
it **embeds llama.cpp as a library and replaces `llama-server`**, with slots declared at agent launch
(`paddler agent --slots 4`) rather than probed. Adoption means migrating the serving layer, not the
gateway. Worth reading as prior art on the llama.cpp half.

**llm-d flow control** — GA in Red Hat AI Inference 3.5 (2026-08), and *recognisably the same design
as llmproxy's scheduler, arrived at independently*: priority bands → inter-flow fairness → FCFS,
gated by a saturation detector combining queue depth and KV pressure. Out on two hard constraints:
**Kubernetes is not optional** (its "standalone mode" removes the Gateway API dependency, not the
cluster) and **the saturation signal is vLLM's Prometheus vocabulary**, so there is no llama.cpp path.
Its fairness policies are round-robin and strict-fairness — not deficit round-robin, so the algorithm
is coarser than llmproxy's.

> **This strengthens rather than weakens the case.** Red Hat, Google and the vLLM project converging
> on the architecture independently validates it; Paddler earning 1,665★ for a strictly smaller slice
> is direct evidence of unmet demand for the larger one.

### 3.3 Dismissal list

- **RouteLLM** — dead (last commit 2024-08-09); also solved model *selection*, not admission.
- **HF text-generation-inference** — archived 2026-03-21.
- **Portkey** — OSS frozen 2026-05-25; Palo Alto Networks acquisition closed 2026-05-29.
- **Helicone** — acquired by Mintlify, maintenance mode; the actual proxy binary is GPLv3 in a
  separate repo, untouched since 2025-11-21.
- **Envoy AI Gateway / kgateway / Higress** — hard Kubernetes (Envoy documents K8s ≥1.32); their
  admission control is borrowed EPP.
- **Kong AI Gateway** — no capacity concept; every multi-backend feature is Enterprise-licensed.
- **Apache APISIX** — strong 2026 AI features (`ai-cache`, semantic balancing) but rate limiting and
  caching only.
- **Bifrost** — "adaptive load balancer" is health/latency-inferred and Enterprise-gated; its
  concurrency knob throttles Bifrost's own worker pool, not the backend.
- **vLLM production-stack / KServe / AIBrix (full)** — hard K8s; KServe's llama.cpp support is an
  open, unaccepted feature request.
- **SGLang gateway / vllm-project/router** — engine-specific; global token bucket, not per-backend.
- **llama-swap / llama.cpp Router Mode / GPUStack / LocalAI / Harbor** — model swapping, deploy-time
  bin-packing, or meta-launchers. Different problem.
- **Traefik AI Gateway / Cloudflare AI Gateway** — no OSS path / SaaS only.

---

## 4. Extraction cost

**Inbound (severed 2026-08-31).** Four imports: `prompt_security.record_security_event`
(`health.py:37`, module scope), `ship_version.log_ship_version`, `observability.degradation`,
`metrics.render_prometheus` (the last three lazy). Now routed through `llmproxy/hooks.py` (the
integration seam) and `llmproxy/metrics.py` (vendored renderer), with `__main__.py` holding the only
— soft, guarded — originfleet imports.

**Outbound.** 38 files import from llmproxy, but production coupling is narrow: ~14 import
`model_catalog`; a handful pull `normalize_endpoint`, `DEFAULT_ENDPOINTS`, `estimate_input_tokens`,
`FLOOR_S`, `cache_stats`, `normalize_and_validate`. **No production code outside `__main__` touches
`ProxyService`** — all 33 such imports are white-box tests.

**One invariant crosses the boundary:** `framework/timeout_advice.py` imports
`FLOOR_S`/`_DEFAULT_FLOOR_S`, a documented synced mirror of `models.yaml` pinned by
`test_timeout_floor_yaml_sync`.

**Open-source readiness.** No secrets. Fleet-specific *code* is ~20 lines: seed registrations in
`acl.py` (an env override format already exists, so these are defaults) and alias tables in
`usage_rates.py`. Third-party surface is six: `httpx`, `starlette`, `uvicorn`, `PyYAML`,
`jsonschema`, `json_repair`. **The repo has no LICENSE file** — a prerequisite, not a blocker.

### 4.1 A correction to the survey's own conclusion

The landscape survey closed by suggesting ~40% of llmproxy is undifferentiated commodity that
LiteLLM could absorb — naming the model catalog, per-agent budgets, cost tracking and the event log.
**That does not survive contact with the code.** The event log is load-bearing (the timeout model
bootstraps from `proxy_completions`; DRR budgets persist through it); per-agent budgets are
denominated in slot-seconds and feed DRR directly; the catalog is not a provider list but the
differentiators' configuration surface (timeout floors, failover targets, thinking kwargs, vision
flags). Genuinely commodity: `acl.py` (289 lines), `usage_rates.py` (205) and parts of the HTTP
surface — **10–15%**, and shedding it would buy close to nothing.

---

## 5. Method and limits

- Capability inventory: full read of all 30 modules by a `dsh` auditor persona, cross-checked
  first-hand on `scheduler.py`, `cost_model.py`, `timeout_model.py`. Its flag findings were
  code-truth; runtime state was read separately from the live container.
- Landscape: GitHub API 2026-08-31 plus primary documentation; decision-critical claims verified
  against source or docs.
- **This is a paper evaluation.** No alternative was stood up and driven with real traffic. The
  dismissals rest on documentation and source reading — strong for capability questions, weak for
  operational ones.
- Unverified items flagged by the survey: LiteLLM's contested MIT/enterprise boundary; Bifrost's
  `json_schema` retry behaviour; Higress's InferencePool admission (inferred); Dynamo's metric
  collection mechanism; **whether Paddler retains a mode for fronting an external server** (the
  single item most capable of changing the near-miss analysis); Portkey's claimed fair-share;
  Cloudflare's non-self-hostability.
