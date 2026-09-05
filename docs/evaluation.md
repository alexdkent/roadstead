<!-- FRESHNESS: 2026-08-31 — original evaluation. Landscape figures pulled from the GitHub API on 2026-08-31; re-verify project activity before relying on the dismissal list after ~2026-11. The capability inventory is code-truth as of the 2026-08-31 snapshot, plus live runtime-flag state. -->

# Roadstead vs the open-source field — 2026-08-31

**Status:** current. This is a survey of the open-source LLM gateway landscape, done to check
whether Roadstead's capabilities have a counterpart anywhere else. They don't: two of them have no
counterpart in ~25 projects surveyed, and everything that comes close inherited its design from a
single shared ancestor.

---

## 1. The gateway market has bifurcated

Every project with genuine capacity-aware admission control inherited it from the Kubernetes
Gateway API Inference Extension, whose mechanism is scraping vLLM's Prometheus metrics from pods in
an `InferencePool`. Both halves of that are load-bearing: llama.cpp speaks a different metric
vocabulary than vLLM, and a self-hosted process is not a pod. Everything non-Kubernetes sits below
that line and treats a backend as an opaque endpoint with a health bit.

Roadstead sits on the fault line: it runs without Kubernetes, gates admission on real backend
capacity, discovers that capacity at runtime, speaks both llama.cpp and vLLM, and fair-shares
between callers. Nothing surveyed satisfies all of that at once.

---

## 2. What Roadstead does

A source-level audit catalogued 47 distinct capabilities. Most are commodity. Four are not, and
they form one coherent idea: the proxy empirically models its backends' capacity and latency
behavior, and uses that model for both admission control and deadline setting.

### 2.1 DRR fair-share denominated in slot-seconds

Three strict priority bands (interactive > foreground > background); deficit round-robin across
callers within each band; per-endpoint concurrency gating; a configurable background floor
guarantee so low-priority work cannot starve. Starvation rescue keys on head-of-queue wait, not
balance sign. `scheduler.py:188-233,542-614`, `agent_budget.py:14-61,163-224`.

The unit of fairness is backend occupancy time, EWMA-calibrated per endpoint, charged to the DRR
balance up front and retroactively corrected once actual duration is known (`cost_model.py`). Most
gateways fair-share over request count or tokens; occupancy time is the correct currency when
backends have fixed slot counts.

### 2.2 Runtime capacity discovery by probing the engine

The poller reads llama.cpp `/props` for real `n_parallel`/`total_slots` and per-slot `n_ctx`,
feeding admission, the context gate, DRR capacity and the timeout model (`health.py:761-809`).

Asymmetric, and the asymmetry matters: vLLM exposes only `max_model_len`, so vLLM concurrency stays
config-seeded with a `max_slots_drift` shadow alert (`health.py:810-826`). It is not "discovers
capacity from both" — that would overstate what's actually happening.

### 2.3 Intent-declared timeouts from a learned latency model

Callers declare priority and interactivity; the proxy owns the number. It learns from its own
measured end-to-end latency (queue wait + inference), conditioned on `(endpoint, tier, input
bucket, output bucket)` across 8x5 buckets, returning `recommended = max(p99 * 1.5, floor)` under
load-aware (`surge_factor`) and context-length-aware (`size_stretch`) multipliers, clamped to
per-class ceilings. It bootstraps from `proxy_completions` at startup, so it survives restarts.
`timeout_model.py:346-646`.

The deadline is then soft: the streaming path can extend it by probing backend decode-progress
counters rather than killing work that is still advancing (`lifecycle.py:1422-1637`).

### 2.4 A response-correction layer

Enforced live: GBNF grammar validate-and-repair, JSON-schema backstop with bounded retry,
degeneration detection and re-dispatch, truncation integrity, thinking recovery, egress conformance
shadow-detection, and two SSE repairs — synthesizing a missing terminal `finish_reason` chunk (only
when the backend sent its own `[DONE]`, i.e. asserted completeness) and splitting coalesced
terminal chunks.

Plus engine-quirk workarounds: a vLLM phantom tool-call sanitizer for `qwen3_xml`, Anthropic to
OpenAI image-block normalization, strict-alternation fixes for Mistral templates, bare
`json_object` stripping on whitespace-banned backends.

> **Code-truth vs runtime-truth.** Several of these gates read as off-by-default in `flags.py`.
> They are shadow-first gates flipped on in production: the runtime flag store carries
> `context_gate_enforce`, `inject_stream_usage`, `smart_default_timeout`, `unknown_endpoint_enforce`
> all true, and deployment env carries the uniform-correction, schema-backstop and
> endpoint-cooldown flags on. The one genuinely inert gate is `vision_capability_enforce`.

---

## 3. The field

~25 projects surveyed. The decisive test is a six-way conjunction: run without Kubernetes, gate
admission on real backend capacity, discover that capacity at runtime, speak llama.cpp, speak vLLM,
fair-share between callers.

| Project | No K8s | Capacity admission | Runtime discovery | llama.cpp | vLLM | Caller fair-share |
|---|---|---|---|---|---|---|
| **Roadstead** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ DRR |
| Paddler | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| llm-d flow control | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| NVIDIA Dynamo | ✅ | ✅ | partial | dev flag | ✅ | hints only |
| Ray Serve LLM | ✅ | ✅ | ❌ static | ❌ | ✅ | ❌ |
| AIBrix standalone | ✅ | opt-in | ✅ | ❌ | ✅ | reduced |
| LiteLLM | ✅ | ❌ static ceiling | ❌ | opaque | opaque | beta |
| Olla | ✅ | ❌ by design | ❌ | ✅ | ✅ | ❌ |

**No project satisfies all six.**

**Olla is the most instructive data point.** It supports llama.cpp and vLLM first-class and
documents llama.cpp's `/slots` endpoint — it had every opportunity to build this. Its documented
decision is the opposite: pass the 503 through and tell the client to back off ("Slot monitoring is
not available through Olla proxy endpoints"). Its config reference has no concurrency, queue or
capacity setting at all. That's the category boundary, written down.

### 3.1 Two capabilities have no counterpart anywhere

- **The correction layer.** Not one surveyed project offers grammar validation-with-repair, a
  schema retry backstop, degeneration detection, truncation integrity or malformed-SSE repair as a
  gateway feature. The industry answer to structured output is constrained decoding inside the
  engine (xgrammar, outlines, lm-format-enforcer) — a materially different guarantee: it prevents
  malformed tokens at generation time and does nothing about a truncated stream, a degenerate
  repetition loop, a coalesced terminal chunk or a missing `finish_reason`.
- **Intent-declared timeouts.** Every product surveyed takes a timeout as a number. Nothing
  computes it from declared intent against a learned latency distribution.

### 3.2 The two near-misses, ruled out

**Paddler** (Rust, Apache-2.0, 1,665 stars, v4.1.0 2026-07-19) — slot-aware buffered admission with
live llama.cpp capacity, single non-K8s binary, even exposes GBNF. Ruled out on three counts: no
vLLM at all; no priority or fair-share (FIFO buffer, no bands, no per-caller accounting); and as of
v4 it embeds llama.cpp as a library and replaces `llama-server`, with slots declared at agent launch
(`paddler agent --slots 4`) rather than probed. Adoption would mean migrating the serving layer, not
just the gateway. Worth reading as prior art on the llama.cpp half.

**llm-d flow control** — GA in Red Hat AI Inference 3.5 (2026-08), and recognizably the same design
as Roadstead's scheduler, arrived at independently: priority bands, then inter-flow fairness, then
FCFS, gated by a saturation detector combining queue depth and KV pressure. Ruled out on two hard
constraints: Kubernetes is not optional (its "standalone mode" removes the Gateway API dependency,
not the cluster) and the saturation signal is vLLM's Prometheus vocabulary, so there is no
llama.cpp path. Its fairness policies are round-robin and strict-fairness — not deficit
round-robin, so the algorithm is coarser.

> This strengthens rather than weakens the case. Red Hat, Google and the vLLM project converging on
> the architecture independently validates it; Paddler earning 1,665 stars for a strictly smaller
> slice is direct evidence of unmet demand for the larger one.

### 3.3 Dismissal list

- **RouteLLM** — dead (last commit 2024-08-09); also solved model selection, not admission.
- **HF text-generation-inference** — archived 2026-03-21.
- **Portkey** — OSS frozen 2026-05-25; Palo Alto Networks acquisition closed 2026-05-29.
- **Helicone** — acquired by Mintlify, in maintenance mode; the actual proxy binary is GPLv3 in a
  separate repo, untouched since 2025-11-21.
- **Envoy AI Gateway / kgateway / Higress** — hard Kubernetes (Envoy documents K8s >=1.32); their
  admission control is borrowed from the Gateway API Inference Extension.
- **Kong AI Gateway** — no capacity concept; every multi-backend feature is Enterprise-licensed.
- **Apache APISIX** — strong 2026 AI features (`ai-cache`, semantic balancing) but rate limiting and
  caching only.
- **Bifrost** — its "adaptive load balancer" is health/latency-inferred and Enterprise-gated; its
  concurrency knob throttles Bifrost's own worker pool, not the backend.
- **vLLM production-stack / KServe / AIBrix (full)** — hard K8s; KServe's llama.cpp support is an
  open, unaccepted feature request.
- **SGLang gateway / vllm-project/router** — engine-specific; global token bucket, not per-backend.
- **llama-swap / llama.cpp Router Mode / GPUStack / LocalAI / Harbor** — model swapping,
  deploy-time bin-packing, or meta-launchers. A different problem.
- **Traefik AI Gateway / Cloudflare AI Gateway** — no OSS path, or SaaS only.

---

## 4. Method and limits

- Capability inventory: full read of all modules, cross-checked first-hand on `scheduler.py`,
  `cost_model.py`, `timeout_model.py`. Flag findings were code-truth; runtime state was read
  separately from a live deployment.
- Landscape: GitHub API 2026-08-31 plus primary documentation; decision-critical claims verified
  against source or docs.
- This is a paper evaluation. No alternative was stood up and driven with real traffic. The
  dismissals rest on documentation and source reading — strong for capability questions, weak for
  operational ones.
- Unverified items flagged by the survey: LiteLLM's contested MIT/enterprise boundary; Bifrost's
  `json_schema` retry behavior; Higress's InferencePool admission (inferred); Dynamo's metric
  collection mechanism; whether Paddler retains a mode for fronting an external server (the single
  item most capable of changing the near-miss analysis); Portkey's claimed fair-share;
  Cloudflare's non-self-hostability.
