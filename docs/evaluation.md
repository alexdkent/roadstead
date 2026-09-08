<!-- FRESHNESS: 2026-08-31 original evaluation, §3.4 added 2026-09-08. Landscape figures pulled from the GitHub API on 2026-08-31; re-verify project activity before relying on the dismissal list after ~2026-11. The capability inventory is code-truth as of the 2026-08-31 snapshot, plus live runtime-flag state. §3.4 (NVIDIA PAIR) is a later addendum on a project that did not exist at survey time and is paper-only — nothing in it has been run. -->

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
| NVIDIA PAIR | ✅ | ❌ steering only | partial | ❌ | ❌ | ❌ |

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


### 3.4 Adjacent, not competing: NVIDIA PAIR (added 2026-09-08)

**Personal AI Router**, `github.com/NVIDIA/Personal-AI-Router`, Apache-2.0, public beta v0.1.1
announced 2026-09-03 at IFA. It postdates the survey above, and it is the first entry in this
document that is local-first *and* multi-host, so it is the closest neighbour the list has. It is
also the only one that could plausibly be a Roadstead **backend** rather than an alternative to
Roadstead, which is a relationship this document had no prior example of.

What it does: auto-discovers compatible machines on a LAN over mDNS (or by manual IP), pairs them
behind a gate that blocks all node-to-node traffic until mTLS is established, and routes each
*independent* request to one eligible node. Scheduling reads node readiness, whether a supported
engine is enabled, **whether that exact model is present on that node**, active jobs on node and
engine, and **GPU utilization including non-inference load** — it will steer away from a machine
that is running a game. It fronts Ollama and LM Studio, and re-exposes Ollama-compatible,
LM-Studio-compatible and OpenAI-compatible doors so no agent harness has to learn a cluster API.
NVIDIA states plainly that it does not pool VRAM and does not shard a request: one request, one
node.

**Why it is not a competitor.** It is the "local tools assume one user, so fairness is
uninteresting to them" case from §1, built well. There is no caller identity, no priority band, no
queue, no deferral, and no admission decision — a request is *steered*, never *held*. Its scarce
resource is an idle machine; Roadstead's is a slot on a machine that is never idle. Both are correct
for their own problem, and the two problems do not overlap: PAIR helps when you have hardware nobody
is using, and is silent on what to do when you don't.

**Two convergences worth recording.** NVIDIA states the no-sharding boundary as flatly as this
project does — two independent teams landing on *the router is a scheduler, not a distributed
inference engine* is evidence the boundary is drawn in the right place. And PAIR is the second
NVIDIA entry in the table: Dynamo is the datacentre answer, PAIR the household one, and neither
does caller fair-share.

**The one capability here that Roadstead does not have.** PAIR schedules against **GPU
utilization from work that is not inference**. Roadstead admits against slot counts and knows
nothing about a co-tenant on the same GPU; where that matters in the deployment this was extracted
from, it is handled out-of-band by a separate GPU-slot dispatcher (`on_demand.py`) that Roadstead
leases from rather than models. PAIR treats co-tenancy as one scheduler's problem; Roadstead treats
it as two systems that have to agree, which is the more fragile arrangement. That is a real gap and
it is not closed by anything in §2.

**As a backend.** Fronting a PAIR fabric as a Roadstead endpoint is coherent but strictly
subtractive on the south face, and the reasons are the descriptor fields in
`providers/base.py`:

- `publishes_slot_count` / `publishes_slot_context` / `publishes_context_ceiling` — **all false.**
  Ollama and LM Studio publish no slot count, and PAIR does not synthesize one across the fabric.
  This is blinder than vLLM, which at least yields `max_model_len`. `max_slots` becomes a guess
  about a set of machines whose *membership changes*, and the `max_slots_drift` reconciler has
  nothing to compare against.
- `lists_available_models` — **true, and the one field that improves.** PAIR knows which models sit
  on which node, so an operator picks from a list rather than guessing a slug.
- `grammar_field` — **`None`.** Neither fronted engine takes GBNF, so a grammar-carrying request to
  such an endpoint must *refuse* under the `ProviderError` doctrine rather than downgrade. Any tier
  that relies on an enforced grammar cannot route here at all.
- The **correction layer keeps working**, since it reads response shape rather than engine identity
  — but Ollama is a third engine whose quirks (`mislabels_truncated_tool_calls`,
  `reasoning_budget_field`, terminal-chunk behaviour) have never been characterised here. Assuming
  llama.cpp's answers because Ollama wraps llama.cpp is exactly the inference this descriptor exists
  to forbid.
- **Attribution goes dark.** `attribution.endpoint` is a load-bearing promise: the caller is told
  what actually served it. Behind PAIR the choice is PAIR's, and the Ollama response schema has no
  field to carry it back. A per-node truth becomes a per-fabric one.
- The latency model would learn a **mixture distribution** over heterogeneous hardware whose
  composition changes when somebody shuts a lid — `(endpoint, tier, in-bucket, out-bucket)` buckets
  assume one endpoint means one machine.

Declaring each node as its own ordinary Roadstead endpoint instead recovers every one of those, at
the cost of running llama.cpp on a machine you may not administer. **That trade — a signed
cross-platform installer and a pairing flow, versus a `llama-server` you have to keep alive on
someone else's gaming PC — is the actual argument for PAIR, and it is an operational argument
rather than an architectural one.** Where it wins, it wins on people, not on scheduling.

🚨 **`kind` has no honest value for it.** The enum offers `local` (finite, scarce, fair-shared in
slot-seconds) and `remote` (elastic, governed by cost). Opportunistic borrowed capacity is finite
*and* free *and* may vanish mid-request — none of the three. It would be declared `local` today,
which is the least wrong option and still wrong.

**Unverified.** Nothing here has been run. Open questions, in the order they would change the
analysis: whether PAIR reports which node served a request through any channel a proxy could read;
whether it exposes queue depth or per-node concurrency anywhere; and whether an unreachable or
gaming-busy fabric fails fast or hangs, which decides whether it can be given the deferrable-error
treatment `on_demand.py` gives a dispatcher that is down.

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
  Cloudflare's non-self-hostability. NVIDIA PAIR (§3.4) was added after the survey and is
  paper-only in the strongest sense — it was read, not run, and its own headline number is
  labelled by NVIDIA as a configuration-specific demo rather than a benchmark.
