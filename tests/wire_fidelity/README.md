# Wire-fidelity tests

Everything else in the suite runs against `roadstead.testing` — which encodes
what inference engines did **on the day it was written**. That is a snapshot
with no alarm attached to it. This directory is the alarm.

## The idea

`conformance.py` states the south-face contract once, as executable assertions
traced line-by-line to the code that reads each field (`roadstead/health.py`,
`roadstead/backend.py`). Three files run it:

| file | against | when |
|---|---|---|
| `test_fake_backend_conforms.py` | `roadstead.testing` | every run — fast |
| `test_real_engine.py` | a real `llama-server` | opt-in, `-m wire_fidelity` |
| `test_real_vllm.py` | a real **vLLM** | opt-in, `-m wire_fidelity` |

The two real-engine files audit **opposite things**, which is why they are not one
parameterised file. llama.cpp is checked for what it *publishes*, because discovery
depends on those fields existing. vLLM is checked for what it does **not** publish,
because the decision resting on that absence — concurrency stays config-seeded, with
a drift alert — is only correct while the absence holds. 🚨 An absence is precisely
what a programmable fake can never confirm: it withholds what it was told to withhold
and agrees with the descriptor by construction.

When the fake and a real engine disagree, the fake has drifted from the thing it
claims to imitate, and every test that trusts it has been proving something about a
fiction.

## Running it

```sh
docker compose -f tests/wire_fidelity/compose.yaml up -d
# first run downloads a ~0.5GB model; the healthcheck allows for it

ROADSTEAD_WIRE_FIDELITY_URL=http://127.0.0.1:18080 \
ROADSTEAD_WIRE_FIDELITY_COMPOSE=1 \
    .venv/bin/pytest -m wire_fidelity -v -s tests/wire_fidelity/

docker compose -f tests/wire_fidelity/compose.yaml down
```

`-s` matters: two tests **print what they found** rather than only asserting.

Without `ROADSTEAD_WIRE_FIDELITY_URL` the whole file skips, so a normal
`pytest` run is unaffected — `wire_fidelity` is deselected by default via
`addopts` in `pyproject.toml`.

`ROADSTEAD_WIRE_FIDELITY_COMPOSE=1` asserts you launched via *this* compose
file. The launch-config-dependent tests (per-slot context, the `n_ctx` units)
need to know `--ctx-size` and `--parallel`; they skip against an arbitrary
engine rather than guessing.

## Status: executed 2026-08-31 ✅

Run against a real `llama-server` (**b5350**, Qwen2.5-0.5B-Instruct Q8_0, `--ctx-size 8192
--parallel 4`) on the `roadstead` dev box. **6 passed.** Two guesses in the original compose file
were wrong and are now corrected against the live registry:

| was | is | how it failed |
|---|---|---|
| `ghcr.io/ggml-org/llama.cpp:server-b4739` | `…:server-b5350` | tag does not exist |
| `ggml-org/Qwen2.5-0.5B-Instruct-Q8_0-GGUF` | `Qwen/Qwen2.5-0.5B-Instruct-GGUF:q8_0` | repo does not exist — and the container reports it as *"model is private or does not exist; if you are accessing a gated model, please provide a valid HF token"*, which reads like an auth problem and is not one |

## What a real engine actually publishes

`/props`, whole top level:

```
bos_token · build_info · chat_template · default_generation_settings
eos_token · modalities · model_path · total_slots
```

| field | value | |
|---|---|---|
| `default_generation_settings.n_ctx` | **2048** | 8192/4 → **per-slot confirmed**, on a build 611 versions newer than the note that first established it |
| `total_slots` | 4 | the only slot source that exists |
| `default_generation_settings.n_parallel` | **absent** | …and it is what `health.py` *prefers* |
| top-level `n_ctx` | **absent** | not an aggregate; not anything |
| `slots` | **absent** | |

`/v1/models` `data[0]`: `id` is the **full GGUF path** (a deployment sets a friendly name with
`-a`), `meta` = `{vocab_type, n_vocab, n_ctx_train, n_embd, n_params, size}`, and **no `root`**, **no
`max_model_len`** — both vLLM-only.

Streaming: `finish_reason` **rides alone on a terminal chunk with an empty delta**, as the correction
layer assumes.

## The open question, settled — and not the way either answer expected

`health.py` divides a top-level `props["n_ctx"]` by the slot count, i.e. reads it as an aggregate,
while conceding it is "unconfirmed whether it's ever populated as an aggregate". **It is not
populated at all.** That fallback is dead code against a current build.

The sharper finding is the one next to it: the fake was publishing `n_parallel`, which a real engine
does not, and `health.py` *prefers* `n_parallel` over `total_slots`. So the fallback that real
discovery entirely depends on **was never exercised by any test**. Deleting it as redundant would
have kept the suite green and broken production capacity discovery.

`roadstead.testing` now defaults to the verified narrow shape. The old superset is
`props_profile="legacy"`, kept so the aggregate fallback stays drivable for older builds. Full
write-up in `docs/ledger.md`.


## Re-run against a SECOND real llama.cpp, 2026-09-01 ✅

The findings above come from **b5350** on a 0.5B model in compose. Re-run against a
production engine — build **`b1-6d05498`**, a 27B Q4_0 on a different accelerator,
`total_slots: 6`, per-slot `n_ctx` 262144. **6 passed, 1 skipped.**

**Every b5350 finding survived a very large build gap**: still no top-level `n_ctx`,
still no `default_generation_settings.n_parallel`, `total_slots` still the only slot
source that exists, and `finish_reason` still rides alone on an empty-delta chunk.

**`default_generation_settings.n_ctx` is per-slot — now corroborated independently.**
This build serves `/slots` (`endpoint_slots: true`), and each of the six entries
reports its own `n_ctx` of 262144, matching `default_generation_settings.n_ctx`
exactly. The per-slot reading has therefore been confirmed twice by two different
mechanisms, which is worth more than the single division that established it.

**The props top level has grown from 8 keys to 19** — new since b5350:
`chat_template_caps · cors_proxy_enabled · endpoint_metrics · endpoint_props ·
endpoint_slots · is_sleeping · media_marker · model_alias · model_ftype · ui ·
ui_settings`. None of them breaks discovery, and none is read. Recorded because the
*direction* matters: this endpoint grows fields rather than losing them, so a test
that asserts an exact key set would fail on every build bump while nothing was wrong.

### 🚨 What the run actually caught was a bug in these tests

Two of them compared against `COMPOSE_PARALLEL` / `EXPECTED_PER_SLOT` **without** the
`_launched_by_compose()` gate their siblings use. Pointed at any engine but compose's
they failed on the *launch config* rather than on the wire shape —
`262144 != 2048`, and `6 != 4`, both correct values for that server. That defeats the
purpose: this directory is supposed to be an alarm you can aim at a production
backend, and until now it could only be aimed at the one it ships with.

Each is now split — the engine-general claim (`n_ctx` is a positive int and discovery
resolves *some* slot count, in the documented order) runs everywhere; the exact
number is asserted only under the gate. 🚨 That distinction is the general rule here:
**the number is a property of the launch, the resolution order is a property of the
engine**, and only the second is what this directory exists to watch.


## The vLLM half — executed 2026-09-01 ✅

Run read-only against a real vLLM serving live traffic (DeepSeek-V4-Flash, two-node
pipeline-parallel, `max_model_len` 1,048,576). **5 passed**, 2 skipped (the inference
pair — see below). **Every descriptor claim in `providers/vllm.py` held.**

```sh
ROADSTEAD_WIRE_FIDELITY_VLLM_URL=http://<host>:<port> \
    .venv/bin/pytest -m wire_fidelity -v -s tests/wire_fidelity/test_real_vllm.py
```

| claim | confirmed by |
|---|---|
| `publishes_slot_count=False` | `/props` **404**, `/slots` **404**, no `max_num_seqs` anywhere readable |
| `publishes_slot_context=False` | nothing per-slot is published — there is no slot to have one |
| `publishes_context_ceiling=True` | `data[0].max_model_len`, the per-request ceiling **directly** |
| `publishes_prefix_cache_metrics=True` | `vllm:prefix_cache_{queries,hits}_total`, the exact names `compute_cache_stats` reads |
| fingerprint prefers `root` | `root` is a weights path; `id` is the `--served-model-name`, and they differ |

`/v1/models` `data[0]` keys, whole: `created · id · max_model_len · object · owned_by
· parent · permission · root`. **No `meta`** — the llama.cpp fingerprint fallback is
as engine-specific as the `root` path it falls back from.

### 🚨 The near miss: `kv_cache_max_concurrency` is not a slot count

`/metrics` publishes `vllm:cache_config_info`, whose labels include
`kv_cache_max_concurrency` — a float that looks exactly like the number vLLM is
documented not to expose. It is `kv_cache_size_tokens / max_model_len`: how many
*full-context* requests the KV cache would hold. On this long-context server it reads
**1.67**, while the engine comfortably fields many more short ones. Seeding
`max_slots` from it would cap a busy endpoint at one and do it wearing the authority
of a discovered fact. `check_no_published_concurrency` searches for `max_num_seqs`
and `max_num_batched_tokens` by name and fails loudly if either appears — that would
make vLLM slot discovery real, and is an alarm to act on rather than route around.

### Inference is a SECOND opt-in — run 2026-09-01, 7/7 ✅

The five assertions above are GETs and cost nothing, so they are safe against an
endpoint carrying real traffic. The `finish_reason` terminal-chunk rule and the sync
completion shape **POST**, and the only vLLM within reach of this project is one
serving live traffic. They skip unless:

```sh
ROADSTEAD_WIRE_FIDELITY_VLLM_INFERENCE=1
```

Pointing the test at an engine is not consent to generate on it.

🚨 **Run with consent 2026-09-01: the terminal-chunk rule holds on vLLM too.**
`finish_reason` rides alone on a chunk with an empty `delta`, and the stream carries
its own `[DONE]`. That rule was established on llama.cpp and `correction.py` has
applied it to *every* backend ever since — so until this run the second engine was
being repaired against a rule measured on the first. Both engines now agree, which is
what makes the repair safe rather than merely untested.
