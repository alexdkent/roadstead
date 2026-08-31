# Wire-fidelity tests

Everything else in the suite runs against `roadstead.testing` — which encodes
what inference engines did **on the day it was written**. That is a snapshot
with no alarm attached to it. This directory is the alarm.

## The idea

`conformance.py` states the south-face contract once, as executable assertions
traced line-by-line to the code that reads each field (`roadstead/health.py`,
`roadstead/backend.py`). Two files run it:

| file | against | when |
|---|---|---|
| `test_fake_backend_conforms.py` | `roadstead.testing` | every run — fast |
| `test_real_engine.py` | a real `llama-server` | opt-in, `-m wire_fidelity` |

When the two disagree, the fake has drifted from the thing it claims to imitate,
and every test that trusts it has been proving something about a fiction.

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
