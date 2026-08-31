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

## ⚠️ Status: authored, never executed

**Written 2026-08-31 on a machine with no Docker. The compose file has not been
brought up and `test_real_engine.py` has never run against a real engine.**
Treat the image tag, the env-var names (`LLAMA_ARG_*`) and the model repo as
*plausible and unverified*; the first person with a Docker daemon should expect
to correct them. The assertions in `conformance.py` are firmer — they are
derived from the fields Roadstead actually reads, and they already run green
against the fake on every suite run.

## The open question this exists to settle

`health.py` reads two different context fields with two different unit
assumptions:

- `default_generation_settings.n_ctx` — taken **as-is**, per-slot. Confirmed
  live: `--ctx-size 131072 --parallel 4` reports `32768` there.
- top-level `props["n_ctx"]` — **divided** by the slot count, i.e. assumed to be
  an aggregate. The comment in `health.py` says outright it is "unconfirmed
  whether it's ever populated as an aggregate".

The fake emits the **same number in both places**, which cannot be correct for
both readings. Nothing breaks today because the reader prefers the first and
never reaches the fallback — but the fallback is emulated wrongly, and a test
that exercised it against the fake would be checking a fiction.

`test_record_the_n_ctx_units` prints what a real engine puts in each. It
deliberately does not assert a preference: the answer should come from the
engine, not from the fake. Once it has been run, record the result in
`docs/ledger.md` and either fix the fake or delete
`test_the_fakes_top_level_n_ctx_is_a_known_fidelity_gap`.
