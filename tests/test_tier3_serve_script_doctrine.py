"""tier3's serve script is load-bearing and was UNGUARDED.

`models.yaml` documents several tier3 flags as load-bearing and cited
`tier3_gates.py` as the enforcement — but that file lives only on anvil at
`/opt/anvil-bench/` and is not in this repo, so neither CI nor `ship.sh`
could run it. Nothing caught drift in the flags whose regressions are SILENT:

  * `enable_thinking: true` makes the model spend the whole budget reasoning and
    return EMPTY content. status=ok, no error, no alert — callers just get "".
  * dropping `disable_any_whitespace` lets structured output run away on
    whitespace: 10/10 failures on a real gate, each burning the full budget and
    30-60 s of a tier3 slot, surfacing downstream as "the model had nothing to say".
  * dropping `--long-prefill-token-threshold` lets one giant prefill starve an
    interactive request for 45 s (measured; 1.86 s with it).

So the script is now VENDORED into the repo and this pins it against
models.yaml. The vendored copy is the reviewable source of truth; anvil still
runs its own copy at /opt/anvil-bench/serve_tier3_prod.sh, so when you change
one, change both — `test_vendored_script_matches_catalog` is what catches the
models.yaml half of that drift.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "infra" / "anvil" / "vllm" / "serve_tier3_prod.sh"
CATALOG = REPO / "originfleet" / "llmproxy" / "models.yaml"


@pytest.fixture(scope="module")
def script() -> str:
    assert SCRIPT.is_file(), f"vendored serve script missing at {SCRIPT}"
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def tier3() -> dict:
    cat = yaml.safe_load(CATALOG.read_text())
    models = cat["models"] if "models" in cat else cat
    # tier3 is the `reasoner` stanza (role llama-thinker).
    stanza = models.get("reasoner")
    assert stanza is not None, "reasoner (tier3) stanza not found in models.yaml"
    return stanza


def _flag(script: str, name: str) -> str | None:
    """Value following `--name` on the docker run line, or None."""
    m = re.search(rf"--{re.escape(name)}\s+(\S+)", script)
    return m.group(1) if m else None


def test_no_comments_inside_the_docker_run_continuation(script: str) -> None:
    """A '#' line inside a trailing-backslash command is NOT a comment — it
    swallows every flag after it. This happened TWICE on 2026-07-31: the first
    time it silently dropped enable_thinking=false, the tool/reasoning parsers
    and the structured-outputs config; tier3 came up "healthy" with five flags
    missing. Only an explicit `docker inspect` of the running args caught it."""
    body = re.search(r"^exec docker run.*?chat-template-kwargs.*?$",
                     script, re.S | re.M)
    assert body, "could not locate the docker run block"
    offenders = [ln for ln in body.group(0).splitlines() if ln.lstrip().startswith("#")]
    assert not offenders, (
        "comment line(s) inside the docker run continuation will swallow every "
        f"flag after them: {offenders}"
    )


def test_enable_thinking_is_false(script: str) -> None:
    """TRUE returns EMPTY content on every caller. Measured: max_tokens=120
    thinking=true -> out=120, finish=length, content_len=0, status=ok."""
    assert '"enable_thinking": false' in script, (
        "enable_thinking must be false — true makes tier3 spend the whole token "
        "budget reasoning and return nothing, with no error anywhere"
    )
    assert '"enable_thinking": true' not in script


def test_structured_outputs_disable_any_whitespace(script: str) -> None:
    """Without this, structured output runs away emitting whitespace at a
    structural position until max_tokens (10/10 on a real gate)."""
    cfg = re.search(r"--structured-outputs-config\s+'([^']+)'", script)
    assert cfg, "--structured-outputs-config missing"
    assert '"disable_any_whitespace":true' in cfg.group(1).replace(" ", ""), (
        "disable_any_whitespace must be true — it is the fix for the structured "
        "output whitespace runaway, not an optimisation"
    )


def test_disable_any_whitespace_is_declared_in_the_catalog(script: str, tier3: dict) -> None:
    """BIDIRECTIONAL pin: script flag ⟺ ``policy.disable_any_whitespace``.

    The proxy cannot introspect a backend LAUNCH flag, so models.yaml declares it
    and ``Correction.apply_json_object_guard`` acts on the declaration — it strips
    a bare ``response_format:{"type":"json_object"}``, which on a whitespace-banned
    grammar makes the legal complete document ``{}`` the greedy path (measured
    2026-08-01: bare json_object -> ``{}``, 2 chars, finish_reason=stop; the same
    prompt with no response_format -> 549 chars of valid JSON).

    Pinning only one direction leaves two silent failures. Script-without-catalog:
    the guard never fires and callers get ``{}`` (3,100 forum-agent defers, executions
    ~300/day -> 13). Catalog-without-script: the proxy strips a constraint the
    backend would have honoured. So the two must move together — as must the
    vendored script and anvil's own copy at /opt/anvil-bench/serve_tier3_prod.sh.
    """
    cfg = re.search(r"--structured-outputs-config\s+'([^']+)'", script)
    in_script = bool(cfg) and '"disable_any_whitespace":true' in cfg.group(1).replace(" ", "")
    declared = bool(tier3.get("policy", {}).get("disable_any_whitespace"))
    assert in_script == declared, (
        f"drift: serve script disable_any_whitespace={in_script} but models.yaml "
        f"reasoner.policy.disable_any_whitespace={declared}. If you removed the "
        "server flag, remove the catalog declaration too (the proxy must stop "
        "stripping bare json_object); if you added it, declare it."
    )


def test_long_prefill_threshold_present_and_sane(script: str) -> None:
    """Slot fairness: caps per-request prefill tokens per scheduler step so one
    giant prefill cannot starve interactive traffic (45.26 s -> 1.86 s measured)."""
    val = _flag(script, "long-prefill-token-threshold")
    assert val is not None, "--long-prefill-token-threshold missing (slot fairness)"
    batched = _flag(script, "max-num-batched-tokens")
    assert batched is not None
    assert 0 < int(val) < int(batched), (
        f"threshold ({val}) must be >0 and BELOW --max-num-batched-tokens "
        f"({batched}); at or above it the cap never binds and fairness is lost"
    )


def test_patched_stable_abi_extension_is_mounted(script: str) -> None:
    """The tier3 memory leak (~1 GiB/day) is torch 2.11's stable-ABI std::string
    unbox: `ToImpl<std::string>::call` does `new std::string(...)` and returns a
    COPY, never freeing it (pytorch#190493 -- on torch `main` ONLY; 2.11, 2.12 and
    2.13 all still leak). vLLM passes `kv_cache_dtype` as a `str` on every
    `reshape_and_cache_flash`, and FlashInfer + spec-decode forces PIECEWISE
    cudagraphs, so that op dispatches EAGERLY on every engine step.

    The header compiles INTO vllm/_C_stable_libtorch.abi3.so, so a new torch wheel
    would NOT fix it -- the leaking code is inlined in the consumer. The fix is a
    locally rebuilt extension bind-mounted over the image's copy.

    This is guarded because the fix is ONE `-v` line with no other trace: drop it
    and the leak returns silently, at full rate, with a green suite and a healthy
    engine. Measured 2,719-4,176 B/step before, 1,186 B/step after.

    ⚠️ The .so is a HOST artifact (/opt/anvil-bench/patched/) that this repo
    cannot yet reproduce -- the upstream image rebuild is blocked on an Ubuntu
    snapshot repo unreachable inside buildkit. A anvil rebuild drops the fix.
    """
    so = "_C_stable_libtorch.abi3.so"
    assert so in script, (
        f"the patched {so} bind-mount is missing -- the tier3 memory leak "
        "(~1 GiB/day) returns SILENTLY without it: no error, no failed test, "
        "just anvil running out of memory again in ~4 days"
    )
    mount = [ln for ln in script.splitlines() if so in ln]
    assert len(mount) == 1, f"expected exactly one {so} mount, found {len(mount)}"
    line = mount[0]
    assert "/opt/anvil-bench/patched/" in line, (
        "the patched extension must come from /opt/anvil-bench/patched/"
    )
    assert line.rstrip().rstrip("\\").rstrip().endswith(":ro"), (
        "mount the patched extension READ-ONLY (:ro) -- it is a build artifact"
    )


def test_concurrent_partial_prefill_flags_are_absent(script: str) -> None:
    """vLLM's V1 engine raises NotImplementedError: 'Concurrent Partial Prefill
    is not supported' and CRASH-LOOPS. The fields exist on the legacy
    SchedulerConfig, which makes them look available. They are not."""
    for flag in ("max-num-partial-prefills", "max-long-partial-prefills"):
        assert f"--{flag}" not in script, (
            f"--{flag} is rejected by the V1 engine at startup and crash-loops"
        )


def test_vendored_script_matches_catalog(script: str, tier3: dict) -> None:
    """models.yaml seeds the proxy's admission ceiling and context gate. vLLM
    does NOT expose --max-num-seqs over its API, so this seed is the only source
    of truth for the proxy — drift means silent over-admission (a stale 32 once
    over-admitted the thinker by 12)."""
    assert int(_flag(script, "max-num-seqs")) == int(tier3["slots"]), (
        "models.yaml `slots` must equal the script's --max-num-seqs"
    )
    assert int(_flag(script, "max-num-seqs")) == int(
        tier3["policy"]["documented_max_num_seqs"]
    ), "models.yaml `documented_max_num_seqs` must equal the script's --max-num-seqs"
    assert int(_flag(script, "max-model-len")) == int(tier3["context_per_slot"]), (
        "models.yaml `context_per_slot` must equal the script's --max-model-len"
    )
