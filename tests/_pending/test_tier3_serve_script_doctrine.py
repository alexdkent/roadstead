"""tier3's serve scripts are load-bearing and were UNGUARDED.

`models.yaml` documents several tier3 flags as load-bearing and cited
`tier3_gates.py` as the enforcement — but that file lives only on anvil at
`/opt/anvil-bench/` and is not in this repo, so neither CI nor `ship.sh`
could run it. Nothing caught drift in the flags whose regressions are SILENT.

2026-08-22 cutover: tier3 moved off the single-node Laguna script
(`serve_tier3_prod.sh`, KEPT as the rollback until Phase 6) onto a two-node
DeepSeek-V4-Flash-0731 vLLM TP=2 pair — `tier3_env.sh` (shared env + engine
args) sourced by `serve_tier3_v4flash_head.sh` (anvil, rank 0, :9083) and
`serve_tier3_v4flash_worker.sh` (anvil2, rank 1, --headless). This file now
pins the NEW pair against models.yaml; the Laguna-only guards (the structured-
outputs whitespace flag, the long-prefill threshold, the patched stable-ABI
mount) either moved with the invariant they protect or were retired below with
the reasoning kept in a comment, never left asserting on a file the new
scripts don't have.

The vendored copies are the reviewable source of truth; the hosts run their
own copies at /opt/tier3/, so when you change one, change both.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
VLLM_DIR = REPO / "infra" / "anvil" / "vllm"
LAGUNA_SCRIPT = VLLM_DIR / "serve_tier3_prod.sh"
ENV_SCRIPT = VLLM_DIR / "v4flash" / "tier3_env.sh"
HEAD_SCRIPT = VLLM_DIR / "v4flash" / "serve_tier3_v4flash_head.sh"
WORKER_SCRIPT = VLLM_DIR / "v4flash" / "serve_tier3_v4flash_worker.sh"
CATALOG = REPO / "originfleet" / "llmproxy" / "models.yaml"


@pytest.fixture(scope="module")
def env_script() -> str:
    assert ENV_SCRIPT.is_file(), f"vendored tier3_env.sh missing at {ENV_SCRIPT}"
    return ENV_SCRIPT.read_text()


@pytest.fixture(scope="module")
def head_script() -> str:
    assert HEAD_SCRIPT.is_file(), f"vendored head launcher missing at {HEAD_SCRIPT}"
    return HEAD_SCRIPT.read_text()


@pytest.fixture(scope="module")
def worker_script() -> str:
    assert WORKER_SCRIPT.is_file(), f"vendored worker launcher missing at {WORKER_SCRIPT}"
    return WORKER_SCRIPT.read_text()


@pytest.fixture(scope="module")
def laguna_script() -> str:
    assert LAGUNA_SCRIPT.is_file(), f"Laguna rollback script missing at {LAGUNA_SCRIPT}"
    return LAGUNA_SCRIPT.read_text()


@pytest.fixture(scope="module")
def tier3() -> dict:
    cat = yaml.safe_load(CATALOG.read_text())
    models = cat["models"] if "models" in cat else cat
    # tier3 is the `reasoner` stanza (role llama-thinker).
    stanza = models.get("reasoner")
    assert stanza is not None, "reasoner (tier3) stanza not found in models.yaml"
    return stanza


def _flag(script: str, name: str) -> str | None:
    """Value following `--name` in a bash-array element, or None. Matches both
    a plain `--name value` docker-run line and a `--name value` array element,
    since COMMON_ARGS in tier3_env.sh is a bash array, not a single line."""
    m = re.search(rf"--{re.escape(name)}\s+(\S+)", script)
    return m.group(1) if m else None


def test_no_comments_inside_the_docker_run_continuation(head_script: str, worker_script: str) -> None:
    """A '#' line inside a trailing-backslash command is NOT a comment — it
    swallows every flag after it. This happened TWICE on 2026-07-31 against the
    Laguna script: the first time it silently dropped enable_thinking=false, the
    tool/reasoning parsers and the structured-outputs config; tier3 came up
    "healthy" with five flags missing. Only an explicit `docker inspect` of the
    running args caught it.

    Scoped to the two thin launchers, NOT tier3_env.sh: that file's
    COMMON_ARGS/ENVFLAGS are bash ARRAYS (`NAME=(...)`), where a `#` line is a
    real comment and does not swallow anything — only a `docker run ... \\`
    continuation has this hazard."""
    for name, script in (("head", head_script), ("worker", worker_script)):
        body = re.search(r"^exec docker run.*", script, re.S | re.M)
        assert body, f"could not locate the docker run block in the {name} script"
        offenders = [ln for ln in body.group(0).splitlines() if ln.lstrip().startswith("#")]
        assert not offenders, (
            f"{name} script: comment line(s) inside the docker run continuation "
            f"will swallow every flag after them: {offenders}"
        )


def test_enable_thinking_is_false(head_script: str, worker_script: str) -> None:
    """False returns EMPTY content on every caller when thinking leaks. The
    template-kwargs KEY changed at this cutover: DeepSeek-V4-Flash's chat
    template reads ``thinking``, not Laguna's ``enable_thinking`` — same
    intent (thinking off by default), new form. Only the HEAD carries it — the
    worker is --headless and serves no chat template at all."""
    assert '"thinking":false' in head_script.replace(" ", ""), (
        "thinking must be false — true makes tier3 spend the whole token "
        "budget reasoning and return nothing, with no error anywhere"
    )
    assert '"thinking":true' not in head_script.replace(" ", "")
    assert "--default-chat-template-kwargs" not in worker_script, (
        "the headless worker serves no chat completions and needs no "
        "chat-template-kwargs flag"
    )


def test_structured_outputs_config_is_absent(env_script: str) -> None:
    """The new script passes NO --structured-outputs-config at all — verified
    2026-08-22: 10/10 well-formed structured outputs without it. Laguna carried
    ``disable_any_whitespace: true`` as the fix for a whitespace runaway on its
    grammar backend; that backend/config combination is not in play here."""
    assert "--structured-outputs-config" not in env_script, (
        "the new tier3_env.sh must not declare --structured-outputs-config — "
        "verified clean without it; adding one back needs its own re-verification, "
        "not a copy of the Laguna flag"
    )


def test_disable_any_whitespace_is_declared_in_the_catalog(env_script: str, tier3: dict) -> None:
    """BIDIRECTIONAL pin, now in the OPPOSITE direction from the Laguna era.

    Ledger `disable_any_whitespace_is_declared_in_the_catalog`: banning
    whitespace made a bare ``json_object`` grammar legal to close immediately,
    returning ``"{}"`` 10/10 — that was the reason the flag existed at all on
    Laguna. The new script drops the flag entirely (see
    ``test_structured_outputs_config_is_absent``), so the catalog's
    ``policy.disable_any_whitespace`` declaration must be dropped with it, or
    ``Correction.apply_json_object_guard`` keeps stripping a bare
    ``response_format:{"type":"json_object"}`` against a backend that no
    longer bans whitespace — the proxy would strip a constraint the backend
    would have honoured, for no reason.
    """
    in_script = "--structured-outputs-config" in env_script
    declared = bool(tier3.get("policy", {}).get("disable_any_whitespace"))
    assert in_script == declared, (
        f"drift: tier3_env.sh declares --structured-outputs-config={in_script} "
        f"but models.yaml reasoner.policy.disable_any_whitespace={declared}. "
        "The new pair runs with neither — both must be absent together."
    )
    assert not declared, (
        "reasoner.policy.disable_any_whitespace must be REMOVED for the "
        "DeepSeek-V4-Flash pair: the new script passes no "
        "--structured-outputs-config, so the catalog declaration is stale"
    )


def test_long_prefill_threshold_not_applicable_to_v4flash(env_script: str) -> None:
    """Laguna-only guard, retired for the new pair. ``--long-prefill-token-
    threshold`` capped per-request prefill tokens per scheduler step so one
    giant prefill couldn't starve interactive traffic on Laguna's scheduler
    (45.26s -> 1.86s measured there). The verified DeepSeek-V4-Flash config
    (tier3_env.sh COMMON_ARGS) does not carry it — chunked prefill +
    async-scheduling are the fairness mechanism on this engine build instead.
    Not re-scoped to the Laguna script because that guard already exists
    below via ``test_long_prefill_threshold_present_on_laguna_rollback``."""
    assert "--long-prefill-token-threshold" not in env_script, (
        "tier3_env.sh should not carry --long-prefill-token-threshold — it "
        "was a Laguna scheduler-fairness flag; if it's back, this test needs "
        "re-scoping, not deletion"
    )


def test_long_prefill_threshold_present_on_laguna_rollback(laguna_script: str) -> None:
    """The Laguna script is the rollback until Phase 6 and must still carry
    the slot-fairness flag that made it safe to run (45.26s -> 1.86s
    measured)."""
    val = _flag(laguna_script, "long-prefill-token-threshold")
    assert val is not None, "--long-prefill-token-threshold missing (slot fairness)"
    batched = _flag(laguna_script, "max-num-batched-tokens")
    assert batched is not None
    assert 0 < int(val) < int(batched), (
        f"threshold ({val}) must be >0 and BELOW --max-num-batched-tokens "
        f"({batched}); at or above it the cap never binds and fairness is lost"
    )


def test_patched_stable_abi_mount_is_laguna_only(env_script: str) -> None:
    """The patched ``_C_stable_libtorch.abi3.so`` bind-mount was a Laguna-image
    leak fix (torch 2.11's stable-ABI std::string unbox, pytorch#190493) tied
    to that image's build. It is not part of the verified DeepSeek-V4-Flash
    config and must not silently reappear via a copy-paste of DOCKERFLAGS."""
    so = "_C_stable_libtorch.abi3.so"
    assert so not in env_script, (
        f"tier3_env.sh must not mount the patched {so} — that fix was scoped "
        "to the retired Laguna image; carrying it forward unexamined is a "
        "leftover, not a guard"
    )


def test_patched_stable_abi_extension_is_mounted_on_laguna_rollback(laguna_script: str) -> None:
    """The tier3 memory leak (~1 GiB/day) is torch 2.11's stable-ABI std::string
    unbox: `ToImpl<std::string>::call` does `new std::string(...)` and returns a
    COPY, never freeing it (pytorch#190493 -- on torch `main` ONLY; 2.11, 2.12 and
    2.13 all still leak). The Laguna image needs the bind-mounted fix; guarded
    here because the fix is ONE `-v` line with no other trace and dropping it
    returns the leak silently."""
    so = "_C_stable_libtorch.abi3.so"
    assert so in laguna_script, (
        f"the patched {so} bind-mount is missing from the Laguna rollback script "
        "-- the tier3 memory leak (~1 GiB/day) returns SILENTLY without it"
    )
    mount = [ln for ln in laguna_script.splitlines() if so in ln]
    assert len(mount) == 1, f"expected exactly one {so} mount, found {len(mount)}"
    line = mount[0]
    assert "/opt/anvil-bench/patched/" in line, (
        "the patched extension must come from /opt/anvil-bench/patched/"
    )
    assert line.rstrip().rstrip("\\").rstrip().endswith(":ro"), (
        "mount the patched extension READ-ONLY (:ro) -- it is a build artifact"
    )


def test_concurrent_partial_prefill_flags_are_absent(env_script: str) -> None:
    """vLLM's V1 engine raises NotImplementedError: 'Concurrent Partial Prefill
    is not supported' and CRASH-LOOPS. The fields exist on the legacy
    SchedulerConfig, which makes them look available. They are not."""
    for flag in ("max-num-partial-prefills", "max-long-partial-prefills"):
        assert f"--{flag}" not in env_script, (
            f"--{flag} is rejected by the V1 engine at startup and crash-loops"
        )


def test_vendored_script_matches_catalog(env_script: str, tier3: dict) -> None:
    """models.yaml seeds the proxy's admission ceiling and context gate. vLLM
    does NOT expose --max-num-seqs over its API, so this seed is the only source
    of truth for the proxy — drift means silent over-admission (a stale 32 once
    over-admitted the thinker by 12). Checked against tier3_env.sh's
    COMMON_ARGS, the shared source both nodes launch from."""
    assert int(_flag(env_script, "max-num-seqs")) == int(tier3["slots"]), (
        "models.yaml `slots` must equal tier3_env.sh's --max-num-seqs"
    )
    assert int(_flag(env_script, "max-num-seqs")) == int(
        tier3["policy"]["documented_max_num_seqs"]
    ), "models.yaml `documented_max_num_seqs` must equal tier3_env.sh's --max-num-seqs"
    assert int(_flag(env_script, "max-model-len")) == int(tier3["context_per_slot"]), (
        "models.yaml `context_per_slot` must equal tier3_env.sh's --max-model-len"
    )


def test_both_nodes_source_the_same_env_and_split_head_from_worker(
    head_script: str, worker_script: str
) -> None:
    """The pair must launch from ONE shared env file — if the two nodes drift
    onto separate copies of COMMON_ARGS/ENVFLAGS, TP=2 either fails to form or
    silently disagrees on engine args between ranks. The head is the only one
    that serves traffic; the worker is headless."""
    for name, script in (("head", head_script), ("worker", worker_script)):
        assert "source" in script and "tier3_env.sh" in script, (
            f"{name} script must source tier3_env.sh (the shared engine-args source "
            "of truth) rather than declaring its own COMMON_ARGS/ENVFLAGS"
        )
    assert re.search(r"--port\s+9083", head_script), "head (rank 0) must serve :9083"
    assert "--headless" in worker_script, "worker (rank 1) must be --headless"
    assert "--headless" not in head_script, "head must NOT be headless -- it serves traffic"
    assert not re.search(r"--port\s+\d+", worker_script), (
        "worker is headless and must not bind a serving port"
    )


def test_image_is_pinned_by_digest(env_script: str) -> None:
    """A `:latest` or other mutable tag lets a host silently pull a different
    build on the next `docker run` -- exactly the class of drift the module
    docstring's nightly-build history warns about (a soak-tested nightly is
    not fungible with whatever `:latest` resolves to tomorrow)."""
    m = re.search(r'^IMAGE="([^"]+)"', env_script, re.M)
    assert m, "IMAGE= assignment not found in tier3_env.sh"
    image = m.group(1)
    assert "@sha256:" in image, (
        f"IMAGE must be pinned by digest (@sha256:...), not a tag: {image!r}"
    )


def test_tier3_declares_its_second_host(tier3: dict) -> None:
    """tier3 is the fleet's first model spanning two boxes, and `host:` is a SCALAR.

    Without `co_hosts` the Inference page renders `▸ anvil` and stops — it keeps
    saying `anvil · healthy` while anvil2 (rank 1, --headless, no port of its
    own) is DEAD. Not blank, not an error: a confident half-truth, which is the
    worst shape an operator page can take.

    Pinned here rather than in the frontend because the CATALOG is the authority —
    if this declaration is dropped, every surface downstream silently reverts to
    the half-truth with nothing failing.
    """
    assert tier3.get("host") == "anvil", (
        "tier3's PRIMARY host must stay `anvil` — it is rank 0, the box that "
        "serves port 9083, and host->IP resolution, telemetry units and port "
        "probing all key off it."
    )
    co = tier3.get("co_hosts") or []
    assert "anvil2" in co, (
        "tier3 serves vLLM TP=2 across anvil + anvil2; the second rank must be "
        "declared in `co_hosts` or the Inference page claims tier3 runs on one box."
    )


def test_co_hosts_survives_into_the_catalog(tier3: dict) -> None:
    """A models.yaml key absent from the parser is discarded WITHOUT A WARNING —
    the `min_expected_slots` / `disable_any_whitespace` trap, twice bitten. Assert
    the declaration actually reaches ModelEntry rather than only existing in YAML."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from roadstead import model_catalog

    entry = model_catalog.load_catalog().entry("tier3")
    assert entry is not None, "tier3 did not resolve in the catalog"
    assert "anvil2" in entry.co_hosts, (
        "models.yaml declares co_hosts: [anvil2] but ModelEntry.co_hosts is "
        f"{entry.co_hosts!r} — the key was dropped by the parser in "
        "model_catalog (add it to the ModelEntry construction)."
    )


def _flag_in_docker_run(script: str, flag: str) -> bool:
    """Is `flag` actually PASSED, as opposed to merely mentioned?

    🚨 Written after the naive `flag in script` version was sabotage-tested and did
    NOT fail: this file's own explanatory comments name `--reasoning-config` twice, so
    the substring check was satisfied by prose and would have reported the flag present
    after it was deleted from the command. A guard anything can satisfy is not a guard.

    So: look only inside the `exec docker run` continuation, and only at lines that are
    not comments.
    """
    body = re.search(r"^exec docker run.*", script, re.S | re.M)
    if not body:
        return False
    return any(flag in ln for ln in body.group(0).splitlines()
               if not ln.lstrip().startswith("#"))


def test_thinking_budget_ratio_matches_the_script(head_script: str, tier3: dict) -> None:
    """BIDIRECTIONAL pin: `--reasoning-config` in the launcher ⟺ a declared ratio.

    vLLM honours the per-request `thinking_token_budget` ONLY when the server was
    started with `--reasoning-config`; without it the server refuses the whole request
    ("thinking_token_budget is set but reasoning_config is not configured"). So the two
    must move together in both directions:

      * flag present, declaration missing -> the proxy never caps reasoning, and this
        model's natural length is BIMODAL: measured 2026-08-23, it sometimes ran past a
        12,000-token ceiling and returned `content: ""` after 8-13 minutes;
      * declaration present, flag removed -> the proxy injects a parameter the backend
        rejects, and EVERY thinking request to tier3 400s. That direction does not
        degrade the endpoint, it breaks it.

    ⚠️ `--reasoning-parser` alone does NOT satisfy this. vLLM's auto-init only stamps
    the parser name INTO an existing reasoning_config object
    (`EngineArgs._set_default_reasoning_config_args` returns early when it is None), so
    the parser flag can be present while the whole path stays disabled — which is
    exactly the state this endpoint was in until 2026-08-23.
    """
    in_script = _flag_in_docker_run(head_script, "--reasoning-config")
    declared = float(tier3.get("policy", {}).get("thinking_budget_ratio") or 0.0)
    assert in_script == bool(declared), (
        f"drift: head launcher passes --reasoning-config={in_script} but "
        f"models.yaml reasoner.policy.thinking_budget_ratio={declared}. "
        "vLLM rejects the request outright when the flag is missing — these must "
        "change together."
    )
    if declared:
        assert 0 < declared < 1, f"ratio must be a fraction of max_tokens, got {declared}"
        # A cap that leaves no room for the answer recreates the failure it prevents.
        assert declared <= 0.8, (
            f"ratio {declared} leaves under 20% of max_tokens for the ANSWER; the "
            "whole point of the cap is answer headroom"
        )


def test_reasoning_parser_is_still_present_for_the_budget_to_work(head_script: str) -> None:
    """The budget needs BOTH flags. `--reasoning-config '{}'` self-derives its
    delimiters from the parser, so removing `--reasoning-parser` while keeping the
    config would leave the config unable to initialize its token ids and the cap
    silently inert."""
    if _flag_in_docker_run(head_script, "--reasoning-config"):
        assert _flag_in_docker_run(head_script, "--reasoning-parser"), (
            "--reasoning-config without --reasoning-parser: the config cannot derive "
            "reasoning_start_str/reasoning_end_str and the budget never applies"
        )
