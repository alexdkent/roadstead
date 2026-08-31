"""Two standing guards against a model swap silently invalidating this stanza's
model-dependent policy — the 2026-08-23 tier3 cutover, which nothing noticed for
a day (ledger `tier3-reasoning-parser-default-mismatch`).

  1. `model_fingerprint_drift`  — the WEIGHTS changed under a declaration.
  2. `thinking_switch_broken`   — the declared thinking switch stopped switching,
                                  for any reason, including a template change
                                  under the SAME weights.

They are complementary and neither subsumes the other: (1) is inference from
config, cheap, and fires the moment weights change; (2) is EVIDENCE — the one
real call the declaration never had behind it.

🚨 EVERY TEST HERE HAS A NEGATIVE CONTROL. The defect being guarded shipped
because fifteen gates checked that something EXISTED. A guard that cannot be
observed going red is not a guard, so each "stays quiet" test is paired with a
"fires" test built from the ACTUAL failure, not an imagined one.

The live probe values below were measured against the running backends
2026-08-24 (both the passing and the failing arms):

  thinker      root=/srv/models/deepseek-v4-flash-0731
               key `thinking`        -> 193 chars reasoning   (declared: PASS)
  creative     params=27320697856;vocab=248320;ftype=Q4_0
               key `enable_thinking` -> 349 chars reasoning   (declared: PASS)
               key `thinking`        ->   0 chars reasoning   (wrong key: FIRES)
  tier2-chat   params=34660610688;vocab=248320;ftype=Q4_K - Medium
               key `enable_thinking` -> 835 chars reasoning   (declared: PASS)
               key `thinking`        ->   0 chars reasoning   (wrong key: FIRES)
"""
import asyncio
import importlib
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

backend = importlib.import_module("roadstead.backend")
config = importlib.import_module("roadstead.config")
health = importlib.import_module("roadstead.health")
model_catalog = importlib.import_module("roadstead.model_catalog")


# ---------------------------------------------------------------------------
# Fingerprint EXTRACTION — the parsing, against real captured payloads.
# ---------------------------------------------------------------------------

# The autouse `_no_network_probes` fixture stubs these on the CLASS, which is
# exactly what we want everywhere else and exactly wrong here: these tests
# exist to exercise the REAL parsing. Capture the real functions at import
# time — before any fixture runs — and call them unbound against a pool whose
# `_client_for` we control.
_REAL_FINGERPRINT = backend.BackendClientPool.probe_model_fingerprint
_REAL_THINKING = backend.BackendClientPool.probe_thinking_switch


class _FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def _pool_returning(payload, status=200):
    pool = backend.BackendClientPool()
    client = types.SimpleNamespace()

    async def _get(_path):
        return _FakeResp(payload, status)

    client.get = _get
    pool._client_for = lambda *a, **k: client   # noqa: SLF001
    return pool


def _ep(**kw):
    base = dict(endpoint_class="x", role="x", host="h", port=1)
    base.update(kw)
    return config.EndpointConfig(**base)


# Captured verbatim from the live backends 2026-08-24.
VLLM_MODELS = {"data": [{"id": "llama-thinker", "object": "model",
                         "root": "/srv/models/deepseek-v4-flash-0731",
                         "max_model_len": 1048576}]}
LLAMACPP_MODELS = {"data": [{"id": "creative", "object": "model",
                             "meta": {"vocab_type": 2, "n_vocab": 248320,
                                      "n_ctx": 262144, "n_embd": 5120,
                                      "n_params": 27320697856,
                                      "size": 16045481984, "ftype": "Q4_0"}}]}
# tier1's build genuinely omits ftype — absent fields must be dropped, not faked.
LLAMACPP_NO_FTYPE = {"data": [{"id": "gemma", "meta": {"n_vocab": 262144,
                                                       "n_params": 7518069290}}]}


def test_vllm_fingerprint_is_the_weights_path_not_the_served_alias():
    """THE WHOLE POINT. `id` is `llama-thinker` and stayed that way across the
    cutover; `root` is what actually moved."""
    pool = _pool_returning(VLLM_MODELS)
    got = asyncio.run(_REAL_FINGERPRINT(pool, _ep()))
    assert got == "/srv/models/deepseek-v4-flash-0731"
    assert "llama-thinker" not in got, (
        "keying the fingerprint on the served alias would make this guard "
        "incapable of ever firing — that alias is exactly what did NOT change")


def test_llamacpp_fingerprint_folds_the_meta_block():
    pool = _pool_returning(LLAMACPP_MODELS)
    got = asyncio.run(_REAL_FINGERPRINT(pool, _ep()))
    assert got == "params=27320697856;vocab=248320;ftype=Q4_0"


def test_llamacpp_fingerprint_omits_fields_the_build_does_not_report():
    pool = _pool_returning(LLAMACPP_NO_FTYPE)
    got = asyncio.run(_REAL_FINGERPRINT(pool, _ep()))
    assert got == "params=7518069290;vocab=262144"


def test_fingerprint_is_none_when_the_backend_cannot_tell():
    """None must mean "cannot tell". If an unreachable backend produced a
    fingerprint of "" the drift alert would compare "" against a declaration
    and scream on every restart — an alert that cries wolf gets muted, and a
    muted alert is worse than no alert."""
    for payload, status in ((VLLM_MODELS, 500), ({"data": []}, 200),
                            ({"data": [{"id": "x"}]}, 200)):
        pool = _pool_returning(payload, status)
        assert asyncio.run(_REAL_FINGERPRINT(pool, _ep())) is None


# ---------------------------------------------------------------------------
# The DRIFT alert — quiet when it should be, loud when it must be.
# ---------------------------------------------------------------------------

def _alerts_for(endpoints):
    """Call the REAL alert builder. `model_swap_alerts` is a pure function
    precisely so this test does not have to mock a scheduler + budget manager +
    metrics registry + queue DB just to reach two comparisons — and so it cannot
    quietly become a re-implementation that passes while production drifts."""
    return [a.name for a in health.model_swap_alerts(endpoints)]


def test_drift_alert_fires_on_the_real_2026_08_23_swap():
    """Declared Laguna, serving DeepSeek — the exact live pair measured above."""
    ep = _ep(model_fingerprint="/srv/models/laguna-s-2.1",
             discovered_model_fingerprint="/srv/models/deepseek-v4-flash-0731")
    assert "model_fingerprint_drift" in _alerts_for({"thinker": ep})


def test_drift_alert_is_silent_when_they_agree():
    ep = _ep(model_fingerprint="/srv/models/deepseek-v4-flash-0731",
             discovered_model_fingerprint="/srv/models/deepseek-v4-flash-0731")
    assert _alerts_for({"thinker": ep}) == []


def test_drift_alert_is_silent_when_either_side_is_unknown():
    """Undeclared, or unreachable. Both are "cannot tell" and must stay quiet —
    otherwise every endpoint without a declaration alerts forever."""
    assert _alerts_for({"a": _ep(discovered_model_fingerprint="x")}) == []
    assert _alerts_for({"b": _ep(model_fingerprint="x")}) == []


# ---------------------------------------------------------------------------
# The CANARY — the evidence half.
# ---------------------------------------------------------------------------

def test_canary_alert_fires_when_the_declared_key_switches_nothing():
    """The measured negative control: sending DeepSeek's `thinking` to a Qwen
    endpoint returned 0 chars of reasoning on BOTH tier2 backends. A stanza
    declaring that key is the 2026-08-23 defect, and this is what catches it."""
    ep = _ep(thinking_canary_state=(
        "sent chat_template_kwargs {thinking: true} and got 0 chars of "
        "reasoning (139 chars of content) — the declared switch is not "
        "switching anything on the model this endpoint is serving now"))
    assert "thinking_switch_broken" in _alerts_for({"creative": ep})


def test_canary_alert_is_silent_when_ok_or_unknown():
    assert _alerts_for({"a": _ep(thinking_canary_state="ok")}) == []
    assert _alerts_for({"b": _ep(thinking_canary_state="")}) == []


def test_canary_counts_reasoning_not_content():
    """tier2-chat's real probe came back with 835 chars of reasoning and ZERO
    content — reasoning ate the 200-token budget. That is a PASS: the switch
    worked, which is the only thing being asked. Judging on content would
    invert this into a false alarm, and judging via `BackendClient.call` would
    have raised `empty_completion_error` on it — which is why the probe
    deliberately does not go through `call`."""
    for r, ok in (({"reasoning_chars": 835, "content_chars": 0}, True),
                  ({"reasoning_chars": 0, "content_chars": 139}, False)):
        assert (r["reasoning_chars"] > 0) is ok


def test_canary_probe_nonces_the_prompt():
    """A fixed probe payload is answered by the prefix cache, and a cached
    answer proves nothing about today's template — a probe that verifies the
    CACHE keeps passing after the thing it watches has broken."""
    sent = {}
    pool = backend.BackendClientPool()
    client = types.SimpleNamespace()

    async def _post(_path, json=None):
        sent.update(json or {})
        return _FakeResp({"choices": [{"message": {"reasoning": "r", "content": "c"}}]})

    client.post = _post
    pool._client_for = lambda *a, **k: client   # noqa: SLF001
    asyncio.run(_REAL_THINKING(pool, _ep(), "thinking", "NONCE-XYZ"))
    assert "NONCE-XYZ" in sent["messages"][0]["content"]
    assert sent["chat_template_kwargs"] == {"thinking": True}


def test_canary_probe_returns_none_rather_than_guessing():
    """Non-200 / malformed → None → the poller records "" → the alert stays
    silent. Silence must never be reported as breakage."""
    pool = backend.BackendClientPool()
    client = types.SimpleNamespace()

    async def _post(_path, json=None):
        return _FakeResp({"choices": []}, 500)

    client.post = _post
    pool._client_for = lambda *a, **k: client   # noqa: SLF001
    assert asyncio.run(_REAL_THINKING(pool, _ep(), "thinking", "n")) is None


# ---------------------------------------------------------------------------
# The declarations are really wired, and really match the live fleet.
# ---------------------------------------------------------------------------

def test_declared_fingerprints_reach_config():
    kw = model_catalog.build_endpoint_kwargs()
    assert kw["thinker"]["model_fingerprint"] == "/srv/models/deepseek-v4-flash-0731"
    assert kw["creative"]["model_fingerprint"] == \
        "params=27320697856;vocab=248320;ftype=Q4_0"
    ep = config.EndpointConfig(**kw["thinker"])
    assert ep.model_fingerprint and not ep.discovered_model_fingerprint


def test_every_endpoint_declaring_a_thinking_switch_also_declares_a_fingerprint():
    """The two guards are a PAIR. A stanza that says which key switches
    reasoning, without saying which weights that claim is about, has the half
    that cannot notice it went stale."""
    for cls, kw in model_catalog.build_endpoint_kwargs().items():
        if kw.get("thinking_kwargs"):
            assert kw.get("model_fingerprint"), cls


def test_every_backend_probe_is_stubbed_in_the_unit_suite():
    """The unit suite must not touch the network, and the fixture that enforces
    that stubs probes BY NAME — so the name list must be complete.

    🚨 THIS TEST EXISTS BECAUSE THE LIST FAILED OPEN. Adding
    `probe_model_fingerprint` and `probe_thinking_switch` to the pool without
    adding them here made the poller dial real fleet hosts from `pytest`, and
    `probe_thinking_switch` would have run a real GENERATION against live tier3
    from a unit test. Nothing failed — the tests passed and only teardown timed
    out (1.3s -> 61-181s on the e2e suite), which is precisely why a guard that
    goes stale silently is worse than no guard. Now a new probe fails HERE, on
    the day it is written, with a message saying what to do.
    """
    from tests.conftest import STUBBED_PROBES

    # Deliberately NOT stubbed, and this predates 2026-08-24 — the coverage
    # test found them, it did not create them. Both are driven only by tests
    # that supply their own fake client to exercise the real parsing, and
    # stubbing them breaks exactly those tests. Listed here so the gap is a
    # RECORDED decision rather than an omission that looks identical to one.
    # If either ever starts being called from the poller on a real host, stub
    # it and fix those tests instead.
    deliberately_unstubbed = {"probe_prefix_cache", "probe_progress_counters"}

    actual = {n for n in dir(backend.BackendClientPool)
              if n.startswith("probe_")
              and callable(getattr(backend.BackendClientPool, n, None))}
    missing = actual - set(STUBBED_PROBES) - deliberately_unstubbed
    assert not missing, (
        f"BackendClientPool grew {sorted(missing)} but tests/llmproxy/conftest.py "
        f"STUBBED_PROBES does not cover it. Unstubbed, that probe makes REAL "
        f"network calls to fleet hosts from the unit suite. Add it to "
        f"STUBBED_PROBES with the return value that means 'cannot tell'.")
    stale = (set(STUBBED_PROBES) | deliberately_unstubbed) - actual
    assert not stale, (
        f"STUBBED_PROBES lists {sorted(stale)}, which no longer exists on "
        f"BackendClientPool — drop it so the list keeps meaning something.")


def test_canary_interval_is_slow_enough_to_be_affordable():
    """It costs a real generation. A per-poll cadence (10s) would add ~8,600
    generations/day across the fleet to catch a defect that lasted a day."""
    assert config.thinking_canary_interval_s() >= 600


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok  {name}")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print("FAILED" if failed else "all passed")
    sys.exit(1 if failed else 0)
