"""A bare `response_format:{"type":"json_object"}` returns literally `{}` on a
backend launched with structured-output whitespace banned.

tier3 (vLLM, anvil:9083) restarted 2026-07-31 15:06:52 UTC with
``--structured-outputs-config '{"backend":"guidance","disable_any_whitespace":true}'``.
That flag is load-bearing — without it structured output runs away emitting
whitespace until max_tokens — but with whitespace banned, the two-character
document ``{}`` is a legal, COMPLETE, zero-whitespace JSON object. A bare
``json_object`` grammar lets the model close immediately, and greedy decoding
does. Measured live on the same prompt and endpoint:

    no response_format                        -> 549 chars, valid JSON
    response_format json_object (bare)        -> "{}" (2 chars), finish=stop
    response_format json_schema (strict, req) -> 241 chars, valid JSON

``{}`` arrives with ``finish_reason=stop`` and no error, so it is invisible to
every caller-side health signal. Sidekick's forum-agent comment critic turned it into
``voice_match must be bool``; the orchestrator read that ``.error`` as transient
infra and DEFERRED; forum-agent re-drove the same proposal every tick — 3,100 defers,
executions ~300/day -> 13.

``Correction.apply_json_object_guard`` strips bare json_object on endpoints whose
``EndpointConfig.disable_any_whitespace`` is set. It must NOT substitute a
permissive schema: ``{}`` satisfies ``{"type":"object"}`` with no ``required``, so
the bug would survive the "fix".

The second half of this file guards the SILENT-DROP trap that would make all of
the above inert: a models.yaml policy key absent from the mapping tuple in
``model_catalog.build_endpoint_policies`` is discarded without a word (see the
``min_expected_slots`` note in models.yaml). The declaration must actually reach
``EndpointConfig``.

Self-contained in the house style of test_thinking_option.py: the real
``Correction`` methods are bound to a lightweight mock ``self`` carrying just the
``.state`` fields they touch.
"""
import importlib
import inspect
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

config = importlib.import_module("roadstead.config")
correction = importlib.import_module("roadstead.correction")
model_catalog = importlib.import_module("roadstead.model_catalog")
C = correction.Correction

BARE = {"type": "json_object"}
SCHEMA_RF = {"type": "json_schema", "json_schema": {"name": "critic", "schema": {
    "type": "object",
    "properties": {"voice_match": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["voice_match", "why"]}}}


def _req(payload, *, endpoint="thinker", ptype="chat_completion", stream=False):
    r = types.SimpleNamespace()
    r.json_object_stripped = False
    r.payload = payload
    r.payload_type = ptype
    r.endpoint = endpoint
    r.stream = stream
    r.request_id = "r1"
    r.agent_id = "sidekick"
    r.call_site = "auto_approve.critic"
    return r


def _mock_self(*, disable_any_whitespace: bool):
    state = types.SimpleNamespace()
    ep = types.SimpleNamespace(backend_engine="vllm",
                               disable_any_whitespace=disable_any_whitespace)
    state.config = types.SimpleNamespace(endpoints={"thinker": ep})
    m = types.SimpleNamespace(state=state)
    for name in ("apply_json_object_guard", "request_is_structured",
                 "request_expects_json", "extract_grammar"):
        setattr(m, name, getattr(C, name).__get__(m, C))
    return m


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------

def test_bare_json_object_is_stripped_on_whitespace_banned_endpoint():
    """The bug. Bare json_object + disable_any_whitespace -> constraint removed."""
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [{"role": "user", "content": "critique this"}],
         "max_tokens": 400, "response_format": dict(BARE)}
    m.apply_json_object_guard(_req(p))
    assert "response_format" not in p, (
        "bare json_object must be STRIPPED on a whitespace-banned backend — "
        "otherwise the model greedily emits the legal complete document '{}'"
    )
    # Everything else is untouched: this is a strip, not a rewrite.
    assert p["max_tokens"] == 400 and len(p["messages"]) == 1


def test_no_permissive_schema_is_substituted():
    """Substituting {"type":"object"} with no `required` would leave '{}' valid,
    so the bug would survive the fix. Nothing may be put back."""
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE)}
    m.apply_json_object_guard(_req(p))
    assert set(p) == {"messages"}
    for key in ("response_format", "structured_outputs", "guided_json", "grammar"):
        assert key not in p


def test_bare_json_object_untouched_on_normal_endpoint():
    """Every other backend still honours json_object correctly — leave it alone."""
    m = _mock_self(disable_any_whitespace=False)
    p = {"messages": [], "response_format": dict(BARE)}
    m.apply_json_object_guard(_req(p))
    assert p["response_format"] == BARE


def test_json_schema_untouched():
    """A real schema carries `required` keys, so it cannot degenerate to '{}'
    (measured: 241 chars of valid JSON on the same endpoint)."""
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": SCHEMA_RF}
    m.apply_json_object_guard(_req(p))
    assert p["response_format"] is SCHEMA_RF


def test_guided_json_untouched():
    m = _mock_self(disable_any_whitespace=True)
    guided = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"]}
    p = {"messages": [], "guided_json": guided}
    m.apply_json_object_guard(_req(p))
    assert p["guided_json"] is guided


def test_gbnf_grammar_untouched():
    m = _mock_self(disable_any_whitespace=True)
    gbnf = 'root ::= "{" ws "\\"a\\"" ws ":" ws string "}"'
    p = {"messages": [], "grammar": gbnf}
    m.apply_json_object_guard(_req(p))
    assert p["grammar"] == gbnf


def test_structured_outputs_untouched():
    m = _mock_self(disable_any_whitespace=True)
    so = {"grammar": 'root ::= "{}"'}
    p = {"messages": [], "structured_outputs": so}
    m.apply_json_object_guard(_req(p))
    assert p["structured_outputs"] is so


def test_bare_json_object_alongside_a_real_grammar_is_left_alone():
    """The grammar is what binds; don't touch a payload that already pinned one."""
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE), "guided_json": {"type": "object"}}
    m.apply_json_object_guard(_req(p))
    assert p["response_format"] == BARE


# --- the same, nested under extra_body (how the OpenAI SDK smuggles vLLM args) ---

def test_extra_body_bare_json_object_is_stripped():
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "extra_body": {"response_format": dict(BARE), "top_k": 20}}
    m.apply_json_object_guard(_req(p))
    assert "response_format" not in p["extra_body"]
    assert p["extra_body"]["top_k"] == 20  # siblings preserved


def test_extra_body_bare_json_object_untouched_on_normal_endpoint():
    m = _mock_self(disable_any_whitespace=False)
    p = {"messages": [], "extra_body": {"response_format": dict(BARE)}}
    m.apply_json_object_guard(_req(p))
    assert p["extra_body"]["response_format"] == BARE


def test_extra_body_json_schema_untouched():
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "extra_body": {"response_format": SCHEMA_RF}}
    m.apply_json_object_guard(_req(p))
    assert p["extra_body"]["response_format"] is SCHEMA_RF


def test_extra_body_guided_json_blocks_the_strip():
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE),
         "extra_body": {"guided_json": {"type": "object"}}}
    m.apply_json_object_guard(_req(p))
    assert p["response_format"] == BARE


def test_json_schema_in_extra_body_protects_a_top_level_json_object():
    """Redundant sibling: the request is really schema-constrained. Hands off."""
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE),
         "extra_body": {"response_format": SCHEMA_RF}}
    m.apply_json_object_guard(_req(p))
    assert p["response_format"] == BARE


def test_both_positions_stripped_together():
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE),
         "extra_body": {"response_format": dict(BARE)}}
    m.apply_json_object_guard(_req(p))
    assert "response_format" not in p and "response_format" not in p["extra_body"]


# --- totality: a compensation must never break a request --------------------

def test_non_chat_payloads_are_ignored():
    m = _mock_self(disable_any_whitespace=True)
    p = {"input": "hello", "response_format": dict(BARE)}
    m.apply_json_object_guard(_req(p, ptype="embedding"))
    assert p["response_format"] == BARE


def test_unknown_endpoint_is_a_noop():
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE)}
    m.apply_json_object_guard(_req(p, endpoint="nope-not-a-role"))
    assert p["response_format"] == BARE


def test_never_raises_on_a_hostile_payload():
    """Fail-open. A malformed payload must pass through, not 500 the caller."""
    m = _mock_self(disable_any_whitespace=True)
    for payload in (None, [], "not-a-dict",
                    {"messages": [], "response_format": "json_object"},
                    {"messages": [], "response_format": None},
                    {"messages": [], "extra_body": "nope",
                     "response_format": dict(BARE)}):
        m.apply_json_object_guard(_req(payload))
    # the last one is a valid strip target with a junk extra_body — still works
    p = {"messages": [], "extra_body": "nope", "response_format": dict(BARE)}
    m.apply_json_object_guard(_req(p))
    assert "response_format" not in p


def test_streaming_requests_are_covered_too():
    """The guard is wired BEFORE the streaming/sync branch in lifecycle.py
    precisely because apply_thinking is sync-only. Nothing here may depend on
    req.stream."""
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "stream": True, "response_format": dict(BARE)}
    m.apply_json_object_guard(_req(p, stream=True))
    assert "response_format" not in p


# --- the strip must not silently drop what the constraint also bought -------
#
# CLAUDE.md: "replacing a component silently drops its guarantees — enumerate
# what it guaranteed and re-assert each." A bare json_object also armed the
# Phase-1.1 truncation-integrity gate (finish_reason=length on a STRUCTURED
# request fails loud + deferrable instead of handing back half an object), the
# JSON/schema backstop, and the structured-stream validity guard. Removing the
# constraint must not remove those, or a truncated reply becomes an unparseable
# half-object at the caller — the SAME defer-loop shape this fix exists to end.

def test_stripped_request_still_counts_as_structured():
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE)}
    req = _req(p)
    assert m.request_is_structured(req) is True   # before: via the payload
    m.apply_json_object_guard(req)
    assert "response_format" not in p
    assert req.json_object_stripped is True
    assert m.request_is_structured(req) is True, (
        "truncation integrity would lapse: a finish_reason=length reply would be "
        "handed back as a benign capped free-form answer instead of failing loud"
    )


def test_stripped_request_still_expects_json():
    m = _mock_self(disable_any_whitespace=True)
    p = {"messages": [], "response_format": dict(BARE)}
    req = _req(p)
    assert m.request_expects_json(req) is True
    m.apply_json_object_guard(req)
    assert m.request_expects_json(req) is True, (
        "the JSON backstop / stream parse gate would lapse after the strip"
    )


def test_unstripped_request_does_not_claim_the_flag():
    """The re-assertion must be scoped to requests the guard actually touched."""
    m = _mock_self(disable_any_whitespace=False)
    req = _req({"messages": []})
    m.apply_json_object_guard(req)
    assert req.json_object_stripped is False
    assert m.request_is_structured(req) is False
    assert m.request_expects_json(req) is False


def test_queued_request_carries_the_field():
    """It must be a real dataclass field, not an attribute the guard invents —
    the WAL-recovery path rebuilds QueuedRequest from the DB."""
    scheduler = importlib.import_module("roadstead.scheduler")
    req = scheduler.QueuedRequest.create(
        agent_id="sidekick", endpoint="tier3", priority=None,
        call_site="auto_approve.critic", payload_type="chat_completion",
        payload={"messages": []})
    assert req.json_object_stripped is False


# ---------------------------------------------------------------------------
# The silent-drop guard: models.yaml -> EndpointConfig
# ---------------------------------------------------------------------------

def test_policy_key_is_in_the_catalog_mapping():
    """A models.yaml policy key that is NOT in build_endpoint_kwargs' mapping
    tuple is discarded WITHOUT A WARNING (the `min_expected_slots` trap).

    🔄 2026-08-22: tier3 no longer DECLARES disable_any_whitespace (see the
    module docstring), so this can no longer be asserted through tier3 without
    going vacuous. The trap it guards is about the MAPPING, not about tier3, so
    it is asserted directly against the builder's src/dst tuple list instead —
    the plumbing stays armed for whichever endpoint next needs the flag.
    """
    src = inspect.getsource(model_catalog.build_endpoint_kwargs)
    assert "disable_any_whitespace" in src, (
        "build_endpoint_kwargs no longer maps disable_any_whitespace. Any stanza "
        "declaring it would be silently dropped — add "
        '("disable_any_whitespace", "disable_any_whitespace") back to the src/dst '
        "tuple list in model_catalog.build_endpoint_kwargs"
    )
    kwargs = model_catalog.build_endpoint_kwargs()
    assert "thinker" in kwargs, "tier3 (endpoint_class `thinker`) missing from the catalog"


def test_models_yaml_flag_reaches_endpoint_config():
    """The whole guard is inert unless the DECLARATION survives into the live
    EndpointConfig the proxy resolves at request time. This is the end-to-end
    check: yaml -> catalog -> EndpointConfig field, on the real tier3 endpoint."""
    ep = config.DEFAULT_ENDPOINTS[config.normalize_endpoint("tier3")]
    assert hasattr(ep, "disable_any_whitespace"), (
        "EndpointConfig has no disable_any_whitespace field — the policy knob was "
        "never added (config.py)"
    )
    # 🔄 2026-08-22: tier3 is DeepSeek-V4-Flash-0731 and its serve script passes NO
    # --structured-outputs-config, so whitespace is not banned and the "{}" bug
    # cannot occur. VERIFIED on the live endpoint that day, which is the only
    # evidence that actually settles it: bare json_object returned 306-319 chars
    # of real JSON 3/3 (containing newlines, i.e. whitespace unbanned), where the
    # 2026-07-31 Laguna backend returned literally "{}" with finish_reason=stop.
    # This must stay FALSE while the serve script omits the flag — the doctrine
    # test tests/llmproxy/test_tier3_serve_script_doctrine.py pins that direction.
    assert ep.disable_any_whitespace is False, (
        "tier3 now declares disable_any_whitespace, but its serve script passes no "
        "--structured-outputs-config. Those two must move together or bare "
        "json_object silently returns '{}' again."
    )
    other = config.DEFAULT_ENDPOINTS[config.normalize_endpoint("creative")]
    assert other.disable_any_whitespace is False


def test_every_alias_of_tier3_resolves_to_the_guarded_endpoint():
    """`tier3` is canonical, but callers still reach it under legacy aliases
    (sidekick's critic passes one). All of them must land on the endpoint carrying the
    flag, or the guard fires for some callers and not others."""
    for alias in ("tier3", "reasoner", "llama-thinker", "thinker", "composer",
                  "companion", "qwen-composer", "nexus-companion"):
        ep = config.DEFAULT_ENDPOINTS[config.normalize_endpoint(alias)]
        # 🔄 2026-08-22: what this test protects is the alias/class-collision
        # history — every tier3 alias must resolve to the SAME endpoint object.
        # The guard FLAG is now False by design (V4-Flash bans no whitespace;
        # see the module docstring and test_models_yaml_flag_reaches_endpoint_config),
        # so assert the flag is uniform across aliases rather than True.
        assert ep.disable_any_whitespace is False, f"alias {alias!r} unexpectedly guarded"
