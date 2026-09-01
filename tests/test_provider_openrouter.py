"""The first REMOTE provider, and the interface it was supposed to prove.

An interface with one kind of implementor has not been tested as an interface —
so this file is as much about `providers/base.py` as about OpenRouter. Every
assumption the local engines let us keep is broken here on purpose, and each one
is a test:

  * reached at a base URL with a PATH, not at `host:port` — so the transport had
    to stop building `http://{host}:{port}` itself;
  * needs a credential, which the provider supplies and the transport never
    learns about;
  * fronts a CATALOGUE, so there is no served model to discover and no weights
    to fingerprint — the poller must not run those probes;
  * publishes prices and no occupancy, which is the asymmetry the descriptor
    exists to state;
  * cannot enforce a GBNF grammar — and REFUSES rather than dropping it.

The end-to-end half runs against `roadstead.testing`'s remote shape on a real
socket, which 401s without a bearer token and serves a two-entry catalogue. No
network: the fake is local, and the one probe the poller would otherwise make is
captured unbound below, exactly as the model-swap guards do.
"""
from __future__ import annotations

import os

import pytest

from roadstead.backend import BackendClientPool, BackendError
from roadstead.config import EndpointConfig
from roadstead.providers import (
    LLAMACPP,
    OPENROUTER,
    ProviderMisconfigured,
    UnsupportedRequest,
    provider_for,
)
from roadstead.testing import FakeBackend, FakeBackendServer

#: The autouse no-network fixture stubs `probe_json` on the CLASS, which is
#: right everywhere else and exactly wrong here — these tests exist to exercise
#: the real GET against a local fake. Captured at import, before any fixture
#: runs, and called unbound (the pattern test_model_swap_guards established).
_REAL_PROBE_JSON = BackendClientPool.probe_json

KEY_ENV = "ROADSTEAD_TEST_OPENROUTER_KEY"
MODEL = "openai/gpt-oss-120b"


def _ep(server: FakeBackendServer | None = None, **kw) -> EndpointConfig:
    base = dict(
        endpoint_class="spill", role="spill", backend_engine="openrouter",
        served_model_id=MODEL, api_key_env=KEY_ENV, max_slots=4,
    )
    base.update(kw)
    if server is not None:
        base["base_url"] = server.base_url
    return EndpointConfig(**base)


@pytest.fixture
def remote():
    """A remote-shaped fake on a real socket, with the key in the environment."""
    server = FakeBackendServer(
        FakeBackend(engine="openrouter", served_model_id=MODEL)).start()
    os.environ[KEY_ENV] = server.controller.api_key
    try:
        yield server
    finally:
        os.environ.pop(KEY_ENV, None)
        server.stop()


# ---------------------------------------------------------------------------
# The connection shape host:port could not express
# ---------------------------------------------------------------------------

def test_base_url_supersedes_host_port():
    assert _ep(base_url="https://openrouter.ai/api/v1").backend_url == (
        "https://openrouter.ai/api/v1")
    # …and a trailing slash does not become a double slash in every path.
    assert _ep(base_url="https://openrouter.ai/api/v1/").backend_url == (
        "https://openrouter.ai/api/v1")


def test_local_endpoints_keep_the_host_port_url():
    """The general form must not disturb the shape every local endpoint uses."""
    ep = EndpointConfig(endpoint_class="chat", role="chat", host="10.0.0.9",
                        port=9083)
    assert ep.backend_url == "http://10.0.0.9:9083"


def test_routes_are_relative_to_the_base_path():
    """`/chat/completions`, not `/v1/chat/completions`: the version lives in the
    base URL here. Getting this wrong is a 404 from a stranger's server."""
    assert OPENROUTER.path_for("chat_completion") == "/chat/completions"
    assert LLAMACPP.path_for("chat_completion") == "/v1/chat/completions"


def test_payload_types_this_provider_has_no_route_for_are_refused():
    """A routing mistake should surface as a refusal here, not as a 404 there."""
    for kind in ("embedding", "rerank"):
        with pytest.raises(UnsupportedRequest):
            OPENROUTER.path_for(kind)


# ---------------------------------------------------------------------------
# Credentials: the provider's business, never the transport's
# ---------------------------------------------------------------------------

def test_auth_header_is_built_from_the_environment(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-or-v1-example")
    headers = OPENROUTER.request_headers(_ep(), "req-1")
    assert headers["Authorization"] == "Bearer sk-or-v1-example"
    assert headers["X-Request-ID"] == "req-1"


def test_a_missing_key_refuses_instead_of_sending_an_unauthenticated_request(
        monkeypatch):
    """A 401 from upstream would be a true statement about the wrong component,
    and it would count against the endpoint's health as though the BACKEND were
    at fault."""
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(ProviderMisconfigured):
        OPENROUTER.request_headers(_ep(), "req-1")
    with pytest.raises(ProviderMisconfigured):
        OPENROUTER.request_headers(_ep(api_key_env=""), "req-1")


def test_the_key_itself_is_never_config(monkeypatch):
    """Only the NAME of the variable is configured — a key in a config file is a
    key in a git history, and this repo is heading for public."""
    monkeypatch.setenv(KEY_ENV, "sk-secret")
    ep = _ep()
    assert "sk-secret" not in repr(ep)
    assert ep.api_key_env == KEY_ENV


# ---------------------------------------------------------------------------
# A constraint it cannot honour is REFUSED, a hint is dropped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"messages": [], "grammar": "root ::= object"},
    {"messages": [], "extra_body": {"grammar": "root ::= object"}},
    {"messages": [], "structured_outputs": {"grammar": "root ::= object"}},
])
def test_a_grammar_is_refused_not_dropped(payload):
    """🚨 The whole argument for `ProviderError` existing.

    Dropping the grammar would return free-form text to a caller that required
    conforming output, and the caller could not distinguish that from a model
    that answered badly — the same shape as the `finish_reason` repair that
    became a silencer (CLAUDE.md). Refusing is loud and correct."""
    with pytest.raises(UnsupportedRequest):
        OPENROUTER.prepare_chat_payload(payload, model_id=MODEL)


def test_engine_hints_are_dropped_silently():
    """These are how we talk to hardware we own — a llama.cpp KV-cache slot, a
    chat-template variable for a template we are not applying, a mirror of a
    vLLM launch flag. No caller asked for them, so silence is correct."""
    out = OPENROUTER.prepare_chat_payload({
        "model": "role-alias",
        "messages": [{"role": "user", "content": "hi"}],
        "id_slot": 3,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_token_budget": 8000,
    }, model_id=MODEL)
    assert "id_slot" not in out
    assert "chat_template_kwargs" not in out
    assert "thinking_token_budget" not in out
    assert out["model"] == MODEL          # it routes on the slug
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_the_shared_repairs_still_apply():
    """A remote provider is still handed our callers' payload shape: a top-level
    `system`, an `extra_body` the OpenAI SDK used to merge, an Anthropic image
    block. Those are OpenAI-shape repairs, not engine ones."""
    out = OPENROUTER.prepare_chat_payload({
        "system": "be brief",
        "messages": [{"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": "AAA"}}]}],
        "extra_body": {"temperature": 0.2},
    }, model_id=MODEL)
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1]["content"][0]["type"] == "image_url"
    assert out["temperature"] == 0.2
    assert "extra_body" not in out and "system" not in out


def test_a_clean_payload_is_returned_unchanged():
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    assert OPENROUTER.prepare_chat_payload(payload, model_id=MODEL) is payload


# ---------------------------------------------------------------------------
# The descriptor: what a remote provider does and does not know
# ---------------------------------------------------------------------------

def test_it_reports_no_occupancy_and_does_not_pretend_otherwise():
    """🚨 Remote capacity is not local capacity. What a remote provider sells is
    money and rate limit, not occupancy — so there are no slots to discover, and
    an endpoint here stays config-capped. Slot-seconds remain the unit of LOCAL
    fairness because local slots are the scarce thing."""
    d = OPENROUTER.descriptor
    assert d.kind == "remote"
    assert d.publishes_slot_count is False
    assert d.publishes_slot_context is False
    assert d.publishes_token_costs is True, (
        "the one fact a remote provider has that no local one does")
    assert d.grammar_field is None
    assert d.publishes_served_model_id is False


def test_catalogue_parsing_picks_our_row_not_the_first_one():
    """A context ceiling read off the wrong row is a confidently wrong number
    feeding the admission context gate."""
    body = {"data": [
        {"id": "someone-else/model", "context_length": 8192},
        {"id": MODEL, "context_length": 131072},
    ]}
    report = OPENROUTER.parse_capacity({"models": body, "model_id": MODEL})
    assert report is not None
    assert report.context_per_slot == 131072
    assert report.slots is None, "there is no slot count to report"


def test_a_model_missing_from_the_catalogue_reports_nothing():
    body = {"data": [{"id": "someone-else/model", "context_length": 8192}]}
    assert OPENROUTER.parse_capacity(
        {"models": body, "model_id": MODEL}) is None
    assert OPENROUTER.parse_capacity({"models": {}, "model_id": MODEL}) is None


# ---------------------------------------------------------------------------
# End to end, on a real socket
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_call_reaches_the_remote_shape_with_its_credential(remote):
    """The composition test: base URL + base path + relative route + auth
    header, all the way through the real transport."""
    pool = BackendClientPool()
    ep = _ep(remote)
    try:
        resp = await pool.call(
            ep, {"messages": [{"role": "user", "content": "ping"}]},
            "chat_completion", "rid-1", timeout_s=10)
    finally:
        await pool.close()
    assert resp.status_code == 200
    rec = remote.controller.requests[-1]
    assert rec.path == "/api/v1/chat/completions"
    assert rec.headers["authorization"] == f"Bearer {remote.controller.api_key}"
    assert rec.body["model"] == MODEL


@pytest.mark.asyncio
async def test_the_wrong_credential_is_rejected_by_the_far_end(remote):
    """Proves the fake actually enforces auth — a test that passes because
    nobody checked would prove nothing about the header above."""
    os.environ[KEY_ENV] = "sk-wrong"
    pool = BackendClientPool()
    try:
        with pytest.raises(BackendError) as exc:
            await pool.call(_ep(remote), {"messages": []},
                            "chat_completion", "rid-2", timeout_s=10)
    finally:
        await pool.close()
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_a_missing_credential_fails_before_a_socket_is_used(remote):
    os.environ.pop(KEY_ENV, None)
    pool = BackendClientPool()
    before = len(remote.controller.requests)
    try:
        with pytest.raises(BackendError) as exc:
            await pool.call(_ep(remote), {"messages": []},
                            "chat_completion", "rid-3", timeout_s=10)
    finally:
        await pool.close()
    assert exc.value.status_code == 400
    assert len(remote.controller.requests) == before, (
        "an endpoint that cannot authenticate must not send the request anyway")


@pytest.mark.asyncio
async def test_discovery_reads_the_catalogue_over_the_wire(remote):
    """`discover_capacity` through the generic `probe_json`, with the provider
    owning the route and the parsing and the pool owning the connection."""
    pool = BackendClientPool()
    ep = _ep(remote)
    remote.controller.catalogue_context_length = 96_000

    async def probe_json(cfg, path, headers=None, timeout_s=5.0):
        return await _REAL_PROBE_JSON(pool, cfg, path, headers, timeout_s)

    pool.probe_json = probe_json
    try:
        report = await OPENROUTER.discover_capacity(pool, ep)
    finally:
        await pool.close()
    assert report is not None
    assert report.context_per_slot == 96_000
    assert report.slots is None
    # …and the prices ride the SAME pass. One probe, two facts: a second round
    # trip to learn what the first response already carried would be a second
    # thing to keep in step with the first.
    assert report.publishes_prices
    assert report.input_usd_per_mtok == pytest.approx(0.5)
    assert report.output_usd_per_mtok == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_a_published_price_reaches_the_price_book_over_the_wire(remote):
    """The end of the wire that Workstream D added: a real socket, a real
    catalogue, and a rate the ledger will bill at.

    🚨 It arrives as REAL money — the descriptor says `publishes_token_costs`,
    so this endpoint is priced by what the provider charges and never by
    `usage_rates.py`'s avoided-cost table. An endpoint that fell through to that
    table would book a SAVING for a call made over the internet.
    """
    from roadstead.config import ProxyConfig
    from roadstead.spend import SOURCE_IMPUTED, SOURCE_PROVIDER
    from roadstead.state import ProxyState

    pool = BackendClientPool()
    ep = _ep(remote)
    remote.controller.catalogue_prompt_cost = "0.000002"     # $2.00 / Mtok
    remote.controller.catalogue_completion_cost = "0.000008"  # $8.00 / Mtok

    async def probe_json(cfg, path, headers=None, timeout_s=5.0):
        return await _REAL_PROBE_JSON(pool, cfg, path, headers, timeout_s)

    pool.probe_json = probe_json
    try:
        report = await OPENROUTER.discover_capacity(pool, ep)
    finally:
        await pool.close()

    state = ProxyState(ProxyConfig())
    assert state.prices.price("spill-chat").source == SOURCE_IMPUTED, (
        "the endpoint was already priced, so this proves nothing")
    from roadstead.health import Health

    Health(state).apply_discovered_capacity("spill-chat", ep, report)
    price = state.prices.price("spill-chat")
    assert price.source == SOURCE_PROVIDER
    assert price.real is True
    assert price.cost_usd(1_000_000, 1_000_000) == pytest.approx(10.0)
    # And it is billed as such, against the caller.
    state.spend.charge("a", "spill-chat", 1_000_000, 1_000_000, now=0.0)
    assert state.spend.get("a").spent_usd == pytest.approx(10.0)
    assert state.spend.get("a").avoided_usd == 0.0


@pytest.mark.asyncio
async def test_discovery_says_cannot_tell_when_the_key_is_missing(remote):
    """None, not an exception: the poller reads it as "cannot tell", the circuit
    breaker opens after its usual consecutive failures, and dispatch stops. That
    beats 502-ing live traffic one call at a time to learn the same thing."""
    os.environ.pop(KEY_ENV, None)
    pool = BackendClientPool()
    try:
        assert await OPENROUTER.discover_capacity(pool, _ep(remote)) is None
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_streaming_goes_through_the_same_seam(remote):
    """The stream path resolves its provider independently of `call`, so it can
    drift independently too. Both faces of the transport, one provider."""
    pool = BackendClientPool()
    events = []
    try:
        async for ev in pool.stream(
                _ep(remote), {"messages": [{"role": "user", "content": "hi"}],
                              "stream": True},
                "chat_completion", "rid-s", timeout_s=10):
            events.append(ev)
    finally:
        await pool.close()
    assert events and events[-1].event_type == "done"
    rec = remote.controller.requests[-1]
    assert rec.path == "/api/v1/chat/completions"
    assert rec.headers["authorization"] == f"Bearer {remote.controller.api_key}"


@pytest.mark.asyncio
async def test_a_refused_request_fails_the_stream_before_it_opens(remote):
    """A refusal raised inside an async generator has to surface on the first
    iteration, not vanish — the stream path's failure mode is different enough
    from `call`'s to be worth its own test."""
    pool = BackendClientPool()
    before = len(remote.controller.requests)
    agen = pool.stream(
        _ep(remote), {"messages": [], "grammar": "root ::= object"},
        "chat_completion", "rid-g", timeout_s=10)
    try:
        with pytest.raises(BackendError) as exc:
            await agen.__anext__()
    finally:
        await pool.close()
    assert exc.value.status_code == 400
    assert "grammar" in exc.value.detail
    assert len(remote.controller.requests) == before


def test_the_registry_resolves_it():
    assert provider_for(_ep()) is OPENROUTER


# ---------------------------------------------------------------------------
# It has to be CONFIGURABLE, or it is a provider nobody can reach
# ---------------------------------------------------------------------------

def test_a_remote_stanza_reaches_endpoint_config(tmp_path):
    """🚨 `build_endpoint_kwargs` copies a fixed set of keys, and a key it does
    not know is SILENTLY DROPPED — the failure mode models.yaml warns about in
    its own comments. A remote endpoint missing any of these is a llama.cpp
    endpoint pointed at nothing.

    It also pins the section split: the CONNECTION comes from the provider and
    the POLICY from the endpoint, so an endpoint stanza never repeats a base URL
    or a key name."""
    from roadstead import model_catalog

    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    engine: openrouter\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    api_key_env: OPENROUTER_API_KEY\n"
        "endpoints:\n"
        "  spill:\n"
        "    provider: openrouter\n"
        "    kind: chat\n"
        f"    model: {MODEL}\n"
        "    slots: 4\n",
        encoding="utf-8")
    cat = model_catalog.load_catalog(yaml_path, force=True)
    kw = model_catalog.build_endpoint_kwargs(cat)["spill"]
    ep = EndpointConfig(**kw)

    assert provider_for(ep) is OPENROUTER
    assert ep.backend_url == "https://openrouter.ai/api/v1"
    assert ep.api_key_env == "OPENROUTER_API_KEY"
    assert ep.effective_model_id == MODEL, (
        "a remote endpoint's model is seeded from config — its provider "
        "declares there is nothing to discover")
    assert ep.max_slots == 4, (
        "a POLICY cap on our own concurrency, not a discovered capacity")


def test_one_remote_provider_serves_many_endpoints(tmp_path):
    """The reason providers and endpoints are separate sections. A local
    provider is one server serving one model; a remote one fronts a catalogue,
    so its credential and base URL are declared ONCE and each endpoint adds only
    which model it routes on."""
    from roadstead import model_catalog

    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "providers:\n"
        "  openrouter:\n"
        "    engine: openrouter\n"
        "    base_url: https://openrouter.ai/api/v1\n"
        "    api_key_env: OPENROUTER_API_KEY\n"
        "endpoints:\n"
        "  spill-fast:\n"
        "    provider: openrouter\n"
        "    model: vendor/small\n"
        "    slots: 8\n"
        "  spill-deep:\n"
        "    provider: openrouter\n"
        "    model: vendor/large\n"
        "    slots: 2\n",
        encoding="utf-8")
    kw = model_catalog.build_endpoint_kwargs(
        model_catalog.load_catalog(yaml_path, force=True))
    fast, deep = EndpointConfig(**kw["spill-fast"]), EndpointConfig(**kw["spill-deep"])
    assert fast.backend_url == deep.backend_url == "https://openrouter.ai/api/v1"
    assert fast.api_key_env == deep.api_key_env == "OPENROUTER_API_KEY"
    assert fast.effective_model_id == "vendor/small"
    assert deep.effective_model_id == "vendor/large"
    assert (fast.max_slots, deep.max_slots) == (8, 2)


def test_planned_endpoints_are_documented_but_not_routed(tmp_path):
    """The shipped catalog carries remote endpoints nobody has a credential for.
    They must document the shape without entering the routing table — an
    endpoint that cannot serve is worse than no endpoint."""
    from roadstead import model_catalog

    cat = model_catalog.load_catalog(force=True)
    planned = [e.name for e in cat.endpoints.values() if not e.routed]
    assert planned, "the example lost its planned remote endpoints"
    kw = model_catalog.build_endpoint_kwargs(cat)
    for name in planned:
        assert name not in kw, f"{name} is planned but was routed anyway"


def test_a_local_stanza_is_unaffected_by_the_generalised_engine_mirror(tmp_path):
    """The negative control. `shim` and a decorated `llama.cpp (Vulkan)` still
    resolve to the default provider and are still not mirrored, so nothing about
    a local endpoint moved when the mirror stopped reading `== "vllm"`. The
    address still arrives, now from the provider section."""
    from roadstead import model_catalog

    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "hosts: {box: 192.0.2.1}\n"
        "providers:\n"
        "  local:\n"
        "    engine: llama.cpp (Vulkan)\n"
        "    host: box\n"
        "    port: 9000\n"
        "endpoints:\n"
        "  chat:\n"
        "    provider: local\n"
        "    slots: 4\n",
        encoding="utf-8")
    kw = model_catalog.build_endpoint_kwargs(
        model_catalog.load_catalog(yaml_path, force=True))["chat"]
    assert "backend_engine" not in kw
    assert "base_url" not in kw and "api_key_env" not in kw
    ep = EndpointConfig(**kw)
    assert provider_for(ep) is LLAMACPP
    assert ep.backend_url == "http://192.0.2.1:9000"
