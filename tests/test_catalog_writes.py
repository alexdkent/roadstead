"""Roadmap J2 — creating, editing and deleting catalog entries at runtime.

🚨 The design claim these tests exist to hold: **the overlay contributes catalog
STANZAS, not a second model of an endpoint.** A runtime fragment is merged into
`models.yaml`'s raw dict *before coercion*, so it is parsed, defaulted and
validated by exactly the code that parses a file-authored stanza. If that ever
becomes two formats, every rule below has to be written twice and one copy will
drift.

The other claim: `models.yaml` is never written. It keeps its comments, its
hand-authored intent, and the property that a bad save cannot destroy it.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from roadstead import model_catalog
from roadstead.config import ProxyConfig
from roadstead.management import reconcile_endpoints
from roadstead.service import ProxyService

from .test_management_plane import _Req, _body, _svc

_CATALOG = pathlib.Path(model_catalog._DEFAULT_PATH)


@pytest.fixture(autouse=True)
def _clean_overlay():
    model_catalog.set_runtime_overlay(None)
    yield
    model_catalog.set_runtime_overlay(None)


async def _put(svc, section, name, body, method="PUT"):
    key = "provider" if section == "providers" else "endpoint"
    return await svc.handle_admin_catalog_entry(
        _Req(method=method, path_params={key: name}, body=body))


# --------------------------------------------------------------------------- #
# The merge, at the level it actually happens
# --------------------------------------------------------------------------- #

def test_a_runtime_stanza_is_parsed_by_the_file_parser():
    """🚨 The load-bearing one. A created endpoint must come out of the catalog
    indistinguishable from a declared one — same defaults, same capability
    coercion, same types — because it went through the same function."""
    cat = model_catalog.load_catalog(overlay={"endpoints": {"made-up": {
        "provider": "small-box", "kind": "chat", "status": "active",
        "slots": 3, "context_per_slot": 4096,
        "capabilities": {"streaming": True, "vision": False}}}})
    made = cat.endpoints["made-up"]
    declared = cat.endpoints["tier1"]
    assert type(made) is type(declared)
    assert made.routed is True and made.slots == 3
    # Coerced by the same code: a false capability is dropped, not kept as False.
    assert made.capabilities.get("streaming") is True
    assert not made.capabilities.get("vision")
    # And it is reachable by name through the same index the file's entries use.
    assert cat.entry("made-up") is made


def test_partial_merge_create_and_tombstone():
    base = model_catalog.load_catalog()
    cat = model_catalog.load_catalog(overlay={"endpoints": {
        "tier1": {"slots": 99},          # partial: merged over the file's
        "tier3": None,                   # tombstone
    }})
    assert cat.endpoints["tier1"].slots == 99
    assert base.endpoints["tier1"].slots != 99, "fixture stale"
    # 🚨 The rest of the file's stanza SURVIVES a partial — an edit to one field
    # that silently blanked the others is the loss this shape exists to avoid.
    assert cat.endpoints["tier1"].capabilities == base.endpoints["tier1"].capabilities
    assert "tier3" not in cat.endpoints


def test_a_candidate_catalog_is_never_cached_or_served():
    """A validation build must not leak into the process. It is, by definition,
    a catalog nobody has agreed to run."""
    before = sorted(model_catalog.load_catalog().endpoints)
    model_catalog.load_catalog(overlay={"endpoints": {"ghost": {
        "provider": "small-box", "kind": "chat", "slots": 1}}})
    assert sorted(model_catalog.load_catalog().endpoints) == before


def test_the_runtime_overlay_is_seen_by_every_caller():
    """🚨 Global rather than a threaded parameter, because nine sites call
    `load_catalog()` and a missed one would describe the fleet as the FILE has
    it while every other surface describes it as it IS."""
    model_catalog.set_runtime_overlay({"endpoints": {"ghost": {
        "provider": "small-box", "kind": "chat", "status": "active",
        "slots": 1, "context_per_slot": 2048, "timeout_floor_s": 42}}})
    assert "ghost" in model_catalog.load_catalog().endpoints
    assert "ghost" in model_catalog.build_endpoint_kwargs()
    # The class maps are read on the hot path and are built from the catalog —
    # a runtime endpoint's floor has to reach them or its deadline is wrong.
    assert model_catalog.build_class_floors()["ghost"] == 42.0


# --------------------------------------------------------------------------- #
# Through the plane
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_create_edit_and_delete_reach_the_routing_table(tmp_path):
    svc = _svc(tmp_path)
    before = _CATALOG.read_bytes()

    r = await _put(svc, "providers", "invented", {
        "engine": "llama.cpp", "host": "192.0.2.77", "port": 9099})
    assert r.status_code == 200, r.body

    r = await _put(svc, "endpoints", "scratch", {
        "provider": "invented", "kind": "chat", "status": "active",
        "slots": 3, "context_per_slot": 8192,
        "capabilities": {"streaming": True}})
    assert r.status_code == 200, r.body
    assert json.loads(r.body)["reconciled"]["added"] == ["scratch"]
    assert "scratch" in svc._state.config.endpoints
    assert svc._state.config.endpoints["scratch"].max_slots == 3

    # 🚨 An EDIT must reach the table. Reconcile leaves existing entries alone so
    # discovery's corrections survive, so an edit that named nothing would be
    # accepted and do nothing — the worst kind of success.
    r = await _put(svc, "endpoints", "scratch", {"slots": 7}, method="PATCH")
    assert r.status_code == 200, r.body
    assert svc._state.config.endpoints["scratch"].max_slots == 7
    assert json.loads(r.body)["reconciled"]["rebuilt"] == ["scratch"]
    # …and the merge kept what the PATCH did not mention.
    assert "streaming" in svc._state.config.endpoints["scratch"].capabilities

    r = await svc.handle_admin_catalog_entry(
        _Req(method="DELETE", path_params={"endpoint": "scratch"}))
    assert r.status_code == 200, r.body
    assert "scratch" not in svc._state.config.endpoints

    assert _CATALOG.read_bytes() == before, "models.yaml was written"


@pytest.mark.asyncio
async def test_the_write_is_validated_by_building_the_catalog(tmp_path):
    """🚨 No second validator. A stanza the file loader would COMPLAIN about is
    a 400 here — same check, different consequence, chosen by who is asking:
    a typo must not stop a fleet booting, but an operator at a keyboard can fix
    it now."""
    svc = _svc(tmp_path)
    r = await _put(svc, "endpoints", "bad", {
        "provider": "small-box", "kind": "chat", "status": "active", "slots": 1,
        "capabilities": {"streaming": True},
        "policy": {"not_a_real_knob": 3}})
    assert r.status_code == 400, r.body
    assert "not_a_real_knob" in json.loads(r.body)["error"]
    assert "bad" not in svc._state.config.endpoints


@pytest.mark.asyncio
@pytest.mark.parametrize("section,name,body,fragment", [
    ("endpoints", "x", {"slotz": 2}, "unknown field"),
    ("providers", "p", {"api_key": "sk-secret"}, "unknown field"),
])
async def test_unknown_fields_are_refused_never_dropped(tmp_path, section, name,
                                                        body, fragment):
    """The surface whose purpose is exposing silent drops must not have one."""
    svc = _svc(tmp_path)
    r = await _put(svc, section, name, body)
    assert r.status_code == 400
    assert fragment in json.loads(r.body)["error"]


@pytest.mark.asyncio
async def test_a_provider_still_in_use_cannot_be_deleted(tmp_path):
    """Named, not merely refused — 'which ones' is the next question."""
    svc = _svc(tmp_path)
    r = await svc.handle_admin_catalog_entry(
        _Req(method="DELETE", path_params={"provider": "small-box"}))
    assert r.status_code == 400
    assert "tier1" in json.loads(r.body)["error"]


@pytest.mark.asyncio
async def test_an_endpoint_with_work_in_flight_cannot_be_deleted(tmp_path, monkeypatch):
    """J1's rule, on the delete path too: the request path reads the endpoint's
    config AFTER dispatch, so removing it under live work is a null deref."""
    svc = _svc(tmp_path)
    monkeypatch.setattr(svc._state.scheduler, "endpoint_snapshot",
                        lambda n: {"in_flight": 2, "max_slots": 4, "queued": 0})
    r = await svc.handle_admin_catalog_entry(
        _Req(method="DELETE", path_params={"endpoint": "tier1"}))
    assert r.status_code == 400
    assert "in flight" in json.loads(r.body)["error"]
    assert "tier1" in svc._state.config.endpoints


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["../etc", "a b", "", "x" * 65, "-lead"])
async def test_unusable_names_are_refused(tmp_path, name):
    """A catalog name is a dict key, a URL segment, a DRR budget key and a
    metrics label. Bounded and boring on purpose."""
    svc = _svc(tmp_path)
    r = await _put(svc, "endpoints", name, {"provider": "small-box"})
    assert r.status_code == 400, f"{name!r} was accepted"


# --------------------------------------------------------------------------- #
# The table is replaced, not mutated
# --------------------------------------------------------------------------- #

def test_reconcile_replaces_the_routing_table_rather_than_mutating_it():
    """🚨 Found by running it: the capacity poller iterates `config.endpoints`
    and awaits between items, so a reconcile landing mid-loop raised
    `RuntimeError: dictionary changed size during iteration`. Rebinding the
    attribute lets an in-flight iteration finish over the table it started with
    — and is the alternative to auditing 27 call sites for an `await` and being
    wrong about one of them.
    """
    config = ProxyConfig(queue_db_path=":memory:")
    config.endpoints = dict(config.endpoints)
    held = config.endpoints                      # what a live iterator holds

    model_catalog.set_runtime_overlay({"endpoints": {"newcomer": {
        "provider": "small-box", "kind": "chat", "status": "active",
        "slots": 1, "context_per_slot": 2048}}})
    reconcile_endpoints(config, names={"newcomer"})

    assert "newcomer" in config.endpoints
    assert "newcomer" not in held, (
        "the reconcile mutated the dict an iterator was holding — that is the "
        "RuntimeError this replacement exists to prevent")
    assert config.endpoints is not held


@pytest.mark.asyncio
async def test_a_created_endpoint_survives_a_restart(tmp_path):
    store = tmp_path / "admin_overlay.json"
    svc = _svc(tmp_path)
    await _put(svc, "providers", "invented",
               {"engine": "llama.cpp", "host": "192.0.2.77", "port": 9099})
    await _put(svc, "endpoints", "scratch", {
        "provider": "invented", "kind": "chat", "status": "active",
        "slots": 2, "context_per_slot": 4096})
    saved = json.loads(store.read_text())
    assert saved["catalog"]["endpoints"]["scratch"]["slots"] == 2
    assert saved["catalog"]["providers"]["invented"]["host"] == "192.0.2.77"

    again = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q2.db"),
                                     admin_store_path=str(store)))
    assert "scratch" in again._state.config.endpoints
    assert again._state.config.endpoints["scratch"].max_slots == 2


def test_the_legacy_endpoints_section_is_migrated_not_dropped(tmp_path):
    """🚨 The pre-J2 file had `endpoints:` at the top level for status
    overrides. Dropped, an operator upgrades and silently finds every promoted
    endpoint out of service — the failure this store exists to prevent."""
    from roadstead.management import AdminOverlay
    store = tmp_path / "old.json"
    store.write_text(json.dumps({"version": 1,
                                 "endpoints": {"spill-chat": {"status": "active"}}}))
    overlay = AdminOverlay(store)
    assert overlay.catalog["endpoints"]["spill-chat"] == {"status": "active"}


@pytest.mark.asyncio
async def test_startup_does_not_delete_an_endpoint_configured_in_CODE(tmp_path):
    """🚨 Found by a spill test going red, and it is the more important half of
    this workstream's blast radius.

    An earlier reconcile imposed the whole catalog at startup — everything it
    routes in, everything else out. The shipped catalog declares `spill-chat` as
    `planned`, a caller had added a routed one to `config.endpoints` directly,
    and startup deleted it. Embedding Roadstead and configuring endpoints in
    code is a supported arrangement; a reconcile that assumes the catalog is the
    only possible author of the routing table breaks it silently.

    So reconcile touches only the names its caller says changed.
    """
    import copy

    store = tmp_path / "admin_overlay.json"
    config = ProxyConfig(queue_db_path=str(tmp_path / "q.db"),
                         admin_store_path=str(store))
    # A routed endpoint the catalog does NOT route (it declares it `planned`)…
    hand_built = copy.deepcopy(config.endpoints["tier1"])
    hand_built.endpoint_class = hand_built.role = "spill-chat"
    config.endpoints["spill-chat"] = hand_built
    # …and one the catalog has never heard of at all.
    bespoke = copy.deepcopy(config.endpoints["tier1"])
    bespoke.endpoint_class = bespoke.role = "bespoke"
    config.endpoints["bespoke"] = bespoke

    svc = ProxyService(config)

    assert "spill-chat" in svc._state.config.endpoints, (
        "startup deleted an endpoint the caller configured in code, because the "
        "catalog declares that name as planned")
    assert "bespoke" in svc._state.config.endpoints, (
        "startup deleted an endpoint the catalog has never heard of")
