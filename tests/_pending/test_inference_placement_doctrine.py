"""Doctrine test — the inference PLACEMENT charter.

``llmproxy/models.yaml`` already says WHERE every model runs. ``meta.vehicles``
says what each host is FOR, and what its spare capacity is already committed to.
This test makes that charter enforceable so a placement decision cannot be made
by eye ("box X looks underutilized") and land silently:

  * every host in ``meta.hosts`` is claimed by exactly one vehicle, and every
    vehicle names only hosts that exist;
  * every vehicle declares a complete charter (purpose, what it admits, what its
    headroom is for, what it forbids);
  * every LIVE entry (``status`` active / on_demand) sits on a host whose vehicle
    admits its ``kind`` at that residency — so an always-on LLM on nexus, or a
    second resident model on the boxa, fails here;
  * a vehicle with ``exclusive_roles`` holds EXACTLY those roles — this is what
    pins anvil + anvil2 as a single-purpose tier3 appliance (D6);
  * ``meta.idle_silicon`` stays populated, so the deliberately-unused
    accelerators keep their recorded reason instead of being rediscovered.

Fix a failure by editing ``meta.vehicles`` DELIBERATELY (widening a charter is an
operator decision, and the ``charter`` line should say why), not by loosening it
to make the run green. Placement narrative: ``system_architecture_overview.md``
§9.3; skills: ``model-selection``, ``inference-substrate``.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]                       # .../OriginFleet
_MODELS_YAML = _REPO_ROOT / "originfleet" / "originfleet" / "llmproxy" / "models.yaml"

#: Statuses that describe a model actually placed on hardware today.
_LIVE_STATUSES = frozenset({"active", "on_demand"})

_REQUIRED_VEHICLE_FIELDS = ("hosts", "charter", "admits", "headroom_is", "forbids")


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(_MODELS_YAML.read_text())


@pytest.fixture(scope="module")
def vehicles(raw) -> dict:
    return raw["meta"]["vehicles"]


def _live_models(raw) -> dict[str, dict]:
    return {
        name: entry
        for name, entry in (raw.get("models") or {}).items()
        if isinstance(entry, dict) and entry.get("status") in _LIVE_STATUSES
    }


def _vehicle_for_host(vehicles: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for vname, v in vehicles.items():
        for host in v.get("hosts") or ():
            out[host] = vname
    return out


def _admission_violations(vehicles: dict, models: dict[str, dict]) -> list[str]:
    """Every live entry must be admitted by its host's vehicle charter.

    Factored out so ``test_the_guard_can_actually_fail`` can feed it a planted
    violation — a green run must mean the rule held, not that nothing was checked.
    """
    by_host = _vehicle_for_host(vehicles)
    problems: list[str] = []
    for name, entry in models.items():
        host = entry.get("host")
        kind = entry.get("kind")
        status = entry.get("status")
        vname = by_host.get(host)
        if vname is None:
            problems.append(
                f"{name!r} is placed on host {host!r}, which no vehicle in "
                f"meta.vehicles claims. Declare the host's charter before placing "
                f"work on it."
            )
            continue
        v = vehicles[vname]
        admits = v.get("admits") or {}
        if kind not in admits:
            problems.append(
                f"{name!r} (kind={kind!r}) is placed on {host!r} — vehicle "
                f"{vname!r}, which admits only {sorted(admits)}.\n"
                f"    charter: {str(v.get('charter', '')).strip()}\n"
                f"    headroom_is: {str(v.get('headroom_is', '')).strip()}\n"
                f"    forbids: {str(v.get('forbids', '')).strip()}"
            )
            continue
        if status not in (admits.get(kind) or ()):
            problems.append(
                f"{name!r} is {status!r} on {host!r}, but vehicle {vname!r} admits "
                f"kind {kind!r} only at residency {sorted(admits[kind])}.\n"
                f"    headroom_is: {str(v.get('headroom_is', '')).strip()}"
            )
    return problems


def test_every_host_is_claimed_by_exactly_one_vehicle(raw, vehicles):
    hosts = set(raw["meta"]["hosts"])
    claimed: dict[str, list[str]] = {}
    for vname, v in vehicles.items():
        for host in v.get("hosts") or ():
            claimed.setdefault(host, []).append(vname)

    unknown = {h: vs for h, vs in claimed.items() if h not in hosts}
    assert not unknown, (
        f"meta.vehicles names hosts that are not in meta.hosts: {unknown}"
    )
    doubled = {h: vs for h, vs in claimed.items() if len(vs) > 1}
    assert not doubled, f"a host may belong to exactly one vehicle: {doubled}"
    unclaimed = hosts - set(claimed)
    assert not unclaimed, (
        f"hosts with no vehicle charter: {sorted(unclaimed)}. Every inference host "
        f"must declare what it is FOR before anything is placed on it."
    )


def test_every_vehicle_declares_a_complete_charter(vehicles):
    incomplete: dict[str, list[str]] = {}
    for vname, v in vehicles.items():
        missing = [
            f
            for f in _REQUIRED_VEHICLE_FIELDS
            if not str(v.get(f) or "").strip() and not v.get(f)
        ]
        if missing:
            incomplete[vname] = missing
    assert not incomplete, (
        f"vehicles missing required charter fields {list(_REQUIRED_VEHICLE_FIELDS)}: "
        f"{incomplete}. `headroom_is` and `forbids` are the load-bearing ones — an "
        f"empty one reads as 'this box has spare capacity', which is the exact claim "
        f"this charter exists to refute."
    )


def test_every_live_model_is_admitted_by_its_vehicles_charter(raw, vehicles):
    problems = _admission_violations(vehicles, _live_models(raw))
    assert not problems, "placement charter violations:\n\n" + "\n\n".join(problems)


def test_the_guard_can_actually_fail(raw, vehicles):
    """Plant an always-on LLM on nexus and assert the checker rejects it."""
    planted = {
        "some_new_llm": {
            "kind": "chat",
            "status": "active",
            "host": "nexus",
        }
    }
    problems = _admission_violations(vehicles, planted)
    assert problems, (
        "the admission check accepted an always-on chat model on nexus — the "
        "charter has been loosened to the point of being vacuous"
    )
    assert "nexus" in problems[0] and "forbids" in problems[0]

    # ...and a host nobody claims is caught too.
    orphan = {"x": {"kind": "chat", "status": "active", "host": "not-a-host"}}
    assert _admission_violations(vehicles, orphan)


def test_exclusive_vehicles_hold_exactly_their_declared_roles(raw, vehicles):
    live = _live_models(raw)
    problems: list[str] = []
    for vname, v in vehicles.items():
        declared = v.get("exclusive_roles")
        if declared is None:
            continue
        hosts = set(v.get("hosts") or ())
        actual = {name for name, e in live.items() if e.get("host") in hosts}
        if actual != set(declared):
            problems.append(
                f"vehicle {vname!r} is exclusive to {sorted(declared)} but hosts "
                f"{sorted(actual)} are live on {sorted(hosts)}.\n"
                f"    forbids: {str(v.get('forbids', '')).strip()}"
            )
    assert not problems, (
        "closed-appliance violations:\n\n" + "\n\n".join(problems)
    )


def test_tier3_pair_is_a_closed_appliance(vehicles):
    """D6 (2026-08-20) in its own named test, because it is the rule most likely
    to be argued with: anvil + anvil2 carry tier3 and nothing else."""
    tier3 = vehicles["tier3"]
    assert set(tier3["hosts"]) == {"anvil", "anvil2"}
    assert tier3.get("exclusive_roles") == ["reasoner"], (
        "tier3 must stay declared as a single-role appliance (D6)"
    )
    assert set(tier3["admits"]) == {"chat"}


def test_nexus_admits_no_always_on_llm(vehicles):
    """The specific rule behind the most common bad proposal: nexus's free
    memory is the on-demand swap pool, not room for a resident model."""
    admits = vehicles["nexus"]["admits"]
    assert "chat" not in admits, (
        "nexus must not admit a resident chat model — its headroom IS the "
        "on-demand swap pool (a 122B resident once made a 7 GB checkpoint load "
        "take ~700s instead of seconds)"
    )
    assert admits.get("media") == ["on_demand"], (
        "nexus's media services are swappable by definition; an always-on one "
        "consumes the pool it is supposed to share"
    )


def test_idle_silicon_register_is_complete(raw):
    rows = raw["meta"].get("idle_silicon")
    assert rows, (
        "meta.idle_silicon must stay populated — it is the register of "
        "accelerators that are idle ON PURPOSE, and deleting a row invites the "
        "next builder to rediscover it as free capacity"
    )
    hosts = set(raw["meta"]["hosts"])
    for row in rows:
        assert str(row.get("what") or "").strip(), f"idle_silicon row has no `what`: {row}"
        assert str(row.get("why_idle") or "").strip(), (
            f"idle_silicon row {row.get('what')!r} has no `why_idle` — the reason "
            f"IS the record; without it the row is just an inventory line"
        )
        assert row.get("host") in hosts, (
            f"idle_silicon row {row.get('what')!r} names host {row.get('host')!r}, "
            f"which is not in meta.hosts"
        )


def test_charter_is_not_silently_reshaped(raw, vehicles):
    """The charter must keep describing the four inference vehicles the fleet is
    built around, plus nexus's dual role. A rename or a merge is an operator
    decision, so it should break here and be re-declared, not drift."""
    assert set(vehicles) >= {"tier3", "tier2-chat", "tier2-analyst", "tier1", "nexus"}
    # nexus owns BOTH always-on retrieval/perception AND the on-demand pool.
    nexus_admits = vehicles["nexus"]["admits"]
    assert {"embed", "rerank"} <= set(nexus_admits)
    assert {"ocr", "stt"} <= set(nexus_admits), (
        "nexus owns always-on PERCEPTION (OCR + streaming ASR) as well as "
        "retrieval — operator decision 2026-08-29"
    )
    # Untouched copy in, untouched copy out: this test must not mutate the fixture.
    assert vehicles == copy.deepcopy(raw["meta"]["vehicles"])
