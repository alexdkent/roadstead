"""The management plane (roadmap Workstream E) — `/rs/v1/admin/*`.

Six doctrines are pinned here, each a decision rather than an implementation
detail, and each observed going red by mutating the code it watches:

1. **The read plane's job is the GAP.** A view that echoed the config back would
   be a worse `cat`; these views report the declared value beside the one in
   force, and `GET /rs/v1/admin/config` reads back every knob an operator wrote
   that no code reads.
2. **A management surface never emits a credential** — not the key, not the
   digest, not the value behind an `api_key_env`. Swept over every read view.
3. **A runtime edit never rewrites the operator's config file**, and one that
   cannot be persisted still takes effect and says so.
4. **The write boundary rejects an unknown field**, naming the known set. This
   is the surface whose whole purpose is to expose silent drops.
5. **An edit changes policy, never history** — a weight change moves the
   replenish rate and leaves the deficit already run.
6. **The plane mints no error code.** §1.6 and §1.7 made the same call; §3
   makes it a third time, and a test reads the document back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import ipaddress

import pytest

from roadstead import model_catalog

from roadstead import config as config_mod
from roadstead import hooks, model_catalog
from roadstead.agent_budget import BudgetManager
from roadstead.config import AgentQuotaConfig, LLMPriority, ProxyConfig
from roadstead.identity import KeyRegistry
from roadstead.management import (
    EDITABLE_QUOTA_FIELDS,
    PREFIX,
    AdminOverlay,
    Invalid,
    validate_key_create,
    validate_quota_patch,
)
from roadstead.routes import make_routes
from roadstead.service import ProxyService

from tests.admin_key import ADMIN_HEADERS, enrol_admin

_ROOT = Path(__file__).resolve().parents[1]
API_DOC = _ROOT / "docs" / "api.md"

#: Loopback is in the ACL's built-in admin nets, so an unauthenticated request
#: from here reaches the admin surfaces — the default-deny local-first shape.
_ADMIN_HOST = "127.0.0.1"


class _Req:
    """The slice of a Starlette request these handlers touch."""

    def __init__(self, *, host=_ADMIN_HOST, headers=None, method="GET",
                 body=None, path_params=None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        # 🚨 Default to an AUTHENTICATED admin request. These tests are
        # about what the plane does, not about who may reach it;
        # `headers={}` still means "no credential" for the ones that
        # care. See tests/admin_key.py.
        self.headers = dict(ADMIN_HEADERS) if headers is None else dict(headers)
        # 🚨 identity.py's CSRF gate (`admin_denial._csrf_denial`) requires
        # `Content-Type: application/json` on every mutating admin request as
        # of 2026-09-04. A real caller sending a JSON body already sets this;
        # this double calls the handler directly and skips the header a real
        # HTTP client would add for free — default it here rather than at
        # every call site, same reasoning as `ADMIN_HEADERS` above.
        if method not in ("GET", "HEAD", "OPTIONS"):
            self.headers.setdefault("Content-Type", "application/json")
        self.method = method
        self.query_params: dict = {}
        self.path_params = path_params or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _svc(tmp_path, **cfg) -> ProxyService:
    svc = ProxyService(ProxyConfig(
        queue_db_path=str(tmp_path / "q.db"),
        admin_store_path=str(tmp_path / "admin_overlay.json"),
        **cfg,
    ))
    # The admin plane needs a credential as of 2026-09-01; these tests are
    # about the plane, so they get one. tests/admin_key.py says why.
    enrol_admin(svc)
    return svc


async def _body(response):
    return json.loads(response.body)


# ---------------------------------------------------------------------------
# The allowlists — three of them, and they must agree
# ---------------------------------------------------------------------------

def test_the_quota_allowlists_agree_with_each_other_and_with_the_dataclass():
    """🚨 A knob writable in the FILE and not through the API (or the reverse)
    is the same surprise as one nothing reads at all — it just takes longer to
    find, because both halves look correct in isolation.

    Three lists have to say the same thing: what `agents.yaml` accepts, what a
    PATCH accepts, and what `AgentQuotaConfig` actually holds. Pinned in one
    place so adding a quota knob fails here until all three are updated, rather
    than shipping a field an operator can write in one of two ways.
    """
    dataclass_fields = {
        f.name for f in dataclasses.fields(AgentQuotaConfig)
    } - {"agent_id"}
    assert set(EDITABLE_QUOTA_FIELDS) == config_mod._AGENT_CONFIG_FIELDS
    assert set(EDITABLE_QUOTA_FIELDS) == dataclass_fields


def test_the_file_parser_actually_reads_every_field_it_allows():
    """The allowlist above is a claim about the parser; this checks the parser.

    A name in `_AGENT_CONFIG_FIELDS` that `load_agent_configs` does not branch
    on would silence the unknown-key notice for a knob that is still dropped —
    the guard defeating the thing it guards.
    """
    source = (_ROOT / "roadstead" / "config.py").read_text(encoding="utf-8")
    body = source.split("def load_agent_configs(", 1)[1]
    for name in config_mod._AGENT_CONFIG_FIELDS:
        assert f'"{name}" in cfg' in body, (
            f"{name} is allowed by _AGENT_CONFIG_FIELDS but load_agent_configs "
            f"never reads it — it is dropped, and now silently, because the "
            f"notice believes it is known")


# ---------------------------------------------------------------------------
# The gap — what you wrote that is not in force
# ---------------------------------------------------------------------------

def test_an_unknown_policy_key_is_reported_rather_than_dropped(tmp_path):
    """`models.yaml` `policy:` keys outside `_POLICY_PASSTHROUGH` are dropped.

    That is the failure `docs/internals.md` says has bitten repeatedly, and its whole
    cost is that a dropped knob is indistinguishable from a knob that was never
    load-bearing. It stays dropped — a typo must not stop a fleet booting — but
    it is now RETAINED as a notice, which is what makes it findable a week
    later by somebody who did not read the boot log.
    """
    hooks.clear_config_notices()
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(
        "hosts: {box: 192.0.2.1}\n"
        "providers:\n"
        "  local: {engine: llama.cpp, host: box, port: 9000}\n"
        "endpoints:\n"
        "  chat:\n"
        "    provider: local\n"
        "    slots: 4\n"
        "    policy:\n"
        "      slot_affinity: true\n"
        "      slot_afinity: true\n",   # the typo that costs a day
        encoding="utf-8")
    model_catalog.build_endpoint_kwargs(
        model_catalog.load_catalog(yaml_path, force=True))

    notices = [n for n in hooks.config_notices()
               if n["subject"] == "endpoints.chat.policy"]
    assert notices, "a dropped policy key produced no notice"
    assert "slot_afinity" in notices[-1]["keys"]
    assert "slot_affinity" not in notices[-1]["keys"], (
        "the notice reported a key that IS read — a false positive here trains "
        "an operator to ignore the list")


def test_an_unknown_agent_quota_key_is_reported(tmp_path):
    hooks.clear_config_notices()
    path = tmp_path / "agents.yaml"
    path.write_text("batch:\n  spill_ok: true\n  spil_ok: true\n", encoding="utf-8")
    loaded = config_mod.load_agent_configs(path)

    assert loaded["batch"].spill_ok is True
    notices = [n for n in hooks.config_notices() if n["subject"] == "batch"]
    assert notices and notices[-1]["keys"] == ["spil_ok"]


@pytest.mark.asyncio
async def test_the_config_route_reads_the_notices_back(tmp_path):
    """The seam is only worth having if something reads it.

    `hooks.degradation` had a sink and no reader for a long time and that was
    fine — it reports OUT. A notice reports IN, so a retained notice nothing
    surfaces is just a slower log line.
    """
    hooks.clear_config_notices()
    hooks.config_notice(source="models.yaml", subject="endpoints.x.policy",
                        problem="unknown_key", detail="nothing reads `wombat`",
                        keys=["wombat"])
    svc = _svc(tmp_path)
    payload = await _body(await svc.handle_admin_config(_Req()))

    assert any(n["detail"] == "nothing reads `wombat`"
               for n in payload["notices"])
    assert payload["sources"]["admin_store"]["writable"] is True


# ---------------------------------------------------------------------------
# The overlay — layered over the files, never written into them
# ---------------------------------------------------------------------------

def test_the_overlay_round_trips_through_its_store(tmp_path):
    path = tmp_path / "admin.json"
    overlay = AdminOverlay(path)
    overlay.add_key({"id": "k1", "agent_id": "batch", "key_sha256": "a" * 64})
    overlay.set_agent("batch", {"weight": 2.0})
    overlay.revoke_key("gone")
    overlay.persist()

    reloaded = AdminOverlay(path)
    assert [k["id"] for k in reloaded.keys] == ["k1"]
    assert reloaded.agents == {"batch": {"weight": 2.0}}
    assert reloaded.revoked == ["gone"]


def test_the_overlay_file_is_written_owner_only(tmp_path):
    """The store holds key DIGESTS, and this module's own doctrine (its module
    docstring, above) is that a digest is a working credential for anyone who
    can compute one against it — `key_sha256` enrols a key BY digest directly.
    The file on disk gets the same care the wire format does."""
    path = tmp_path / "admin.json"
    overlay = AdminOverlay(path)
    overlay.add_key({"id": "k1", "agent_id": "batch", "key_sha256": "a" * 64})
    overlay.persist()

    mode = path.stat().st_mode & 0o777
    assert mode == 0o600, f"admin overlay is {oct(mode)}, expected 0600"


def test_a_corrupt_store_does_not_block_startup(tmp_path):
    """An overlay is a convenience; the config files are the deployment.

    Refusing to boot over a truncated JSON file would take a fleet down for a
    layer whose entire content is "what somebody changed through the API since".
    """
    path = tmp_path / "admin.json"
    path.write_text("{not json", encoding="utf-8")
    overlay = AdminOverlay(path)
    assert overlay.keys == [] and overlay.agents == {}


def test_enrolment_is_applied_before_revocation(tmp_path):
    """🚨 A key id in BOTH sections ends up revoked, and the order is why.

    A tombstone is a later statement than the enrolment it follows. Apply them
    the other way round and a revocation is silently undone, at the next
    restart, by the record of the key it revoked — a credential that comes back
    from the dead at a moment unrelated to anything anybody did.
    """
    overlay = AdminOverlay(None)
    overlay.keys = [{"id": "leaked", "agent_id": "batch", "key_sha256": "b" * 64}]
    overlay.revoked = ["leaked"]
    registry = KeyRegistry()
    overlay.apply(registry, ProxyConfig())

    assert not registry.configured
    assert [row["key_id"] for row in registry.snapshot()] == []


def test_a_reenrolled_key_id_is_no_longer_tombstoned():
    """Enrolment clears the tombstone for that id, and it has to.

    Without it, an id reused after a revocation works perfectly until the next
    restart and then stops — because the apply order (enrol, then revoke) would
    tombstone the new key with the old key's record. A credential that dies at a
    moment unrelated to anything anybody did is the worst kind of failure to be
    on call for.
    """
    overlay = AdminOverlay(None)
    overlay.revoke_key("ops")
    overlay.add_key({"id": "ops", "agent_id": "ops", "key_sha256": "c" * 64})

    registry = KeyRegistry()
    overlay.apply(registry, ProxyConfig())
    assert [row["key_id"] for row in registry.snapshot()] == ["ops"]


def test_a_runtime_revocation_survives_a_restart_of_an_env_declared_key(tmp_path,
                                                                       monkeypatch):
    """The tombstone is the only mechanism that can revoke what this layer
    cannot edit. A keys file and an environment variable are both operator
    property; the overlay never rewrites either, so without the tombstone a
    leaked env key would come back on every restart.
    """
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "leaked-secret=batch")
    registry = KeyRegistry.from_env()
    key_id = registry.snapshot()[0]["key_id"]
    assert registry.resolve("leaked-secret") is not None

    overlay = AdminOverlay(tmp_path / "admin.json")
    overlay.revoke_key(key_id)
    overlay.persist()

    # A fresh process: env re-read, overlay re-applied.
    restarted = KeyRegistry.from_env()
    AdminOverlay(tmp_path / "admin.json").apply(restarted, ProxyConfig())
    assert restarted.resolve("leaked-secret") is None


# ---------------------------------------------------------------------------
# Redaction — the rule that turns a read surface into a key store if broken
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_read_view_carries_a_secret_or_its_digest(tmp_path, monkeypatch):
    """🚨 Swept over every read view at once, deliberately.

    `KeyRegistry.snapshot` already refuses to print a digest and has its own
    test. This one is about the PLANE: a view assembled from several sources is
    exactly where a digest gets picked up again by somebody adding a field, and
    a per-view assertion would not have covered the view that did not exist yet.
    """
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "sk-live-secret=batch")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-topsecret")
    svc = _svc(tmp_path)
    digest = hashlib.sha256(b"sk-live-secret").hexdigest()

    blob = ""
    for handler in (svc.handle_admin_keys, svc.handle_admin_callers,
                    svc.handle_admin_providers, svc.handle_admin_config):
        blob += (await handler(_Req())).body.decode()

    assert "sk-live-secret" not in blob
    assert digest not in blob
    assert "sk-or-v1-topsecret" not in blob, (
        "a provider credential's VALUE reached a read view — the plane "
        "publishes the env var's NAME and whether it resolved, never its value")
    # The negative control: the sweep must be looking at something. Without
    # this, a handler that returned `{}` would pass every assertion above.
    assert "OPENROUTER_API_KEY" in blob, (
        "the providers view no longer names the credential variable, so the "
        "sweep above is asserting nothing about it")


@pytest.mark.asyncio
async def test_the_credential_is_reported_as_present_or_missing(tmp_path,
                                                                monkeypatch):
    """An `api_key_env` naming a variable nobody exported makes an endpoint fail
    its health probe for a reason no health page states. This is that reason,
    stated — as a boolean, which is all a boolean can leak."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    svc = _svc(tmp_path)
    view = await _body(await svc.handle_admin_providers(_Req()))

    remote = [p for p in view["providers"] if p["credential"]["env_var"]]
    assert remote, "the example catalog lost its remote provider"
    assert remote[0]["credential"]["present"] is False

    monkeypatch.setenv(remote[0]["credential"]["env_var"], "anything")
    view = await _body(await svc.handle_admin_providers(_Req()))
    assert [p for p in view["providers"]
            if p["provider"] == remote[0]["provider"]][0]["credential"]["present"]


# ---------------------------------------------------------------------------
# Keys — enrolment and revocation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_generated_secret_is_returned_once_and_never_stored(tmp_path):
    """🚨 The registry keeps the digest, so the secret cannot be reissued.

    Losing it costs a revoke-and-enrol, which is the correct price: a surface
    that could re-read a key is a key store, and a key store on the admin plane
    is a much worse thing to leave running than an operator's inconvenience.
    """
    svc = _svc(tmp_path)
    created = await _body(await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "batch", "priority": "P2_BATCH"})))
    secret = created["key"]

    assert secret.startswith("rs-")
    assert svc._state.identity.keys.resolve(secret).agent_id == "batch"

    listed = await _body(await svc.handle_admin_keys(_Req()))
    assert secret not in json.dumps(listed)
    stored = json.loads((tmp_path / "admin_overlay.json").read_text())
    assert secret not in json.dumps(stored)
    assert stored["keys"][0]["key_sha256"] == hashlib.sha256(
        secret.encode()).hexdigest()


@pytest.mark.asyncio
async def test_a_digest_may_be_enrolled_and_returns_no_secret(tmp_path):
    """The migration path: an operator moving an existing credential in sends
    its digest, and nothing comes back to store — there is no secret here for
    this process to have known. The response must not pretend otherwise, and a
    `key` field that was sometimes absent would be worse than one that never
    exists, because a script would read it as an empty string."""
    svc = _svc(tmp_path)
    digest = hashlib.sha256(b"an-existing-key").hexdigest()
    created = await _body(await svc.handle_admin_keys(
        _Req(method="POST",
             body={"agent_id": "migrated", "key_sha256": digest, "id": "old"})))

    assert "key" not in created
    assert created["key_id"] == "old"
    assert svc._state.identity.keys.resolve("an-existing-key").agent_id == "migrated"

    # And the same digest twice is a 409, not a second identity on one secret —
    # the registry's own rule, surfaced rather than swallowed.
    again = await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "other", "key_sha256": digest}))
    assert again.status_code == 409


@pytest.mark.asyncio
async def test_the_first_key_says_the_identity_regime_just_changed(tmp_path):
    """🚨 §1.5 rule 2 is invisible until it fires.

    With no keys configured a presented key is IGNORED and the address decides.
    The first enrolment silently flips that for every caller on the deployment:
    from then on an unrecognised key is a 401 instead of being ignored. The
    operator enrolling their first credential is usually not the person who
    finds out, so the response says it.
    """
    svc = _svc(tmp_path)
    first = await _body(await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "batch"})))
    assert any("401" in w for w in first["warnings"])

    second = await _body(await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "other"})))
    assert "warnings" not in second, (
        "every enrolment warned — a warning that fires always says nothing, "
        "which is the `substituted: true` failure in a different costume")


@pytest.mark.asyncio
async def test_a_plaintext_secret_cannot_be_submitted(tmp_path):
    """🚨 There is no `key` field. A secret sent in a request body lands in an
    access log, a proxy buffer and a shell history — which is exactly why
    `key_sha256:` exists in the keys file. An operator migrating an existing
    credential sends its digest."""
    svc = _svc(tmp_path)
    refused = await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "batch", "key": "sk-mine"}))
    assert refused.status_code == 400
    assert "key_sha256" in json.loads(refused.body)["error"]


@pytest.mark.asyncio
async def test_revoking_an_env_key_works_and_names_what_it_cannot_do(tmp_path,
                                                                     monkeypatch):
    """🚨 Revocation is never refused on provenance grounds.

    "That key came from the environment, use a different tool" is a correctness
    argument answered, in the moment, by a breach. It works — and the response
    says the declaration will outlive the reason it is dead, which is the part
    an operator has to act on.
    """
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "leaked=batch")
    svc = _svc(tmp_path)
    key_id = svc._state.identity.keys.snapshot()[0]["key_id"]

    done = await _body(await svc.handle_admin_key(
        _Req(method="DELETE", path_params={"key_id": key_id})))

    assert done["revoked"] == key_id and done["source"] == "env"
    assert any("ROADSTEAD_API_KEYS" in w for w in done["warnings"])
    assert svc._state.identity.keys.resolve("leaked") is None


@pytest.mark.asyncio
async def test_keys_group_by_agent_id_because_that_is_the_budget_holder(tmp_path):
    """🚨 This answers the roadmap's open "multi-tenancy depth" question.

    Keys are flat and stay flat: the quota holder, the DRR share and the spend
    cap are all keyed on `agent_id`, not on the key, so several keys naming one
    `agent_id` ALREADY give a team one budget with per-key revocation. The view
    groups by it so the structure is visible rather than inferable.
    """
    svc = _svc(tmp_path)
    for _ in range(3):
        await svc.handle_admin_keys(
            _Req(method="POST", body={"agent_id": "research-team"}))
    view = await _body(await svc.handle_admin_keys(_Req()))

    grouped = set(view["by_agent"]["research-team"]["keys"])  # key_ids
    assert len(grouped) == 3
    # The flat list mirrors the grouping. Asserted as containment rather than as
    # a total, because the process also holds a self-minted bootstrap admin key
    # and an operator is meant to SEE that credential in this view — a count
    # here would make the test fail for the right thing happening.
    assert grouped <= {row["key_id"] for row in view["keys"]}


@pytest.mark.asyncio
async def test_revoking_the_last_key_discloses_the_regime_change(tmp_path):
    """🚨 Found by the e2e journey, and the mirror of the first-key warning.

    Revoking the last key empties the registry, and §1.5 rule 2 then says the
    registry is not in play AT ALL: a presented key — including the one just
    revoked — is ignored again and the source address decides. Nothing is
    escalated (the credential confers nothing either way), but "revoked means
    refused" stops being true for a caller whose address is enrolled, and an
    operator who revoked a key in an incident needs to know that in the
    response rather than from a graph.

    🚨 Rule 2 is NOT narrowed to make the surprise go away. It exists so a
    deployment with no keys is not broken by the placeholder Authorization
    header every OpenAI client sends, and a registry that stayed "in play" once
    populated would 401 exactly the deployment that has just deliberately
    emptied it.
    """
    svc = _svc(tmp_path)
    first = await _body(await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "a"})))
    second = await _body(await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "b"})))

    done = await _body(await svc.handle_admin_key(
        _Req(method="DELETE", path_params={"key_id": first["key_id"]})))
    assert "warnings" not in done, (
        "a revocation that left the registry populated warned about it anyway")
    assert svc._state.identity.keys.resolve(second["key"]) is not None

    last = await _body(await svc.handle_admin_key(
        _Req(method="DELETE", path_params={"key_id": second["key_id"]})))
    assert any("LAST key" in w for w in last["warnings"])
    assert not svc._state.identity.keys.configured


@pytest.mark.asyncio
async def test_two_disclosures_about_one_action_both_survive(tmp_path,
                                                              monkeypatch):
    """🚨 `warnings` is a LIST because two of them can be true at once.

    Revoking an env-declared key that is also the last one is both "the
    environment will outlive this" and "the registry is empty now". A single
    string field means the second overwrites the first — which is a config key
    dropped in silence, wearing a response body.
    """
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "only-one=batch")
    svc = _svc(tmp_path)
    key_id = svc._state.identity.keys.snapshot()[0]["key_id"]

    done = await _body(await svc.handle_admin_key(
        _Req(method="DELETE", path_params={"key_id": key_id})))

    assert len(done["warnings"]) == 2
    assert any("LAST key" in w for w in done["warnings"])
    assert any("ROADSTEAD_API_KEYS" in w for w in done["warnings"])


# ---------------------------------------------------------------------------
# The write boundary
# ---------------------------------------------------------------------------

def test_an_unknown_quota_field_is_rejected_and_the_known_set_is_named():
    """🚨 Not dropped. This is the surface whose purpose is exposing silent
    drops; reproducing one here would be a joke at the operator's expense."""
    with pytest.raises(Invalid) as exc:
        validate_quota_patch({"spil_ok": True})
    assert "spil_ok" in str(exc.value)
    assert "spill_ok" in str(exc.value), (
        "the refusal did not name the known set, so an operator with a typo "
        "learns only that they were wrong")


def test_null_is_uncapped_and_zero_is_a_real_cap():
    """🚨 One keystroke apart, opposite policies. `None` means nobody set a cap;
    `0.0` means "no paid spend at all". A bare `float()` collapses them, and the
    caller that gets the wrong one is the one an operator was trying to stop."""
    assert validate_quota_patch({"daily_spend_usd": None})["daily_spend_usd"] is None
    assert validate_quota_patch({"daily_spend_usd": 0})["daily_spend_usd"] == 0.0


def test_a_weight_of_zero_is_refused():
    """A zero weight is a caller that never replenishes — starvation written as
    a quota. §1.6's rule is that a threshold degrades and never rejects; a knob
    that can express permanent starvation is that rejection by another route."""
    with pytest.raises(Invalid):
        validate_quota_patch({"weight": 0})


def test_no_editable_field_can_express_a_rejection():
    """🚨 §1.6 is INHERITED at the write boundary, not re-implemented there.

    The plane cannot mint a policy the admission path refuses to honour because
    there is no field with which to say "refuse this caller" — every knob here
    changes a share, a band or a cap, and crossing a cap costs one band and paid
    spill. A field named `enabled`, `blocked` or `max_requests` would break that
    without touching a line of admission code.
    """
    for name in EDITABLE_QUOTA_FIELDS:
        assert not re.search(r"enabled|blocked|denied|reject|max_requests", name)


def test_a_key_body_rejects_an_unknown_field():
    with pytest.raises(Invalid) as exc:
        validate_key_create({"agent_id": "a", "admn": True})
    assert "admn" in str(exc.value)


def test_a_key_without_an_agent_id_is_refused():
    """The agent_id is the fair-share key, the quota holder and the budget
    holder; a key without one is a credential that authenticates to nothing."""
    with pytest.raises(Invalid):
        validate_key_create({"priority": "P1_TURN_SUPPORT"})


# ---------------------------------------------------------------------------
# Quota edits — policy, not history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_weight_edit_reaches_the_live_budget(tmp_path):
    """Without this the edit applies only to callers the proxy has NOT seen —
    which reads as "the edit did nothing" for exactly the busy caller it was
    aimed at, and is the shape of every knob that turns out to be documentation.
    """
    svc = _svc(tmp_path)
    svc._budget_mgr.set_total_capacity(8.0)
    svc._budget_mgr.get_or_create("batch", weight=1.0)
    # A second caller, because DRR shares are RELATIVE: with one agent the rate
    # is the whole fleet capacity whatever its weight, so a single-caller
    # assertion would pass against a `reweight` that did nothing at all.
    svc._budget_mgr.get_or_create("chat", weight=1.0)
    before = svc._budget_mgr.agents["batch"].replenish_rate

    await svc.handle_admin_caller(
        _Req(method="PATCH", path_params={"agent_id": "batch"},
             body={"weight": 4.0}))

    assert svc._budget_mgr.agents["batch"].weight == 4.0
    assert svc._budget_mgr.agents["batch"].replenish_rate != before
    assert svc._config.agent_config("batch").weight == 4.0


@pytest.mark.asyncio
async def test_a_weight_edit_leaves_the_deficit_already_run(tmp_path):
    """🚨 Changes the RATE, never the BALANCE.

    Re-crediting on a config edit hands a fresh allowance to precisely the
    caller an operator is reweighting *because* it consumes too much — a
    fairness reset wearing a config edit's clothes.
    """
    svc = _svc(tmp_path)
    svc._budget_mgr.set_total_capacity(8.0)
    budget = svc._budget_mgr.get_or_create("batch")
    budget.charge(25.0, now=100.0)
    spent = budget.balance

    await svc.handle_admin_caller(
        _Req(method="PATCH", path_params={"agent_id": "batch"},
             body={"weight": 9.0}))

    assert svc._budget_mgr.agents["batch"].balance == spent


@pytest.mark.asyncio
async def test_the_view_keeps_declared_and_runtime_apart(tmp_path):
    """The gap again, for callers. Which block a value appears in is where it
    came from — an override is visible as a DIFFERENCE rather than a label, so
    an operator can see both what the file says and what is actually in force
    without reading the file."""
    agents_yaml = tmp_path / "agents.yaml"
    agents_yaml.write_text("batch:\n  weight: 3.0\n", encoding="utf-8")
    svc = _svc(tmp_path)
    svc._config.agents.update(config_mod.load_agent_configs(agents_yaml))
    svc._state.admin_overlay.apply(svc._state.identity.keys, svc._config)

    await svc.handle_admin_caller(
        _Req(method="PATCH", path_params={"agent_id": "batch"},
             body={"spill_ok": True}))
    view = await _body(await svc.handle_admin_callers(_Req()))
    batch = [c for c in view["callers"] if c["agent_id"] == "batch"][0]

    assert batch["quota"]["declared"] == {"weight": 3.0}
    assert batch["quota"]["runtime"] == {"spill_ok": True}
    assert batch["quota"]["in_force"]["weight"] == 3.0
    assert batch["quota"]["in_force"]["spill_ok"] is True


@pytest.mark.asyncio
async def test_the_two_kinds_of_money_stay_apart(tmp_path):
    """🚨 §1.6. An invoice and a saving are reported as separate fields for the
    same reason `spend.py` stores them as separate fields — a total would be
    neither number, on the surface an operator uses to decide whether a caller
    costs too much."""
    svc = _svc(tmp_path)
    svc._config.agent_config("batch")
    view = await _body(await svc.handle_admin_callers(_Req()))
    spend = [c for c in view["callers"] if c["agent_id"] == "batch"][0]["spend"]

    assert {"spent_usd", "avoided_usd", "spent_today_usd"} <= set(spend)
    assert not any("total" in k for k in spend), (
        "a combined figure appeared on the money view — it is neither an "
        "invoice nor a saving")


@pytest.mark.asyncio
async def test_an_unpersistable_change_still_takes_effect_and_says_so(tmp_path):
    """🚨 Refusing a revocation because the disk is unwritable is the breach
    argument again. The change applies in memory, and `persisted: false` with a
    reason is a fact an operator can act on — unlike a 503."""
    # Built without `admin_store_path` ON PURPOSE — that is what makes the
    # write unpersistable — so it cannot use `_svc`, and needs its own
    # credential now that the plane requires one.
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    enrol_admin(svc)
    created = await _body(await svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "batch"})))

    assert created["persisted"] is False
    assert "ROADSTEAD_ADMIN_STORE" in created["reason"]
    assert svc._state.identity.keys.resolve(created["key"]).agent_id == "batch"


# ---------------------------------------------------------------------------
# Providers — declared, discovered, in force
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capacity_reports_the_declared_seed_beside_what_is_in_force(tmp_path):
    """The seed in `models.yaml` is overwritten by discovery, so by the time an
    operator asks, the number they wrote is gone. Both are reported, which is
    the only way to tell "discovery agreed" from "discovery never ran"."""
    svc = _svc(tmp_path)
    name = next(iter(svc._config.endpoints))
    svc._config.endpoints[name].max_slots = 99

    view = await _body(await svc.handle_admin_providers(_Req()))
    row = [e for e in view["endpoints"] if e["endpoint"] == name][0]

    assert row["capacity"]["slots"]["in_force"] == 99
    assert row["capacity"]["slots"]["declared"] != 99, (
        "the declared seed is being read from the live config rather than the "
        "catalog, so the gap can never be visible")


@pytest.mark.asyncio
async def test_every_endpoint_the_catalog_declares_is_reported(tmp_path):
    """🚨 The plane answers ONE question: what did you write that is not in force?

    A `planned` endpoint is the plainest possible answer to it — written down,
    parsed, validated, deliberately not serving — and until 2026-09-01 it
    appeared in NO admin view at all. `_providers_view` iterated
    `state.config.endpoints`, which `model_catalog` builds from `cat.routed()`,
    so the two `planned` spill endpoints were filtered out one layer below the
    view and nothing said so. An operator could not see that `spill-chat`
    existed, nor that the only thing between it and service was an unset
    environment variable.

    Driven from the CATALOG rather than from a list of names: a guard naming
    `spill-chat` would pass against a view that had learned to report exactly
    that one endpoint, and would need editing every time the example catalog
    changed.
    """
    svc = _svc(tmp_path)
    view = await _body(await svc.handle_admin_providers(_Req()))
    reported = {e["endpoint"] for e in view["endpoints"]}
    declared = set(model_catalog.load_catalog().endpoints)
    missing = sorted(declared - reported)
    assert not missing, (
        f"the catalog declares {missing} and the management plane reports "
        "neither the endpoint nor the fact that it is not in force")

    # And the guard is not vacuous: the example catalog must still contain an
    # endpoint that is declared and NOT routed, or this proves only that routed
    # endpoints are reported.
    unrouted = sorted(e["endpoint"] for e in view["endpoints"] if not e["routed"])
    assert unrouted, (
        "no unrouted endpoint in the example catalog — this guard would pass "
        "against the bug it was written for")


@pytest.mark.asyncio
async def test_discoverability_is_a_descriptor_property(tmp_path):
    """🚨 Branch on a capability, never on an engine name (docs/internals.md).

    "Is this slot count real or config-seeded" is a property of the ENGINE kind
    — llama.cpp publishes `/props`, vLLM publishes nothing of the sort — and the
    view reads the descriptor rather than comparing a string, so a third engine
    is right here for free.
    """
    svc = _svc(tmp_path)
    view = await _body(await svc.handle_admin_providers(_Req()))
    routed = {e["endpoint"]: e["capacity"]["slots"]["discoverable"]
              for e in view["endpoints"] if e["routed"]}
    assert set(routed.values()) == {True, False}, (
        "every routed endpoint reports the same discoverability, so the example "
        "fleet no longer exercises the asymmetry this field exists to state")

    # 🚨 An endpoint nothing is serving reports None, never False. False is a
    # claim about an engine that was consulted — "this backend publishes no slot
    # count" — and for a `planned` stanza no backend was consulted at all. The
    # same null-not-zero rule the in-force numbers follow, and the reason this
    # view exists: "discovery said no" and "discovery never ran" must not be
    # indistinguishable.
    unrouted = {e["endpoint"]: e["capacity"]["slots"]["discoverable"]
                for e in view["endpoints"] if not e["routed"]}
    assert unrouted, (
        "the example catalog has no unrouted endpoint left, so nothing here "
        "exercises an endpoint that is declared and not in force")
    assert set(unrouted.values()) == {None}, unrouted
    for name, e in ((e["endpoint"], e) for e in view["endpoints"]
                    if not e["routed"]):
        assert e["capacity"]["slots"]["in_force"] is None, name
        assert e["price"] is None, name
        assert e["not_in_force"]["reason"], name

    source = (_ROOT / "roadstead" / "management.py").read_text(encoding="utf-8")
    assert '== "vllm"' not in source and "== 'vllm'" not in source


# ---------------------------------------------------------------------------
# The contract — docs/api.md §3 is executable
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def doc() -> str:
    return API_DOC.read_text(encoding="utf-8")


def _admin_section(doc: str) -> str:
    return doc.split("## 3. Admin / control plane", 1)[1].split("### 3.2", 1)[0]


def _documented_routes(doc: str) -> set[str]:
    """Admin paths published in §3's route TABLE.

    🚨 Table rows only, and that is the whole point of the function. An earlier
    version matched anywhere in §3 and was satisfied by a route mentioned in the
    surrounding prose — so renaming a row left the guard green, which a mutation
    caught. The table is what an operator reads as the contract; a sentence
    about a route is not a publication of it.
    """
    paths: set[str] = set()
    for line in _admin_section(doc).splitlines():
        if not line.startswith("|"):
            continue
        for path in re.findall(r"(/(?:rs/)?v1/admin/[^\s`·|]*)", line):
            paths.add(path.replace("{ep}", "{endpoint}"))
    return paths


def test_every_management_route_is_documented(doc, monkeypatch):
    """§3 publishes the route table and the suite reads it back, the same way
    §2.1's codes and §1.6's admission table are pinned. A route added to the
    code and not the document is a control surface nobody can find.

    🚨 The UI is enabled for the sweep. `ROADSTEAD_ADMIN_UI` gates whether two
    routes are REGISTERED, so leaving it unset would hide them from the guard —
    a documented surface that the contract test cannot see is the guard going
    blind rather than green.
    """
    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")
    documented = _documented_routes(doc)
    assert documented, "the table parser found no routes — it is asserting nothing"
    routed = {r.path for r in make_routes(_FakeSvc())
              if r.path.startswith(PREFIX)}
    assert routed, "the sweep found no management routes"
    for path in routed:
        assert path in documented, (
            f"{path} is served and is not a row in docs/api.md §3's route table")


def test_the_document_publishes_no_route_that_is_not_served(doc, monkeypatch):
    """The direction that actually rots: a route removed from the code leaves
    its row behind, and an operator reads a control surface that 404s."""
    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")   # see the sweep above
    served = {r.path for r in make_routes(_FakeSvc())}
    for path in _documented_routes(doc):
        assert path in served, (
            f"docs/api.md §3's route table publishes {path}, which nothing serves")


def test_the_legacy_control_routes_are_aliased_not_duplicated(doc):
    """🚨 Same handler at both spellings.

    `/v1/admin/*` stays because §3 published it and external consumers read it;
    `/rs/v1/admin/*` exists so an operator has one prefix rather than two. An
    alias that silently stopped aliasing is a control surface that works on one
    spelling and 404s on the other — and the one an operator reaches for is
    whichever the documentation showed them last.
    """
    routes = {(r.path, tuple(sorted(r.methods))): r.endpoint
              for r in make_routes(_FakeSvc())}
    legacy = [(path, methods) for path, methods in routes
              if path.startswith("/v1/admin/")]
    assert len(legacy) == 4, f"expected four legacy control routes, got {legacy}"
    for path, methods in legacy:
        twin = (PREFIX + path[len("/v1/admin"):], methods)
        assert twin in routes, f"{path} has no {PREFIX} twin"
        assert routes[twin] is routes[(path, methods)], (
            f"{twin[0]} is served by a DIFFERENT handler than {path} — an "
            f"alias that diverges is worse than no alias")


def test_the_management_plane_mints_no_error_code(doc):
    """🚨 The third time this call has been made (§1.6, §1.7, now §3).

    A plane that added `store_unwritable` or `quota_invalid` would be publishing
    a fourth spelling of a refusal every client already classifies — and the
    unwritable-store case is not an error at all, because the change took
    effect. Every refusal here is `access_denied`, `invalid_api_key` or
    `invalid_request_error`.
    """
    section = doc.split("### 2.1 Codes", 1)[1].split("### 2.2", 1)[0]
    published = set(re.findall(r"`([a-z_]+)`", section))
    for minted in ("store_unwritable", "quota_invalid", "key_exists",
                   "admin_required", "unknown_caller"):
        assert minted not in published
    source = (_ROOT / "roadstead" / "management.py").read_text(encoding="utf-8")
    for code in re.findall(r'_error\(\s*"([a-z_]+)"', source):
        assert f"`{code}`" in doc, (
            f"management.py emits the code {code!r}, which docs/api.md §2.1 "
            f"does not publish")


class _FakeSvc:
    """`make_routes` only reads attribute names off the service to build
    closures, so route SHAPE is checkable without standing a proxy up."""

    def __getattr__(self, name):
        async def handler(*args, **kwargs):  # pragma: no cover — never called
            raise AssertionError("route handler invoked in a shape test")
        return handler


@pytest.mark.asyncio
async def test_the_config_view_reports_the_REACH_set_not_the_grant_lists(tmp_path):
    """🚨 The plane's own thesis, applied to the plane.

    `ROADSTEAD_ADMIN_NETS` names who may REACH the admin plane, and naming any
    net DROPS the docker-internal default. This view reported
    `builtin: [..., "172.16.0.0/12", ...]` unconditionally — from
    `builtin_admin_nets()`, which is the *identity-grant* net list
    (`_internal_nets`), a different question — so after an operator named their
    own nets it advertised a grant that was no longer in force. On the surface
    that exists to expose exactly that gap.

    `acl.reach_nets()` is the right source and its docstring says it is for the
    management plane; only the STARTUP LOG was calling it. Two sources that
    agree most of the time, which is where §3.5 says the expensive failures
    live.

    Asserted against `may_reach_admin` rather than against a literal list — the
    thing that decides, not a transcription of it.
    """
    svc = _svc(tmp_path)
    acl = svc._state.acl
    acl.register_admin_net("192.0.2.0/24")   # RFC 5737, per the scrub guard

    view = await _body(await svc.handle_admin_config(_Req()))
    nets = view["sources"]["admin_nets"]

    assert nets["docker_default_dropped"] is True
    for probe, expected in (("172.18.0.2", False),   # docker: default dropped
                            ("192.0.2.15", True),    # the operator's own net
                            ("127.0.0.1", True)):    # loopback always survives
        assert acl.may_reach_admin(probe) is expected, probe
        covered = any(ipaddress.ip_address(probe) in ipaddress.ip_network(n)
                      for n in nets["in_force"])
        assert covered is expected, (
            f"the view says {probe} is {'in' if covered else 'not in'} the reach "
            f"set and may_reach_admin says {expected} — the view is describing a "
            f"fleet this proxy is not running")


@pytest.mark.asyncio
async def test_a_fresh_install_still_reports_the_docker_default_as_in_force(tmp_path):
    """The other direction, so the fix is not just 'always say dropped'."""
    svc = _svc(tmp_path)
    nets = (await _body(await svc.handle_admin_config(_Req())))["sources"]["admin_nets"]
    assert nets["docker_default_dropped"] is False
    assert "172.16.0.0/12" in nets["in_force"]
    assert svc._state.acl.may_reach_admin("172.18.0.2") is True
