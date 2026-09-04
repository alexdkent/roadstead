"""A key may act as another caller, but only names its operator granted. 🚨

**Why this exists at all.** `docs/api.md` §1.5 rule 3 says a key OVERRIDES a
body-declared `agent_id`, and the reason is unchanged: the `agent_id` is the DRR
fair-share key, the quota holder and the budget holder, so a caller that could
type one could type one with a better weight. `/v1/submit` worked that way and
was removed for it.

But that rule quietly assumes the SECURITY boundary and the FAIRNESS boundary are
the same object, and on a real fleet they are not. Measured on the proxy this one
replaces: **38 distinct `agent_id`s, 197k requests/week**, and most of them are
sibling processes inside ONE container sharing a filesystem and a uid — one trust
domain, 38 fair-share identities. The two ways to force those to be one thing are
both bad, and both were considered:

  * one key per `agent_id` — 38 secrets where one boundary is, all readable by
    the same uid. Labelling wearing authentication's clothes.
  * one key, identities collapsed — destroys what the DRR weights exist for. The
    biggest caller is 47% of all traffic; it would share a balance with every
    interactive user turn in the same container.

The third answer is an **allowlist**: the operator grants the names, the caller
picks among them, and a name outside the grant is refused. That keeps the
property that made rule 3 right — a caller still cannot claim an identity nobody
gave it — while letting one credential carry many fair shares.

🚨 **The refusal asymmetry is the delicate part** and has its own test below: an
assertion outside a NON-EMPTY allowlist is a 403, and an assertion against a key
with NO allowlist is ignored exactly as rule 3 has always said. The difference is
whether an operator opted in — not what the caller sent.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from roadstead.identity import IdentityResolver, KeyRegistry, Principal
from roadstead.acl import IPIdentityMap

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _resolver(**kw) -> IdentityResolver:
    reg = KeyRegistry()
    reg.register(secret="kk", agent_id="originfleet", key_id="c2", **kw)
    return IdentityResolver(IPIdentityMap(), reg)


def _principal(**kw) -> Principal:
    return Principal(agent_id="originfleet", source="api_key", key_id="c2", **kw)


# --------------------------------------------------------------------------- #
# The four cases
# --------------------------------------------------------------------------- #

def test_nothing_declared_leaves_the_caller_alone():
    r = _resolver(may_assert=["chat-agent"])
    p = _principal(may_assert=frozenset({"chat-agent"}))
    for declared in (None, "", "   ", "originfleet"):
        out = r.delegate(p, declared)
        assert out.ok and out.principal.agent_id == "originfleet", declared


def test_a_permitted_name_moves_the_fair_share_key():
    r = _resolver(may_assert=["chat-agent", "knowledge_store"])
    out = r.delegate(_principal(may_assert=frozenset({"chat-agent", "knowledge_store"})), "chat-agent")
    assert out.ok
    assert out.principal.agent_id == "chat-agent"


def test_a_name_outside_a_granted_allowlist_is_a_403_that_names_the_grant():
    """🚨 Refused, not quietly re-billed to the key's own identity.

    Silently charging the credential would be the silencer shape: the work
    happens, the bill lands somewhere else, and the DRR weights an operator
    tuned are not the ones in force — with nothing anywhere to say so.
    """
    r = _resolver(may_assert=["chat-agent"])
    out = r.delegate(_principal(may_assert=frozenset({"chat-agent"})), "knowledge_store")
    assert not out.ok
    assert out.denial.status == 403
    assert out.denial.code == "access_denied"
    # The permitted set is named: an operator reading the caller's log has to be
    # able to see what the grant actually is without reaching for the registry.
    assert "chat-agent" in out.denial.message


def test_a_key_with_no_allowlist_IGNORES_a_declared_agent_id():
    """🚨 Ignored, NOT refused — §1.5 rule 3, unchanged.

    This is deliberately not the case above. An operator who wrote `may_assert`
    asked for the field to mean something, so a bad value there is an error worth
    a sentence. An operator who wrote none has a caller sending a field it
    carried over from `/v1/submit`, and 403-ing that would break every such
    caller on upgrade over a claim that was already inert.
    """
    r = _resolver()
    out = r.delegate(_principal(), "chat-agent")
    assert out.ok
    assert out.principal.agent_id == "originfleet"


def test_an_address_derived_caller_can_never_delegate():
    """No special rule needed — `may_assert` is empty on an address principal by
    construction, so it lands in the ignored case. Pinned because the *absence*
    of a rule is what is being relied on."""
    r = _resolver()
    out = r.delegate(Principal(agent_id="internal", source="ip"), "chat-agent")
    assert out.ok and out.principal.agent_id == "internal"


# --------------------------------------------------------------------------- #
# What delegation must NOT carry
# --------------------------------------------------------------------------- #

def test_delegation_moves_the_identity_and_grants_no_POLICY():
    """🚨 The band, deadline floor, admin scope and key_id all survive unchanged.

    A key that could hand itself a different policy by naming another agent
    would be the self-asserted `agent_id` bug restored rather than fenced —
    which is the whole reason the body cannot set these directly.
    """
    from roadstead.config import LLMPriority

    r = _resolver(may_assert=["chat-agent"], admin=True, priority=LLMPriority.P1_TURN_SUPPORT)
    p = _principal(may_assert=frozenset({"chat-agent"}), admin=True,
                   priority=LLMPriority.P1_TURN_SUPPORT, min_timeout_s=600.0)
    out = r.delegate(p, "chat-agent").principal
    assert out.agent_id == "chat-agent"
    assert out.admin is True and out.priority is LLMPriority.P1_TURN_SUPPORT
    assert out.min_timeout_s == 600.0
    assert out.key_id == "c2" and out.source == "api_key"
    # And the grant itself travels, so a delegated principal cannot be used to
    # launder a SECOND, wider assertion.
    assert out.may_assert == frozenset({"chat-agent"})


def test_the_grant_is_read_in_exactly_one_place():
    """🚨 AST, over the whole package. A second site deciding what a credential
    permits is the `_remote_ip` shape — and this one decides who is billed.

    Attribute access only: `management.py` moves the grant around as a dict key
    (`entry.get("may_assert")`), which is transport, not a decision.
    """
    offenders = []
    for path in sorted((_ROOT / "roadstead").rglob("*.py")):
        if path.name == "identity.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "may_assert":
                offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert not offenders, (
        "what a credential may act as is decided outside identity.py: "
        f"{offenders} — route it through IdentityResolver.delegate")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def test_the_keys_file_carries_the_grant(tmp_path):
    f = tmp_path / "keys.yaml"
    f.write_text(
        "keys:\n"
        "  - id: c2\n"
        "    key: kk\n"
        "    agent_id: originfleet\n"
        "    may_assert: [chat-agent, knowledge_store]\n"
        "  - id: single\n"
        "    key: jj\n"
        "    agent_id: bridge-agent\n"
        "    may_assert: bridge-agent-batch\n",       # a scalar has one reading
        encoding="utf-8")
    reg = KeyRegistry()
    reg._load_file(f)
    assert reg.resolve("kk").may_assert == frozenset({"chat-agent", "knowledge_store"})
    assert reg.resolve("jj").may_assert == frozenset({"bridge-agent-batch"})


def test_an_unreadable_grant_fails_CLOSED(tmp_path, caplog):
    """🚨 A widening that failed open would be strictly worse than none. A
    narrowing (`bind`) that failed open would be too — same argument, and this
    is the direction where it costs money rather than access."""
    f = tmp_path / "keys.yaml"
    f.write_text(
        "keys:\n  - id: c2\n    key: kk\n    agent_id: originfleet\n"
        "    may_assert: {chat-agent: true}\n", encoding="utf-8")
    reg = KeyRegistry()
    reg._load_file(f)
    assert reg.resolve("kk").may_assert == frozenset()


def test_the_management_plane_refuses_a_key_that_lists_itself():
    """It reads as though the list is exhaustive; an operator who believes that
    will later wonder why the key still works with the entry removed."""
    from roadstead.management import Invalid, validate_key_create

    ok = validate_key_create({"agent_id": "originfleet",
                              "key_sha256": "a" * 64,
                              "may_assert": ["chat-agent", "chat-agent", "knowledge_store"]})
    assert ok["may_assert"] == ["chat-agent", "knowledge_store"]      # deduped, ordered

    with pytest.raises(Invalid, match="own agent_id"):
        validate_key_create({"agent_id": "originfleet", "key_sha256": "a" * 64,
                             "may_assert": ["originfleet"]})
    with pytest.raises(Invalid, match="non-empty strings"):
        validate_key_create({"agent_id": "c2", "key_sha256": "a" * 64,
                             "may_assert": [""]})


# --------------------------------------------------------------------------- #
# Durability — the bug running it found
# --------------------------------------------------------------------------- #

def test_the_grant_survives_a_restart(tmp_path):
    """🚨 Found by running it, not by the suite.

    The enrolment wrote `may_assert` to the live registry and left it out of the
    OVERLAY record, so the grant lived only in memory. The key kept working
    across a restart and its delegation did not — and the failure is not a 403,
    it is **silent re-billing**: a key with no grant IGNORES a declared
    `agent_id` (§1.5 rule 3), so the delegated caller's work would quietly land
    on the credential's own budget, at a restart unrelated to anything anyone
    changed.

    Driven through the overlay's own round trip rather than through a mocked
    dict, because the defect was in what got WRITTEN.
    """
    from roadstead.config import ProxyConfig
    from roadstead.management import AdminOverlay

    store = tmp_path / "overlay.json"
    overlay = AdminOverlay(store)
    overlay.add_key({
        "id": "c2", "agent_id": "originfleet",
        "key_sha256": "b" * 64, "priority": "P3_INGESTION",
        "min_timeout_s": None, "admin": False, "admin_readonly": False,
        "expires_at": None, "bind": [], "may_assert": ["chat-agent", "knowledge_store"],
    })
    overlay.persist()

    # A fresh process: a new overlay read off disk, layered over a new registry.
    reloaded = AdminOverlay(store)
    reg = KeyRegistry()
    reloaded.apply(reg, ProxyConfig())

    principal = next(iter(reg._by_digest.values()))
    assert principal.agent_id == "originfleet"
    assert principal.may_assert == frozenset({"chat-agent", "knowledge_store"}), (
        "the delegation grant did not survive a restart — the key still works "
        "and its callers are now billed to the credential instead of refused")


def test_enrolment_echoes_the_grant_it_just_made():
    """🚨 Of every field on this response, this is the one an operator most
    needs confirmed: it WIDENS. Reading it back from a separate list is not the
    same as the call that made it saying what it did."""
    from roadstead.management import validate_key_create

    spec = validate_key_create({"agent_id": "originfleet",
                                "key_sha256": "c" * 64,
                                "may_assert": ["chat-agent"]})
    assert spec["may_assert"] == ["chat-agent"]


# --------------------------------------------------------------------------- #
# Disclosure — "ignored" must not also mean "invisible"
# --------------------------------------------------------------------------- #

def test_the_identity_block_reports_who_was_billed():
    """🚨 The complement to "ignored, not refused".

    Refusing a declared `agent_id` on a key with no grant would break every
    caller carrying a `/v1/submit` habit, so it is ignored — and an ignored
    claim that is also invisible is the `finish_reason` silencer in another
    costume: two different outcomes, one 200, identical bodies.
    """
    from roadstead.enriched import identity_block

    class _R:
        agent_id = "originfleet"
        declared_agent_id = "chat-agent"

    assert identity_block(_R()) == {
        "agent_id": "originfleet", "declared": "chat-agent", "honoured": False}

    class _Ok(_R):
        agent_id = "chat-agent"
    assert identity_block(_Ok()) == {
        "agent_id": "chat-agent", "declared": "chat-agent", "honoured": True}

    class _Silent:
        agent_id = "originfleet"
        declared_agent_id = ""
    # 🚨 No `honoured` when nothing was declared — there was nothing to ignore,
    # and a field that is True on every ordinary call means nothing.
    assert identity_block(_Silent()) == {"agent_id": "originfleet"}


def test_the_block_leaks_no_band_or_queue_position():
    """§1.6: a caller cannot observe its own spend demotion. Pinned as a field
    SET, so a later addition to this block is a deliberate decision about what a
    caller may see rather than something that arrived with a refactor."""
    from roadstead.enriched import identity_block

    class _R:
        agent_id = "a"
        declared_agent_id = "b"
        priority = "P0_REALTIME"
        band = "interactive"

    assert set(identity_block(_R())) == {"agent_id", "declared", "honoured"}
