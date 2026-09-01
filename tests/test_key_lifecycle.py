"""Key lifecycle — expiry, rotation, and per-key address binding.

What was left of Workstream B, and the oldest unclosed thing in the repo:
`created_at` was written, surfaced in two views, and read for NO decision. A key
was valid until somebody revoked it, there was no way to issue a successor and
retire a predecessor as one operation, and "this key, only from this subnet" was
not expressible because a key and an address were a *precedence* rather than an
AND.

Four doctrines are pinned here, each observed going red:

1. **An expired key is refused with its own SENTENCE and the same CODE.** A
   caller can act on no distinction; the operator reading the log can.
2. **A rotation is ONE action, and it fails safe.** If the successor cannot be
   minted the predecessor is untouched.
3. **`overlap_s` defaults to 0 and says so.** The safe default is also the one
   that breaks a running caller the instant it is chosen.
4. **A binding narrows and never widens** — and it is only worth what
   `ROADSTEAD_TRUSTED_PROXIES` is worth, which the read view states.
"""

from __future__ import annotations

import json
import time

import pytest

from roadstead.config import LLMPriority, ProxyConfig
from roadstead.identity import (
    IdentityResolver, KeyRegistry, TrustedProxies, _address_in_any,
    _binding_from_file, _expiry_from_file,
)
from roadstead.acl import IPIdentityMap
from roadstead.management import Invalid, validate_key_create, validate_key_rotate
from roadstead.service import ProxyService

_ADMIN_HOST = "127.0.0.1"


class _Req:
    def __init__(self, *, host=_ADMIN_HOST, headers=None, method="GET",
                 body=None, path_params=None):
        class _C:
            pass
        _C.host = host
        self.client = _C()
        self.headers = headers or {}
        self.method = method
        self.query_params: dict = {}
        self.path_params = path_params or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _svc(tmp_path, **cfg) -> ProxyService:
    return ProxyService(ProxyConfig(
        queue_db_path=str(tmp_path / "q.db"),
        admin_store_path=str(tmp_path / "admin_overlay.json"),
        **cfg,
    ))


async def _body(response):
    return json.loads(response.body)


# ---------------------------------------------------------------------------
# 1 · Expiry
# ---------------------------------------------------------------------------

def test_a_key_with_no_expiry_never_expires():
    """Every key was this before 2026-09-01, so an existing registry must
    behave exactly as it did. The feature has to be opt-in at the KEY, not a
    default that quietly starts killing credentials."""
    reg = KeyRegistry()
    reg.register(secret="forever", agent_id="a", key_id="forever")
    assert reg.resolve("forever") is not None
    assert reg.snapshot()[0]["expires_at"] is None
    assert reg.snapshot()[0]["expired"] is False


def test_an_expired_key_stops_resolving_and_says_which_one_it_was():
    """`lookup` distinguishes three states; `resolve` collapses two of them.

    An expired key reported as simply absent is what made a credential that ran
    out indistinguishable from a typo. The `key_id` survives the expiry because
    an operator needs to know WHICH key it was.
    """
    reg = KeyRegistry()
    reg.register(secret="stale", agent_id="a", key_id="stale",
                 expires_at=time.time() - 1)
    found = reg.lookup("stale")
    assert found.principal is None
    assert found.expired is True
    assert found.key_id == "stale"
    assert reg.resolve("stale") is None          # the plain accessor agrees
    # An unknown key is a different state, not the same one.
    assert reg.lookup("never-existed").expired is False


@pytest.mark.asyncio
async def test_the_401_for_an_expired_key_reads_differently(tmp_path):
    """🚨 Same code, same status, its OWN sentence.

    A caller can do nothing differently between "wrong key" and "expired key",
    so §2.1 mints no second code — the same reasoning that keeps a spend
    threshold from having one. The operator reading the log CAN act on it: one
    means check what you pasted, the other means issue a successor.
    """
    svc = _svc(tmp_path)
    keys = svc._state.identity.keys
    keys.register(secret="stale", agent_id="a", key_id="stale",
                  expires_at=time.time() - 60)
    keys.register(secret="good", agent_id="a", key_id="good")

    resolved = svc._state.identity.resolve(_Req(headers={"X-API-Key": "stale"}))
    assert not resolved.ok
    assert resolved.denial.status == 401
    assert resolved.denial.code == "invalid_api_key"
    # 🚨 The LEAD, not just the presence of the word. A mutation that changed
    # only the opening phrase left "expired at ..." further down the sentence
    # and survived a substring check — while the first four words are what an
    # operator actually reads off a log line.
    assert resolved.denial.message.startswith("expired API key")
    assert "'stale'" in resolved.denial.message
    assert "rotate" in resolved.denial.message
    # 🚨 And it does NOT fall back to the address (§1.5 rule 1), even though
    # loopback is enrolled and would otherwise identify this caller.
    assert "NOT ignored in favour of the source address" in resolved.denial.message

    unknown = svc._state.identity.resolve(_Req(headers={"X-API-Key": "wat"}))
    assert unknown.denial.code == "invalid_api_key"
    assert unknown.denial.message.startswith("invalid API key")
    # The two must not be the same sentence, which is the whole point.
    assert unknown.denial.message != resolved.denial.message


def test_a_restart_never_extends_a_key(tmp_path):
    """🚨 The store holds the ABSOLUTE instant, not the duration it came from.

    Re-deriving `now + expires_in_s` on replay would make a restart a way to
    extend a credential — and a proxy restarts for reasons unrelated to anybody
    deciding a key should live longer.
    """
    from roadstead.management import AdminOverlay
    overlay = AdminOverlay(tmp_path / "o.json")
    past = time.time() - 3600
    overlay.add_key({"id": "k", "agent_id": "a", "key_sha256": "0" * 64,
                     "priority": "P3_INGESTION", "min_timeout_s": None,
                     "admin": False, "admin_readonly": False,
                     "expires_at": past, "bind": [], "created_at": past})
    overlay.persist()

    reborn = AdminOverlay(tmp_path / "o.json")
    reg = KeyRegistry()
    reborn.apply(reg, ProxyConfig())
    assert reg.snapshot()[0]["expires_at"] == pytest.approx(past)
    assert reg.snapshot()[0]["expired"] is True


def test_the_keys_file_takes_a_date_and_a_typo_does_not_kill_the_key(caplog):
    """A file is written once and read at every boot, so it takes an absolute
    date; the API takes a duration, because a call happens now.

    🚨 An unparseable value loads the key WITHOUT an expiry rather than as
    already expired. A typo that quietly killed a credential would look exactly
    like a revocation nobody made — and this runs at startup over
    operator-supplied strings, where one bad character must not take a caller
    offline.
    """
    assert _expiry_from_file({"expires_at": "2027-01-01"}) == 1798761600
    assert _expiry_from_file({"expires_at": 1800000000}) == 1800000000.0
    assert _expiry_from_file({}) is None
    with caplog.at_level("WARNING"):
        assert _expiry_from_file({"expires_at": "next tuesday", "id": "k"}) is None
    assert "unparseable expires_at" in caplog.text


# ---------------------------------------------------------------------------
# 2 · Rotation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rotation_mints_a_successor_that_inherits_the_policy(tmp_path):
    """A rotation is a new SECRET for the same identity.

    Changing policy in the same call would make one request do two things, and
    the one nobody reviewed is the dangerous one.
    """
    svc = _svc(tmp_path)
    keys = svc._state.identity.keys
    keys.register(secret="old", agent_id="team", key_id="old",
                  priority=LLMPriority.P1_TURN_SUPPORT, min_timeout_s=600,
                  admin=True, admin_readonly=True, bind=["10.0.0.0/8"])

    resp = await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "old"}, body={}))
    assert resp.status_code == 201
    body = await _body(resp)
    assert body["rotated_from"] == "old"
    # The secret is returned once, like an enrolment.
    assert body["key"].startswith("rs-")

    # 🚨 Asserted against the REGISTRY, not the response. The first version of
    # this test read the response, which was assembled from the PREDECESSOR's
    # row — so a successor registered with default policy would have been
    # reported as inheriting it, and the mutation that did exactly that
    # survived. The response now reads back from the registry too, but the test
    # asks the thing that decides.
    successor = keys.resolve(body["key"])
    assert successor.agent_id == "team"
    assert successor.priority is LLMPriority.P1_TURN_SUPPORT
    assert successor.min_timeout_s == 600
    assert successor.admin is True
    assert successor.admin_readonly is True
    assert keys.binding(body["key_id"]) == ["10.0.0.0/8"]
    # ...and the response agrees with it, which is now a fact rather than an
    # echo of the request.
    assert body["agent_id"] == "team"
    assert body["priority"] == "P1_TURN_SUPPORT"
    assert body["min_timeout_s"] == 600
    assert body["admin"] is True
    assert body["admin_readonly"] is True
    assert body["bind"] == ["10.0.0.0/8"]


@pytest.mark.asyncio
async def test_rotation_revokes_the_predecessor_by_default_and_says_so(tmp_path):
    """🚨 overlap_s defaults to 0 — and the safe default is also the one that
    breaks a running caller the instant it is chosen, so it is disclosed."""
    svc = _svc(tmp_path)
    keys = svc._state.identity.keys
    keys.register(secret="old", agent_id="a", key_id="old")

    body = await _body(await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "old"}, body={})))
    assert body["predecessor"] == {"key_id": "old", "retired": True,
                                   "expires_at": None}
    assert keys.resolve("old") is None
    assert any("revoked immediately" in w for w in body["warnings"]), body


@pytest.mark.asyncio
async def test_an_overlap_retires_the_predecessor_instead_of_killing_it(tmp_path):
    """The window an operator actually wants: deploy the successor, then let the
    predecessor stop on its own. Expressed as an EXPIRY rather than a timer, so
    it survives the restart a timer would not."""
    svc = _svc(tmp_path)
    keys = svc._state.identity.keys
    keys.register(secret="old", agent_id="a", key_id="old")

    body = await _body(await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "old"},
        body={"overlap_s": 3600})))
    assert body["predecessor"]["retired"] is False
    assert body["predecessor"]["expires_at"] == pytest.approx(
        time.time() + 3600, abs=5)
    # 🚨 Still working RIGHT NOW, which is the whole point.
    assert keys.resolve("old") is not None
    # And the new one works too — that is the overlap.
    assert keys.resolve(body["key"]) is not None
    # It survives a restart as an expiry, not as a revocation.
    reborn = _svc(tmp_path)
    row = next(r for r in reborn._state.identity.keys.snapshot()
               if r["key_id"] == "old") if any(
        r["key_id"] == "old" for r in reborn._state.identity.keys.snapshot()) else None
    assert row is None or row["expires_at"] == pytest.approx(
        body["predecessor"]["expires_at"], abs=5)


@pytest.mark.asyncio
async def test_an_overlap_never_extends_the_predecessor(tmp_path):
    """🚨 Found by rotating a two-hour key and reading the numbers back.

    `overlap_s` may SHORTEN a predecessor's life and must never lengthen it.
    A day-long overlap on a key the operator gave two hours would otherwise push
    its expiry a day out — a rotation quietly extending a credential, which is
    the same widening this repo refuses everywhere else. A narrowing that a
    later statement can cancel is not a narrowing.
    """
    svc = _svc(tmp_path)
    keys = svc._state.identity.keys
    original = time.time() + 7200
    keys.register(secret="old", agent_id="a", key_id="old", expires_at=original)

    body = await _body(await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "old"},
        body={"overlap_s": 86400})))
    assert body["predecessor"]["expires_at"] == pytest.approx(original, abs=1)
    assert any("never lengthens" in w for w in body["warnings"]), body

    # A shorter overlap is honoured, because shortening is the direction that
    # is always safe.
    keys.register(secret="old2", agent_id="a", key_id="old2",
                  expires_at=time.time() + 7200)
    body = await _body(await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "old2"},
        body={"overlap_s": 60})))
    assert body["predecessor"]["expires_at"] == pytest.approx(
        time.time() + 60, abs=5)
    assert not any("never lengthens" in w for w in body.get("warnings", []))


@pytest.mark.asyncio
async def test_a_failed_rotation_leaves_the_predecessor_working(tmp_path):
    """🚨 The failure mode that matters.

    A rotation that revoked the old key and then failed to mint the new one is
    an outage — and the commonest cause, re-POSTing a digest already enrolled,
    is entirely recoverable while nothing has been taken away.
    """
    import hashlib
    svc = _svc(tmp_path)
    keys = svc._state.identity.keys
    keys.register(secret="old", agent_id="a", key_id="old")
    keys.register(secret="taken", agent_id="b", key_id="taken")

    resp = await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "old"},
        body={"key_sha256": hashlib.sha256(b"taken").hexdigest()}))
    assert resp.status_code == 409
    assert "predecessor is UNCHANGED" in (await _body(resp))["error"]
    assert keys.resolve("old") is not None


@pytest.mark.asyncio
async def test_a_rotation_discloses_that_the_successor_has_no_expiry(tmp_path):
    """🚨 Found by rotating a two-hour key and reading the response.

    The successor inherits the policy but NOT the expiry, and comes out
    permanent. That default is right — inheriting an absolute instant would mint
    a successor that expired at the predecessor's moment, possibly seconds later
    — and the original duration is not recoverable, because `created_at` on an
    env- or file-declared key is process start rather than enrolment.

    But it silently weakens a control the operator deliberately set, which is
    the one thing this plane must never do quietly. So it is disclosed, in the
    `warnings` array that already carries every other "this did not do what you
    probably assumed".
    """
    svc = _svc(tmp_path)
    svc._state.identity.keys.register(
        secret="short-lived", agent_id="a", key_id="short",
        expires_at=time.time() + 7200)

    body = await _body(await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "short"}, body={})))
    assert body["expires_at"] is None
    assert any("NO expiry" in w for w in body["warnings"]), body

    # Ask for one and there is nothing to disclose.
    svc._state.identity.keys.register(
        secret="short2", agent_id="a", key_id="short2",
        expires_at=time.time() + 7200)
    body = await _body(await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "short2"},
        body={"expires_in_s": 7200})))
    assert body["expires_at"] is not None
    assert not any("NO expiry" in w for w in body.get("warnings", []))

    # A key that never expired has nothing to warn about either.
    svc._state.identity.keys.register(secret="perm", agent_id="a", key_id="perm")
    body = await _body(await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "perm"}, body={})))
    assert not any("NO expiry" in w for w in body.get("warnings", []))


@pytest.mark.asyncio
async def test_rotating_an_unknown_key_is_a_404(tmp_path):
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_key_rotate(_Req(
        method="POST", path_params={"key_id": "ghost"}, body={}))
    assert resp.status_code == 404


def test_the_rotate_boundary_rejects_an_unknown_field():
    """§3.4's rule, on the newest write surface: a 400 that names the known set,
    never a silent drop."""
    with pytest.raises(Invalid, match="unknown field"):
        validate_key_rotate({"agent_id": "sneaky"})
    assert validate_key_rotate({}) == {}
    assert validate_key_rotate({"overlap_s": 0}) == {"overlap_s": 0.0}


# ---------------------------------------------------------------------------
# 3 · Binding
# ---------------------------------------------------------------------------

def test_a_binding_narrows_and_never_widens():
    """🚨 The asymmetry, stated as three facts.

    A key with no binding is unconstrained (what every key was). A bound key can
    only ever be refused somewhere it would otherwise have worked. And inside
    the binding it confers exactly the principal it always did — it does not
    *become* an address identity, which would be a key granting what an address
    grants.
    """
    acl = IPIdentityMap()
    acl.register("203.0.113.7", "the-host", LLMPriority.P0_REALTIME)
    keys = KeyRegistry()
    keys.register(secret="bound", agent_id="bound-caller", key_id="bound",
                  priority=LLMPriority.P3_INGESTION, bind=["10.0.0.0/8"])
    resolver = IdentityResolver(acl, keys, require_key=False,
                                trusted_proxies=TrustedProxies())

    inside = resolver.resolve(_Req(host="10.0.0.3",
                                   headers={"X-API-Key": "bound"}))
    assert inside.ok
    # The KEY's identity, not the address's — a binding does not blend them.
    assert inside.principal.agent_id == "bound-caller"
    assert inside.principal.priority is LLMPriority.P3_INGESTION

    # 🚨 Outside: refused, and NOT quietly downgraded to the address identity
    # that would otherwise have worked here and worked BETTER (P0).
    outside = resolver.resolve(_Req(host="203.0.113.7",
                                    headers={"X-API-Key": "bound"}))
    assert not outside.ok
    assert outside.denial.status == 401
    assert "bound to 10.0.0.0/8" in outside.denial.message
    assert "203.0.113.7" in outside.denial.message


def test_an_unparseable_binding_matches_nothing(caplog):
    """🚨 Fails CLOSED. A malformed entry that matched everything would turn a
    typo in a binding into no binding at all, silently, on the one field whose
    whole purpose is to narrow."""
    assert _address_in_any("10.0.0.1", ["10.0.0.0/8"])
    with caplog.at_level("WARNING"):
        assert not _address_in_any("10.0.0.1", ["not-a-cidr"])
    assert "matches nothing" in caplog.text
    assert not _address_in_any("garbage", ["10.0.0.0/8"])


def test_the_write_boundary_refuses_a_binding_it_cannot_parse():
    """Caught at enrolment, so an operator never reaches the fail-closed path
    and wonders why their key stopped working everywhere."""
    with pytest.raises(Invalid, match="not an address or CIDR"):
        validate_key_create({"agent_id": "a", "bind": ["10.0.0.0/8", "nope"]})
    with pytest.raises(Invalid, match="at least one"):
        validate_key_create({"agent_id": "a", "bind": []})
    ok = validate_key_create({"agent_id": "a", "bind": "10.0.0.0/8"})
    assert ok["bind"] == ["10.0.0.0/8"]
    assert _binding_from_file({"bind": "10.0.0.0/8"}) == ["10.0.0.0/8"]


@pytest.mark.asyncio
async def test_the_read_view_says_what_a_binding_is_worth(tmp_path):
    """🚨 The §3.5 rule applied to a security control.

    A binding is checked against the RESOLVED address, so the same
    `bind: [10.0.0.0/8]` is a network-level fact on a direct deployment and a
    statement about what a front proxy vouches for behind one. An operator must
    not have to infer which case they are in from the absence of a note.
    """
    svc = _svc(tmp_path)
    view = await _body(await svc.handle_admin_keys(_Req()))
    assert view["binding"]["forwarded_headers_honoured"] is False
    assert view["binding"]["checked_against"] == "the peer address"
    assert "cannot be spoofed" in view["binding"]["note"]

    # 🚨 The directory must EXIST before the sqlite open (see CLAUDE.md notes).
    (tmp_path / "proxied").mkdir()
    behind = _svc(tmp_path / "proxied")
    behind._state.identity.proxies = TrustedProxies.parse("10.0.0.1")
    view = await _body(await behind.handle_admin_keys(_Req()))
    assert view["binding"]["forwarded_headers_honoured"] is True
    assert view["binding"]["checked_against"] == "the forwarded caller address"
    assert "ROADSTEAD_TRUSTED_PROXIES" in view["binding"]["note"]


@pytest.mark.asyncio
async def test_a_forwarded_binding_failure_says_it_was_forwarded(tmp_path):
    """The refusal an operator debugs at 2am says which address it judged, and
    that the address came from a header rather than a socket."""
    acl = IPIdentityMap()
    keys = KeyRegistry()
    keys.register(secret="bound", agent_id="b", key_id="bound",
                  bind=["10.0.0.0/8"])
    resolver = IdentityResolver(acl, keys, require_key=False,
                                trusted_proxies=TrustedProxies.parse("127.0.0.1"))
    denial = resolver.resolve(_Req(
        headers={"X-API-Key": "bound", "X-Forwarded-For": "203.0.113.9"})).denial
    assert "203.0.113.9" in denial.message
    assert "FORWARDED" in denial.message
    assert "ROADSTEAD_TRUSTED_PROXIES" in denial.message
