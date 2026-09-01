"""The audit trail, and the read-only admin scope.

Both land together because both change what an admin identity IS, and the two
answer halves of one question. `/rs/v1/admin/config` and the other read views
report a **state**; a state cannot say who put it there. And an operator who
wants somebody to be able to *look* has, until now, had to give them the ability
to change everything.

Four doctrines are pinned here, each observed going red:

1. **A scope NARROWS, never widens.** `admin_readonly` cannot grant anything —
   not on a non-admin key, not by overlapping an address grant, not by being
   absent.
2. **The read/write split comes from the HTTP METHOD**, not from a list of write
   routes, so it cannot fall behind `routes.py` in the widening direction.
3. **`identity.py` decides what a scope permits.** One place, and the gate
   renders rather than decides.
4. **Every mutating admin route records.** A trail covering only some of them is
   worse than none: a reader assumes completeness.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from roadstead.acl import IPIdentityMap
from roadstead.config import LLMPriority, ProxyConfig
from roadstead.identity import (
    SAFE_METHODS, IdentityResolver, KeyRegistry, Principal, parse_identity_spec,
)
from roadstead.management import Invalid, PREFIX, validate_key_create
from roadstead.service import ProxyService

_ROOT = Path(__file__).resolve().parents[1]
_ADMIN_HOST = "127.0.0.1"


class _Req:
    def __init__(self, *, host=_ADMIN_HOST, headers=None, method="GET",
                 body=None, path_params=None, query=None):
        class _C:
            pass
        _C.host = host
        self.client = _C()
        self.headers = headers or {}
        self.method = method
        self.query_params = query or {}
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


def _enrol(svc, *, admin=False, admin_readonly=False, agent_id="ops"):
    """Enrol a key through the registry and return its secret."""
    secret = "k-" + agent_id + ("-ro" if admin_readonly else "")
    svc._state.identity.keys.register(
        secret=secret, agent_id=agent_id, admin=admin,
        admin_readonly=admin_readonly, key_id=agent_id + "-key")
    return {"X-API-Key": secret}


# ---------------------------------------------------------------------------
# 1 · A scope NARROWS
# ---------------------------------------------------------------------------

def test_may_admin_write_is_the_conjunction_and_never_grants():
    """🚨 `admin_readonly` alone reaches True for nobody.

    The self-asserted `agent_id` bug in another costume would be a field that,
    set on an identity the operator did not make an admin, produced any admin
    ability at all.
    """
    assert Principal(agent_id="a", admin=True).may_admin_write
    assert not Principal(agent_id="a", admin=True, admin_readonly=True).may_admin_write
    assert not Principal(agent_id="a").may_admin_write
    # The one that matters: readonly WITHOUT admin grants nothing.
    assert not Principal(agent_id="a", admin_readonly=True).may_admin_write
    assert not Principal(agent_id="a", admin_readonly=True).admin


def test_the_grammar_carries_readonly_and_warns_when_it_grants_nothing(caplog):
    """One grammar for keys and the ACL, so `:readonly` works in both."""
    assert parse_identity_spec("ops:admin:readonly") == (
        "ops", LLMPriority.P3_INGESTION, None, True, True)
    # Order does not matter — segments are recognised by shape.
    assert parse_identity_spec("ops:readonly:P1_TURN_SUPPORT:600:admin") == (
        "ops", LLMPriority.P1_TURN_SUPPORT, 600.0, True, True)
    # 🚨 Reported, never dropped in silence: an operator who wrote it believes
    # they issued a safer credential than they did.
    with caplog.at_level("WARNING"):
        agent, _, _, admin, readonly = parse_identity_spec("ops:readonly")
    assert (admin, readonly) == (False, True)
    assert "no effect without 'admin'" in caplog.text


def test_a_key_create_refuses_readonly_without_admin():
    """The same disclosure at the write boundary, as a 400 rather than a log.

    §3.4's rule: on the surface whose purpose is exposing silent drops, a field
    that would do nothing is refused and says what to do instead.
    """
    with pytest.raises(Invalid, match="grants nothing on its own"):
        validate_key_create({"agent_id": "a", "admin_readonly": True})
    ok = validate_key_create({"agent_id": "a", "admin": True,
                              "admin_readonly": True})
    assert ok["admin"] is True and ok["admin_readonly"] is True


def test_a_readonly_acl_entry_beats_the_builtin_loopback_grant():
    """🚨 The narrowing wins over an overlapping wider grant.

    `127.0.0.1=ops:admin:readonly` is the line an operator actually writes, and
    loopback is a BUILT-IN admin net. If the widest overlapping grant won, that
    line would silently be a full grant for every operator who wrote it — a
    narrowing another grant can cancel is not a narrowing, it is a comment.
    """
    acl = IPIdentityMap()
    acl.register("127.0.0.1", "ops")
    acl.register_admin_net("127.0.0.1", readonly=True)
    assert acl.is_admin("127.0.0.1")             # still an admin
    assert acl.is_admin_readonly("127.0.0.1")    # but a narrowed one
    # An address nobody narrowed is untouched.
    assert not acl.is_admin_readonly("10.0.0.1")


# ---------------------------------------------------------------------------
# 2 · The split comes from the METHOD
# ---------------------------------------------------------------------------

def test_safe_methods_are_the_read_set_and_the_default_is_deny():
    """An unanticipated method is a MUTATION, not a read.

    Default-deny in the only direction that is safe: a new method waved through
    to a read-only credential is a silent widening, and a new method refused is
    a visible 403 somebody fixes.
    """
    assert SAFE_METHODS == frozenset({"GET", "HEAD", "OPTIONS"})
    for method in ("POST", "DELETE", "PATCH", "PUT", "PURGE", "FROBNICATE"):
        assert method not in SAFE_METHODS


@pytest.mark.asyncio
async def test_a_readonly_key_reads_the_plane_and_cannot_change_it(tmp_path):
    """The end-to-end shape, through the real handlers."""
    svc = _svc(tmp_path)
    headers = _enrol(svc, admin=True, admin_readonly=True)

    # Every GET is allowed.
    for handler in (svc.handle_admin_config, svc.handle_admin_keys,
                    svc.handle_admin_callers, svc.handle_admin_providers,
                    svc.handle_admin_audit):
        resp = await handler(_Req(headers=headers))
        assert resp.status_code == 200, handler.__name__

    # Every mutation is refused, with its OWN message.
    post = await svc.handle_admin_keys(_Req(
        headers=headers, method="POST", body={"agent_id": "x"}))
    assert post.status_code == 403
    body = await _body(post)
    assert "read-only admin scope" in body["error"]
    # 🚨 NOT the address-shaped refusal: that would send an operator holding a
    # deliberately narrowed credential off to widen an ACL.
    assert "access denied for" not in body["error"]

    patch = await svc.handle_admin_caller(_Req(
        headers=headers, method="PATCH", path_params={"agent_id": "a"},
        body={"weight": 2.0}))
    assert patch.status_code == 403

    delete = await svc.handle_admin_key(_Req(
        headers=headers, method="DELETE", path_params={"key_id": "ops-key"}))
    assert delete.status_code == 403
    # The key it tried to revoke is still registered.
    assert svc._state.identity.keys.resolve("k-ops-ro") is not None


@pytest.mark.asyncio
async def test_a_full_admin_key_still_writes(tmp_path):
    """The control. Without it the test above passes on a plane that refuses
    everybody, which is not the property being claimed."""
    svc = _svc(tmp_path)
    headers = _enrol(svc, admin=True)
    resp = await svc.handle_admin_caller(_Req(
        headers=headers, method="PATCH", path_params={"agent_id": "a"},
        body={"weight": 2.0}))
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_the_legacy_control_routes_inherit_the_split(tmp_path):
    """🚨 The four routes that predate the management plane get it for free.

    They share one gate, which is why the split was put in the gate rather than
    at each management handler: a route list would have to be extended by hand,
    and the failure when it is not is silent and widening.
    """
    svc = _svc(tmp_path)
    headers = _enrol(svc, admin=True, admin_readonly=True)
    flags = await svc.handle_admin_flags(_Req(
        headers=headers, method="POST", body={"context_gate_enforce": True}))
    assert flags.status_code == 403
    assert svc._state.flags.get("context_gate_enforce") is False
    # ...and the GET half of the same handler still works.
    read = await svc.handle_admin_flags(_Req(headers=headers, method="GET"))
    assert read.status_code == 200


# ---------------------------------------------------------------------------
# 3 · identity.py decides
# ---------------------------------------------------------------------------

def test_only_identity_decides_what_a_scope_permits():
    """🚨 An AST guard, not a substring sweep.

    The refusal must be reached through `IdentityResolver.admin_denial`. A gate
    that resolved a principal and then made its own call on whether a read-only
    identity may POST is a second place deciding — the thing this module exists
    to prevent, and the exact shape of the `_remote_ip` that had to be removed
    from `management.py`.

    Reading `may_admin_write` (or `admin_readonly`) anywhere but `identity.py`
    is that second place, whatever it then does with the answer.
    """
    offenders = []
    pkg = _ROOT / "roadstead"
    for path in sorted(pkg.rglob("*.py")):
        if path.name == "identity.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                    "may_admin_write", "admin_readonly"}:
                offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
    assert not offenders, (
        "what an admin scope permits is decided outside identity.py: "
        f"{offenders}")


@pytest.mark.asyncio
async def test_the_non_admin_403_body_is_unchanged(tmp_path):
    """docs/api.md §2 makes the error surface stable even when behaviour is not.

    The refusal for an identity with no admin scope at all keeps the body it has
    always had — no `code` — so a consumer parsing it is not broken by a feature
    it does not use.
    """
    svc = _svc(tmp_path)
    headers = _enrol(svc, admin=False, agent_id="plain")
    resp = await svc.handle_admin_config(_Req(headers=headers))
    assert resp.status_code == 403
    assert await _body(resp) == {"error": "access denied for 127.0.0.1"}


# ---------------------------------------------------------------------------
# 4 · Every mutating admin route records
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_enrolment_is_attributed_to_the_credential_that_made_it(tmp_path):
    svc = _svc(tmp_path)
    headers = _enrol(svc, admin=True)
    await svc.handle_admin_keys(_Req(
        headers=headers, method="POST",
        body={"agent_id": "newcomer", "priority": "P1_TURN_SUPPORT"}))
    trail = svc._state.admin_overlay.audit
    assert trail, "the enrolment recorded nothing — it is unattributable"
    entry = trail[-1]
    assert entry["action"] == "key.enrol"
    assert entry["actor"]["key_id"] == "ops-key"
    assert entry["actor"]["agent_id"] == "ops"
    assert entry["actor"]["source"] == "api_key"
    assert entry["actor"]["address"] == "127.0.0.1"
    assert entry["detail"]["agent_id"] == "newcomer"
    assert isinstance(entry["at"], float)


@pytest.mark.asyncio
async def test_an_audit_record_never_carries_a_credential(tmp_path):
    """🚨 Not the key, not the digest — the rule that governs every other
    readout on this plane. The overlay stores the digest because it must replay
    the enrolment on restart; the trail has no such need, and a digest is a
    working credential to anyone who can compute one.
    """
    import hashlib
    svc = _svc(tmp_path)
    headers = _enrol(svc, admin=True)
    resp = await svc.handle_admin_keys(_Req(
        headers=headers, method="POST", body={"agent_id": "newcomer"}))
    secret = (await _body(resp))["key"]
    blob = json.dumps(svc._state.admin_overlay.audit)
    assert secret not in blob
    assert hashlib.sha256(secret.encode()).hexdigest() not in blob
    # The enrolling credential's own secret is not in there either.
    assert "k-ops" not in blob


@pytest.mark.asyncio
async def test_an_address_derived_admin_is_recorded_with_no_key(tmp_path):
    """A record showing an address and no `key_id` is a meaningful — and
    slightly alarming — thing for an operator to find. Both halves are recorded
    always, so which factor authorized a change is never inferred."""
    svc = _svc(tmp_path)                       # loopback, no key presented
    await svc.handle_admin_caller(_Req(
        method="PATCH", path_params={"agent_id": "a"}, body={"weight": 3.0}))
    trail = svc._state.admin_overlay.audit
    assert trail, "the quota edit recorded nothing"
    entry = trail[-1]
    assert entry["action"] == "caller.quota"
    assert entry["actor"]["key_id"] is None
    assert entry["actor"]["source"] == "ip"
    assert entry["actor"]["address"] == "127.0.0.1"
    assert entry["detail"] == {"weight": 3.0}


@pytest.mark.asyncio
async def test_every_mutating_admin_route_records_something(tmp_path, monkeypatch):
    """🚨 The completeness guard, driven from `routes.py` rather than a list.

    A trail that covered only the routes which happen to persist through the
    admin overlay would be WORSE than none — an operator reading it assumes
    completeness, and "who paused tier2" is exactly the question it would
    silently fail to answer.

    Every mutating admin route in the route table must leave a record. A new one
    that does not fails here, which is the point: the site list lives in
    `routes.py` and this reads it rather than repeating it.
    """
    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")
    from roadstead.routes import make_routes

    mutating = sorted(
        (r.path, m)
        for r in make_routes(_svc(tmp_path))
        for m in (r.methods or set())
        if m not in SAFE_METHODS and "admin" in r.path
    )
    assert mutating, "the sweep found no mutating admin routes"

    # One exerciser per route shape. A route with no entry here fails the
    # assertion below rather than being quietly skipped.
    async def _exercise(svc, path, method):
        if path.endswith("/keys"):
            return await svc.handle_admin_keys(_Req(
                method=method, body={"agent_id": "audit-probe"}))
        if path.endswith("/keys/{key_id}/rotate"):
            svc._state.identity.keys.register(
                secret="rotate-me", agent_id="r", key_id="rotate-me")
            return await svc.handle_admin_key_rotate(_Req(
                method=method, path_params={"key_id": "rotate-me"}, body={}))
        if path.endswith("/keys/{key_id}"):
            svc._state.identity.keys.register(
                secret="doomed", agent_id="d", key_id="doomed-key")
            return await svc.handle_admin_key(_Req(
                method=method, path_params={"key_id": "doomed-key"}))
        if path.endswith("/callers/{agent_id}"):
            return await svc.handle_admin_caller(_Req(
                method=method, path_params={"agent_id": "a"},
                body={"weight": 4.0}))
        if path.endswith("/flags"):
            return await svc.handle_admin_flags(_Req(
                method=method, body={"context_gate_enforce": True}))
        if path.endswith("/pause"):
            return await svc.handle_admin_endpoint_pause(
                "tier1", _Req(method=method, body={"reason": "probe"}),
                pause=True)
        if path.endswith("/resume"):
            return await svc.handle_admin_endpoint_pause(
                "tier1", _Req(method=method, body={}), pause=False)
        if path.endswith("/maintenance"):
            return await svc.handle_maintenance(_Req(
                method=method, body={"endpoint": "tier1", "duration_s": 5}))
        return None

    unrecorded = []
    for path, method in mutating:
        # One service per route, so a record from an earlier one cannot be
        # mistaken for this one's. 🚨 The directory must EXIST first — sqlite
        # fails with "unable to open database file", which reads as a
        # permissions problem and is not.
        sandbox = tmp_path / f"{method}{path}".replace("/", "_").replace(
            "{", "").replace("}", "")
        sandbox.mkdir(parents=True, exist_ok=True)
        svc = _svc(sandbox)
        before = len(svc._state.admin_overlay.audit)
        resp = await _exercise(svc, path, method)
        if resp is None:
            unrecorded.append(f"{method} {path} (no exerciser in this test)")
            continue
        assert resp.status_code < 400, (
            f"{method} {path} failed with {resp.status_code}: {resp.body[:200]}")
        if len(svc._state.admin_overlay.audit) == before:
            unrecorded.append(f"{method} {path}")
    assert not unrecorded, (
        "these admin mutations leave no audit record — a partial trail is worse "
        f"than none, because a reader assumes completeness: {unrecorded}")


# ---------------------------------------------------------------------------
# The UI's write controls
# ---------------------------------------------------------------------------

_UI = _ROOT / "roadstead" / "ui" / "index.html"


def _ui_script() -> str:
    body = _UI.read_text()
    return body[body.index("<script>") + len("<script>"):body.rindex("</script>")]


def _write_action_names(script: str) -> set[str]:
    """Every page function that issues a non-GET request.

    Found from the METHOD it sends, not from a hand-kept list — the same reason
    the server takes the read/write split from the HTTP method rather than from
    a list of write routes.
    """
    names = set()
    for match in re.finditer(r"^(?:async )?function (\w+)\(", script, re.M):
        name = match.group(1)
        nxt = re.search(r"^(?:async )?function ", script[match.end():], re.M)
        body = script[match.end(): match.end() + (nxt.start() if nxt else len(script))]
        if re.search(r'method:\s*"(POST|PATCH|PUT|DELETE)"', body):
            names.add(name)
    return names


def _call_end(text: str, start: int) -> int:
    """Index just past the `el(...)` call beginning at ``start``.

    Paren-matched rather than a fixed lookahead, which is how the first version
    of this guard reported the Cancel button: a 400-character window ran off the
    end of the element and into the next function, where `patchCaller` was
    merely being DEFINED.
    """
    depth, i, quote = 0, text.index("(", start), None
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise AssertionError("unbalanced element constructor in the UI script")


def test_every_write_control_in_the_ui_is_scope_guarded():
    """🚨 Found by RUNNING the page, not by a test — which is why there is now
    a test.

    Six write controls were wrapped by hand and the seventh, the Fleet view's
    Pause/Resume, was missed. It rendered, it worked, the suite was green, and a
    read-only operator would have discovered their scope by draining an endpoint
    and getting a 403 — after the click. Exactly the failure mode `pick()` exists
    for, one layer up: a page with no compiler, checked by eye.

    Every element carrying an `onclick` that reaches a write action must be
    created through `guarded(...)`. The set of write actions comes from the
    METHOD each one sends, so a new mutating handler is covered the day it is
    written.
    """
    script = _ui_script()
    writers = _write_action_names(script)
    assert writers, "the extraction found no write actions — it is asserting nothing"
    # `control` is the shared POST helper behind pause/resume; `api` is generic.
    assert {"enrol", "revoke", "patchCaller", "setFlag", "recordWindow",
            "control"} <= writers, sorted(writers)

    unguarded = []
    for match in re.finditer(r'el\("(?:button|input)"', script):
        span = script[match.start(): _call_end(script, match.start())]
        if "onclick:" not in span:
            continue
        if not any(re.search(r"\b" + re.escape(n) + r"\(", span) for n in writers):
            continue
        line = script[:match.start()].count("\n") + 1
        if not script[:match.start()].rstrip().endswith("guarded("):
            unguarded.append(f"index.html script line ~{line}")
    assert not unguarded, (
        "these UI write controls are live for a read-only admin scope — the "
        "operator learns their scope from a 403 after clicking: " + str(unguarded))


def test_the_scope_is_known_before_the_first_paint():
    """🚨 The second thing running the page found, and the sharper one.

    The write guard defaults to ALLOWED when `you.may_write` is absent — it must,
    or a server that does not publish the field yields a page nobody can use. So
    the boot order is load-bearing: `render()` fired before `primeChrome()`
    resolved painted every write control LIVE for a read-only operator, correct
    only after the 30-second heartbeat. The first paint is the one somebody
    clicks.

    Nothing about this was visible to a test: the buttons rendered, they worked,
    and the server refused them afterwards — which is the 403-after-the-click the
    guard exists to prevent. Pinned on the SOURCE because the ordering is the
    property, and a headless assertion about paint timing would be a flakier
    statement of the same thing.
    """
    script = _ui_script()
    boot = script[script.rindex("primeChrome"):]
    # render() must be chained off primeChrome(), never raced with it.
    assert re.search(r"primeChrome\(\)\.then\(render\)", script), (
        "the first render is not chained off primeChrome — the page paints "
        "before it knows its own admin scope, and the write guard defaults to "
        "allowed")
    assert not re.search(r"^render\(\)", script, re.M), (
        "a bare top-level render() races the config fetch that carries "
        "`you.may_write`")
    assert "primeChrome" in boot


# ---------------------------------------------------------------------------
# The trail reports its own limits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_view_reports_its_bound_and_its_durability(tmp_path):
    """🚨 As DATA, not as prose an operator has to know to look for.

    An in-memory-only trail is the default (no admin store is configured by
    default), and the bound discards silently unless it is counted. A trail that
    presented itself as complete while being neither durable nor unbounded would
    be the `finish_reason` repair again.
    """
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_audit(_Req())
    body = await _body(resp)
    assert body["persisted"] is True          # a store IS configured here
    assert body["capacity"] == 500
    assert body["dropped"] == 0

    # No store → applies, does not survive, and says so.
    bare = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "b.db")))
    body = await _body(await bare.handle_admin_audit(_Req()))
    assert body["persisted"] is False
    assert "ROADSTEAD_ADMIN_STORE" in body["reason"]


def test_the_trail_is_bounded_and_counts_what_it_dropped():
    from roadstead.management import AdminOverlay
    overlay = AdminOverlay()
    for i in range(AdminOverlay._AUDIT_MAX + 25):
        overlay.record({"at": float(i), "action": "t", "target": str(i),
                        "actor": {}, "detail": {}})
    assert len(overlay.audit) == AdminOverlay._AUDIT_MAX
    assert overlay.audit_dropped == 25
    # The OLDEST went. A trail that dropped the newest would be useless.
    assert overlay.audit[0]["target"] == "25"


@pytest.mark.asyncio
async def test_the_trail_round_trips_through_the_store(tmp_path):
    """It rides the same off-loop write as the change it records, so a restart
    that keeps the change keeps the record of who made it."""
    svc = _svc(tmp_path)
    await svc.handle_admin_caller(_Req(
        method="PATCH", path_params={"agent_id": "a"}, body={"weight": 5.0}))
    stored = json.loads((tmp_path / "admin_overlay.json").read_text())
    entries = stored.get("audit", {}).get("entries", [])
    assert entries, "the change persisted and the record of who made it did not"
    assert entries[-1]["action"] == "caller.quota"

    reborn = _svc(tmp_path)
    revived = reborn._state.admin_overlay.audit
    assert revived, "the trail did not survive the restart the change did"
    assert revived[-1]["detail"] == {"weight": 5.0}


@pytest.mark.asyncio
async def test_a_control_route_record_reaches_disk_on_its_own(tmp_path):
    """🚨 The one a live run found, and no unit test would have.

    The four control routes that predate the management plane change no overlay
    state of their own. When their audit record rode a LATER overlay write, the
    most recent entry — the one an operator looks for after an incident — was
    exactly the one a restart lost. A drain is a control action taken in a hurry,
    often right before the restart that would drop the record of it.

    Nothing else writes the store in this test, so the entry is on disk or it is
    nowhere.
    """
    svc = _svc(tmp_path)
    await svc.handle_admin_endpoint_pause(
        "tier1", _Req(method="POST", body={"reason": "gpu swap"}), pause=True)
    stored = json.loads((tmp_path / "admin_overlay.json").read_text())
    entries = stored.get("audit", {}).get("entries", [])
    assert entries, "a drain recorded nothing that survives a restart"
    assert entries[-1]["action"] == "endpoint.pause"
    assert entries[-1]["detail"] == {"reason": "gpu swap"}


@pytest.mark.asyncio
async def test_an_unwritable_store_does_not_break_a_control_action(tmp_path):
    """The trail must never be a reason the control plane stops working.

    Same rule as everywhere else here: an unpersistable change still takes
    effect. A drain that failed because its audit record could not be written
    would be a worse failure than the one the trail protects against.
    """
    svc = _svc(tmp_path)
    svc._state.admin_overlay._path = tmp_path / "no" / "such" / "dir" / "o.json"
    resp = await svc.handle_admin_endpoint_pause(
        "tier1", _Req(method="POST", body={}), pause=True)
    assert resp.status_code == 200
    assert "tier1" in svc._state.paused_endpoints
    # It applied in memory even though it will not survive.
    assert svc._state.admin_overlay.audit[-1]["action"] == "endpoint.pause"


@pytest.mark.asyncio
async def test_the_view_is_newest_first(tmp_path):
    svc = _svc(tmp_path)
    for w in (1.0, 2.0, 3.0):
        await svc.handle_admin_caller(_Req(
            method="PATCH", path_params={"agent_id": "a"}, body={"weight": w}))
    body = await _body(await svc.handle_admin_audit(_Req()))
    assert [e["detail"]["weight"] for e in body["entries"]] == [3.0, 2.0, 1.0]
