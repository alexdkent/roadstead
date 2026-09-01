"""The operator UI (roadmap Workstream G) — `GET /rs/v1/admin/ui`.

The HTTP half of management landed with E; this is the face on it. Two
constraints have bound it since the roadmap was written and both still do: it
must not drag a frontend toolchain into a package whose dependency list is six
entries on purpose, and it must not violate the concurrency invariant. So it is
ONE static HTML file with vanilla JS, served by Starlette, and the only work it
does on the server is a file read that goes off-loop.

What is pinned here, and why each one is a decision rather than a detail:

1. **The page has NO external references, and the CSP says so rather than
   trusting a reviewer.** It ships in the wheel, so a `<script src>` or a webfont
   would be a new external dependency for everybody who installs Roadstead — the
   same argument that makes `roadstead.testing` public surface.
2. **The routes it calls are pinned against the routes the server serves.** A UI
   is a client of a contract like any other. Its URLs live in one table
   precisely so a test can read them back.
3. **🚨 The FIELDS it reads are pinned against real responses.** This is the one
   that matters. There is no compiler, no schema and no types here: rename
   `capacity.slots.in_force` and the page renders "—" in one cell of one view
   forever, looking entirely healthy. Every read goes through `pick(obj, "a.b")`
   so the paths are extractable, and every path is walked against a real
   response from a real service. It found a live bug the first time it ran.
4. **The route does not EXIST unless enabled**, and the refusal is a shape a
   browser can act on — a 401 with `WWW-Authenticate`, where the JSON plane
   would answer 403.
5. **Basic carries an API key, not a password.** A browser cannot attach a
   bearer token to a navigation and `EventSource` cannot set a header at all, so
   a browser-facing surface needs a scheme the browser itself carries. Minting a
   password to go with it would be a second credential kind with its own store,
   rotation and revocation, parallel to a registry that already does all three.
"""

from __future__ import annotations

import ast
import base64
import json
import os
import re
import tomllib
from pathlib import Path

import pytest

from roadstead import hooks
from roadstead.acl import IPIdentityMap
from roadstead.config import LLMPriority, ProxyConfig
from roadstead.identity import IdentityResolver, KeyRegistry, presented_key
from roadstead.management import PREFIX, admin_ui_enabled
from roadstead.routes import make_routes
from roadstead.service import ProxyService

_ROOT = Path(__file__).resolve().parents[1]
UI = _ROOT / "roadstead" / "ui" / "index.html"

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


def _basic(secret: str, user: str = "operator") -> dict:
    blob = base64.b64encode(f"{user}:{secret}".encode()).decode()
    return {"Authorization": f"Basic {blob}"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("ROADSTEAD_API_KEYS", "ROADSTEAD_API_KEYS_FILE", "ROADSTEAD_ACL",
                "LLM_PROXY_ACL", "ROADSTEAD_ADMIN_NETS", "ROADSTEAD_REQUIRE_API_KEY",
                "ROADSTEAD_TRUSTED_PROXIES", "ROADSTEAD_ADMIN_UI"):
        monkeypatch.delenv(var, raising=False)


def _svc(tmp_path) -> ProxyService:
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db"),
                                    admin_store_path=str(tmp_path / "o.json")))


# ---------------------------------------------------------------------------
# 1. No toolchain, no external anything
# ---------------------------------------------------------------------------

def test_the_page_references_nothing_outside_itself():
    """🚨 It ships in the wheel, so a reference here is a reference for everyone
    who installs Roadstead. A CDN script is also a third party that can read the
    admin credential the browser is attaching to this origin, which is what makes
    "the key is reachable by script on the page" an acceptable trade only while
    the only script on the page is this one."""
    html = UI.read_text(encoding="utf-8")
    external = re.findall(r'(?:src|href)\s*=\s*["\'](?:https?:)?//[^"\']*', html)
    assert not external, f"external references in the shipped UI: {external}"
    assert "<script src" not in html.replace(" ", "")
    assert "<link" not in html, "no external stylesheet; the CSS is inline"
    # A font stack, not a font file: a webfont is a network request.
    assert "@font-face" not in html
    assert "@import" not in html


def test_the_dependency_list_did_not_grow_a_frontend():
    """The constraint the roadmap states in as many words. A build step, a
    node_modules or a React dependency is out — this is what "out" looks like as
    an assertion rather than as an intention."""
    meta = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    names = {re.split(r"[<>=!\[ ]", d)[0].lower().replace("_", "-")
             for d in meta["project"]["dependencies"]}
    assert names == {"httpx", "starlette", "uvicorn", "pyyaml", "jsonschema", "json-repair"}
    for artefact in ("package.json", "package-lock.json", "node_modules", "vite.config.js"):
        assert not (_ROOT / artefact).exists(), f"a frontend toolchain appeared: {artefact}"


def test_the_asset_ships_in_the_wheel():
    """Serving it from the package is the whole delivery: an operator who pip
    installs Roadstead and sets one variable gets the UI. If package-data stops
    naming it, the install silently has no UI and the route 500s at runtime."""
    meta = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = meta["tool"]["setuptools"]["package-data"]["roadstead"]
    assert any(p.startswith("ui/") for p in patterns), patterns
    assert UI.exists()


# ---------------------------------------------------------------------------
# 2. The routes it calls
# ---------------------------------------------------------------------------

def _ui_routes() -> set[str]:
    """The path literals in the page's one API table."""
    html = UI.read_text(encoding="utf-8")
    block = html.split("const API = {", 1)[1].split("\n};", 1)[0]
    return set(re.findall(r'"(/(?:rs/)?v1/[^"]*)"', block))


def test_the_ui_calls_only_routes_the_server_serves(tmp_path, monkeypatch):
    """🚨 The two-ended pin, from the consuming side.

    A page whose URL drifts 404s in the one view nobody opened this week, and
    nothing else notices. The URLs live in ONE table in the page for exactly this
    reason — scattered through the handlers they would not be extractable, and a
    guard that cannot read its subject is not a guard.
    """
    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")
    served = {r.path for r in make_routes(_svc(tmp_path))}
    # The page builds `/rs/v1/admin/keys/` + an id; Starlette declares the same
    # route as `/rs/v1/admin/keys/{key_id}`. Compare on the prefix a literal can
    # actually carry.
    prefixes = {p.split("{", 1)[0].rstrip("/") for p in served}
    for path in sorted(_ui_routes()):
        assert path in served or path.rstrip("/") in prefixes, (
            f"the UI calls {path}, which no route serves")
    assert len(_ui_routes()) >= 8, "the API table went blind, not empty"


def test_the_ui_route_is_registered_only_when_enabled(tmp_path, monkeypatch):
    """🚨 OFF means the route does not EXIST, not that it refuses.

    Same posture as ROADSTEAD_TRUSTED_PROXIES and ROADSTEAD_REQUIRE_API_KEY: a
    capability that widens what is reachable is the operator's decision, and an
    HTML door on a proxy is reachable by things that would never send an API
    request on purpose.
    """
    monkeypatch.delenv("ROADSTEAD_ADMIN_UI", raising=False)
    assert admin_ui_enabled() is False
    off = {r.path for r in make_routes(_svc(tmp_path))}
    assert f"{PREFIX}/ui" not in off
    assert f"{PREFIX}/stream" not in off

    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")
    assert admin_ui_enabled() is True
    on = {r.path for r in make_routes(_svc(tmp_path))}
    assert f"{PREFIX}/ui" in on
    assert f"{PREFIX}/stream" in on
    # 🚨 And nothing ELSE appeared or vanished with it.
    assert on - off == {f"{PREFIX}/ui", f"{PREFIX}/stream"}
    assert off - on == set()


# ---------------------------------------------------------------------------
# 3. The fields it reads — the guard that matters
# ---------------------------------------------------------------------------

def _pick_paths() -> set[str]:
    """Every dotted path the page dereferences off an API response."""
    html = UI.read_text(encoding="utf-8")
    return set(re.findall(r'pick\([^,]+,\s*"([^"]+)"', html))


def _reachable(obj, out: set[str], prefix: str = "", depth: int = 0) -> None:
    """Every dotted path reachable from ``obj``, and from each sub-object as its
    own root — because the page calls ``pick`` on rows and frames, not only on
    whole responses."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(f"{prefix}{k}")
            _reachable(v, out, f"{prefix}{k}.", depth + 1)
            if prefix:  # this sub-object is also a root somewhere
                _reachable(v, out, f"{k}.", depth + 1)
                out.add(k)
    elif isinstance(obj, list):
        for item in obj[:4]:
            _reachable(item, out, prefix, depth)


def _completed_frame_keys() -> dict:
    """The `call.completed` payload, read off the site that publishes it.

    A two-ended pin against the emitter rather than a sample, because producing a
    real completion here would mean standing up a backend to check a field name.
    """
    src = (_ROOT / "roadstead" / "lifecycle.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "publish" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "call.completed"):
            keys = [k.value for k in node.args[1].keys if isinstance(k, ast.Constant)]
            return {k: 0 for k in keys}
    raise AssertionError("no sse.publish('call.completed', {...}) site found")


@pytest.mark.asyncio
async def test_every_field_the_ui_reads_exists_in_a_real_response(tmp_path, monkeypatch):
    """🚨 THE guard. A UI with no types, no schema and no compiler drifts silently.

    Every read goes through `pick(obj, "a.b.c")` — which is a rule with a test,
    not a style — so the paths are extractable, and each one is walked against a
    response a real service really produced. The failure this closes is not a
    crash: it is one cell rendering "—" forever in a view nobody opened, which is
    the same class as the vision capability nothing read.

    It earned its keep on the first run: the live feed was reading `agent_id` and
    `total_ms` off a frame that publishes `agent` and `duration_s`.
    """
    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")
    # A config notice, so the gap view's own row shape is sampled rather than
    # assumed — the healthy state is an empty list, which samples nothing.
    hooks.clear_config_notices()
    hooks.config_notice(source="models.yaml", subject="tier1",
                        problem="unknown_key", detail="nothing reads `wombat`")
    svc = _svc(tmp_path)
    # An alert, so the alert row's fields are sampled rather than assumed.
    svc._state.alerts.append(
        {"name": "sampled_for_the_ui", "severity": "WARNING", "detail": "…"})
    # A live DRR budget, so the fairness column is sampled. `reweight` edits an
    # existing budget and does not create one — which is the correct behaviour
    # and means a caller has to have been admitted before it has a balance.
    svc._state.budget_mgr.get_or_create("ui-sample")
    # A paused endpoint, so `health.admin_paused` and the pause response are real.
    endpoint = sorted(svc._state.config.endpoints)[0]

    samples = []

    async def sample(coro):
        resp = await coro
        assert resp.status_code < 400, (resp.status_code, resp.body[:300])
        samples.append(json.loads(resp.body))
        return samples[-1]

    await sample(svc.handle_status(_Req()))
    await sample(svc.handle_admin_config(_Req()))
    await sample(svc.handle_admin_providers(_Req()))
    await sample(svc.handle_admin_flags(_Req()))
    await sample(svc.handle_admin_endpoint_pause(
        endpoint, _Req(method="POST", body={"reason": "sampling"}), pause=True))
    await sample(svc.handle_admin_providers(_Req()))  # now with a paused endpoint
    await sample(svc.handle_maintenance(
        _Req(method="POST", body={"endpoint": endpoint, "reason": "sampling",
                                  "duration_s": 60})))
    await sample(svc.handle_maintenance_list(_Req()))
    created = await sample(svc.handle_admin_keys(
        _Req(method="POST", body={"agent_id": "ui-sample", "priority": "P2_POST_TURN",
                                  "min_timeout_s": 30, "admin": False})))
    await sample(svc.handle_admin_keys(_Req()))
    await sample(svc.handle_admin_caller(
        _Req(method="PATCH", body={"weight": 2.5, "spill_ok": True},
             path_params={"agent_id": "ui-sample"})))
    await sample(svc.handle_admin_callers(_Req()))
    await sample(svc.handle_admin_key(
        _Req(method="DELETE", path_params={"key_id": created["key_id"]})))
    samples.append(_completed_frame_keys())

    reachable: set[str] = set()
    for s in samples:
        _reachable(s, reachable)

    #: Paths that cannot be produced without a condition this test will not
    #: manufacture. Each one is a real field at a real site; keep this list SHORT
    #: — a long allowlist is the guard being talked out of its job.
    unsampled = {
        # Only present while an endpoint is inside a rate-windowed cooldown.
        "cooling",
        # `sources.trusted_proxies` / `sources.admin_nets` ARE sampled; these are
        # the sub-keys of an empty operator config, present but empty.
    }
    missing = sorted(p for p in _pick_paths()
                     if p not in reachable and p not in unsampled)
    assert not missing, (
        "the UI reads fields no real response carries — it will render '—' "
        "here forever and look healthy:\n  " + "\n  ".join(missing))
    assert len(_pick_paths()) >= 60, "the extraction went blind, not clean"


# ---------------------------------------------------------------------------
# 4. The door
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_page_is_served_to_an_admin_with_its_security_headers(tmp_path):
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_ui(_Req())
    assert resp.status_code == 200
    assert b"ROADSTEAD" in resp.body
    csp = resp.headers["content-security-policy"]
    # 🚨 No host is allowed anywhere in the policy. `connect-src 'self'` and
    # nothing else — a policy that permitted one CDN would permit that CDN to
    # read the credential the browser attaches to this origin.
    assert "default-src 'none'" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "http" not in csp, f"the CSP names an external origin: {csp}"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_the_door_refuses_a_non_admin_in_a_shape_a_browser_can_act_on(tmp_path):
    """🚨 A 401 where the JSON plane would answer 403, and the deviation is the
    point.

    The plane's 401/403 split is right for an API client, which reads the two
    differently. It is a dead end for a browser: a 403 produces no password box,
    so an operator arriving at this URL for the first time — with no credential,
    because there was nowhere to put one — would see a refusal with no way to
    answer it. A 401 is the one status that makes the browser ask.
    """
    svc = _svc(tmp_path)
    resp = await svc.handle_admin_ui(_Req(host="198.51.100.7"))
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"].startswith('Basic realm=')
    assert b"password" in resp.body.lower()

    # The plane behind the door keeps its own contract — this applies to the
    # door, not to what it opens onto.
    plane = await svc.handle_admin_config(_Req(host="198.51.100.7"))
    assert plane.status_code == 403


@pytest.mark.asyncio
async def test_the_challenge_discloses_nothing_about_the_deployment(tmp_path):
    """Identical whether or not any key is configured. A challenge that appeared
    only on a keyed deployment would tell an unauthenticated caller which of the
    two identity regimes §1.5 is in — which is the one thing they can act on."""
    a = await _svc(tmp_path).handle_admin_ui(_Req(host="198.51.100.7"))
    other = tmp_path / "b"
    other.mkdir()
    svc = _svc(other)
    svc._state.identity.keys.register(secret="k", agent_id="somebody", key_id="k1")
    b = await svc.handle_admin_ui(_Req(host="198.51.100.7"))
    assert (a.status_code, a.headers["www-authenticate"]) == \
           (b.status_code, b.headers["www-authenticate"])


# ---------------------------------------------------------------------------
# 5. Basic carries a key, not a password
# ---------------------------------------------------------------------------

def test_a_basic_password_is_an_api_key_and_the_username_is_ignored():
    """🚨 The whole reason Basic is acceptable here.

    Basic was chosen because a browser can carry it and cannot carry a bearer
    token on a navigation. Minting a PASSWORD to go in it would be a second kind
    of credential with its own store, its own rotation and its own revocation,
    running in parallel with a key registry that already does all three — which
    is exactly what identity.py exists to prevent. So the credential stays the
    key and only the wrapper changes.

    The username is ignored rather than checked against `key_id`: the label is
    public, so requiring it adds a string for an operator to remember and buys
    nothing an attacker holding the key does not already have.
    """
    for user in ("operator", "", "anything-at-all", "not-the-key-id"):
        assert presented_key(_Req(headers=_basic("sk-the-key", user))) == "sk-the-key"


@pytest.mark.parametrize("header, expected", [
    ("Basic " + base64.b64encode(b"u:sk-1").decode(), "sk-1"),
    ("basic " + base64.b64encode(b"u:sk-2").decode(), "sk-2"),   # scheme is case-insensitive
    ("Basic " + base64.b64encode(b":sk-3").decode(), "sk-3"),    # empty username
    ("Basic " + base64.b64encode(b"u:sk:with:colons").decode(), "sk:with:colons"),
    ("Basic !!!!not-base64!!!!", ""),                            # not a credential at all
    ("Basic " + base64.b64encode(b"no-colon").decode(), ""),     # not a Basic payload
    ("Negotiate abc", ""),                                       # somebody else's auth
])
def test_basic_payload_shapes(header, expected):
    assert presented_key(_Req(headers={"Authorization": header})) == expected


def test_an_unparseable_basic_header_is_no_credential_rather_than_a_bad_one():
    """A header we cannot decode is not a credential that failed, so it must not
    401. The address decides, exactly as it would with no header at all —
    otherwise a caller behind something that sets an unrelated Basic header is
    refused over bytes nobody meant as ours."""
    keys = KeyRegistry()
    keys.register(secret="good", agent_id="app", key_id="app")
    resolver = IdentityResolver(IPIdentityMap(), keys, require_key=False)
    res = resolver.resolve(_Req(headers={"Authorization": "Basic !!!"}))
    assert res.ok and res.principal.agent_id == "internal"


def test_a_wrong_basic_key_is_401_and_never_falls_back_to_the_address():
    """🚨 §1.5 rule 1 holds through the new wrapper. A revoked credential that
    silently became a weaker working identity is the failure the rule exists for,
    and adding a scheme must not open a door around it — least of all on the
    surface a browser uses."""
    keys = KeyRegistry()
    keys.register(secret="good", agent_id="ops", admin=True, key_id="ops")
    resolver = IdentityResolver(IPIdentityMap(), keys, require_key=False)
    res = resolver.resolve(_Req(host=_ADMIN_HOST, headers=_basic("wrong")))
    assert not res.ok and res.denial.status == 401


def test_basic_is_ignored_entirely_when_no_keys_are_configured():
    """§1.6 rule 2 likewise. A browser that has cached a credential for this
    origin sends it on every request; on a deployment with no registry that must
    stay invisible, or one stale browser 401s a proxy nobody configured."""
    resolver = IdentityResolver(IPIdentityMap(), KeyRegistry(), require_key=False)
    res = resolver.resolve(_Req(headers=_basic("anything")))
    assert res.ok and res.principal.agent_id == "internal"


@pytest.mark.asyncio
async def test_an_admin_key_opens_the_door_from_anywhere_over_basic(tmp_path):
    """The end-to-end of the choice: an operator on a machine no ACL knows,
    behind whatever front proxy, types their admin key into the browser's
    password box and the page loads."""
    svc = _svc(tmp_path)
    svc._state.identity.keys.register(secret="sk-ops", agent_id="ops", admin=True,
                                      key_id="ops")
    stranger = _Req(host="198.51.100.7")
    assert (await svc.handle_admin_ui(stranger)).status_code == 401
    ok = await svc.handle_admin_ui(_Req(host="198.51.100.7", headers=_basic("sk-ops")))
    assert ok.status_code == 200


# ---------------------------------------------------------------------------
# 6. The live stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_stream_is_admin_gated(tmp_path):
    """🚨 It was not, and a frame here names the caller, the endpoint, the tokens
    and the timing of every call the fleet serves — the live version of
    `/rs/v1/admin/callers`, which has been gated since it existed. A stream reads
    as plumbing rather than as a view, which is how it was missed when the rest
    of these surfaces were tightened."""
    svc = _svc(tmp_path)
    refused = await svc.handle_stream(_Req(host="198.51.100.7"))
    assert refused.status_code == 403
    allowed = await svc.handle_stream(_Req(host=_ADMIN_HOST))
    assert allowed.status_code == 200
    assert allowed.media_type == "text/event-stream"


def test_the_stream_alias_is_the_same_handler_and_the_same_gate(tmp_path, monkeypatch):
    """🚨 The alias is load-bearing rather than tidy: `EventSource` cannot set a
    request header AT ALL, so a page can only reach an authenticated stream via
    a credential the browser attaches itself — and a browser attaches cached
    Basic credentials by directory. `/rs/v1/admin/stream` is the one directory
    the page was challenged in.

    Same handler, same gate — the pattern the four control routes established. An
    alias that stopped aliasing would be a stream that works on one spelling and
    404s on the other, which is the failure that pinned those four.
    """
    monkeypatch.setenv("ROADSTEAD_ADMIN_UI", "1")
    routes = {r.path: r for r in make_routes(_svc(tmp_path))}
    assert routes["/v1/stream"].endpoint is routes[f"{PREFIX}/stream"].endpoint


# ---------------------------------------------------------------------------
# 7. The invariant
# ---------------------------------------------------------------------------

def test_the_asset_read_goes_off_the_loop():
    """A dashboard polling on a fast cadence is exactly the load that finds a
    violation, so the one piece of I/O this surface does must not be on the loop
    thread. It is small and it is cached after the first hit, so this is not
    about throughput — it is that the exception a dashboard makes for itself is
    the one still there when somebody serves a bigger asset from this handler."""
    src = (_ROOT / "roadstead" / "management.py").read_text(encoding="utf-8")
    body = src.split("async def handle_admin_ui(", 1)[1].split("\n    # ----", 1)[0]
    assert "asyncio.to_thread" in body
    assert "read_text" in body
    assert body.index("asyncio.to_thread") < body.index("read_text"), (
        "the read must be handed to the thread, not merely near one")
