"""The `calls_push` scope permits ONE route and grants nothing anywhere else.

🚨 THE DEFECT THIS CLOSES: `POST /v1/calls/log` is admin-gated, and since the
address stopped granting admin (2026-09-01) a pusher that presents no credential
is refused `403 access denied for <ip>` — including a relay on loopback. On the
reference fleet every audio / TTS / diarize / stem push was refused this way from
2026-09-05, silently, because the pusher is fire-and-forget. The fix is a
credential that can push and can do NOTHING else: making the pusher an admin
would have handed it the whole plane (read every caller's traffic, pause a
backend, mint keys) to fix a missing row.

WHAT MAKES "nothing else" CHECKABLE. Nothing here hand-lists routes. The route
table is the real one (`app.routes` on the real ASGI app), and two invariants are
driven over every (route, method) in it:

* **the scope grants nothing a plain key does not** — the ingest key and an
  ordinary non-admin key must get the SAME answer everywhere except the one
  route the scope exists for. That covers admin routes, open routes and any route
  added tomorrow, with no list to fall behind.
* **the admin plane is what refuses it** — the routes the sweep sees refusing
  the plain key with the admin 403 are the admin-gated ones, and the sweep
  asserts it found a healthy number of them (an empty sweep would pass vacuously).
"""
from __future__ import annotations

import pytest

from tests.admin_key import ADMIN_HEADERS

_INGEST = {"X-API-Key": "k-ingest"}
_PLAIN = {"X-API-Key": "k-plain"}
_JSON = {"Content-Type": "application/json"}
_PUSH = {"provider": "cortex-orpheus-tts", "kind": "audio",
         "input_tokens": 120, "output_tokens": 350}
#: The one (method, path) the scope exists for.
_THE_ROUTE = ("POST", "/v1/calls/log")


@pytest.fixture
def keys(proxy):
    reg = proxy.svc._state.identity.keys
    reg.register(secret="k-ingest", agent_id="calls-ingest", key_id="ingest",
                 calls_push=True)
    reg.register(secret="k-plain", agent_id="plain", key_id="plain")
    return reg


def _routes(proxy):
    """Every (method, path template) in the REAL route table."""
    out = []
    for route in proxy.app.routes:
        for method in sorted(route.methods or ()):
            if method in ("HEAD", "OPTIONS"):
                continue
            out.append((method, route.path))
    return sorted(set(out))


def _fill(path: str) -> str:
    import re
    return re.sub(r"\{[^}]+\}", "x", path)


async def _hit(proxy, method, path, headers):
    kw = {"headers": {**_JSON, **headers}}
    if method in ("POST", "PUT", "PATCH"):
        kw["content"] = b"{}"
    resp = await proxy.client.request(method, _fill(path), **kw)
    try:
        error = resp.json().get("error")
    except Exception:  # noqa: BLE001 — not every route answers JSON
        error = None
    return resp.status_code, error if isinstance(error, str) else None


@pytest.mark.asyncio
async def test_the_scoped_key_pushes_and_the_row_lands(proxy, keys):
    resp = await proxy.client.post(
        "/v1/calls/log", json=_PUSH, headers=_INGEST)
    assert resp.status_code == 200, resp.text
    rid = resp.json()["request_id"]
    proxy.svc._queue_db.flush(timeout=5.0)
    row = proxy.svc._queue_db._conn.execute(
        "SELECT endpoint, input_tokens, output_tokens, kind "
        "FROM proxy_completions WHERE request_id=?", (rid,)).fetchone()
    assert row == ("cortex-orpheus-tts", 120, 350, "audio")


@pytest.mark.asyncio
async def test_a_bearer_credential_pushes_too(proxy, keys):
    """The relay sends `Authorization: Bearer`; prove that spelling, not only
    `X-API-Key`."""
    resp = await proxy.client.post(
        "/v1/calls/log", json=_PUSH, headers={"Authorization": "Bearer k-ingest"})
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_a_push_with_no_credential_is_still_refused(proxy, keys):
    resp = await proxy.client.post("/v1/calls/log", json=_PUSH)
    assert resp.status_code == 403
    assert resp.json() == {"error": "access denied for 127.0.0.1"}


@pytest.mark.asyncio
async def test_a_plain_key_cannot_push(proxy, keys):
    resp = await proxy.client.post("/v1/calls/log", json=_PUSH, headers=_PLAIN)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_an_admin_key_still_pushes(proxy, keys):
    """Behaviour the route already had: nothing was taken from admin."""
    resp = await proxy.client.post(
        "/v1/calls/log", json=_PUSH, headers=dict(ADMIN_HEADERS))
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_a_wrong_key_is_a_401_not_a_quiet_fallback(proxy, keys):
    resp = await proxy.client.post(
        "/v1/calls/log", json=_PUSH, headers={"X-API-Key": "k-nope"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_a_push_must_still_declare_json(proxy, keys):
    """The CSRF media-type check applies to the scope like any mutation."""
    resp = await proxy.client.post(
        "/v1/calls/log", content=b'{"provider": "x"}',
        headers={**_INGEST, "Content-Type": "text/plain"})
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_the_scope_is_refused_on_every_other_route_in_the_table(proxy, keys):
    """THE containment test, driven from the route table.

    For every (method, path) other than the ingest route, the ingest key must
    get exactly the answer an ordinary non-admin key gets. If the scope ever
    reached another route — an admin GET, an admin write, a route added next
    month — the two answers would part company here.
    """
    routes = [r for r in _routes(proxy) if r != _THE_ROUTE]
    assert len(routes) >= 40, f"the sweep found only {len(routes)} routes"

    diverged = []
    admin_gated = 0
    for method, path in routes:
        ingest = await _hit(proxy, method, path, _INGEST)
        plain = await _hit(proxy, method, path, _PLAIN)
        if ingest != plain:
            diverged.append((method, path, ingest, plain))
        if plain == (403, "access denied for 127.0.0.1"):
            admin_gated += 1
            assert ingest == (403, "access denied for 127.0.0.1"), (
                f"{method} {path}: an admin route accepted or re-worded the "
                f"ingest scope: {ingest}")
    assert not diverged, (
        "the calls_push scope changes the answer on routes it must not touch: "
        f"{diverged}")
    # Non-vacuity: the plane really is what refused. 20+ mutating and reading
    # admin routes exist today; a sweep that saw a handful would prove nothing.
    assert admin_gated >= 20, (
        f"only {admin_gated} routes refused the plain key with the admin 403 — "
        f"the sweep is not reaching the admin plane")


@pytest.mark.asyncio
async def test_the_scope_reads_nothing_on_the_admin_plane(proxy, keys):
    """The same route, by name: the ingest key cannot even LIST keys, which is
    how a leaked pusher credential would otherwise map the whole registry."""
    for method, path in (("GET", "/rs/v1/admin/keys"),
                         ("GET", "/rs/v1/admin/config"),
                         ("POST", "/rs/v1/admin/keys")):
        resp = await proxy.client.request(
            method, path, headers={**_JSON, **_INGEST},
            **({"content": b'{"agent_id": "escalate", "admin": true}'}
               if method == "POST" else {}))
        assert resp.status_code == 403, (method, path, resp.text)
    reg = proxy.svc._state.identity.keys
    assert "escalate" not in {r["agent_id"] for r in reg.snapshot()}


@pytest.mark.asyncio
async def test_only_post_reaches_the_route_through_the_scope(proxy, keys):
    from roadstead.identity import Principal  # noqa: F401 — import proves the surface

    class _R:
        method = "GET"
        headers = {"X-API-Key": "k-ingest"}
        client = type("C", (), {"host": "127.0.0.1"})()

    denial = proxy.svc._state.identity.calls_push_denial(_R())
    assert denial is not None and denial.status == 403
    assert "POST /v1/calls/log only" in denial.message
