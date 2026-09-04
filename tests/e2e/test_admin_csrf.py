"""P0 audit finding — CSRF on the mutating admin plane via cached HTTP Basic.

A browser attaches a cached ``Authorization: Basic`` credential to ANY request
to this origin, including a cross-site ``<form enctype="text/plain">`` POST —
a CORS *simple request* that skips the preflight a real cross-site JSON POST
would need, and whose body can still be syntactically valid JSON. Nothing
upstream of ``request.json()`` used to check the declared content-type, the
browser's own ``Sec-Fetch-Site`` label, or where the credential came from.

``identity.IdentityResolver.admin_denial._csrf_denial`` closes it, in the one
place every mutating admin route (both `/rs/v1/admin/*` and the legacy
`/v1/admin/*` spellings) already funnels through for its read/write scope
check. These run against the real ASGI app via ``httpx.ASGITransport`` — the
`tests/e2e` house rule — because the fix is precisely about what a real HTTP
request looks like, which a hand-built request double can't be trusted to get
wrong the same way a browser does.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from tests.admin_key import ADMIN_HEADERS, ADMIN_SECRET


def _basic(secret: str, user: str = "operator") -> dict:
    blob = base64.b64encode(f"{user}:{secret}".encode()).decode()
    return {"Authorization": f"Basic {blob}"}


def _bearer(secret: str) -> dict:
    return {"Authorization": f"Bearer {secret}"}


async def _enrolled_agent_ids(client: httpx.AsyncClient) -> set:
    resp = await client.get("/rs/v1/admin/keys", headers=dict(ADMIN_HEADERS))
    assert resp.status_code == 200, resp.text
    return {k["agent_id"] for k in resp.json()["keys"]}


@pytest.mark.asyncio
async def test_a_basic_admin_body_declared_as_text_plain_is_refused_before_parsing(proxy):
    """(a) A cross-site ``<form enctype="text/plain">`` POST is exactly this
    shape: a real credential, a body that happens to be valid JSON, declared
    as something a JSON API never accepts. 415, and the write never happens."""
    before = await _enrolled_agent_ids(proxy.client)

    resp = await proxy.client.post(
        "/rs/v1/admin/keys",
        content=b'{"agent_id": "csrf-form-probe"}',
        headers={**_basic(ADMIN_SECRET), "Content-Type": "text/plain"})
    assert resp.status_code == 415, resp.text
    assert resp.json()["code"] == "invalid_request_error"

    after = await _enrolled_agent_ids(proxy.client)
    assert after == before, "a 415 must never mint a key"


@pytest.mark.asyncio
async def test_a_cross_site_labelled_request_is_refused_even_with_correct_content_type(proxy):
    """(b) `Sec-Fetch-Site: cross-site` is sent by every modern browser on a
    cross-origin request and by nothing else — refuse it outright, independent
    of whether the content-type check would have let it through."""
    before = await _enrolled_agent_ids(proxy.client)

    resp = await proxy.client.post(
        "/rs/v1/admin/keys",
        json={"agent_id": "csrf-cross-site-probe"},
        headers={**_basic(ADMIN_SECRET), "Sec-Fetch-Site": "cross-site"})
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "access_denied"

    after = await _enrolled_agent_ids(proxy.client)
    assert after == before


@pytest.mark.asyncio
async def test_basic_plus_json_plus_the_custom_header_succeeds(proxy):
    """(c) The management UI's own path: a same-origin `fetch()` can set an
    arbitrary header on its own initiative; a forged cross-site submission
    cannot. That is what `X-Roadstead-Request` proves."""
    resp = await proxy.client.post(
        "/rs/v1/admin/keys",
        json={"agent_id": "csrf-legit-ui-probe"},
        headers={**_basic(ADMIN_SECRET), "X-Roadstead-Request": "1"})
    assert resp.status_code == 201, resp.text

    after = await _enrolled_agent_ids(proxy.client)
    assert "csrf-legit-ui-probe" in after


@pytest.mark.asyncio
async def test_basic_plus_json_without_the_custom_header_is_refused(proxy):
    """(c') The third layer on its own: a Basic-authenticated write with the
    right content-type, no cross-site label, and NO `X-Roadstead-Request` is
    still refused. This is the branch a browser would reach if the other two
    checks were ever loosened — and the one the first version of this suite
    never pinned, so dropping the header check would have stayed green."""
    before = await _enrolled_agent_ids(proxy.client)

    resp = await proxy.client.post(
        "/rs/v1/admin/keys",
        json={"agent_id": "csrf-basic-no-header-probe"},
        headers=_basic(ADMIN_SECRET))
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "access_denied"

    after = await _enrolled_agent_ids(proxy.client)
    assert after == before, "a Basic write without the header must never mint a key"


@pytest.mark.asyncio
async def test_bearer_plus_json_needs_no_custom_header(proxy):
    """(d) A Bearer/`X-API-Key` credential is never auto-attached by a browser
    in the first place, so the SDK path is unaffected — only Basic carries the
    extra requirement."""
    resp = await proxy.client.post(
        "/rs/v1/admin/keys",
        json={"agent_id": "csrf-sdk-probe"},
        headers=_bearer(ADMIN_SECRET))
    assert resp.status_code == 201, resp.text

    after = await _enrolled_agent_ids(proxy.client)
    assert "csrf-sdk-probe" in after


@pytest.mark.asyncio
async def test_get_routes_are_unaffected_by_any_of_it(proxy):
    """(e) The whole gate is scoped to mutating methods — a GET with a Basic
    credential, no custom header, cross-site label and all, still just reads."""
    resp = await proxy.client.get(
        "/rs/v1/admin/keys",
        headers={**_basic(ADMIN_SECRET), "Sec-Fetch-Site": "cross-site",
                 "Content-Type": "text/plain"})
    assert resp.status_code == 200, resp.text
