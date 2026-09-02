"""The management plane end to end (roadmap Workstream E).

The unit tests in `tests/test_management_plane.py` pin the views and the write
boundary. This file exists for the one thing they structurally cannot check:
that a credential **enrolled through the API actually authenticates an inference
call, and stops doing so when revoked** — a loop that runs through the identity
resolver, the OpenAI door, the scheduler and back out through the read plane.

That loop is the whole point of the workstream. Workstream B made a key the
caller's identity and left "create one" meaning *edit a file and restart the
process*; if enrolment does not produce a working credential without a restart,
everything else here is decoration.

The gate is checked from a NON-admin address as well, because every route on
this plane is one an unenrolled caller must not reach, and a gate asserted only
from a host that passes it is not asserted at all.
"""

from __future__ import annotations

import httpx
import pytest

_OUTSIDE = ("203.0.113.10", 5555)   # RFC 5737 documentation address, not enrolled


def _client_from(app, addr) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False,
                                      client=addr),
        base_url="http://proxy", timeout=30.0)


@pytest.mark.asyncio
async def test_a_key_enrolled_through_the_api_authenticates_a_real_call(proxy):
    """🚨 The journey Workstream E exists for: enrol → call → revoke → refused,
    with no restart anywhere in it.

    Each step is checked against a DIFFERENT layer, because the ways this can
    break are independent: enrolment against the registry, the call against the
    OpenAI door's identity resolution, revocation against the resolver again,
    and attribution against the read plane.
    """
    created = await proxy.client.post(
        "/rs/v1/admin/keys",
        json={"agent_id": "e2e-batch", "priority": "P2_POST_TURN"}, headers=proxy.admin)
    assert created.status_code == 201, created.text
    secret = created.json()["key"]
    key_id = created.json()["key_id"]

    # 🚨 The first key flips the identity regime for the whole deployment
    # (§1.5 rule 2), and the response has to say so — from here on an
    # unrecognised key is a 401 rather than being ignored.
    assert any("401" in w for w in created.json()["warnings"])

    # A SECOND key, so the revocation below is a statement about the credential
    # rather than about the registry emptying. Revoking the last key returns the
    # deployment to address identity (§1.5 rule 2) and the revoked secret would
    # then be *ignored* rather than refused — correct, disclosed, and a
    # different assertion (see the unit suite).
    keeper = await proxy.client.post("/rs/v1/admin/keys",
                                     json={"agent_id": "e2e-keeper"}, headers=proxy.admin)
    assert keeper.status_code == 201, keeper.text

    served = await proxy.client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {secret}"},
        json={"model": "chat", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 8})
    assert served.status_code == 200, served.text

    # The credential became the fair-share key, not merely an access token —
    # which is what makes it the quota holder and the budget holder too.
    callers = (await proxy.client.get("/rs/v1/admin/callers", headers=proxy.admin)).json()["callers"]
    mine = [c for c in callers if c["agent_id"] == "e2e-batch"]
    assert mine, "the enrolled identity never reached the scheduler's books"
    assert key_id in mine[0]["identities"]["keys"]

    # 🚨 §1.5 rule 1: a presented key that does not resolve is a 401 and NEVER
    # falls back to the source address — even from loopback, which is otherwise
    # enrolled. Checked here rather than only in the unit suite because the
    # fallback, if it came back, would come back at the door.
    wrong = await proxy.client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer not-a-real-key"},
        json={"model": "chat", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 8})
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "invalid_api_key"

    revoked = await proxy.client.request(
        "DELETE", f"/rs/v1/admin/keys/{key_id}", headers=proxy.admin)
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["revoked"] == key_id

    after = await proxy.client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {secret}"},
        json={"model": "chat", "messages": [{"role": "user", "content": "hi"}],
              "max_tokens": 8})
    assert after.status_code == 401, (
        "a revoked credential still served a request — revocation that needs a "
        "restart to take effect is not revocation")


@pytest.mark.asyncio
async def test_every_management_route_refuses_an_unenrolled_address(proxy):
    """The gate, from outside. Admin is narrower than inference (§3): a host
    that is not even enrolled for inference certainly cannot read the fleet's
    key registry, and a plane whose gate was asserted only from a host that
    passes it would read green with no gate at all."""
    outside = _client_from(proxy.app, _OUTSIDE)
    try:
        for path in ("/rs/v1/admin/config", "/rs/v1/admin/keys",
                     "/rs/v1/admin/callers", "/rs/v1/admin/providers"):
            resp = await outside.get(path)
            assert resp.status_code == 403, f"{path} answered {resp.status_code}"
            # 🚨 The refusal now comes from the NETWORK gate rather than the
            # scope check, and says so — an address outside the reach set is
            # refused before any credential is read. Asserted on the address
            # rather than on the whole sentence: which of the two refusals
            # applies is `identity.py`'s to decide, and pinning the exact string
            # here would make this test the second place that decides it.
            assert _OUTSIDE[0] in resp.json()["error"]
        created = await outside.post("/rs/v1/admin/keys",
                                     json={"agent_id": "intruder"})
        assert created.status_code == 403, (
            "an unenrolled address could MINT ITSELF A CREDENTIAL — the write "
            "routes are gated by the same predicate as the reads, and this is "
            "the assertion that says so")
        patched = await outside.patch("/rs/v1/admin/callers/batch",
                                      json={"weight": 99.0})
        assert patched.status_code == 403
    finally:
        await outside.aclose()


@pytest.mark.asyncio
async def test_the_legacy_control_routes_answer_at_both_prefixes(proxy):
    """🚨 The alias is a live behaviour, not a route-table shape.

    `tests/test_management_plane.py` pins that both paths resolve to the same
    handler object; this pins that both actually WORK against a running proxy,
    because an identical closure reached through a broken path is still a 404 to
    an operator following the documentation.
    """
    endpoint = next(iter(proxy.svc._config.endpoints))

    paused = await proxy.client.post(f"/rs/v1/admin/endpoints/{endpoint}/pause", headers=proxy.admin)
    assert paused.status_code == 200, paused.text
    status = (await proxy.client.get("/v1/status")).json()
    assert status["endpoints"][endpoint].get("admin_paused") is True

    # …and undone through the OTHER spelling, which is the pair that matters:
    # an operator who paused from a dashboard and resumes from a shell must not
    # discover that the two prefixes are different systems.
    resumed = await proxy.client.post(f"/v1/admin/endpoints/{endpoint}/resume", headers=proxy.admin)
    assert resumed.status_code == 200, resumed.text
    status = (await proxy.client.get("/v1/status")).json()
    assert not status["endpoints"][endpoint].get("admin_paused")

    for path in ("/rs/v1/admin/flags", "/v1/admin/flags"):
        assert (await proxy.client.get(path, headers=proxy.admin)).json()["flags"]


@pytest.mark.asyncio
async def test_a_quota_edit_is_visible_to_the_next_admission_decision(proxy):
    """An edit that needed a restart would be the exact failure this workstream
    was opened to remove. Checked through the read plane rather than by poking
    the config object, so the assertion covers the wiring an operator sees."""
    patched = await proxy.client.patch(
        "/rs/v1/admin/callers/internal",
        json={"spill_ok": True, "daily_spend_usd": 2.5}, headers=proxy.admin)
    assert patched.status_code == 200, patched.text

    assert proxy.svc._config.agent_config("internal").spill_ok is True
    assert proxy.svc._state.spend_standing("internal").cap_usd == 2.5

    callers = (await proxy.client.get("/rs/v1/admin/callers", headers=proxy.admin)).json()["callers"]
    internal = [c for c in callers if c["agent_id"] == "internal"][0]
    assert internal["quota"]["runtime"] == {"spill_ok": True,
                                            "daily_spend_usd": 2.5}
    assert internal["quota"]["in_force"]["spill_ok"] is True


@pytest.mark.asyncio
async def test_the_providers_view_reports_a_live_endpoint_honestly(proxy):
    """The e2e half of the gap view: here discovery has actually RUN against a
    real socket, so `in_force` is a measured number rather than the seed. On a
    fake backend that publishes `/props`, declared and in-force should both be
    present and the endpoint should read discoverable.

    🚨 Scoped to ROUTED endpoints, which is what the paragraph above was always
    describing. Since 2026-09-01 the view also reports endpoints the catalog
    declares that nothing serves, and for those the same fields are `null` —
    nobody probed them, so there is no measurement to be honest about. The
    unrouted half is asserted below rather than skipped: `null` there is the
    claim, and a view that started reporting `0` would be saying discovery ran
    and found nothing.
    """
    view = (await proxy.client.get("/rs/v1/admin/providers", headers=proxy.admin)).json()
    rows = {e["endpoint"]: e for e in view["endpoints"]}
    assert rows, "the providers view returned no endpoints"

    routed = [r for r in rows.values() if r["routed"]]
    assert routed, "no routed endpoint — discovery had nothing to run against"
    for row in routed:
        assert row["capacity"]["slots"]["in_force"] > 0
        assert isinstance(row["capacity"]["slots"]["discoverable"], bool)

    for row in (r for r in rows.values() if not r["routed"]):
        assert row["capacity"]["slots"]["in_force"] is None, row["endpoint"]
        assert row["capacity"]["slots"]["discoverable"] is None, row["endpoint"]
        assert row["capacity"]["slots"]["declared"], (
            f"{row['endpoint']}: declared capacity is what an unrouted row is "
            "FOR — without it the row says nothing")

    # 🚨 Never a credential, on a live surface either.
    for provider in view["providers"]:
        assert "key" not in str(provider.get("credential", {}).get("present"))
