"""`GET /v1/status` is open. Two of its fields are ADDRESSES, and were not.

Found by an adversarial pre-publication review, 2026-09-05.

`docs/api.md` §3.12 defends the open analytics family with a specific argument:
what it contains is "identities and arithmetic — agent, call_site, endpoint,
token counts, timings — never a payload, a prompt or a completion." That is
accurate for the rest of the family and was wrong for two fields:

  reliability.admin_ips_seen              every source IP seen on an admin route
  reliability.placeholder_bearers.by_address
                                          the addresses of callers presenting a
                                          credential that means nothing

Neither is arithmetic. The first is the operator's own workstation; the second
is a weak-credential list. One unauthenticated GET returned both, which is a
target list and a shortlist of who to try first.

🚨 The route stays OPEN — that is deliberate and documented. What changed is
that the two address-bearing fields degrade to counts for a caller who is not
an admin. The counts are what the fields are actually read for (an
ACL-tightening go/no-go asks "has anything but me touched an admin route?", and
the placeholder inventory asks "is it empty yet?"), so nothing operational is
lost by withholding the addresses from everyone else.
"""
from __future__ import annotations

import json

import pytest

from roadstead.config import ProxyConfig
from roadstead.service import ProxyService


class _Req:
    """Minimal request double — the idiom used across this suite."""

    def __init__(self, host="127.0.0.1", headers=None, method="GET"):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.headers = dict(headers) if headers else {}
        self.headers.setdefault("Content-Type", "application/json")
        self.method = method
        self.query_params: dict = {}
        self.path_params: dict = {}


async def _status(svc, host):
    return json.loads((await svc.handle_status(_Req(host=host))).body)["reliability"]


@pytest.mark.asyncio
async def test_a_stranger_is_told_how_many_admin_ips_not_which(tmp_path, monkeypatch):
    """The go/no-go signal survives; the target list does not."""
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "real-key=enrolled")
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    svc._state.admin_ips_seen.setdefault("/v1/admin/flags", set()).add("198.51.100.42")

    # 127.0.0.1 carries the built-in admin grant, so the operator still sees it.
    assert (await _status(svc, "127.0.0.1"))["admin_ips_seen"] == {
        "/v1/admin/flags": ["198.51.100.42"]}

    # Anyone else gets the count. Not the address, and not a 403 either — the
    # route is open on purpose and must keep answering.
    stranger = await _status(svc, "203.0.113.9")
    assert stranger["admin_ips_seen"] == {"/v1/admin/flags": 1}
    assert "198.51.100.42" not in json.dumps(stranger)


@pytest.mark.asyncio
async def test_a_stranger_cannot_read_the_weak_credential_list(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADSTEAD_BEARER_PLACEHOLDERS", "not-needed")
    monkeypatch.setenv("ROADSTEAD_API_KEYS", "real-key=enrolled")
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    svc._state.identity.resolve(
        _Req(host="203.0.113.10", headers={"Authorization": "Bearer not-needed"},
             method="POST"))

    assert (await _status(svc, "127.0.0.1"))["placeholder_bearers"] == {
        "count": 1, "by_address": {"203.0.113.10": 1}}

    stranger = await _status(svc, "203.0.113.9")
    assert stranger["placeholder_bearers"] == {"count": 1, "by_address_count": 1}
    assert "203.0.113.10" not in json.dumps(stranger)


@pytest.mark.asyncio
async def test_the_rest_of_the_open_status_payload_is_unchanged(tmp_path):
    """The redaction is two fields, not a general narrowing of the route.

    §3.12's openness argument is right about everything else, and a fix that
    quietly closed the analytics family would break the callers that argument
    exists to permit.
    """
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))
    admin = await _status(svc, "127.0.0.1")
    stranger = await _status(svc, "203.0.113.9")
    differing = {k for k in admin if admin[k] != stranger.get(k, ...)}
    assert differing <= {"admin_ips_seen", "placeholder_bearers"}, differing
    # And the route still answers a stranger at all.
    assert stranger["scheduler_alive"] == admin["scheduler_alive"]
