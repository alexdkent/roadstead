"""One admin credential, shared by the tests that exercise the admin PLANE.

🚨 Why this module exists. Until 2026-09-01 an address granted admin, so a
request from ``127.0.0.1`` with no credential at all could pause a backend or
flip a runtime flag — and roughly sixty tests were written against that, none of
them deliberately. They were testing what the plane DOES, and the fact that they
were also asserting it needs no credential was invisible.

When the address grant was removed they all failed at once, which is the right
outcome and the reason the change was safe to make. They are fixed here rather
than one by one: a test about the shape of a capacity view should say nothing
about authentication.

**Authentication itself is pinned elsewhere, deliberately** — ``test_acl.py``,
``test_identity_keys.py`` and ``test_admin_gate.py``. Nothing in this module is
allowed to become the place the gate is tested, or a regression that opened the
plane back up would be masked by the very helper the plane's tests rely on.
"""
from __future__ import annotations

ADMIN_SECRET = "test-admin-secret"
ADMIN_HEADERS = {"X-API-Key": ADMIN_SECRET}


def enrol_admin(svc, *, secret: str = ADMIN_SECRET, agent_id: str = "test-admin",
                key_id: str = "test-admin", **kw) -> dict:
    """Register an admin key straight on a live service and return its headers.

    Goes through the registry rather than the environment on purpose: an env
    var is process-global and would silently put the key registry "in play" for
    tests that assert the no-keys regime of `docs/api.md` §1.5 rule 2.
    """
    svc._state.identity.keys.register(
        secret=secret, agent_id=agent_id, admin=True, key_id=key_id,
        # 🚨 BOOTSTRAP, so `KeyRegistry.configured` stays False and these tests
        # still observe the empty-registry regime of §1.5 rule 2. Without this
        # the helper would silently answer "has an operator adopted API keys?"
        # with yes, and the tests about enrolling the FIRST key — the ones that
        # check the regime-change disclosure — would be testing a registry that
        # was never empty.
        bootstrap=True, **kw)
    return dict(ADMIN_HEADERS)
