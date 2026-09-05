"""The server half of the keepalive ordering invariant (``docs/api.md`` §1.3).

The client must retire an idle connection BEFORE the proxy does. If the server
wins that race, a caller's POST lands on a socket the server has already closed
and fails with ``RemoteProtocolError("Server disconnected without sending a
response.")`` — a transport error for a request that was never attempted.

This assertion came from a quarantined file that imported the client's
``_CLIENT_KEEPALIVE_EXPIRY_S`` and compared the two constants directly. That
file was 21 tests of client code, which is not Roadstead's to test; this one
assertion is the only part of it Roadstead owns, so it is rehomed here rather
than lost.

🚨 ``PROXY_SERVER_KEEPALIVE_S`` reads ``ROADSTEAD_PROXY_SERVER_KEEPALIVE_S``,
so the invariant can be broken from a deployment config without touching code —
which is exactly why the server side deserves its own test.
"""
from __future__ import annotations

import importlib
import os

import pytest

from tests.wire_contract import CLIENT_KEEPALIVE_EXPIRY_S, KEEPALIVE_MIN_MARGIN_S


def _server_keepalive(monkeypatch, value: str | None = None) -> float:
    """Re-import the entrypoint so the module-level env read runs again."""
    import roadstead.__main__ as main_mod
    if value is None:
        monkeypatch.delenv("ROADSTEAD_PROXY_SERVER_KEEPALIVE_S", raising=False)
    else:
        monkeypatch.setenv("ROADSTEAD_PROXY_SERVER_KEEPALIVE_S", value)
    return importlib.reload(main_mod).PROXY_SERVER_KEEPALIVE_S


@pytest.fixture(autouse=True)
def _restore_module(monkeypatch):
    """Reloading ``__main__`` mutates a module other tests import from, so put
    the default back whatever this test did."""
    yield
    monkeypatch.delenv("ROADSTEAD_PROXY_SERVER_KEEPALIVE_S", raising=False)
    importlib.reload(importlib.import_module("roadstead.__main__"))


def test_server_keepalive_clears_the_client_by_the_required_margin(monkeypatch):
    server = _server_keepalive(monkeypatch)
    assert server > CLIENT_KEEPALIVE_EXPIRY_S, (
        f"server idle timeout {server}s does not exceed the client's "
        f"{CLIENT_KEEPALIVE_EXPIRY_S}s — the proxy will close sockets the "
        f"caller is about to reuse")
    # Margin, not merely a positive difference: 5s vs 4.5s was "correct" by the
    # first assertion and still raced, which is why the second one exists.
    assert server - CLIENT_KEEPALIVE_EXPIRY_S >= KEEPALIVE_MIN_MARGIN_S, (
        f"margin {server - CLIENT_KEEPALIVE_EXPIRY_S}s is under the required "
        f"{KEEPALIVE_MIN_MARGIN_S}s")


def test_an_env_override_cannot_silently_break_the_ordering(monkeypatch):
    """The knob is a deployment config, so prove the guard reads the live value
    rather than a constant that happens to be right."""
    assert _server_keepalive(monkeypatch, "3") == 3
    server = _server_keepalive(monkeypatch, "3")
    assert not (server - CLIENT_KEEPALIVE_EXPIRY_S >= KEEPALIVE_MIN_MARGIN_S), (
        "a 3s server keepalive is below the client's 4.5s and MUST fail the "
        "invariant — if this passes, the guard above is reading a constant")


def test_the_documented_values_match_the_code(monkeypatch):
    """``docs/api.md`` §1.3 publishes both numbers. If the code's default moves
    and the table does not, the client side is pinned to a stale figure."""
    from tests.wire_contract import API_DOC
    doc = API_DOC.read_text(encoding="utf-8")
    assert "### 1.3 The keepalive ordering invariant" in doc
    server = _server_keepalive(monkeypatch)
    assert f"**{int(server)}s**" in doc, (
        f"docs/api.md does not publish the server keepalive default of {server}s")
    assert f"**{CLIENT_KEEPALIVE_EXPIRY_S}s**" in doc, (
        "docs/api.md does not publish the client keepalive of "
        f"{CLIENT_KEEPALIVE_EXPIRY_S}s")
