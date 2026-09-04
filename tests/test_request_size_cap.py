"""P2 audit hardening — a request body size cap enforced at the ASGI layer,
BEFORE any handler calls ``request.json()``.

Every door on this proxy parses its body with no upper bound on how large it
may be — a vision chat payload, an admin write, all of them. `__main__.
RequestSizeLimitMiddleware` refuses an oversized body at the ASGI layer, so
nothing downstream (the scheduler, the queue, a backend dispatch) ever sees
the bytes. These assert against the REAL ASGI app via `httpx.ASGITransport`,
same house rule as `tests/test_readyz.py`: route/middleware behaviour is a
SOURCE fact or a TEST fact, never something to probe against the live fleet.
"""

from __future__ import annotations

import tempfile

import httpx
import pytest
import pytest_asyncio

from roadstead.__main__ import build_app
from roadstead.config import DEFAULT_MAX_REQUEST_BYTES, ProxyConfig

_INTERNAL_CLIENT = ("127.0.0.1", 41999)


async def _make_proxy():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ProxyConfig(queue_db_path=f"{tmp}/q.db")
        cfg.poller_interval_s = 3600
        app = build_app(cfg)
        svc = app.state.proxy_service
        await svc.startup()
        transport = httpx.ASGITransport(
            app=app, raise_app_exceptions=False, client=_INTERNAL_CLIENT)
        client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                                   timeout=30.0)
        try:
            yield svc, client
        finally:
            await client.aclose()
            await svc.shutdown()


@pytest_asyncio.fixture
async def proxy(monkeypatch):
    """A tiny cap, set BEFORE `build_app` reads the env — the default (16 MiB)
    would make an oversized-body test slow and memory-heavy for no reason."""
    monkeypatch.setenv("ROADSTEAD_MAX_REQUEST_BYTES", "64")
    async for item in _make_proxy():
        yield item


@pytest_asyncio.fixture
async def proxy_default_cap():
    """The real default (16 MiB) — proves the cap does not get in the way of
    an ordinary small request, which the tiny test cap above can't show."""
    async for item in _make_proxy():
        yield item


def test_the_default_is_sixteen_mebibytes():
    assert DEFAULT_MAX_REQUEST_BYTES == 16 * 1024 * 1024


@pytest.mark.asyncio
async def test_an_oversized_body_is_refused_by_content_length_before_dispatch(proxy):
    svc, client = proxy
    oversized = b"x" * 200  # over the 64-byte test cap
    resp = await client.post(
        "/v1/chat/completions",
        content=oversized,
        headers={"Content-Type": "application/json",
                 "Content-Length": str(len(oversized))})
    assert resp.status_code == 413
    body = resp.json()
    assert body["code"] == "invalid_request_error"
    # Refused before anything was even read as a body: no backend socket
    # exists in this fixture at all, so a 502/504 (rather than this clean
    # 413) would mean the middleware let it through to a dispatch attempt.


@pytest.mark.asyncio
async def test_an_oversized_body_with_no_content_length_is_still_caught(proxy):
    """A caller that lies short, or declares no length at all (chunked), must
    be caught by counting bytes as they actually arrive."""
    svc, client = proxy

    async def _stream():
        for _ in range(10):
            yield b"x" * 20  # 200 bytes total, over the 64-byte cap

    resp = await client.post(
        "/v1/chat/completions",
        content=_stream(),
        headers={"Content-Type": "application/json"})
    assert resp.status_code == 413
    assert resp.json()["code"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_a_body_under_the_cap_is_unaffected(proxy_default_cap):
    svc, client = proxy_default_cap
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]})
    # Refused for an unknown model, not for size — proves the cap did not
    # swallow a legitimate small request.
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_endpoint"


@pytest.mark.asyncio
async def test_get_requests_are_unaffected(proxy_default_cap):
    svc, client = proxy_default_cap
    resp = await client.get("/health")
    assert resp.status_code == 200
