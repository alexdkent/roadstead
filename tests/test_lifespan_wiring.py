"""The ASGI lifespan wiring in ``build_app`` is exercised by nothing else.

Every other test that needs a running service calls ``svc.startup()`` and
``svc.shutdown()`` directly and drives the app through ``httpx.ASGITransport``,
which does **not** run the lifespan protocol. So the one line that connects the
service's lifecycle to the server's had no coverage at all: unhook it and the
whole suite still passes, while the real process would serve requests against a
service that never started.

That mattered on 2026-08-31, when ``on_startup=``/``on_shutdown=`` were replaced
by ``lifespan=`` to lift the load-bearing ``starlette<1.0`` pin. These drive the
raw ASGI lifespan protocol — no ``TestClient``, so no second thread and no
second event loop, which keeps the package's single-loop invariant intact.
"""
from __future__ import annotations

import asyncio
import tempfile

import pytest

from roadstead.__main__ import build_app
from roadstead.config import ProxyConfig

_SCOPE = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}}


def _config(tmp: str) -> ProxyConfig:
    cfg = ProxyConfig(queue_db_path=f"{tmp}/q.db")
    cfg.poller_interval_s = 3600  # no backend sockets here; don't let it race us
    return cfg


class _LifespanDriver:
    """Minimal ASGI server: sends the two lifespan events, records the replies."""

    def __init__(self, app):
        self._app = app
        self._inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self._replied = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _receive(self) -> dict:
        return await self._inbox.get()

    async def _send(self, message: dict) -> None:
        self.sent.append(message)
        self._replied.set()

    async def startup(self) -> dict:
        self._task = asyncio.create_task(
            self._app(_SCOPE, self._receive, self._send))
        self._replied.clear()
        await self._inbox.put({"type": "lifespan.startup"})
        await asyncio.wait_for(self._replied.wait(), timeout=30)
        return self.sent[-1]

    async def shutdown(self) -> dict:
        self._replied.clear()
        await self._inbox.put({"type": "lifespan.shutdown"})
        # The app coroutine re-raises a failed startup/shutdown after replying,
        # which is the server's cue to abort — swallow it, the reply is the fact
        # under test.
        await asyncio.wait_for(
            asyncio.gather(self._task, return_exceptions=True), timeout=90)
        return self.sent[-1]


async def test_lifespan_runs_startup_then_shutdown():
    """The happy path: one startup before serving, one drain on shutdown."""
    with tempfile.TemporaryDirectory() as tmp:
        app = build_app(_config(tmp))
        svc = app.state.proxy_service

        async def _healthy(ep_cfg):
            return True
        svc._backend.probe_health = _healthy

        calls: list[str] = []
        real_startup, real_shutdown = svc.startup, svc.shutdown

        async def spy_startup():
            calls.append("startup")
            await real_startup()

        async def spy_shutdown():
            calls.append("shutdown")
            await real_shutdown()

        svc.startup, svc.shutdown = spy_startup, spy_shutdown

        driver = _LifespanDriver(app)
        assert (await driver.startup())["type"] == "lifespan.startup.complete"
        assert calls == ["startup"]
        # Not just the spy: startup's real effect is a running scheduler loop.
        assert svc._scheduler_task is not None
        assert not svc._scheduler_task.done()

        assert (await driver.shutdown())["type"] == "lifespan.shutdown.complete"
        assert calls == ["startup", "shutdown"]
        # ``shutdown`` requests cancellation of the loop tasks without awaiting
        # them (only in-flight *dispatches* get the bounded drain), so the task
        # is typically still in the "cancelling" state when shutdown returns.
        assert svc._scheduler_task.cancelling() > 0 or svc._scheduler_task.done()


async def test_failed_startup_does_not_run_shutdown():
    """A service that never started must not be drained.

    ``on_shutdown`` handlers never ran when ``on_startup`` raised, and the
    ``try``/``finally`` that replaced them is placed to preserve that: entered
    only after ``svc.startup()`` returns. Draining a half-built service would
    close a queue DB whose writer thread was never started.
    """
    with tempfile.TemporaryDirectory() as tmp:
        app = build_app(_config(tmp))
        svc = app.state.proxy_service

        drained = False

        async def boom():
            raise RuntimeError("startup exploded")

        async def spy_shutdown():
            nonlocal drained
            drained = True

        svc.startup, svc.shutdown = boom, spy_shutdown

        driver = _LifespanDriver(app)
        reply = await driver.startup()
        assert reply["type"] == "lifespan.startup.failed"
        assert "startup exploded" in reply.get("message", "")
        assert drained is False

        # Tidy up the app task, which re-raises after replying.
        with pytest.raises(BaseException):
            await asyncio.wait_for(driver._task, timeout=10)

        # Nothing was left running to leak into the next test.
        assert svc._scheduler_task is None or svc._scheduler_task.done()
