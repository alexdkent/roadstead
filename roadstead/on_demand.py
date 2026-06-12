"""On-demand backend lifecycle — for endpoints whose model is loaded lazily
under the anvil GPU-slot dispatcher lease, then idle-unloaded.

An endpoint opts in via ``EndpointConfig.on_demand=True`` +
``.dispatcher_capability``. Before a request to such an endpoint dispatches,
``ensure_loaded`` acquires (or confirms) the **anvil dispatcher** lease — a
single GPU slot shared FIFO with the other on-demand services (imagegen,
diarize, lyrics, …). Acquiring loads the model (and blocks behind any current
slot holder). A background loop heartbeats each held lease and releases it once
no request has been in-flight for ``hold_idle_s``; the dispatcher then keeps the
model warm for its own ``keep_warm_sec`` and idle-unloads it (or evicts it the
moment another service claims the slot), returning the GPU memory to the pool.

Generic by design — any number of on-demand endpoints are managed from the one
manager, keyed by endpoint name. Adding another on-demand model is just an
``EndpointConfig`` flag + a dispatcher registry entry.

Robustness:
  - Dispatcher unreachable / load failure → ``ensure_loaded`` raises
    ``OnDemandUnavailable`` so the caller fails the request cleanly (the proxy
    surfaces a deferrable error) instead of hanging or dispatching into a dead
    backend.
  - Concurrent ``ensure_loaded`` for one endpoint share a single ``/ensure``
    (per-endpoint lock); the load happens once, the others wait it out.
  - In-flight leak guard: a held lease whose in-flight count is stuck past a
    hard ceiling (well beyond any single request's timeout) is force-released,
    so a lost completion can never pin the shared slot forever.

FUTURE (noted, not built): an explicit per-request "last call — unload when
done" signal carried from the consuming agent through the proxy. On that
request's completion the manager would release the lease immediately instead of
waiting out ``hold_idle_s`` — turning the idle timeout into a fallback rather
than the primary release path. ``request_done`` already centralizes the
completion hook where that early-release would attach.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_DISPATCHER_URL = os.environ.get(
    "ANVIL_DISPATCHER_URL", "http://10.0.0.3:9201"
)

# Lease protocol params — mirror the imagegen lease client.
_LEASE_TTL_S = 300.0          # dispatcher lease TTL; we heartbeat to extend.
_HEARTBEAT_S = 60.0           # 5x headroom under the TTL.
_HOLD_IDLE_S = 60.0           # release the lease this long after the last in-flight
                             # request (short → free the slot for other services;
                             # the dispatcher's keep_warm_sec governs model unload).
_ENSURE_TIMEOUT_S = 450.0     # cold load can take minutes (vLLM NVFP4 + torch.compile).
_LOOP_PERIOD_S = 10.0
# In-flight can never legitimately exceed the max request lifetime (the caller's
# timeout, ≤ ~900s). A stuck count past this ceiling is a leaked completion.
_STUCK_INFLIGHT_S = 1200.0


class OnDemandUnavailable(Exception):
    """The on-demand backend could not be made ready (dispatcher down / load failed)."""


@dataclass
class _EndpointState:
    capability: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lease_id: str | None = None
    inflight: int = 0
    last_activity: float = 0.0
    last_heartbeat: float = 0.0


class OnDemandManager:
    """Manages the dispatcher lease lifecycle for all on-demand endpoints."""

    def __init__(
        self,
        endpoints: dict,
        *,
        dispatcher_url: str = _DEFAULT_DISPATCHER_URL,
        hold_idle_s: float = _HOLD_IDLE_S,
    ) -> None:
        self._url = dispatcher_url.rstrip("/")
        self._hold_idle_s = hold_idle_s
        self._states: dict[str, _EndpointState] = {
            name: _EndpointState(capability=str(cfg.dispatcher_capability))
            for name, cfg in endpoints.items()
            if getattr(cfg, "on_demand", False)
            and getattr(cfg, "dispatcher_capability", "")
        }
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=_ENSURE_TIMEOUT_S, write=10.0, pool=5.0)
        )
        self._loop_task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def active(self) -> bool:
        return bool(self._states)

    def manages(self, endpoint: str) -> bool:
        return endpoint in self._states

    def is_loaded(self, endpoint: str) -> bool:
        """True if we currently hold a lease for ``endpoint`` (model resident)."""
        st = self._states.get(endpoint)
        return st is not None and st.lease_id is not None

    # ----- request path -----

    async def ensure_loaded(self, endpoint: str) -> None:
        """Acquire/confirm the dispatcher lease so the model is resident before
        the request dispatches. Counts the request in-flight. Idempotent while a
        lease is held. Raises ``OnDemandUnavailable`` on dispatcher failure."""
        st = self._states.get(endpoint)
        if st is None:
            return
        async with st.lock:
            st.last_activity = time.monotonic()
            st.inflight += 1
            if st.lease_id is not None:
                return
            try:
                r = await self._client.post(
                    f"{self._url}/ensure",
                    json={"capability": st.capability, "lease_ttl_s": _LEASE_TTL_S},
                    timeout=_ENSURE_TIMEOUT_S,
                )
            except httpx.HTTPError as exc:
                st.inflight = max(0, st.inflight - 1)
                raise OnDemandUnavailable(
                    f"dispatcher /ensure unreachable for {endpoint}: {exc}"
                ) from exc
            if r.status_code != 200:
                st.inflight = max(0, st.inflight - 1)
                raise OnDemandUnavailable(
                    f"dispatcher /ensure {r.status_code} for {endpoint}: {r.text[:200]}"
                )
            data = r.json()
            st.lease_id = data.get("lease_id")
            st.last_heartbeat = time.monotonic()
            logger.info(
                "on_demand: acquired lease %s for %s (capability=%s)",
                st.lease_id, endpoint, st.capability,
            )

    def request_done(self, endpoint: str) -> None:
        """A request to ``endpoint`` reached a terminal outcome. Decrement
        in-flight + bump activity so the idle watchdog can eventually release."""
        st = self._states.get(endpoint)
        if st is None:
            return
        st.inflight = max(0, st.inflight - 1)
        st.last_activity = time.monotonic()

    # ----- background lifecycle -----

    async def start(self) -> None:
        if self._states and self._loop_task is None:
            self._loop_task = asyncio.create_task(self._loop(), name="on_demand.loop")
            logger.info(
                "on_demand: managing %s via dispatcher %s",
                sorted(self._states), self._url,
            )

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=_LOOP_PERIOD_S)
                return
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            for endpoint, st in self._states.items():
                if st.lease_id is None:
                    continue
                # Leak guard: a stuck in-flight count must never pin the slot.
                if st.inflight > 0 and (now - st.last_activity) >= _STUCK_INFLIGHT_S:
                    logger.warning(
                        "on_demand: %s inflight=%d stuck %.0fs — force-releasing",
                        endpoint, st.inflight, now - st.last_activity,
                    )
                    st.inflight = 0
                if now - st.last_heartbeat >= _HEARTBEAT_S:
                    await self._heartbeat(endpoint, st)
                if st.lease_id is not None and st.inflight <= 0 and (
                    now - st.last_activity
                ) >= self._hold_idle_s:
                    await self._release(endpoint, st)

    async def _heartbeat(self, endpoint: str, st: _EndpointState) -> None:
        lid = st.lease_id
        if not lid:
            return
        try:
            r = await self._client.post(
                f"{self._url}/heartbeat",
                json={"lease_id": lid, "extend_s": _LEASE_TTL_S},
                timeout=10.0,
            )
            if r.status_code == 200:
                st.last_heartbeat = time.monotonic()
            else:
                # Lease lost/expired (404) — drop our handle so the next request
                # re-acquires (and reloads if the dispatcher evicted the model).
                logger.warning(
                    "on_demand: heartbeat %s -> %s; dropping lease handle",
                    endpoint, r.status_code,
                )
                st.lease_id = None
        except httpx.HTTPError as exc:
            logger.warning("on_demand: heartbeat %s failed: %s", endpoint, exc)

    async def _release(self, endpoint: str, st: _EndpointState) -> None:
        lid = st.lease_id
        st.lease_id = None
        if not lid:
            return
        try:
            await self._client.post(
                f"{self._url}/release", json={"lease_id": lid}, timeout=10.0
            )
            logger.info("on_demand: released lease %s for %s (idle)", lid, endpoint)
        except httpx.HTTPError as exc:
            logger.warning(
                "on_demand: release %s failed (slot will TTL-expire): %s",
                endpoint, exc,
            )

    async def close(self) -> None:
        self._stop.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
        for endpoint, st in list(self._states.items()):
            if st.lease_id:
                await self._release(endpoint, st)
        await self._client.aclose()
