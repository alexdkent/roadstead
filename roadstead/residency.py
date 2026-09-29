"""Residency — is an endpoint's model LOADED right now, according to the host
dispatcher that decides that. READ-ONLY, and deliberately not a lifecycle.

Some backends are not always resident: a host-side dispatcher loads and evicts
them as GPU-slot leases come and go. From the proxy an evicted backend and a
crashed one look the same — nothing answers — and reading them the same is
wrong in both directions: a crash is a page, an eviction the dispatcher chose is
not. This module answers the one question that tells them apart.

An endpoint opts in by declaring ``policy.residency_tenant: <name>`` — the key
the dispatcher files it under in its ``GET /status`` ``intended_state`` map.
The name→tenant mapping lives in the deployment's catalog, never here.

🚨 **This never loads anything.** Waking a backend is ``on_demand``'s job
(``OnDemandManager.ensure_loaded``) and it is a decision with a blast radius: on
a shared GPU, loading one model can evict another. A deployment may forbid a
request from triggering it, and the way it says so is by not declaring the
endpoint to the manager. Nothing here can take that decision away.

🚨 **``intended_state`` is what the dispatcher INTENDS, not what is running.**
A tenant intended ``resident`` whose process died reads ``resident`` here, so a
residency of ``resident`` never makes an endpoint healthy — health still needs
the endpoint to answer a probe. Residency only ever EXPLAINS a failed probe
(``evicted`` -> expected), it never overrides one.

🚨 **The answer is three-valued and ``unknown`` is never folded into either
other value.** No dispatcher URL configured, a dispatcher that does not answer,
a body that is not the expected shape, a tenant the dispatcher does not list, a
word this module does not know, a reading older than ``_STALE_S`` — every one
is ``unknown``. A confident ``evicted`` derived from a failed read would turn a
real crash into an "expected" one, which is the exact silence this exists to
end.
"""
from __future__ import annotations

import logging
import time

import httpx

from .on_demand import _DEFAULT_DISPATCHER_URL

logger = logging.getLogger(__name__)

RESIDENT = "resident"
EVICTED = "evicted"
UNKNOWN = "unknown"

#: The poller refreshes once per pass (10s by default). A reading this old means
#: the refresh loop stopped, and an old ``evicted`` must not keep excusing a
#: failure long after the dispatcher may have changed its mind.
_STALE_S = 60.0
_READ_TIMEOUT_S = 3.0


class ResidencyReader:
    """Caches the dispatcher's ``intended_state`` map and answers per endpoint."""

    def __init__(self, endpoints: dict, *,
                 dispatcher_url: str = _DEFAULT_DISPATCHER_URL) -> None:
        self._url = dispatcher_url.rstrip("/")
        self._tenants: dict[str, str] = {
            name: str(cfg.residency_tenant).strip()
            for name, cfg in endpoints.items()
            if str(getattr(cfg, "residency_tenant", "") or "").strip()
        }
        self._client: httpx.AsyncClient | None = None
        self._intent: dict[str, str] | None = None
        self._read_at: float = 0.0
        self._last_error: str = ""

    @property
    def tracks(self) -> bool:
        """True iff at least one endpoint declares a tenant. Gates the refresh
        so a deployment that declares nothing makes no dispatcher call at all."""
        return bool(self._tenants)

    def declares(self, endpoint: str) -> bool:
        return endpoint in self._tenants

    async def refresh(self) -> None:
        """One read of the dispatcher. Never raises; any failure leaves the
        cache EMPTY (unknown), not at its last value."""
        if not self._tenants:
            return
        if not self._url:
            self._intent, self._last_error = None, "no dispatcher URL configured"
            return
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(_READ_TIMEOUT_S))
        try:
            r = await self._client.get(f"{self._url}/status")
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")
            intent = r.json().get("intended_state")
            if not isinstance(intent, dict):
                raise ValueError("no intended_state map in the response")
            self._intent = {str(k): str(v) for k, v in intent.items()}
            self._read_at = time.monotonic()
            self._last_error = ""
        except Exception as exc:  # noqa: BLE001 — cannot tell is a value, not a fault
            self._intent = None
            msg = f"{type(exc).__name__}: {exc}"
            if msg != self._last_error:
                logger.warning(
                    "residency: dispatcher read failed (%s) — declared endpoints "
                    "read `unknown` until it answers", msg)
            self._last_error = msg

    def state(self, endpoint: str) -> str | None:
        """``resident`` | ``evicted`` | ``unknown``, or ``None`` when the
        endpoint declares no tenant (the field is then absent, not unknown)."""
        tenant = self._tenants.get(endpoint)
        if tenant is None:
            return None
        if self._intent is None or (time.monotonic() - self._read_at) > _STALE_S:
            return UNKNOWN
        word = self._intent.get(tenant)
        # "unmanaged" (the dispatcher does not manage this tenant's residency)
        # and any word not known here are honest non-answers.
        return word if word in (RESIDENT, EVICTED) else UNKNOWN

    def age_s(self) -> float | None:
        return round(time.monotonic() - self._read_at, 1) if self._intent is not None else None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
