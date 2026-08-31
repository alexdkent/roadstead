"""Roadstead in a container, with a parked in-flight request, ready for SIGTERM.

Measures what `docker stop` actually costs, which is the containerisation half
of the shutdown question in `CLAUDE.md`. `tools/sigterm_drain_probe.py` measured
a raw SIGTERM to a bare process; this measures the thing that will actually
happen in production, where a 10s default sits between SIGTERM and SIGKILL.

🚨 EVERY ENDPOINT IS REPOINTED AT THE IN-PROCESS FAKE. `ProxyConfig()`'s
defaults carry real fleet hosts, and a second proxy pointed at real backends
does not merely borrow capacity — the timeout and cost models LEARN from
measured latency and PERSIST what they learn, so it would contaminate the
production proxy's p99 estimates and DRR slot-second charges, and the
contamination outlives the run. There is no clean rollback for a poisoned
model and nothing alerts on it.

Two independent guards, because config discipline alone is one edit from
failing: the repoint below, and `internal: true` on the compose network so the
container has no route off the host at all.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
import time

import httpx
import uvicorn

from roadstead.__main__ import PROXY_SERVER_KEEPALIVE_S, build_app
from roadstead.config import ProxyConfig
from roadstead.service import _DRAIN_DEADLINE_S
from roadstead.testing import FAULT_TIMEOUT, FakeBackend, FakeBackendServer

HOLD_S = float(os.environ.get("PROBE_HOLD_S", "120"))
LISTEN_PORT = int(os.environ.get("PROBE_PORT", "42161"))

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("probe")

fake = FakeBackendServer(FakeBackend()).start()
log.info("fake backend on %s", fake.url)

cfg = ProxyConfig(queue_db_path="/data/queue.db")
_fleet = {(ep.host, ep.port) for ep in cfg.endpoints.values()}
cfg.endpoints = {
    name: dataclasses.replace(ep, host="127.0.0.1", port=fake.port,
                              max_slots=ep.max_slots or 4,
                              context_per_slot=ep.context_per_slot or 8192)
    for name, ep in cfg.endpoints.items()
}
# Assert the repoint, rather than trusting that it happened.
still_fleet = {(ep.host, ep.port) for ep in cfg.endpoints.values()} & _fleet
assert not still_fleet, f"endpoints still pointing off-box: {still_fleet}"
assert all(ep.host == "127.0.0.1" for ep in cfg.endpoints.values())
log.info("repointed %d endpoints away from %d fleet targets",
         len(cfg.endpoints), len(_fleet))
cfg.poller_interval_s = 0.5

app = build_app(cfg)


def _park_a_request() -> None:
    """Hold one dispatch inside the backend so SIGTERM lands mid-flight."""
    for _ in range(60):
        try:
            if httpx.get(f"http://127.0.0.1:{LISTEN_PORT}/health",
                         timeout=2.0).status_code < 500:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    fake.controller.set_fault(FAULT_TIMEOUT, HOLD_S)
    log.info("PROBE_READY firing a request that parks for %.0fs", HOLD_S)
    try:
        r = httpx.post(
            f"http://127.0.0.1:{LISTEN_PORT}/v1/chat/completions",
            json={"model": "chat", "max_tokens": 16, "stream": False,
                  "timeout_s": 600,
                  "messages": [{"role": "user", "content": "hold"}]},
            timeout=600.0)
        log.info("parked request returned %s", r.status_code)
    except Exception as exc:  # noqa: BLE001
        log.info("parked request ended: %s", type(exc).__name__)


threading.Thread(target=_park_a_request, daemon=True).start()

log.info("PROBE_START drain=%ss uvicorn_graceful=%ss",
         _DRAIN_DEADLINE_S, int(_DRAIN_DEADLINE_S) + 18)
uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="info",
            access_log=False,
            timeout_graceful_shutdown=int(_DRAIN_DEADLINE_S) + 18,
            timeout_keep_alive=PROXY_SERVER_KEEPALIVE_S)
