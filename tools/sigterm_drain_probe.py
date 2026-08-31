#!/usr/bin/env python3
"""Measure what SIGTERM actually costs — the experiment that settled the
shutdown-signal question on 2026-08-31.

    python tools/sigterm_drain_probe.py

Off the default test path on purpose: it spawns real processes, sends real
signals, and the slowest scenario takes ~80 seconds. It is here so the
measurement can be *re-run* rather than merely cited — the finding it produced
is recorded in ``docs/ledger.md`` and summarised in ``CLAUDE.md``.

The question it answers (``docs/handoff.md``, open question 1): the origin
project's knowledge layer contradicted itself on whether SIGTERM hangs behind
slow in-flight requests, and it becomes load-bearing the day this is
containerised, because ``docker stop`` sends SIGTERM and hard-kills after 10s.

What it does: runs the fake backend in *this* process (so faults stay steerable
from here) and the real proxy in a *child* process (so it can receive a real
signal), parks a dispatch inside the backend for a controlled duration, sends
SIGTERM, and times the exit.

Reading the result: the two shutdown budgets are **serial, not nested**.
uvicorn's ``timeout_graceful_shutdown`` bounds the in-flight HTTP *connections*;
only once it expires does uvicorn send ``lifespan.shutdown``, at which point
``ProxyService.shutdown``'s own ``_DRAIN_DEADLINE_S`` drain *begins* — and
uvicorn does not bound the lifespan shutdown at all. Worst case is therefore
their SUM plus the close tail, not the larger of the two.
"""
from __future__ import annotations

import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roadstead.service import _DRAIN_DEADLINE_S  # noqa: E402
from roadstead.testing import FAULT_TIMEOUT, FakeBackend, FakeBackendServer  # noqa: E402

# The child is written out at run time rather than kept as a second file: it is
# not importable code, it is the experiment's subject.
_CHILD_SRC = textwrap.dedent('''
    import dataclasses, logging, os, sys
    import uvicorn
    from roadstead.config import ProxyConfig
    from roadstead.__main__ import build_app, PROXY_SERVER_KEEPALIVE_S
    from roadstead.service import _DRAIN_DEADLINE_S

    backend_port, listen_port, queue_db = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    cfg = ProxyConfig(queue_db_path=queue_db)
    cfg.endpoints = {
        name: dataclasses.replace(
            ep, host="127.0.0.1", port=backend_port,
            max_slots=ep.max_slots or 4,
            context_per_slot=ep.context_per_slot or 8192,
        )
        for name, ep in cfg.endpoints.items()
    }
    cfg.poller_interval_s = 0.5

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"[child] pid={os.getpid()} drain={_DRAIN_DEADLINE_S} "
          f"graceful={int(_DRAIN_DEADLINE_S) + 18}", file=sys.stderr, flush=True)

    uvicorn.run(
        build_app(cfg), host="127.0.0.1", port=listen_port,
        log_level="info", access_log=False,
        timeout_graceful_shutdown=int(_DRAIN_DEADLINE_S) + 18,
        timeout_keep_alive=PROXY_SERVER_KEEPALIVE_S,
    )
''')

SCENARIOS = [
    ("S1 idle", 0.0, "no in-flight work — the container-restart common case"),
    ("S2 short in-flight", 8.0, "one dispatch finishing INSIDE the drain budget"),
    ("S3 outlasts drain", 120.0, "one dispatch outlasting the drain — cancelled"),
]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_healthy(port: int, proc: subprocess.Popen, timeout: float = 40.0) -> float:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"child died during startup rc={proc.returncode}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0).status_code < 500:
                return time.monotonic() - t0
        except Exception:  # noqa: BLE001 — not up yet
            pass
        time.sleep(0.1)
    raise RuntimeError("child never became healthy")


def _persisted(db: str) -> dict:
    """What survived the drain. This is what a SIGKILL would have lost."""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return {
                t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("proxy_agent_budgets", "proxy_completions")
            }
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_scenario(name: str, hold_s: float, note: str) -> dict:
    fake = FakeBackendServer(FakeBackend()).start()
    tmp = tempfile.mkdtemp(prefix="roadstead-sigterm-")
    db = os.path.join(tmp, "queue.db")
    child_py = os.path.join(tmp, "proxy_child.py")
    with open(child_py, "w") as fh:
        fh.write(_CHILD_SRC)
    port = _free_port()
    log = open(os.path.join(tmp, "child.log"), "w+")
    proc = subprocess.Popen(
        [sys.executable, child_py, str(fake.port), str(port), db],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    result: dict = {"scenario": name, "note": note, "hold_s": hold_s}
    caller: dict = {}
    thread: threading.Thread | None = None
    try:
        result["startup_s"] = round(_wait_healthy(port, proc), 2)

        if hold_s > 0:
            # Park a dispatch inside the backend. The fault is set on the fake's
            # controller, not via a header — the proxy forwards only
            # X-Request-ID, so a header fault would never reach the backend.
            fake.controller.set_fault(FAULT_TIMEOUT, hold_s)

            def _fire() -> None:
                t0 = time.monotonic()
                try:
                    r = httpx.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"model": "chat",
                              "messages": [{"role": "user", "content": "hold"}],
                              "max_tokens": 16, "stream": False, "timeout_s": 600},
                        timeout=600.0)
                    caller["status"] = r.status_code
                    caller["body"] = r.text[:160]
                except Exception as exc:  # noqa: BLE001
                    caller["status"] = "transport-error"
                    caller["body"] = f"{type(exc).__name__}: {exc}"
                caller["elapsed_s"] = round(time.monotonic() - t0, 2)

            thread = threading.Thread(target=_fire, daemon=True)
            thread.start()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not fake.controller.requests:
                time.sleep(0.1)
            result["reached_backend"] = bool(fake.controller.requests)
            time.sleep(0.5)

        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=180)
            result["exit_s"] = round(time.monotonic() - t0, 2)
            result["returncode"] = proc.returncode
            result["hard_killed"] = False
        except subprocess.TimeoutExpired:
            result["exit_s"] = None
            result["hard_killed"] = True
            proc.kill()
            proc.wait(timeout=10)

        if thread is not None:
            thread.join(timeout=30)
            result["caller"] = caller
        result["persisted"] = _persisted(db)

        log.flush()
        log.seek(0)
        result["shutdown_log"] = [
            ln.rstrip() for ln in log.read().splitlines()
            if any(k in ln.lower() for k in
                   ("drain", "straggler", "shutting down", "shutdown", "cancel"))
        ][:14]
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
        fake.stop()
    return result


def main() -> None:
    graceful = int(_DRAIN_DEADLINE_S) + 18
    print(f"app drain deadline   _DRAIN_DEADLINE_S = {_DRAIN_DEADLINE_S}s")
    print(f"uvicorn graceful     timeout_graceful_shutdown = {graceful}s\n")

    results = []
    for name, hold, note in SCENARIOS:
        print(f">>> {name}: {note}", flush=True)
        r = run_scenario(name, hold, note)
        results.append(r)
        for key in ("startup_s", "reached_backend", "exit_s", "hard_killed",
                    "caller", "persisted"):
            if key in r:
                print(f"    {key}: {r[key]}")
        for ln in r["shutdown_log"]:
            print(f"    | {ln}")
        print(flush=True)

    print("===== SUMMARY =====")
    print(f"{'scenario':<24}{'SIGTERM→exit':>14}{'hard-killed':>13}{'persisted':>44}")
    for r in results:
        print(f"{r['scenario']:<24}{str(r['exit_s']) + 's':>14}"
              f"{str(r['hard_killed']):>13}{str(r.get('persisted')):>44}")
    worst = max((r["exit_s"] or 0) for r in results)
    print(f"\nworst observed SIGTERM→exit: {worst}s "
          f"(uvicorn {graceful}s THEN app drain {_DRAIN_DEADLINE_S}s — serial, not nested)")
    print(f"→ a container stop-grace-period below ~{int(worst) + 12}s truncates the drain.")


if __name__ == "__main__":
    main()
