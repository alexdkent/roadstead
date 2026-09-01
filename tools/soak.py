#!/usr/bin/env python3
"""Run the proxy under sustained load for minutes and watch what accumulates.

    python tools/soak.py                        # 60s, 16 concurrent
    python tools/soak.py --seconds 600 --concurrency 32
    python tools/soak.py --seconds 300 --stream-fraction 0.5

Off the default test path on purpose. ``tests/e2e/test_soak.py`` is the
*assertion* — bounded to a few seconds, on every run, and it answers "does a
second thread touch single-loop state, and does the accounting balance". This is
the *experiment*, and it answers a different question that no assertion can:
**what accumulates over time.**

Those are genuinely different failures. Contention shows up in seconds; a slow
leak — RSS climbing, a WAL that never checkpoints, connection-pool growth, a
budget map that gains an agent per request and never prunes — shows up in
minutes and is invisible to a test that finishes before it starts.

## What it reports

Per interval, and again as a summary:

* **RSS**, and the slope over the run. The number to look at is the slope, not
  the peak: Python's allocator does not return everything, so a run that rises
  and plateaus is healthy and one that rises linearly is not.
* **In-flight, queue depth, and the peak of each** — the proof that load
  actually overlapped, and the first place a leaked slot shows.
* **DRR budget map size**, which should track distinct callers and not requests.
* **The spend ledger's row count and request total.**
* **queue.db size and WAL size.** A WAL that grows without bound means
  checkpointing is not keeping up with the single writer.
* **Latency percentiles and the status-code histogram.** Under saturation a 429
  or a 503 is a correct answer; a 504 against a fake backend that replies in
  milliseconds is a leaked slot, and a 500 is an unhandled fault.

🚨 **The loop-affinity guard runs here too**, armed exactly as the e2e soak arms
it. A violation that only appears after minutes of load is precisely the one a
bounded test cannot reach.

## What it is not

Not a benchmark. The backend is ``roadstead.testing``'s fake, so the throughput
number says how fast the *proxy* is at scheduling and says nothing about any
model. Comparing two runs of this tool is meaningful; comparing it to a real
fleet is not.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import random
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from roadstead.__main__ import build_app  # noqa: E402
from roadstead.config import ProxyConfig  # noqa: E402
from roadstead.testing import FakeBackend, FakeBackendServer  # noqa: E402


def _arm_affinity(state):
    """Borrow the affinity recorder from ``tests/``.

    It lives there because it is scaffolding rather than product (see its
    docstring), and this is the one place a tool reaches into the test tree — a
    second copy of the invariant would be a second thing to keep correct.

    🚨 Imported INSIDE the function, not at module scope. ``tools/`` is swept by
    ``tests/test_tools_are_importable.py``, which imports every script here; a
    module-level ``sys.path.insert`` would put ``tests/`` on the path of the
    pytest process itself, where it could shadow a module for every test that
    ran afterwards.
    """
    tests_dir = str(Path(__file__).resolve().parents[1] / "tests")
    if tests_dir not in sys.path:
        sys.path.append(tests_dir)
    from loop_affinity import arm
    return arm(state)


def _rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024


def _repointed(host: str, port: int, db: str) -> ProxyConfig:
    base = ProxyConfig(queue_db_path=db)
    import dataclasses
    base.endpoints = {
        cls: dataclasses.replace(
            ep, host=host, port=port,
            max_slots=ep.max_slots or 4,
            context_per_slot=ep.context_per_slot or 8192)
        for cls, ep in base.endpoints.items()
    }
    return base


def _body(rng: random.Random, *, stream: bool, deadline_s: float) -> dict:
    intent = rng.choice(["chat", "reasoning", "fast-chat", "long-context"])
    body: dict = {
        "priority": rng.choice(
            ["P1_TURN_SUPPORT", "P2_POST_TURN", "P3_INGESTION"]),
        "call_site": f"soak.{rng.randrange(6)}",
        "deadline_s": deadline_s,
        "payload": {
            "messages": [{"role": "user",
                          "content": "soak " + "x" * rng.randrange(20, 400)}],
            "max_tokens": rng.randrange(8, 64),
            "stream": stream,
        },
    }
    if rng.random() < 0.5:
        body["intent"] = intent
    else:
        body["model"] = rng.choice(["tier1", "tier2", "tier3"])
    return body


async def _one(client, body, stream: bool) -> tuple[int, float]:
    t0 = time.monotonic()
    try:
        if stream:
            async with client.stream("POST", "/rs/v1/chat", json=body) as resp:
                async for _ in resp.aiter_lines():
                    pass
                code = resp.status_code
        else:
            code = (await client.post("/rs/v1/chat", json=body)).status_code
    except Exception:  # noqa: BLE001 — a transport fault is a result, not a crash
        code = 0
    return code, (time.monotonic() - t0) * 1000.0


async def run(args) -> int:
    rng = random.Random(args.seed)
    srv = FakeBackendServer(FakeBackend()).start()
    tmp = tempfile.TemporaryDirectory()
    db = f"{tmp.name}/queue.db"
    app = build_app(_repointed(srv.host, srv.port, db))
    svc = app.state.proxy_service

    async def _healthy(ep_cfg):
        return True
    svc._backend.probe_health = _healthy

    await svc.startup()
    affinity = _arm_affinity(svc._state)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False,
                                    client=("127.0.0.1", 41999))
    client = httpx.AsyncClient(transport=transport, base_url="http://proxy",
                               timeout=args.deadline + 30)

    statuses: dict[int, int] = {}
    latencies: list[float] = []
    peak_inflight = 0
    peak_queued = 0
    samples: list[tuple[float, float, int, int]] = []
    started = time.monotonic()
    stop = started + args.seconds

    async def sampler():
        nonlocal peak_inflight, peak_queued
        while True:
            inflight = sum(svc._scheduler.endpoint_snapshot(ep)["in_flight"]
                           for ep in svc._state.config.endpoints)
            queued = sum(svc._scheduler.endpoint_snapshot(ep)["queued"]
                         for ep in svc._state.config.endpoints)
            peak_inflight = max(peak_inflight, inflight)
            peak_queued = max(peak_queued, queued)
            await asyncio.sleep(0.02)

    async def reporter():
        while True:
            await asyncio.sleep(args.interval)
            elapsed = time.monotonic() - started
            rss = _rss_mb()
            samples.append((elapsed, rss, peak_inflight, peak_queued))
            wal = Path(db + "-wal")
            print(f"[{elapsed:6.1f}s] rss={rss:7.1f}MB "
                  f"done={sum(statuses.values()):6d} "
                  f"inflight_peak={peak_inflight:3d} queued_peak={peak_queued:4d} "
                  f"budgets={len(svc._state.budget_mgr.agents):3d} "
                  f"ledger={len(svc._state.spend):3d} "
                  f"db={Path(db).stat().st_size / 1e6:6.2f}MB "
                  f"wal={(wal.stat().st_size / 1e6 if wal.exists() else 0):6.2f}MB",
                  flush=True)

    watchers = [asyncio.create_task(sampler()), asyncio.create_task(reporter())]
    inflight: set[asyncio.Task] = set()
    try:
        while time.monotonic() < stop:
            while len(inflight) < args.concurrency:
                stream = rng.random() < args.stream_fraction
                task = asyncio.create_task(
                    _one(client, _body(rng, stream=stream,
                                       deadline_s=args.deadline), stream))
                inflight.add(task)
            done, inflight = await asyncio.wait(
                inflight, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                code, ms = t.result()
                statuses[code] = statuses.get(code, 0) + 1
                latencies.append(ms)
        if inflight:
            for t in await asyncio.gather(*inflight, return_exceptions=True):
                if isinstance(t, tuple):
                    statuses[t[0]] = statuses.get(t[0], 0) + 1
                    latencies.append(t[1])
    finally:
        for w in watchers:
            w.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await w
        affinity.disarm()
        await client.aclose()
        await svc.shutdown()
        srv.stop()

    total = sum(statuses.values())
    lat = sorted(latencies)

    def pct(p: float) -> float:
        return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else 0.0

    print("\n" + "=" * 72)
    print(f"requests   {total} in {args.seconds}s "
          f"({total / max(args.seconds, 1):.1f}/s), concurrency {args.concurrency}")
    print(f"statuses   {dict(sorted(statuses.items()))}")
    print(f"latency    p50={pct(0.5):.0f}ms p95={pct(0.95):.0f}ms "
          f"p99={pct(0.99):.0f}ms max={lat[-1] if lat else 0:.0f}ms")
    print(f"peak       inflight={peak_inflight} queued={peak_queued}")
    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        slope = (last[1] - first[1]) / max(last[0] - first[0], 1e-9) * 60.0
        print(f"rss        {first[1]:.1f}MB -> {last[1]:.1f}MB "
              f"({slope:+.2f} MB/min)")
        print("           🚨 the SLOPE is the number. A run that rises and "
              "plateaus is healthy; linear growth is not.")
    print(f"budgets    {len(svc._state.budget_mgr.agents)} agents "
          f"(should track distinct CALLERS, never requests)")
    print(f"mutations  {affinity.total_mutations} recorded across "
          f"{affinity.armed_methods} armed methods")

    problems = affinity.violations()
    bad_status = {c for c in statuses if c not in (200, 429, 503)}
    print("=" * 72)
    if problems:
        print("🚨 CONCURRENCY INVARIANT VIOLATED — single-loop state was "
              "mutated from more than one thread:")
        for p in problems:
            print(f"   - {p}")
    if bad_status:
        print(f"🚨 unexpected statuses {sorted(bad_status)} — 504 against a "
              f"fake backend means a LEAKED SLOT; 500 an unhandled fault; "
              f"0 a transport error.")
    if not problems and not bad_status:
        print("✅ single-threaded throughout, every request answered.")
    return 1 if (problems or bad_status) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--stream-fraction", type=float, default=0.25)
    ap.add_argument("--deadline", type=float, default=30.0,
                    help="per-request deadline (s); short on purpose, so a "
                         "leaked slot surfaces as a 504 rather than a hang")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
