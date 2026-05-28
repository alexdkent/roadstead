"""Unified test harness for the LLM proxy.

Three modes:
  1. **sim** — discrete-event simulation with synthetic traffic (wraps simulation.py)
  2. **replay** — recorded-replay through the real proxy scheduler
  3. **ab** — A/B backend comparison using recorded corpus

Usage:
    python -m originfleet.llmproxy test sim all_agents_burst
    python -m originfleet.llmproxy test replay --hours 4 --compress 8 --endpoint thinker
    python -m originfleet.llmproxy test ab --hours 4 --endpoint thinker --shadow-host 10.0.0.3 --shadow-port 9084
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import EndpointConfig, ProxyConfig
from .queue import PersistentQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shaping controls
# ---------------------------------------------------------------------------

@dataclass
class ShapingConfig:
    compress: float = 1.0
    multiply: int = 1
    priority_remap: dict[str, str] = field(default_factory=dict)
    endpoint_filter: str | None = None
    call_site_filter: str | None = None


# ---------------------------------------------------------------------------
# A/B comparison result
# ---------------------------------------------------------------------------

@dataclass
class ABResult:
    request_id: str
    call_site: str
    endpoint: str

    primary_duration_s: float = 0.0
    primary_input_tokens: int = 0
    primary_output_tokens: int = 0
    primary_response: dict = field(default_factory=dict)

    shadow_duration_s: float = 0.0
    shadow_input_tokens: int = 0
    shadow_output_tokens: int = 0
    shadow_response: dict = field(default_factory=dict)
    shadow_error: str | None = None

    json_parseable_primary: bool = True
    json_parseable_shadow: bool = True
    token_overlap_jaccard: float = 0.0
    length_ratio: float = 0.0


@dataclass
class ABReport:
    corpus_size: int = 0
    shadow_success_count: int = 0
    shadow_error_count: int = 0

    primary_p50_duration_ms: float = 0.0
    primary_p95_duration_ms: float = 0.0
    shadow_p50_duration_ms: float = 0.0
    shadow_p95_duration_ms: float = 0.0

    primary_decode_tps: float = 0.0
    shadow_decode_tps: float = 0.0

    json_parse_rate_primary: float = 0.0
    json_parse_rate_shadow: float = 0.0
    avg_token_overlap_jaccard: float = 0.0
    avg_length_ratio: float = 0.0

    results: list[ABResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  A/B Comparison Report",
            f"{'='*70}",
            f"  Corpus: {self.corpus_size} requests",
            f"  Shadow: {self.shadow_success_count} ok, {self.shadow_error_count} errors",
            f"",
            f"  {'Metric':<30} {'Primary':>12} {'Shadow':>12}",
            f"  {'-'*30} {'-'*12} {'-'*12}",
            f"  {'Latency p50':<30} {self.primary_p50_duration_ms:>10.0f}ms {self.shadow_p50_duration_ms:>10.0f}ms",
            f"  {'Latency p95':<30} {self.primary_p95_duration_ms:>10.0f}ms {self.shadow_p95_duration_ms:>10.0f}ms",
            f"  {'Decode tok/s':<30} {self.primary_decode_tps:>11.1f} {self.shadow_decode_tps:>11.1f}",
            f"  {'JSON parse rate':<30} {self.json_parse_rate_primary:>11.1f}% {self.json_parse_rate_shadow:>11.1f}%",
            f"  {'Token overlap (Jaccard)':<30} {self.avg_token_overlap_jaccard:>11.3f}",
            f"  {'Length ratio (shadow/primary)':<30} {self.avg_length_ratio:>11.2f}",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------

@dataclass
class ReplayReport:
    corpus_size: int = 0
    replayed: int = 0
    errors: int = 0
    p50_wait_ms: float = 0.0
    p95_wait_ms: float = 0.0
    p50_backend_ms: float = 0.0
    p95_backend_ms: float = 0.0
    throughput_rps: float = 0.0
    baseline_p50_wait_ms: float = 0.0
    baseline_p95_wait_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  Replay Report",
            f"{'='*70}",
            f"  Corpus: {self.corpus_size}, Replayed: {self.replayed}, Errors: {self.errors}",
            f"  Throughput: {self.throughput_rps:.1f} req/s",
            f"",
            f"  {'Metric':<25} {'Replay':>12} {'Baseline':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12}",
            f"  {'Queue wait p50':<25} {self.p50_wait_ms:>10.0f}ms {self.baseline_p50_wait_ms:>10.0f}ms",
            f"  {'Queue wait p95':<25} {self.p95_wait_ms:>10.0f}ms {self.baseline_p95_wait_ms:>10.0f}ms",
            f"  {'Backend p50':<25} {self.p50_backend_ms:>10.0f}ms",
            f"  {'Backend p95':<25} {self.p95_backend_ms:>10.0f}ms",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def _token_set(text: str) -> set[str]:
    return set(text.lower().split())


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _extract_text(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content", "") or ""


def _check_json_parseable(response: dict) -> bool:
    text = _extract_text(response)
    if not text:
        return True
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return "{" not in text and "[" not in text


# ---------------------------------------------------------------------------
# ProxyTestHarness
# ---------------------------------------------------------------------------

class ProxyTestHarness:
    """Unified test harness for the LLM proxy."""

    def __init__(
        self,
        proxy_url: str = "http://127.0.0.1:42161",
        db_path: str | None = None,
    ) -> None:
        self._proxy_url = proxy_url
        self._db = PersistentQueue(db_path) if db_path else None

    # --- Mode 1: Simulation (delegates to existing SimRunner) ---

    def run_sim(self, scenario_name: str, **overrides: Any) -> Any:
        from .simulation import BUILTIN_SCENARIOS as SCENARIOS, SimRunner
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario_name!r}; "
                f"available: {sorted(SCENARIOS)}"
            )
        scenario = SCENARIOS[scenario_name]
        runner = SimRunner(scenario)
        return runner.run()

    # --- Mode 2: Recorded replay through the proxy ---

    async def run_replay(
        self,
        hours: float = 4.0,
        endpoint: str | None = None,
        call_site: str | None = None,
        shaping: ShapingConfig | None = None,
    ) -> ReplayReport:
        if self._db is None:
            raise RuntimeError("replay requires a db_path (proxy queue.db)")
        shaping = shaping or ShapingConfig()

        corpus = self._db.export_corpus(
            hours=hours,
            endpoint=shaping.endpoint_filter or endpoint,
            call_site=shaping.call_site_filter or call_site,
        )
        if not corpus:
            logger.warning("no corpus records found")
            return ReplayReport()

        report = ReplayReport(corpus_size=len(corpus))

        baseline_waits = [r["queue_wait_ms"] for r in corpus]
        report.baseline_p50_wait_ms = _percentile(baseline_waits, 50)
        report.baseline_p95_wait_ms = _percentile(baseline_waits, 95)

        timestamps = [r["completed_at"] for r in corpus if r["completed_at"]]
        if len(timestamps) >= 2:
            time_span = max(timestamps) - min(timestamps)
        else:
            time_span = 1.0

        replay_waits: list[float] = []
        replay_backends: list[float] = []
        errors = 0

        async with httpx.AsyncClient(timeout=360.0) as client:
            t0 = time.monotonic()
            for i, rec in enumerate(corpus):
                for _ in range(shaping.multiply):
                    if rec.get("payload") is None:
                        continue

                    priority = rec.get("priority", 1)
                    if shaping.priority_remap:
                        for old, new in shaping.priority_remap.items():
                            if str(priority) == old:
                                priority = int(new)

                    submit = {
                        "agent_id": rec["agent_id"],
                        "endpoint": rec["endpoint"],
                        "priority": priority,
                        "call_site": rec["call_site"],
                        "payload_type": "chat_completion",
                        "payload": rec["payload"],
                        "timeout_s": 300.0,
                    }

                    try:
                        resp = await client.post(
                            f"{self._proxy_url}/v1/submit",
                            json=submit,
                        )
                        result = resp.json()
                        if result.get("status") == "ok":
                            replay_waits.append(result.get("queue_wait_ms", 0))
                            replay_backends.append(result.get("backend_latency_ms", 0))
                        else:
                            errors += 1
                    except Exception:
                        errors += 1

                    if shaping.compress > 1 and i < len(corpus) - 1:
                        gap = (corpus[i + 1]["completed_at"] - rec["completed_at"])
                        compressed_gap = gap / shaping.compress
                        if compressed_gap > 0:
                            await asyncio.sleep(compressed_gap)

            elapsed = time.monotonic() - t0

        report.replayed = len(replay_waits) + errors
        report.errors = errors
        report.p50_wait_ms = _percentile(replay_waits, 50)
        report.p95_wait_ms = _percentile(replay_waits, 95)
        report.p50_backend_ms = _percentile(replay_backends, 50)
        report.p95_backend_ms = _percentile(replay_backends, 95)
        report.throughput_rps = report.replayed / elapsed if elapsed > 0 else 0
        return report

    # --- Mode 3: A/B backend comparison ---

    async def run_ab(
        self,
        hours: float = 4.0,
        endpoint: str | None = None,
        call_site: str | None = None,
        shadow_host: str = "",
        shadow_port: int = 0,
        concurrency: int = 4,
    ) -> ABReport:
        if self._db is None:
            raise RuntimeError("ab requires a db_path (proxy queue.db)")
        if not shadow_host or not shadow_port:
            raise ValueError("shadow_host and shadow_port required for A/B mode")

        corpus = self._db.export_corpus(
            hours=hours, endpoint=endpoint, call_site=call_site,
        )
        if not corpus:
            logger.warning("no corpus records found")
            return ABReport()

        report = ABReport(corpus_size=len(corpus))
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient(timeout=360.0) as client:
            tasks = []
            for rec in corpus:
                if rec.get("payload") is None:
                    continue
                tasks.append(self._ab_one(
                    client, rec, shadow_host, shadow_port, sem, report,
                ))
            await asyncio.gather(*tasks)

        ok_results = [r for r in report.results if r.shadow_error is None]
        if ok_results:
            report.shadow_success_count = len(ok_results)

            primary_durs = [r.primary_duration_s * 1000 for r in ok_results]
            shadow_durs = [r.shadow_duration_s * 1000 for r in ok_results]
            report.primary_p50_duration_ms = _percentile(primary_durs, 50)
            report.primary_p95_duration_ms = _percentile(primary_durs, 95)
            report.shadow_p50_duration_ms = _percentile(shadow_durs, 50)
            report.shadow_p95_duration_ms = _percentile(shadow_durs, 95)

            p_tps = [
                r.primary_output_tokens / r.primary_duration_s
                for r in ok_results if r.primary_duration_s > 0
            ]
            s_tps = [
                r.shadow_output_tokens / r.shadow_duration_s
                for r in ok_results if r.shadow_duration_s > 0
            ]
            report.primary_decode_tps = sum(p_tps) / len(p_tps) if p_tps else 0
            report.shadow_decode_tps = sum(s_tps) / len(s_tps) if s_tps else 0

            jaccards = [r.token_overlap_jaccard for r in ok_results]
            report.avg_token_overlap_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0

            ratios = [r.length_ratio for r in ok_results if r.length_ratio > 0]
            report.avg_length_ratio = sum(ratios) / len(ratios) if ratios else 0

            p_parse = sum(1 for r in ok_results if r.json_parseable_primary)
            s_parse = sum(1 for r in ok_results if r.json_parseable_shadow)
            report.json_parse_rate_primary = p_parse / len(ok_results) * 100
            report.json_parse_rate_shadow = s_parse / len(ok_results) * 100

        report.shadow_error_count = len(report.results) - report.shadow_success_count
        return report

    async def _ab_one(
        self,
        client: httpx.AsyncClient,
        rec: dict,
        shadow_host: str,
        shadow_port: int,
        sem: asyncio.Semaphore,
        report: ABReport,
    ) -> None:
        async with sem:
            # Corpus payloads are stored pre-normalization (top-level
            # `system` + `extra_body`). The proxy applies
            # _normalize_chat_payload before hitting the primary backend,
            # so the shadow MUST get the same treatment — otherwise it
            # silently loses system+grammar and the A/B comparison is
            # invalid (shadow always looks worse). See backend.py.
            from .backend import _normalize_chat_payload
            payload = _normalize_chat_payload(rec["payload"])
            path = "/v1/chat/completions"

            ab = ABResult(
                request_id=rec["request_id"],
                call_site=rec["call_site"],
                endpoint=rec["endpoint"],
                primary_duration_s=rec.get("duration_s", 0),
                primary_input_tokens=rec.get("input_tokens", 0),
                primary_output_tokens=rec.get("output_tokens", 0),
                primary_response=rec.get("response") or {},
            )

            t0 = time.monotonic()
            try:
                resp = await client.post(
                    f"http://{shadow_host}:{shadow_port}{path}",
                    json=payload,
                    headers={"X-Request-ID": f"ab-{rec['request_id']}"},
                )
                shadow_dur = time.monotonic() - t0
                if resp.status_code >= 400:
                    ab.shadow_error = f"HTTP {resp.status_code}"
                else:
                    body = resp.json()
                    usage = body.get("usage", {})
                    ab.shadow_duration_s = shadow_dur
                    ab.shadow_input_tokens = usage.get("prompt_tokens", 0)
                    ab.shadow_output_tokens = usage.get("completion_tokens", 0)
                    ab.shadow_response = body

                    p_text = _extract_text(ab.primary_response)
                    s_text = _extract_text(ab.shadow_response)
                    ab.token_overlap_jaccard = _jaccard(p_text, s_text)
                    p_len = len(p_text)
                    s_len = len(s_text)
                    ab.length_ratio = s_len / p_len if p_len > 0 else 0

                    ab.json_parseable_primary = _check_json_parseable(ab.primary_response)
                    ab.json_parseable_shadow = _check_json_parseable(ab.shadow_response)
            except Exception as exc:
                ab.shadow_error = str(exc)[:200]

            report.results.append(ab)
