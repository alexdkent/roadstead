"""ProxyHttpHandlers — the Starlette HTTP surface for the LLMProxy.

The ~26 `handle_*` request handlers (OpenAI front door, admin, metrics, status,
SSE, fleet usage) + the admin-IP audit. Thin translators: parse the request,
read/mutate the shared :class:`ProxyState` or call the Lifecycle/Health
collaborators, serialize a response. ProxyService keeps identical-signature
`handle_*` delegators so `routes.make_routes(svc)` binds unchanged (contract §1).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from . import cache_stats
from .config import LLMPriority, normalize_endpoint
from .constants import _PAYLOAD_KIND
from .lifecycle import _openai_error
from .observability import structured_empty_rates
from .sse_hub import DROP_SENTINEL

if TYPE_CHECKING:
    from .health import Health
    from .lifecycle import Lifecycle


_TIER_RE = re.compile(r"^tier\d+$")
_ADVERTISED_ID: "dict[str, str] | None" = None


def _advertised_model_id(ep_cfg) -> str:
    """The name /v1/models advertises for an endpoint: its CANONICAL tier name.

    The 2026-07-30 tier migration made ``tier1``/``tier2``/``tier3`` the
    canonical names; the ``role:`` strings (``gemma-router``, ``creative``,
    ``llama-thinker``) are legacy and only kept resolving for compat. /v1/models
    is where third-party clients LEARN a model name and then pin it in their own
    config, so advertising a legacy role there mints new callers on the old name
    indefinitely — which is exactly the thing the migration was ending.

    Falls back to ``role`` for endpoints with no tier (embed/rerank are not
    tiers). Built once from the catalog; the catalog is the authority, so a
    renamed tier follows automatically.
    """
    global _ADVERTISED_ID
    if _ADVERTISED_ID is None:
        from .model_catalog import load_catalog
        mapping: dict[str, str] = {}
        for e in load_catalog().proxy_endpoints():
            tier = next((a for a in e.aliases if _TIER_RE.match(a)), None)
            if tier:
                mapping[e.role] = tier
        _ADVERTISED_ID = mapping
    return _ADVERTISED_ID.get(ep_cfg.role, ep_cfg.role)


def _embedding_texts(raw) -> "tuple[list, str]":
    """OpenAI ``input`` -> the shim's ``texts`` list. Returns ``(texts, error)``.

    OpenAI also allows PRE-TOKENIZED input (``[int]`` or ``[[int]]``). The bge-m3 shim
    takes text only and exposes no detokenizer, so that form is refused with a clear
    message rather than silently embedding the string ``"[1, 2, 3]"`` — a wrong vector
    is far worse here than an error, because nothing downstream can tell it apart from
    a right one.
    """
    if isinstance(raw, str):
        return ([raw], "") if raw else ([], "input must not be empty")
    if isinstance(raw, list):
        if not raw:
            return [], "input must not be empty"
        if all(isinstance(x, str) for x in raw):
            if any(not x for x in raw):
                return [], "input must not contain empty strings"
            return raw, ""
        if all(isinstance(x, int) for x in raw) or all(isinstance(x, list) for x in raw):
            return [], ("pre-tokenized input is not supported by the bge-m3 backend; "
                        "send text (a string or a list of strings)")
        return [], "input must be a string or a list of strings"
    if raw is None:
        return [], "input is required"
    return [], f"input must be a string or a list of strings, got {type(raw).__name__}"


def _encode_embedding(vec: list, fmt: str):
    """Float list, or OpenAI's base64 form (little-endian float32).

    base64 is not exotic — the official OpenAI Python SDK REQUESTS it by default and
    then decodes it itself, so a door that only ever returns float lists breaks the
    most likely real client.
    """
    if fmt != "base64":
        return vec
    import base64
    import struct
    return base64.b64encode(
        struct.pack(f"<{len(vec)}f", *(float(x) for x in vec))).decode("ascii")


def _estimated_embed_tokens(texts: list) -> int:
    """4 chars ~= 1 token, matching cost_model.estimate_input_tokens. An ESTIMATE."""
    return max(1, sum(len(t) for t in texts) // 4)
    from .state import ProxyState

logger = logging.getLogger(__name__)


def _parse_duration(s: str, default: int) -> int:
    """Parse a '24h'/'90m'/'3600s' duration to seconds (ported from the host
    telemetry daemon so the unified frontend's window params are identical)."""
    import re
    m = re.match(r"^(\d+)([smhd])$", (s or "").strip())
    if not m:
        return default
    n, u = int(m.group(1)), m.group(2)
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u]


def _clamp_window(s: str, default: int, *, cap: int = 7 * 86400) -> int:
    return max(60, min(_parse_duration(s, default), cap))


def _bin_seconds_for(window_s: int) -> int:
    """Bin width that keeps a series in ~60-150 points (matches the daemon)."""
    if window_s <= 3600:
        return 60
    if window_s <= 6 * 3600:
        return 300
    if window_s <= 86400:
        return 600
    return 3600


def _to_int(value: object, default: int) -> int:
    """Parse a query-param / body field to int, falling back to ``default``
    on anything malformed — so a bad ``?bin=abc`` uses the default instead of
    escaping as an unhandled 500 from these read-only/best-effort handlers."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float) -> float:
    """Float counterpart to ``_to_int`` — default on malformed input."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class ProxyHttpHandlers:
    """The Starlette HTTP surface over the shared ProxyState."""

    def __init__(self, state: "ProxyState", lifecycle: "Lifecycle",
                 health: "Health") -> None:
        self.state = state
        self.lifecycle = lifecycle
        self.health = health

    async def handle_openai_chat(self, body: dict, request: Request) -> Response:
        remote_ip = request.client.host if request.client else "unknown"
        identity = self.state.acl.identify(remote_ip)
        if not identity:
            # Phase 5D: OpenAI-shaped 403 (was a bare {"error": "<str>"} that a
            # strict OpenAI client crashes on doing resp.error.message).
            return _openai_error(
                f"access denied for {remote_ip}", "access_denied", 403,
                code="access_denied")
        agent_id, default_priority = identity
        model = body.get("model", "qwen-analyst")
        # Phase 5D: validate the model maps to a known endpoint BEFORE enqueue.
        # Otherwise an unknown model burns a scheduler slot + DRR charge and
        # fails late with a confusing 502; OpenAI clients expect 404/model_not_found.
        if normalize_endpoint(str(model)) not in self.state.config.endpoints:
            return _openai_error(
                f"unknown model {model!r}", "model_not_found", 404,
                code="unknown_endpoint")

        # Honor a client-supplied deadline (goose recipes can run long): a
        # ``timeout_s`` body field or an ``X-Timeout-S`` header. Popped from the
        # body so it isn't forwarded to the backend (which would reject the
        # unknown field). When the client supplies NOTHING, omit timeout_s so
        # handle_submit chooses the (flag-gated) smart/flat default — instead of
        # baking the flat 180s here, which denied the OpenAI door the data-driven
        # default framework callers get (Phase 5a). Coercion of a supplied value
        # is handled uniformly in handle_submit (a malformed value falls back to
        # the default there, same as before).
        client_timeout = body.pop("timeout_s", None) or request.headers.get("X-Timeout-S")

        submit_body = {
            "agent_id": agent_id,
            "endpoint": model,
            "priority": int(default_priority),
            "call_site": f"{agent_id}.openai_compat",
            "caller_id": agent_id,
            "payload_type": "chat_completion",
            "payload": body,
        }
        if client_timeout is not None:
            submit_body["timeout_s"] = client_timeout
        return await self.lifecycle.handle_submit(submit_body, request, openai=True)
    async def handle_openai_embeddings(self, body: dict, request: Request) -> Response:
        """POST /v1/embeddings — the OpenAI-compatible embeddings door.

        This is a TRANSLATOR in both directions, which it was not until 2026-08-01.
        The bge-m3 shim speaks its own dialect on BOTH sides and the two never met:

          request   OpenAI ``{"input": str | [str]}``   ->  shim ``{"texts": [str]}``
          response  shim ``{"dense": [[float]], ...}``  ->  OpenAI ``{object, data, usage}``

        Before, the raw OpenAI body was forwarded verbatim, so the shim rejected every
        call for a missing ``texts`` field and the door returned 502 for its whole life.
        The bug was invisible because no fleet caller uses it — agents embed through
        ``/v1/submit``, which already speaks ``texts`` — so this path only ever served
        external OpenAI clients, and nothing in-repo exercised it.
        """
        remote_ip = request.client.host if request.client else "unknown"
        identity = self.state.acl.identify(remote_ip)
        if not identity:
            return _openai_error(
                f"access denied for {remote_ip}", "access_denied", 403,
                code="access_denied")
        agent_id, default_priority = identity

        texts, err = _embedding_texts(body.get("input"))
        if err:
            return _openai_error(err, "invalid_request_error", 400,
                                 code="invalid_request_error")

        fmt = body.get("encoding_format") or "float"
        if fmt not in ("float", "base64"):
            return _openai_error(
                f"unsupported encoding_format {fmt!r} (expected 'float' or 'base64')",
                "invalid_request_error", 400, code="invalid_request_error")

        submit_body = {
            "agent_id": agent_id,
            "endpoint": "bge-m3-embed",
            "priority": int(default_priority),
            "call_site": f"{agent_id}.openai_compat_embed",
            "payload_type": "embedding",
            # BOTH dialects, deliberately. The bge-m3 shim reads `texts` and ignores
            # unknown keys (verified against the live shim); an OpenAI-shaped embeddings
            # server reads `input` and sizes its reply from it. Sending only `texts`
            # makes such a server return ONE vector for an N-input request — a silent
            # under-count, which is worse than an error because the caller gets a
            # well-formed list of the wrong length. `texts` is the normalized list, so
            # the two never disagree about content.
            "payload": {"texts": texts, "input": texts},
            "timeout_s": 60.0,
        }
        # openai=True keeps ERRORS OpenAI-shaped (the internal {status,response}
        # envelope used to leak). Success still returns the bare BACKEND body, so the
        # OpenAI shape is applied here rather than in the shared sync path — embeddings
        # are the only payload_type needing it, and the chat hot path stays untouched.
        resp = await self.lifecycle.handle_submit(submit_body, request, openai=True)
        if getattr(resp, "status_code", 500) != 200:
            return resp
        try:
            backend = json.loads(resp.body)
        except (ValueError, AttributeError, TypeError):
            return _openai_error("embedding backend returned an unreadable body",
                                 "backend_error", 502, code="backend_error")
        # TWO backend dialects, and assuming one is how this broke a second time.
        # The nexus bge-m3 shim answers `{"dense": [[float]]}`; a stock OpenAI-shaped
        # embeddings server (and the e2e fake backend) answers with `data`/`object`
        # already correct. Translate the former, pass the latter through — a backend
        # that already speaks OpenAI must not be re-wrapped.
        if isinstance(backend.get("data"), list) and backend.get("object") == "list":
            if fmt == "base64":
                # Only the ENCODING differs; leave index/usage/model as the backend set
                # them, since those are its measurements and not ours to invent.
                for row in backend["data"]:
                    if isinstance(row, dict) and isinstance(row.get("embedding"), list):
                        row["embedding"] = _encode_embedding(row["embedding"], "base64")
            return JSONResponse(backend)

        vectors = backend.get("dense")
        if not isinstance(vectors, list) or not vectors:
            # Assert PRESENCE: an embeddings reply with no vectors is a failure, not an
            # empty success. Returning {"data": []} here would let a caller treat a dead
            # backend as "nothing to embed".
            return _openai_error(
                "embedding backend returned no vectors "
                f"(keys={sorted(backend)[:6]})", "backend_error", 502,
                code="backend_error")
        # Prefer the backend's own usage when it reports one; fall back to the estimate
        # only because the bge-m3 shim reports no token counts at all. A measurement
        # always beats our 4-chars-per-token guess.
        est = _estimated_embed_tokens(texts)
        usage = backend.get("usage")
        if not isinstance(usage, dict) or "prompt_tokens" not in usage:
            # Same estimator the proxy already records as estimated_input_tokens, so the
            # door and telemetry agree rather than being two differently-wrong numbers.
            # An ESTIMATE, not a measurement — do not bill from it.
            usage = {"prompt_tokens": est, "total_tokens": est}
        return JSONResponse({
            "object": "list",
            "data": [{"object": "embedding", "index": i,
                      "embedding": _encode_embedding(v, fmt)}
                     for i, v in enumerate(vectors)],
            "model": body.get("model") or "bge-m3",
            "usage": usage,
        })
    async def handle_models(self, request: Request) -> Response:
        """GET /v1/models — OpenAI-compatible listing.

        Two things here are load-bearing for third-party OpenAI clients, both
        learned from wiring poolside's `pool` CLI 2026-08-02:

        1. ORDER IS A DEFAULT. A client with no configured model takes
           ``data[0]``. Ours used to be whatever ``config.endpoints`` happened to
           iterate first, which was the (now removed) 122B backup class — so pool
           defaulted onto a decorative endpoint. Chat roles sort first, biggest
           context first, so ``data[0]`` is the fleet's deepest chat model.
        2. ``max_model_len`` IS THE CONTEXT FIELD CLIENTS ACTUALLY READ. The
           OpenAI model schema has no context field at all; vLLM's own
           /v1/models emits ``max_model_len``, so that is what a client built
           against vLLM looks for. We were emitting only the bespoke
           ``context_per_slot``, which nothing outside the fleet understands, so
           pool fell back to its built-in default (128K) and budgeted its context
           against that instead of the 700K we actually serve.

        ``context_per_slot``/``endpoint_class``/``max_slots`` are kept for the
        fleet's own consumers (Inference page, tooling).
        """
        _EMBEDDINGS = {"embed", "rerank"}
        rows = []
        for ep_name, ep_cfg in self.state.config.endpoints.items():
            ctx = ep_cfg.context_per_slot
            rows.append({
                # The CANONICAL tier name (tier1/tier2/tier3), never the legacy
                # `role`. `role` is still `gemma-router`/`creative`/
                # `llama-thinker` for compat, but the 2026-07-30 tier migration
                # made the tier names canonical and this is the surface every
                # NEW external client copies its model name from — advertising
                # `llama-thinker` here would mint fresh callers on a legacy name
                # forever. Non-tier endpoints (embed/rerank) have no tier and
                # keep their role. Guarded by
                # test_v1_models_advertises_canonical_tier_names.
                "id": _advertised_model_id(ep_cfg),
                "object": "model",
                # OpenAI clients expect `created`; vLLM emits it. Static per
                # boot is fine — nothing consumes the value, only the key.
                "created": self.state.boot_time_epoch,
                "owned_by": "collective",
                # The served context window, in the field vLLM uses (see above).
                "max_model_len": ctx,
                "endpoint_class": ep_name,
                # The legacy role, kept so in-fleet consumers that still key off
                # it don't break. External clients should use `id`.
                "role": ep_cfg.role,
                "max_slots": ep_cfg.max_slots,
                "context_per_slot": ctx,
            })
        # Chat before embed/rerank; within each, largest context first. Ties
        # break on id so the order is stable across boots.
        rows.sort(key=lambda r: (
            r["endpoint_class"] in _EMBEDDINGS,
            -(r["max_model_len"] or 0),
            r["id"],
        ))
        return JSONResponse({"object": "list", "data": rows})
    async def handle_prometheus_metrics(self, request: Request) -> Response:
        """GET /metrics — Prometheus text exposition of CURRENT QoS aggregates
        (timeseries_migration_plan Phase 3.1). In-memory reads only (scheduler +
        the 300s rolling window + since-boot counters), so ~zero scrape cost; the
        proxy_timeouts/_completions event tables stay the forensic system-of-
        record. Layer-split timeout counts + would-timeout% remain on
        /v1/timeouts (SQL-windowed), not here. Never raises — degrades to empty."""
        from originfleet.framework.metrics import Metric, render_prometheus

        now = time.monotonic()
        out: list[Metric] = []
        _Q = {50: "0.5", 95: "0.95"}
        _se_rates = structured_empty_rates(self.state.structured_empty_window, now)
        try:
            for ep_name in self.state.config.endpoints:
                lbl = {"endpoint": ep_name}
                snap = self.state.scheduler.endpoint_snapshot(ep_name)
                if snap.get("max_slots") is not None:
                    out.append(Metric("llmproxy_endpoint_slots_total",
                                      snap["max_slots"], lbl, "gauge",
                                      help="configured max concurrent slots"))
                if snap.get("in_flight") is not None:
                    out.append(Metric("llmproxy_endpoint_inflight",
                                      snap["in_flight"], lbl, "gauge",
                                      help="requests dispatched and in flight"))
                if snap.get("queued") is not None:
                    out.append(Metric("llmproxy_endpoint_queued", snap["queued"],
                                      lbl, "gauge", help="requests waiting in queue"))
                for band, n in (snap.get("queue_by_band") or {}).items():
                    out.append(Metric("llmproxy_endpoint_queued_by_band", n,
                                      {"endpoint": ep_name, "band": str(band)},
                                      "gauge", help="queued requests by priority band"))
                for q, qlabel in _Q.items():
                    w = self.state.metrics.percentile("queue_wait_ms", q,
                                                 endpoint=ep_name, now=now)
                    if w is not None:
                        out.append(Metric("llmproxy_queue_wait_ms", w,
                                          {"endpoint": ep_name, "quantile": qlabel},
                                          "gauge", help="queue-wait latency (ms)"))
                    b = self.state.metrics.percentile("backend_latency_ms", q,
                                                 endpoint=ep_name, now=now)
                    if b is not None:
                        out.append(Metric("llmproxy_backend_latency_ms", b,
                                          {"endpoint": ep_name, "quantile": qlabel},
                                          "gauge", help="backend latency (ms)"))
                out.append(Metric("llmproxy_recent_timeouts_5m",
                                  self.state.metrics.count(endpoint=ep_name,
                                                      status="timeout", now=now),
                                  lbl, "gauge",
                                  help="timeouts in the last 5 min"))
                out.append(Metric("llmproxy_recent_requests_5m",
                                  self.state.metrics.count(endpoint=ep_name, now=now),
                                  lbl, "gauge",
                                  help="requests in the last 5 min"))
                ep_cfg = self.state.config.endpoints[ep_name]
                ss_consumed = self.state.metrics.slot_seconds_consumed(ep_name, now)
                out.append(Metric("llmproxy_slot_seconds_5m", ss_consumed, lbl,
                                  "gauge", help="slot-seconds consumed in 5 min"))
                if ep_cfg.max_slots > 0:
                    ss_available = ep_cfg.max_slots * 300.0
                    util = (ss_consumed / ss_available * 100) if ss_available > 0 else 0
                    out.append(Metric("llmproxy_endpoint_utilization_pct",
                                      round(util, 1), lbl, "gauge",
                                      help="5-min slot utilization %"))
                h = self.state.endpoint_health.get(ep_name, {})
                out.append(Metric("llmproxy_endpoint_healthy",
                                  1 if h.get("healthy", True) else 0, lbl, "gauge",
                                  help="1 if the endpoint is healthy (not paused)"))
                # Empty-structured rate (ledger tier3-json-object-empty-brace):
                # the fraction of this endpoint's structured responses that came
                # back a well-formed JSON object with no answer in it. VM builds
                # the history; the standing alert is the page. Only emitted when
                # the endpoint cleared the sample floor — an unevaluated
                # endpoint must not publish a 0/0 that reads as "healthy".
                se = _se_rates.get(ep_name)
                if se and se.get("evaluated"):
                    out.append(Metric("llmproxy_structured_empty_rate",
                                      se["rate"], lbl, "gauge",
                                      help="fraction of structured responses "
                                           "carrying no answer (30m window)"))
                    out.append(Metric("llmproxy_structured_samples_30m",
                                      se["n"], lbl, "gauge",
                                      help="structured responses in the window"))

            stats = self.state.scheduler.stats()
            for key, name in (("total_dispatched", "llmproxy_dispatched_total"),
                              ("total_completed", "llmproxy_completed_total"),
                              ("total_timeouts", "llmproxy_timeouts_total")):
                if stats.get(key) is not None:
                    out.append(Metric(name, stats[key], {}, "counter",
                                      help="since-boot scheduler counter"))

            # Liveness + active-alert gauges (audit 2026-07-02): before these,
            # a dead poller/scheduler/DB-writer or a standing ALERT was visible
            # only on /health and the log — nothing a TSDB rule could fire on.
            out.append(Metric("llmproxy_scheduler_alive",
                              1 if self.health.scheduler_loop_alive() else 0, {},
                              "gauge", help="1 if the scheduler loop ticked recently"))
            out.append(Metric("llmproxy_poller_alive",
                              1 if self.health.poller_alive() else 0, {}, "gauge",
                              help="1 if the capacity poller iterated recently"))
            out.append(Metric("llmproxy_writer_thread_alive",
                              1 if self.state.queue_db.writer_alive() else 0, {},
                              "gauge", help="1 if the SQLite writer thread is alive"))
            by_sev: dict[str, int] = {}
            for a in (self.state.alerts or []):
                sev = str(a.get("severity", "WARNING"))
                by_sev[sev] = by_sev.get(sev, 0) + 1
            for sev in ("CRITICAL", "ERROR", "WARNING", "INFO"):
                out.append(Metric("llmproxy_alerts_active",
                                  by_sev.get(sev, 0), {"severity": sev}, "gauge",
                                  help="standing alert conditions by severity"))

            # Per-caller truncation gauge (audit 2026-07-12, L-2): surfaces WHO
            # is hitting output caps on WHICH model, split structured vs
            # freetext, from the in-memory tally keyed "endpoint|agent_id". A
            # climbing structured series is a caller max_tokens-too-low signal
            # (e.g. the kv4 grader). Emitted as two series per caller so a TSDB
            # rule can alert on structured truncations alone.
            for key, t in (self.state.truncation_by_model_caller or {}).items():
                endpoint, _, caller = str(key).partition("|")
                for structured_flag, field in (("true", "structured"),
                                                ("false", "freetext")):
                    out.append(Metric(
                        "llmproxy_truncations_total",
                        int(t.get(field, 0) or 0),
                        {"endpoint": endpoint, "caller": caller,
                         "structured": structured_flag},
                        "gauge",
                        help="output-cap (finish_reason=length) hits by caller"))

            # Per-endpoint empty-completion gauge (audit 2026-07-12, C-3): the
            # post-boxa-consolidation "empty completion on creative" reliability
            # signature. A 2xx with no content/tool_calls, counted each time the
            # backend.call() fail-loud gate trips. Watch it — investigate if it
            # climbs.
            for endpoint, n in (self.state.empty_completion_by_endpoint or {}).items():
                out.append(Metric("llmproxy_empty_completion_total", int(n or 0),
                                  {"endpoint": str(endpoint)}, "gauge",
                                  help="empty (position-0-EOS) completions by endpoint"))
        except Exception:
            logger.exception("llmproxy /metrics render failed")
            out = []
        return Response(render_prometheus(out),
                        media_type="text/plain; version=0.0.4")
    async def handle_status(self, request: Request) -> Response:
        now = time.monotonic()
        endpoints = {}
        for ep_name in self.state.config.endpoints:
            snap = self.state.scheduler.endpoint_snapshot(ep_name)
            snap["p50_wait_ms"] = self.state.metrics.percentile(
                "queue_wait_ms", 50, endpoint=ep_name, now=now,
            )
            snap["p95_wait_ms"] = self.state.metrics.percentile(
                "queue_wait_ms", 95, endpoint=ep_name, now=now,
            )
            snap["throughput_rps"] = round(
                self.state.metrics.throughput_rps(ep_name, now), 2,
            )
            ep_cfg = self.state.config.endpoints[ep_name]
            if ep_cfg.max_slots > 0:
                ss_consumed = self.state.metrics.slot_seconds_consumed(ep_name, now)
                ss_available = ep_cfg.max_slots * 300.0  # 5-min window
                snap["utilization_pct"] = round(
                    (ss_consumed / ss_available * 100) if ss_available > 0 else 0, 1,
                )
            h = self.state.endpoint_health.get(ep_name, {})
            admin_paused = ep_name in self.state.paused_endpoints
            # `healthy` must reflect an ADMIN pause too. The poller STOPS probing a
            # paused endpoint (handle_pause), so `endpoint_health[ep].healthy`
            # freezes at its last (usually True) value — which made /status report
            # an evicted classify as healthy+serving the whole eviction window
            # (blind spot, 2026-07-09). A paused endpoint is not serving, so fold
            # the pause in. Keep `paused` PROBE-only (it means "unexpectedly down"
            # for check_alerts Phase 2.5 — an operator/coordinator pause reads
            # paused:False by design); surface the deliberate pause via the
            # explicit `admin_paused` field instead.
            snap["healthy"] = h.get("healthy", True) and not admin_paused
            snap["paused"] = not h.get("healthy", True)  # check_alerts (Phase 2.5) keys on this
            if admin_paused:
                snap["admin_paused"] = True
            # Step 4b — rate-windowed cooldown surface (shadow report + enforce
            # state). `cooldown_trips` accrues even in shadow mode so the operator
            # can review the trip rate before flipping enforce; `cooling` +
            # `cooldown_remaining_s` reflect an ACTIVE (enforce) cooldown.
            trips = self.state.endpoint_cooldown_trips.get(ep_name, 0)
            if trips:
                snap["cooldown_trips"] = trips
            _cd_until = self.state.endpoint_cooldown_until.get(ep_name, 0.0)
            if _cd_until and now < _cd_until:
                snap["cooling"] = True
                snap["cooldown_remaining_s"] = round(_cd_until - now, 1)
            # Survivorship fix (2026-06-06): the timeout-advice model + shadow
            # only ingest status==ok, so they reported a misleading "0 would
            # timeout" while requests were actually timing out. Surface the real
            # 5-min timeout count per endpoint so a partial stall is VISIBLE
            # (feeds the endpoint_stalled alert + health-verifier/dashboards).
            snap["recent_timeouts"] = self.state.metrics.count(
                endpoint=ep_name, status="timeout", now=now)
            # Phase 2a — ACTUAL prefix-cache hit rate (last cache-stats cycle,
            # from the vLLM /metrics global counters). Absent for llama.cpp roles
            # (no counter) → field omitted = n/a.
            _cache = self.state.endpoint_cache_hit_rate.get(ep_name)
            if _cache and _cache.get("hit_rate") is not None:
                snap["cache_hit_rate"] = _cache["hit_rate"]
            endpoints[ep_name] = snap

        agents = {
            b["agent_id"]: b
            for b in self.state.budget_mgr.snapshot()
        }

        # Loop liveness is computed AT READ TIME: the poller is what evaluates
        # self.state.alerts, so a dead poller (or scheduler) can never report itself
        # through that path — only through this one.
        alerts = list(self.state.alerts)
        if not self.health.scheduler_loop_alive():
            alerts.append({
                "name": "scheduler_loop_dead", "severity": "CRITICAL",
                "detail": "scheduler loop task not running — dispatch is DOWN",
            })
        if not self.health.poller_alive():
            alerts.append({
                "name": "poller_dead", "severity": "CRITICAL",
                "detail": "capacity poller task not running — health probing, "
                          "alerting, and budget persistence have stopped",
            })

        return JSONResponse({
            "endpoints": endpoints,
            "agents": agents,
            "alerts": alerts,  # Phase 2.5 — health-verifier/log_scan surface
            # Runtime feature flags (flags.py) — read by dashboards + the
            # scheduled shadow→enforce flip checks.
            "flags": self.state.flags.as_dict(),
            "cache": self.state.cache.stats(),
            "scheduler": {
                **self.state.scheduler.stats(),
                "uptime_s": round(time.monotonic() - self.state.started_at, 0),
            },
            # Phase 5B reliability counters.
            "reliability": {
                "slot_leak_reclaimed": self.state.slot_leak_reclaimed,
                "drain_straggler_cancelled": self.state.drain_straggler_cancelled,
                "writer_thread_alive": self.state.queue_db.writer_alive(),
                "writer_thread_restarts": self.state.queue_db.writer_restarts(),
                "write_q_dropped": self.state.queue_db.write_q_dropped(),
                # Critical-loop liveness (read-time; see alerts note above).
                "scheduler_alive": self.health.scheduler_loop_alive(),
                "poller_alive": self.health.poller_alive(),
                # Source IPs seen on admin-ish routes since boot — the
                # ACL-tightening go/no-go reads this instead of grepping logs.
                "admin_ips_seen": {
                    route: sorted(ips)
                    for route, ips in self.state.admin_ips_seen.items()
                },
                # Unknown-endpoint submits since boot (shadow counter for the
                # unknown_endpoint_enforce flip check). Empty = safe to flip.
                "unknown_endpoint_submits": self.state.unknown_endpoint_submits,
                "vision_capability_violations":
                    self.state.vision_capability_violations,
                # Context-gate hits since boot (shadow counter for the
                # context_gate_enforce flip check — compare against actual
                # backend overflow errors before flipping).
                "context_overflows_shadow": self.state.context_overflows,
                # Smart-default-timeout shadow (Phase 5a): for callers that OMIT
                # timeout_s, what the data-driven default WOULD be (smart_s_*)
                # vs the flat 180s. Recorded whether smart_default_timeout is on
                # or off; the flip go/no-go reads mean = smart_s_sum / count.
                "smart_default_shadow": self.state.smart_default_shadow,
                # Thinking option (per-request native reasoning) health. All 0
                # until a caller opts in with thinking:true. Watch
                # thinking_truncated to tune COLLECTIVE_PROXY_THINKING_BUDGET down.
                "thinking_requests": self.state.thinking_requests,
                "thinking_clean": self.state.thinking_clean,
                "thinking_recovered": self.state.thinking_recovered,
                "thinking_truncated": self.state.thinking_truncated,
                "thinking_fallback": self.state.thinking_fallback,
                # Thinking applied but the response carried NO reasoning — the
                # backend template ignored enable_thinking. Non-zero here means
                # the opt-in is silently doing nothing on that endpoint.
                "thinking_noop": self.state.thinking_noop,
                # WS-4 shadow egress detector — silent grammar-drop over ALL
                # grammar-bearing responses (read-only/zero-risk). Per-call_site
                # rate + a flat fleet rate (health-verifier thresholds the scalar).
                "silent_drop_by_call_site": {
                    cs: {**t, "rate": round(t["dropped"] / t["checked"], 4)}
                    for cs, t in self.state.shadow_drop.items() if t["checked"]
                },
                "silent_drop_rate": round(
                    sum(t["dropped"] for t in self.state.shadow_drop.values())
                    / max(1, sum(t["checked"] for t in self.state.shadow_drop.values())),
                    4,
                ),
                # Empty-completion (position-0-EOS) rescue: retries dispatched
                # with min_tokens, and how many produced a real response.
                "empty_rescue_attempts": self.state.empty_rescue_attempts,
                "empty_rescue_recovered": self.state.empty_rescue_recovered,
                # Truncation / structured-validity guard (operator mandate
                # 2026-07-11): per-(model, caller) finish_reason=length tallies
                # (structured vs freetext) and structured json.loads failures.
                # Grep markers: LLMPROXY_TRUNCATION / LLMPROXY_STRUCTURED_INVALID.
                "truncation_total": self.state.truncation_total,
                "truncation_by_model_caller": self.state.truncation_by_model_caller,
                "structured_parse_failure_total":
                    self.state.structured_parse_failure_total,
                "structured_parse_failures_by_model_caller":
                    self.state.structured_parse_failures_by_model_caller,
                # Egress degeneration guard — repetition-loop responses detected,
                # and how many an anti-repetition re-dispatch recovered.
                "degeneration_detected": self.state.degeneration_detected,
                "degeneration_recovered": self.state.degeneration_recovered,
                "degeneration_unrecovered": self.state.degeneration_unrecovered,
                "degeneration_by_call_site": {
                    cs: t for cs, t in self.state.degeneration_by_call_site.items()
                    if t["detected"]
                },
                # Phase 3 schema-repair backstop — structured/tool responses that
                # failed their JSON contract, and how they were resolved (in-memory
                # json-repair / one error-fed-back retry / failed-loud deferrable).
                # stream = detected on a streaming reassembly (detect-only).
                "schema_detected": self.state.schema_detected,
                "schema_repaired": self.state.schema_repaired,
                "schema_retry_recovered": self.state.schema_retry_recovered,
                "schema_unrecoverable": self.state.schema_unrecoverable,
                "schema_invalid_stream": self.state.schema_invalid_stream,
                "schema_by_call_site": {
                    cs: t for cs, t in self.state.schema_by_call_site.items()
                    if t.get("detected")
                },
                # Empty-structured responses (2026-08-01, ledger
                # `tier3-json-object-empty-brace`): a STRUCTURED request that
                # came back a well-formed JSON object with no answer in it.
                # `{}` passes every other guard here, so this is the only place
                # the 31-hour silent outage would have shown up. Grep marker:
                # LLMPROXY_STRUCTURED_EMPTY. `structured_empty_rate` is the
                # live per-endpoint sliding window the standing alert reads.
                "structured_empty_total": self.state.structured_empty_total,
                "structured_empty_by_call_site":
                    self.state.structured_empty_by_call_site,
                "structured_empty_rate": structured_empty_rates(
                    self.state.structured_empty_window, time.monotonic()),
            },
            # Phase 5F — endpoints an operator has drained for maintenance.
            "paused_endpoints": sorted(self.state.paused_endpoints),
        })
    async def handle_admin_flags(self, request: Request) -> Response:
        """GET: current runtime flags. POST: update a subset (JSON object of
        flag→bool), persisted across restarts. Internal-only (ACL). This is the
        flip surface for the shadow→enforce switches and kill-switches — flags
        change behaviour immediately, no process restart."""
        remote_ip = request.client.host if request.client else "unknown"
        self.audit_admin_ip("/v1/admin/flags", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if request.method == "GET":
            return JSONResponse({"flags": self.state.flags.as_dict()})
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            # File write runs off-loop; flag reads elsewhere are plain dict
            # lookups on the loop thread (single mutation source — this handler).
            updated = await asyncio.to_thread(self.state.flags.set_many, body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        logger.warning("runtime flags updated by %s: %s", remote_ip, body)
        return JSONResponse({"flags": updated})
    async def handle_admin_endpoint_pause(
        self, endpoint: str, request: Request, *, pause: bool,
    ) -> Response:
        """Pause (drain) or resume an endpoint for maintenance — e.g. a vLLM
        restart to change --max-model-len. Internal-only (ACL).

        PAUSE marks the endpoint unhealthy NOW: background work defers (it
        queues + drains on resume — no loss while deadlines exceed the restart),
        interactive fast-fails with a deferrable error, and the poller stops
        probing it — so no request hits a backend you're about to kill, with no
        ~30s auto-circuit-trip lag. RESUME hands it back to the poller, which
        re-probes, recovers on /health, re-discovers capacity (the new
        max_model_len), and the deferred queue drains."""
        remote_ip = request.client.host if request.client else "unknown"
        self.audit_admin_ip("/v1/admin/endpoints", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        ep = normalize_endpoint(endpoint)
        if ep not in self.state.config.endpoints:
            return JSONResponse(
                {"error": f"unknown endpoint {endpoint!r}"}, status_code=404)
        # Optional free-text reason on PAUSE, recorded on the maintenance window
        # so the timeout burst during the restart reads as PLANNED (with the
        # reason) in /v1/timeouts. Body is optional — never fail the drain on it.
        reason = ""
        try:
            body = await request.json()
            if isinstance(body, dict):
                reason = str(body.get("reason") or "")
        except Exception:  # noqa: BLE001 — empty/invalid body is fine
            pass
        if pause:
            self.state.paused_endpoints.add(ep)
            # Release any already-queued interactive/foreground immediately with
            # a deferrable error (don't make them wait out their deadline).
            self.health.fast_fail_interactive(ep)
            # Open a maintenance window so timeouts during the restart are tagged
            # planned. Closed on RESUME (below).
            if self.state.queue_db is not None:
                self.state.queue_db.maintenance_open(
                    endpoint=ep, reason=reason, operator=remote_ip, source="drain")
            logger.warning(
                "endpoint %s PAUSED by operator (%s) — background defers, "
                "interactive fast-fails; backend safe to restart%s", ep, remote_ip,
                f" (reason: {reason})" if reason else "")
        else:
            self.state.paused_endpoints.discard(ep)
            self.state.dispatch_event.set()  # nudge the scheduler to drain deferred work
            if self.state.queue_db is not None:
                self.state.queue_db.maintenance_close(endpoint=ep)
            logger.warning(
                "endpoint %s RESUMED by operator (%s) — poller will re-probe / "
                "recover / re-discover capacity; deferred queue draining",
                ep, remote_ip)
        return JSONResponse({
            "endpoint": ep,
            "paused": ep in self.state.paused_endpoints,
            "healthy": self.health.endpoint_healthy(ep),
            "paused_endpoints": sorted(self.state.paused_endpoints),
        })
    async def handle_maintenance(self, request: Request) -> Response:
        """Annotate a maintenance window so a timeout burst during a DELIBERATE
        backend restart reads as PLANNED (not an incident) in /v1/timeouts.
        Internal-only (ACL). Use this when you restart a backend WITHOUT the
        pause/resume drain (which records the window automatically).

        Body (JSON):
          endpoint:    str | [str] | "*"   (required) class(es) under maintenance
          reason:      str                 free-text ("thinker restart: 128K")
          duration_s:  float (optional)    backdate a CLOSED window [now-d, now]
          started_at / ended_at: epoch s   explicit bounds (override duration_s)
        With none of duration_s/started_at/ended_at, opens an OPEN window now
        (close it later via the drain resume, or re-POST with ended_at)."""
        remote_ip = request.client.host if request.client else "unknown"
        self.audit_admin_ip("/v1/admin/maintenance", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if self.state.queue_db is None:
            return JSONResponse({"error": "no persistence backend"}, status_code=503)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            body = {}
        raw_eps = body.get("endpoint")
        if not raw_eps:
            return JSONResponse({"error": "endpoint required"}, status_code=400)
        eps = raw_eps if isinstance(raw_eps, list) else [raw_eps]
        norm_eps: list[str] = []
        for e in eps:
            if str(e) == "*":
                norm_eps.append("*")
                continue
            ne = normalize_endpoint(str(e))
            if ne not in self.state.config.endpoints:
                return JSONResponse(
                    {"error": f"unknown endpoint {e!r}"}, status_code=404)
            norm_eps.append(ne)
        reason = str(body.get("reason") or "")
        started_at = body.get("started_at")
        ended_at = body.get("ended_at")
        duration_s = body.get("duration_s")
        now = time.time()
        recorded: list[dict] = []
        try:
            for ep in norm_eps:
                if started_at is not None or ended_at is not None:
                    s = float(started_at) if started_at is not None else now
                    en = float(ended_at) if ended_at is not None else now
                    self.state.queue_db.maintenance_record(
                        endpoint=ep, started_at=s, ended_at=en,
                        reason=reason, operator=remote_ip)
                    recorded.append({"endpoint": ep, "started_at": s,
                                     "ended_at": en, "reason": reason})
                elif duration_s is not None:
                    s = now - float(duration_s)
                    self.state.queue_db.maintenance_record(
                        endpoint=ep, started_at=s, ended_at=now,
                        reason=reason, operator=remote_ip)
                    recorded.append({"endpoint": ep, "started_at": s,
                                     "ended_at": now, "reason": reason})
                else:
                    self.state.queue_db.maintenance_open(
                        endpoint=ep, reason=reason, operator=remote_ip,
                        source="manual")
                    recorded.append({"endpoint": ep, "started_at": now,
                                     "ended_at": None, "reason": reason})
        except (TypeError, ValueError) as exc:
            return JSONResponse(
                {"error": f"bad window bounds: {exc}"}, status_code=400)
        logger.warning(
            "maintenance window(s) recorded by operator (%s): %s reason=%r",
            remote_ip, [r["endpoint"] for r in recorded], reason)
        return JSONResponse({"recorded": recorded})
    async def handle_maintenance_list(self, request: Request) -> Response:
        """List maintenance windows overlapping the last ``hours`` (default 24).
        Admin surface (was unauthenticated — tightened with the rest)."""
        remote_ip = request.client.host if request.client else "unknown"
        self.audit_admin_ip("/v1/admin/maintenance", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse(
                {"error": f"access denied for {remote_ip}"}, status_code=403)
        if self.state.queue_db is None:
            return JSONResponse({"hours": 0, "windows": []})
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        windows = await asyncio.to_thread(self.state.queue_db.maintenance_windows, hours)
        return JSONResponse({"hours": hours, "windows": windows})
    async def handle_metrics(self, request: Request) -> Response:
        return JSONResponse(self.state.metrics_payload(time.monotonic()))
    async def handle_fleet_activity(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "24h"), 86400)
        bin_s = _to_int(request.query_params.get("bin"), _bin_seconds_for(window_s))
        # §5c: heavy GROUP-BY over the whole-fleet completions table runs off the
        # event loop so it can't stall fleet LLM scheduling under a hot dashboard.
        data = await asyncio.to_thread(self.state.queue_db.fleet_activity, window_s, bin_s)
        return JSONResponse(data)
    async def handle_fleet_savings(self, request: Request) -> Response:
        since_q = request.query_params.get("since")
        today_start = float(since_q) if since_q and since_q.isdigit() else None
        data = await asyncio.to_thread(self.state.queue_db.savings_summary, today_start)
        return JSONResponse(data)
    async def handle_top_callers(self, request: Request) -> Response:
        window_s = _clamp_window(request.query_params.get("window", "1h"), 3600)
        per_endpoint = max(1, min(_to_int(request.query_params.get("per_endpoint"), 5), 20))
        data = await asyncio.to_thread(self.state.queue_db.top_callers, window_s, per_endpoint)
        return JSONResponse(data)
    async def handle_fleet_cache_stats(self, request: Request) -> Response:
        """Prefix-cache observability: per-model actual hit-rate + per-call_site
        misalignment offenders + rollup + trend + drift. Computed from the
        periodic snapshots in proxy_cache_stats (see _compute_cache_stats)."""
        window_s = _clamp_window(request.query_params.get("window", "7d"), 30 * 86400)
        snaps = await asyncio.to_thread(self.state.queue_db.cache_stats_snapshots, window_s)
        labels = cache_stats.chat_endpoint_labels()
        engines = {ep: cfg.backend_engine for ep, cfg in self.state.config.endpoints.items()}
        return JSONResponse(cache_stats.build_fleet_payload(snaps, labels, engines))
    async def handle_cache_attribution(self, request: Request) -> Response:
        """Phase 2a — per-caller ACTUAL prefix-cache hit rate (Tier-2).

        The complement to /health/llmproxy.cache (Tier-1 predicted cacheability
        from the misalignment screen): this reports the *measured* hit rate from
        captured cached_tokens, per (call_site,endpoint), plus per-endpoint and
        fleet rollups. Rows the backend couldn't attribute (llama.cpp — no
        cached_tokens counter) are surfaced as unattributed, not folded into a
        false 0% (the ~4.8% global-scrape artifact this fixes). Heavy GROUP-BY →
        off the event loop so a dashboard poll can't stall fleet scheduling."""
        window_s = _clamp_window(request.query_params.get("window", "1h"), 7 * 86400)
        limit = max(1, min(_to_int(request.query_params.get("limit"), 40), 200))
        data = await asyncio.to_thread(
            self.state.queue_db.cache_attribution, window_s, limit)
        # Per-endpoint headline hit_rate: the cached_tokens rollup reads n/a
        # (vLLM emits null per-request), so overlay the REAL per-endpoint rate
        # from the vLLM /metrics scrape (Phase-2a operator decision). The
        # cached_tokens-derived fields (attributed/unattributed/attributable_in)
        # are kept for transparency; hit_rate_source flags the real ones. The
        # per-CALLER (by_call_site) rows stay cached_tokens-based (n/a until the
        # backend emits it — Tier-1's LCP screen predicts per-caller reuse).
        real = self.state.endpoint_cache_hit_rate
        for row in data.get("by_endpoint", []):
            r = real.get(row.get("endpoint"))
            if r and r.get("hit_rate") is not None:
                row["hit_rate"] = r["hit_rate"]
                row["hit_rate_source"] = r.get("source", "backend_prefix_cache_metrics")
        fleet = data.get("fleet")
        if fleet is not None and real:
            # fleet headline = query-weighted mean of the real per-endpoint rates.
            tot_q = sum(v["queries"] for v in real.values() if v.get("queries"))
            if tot_q > 0:
                fleet["hit_rate"] = round(
                    sum(v["hit_rate"] * v["queries"] for v in real.values()
                        if v.get("queries")) / tot_q, 4)
                fleet["hit_rate_source"] = "backend_prefix_cache_metrics"
        return JSONResponse(data)
    async def handle_usage(self, request: Request) -> Response:
        dimension = request.query_params.get("by", "agent")
        if dimension not in ("agent", "call_site", "endpoint", "provider"):
            dimension = "agent"
        hours = min(_to_float(request.query_params.get("hours"), 24), 168)
        rows = await asyncio.to_thread(self.state.queue_db.usage_rollup, dimension, hours)
        return JSONResponse({
            "dimension": dimension,
            "hours": hours,
            "rows": rows,
        })
    async def handle_series(self, request: Request) -> Response:
        endpoint = request.query_params.get("endpoint", "")
        if not endpoint:
            return JSONResponse({"error": "endpoint query param required"}, status_code=400)
        window_s = _clamp_window(request.query_params.get("window", "24h"), 86400)
        bin_s = _to_int(request.query_params.get("bin"), _bin_seconds_for(window_s))
        data = await asyncio.to_thread(
            self.state.queue_db.endpoint_series, endpoint, window_s, bin_s)
        return JSONResponse(data)
    async def handle_calls_log(self, request: Request) -> Response:
        """Ingest a non-LLM service call (audio/imagegen/ocr/translate) that
        never traversed the scheduler, so the proxy is the single fleet
        call-metrics store. Internal/LAN — gated by the same ACL as admin.
        Best-effort: validates the minimum, records, fans out, returns ok."""
        remote_ip = request.client.host if request.client else "unknown"
        self.audit_admin_ip("/v1/calls/log", remote_ip)
        if not self.state.acl.is_admin(remote_ip):
            return JSONResponse({"error": f"access denied for {remote_ip}"}, status_code=403)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "invalid json"}, status_code=400)
        endpoint = str(body.get("endpoint") or body.get("provider") or body.get("unit") or "").strip()
        kind = str(body.get("kind") or "").strip() or "external"
        if not endpoint:
            return JSONResponse({"error": "endpoint/provider/unit required"}, status_code=400)
        if kind in _PAYLOAD_KIND.values() or kind == "llm":
            # The proxy is authoritative for LLM traffic via native recording;
            # refuse a pushed LLM call so the two paths can't double-count.
            return JSONResponse(
                {"error": f"kind {kind!r} is proxy-native; do not push LLM calls"},
                status_code=409)
        if normalize_endpoint(endpoint) in self.state.config.endpoints:
            # Same double-count guard keyed on the ENDPOINT: a push whose
            # endpoint normalizes to a proxy-native LLM class (rerank, embed,
            # chat, ...) duplicates a row the proxy already recorded natively -
            # regardless of the kind label it arrives under (the 2026-06-11
            # rerank double-count arrived as kind='external').
            return JSONResponse(
                {"error": f"endpoint {endpoint!r} is proxy-native "
                          f"({normalize_endpoint(endpoint)}); do not push LLM calls"},
                status_code=409)
        status = str(body.get("status") or ("ok" if body.get("success", True) else "error"))
        request_id = str(body.get("request_id") or f"ext-{uuid.uuid4().hex}")
        in_tok = _to_int(body.get("input_tokens"), 0)
        out_tok = _to_int(body.get("output_tokens"), 0)
        latency_ms = _to_float(body.get("latency_ms"), 0.0)
        duration_s = _to_float(body.get("duration_s"), latency_ms / 1000.0)
        agent_id = str(body.get("agent") or body.get("agent_id") or "unknown")
        call_site = str(body.get("call_site") or kind)
        try:
            self.state.queue_db.persist_external_call(
                request_id=request_id, agent_id=agent_id, endpoint=endpoint,
                call_site=call_site, kind=kind, input_tokens=in_tok,
                output_tokens=out_tok, duration_s=duration_s, status=status,
                caller_id=body.get("caller_id"),
            )
            self.state.sse.publish("call.completed", {
                "request_id": request_id, "agent": agent_id,
                "endpoint": normalize_endpoint(endpoint), "call_site": call_site,
                "kind": kind, "priority": "P2_POST_TURN",
                "input_tokens": in_tok, "output_tokens": out_tok,
                "duration_s": round(duration_s, 3), "queue_wait_ms": 0.0,
                "status": status, "ts": time.time(),
            })
        except Exception as exc:  # noqa: BLE001 — never fail the pusher
            logger.warning("calls/log ingest failed: %s", exc)
            return JSONResponse({"error": "ingest failed"}, status_code=500)
        return JSONResponse({"ok": True, "request_id": request_id})
    async def handle_stream(self, request: Request) -> Response:
        """Server-Sent Events: real-time `call.completed` + periodic `metrics`
        frames. Slow clients are dropped (the browser reconnects + re-syncs via
        the REST endpoints). Mirrors the host-telemetry /stream/v2 shape."""
        q = self.state.sse.subscribe()

        async def event_gen():
            try:
                yield f"event: hello\ndata: {json.dumps({'ts': time.time()})}\n\n".encode()
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event, data = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
                        continue
                    if (event, data) == DROP_SENTINEL:
                        break
                    yield f"event: {event}\ndata: {data}\n\n".encode()
            except asyncio.CancelledError:
                pass
            finally:
                self.state.sse.unsubscribe(q)

        return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })
    async def handle_cost_model(self, request: Request) -> Response:
        return JSONResponse(self.state.cost_model.snapshot())
    async def handle_history(self, request: Request) -> Response:
        hours = min(_to_float(request.query_params.get("hours"), 4), 168)
        bucket_minutes = max(1, min(_to_int(request.query_params.get("bucket_minutes"), 5), 60))
        buckets = await asyncio.to_thread(
            self.state.queue_db.history_buckets, hours, bucket_minutes)
        return JSONResponse({"buckets": buckets})
    async def handle_recent(self, request: Request) -> Response:
        limit = min(_to_int(request.query_params.get("limit"), 50), 200)
        rows = await asyncio.to_thread(self.state.queue_db.recent_requests, limit)
        return JSONResponse({"requests": rows})
    async def handle_inflight(self, request: Request) -> Response:
        """Live list of currently-executing requests + per-endpoint occupancy —
        the proxy is the dispatch authority, so this is the authoritative
        real-time "what's flowing through the LLM systems" view. Pure in-memory
        snapshot; also pushed via the SSE `inflight` frame for sub-second feel."""
        snap = self.state.scheduler.inflight_snapshot(time.monotonic())
        snap["ts"] = time.time()
        return JSONResponse(snap)
    async def handle_timeout_advice(self, request: Request) -> Response:
        """Recommended timeout (+ min/median/p95) for a model/tier/size,
        derived from measured end-to-end latency."""
        qp = request.query_params
        model = qp.get("model")
        if not model:
            return JSONResponse(
                {"error": "missing required param: model"}, status_code=400,
            )
        endpoint = normalize_endpoint(model)
        if endpoint not in self.state.config.endpoints:
            return JSONResponse(
                {"error": f"unknown model {model!r}",
                 "known": sorted(self.state.config.endpoints)},
                status_code=400,
            )
        pri_raw = qp.get("priority") or "P1_TURN_SUPPORT"
        # Accept both numeric strings ("1") and enum names ("P1_TURN_SUPPORT").
        pri_val: str | int = int(pri_raw) if pri_raw.lstrip("-").isdigit() else pri_raw
        try:
            priority = int(LLMPriority.coerce(pri_val))
        except (ValueError, KeyError):
            return JSONResponse(
                {"error": f"unknown priority {qp.get('priority')!r}"},
                status_code=400,
            )
        try:
            est_in = int(qp.get("est_in", "0"))
            est_out = int(qp.get("est_out", "0"))
        except ValueError:
            return JSONResponse(
                {"error": "est_in/est_out must be integers"}, status_code=400,
            )

        # Uplifted for live load + prompt size, bounded by the caller-class
        # ceiling — so the extend-only client actually waits long enough under
        # contention instead of severing a merely-slow call.
        advice = self.state.effective_timeout_advice(endpoint, priority, est_in, est_out)
        return JSONResponse({
            "model": endpoint,
            "priority": LLMPriority(priority).name,
            "est_in": est_in,
            "est_out": est_out,
            **advice,
        })
    async def handle_timeout_shadow_report(self, request: Request) -> Response:
        """Per-(model, tier) shadow summary: would-timeout rate and
        headroom reclaimed vs. the timeout actually applied."""
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        report = await asyncio.to_thread(self.state.queue_db.timeout_shadow_report, hours)
        return JSONResponse({"hours": hours, "report": report})
    async def handle_timeouts_report(self, request: Request) -> Response:
        """Calls that hit their timeout instead of finishing, per
        (model, tier, layer), with the load context when they gave up and
        how many fired below the recommended deadline (premature).

        ``by_abort_reason`` splits the ``stream`` layer into ttft / stall /
        hard_cap / caller_deadline. ``stream_extensions`` is the counterpart the
        DB cannot show: streams that SURVIVED past a proxy-chosen deadline
        because they were still emitting tokens. Those never write a timeout
        row, so without this the progress-governed deadline would be an
        invisible capacity sink — you'd see the kills it prevents and never the
        slot-seconds it spends. Process-lifetime counters (reset on restart),
        read straight off the loop."""
        hours = _to_float(request.query_params.get("hours"), 24)
        hours = min(max(hours, 0.1), 168)
        report = await asyncio.to_thread(self.state.queue_db.timeouts_report, hours)
        return JSONResponse({
            "hours": hours,
            **report,
            "stream_extensions": {
                "count": self.state.stream_deadline_extended,
                "total_s": round(self.state.stream_extension_s_total, 1),
                "hard_cap_aborts": self.state.stream_hard_cap_aborts,
            },
        })
    async def handle_stall_aborts(self, request: Request) -> Response:
        """Backend-stall aborts by ONE caller inside ONE time window.

        The forensic counterpart to /v1/timeouts: that one aggregates and
        answers "is the fleet under pressure?"; this answers "while this
        specific run was alive, did the backend stall underneath it?" — the
        question a consumer needs to tell an upstream substrate failure apart
        from its own agent hanging. `caller` is a PREFIX (ids carry a per-run
        suffix); `since`/`until` are epoch seconds, inclusive.

        Read-only. Returns `{"caller","since","until","count","rows"}`; an
        unknown caller or an empty window is `count: 0`, never an error — the
        consumer treats "no evidence" and "cannot tell" identically."""
        caller = (request.query_params.get("caller") or "").strip()
        since = _to_float(request.query_params.get("since"), 0.0)
        until = _to_float(request.query_params.get("until"), time.time())
        rows = await asyncio.to_thread(
            self.state.queue_db.stall_aborts, caller, since, until)
        return JSONResponse({
            "caller": caller, "since": since, "until": until,
            "count": len(rows), "rows": rows,
        })

    async def handle_health(self, request: Request) -> Response:
        ok = self.health.scheduler_loop_alive()
        poller_ok = self.health.poller_alive()
        unhealthy = [ep for ep, h in self.state.endpoint_health.items() if not h["healthy"]]
        # The proxy is UP iff its scheduler is alive (200). A dead BACKEND
        # degrades status but must NOT 503 the proxy — that would make a monitor
        # restart a healthy front door over a backend blip (alert-don't-kill).
        # A dead POLLER also only degrades: dispatch still works, but health
        # probing / alerting / budget persistence have stopped — surface it.
        status = "ok" if (ok and poller_ok and not unhealthy) else ("degraded" if ok else "down")
        return JSONResponse(
            {
                "status": status,
                "uptime_s": round(time.monotonic() - self.state.started_at, 0),
                "total_dispatched": self.state.scheduler.stats()["total_dispatched"],
                "endpoints": len(self.state.config.endpoints),
                "total_slots": self.state.config.total_fleet_slots,
                "unhealthy_endpoints": unhealthy,
                "scheduler_alive": ok,
                "poller_alive": poller_ok,
            },
            status_code=200 if ok else 503,
        )
    async def handle_readyz(self, request: Request) -> Response:
        """READINESS, which is the opposite question from `/health` above.

        `/health` is deliberately fail-OPEN — a dead backend degrades its status
        but must not 503 the proxy, because a monitor should not restart a
        healthy front door over a backend blip (alert-don't-kill). That is right
        for liveness and useless for routing.

        `/readyz` fails CLOSED (tier2 split plan §8.0 req 4): when the endpoint
        backing the conversational lane is unhealthy or paused, say NOT READY so
        callers stop dispatching rather than queueing at a dead box. Under the
        fleet-wide go-dark decision a chat surface must refuse the turn and say
        so, and it cannot do that if the proxy keeps accepting work.

        🚨 THIS DOES NOT SEE "UP BUT SLOW", and must not be read as if it did.
        It is built on the circuit-breaker health flag, and `Health._probe` RESETS
        `consecutive_failures` to 0 whenever `/health` answers — so a backend that
        responds promptly and generates at a fraction of its speed is, to this
        endpoint, perfectly ready. Slow detection lives in two other places by
        design: `infra/jetty/tier2-chat/bootgate.py` catches a bad SPAWN at boot
        (§8.0 req 1), and the health-verifier ground-truth verifier catches slow DRIFT
        continuously (§8.0 req 2). Do not add a latency guess here to paper over
        that — a readiness endpoint that flaps on a slow prompt is worse than one
        with a documented blind spot.
        """
        critical = [
            name for name, cfg in self.state.config.endpoints.items()
            if getattr(cfg, "readiness_critical", False)
        ]
        scheduler_ok = self.health.scheduler_loop_alive()
        unready: dict[str, str] = {}
        for name in critical:
            if name in self.state.paused_endpoints:
                unready[name] = "paused"          # operator drain
            elif not self.state.endpoint_health.get(name, {}).get("healthy", True):
                unready[name] = "circuit_open"    # consecutive probe failures
        if not scheduler_ok:
            unready["_scheduler"] = "dead"
        ready = not unready
        return JSONResponse(
            {
                "ready": ready,
                # Empty `critical` means NOTHING declared itself conversational,
                # which is a config error rather than a clean bill of health —
                # reported so it cannot masquerade as ready. (An endpoint whose
                # role has not been cut over yet is the expected cause.)
                "readiness_critical_endpoints": critical,
                "unready": unready,
                "scheduler_alive": scheduler_ok,
                "reason": ("ok" if ready else
                           ", ".join(f"{k}={v}" for k, v in sorted(unready.items()))),
            },
            status_code=200 if ready else 503,
        )

    def audit_admin_ip(self, route: str, remote_ip: str) -> None:
        """Track source IPs per admin-ish route (exposed on /v1/status) and log
        the FIRST hit per (route, ip) — the data the ACL-tightening go/no-go
        needs, without per-hit log volume."""
        seen = self.state.admin_ips_seen.setdefault(route, set())
        if remote_ip not in seen:
            seen.add(remote_ip)
            logger.info("admin-audit: first hit on %s from %s", route, remote_ip)
