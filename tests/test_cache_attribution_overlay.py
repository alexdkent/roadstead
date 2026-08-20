"""The cache-attribution overlay is a GAP-FILLER, not a preference.

`/v1/fleet/cache-attribution` computes every hit_rate honestly from summed
per-request `cached_tokens`, then overlays the vLLM `/metrics` prefix-cache rate
on top. That overlay used to be UNCONDITIONAL, which was harmless only while no
backend reported per-request counts.

It stopped being harmless on 2026-08-20 for two reasons:
  1. llama.cpp always DID report `cached_tokens` (three code comments claimed the
     opposite) — those endpoints are fully attributed and dominate the corpus;
  2. tier3's vLLM began reporting it once `--enable-prompt-tokens-details` was
     added to its launch args.

The measured symptom: the FLEET row reported `hit_rate 0.4672` beside its own
columns showing 101730/174386 = 0.5833. `state.endpoint_cache_hit_rate` is
populated only for `backend_engine == "vllm"`, so the "query-weighted mean of
real per-endpoint rates" was a mean of ONE endpoint — and those counters are
cumulative-since-backend-boot, so a window-scoped `fleet` field was carrying one
backend's LIFETIME number.

These tests pin the rule: real per-request measurement wins; the overlay only
fills a genuine gap.
"""

from __future__ import annotations

import json

import pytest

from originfleet.llmproxy.config import ProxyConfig
from originfleet.llmproxy.service import ProxyService


def _attribution(monkeypatch, rows, fleet, real):
    """Drive handle_cache_attribution with a canned rollup + canned overlay."""
    svc = ProxyService(ProxyConfig())
    payload = {"window_s": 3600, "now": 0, "by_call_site": [],
               "by_endpoint": rows, "fleet": fleet}
    monkeypatch.setattr(svc._state.queue_db, "cache_attribution",
                        lambda *a, **k: payload, raising=False)
    svc._state.endpoint_cache_hit_rate = real

    class _Req:
        query_params = {}

    import asyncio
    resp = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        svc._http.handle_cache_attribution(_Req()))
    return json.loads(bytes(resp.body).decode())


def test_overlay_does_not_clobber_a_row_with_real_per_request_data(monkeypatch):
    """The regression. An endpoint that DID report cached_tokens must keep its
    own measured rate, not have it replaced by a lifetime backend counter."""
    out = _attribution(
        monkeypatch,
        rows=[{"endpoint": "thinker", "attributed_calls": 10,
               "attributable_input_tokens": 1000, "cached_tokens": 800,
               "hit_rate": 0.8}],
        fleet={"attributable_input_tokens": 1000, "cached_tokens": 800,
               "hit_rate": 0.8},
        real={"thinker": {"hit_rate": 0.4672, "queries": 3_150_000_000,
                          "source": "backend_prefix_cache_metrics"}},
    )
    row = out["by_endpoint"][0]
    assert row["hit_rate"] == 0.8, (
        "the overlay clobbered a real per-request measurement with the backend's "
        "cumulative-since-boot counter")
    assert row.get("hit_rate_source") != "backend_prefix_cache_metrics"


def test_overlay_still_fills_a_genuine_gap(monkeypatch):
    """The overlay's original and still-valid job: an endpoint whose backend
    reports nothing per-request gets the /metrics rate rather than n/a."""
    out = _attribution(
        monkeypatch,
        rows=[{"endpoint": "thinker", "attributed_calls": 0,
               "attributable_input_tokens": 0, "cached_tokens": 0,
               "hit_rate": None}],
        fleet={"attributable_input_tokens": 0, "cached_tokens": 0,
               "hit_rate": None},
        real={"thinker": {"hit_rate": 0.4672, "queries": 100,
                          "source": "backend_prefix_cache_metrics"}},
    )
    row = out["by_endpoint"][0]
    assert row["hit_rate"] == 0.4672
    assert row["hit_rate_source"] == "backend_prefix_cache_metrics"


def test_fleet_headline_matches_its_own_columns_when_data_exists(monkeypatch):
    """🚨 THE MEASURED DEFECT. A `fleet` field must not report one backend's
    lifetime rate while the columns printed beside it describe a windowed sum of
    the whole fleet — that is a number answering a different question."""
    out = _attribution(
        monkeypatch,
        rows=[{"endpoint": "creative", "attributed_calls": 37,
               "attributable_input_tokens": 47701, "cached_tokens": 8725,
               "hit_rate": 0.1829}],
        fleet={"attributable_input_tokens": 174386, "cached_tokens": 101730,
               "hit_rate": 0.5833},
        real={"thinker": {"hit_rate": 0.4672, "queries": 3_150_000_000,
                          "source": "backend_prefix_cache_metrics"}},
    )
    fleet = out["fleet"]
    assert fleet["hit_rate"] == 0.5833, (
        f"fleet hit_rate is {fleet['hit_rate']}, but its own columns give "
        f"{fleet['cached_tokens']}/{fleet['attributable_input_tokens']} = 0.5833 — "
        "the overlay is reporting a single vLLM backend's lifetime rate under a "
        "fleet-scope, window-scoped label")


def test_fleet_overlay_applies_when_nothing_was_attributable(monkeypatch):
    """Anti-vacuity: the fleet overlay must still work in the case it was
    written for, or the fix above would have simply deleted a feature."""
    out = _attribution(
        monkeypatch,
        rows=[],
        fleet={"attributable_input_tokens": 0, "cached_tokens": 0,
               "hit_rate": None},
        real={"thinker": {"hit_rate": 0.4672, "queries": 100,
                          "source": "backend_prefix_cache_metrics"}},
    )
    assert out["fleet"]["hit_rate"] == 0.4672
    assert out["fleet"]["hit_rate_source"] == "backend_prefix_cache_metrics"
