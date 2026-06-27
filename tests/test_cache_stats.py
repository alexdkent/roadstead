"""Unit tests for the prefix-cache observability core (llmproxy/cache_stats.py).

Pure-function coverage: the cache-ability screen (LCP/Jaccard/verdict/ROI),
payload flattening, and the /v1/fleet/cache-stats contract builder (per-model
actual + windowed hit-rate, offenders, rollup, trend, drift).
"""
from __future__ import annotations

import json

from originfleet.llmproxy import cache_stats as cs


def test_prompt_text_shapes():
    assert "hello world" in cs.prompt_text(json.dumps(
        {"messages": [{"role": "user", "content": "hello world"}]}))
    # content as a list of blocks
    assert "blk" in cs.prompt_text(json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "blk"}]}]}))
    # submit envelope + bare prompt + garbage
    assert "env" in cs.prompt_text(json.dumps({"payload": {"prompt": "env"}}))
    assert cs.prompt_text("not json") == ""


def test_screen_flags_misaligned_vs_aligned():
    # MISALIGNED: prompts share lots of words but diverge at char 0 (no front prefix)
    mis = [("site.mis", f"variable-head-{i} STABLE BIG SHARED BODY of identical words here repeated")
           for i in range(8)]
    # ALIGNED: identical long front prefix, tiny variable tail
    pre = "STABLE PERSONA AND INSTRUCTIONS AND SCHEMA " * 20
    al = [("site.aln", pre + f"tail-{i}") for i in range(8)]
    rows = cs.screen(mis + al, min_reqs=3)
    by = {r["call_site"]: r for r in rows}
    assert by["site.mis"]["verdict"] == "misaligned"
    assert by["site.mis"]["lcp_pct"] < cs.MISALIGN_LCP_PCT
    assert by["site.aln"]["verdict"] == "aligned"
    assert by["site.aln"]["lcp_pct"] >= cs.MISALIGN_LCP_PCT
    # ROI-ranked, and a min_reqs filter
    assert rows == sorted(rows, key=lambda r: r["roi"], reverse=True)
    assert cs.screen([("x", "a"), ("x", "b")], min_reqs=5) == []


def test_screen_low_overlap():
    # every word carries the index → no shared words (Jaccard ~0) and prompts
    # diverge almost immediately → genuinely unique, nothing to cache.
    rows = cs.screen([("u", f"alpha{i} beta{i} gamma{i} delta{i} epsilon{i} zeta{i} eta{i}")
                      for i in range(6)], min_reqs=3)
    assert rows and rows[0]["verdict"] == "low-overlap"


def _snap(at, ep, hits, queries, screen):
    return {"snapshot_at": at, "endpoint": ep, "cum_hits": hits,
            "cum_queries": queries, "screen": screen}


def test_build_fleet_payload_actual_and_window():
    labels = {"creative": "creative", "chat": "analyst"}
    engines = {"creative": "vllm", "chat": "llama.cpp"}
    snaps = [
        _snap(100.0, "creative", 100, 1000, [{"call_site": "a", "verdict": "misaligned",
                                              "reqs": 10, "lcp_pct": 2.0, "jacc": 0.6,
                                              "wasted_tokens": 50, "roi": 500}]),
        _snap(200.0, "creative", 300, 2000, [{"call_site": "a", "verdict": "misaligned",
                                              "reqs": 10, "lcp_pct": 2.0, "jacc": 0.6,
                                              "wasted_tokens": 50, "roi": 500}]),
        # llama.cpp endpoint: no counters → actual n/a, screen still present
        _snap(200.0, "chat", None, None, [{"call_site": "b", "verdict": "low-overlap",
                                          "reqs": 5, "lcp_pct": 1.0, "jacc": 0.1,
                                          "wasted_tokens": 0, "roi": 0}]),
    ]
    out = cs.build_fleet_payload(snaps, labels, engines)
    m = {x["endpoint"]: x for x in out["models"]}
    # creative: lifetime 300/2000=0.15; window Δ=(300-100)/(2000-1000)=0.2
    assert m["creative"]["actual_hit_rate"] == 0.15
    assert m["creative"]["window_hit_rate"] == 0.2
    assert m["creative"]["engine"] == "vllm"
    # llama.cpp: actual unavailable
    assert m["chat"]["actual_hit_rate"] is None
    assert m["chat"]["window_hit_rate"] is None
    # offenders only the misaligned one
    assert [o["call_site"] for o in out["offenders"]] == ["a"]
    assert out["rollup"]["captured_pct"] == 0.2  # fleet Δhits/Δqueries


def test_build_fleet_payload_drift():
    labels = {"creative": "creative"}
    engines = {"creative": "vllm"}
    good = {"call_site": "c", "verdict": "aligned", "reqs": 5, "lcp_pct": 80.0,
            "jacc": 0.7, "wasted_tokens": 0, "roi": 0}
    broke = {**good, "verdict": "misaligned", "lcp_pct": 5.0, "wasted_tokens": 40, "roi": 200}
    snaps = [_snap(1.0, "creative", 1, 10, [good]),
             _snap(2.0, "creative", 2, 20, [good]),
             _snap(3.0, "creative", 3, 30, [broke])]
    out = cs.build_fleet_payload(snaps, labels, engines)
    assert out["drift"] and out["drift"][0]["call_site"] == "c"
    assert out["drift"][0]["from"] >= cs.MISALIGN_LCP_PCT and out["drift"][0]["to"] == 5.0
