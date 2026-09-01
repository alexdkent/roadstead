"""Tests for the unified ProxyTestHarness."""

from __future__ import annotations

import json
import time

import pytest

from roadstead.config import ProxyConfig
from roadstead.queue import PersistentQueue
from roadstead.test_harness import (
    ABReport,
    ABResult,
    ProxyTestHarness,
    ReplayReport,
    ShapingConfig,
    _check_json_parseable,
    _jaccard,
    _percentile,
)


def test_percentile_basic():
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([1, 2, 3, 4, 5], 95) == 5
    assert _percentile([], 50) == 0.0


def test_jaccard_identical():
    assert _jaccard("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert _jaccard("hello world", "foo bar") == 0.0


def test_jaccard_partial():
    j = _jaccard("hello world foo", "hello world bar")
    assert 0.4 < j < 0.7


def test_jaccard_empty():
    assert _jaccard("", "") == 1.0
    assert _jaccard("hello", "") == 0.0


def test_check_json_parseable_valid():
    resp = {"choices": [{"message": {"content": '{"key": "value"}'}}]}
    assert _check_json_parseable(resp) is True


def test_check_json_parseable_plain_text():
    resp = {"choices": [{"message": {"content": "just plain text"}}]}
    assert _check_json_parseable(resp) is True


def test_check_json_parseable_broken_json():
    resp = {"choices": [{"message": {"content": '{"key": broken}'}}]}
    assert _check_json_parseable(resp) is False


def test_sim_mode_runs():
    harness = ProxyTestHarness()
    result = harness.run_sim("all_agents_burst")
    assert result.total_submitted > 0
    assert result.total_dispatched > 0


def test_export_corpus_for_replay(tmp_path):
    db_path = str(tmp_path / "queue.db")
    q = PersistentQueue(db_path)

    payload = {"messages": [{"role": "user", "content": "test prompt"}]}
    response = {"choices": [{"message": {"content": "test response"}}]}

    q.persist_complete(
        "req_001", "forum-agent", "tier3", "forum-agent.proposal_emitter", 3,
        6000, 250, 30.0, 5.0, "ok",
        payload=payload, response=response,
    )

    corpus = q.export_corpus(hours=1, endpoint="tier3")
    assert len(corpus) == 1
    assert corpus[0]["payload"]["messages"][0]["content"] == "test prompt"
    assert corpus[0]["response"]["choices"][0]["message"]["content"] == "test response"


def test_ab_report_summary():
    report = ABReport(
        corpus_size=100,
        shadow_success_count=95,
        shadow_error_count=5,
        primary_p50_duration_ms=500,
        primary_p95_duration_ms=2000,
        shadow_p50_duration_ms=300,
        shadow_p95_duration_ms=1500,
        primary_decode_tps=9.6,
        shadow_decode_tps=20.0,
        json_parse_rate_primary=99.0,
        json_parse_rate_shadow=98.0,
        avg_token_overlap_jaccard=0.85,
        avg_length_ratio=1.05,
    )
    summary = report.summary()
    assert "A/B Comparison Report" in summary
    assert "100" in summary
    assert "9.6" in summary
    assert "20.0" in summary


def test_replay_report_summary():
    report = ReplayReport(
        corpus_size=50,
        replayed=48,
        errors=2,
        p50_wait_ms=3.0,
        p95_wait_ms=15.0,
        baseline_p50_wait_ms=61000,
        baseline_p95_wait_ms=84000,
    )
    summary = report.summary()
    assert "Replay Report" in summary
    assert "50" in summary


def test_shaping_config_defaults():
    s = ShapingConfig()
    assert s.compress == 1.0
    assert s.multiply == 1
    assert s.priority_remap == {}
