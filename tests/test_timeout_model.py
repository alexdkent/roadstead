"""Unit tests for the timeout-advice model (pure computation).

Covers bucketing, percentile method, the coarsening fallback hierarchy,
floor enforcement, the ``recommended = max(p99*margin, floor)`` rule,
ok-only sampling, age pruning, role→class normalization, and a doctrine
check that every endpoint class has a floor.
"""

from __future__ import annotations

from originfleet.llmproxy.config import DEFAULT_ENDPOINTS
from originfleet.llmproxy.timeout_model import (
    FLOOR_S,
    TimeoutModel,
    _bucket,
    _IN_EDGES,
    _OUT_EDGES,
    percentile,
)


# ----- primitives -----

def test_bucket_edges():
    # _OUT_EDGES = [128, 512, 2048, 8192] → 5 buckets; bisect_left puts an
    # exact edge value in the lower bucket.
    assert _bucket(0, _OUT_EDGES) == 0
    assert _bucket(127, _OUT_EDGES) == 0
    assert _bucket(128, _OUT_EDGES) == 0
    assert _bucket(129, _OUT_EDGES) == 1
    assert _bucket(512, _OUT_EDGES) == 1
    assert _bucket(513, _OUT_EDGES) == 2
    assert _bucket(9000, _OUT_EDGES) == 4
    # _IN_EDGES = [1024, 4096, 16384] → 4 buckets
    assert _bucket(0, _IN_EDGES) == 0
    assert _bucket(20000, _IN_EDGES) == 3


def test_percentile_nearest_rank():
    vals = [10.0, 20.0, 30.0, 40.0]
    assert percentile(vals, 50) == 30.0
    assert percentile(vals, 95) == 40.0
    assert percentile([], 50) == 0.0


# ----- advise -----

def _seed(model, ep, pri, latency_ms, n, *, est_in=2000, est_out=256, now=1000.0):
    for _ in range(n):
        model.record(ep, pri, est_in, est_out, latency_ms, "ok", now)


def test_advise_uses_cell_when_enough_samples():
    m = TimeoutModel(min_samples=5)
    _seed(m, "thinker", 1, 5000.0, 10)
    advice = m.advise("thinker", 1, 2000, 256)
    assert advice["source"] == "cell"
    assert advice["sample_count"] == 10
    assert advice["median_ms"] == 5000.0
    assert advice["min_ms"] == 5000.0


def test_advise_falls_back_to_tier_then_floor():
    m = TimeoutModel(min_samples=30)
    # 10 samples in each of 4 distinct output buckets, same tier — no
    # single cell/out-bucket reaches 30, but (ep, tier) totals 40.
    for est_out in (64, 256, 1024, 4096):
        _seed(m, "thinker", 1, 5000.0, 10, est_out=est_out)
    advice = m.advise("thinker", 1, 2000, 256)
    assert advice["source"] == "tier"
    assert advice["sample_count"] == 40

    # Same model, unseen tier: the endpoint-level fallback still catches
    # the cross-tier samples (source="endpoint"), not floor.
    other_tier = m.advise("thinker", 4, 2000, 256)
    assert other_tier["source"] == "endpoint"

    # A model with no samples at all → floor only.
    cold = m.advise("embed", 1, 2000, 256)
    assert cold["source"] == "floor"
    assert cold["recommended_timeout_s"] == FLOOR_S["embed"]


def test_recommended_is_p99_times_margin_when_above_floor():
    # rerank floor is 10s; seed 10s latencies so p99*1.5 = 15s wins.
    # Query matches the seed's token buckets so it resolves at cell level.
    m = TimeoutModel(min_samples=5, margin=1.5)
    _seed(m, "rerank", 1, 10000.0, 100)
    advice = m.advise("rerank", 1, 2000, 256)
    assert advice["source"] == "cell"
    assert advice["recommended_ms"] == 15000.0
    assert advice["recommended_timeout_s"] == 15


def test_floor_dominates_when_samples_are_fast():
    # Fast 1s latencies: p99*1.5 = 1.5s < 10s floor → floor wins.
    m = TimeoutModel(min_samples=5, margin=1.5)
    _seed(m, "rerank", 1, 1000.0, 100)
    advice = m.advise("rerank", 1, 2000, 256)
    assert advice["recommended_ms"] == FLOOR_S["rerank"] * 1000.0
    assert advice["source"] == "cell"  # cell had the samples; floor still clamps


def test_only_ok_samples_counted():
    m = TimeoutModel(min_samples=1)
    m.record("thinker", 1, 2000, 256, 5000.0, "error", 1000.0)
    m.record("thinker", 1, 2000, 256, 5000.0, "timeout", 1000.0)
    m.record("thinker", 1, 2000, 256, 0.0, "ok", 1000.0)  # zero latency skipped
    assert m.advise("thinker", 1, 2000, 256)["source"] == "floor"


def test_prune_drops_stale_samples():
    m = TimeoutModel(min_samples=1, window_s=100.0)
    _seed(m, "thinker", 1, 5000.0, 5, now=1000.0)
    assert m.advise("thinker", 1, 2000, 256)["source"] == "cell"
    m.prune(now=1000.0 + 101.0)
    assert m.advise("thinker", 1, 2000, 256)["source"] == "floor"
    assert m.snapshot()["cells"] == 0


def test_role_name_normalizes_to_class():
    m = TimeoutModel(min_samples=1)
    # Record under the role name; query by class — same cell.
    _seed_ep = "llama-thinker"
    for _ in range(5):
        m.record(_seed_ep, 1, 2000, 256, 5000.0, "ok", 1000.0)
    advice = m.advise("thinker", 1, 2000, 256)
    assert advice["source"] == "cell"
    assert advice["sample_count"] == 5


def test_every_endpoint_class_has_a_floor():
    """Doctrine: a new endpoint class must add a FLOOR_S entry."""
    for ep_class in DEFAULT_ENDPOINTS:
        assert ep_class in FLOOR_S, f"missing FLOOR_S entry for {ep_class!r}"


def test_timeout_floor_yaml_sync():
    """Doctrine: the hardcoded FLOOR_S mirror MUST equal models.yaml's
    per-class ``timeout_floor_s`` for every class the yaml declares.

    models.yaml is the authoritative source (it seeds the SERVER TimeoutModel
    via build_class_floors). FLOOR_S is a synced fallback that ALSO backs the
    client-side floor_for() sub-floor-honor decision (framework/timeout_advice),
    which cannot read the catalog. If the two drift, the client honors a
    different floor than the server enforces. This test caught the inert-yaml
    bug where classify 45→180 / composer 180→360 (commit 35df6545) were bumped
    in the yaml but never took effect. Bump BOTH in lockstep; this pins it."""
    from originfleet.llmproxy.model_catalog import build_class_floors

    yaml_floors = build_class_floors()
    assert yaml_floors, "models.yaml declares no timeout_floor_s — regression"
    mismatched = {
        ep: (FLOOR_S.get(ep), floor)
        for ep, floor in yaml_floors.items()
        if FLOOR_S.get(ep) != floor
    }
    assert not mismatched, (
        "FLOOR_S mirror drifted from models.yaml timeout_floor_s "
        f"(class: (FLOOR_S, yaml)): {mismatched}"
    )
