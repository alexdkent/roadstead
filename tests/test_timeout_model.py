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


# ----- adaptive load × size uplift + split ceilings (2026-07-05) -----

from originfleet.llmproxy.timeout_model import (  # noqa: E402
    _BACKGROUND_CEILING_S,
    _INTERACTIVE_CEILING_S,
    _SIZE_STRETCH_REF_TOKENS,
    apply_load_and_ceiling,
    resolve_ceiling_s,
    size_stretch,
    surge_factor,
)


def test_surge_is_one_at_or_under_capacity():
    # in_flight + queued <= max_slots → no backlog → factor 1.0 (never shrinks).
    assert surge_factor(0, 0, 4) == 1.0
    assert surge_factor(4, 0, 4) == 1.0
    assert surge_factor(2, 2, 4) == 1.0


def test_surge_scales_with_backlog_and_clamps():
    # backlog of one capacity-unit over → 1 + k(0.5)*1 = 1.5
    assert surge_factor(4, 4, 4) == 1.5
    # huge backlog clamps `over` at surge_max (3.0) → 1 + 0.5*3 = 2.5
    assert surge_factor(100, 0, 4) == 2.5


def test_surge_unknown_capacity_is_neutral():
    assert surge_factor(10, 10, 0) == 1.0


def test_size_stretch_only_past_the_reference():
    """The 16K..82K band is unchanged by the D1 recalibration (2026-08-03); the
    curve past it lives in ``test_timeout_sizing.py``.  Note the reference is
    ``_SIZE_STRETCH_REF_TOKENS``, deliberately NOT ``_IN_EDGES[-1]`` — widening
    the empirical buckets must not move where the stretch starts."""
    ref = _SIZE_STRETCH_REF_TOKENS
    assert size_stretch(1000) == 1.0
    assert size_stretch(ref) == 1.0                      # exactly at the 16K ref
    # 2x the reference → (32768-16384)/16384 = 1 → 1.5
    assert size_stretch(ref * 2) == 1.5
    # the legacy linear term still clamps at size_max (4) → 1 + 0.5*4 = 3.0 …
    assert size_stretch(ref * 5) == 3.0
    # … but past it the superlinear term takes over instead of flatlining.
    assert size_stretch(ref * 100) > 3.0


def test_ceiling_tier_bands_and_role_override():
    # interactive tier → interactive band; background tier → background band.
    assert resolve_ceiling_s("thinker", interactive=True, floor_s=180) == _INTERACTIVE_CEILING_S
    assert resolve_ceiling_s("thinker", interactive=False, floor_s=180) == _BACKGROUND_CEILING_S
    # a per-role override wins over the band, on EITHER tier (creative song-
    # compose runs at an interactive tier but must keep its generous ceiling).
    assert resolve_ceiling_s(
        "creative", interactive=True, role_ceilings={"creative": 1800}, floor_s=900,
    ) == 1800.0


def test_ceiling_never_below_floor():
    # A band tighter than the class floor must be lifted to the floor — a
    # ceiling that strangles below the model's guaranteed deadline would
    # re-introduce the sub-floor-cliff regression.
    assert resolve_ceiling_s("creative", interactive=True, floor_s=900) == 900.0


def test_apply_load_and_ceiling_bounds_and_reports_factors():
    # No load, small prompt → recommendation unchanged, factors 1.0.
    eff, surge, stretch = apply_load_and_ceiling(
        180_000.0, in_flight=0, queued=0, max_slots=4, est_in=100,
        ceiling_ms=600_000.0,
    )
    assert (eff, surge, stretch) == (180_000.0, 1.0, 1.0)
    # Load + big prompt widen the deadline, but the ceiling caps it.
    eff2, surge2, stretch2 = apply_load_and_ceiling(
        400_000.0, in_flight=100, queued=0, max_slots=4, est_in=_IN_EDGES[-1] * 4,
        ceiling_ms=600_000.0,
    )
    assert surge2 > 1.0 and stretch2 > 1.0
    assert eff2 == 600_000.0  # capped at ceiling


def test_ceiling_yaml_sync():
    """The per-role ceiling overrides in models.yaml are parsed and non-empty.

    Companion (122B, slow) + creative (long-form author) carry an explicit
    timeout_ceiling_s so the interactive band can't strangle them. This pins
    that the yaml field is wired through build_class_ceilings (mirrors the
    timeout_floor_yaml_sync doctrine)."""
    from originfleet.llmproxy.model_catalog import build_class_ceilings

    ceilings = build_class_ceilings()
    assert ceilings, "models.yaml declares no timeout_ceiling_s — regression"
    # every declared ceiling is a positive float
    assert all(isinstance(v, float) and v > 0 for v in ceilings.values())
