"""Tests for the duration-weighted cost model."""

import pytest

from originfleet.llmproxy.cost_model import (
    CostModel,
    EndpointCostModel,
    EWMATracker,
    estimate_input_tokens,
)


class TestEWMATracker:
    def test_first_sample_is_value(self):
        e = EWMATracker(alpha=0.1)
        e.update(100.0)
        assert e.value == 100.0
        assert e.sample_count == 1

    def test_converges_to_mean(self):
        e = EWMATracker(alpha=0.1)
        for _ in range(200):
            e.update(50.0)
        assert abs(e.value - 50.0) < 0.1

    def test_p95_above_mean(self):
        e = EWMATracker(alpha=0.1)
        for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            e.update(float(v))
        assert e.p95 > e.value

    def test_convergence_within_20_samples(self):
        e = EWMATracker(alpha=0.3)
        true_mean = 42.0
        for _ in range(20):
            e.update(true_mean)
        assert abs(e.value - true_mean) / true_mean < 0.1


class TestEndpointCostModel:
    def test_estimate_cost_basic(self):
        m = EndpointCostModel(endpoint="chat", max_slots=4, prefill_k=0.0004)
        m.decode_tps = [57.0, 50.0, 43.0, 38.0]
        cost = m.estimate_cost_ss(
            input_tokens=1000,
            max_output_tokens=256,
            call_site="test",
            current_occupancy=0,
        )
        # prefill = 0.0004 * 1000 = 0.4s
        # decode = 256 / 57 ≈ 4.49s (occupancy 1)
        assert 4.0 < cost < 6.0

    def test_higher_occupancy_costs_more(self):
        m = EndpointCostModel(endpoint="chat", max_slots=4, prefill_k=0.0004)
        m.decode_tps = [57.0, 50.0, 43.0, 38.0]
        cost_low = m.estimate_cost_ss(1000, 256, "test", 0)
        cost_high = m.estimate_cost_ss(1000, 256, "test", 3)
        assert cost_high > cost_low

    def test_call_site_ewma_refines_estimate(self):
        m = EndpointCostModel(endpoint="chat", max_slots=4)
        m.decode_tps = [50.0, 45.0, 40.0, 35.0]

        # First estimate uses max_tokens (conservative)
        cost_before = m.estimate_cost_ss(100, 1024, "test.site", 0)

        # Feed actual completions with much shorter output
        for _ in range(10):
            m.update_from_completion("test.site", 100, 50, 1.0, 1)

        cost_after = m.estimate_cost_ss(100, 1024, "test.site", 0)
        # After calibration, estimate should be lower (output EWMA < max_tokens)
        assert cost_after < cost_before

    def test_calibration_updates_decode_tps(self):
        m = EndpointCostModel(endpoint="chat", max_slots=4)
        m.decode_tps = [50.0, 45.0, 40.0, 35.0]

        # Observe faster-than-expected decode at occupancy 1
        for _ in range(10):
            m.update_from_completion("test", 100, 200, 2.0, 1)
            # 200 tokens in ~2s (minus prefill) ≈ 100+ tps

        assert m.decode_tps[0] > 50.0  # Should have moved upward


class TestCostModel:
    def test_register_and_estimate(self):
        cm = CostModel()
        cm.register_endpoint("chat", 4, prefill_k=0.0004, decode_tps=[57, 50, 43, 38])
        cost = cm.estimate_cost("chat", 1000, 256, "test", 0)
        assert cost > 0

    def test_unknown_endpoint_fallback(self):
        cm = CostModel()
        cost = cm.estimate_cost("nonexistent", 1000, 256, "test", 0)
        # Fallback: max_output / 40 = 6.4
        assert abs(cost - 6.4) < 0.1

    def test_update_max_slots(self):
        cm = CostModel()
        cm.register_endpoint("chat", 4)
        cm.update_max_slots("chat", 6)
        m = cm.get("chat")
        assert m.max_slots == 6
        assert len(m.decode_tps) == 6

    def test_snapshot_serializable(self):
        cm = CostModel()
        cm.register_endpoint("chat", 4)
        snap = cm.snapshot()
        assert "chat" in snap
        assert "prefill_k" in snap["chat"]
        assert "decode_tps" in snap["chat"]


class TestEstimateInputTokens:
    def test_basic(self):
        tokens = estimate_input_tokens({
            "messages": [{"role": "user", "content": "Hello world, how are you?"}]
        })
        assert tokens == len("Hello world, how are you?") // 4

    def test_empty(self):
        tokens = estimate_input_tokens({})
        assert tokens >= 1

    def test_multipart(self):
        tokens = estimate_input_tokens({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me about cats."},
            ]
        })
        total_chars = len("You are a helpful assistant.") + len("Tell me about cats.")
        assert tokens == total_chars // 4


class TestRetroactiveAdjustment:
    def test_overestimate_refunds(self):
        cm = CostModel()
        cm.register_endpoint("chat", 4, decode_tps=[50, 45, 40, 35])

        # Record a completion that was faster than the default model predicts
        cm.record_completion("chat", "test", 100, 50, 0.5, 1)
        # The model should have been updated
        m = cm.get("chat")
        assert m.output_length_ewma.get("test") is not None
