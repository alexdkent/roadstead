"""Tests for the duration-weighted cost model."""

import pytest

from roadstead.cost_model import (
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

    def test_top_level_system_string_counted(self):
        # The Anthropic-shaped `system` field (inlined into messages later by
        # backend._normalize_chat_payload) must count toward est_in.
        msg = "Tell me about cats."
        system = "S" * 400
        base = estimate_input_tokens({"messages": [{"role": "user", "content": msg}]})
        with_sys = estimate_input_tokens({
            "system": system,
            "messages": [{"role": "user", "content": msg}],
        })
        assert with_sys == base + len(system) // 4

    def test_top_level_system_list_counted(self):
        parts = [{"type": "text", "text": "A" * 100}, {"type": "text", "text": "B" * 60}]
        tokens = estimate_input_tokens({"system": parts, "messages": []})
        assert tokens == 160 // 4

    def test_tools_schemas_counted(self):
        import json as _json
        tools = [{
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search the corpus for matching files.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"}, "limit": {"type": "integer"}}},
            },
        }]
        msg = "find the report"
        with_tools = estimate_input_tokens({
            "tools": tools,
            "messages": [{"role": "user", "content": msg}],
        })
        assert with_tools == (len(msg) + len(_json.dumps(tools))) // 4

    def test_messages_only_unchanged(self):
        # No system/tools → identical to the historic messages-only estimate.
        payload = {"messages": [{"role": "user", "content": "x" * 80}]}
        assert estimate_input_tokens(payload) == 80 // 4


class TestToolHeavyPayloadsAreNotHalved:
    """A tool-calling chat transcript keeps most of its tokens OUTSIDE
    ``msg["content"]``.

    An assistant turn that calls a tool has ``content: null`` and carries the
    whole call in ``tool_calls[].function.arguments``; the tool's reply comes
    back as a separate message. A messages[].content-only walk therefore sees
    almost none of a long agentic transcript.

    Measured live on one caller (2026-08-03): est_in read 123,466 while the
    backend reported ``input_tokens`` 226,014 — a ~1.8x undercount. That
    undercount shrinks BOTH the size_stretch and the streaming TTFT allowance
    (``_STREAM_TTFT_DEADLINE_S + est_in/1000``) on exactly the callers that need
    them most. Same class of bug the ``system``/``tools`` fix already closed one
    layer up.
    """

    @staticmethod
    def _openai_tool_transcript() -> tuple[dict, int]:
        """An OpenAI-shaped tool loop, plus the true character count."""
        args = '{"path": "' + "a" * 40_000 + '"}'
        result = "R" * 60_000
        prompt = "P" * 4_000
        payload = {
            "model": "thinker",
            "stream": True,
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": args},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": result},
            ],
        }
        return payload, len(prompt) + len("read_file") + len(args) + len(result)

    def test_openai_tool_calls_arguments_are_counted(self):
        payload, real_chars = self._openai_tool_transcript()
        est = estimate_input_tokens(payload)
        assert est >= real_chars * 0.9 / 4, (
            f"est_in {est} is far below the {real_chars // 4} tokens actually "
            "in this transcript — the tool_calls arguments were skipped")

    def test_the_undercount_would_have_been_about_half(self):
        """Guard on the guard: prove the content-only walk really does miss most
        of this payload, so the assertion above cannot pass vacuously."""
        payload, real_chars = self._openai_tool_transcript()
        content_only = sum(
            len(m.get("content") or "") for m in payload["messages"])
        assert content_only < real_chars * 0.7, (
            "fixture does not reproduce the undercount")

    def test_anthropic_tool_use_and_tool_result_blocks_are_counted(self):
        """The other wire shape the proxy accepts: tool traffic lives in typed
        CONTENT BLOCKS, and only ``{"type": "text"}`` blocks were being read."""
        tool_input = {"query": "Q" * 30_000}
        result_text = "R" * 50_000
        payload = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "search",
                     "input": tool_input},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1",
                     "content": [{"type": "text", "text": result_text}]},
                ]},
            ],
        }
        est = estimate_input_tokens(payload)
        assert est >= (30_000 + 50_000) * 0.9 / 4, (
            f"est_in {est} misses the tool_use input / tool_result content")

    def test_a_tool_result_carried_as_a_plain_string_block_is_counted(self):
        body = "R" * 20_000
        payload = {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": body},
        ]}]}
        assert estimate_input_tokens(payload) >= 20_000 * 0.9 / 4

    def test_legacy_function_call_shape_is_counted(self):
        args = '{"q": "' + "z" * 10_000 + '"}'
        payload = {"messages": [
            {"role": "assistant", "content": None,
             "function_call": {"name": "search", "arguments": args}},
        ]}
        assert estimate_input_tokens(payload) >= 10_000 * 0.9 / 4

    def test_binary_parts_are_not_counted_as_characters(self):
        """An image part's data URI is not prompt text — counting its base64
        would overstate est_in by megabytes."""
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + "A" * 200_000}},
        ]}]}
        assert estimate_input_tokens(payload) < 1_000

    def test_malformed_messages_do_not_raise(self):
        for payload in (
            {"messages": [None, 7, "loose string"]},
            {"messages": [{"role": "assistant", "tool_calls": "not-a-list"}]},
            {"messages": [{"role": "assistant", "tool_calls": [None, 3]}]},
            {"messages": [{"content": [None, 3, {"type": "text"}]}]},
        ):
            assert estimate_input_tokens(payload) >= 1


class TestZeroSlotEndpoint:
    def test_completion_after_slots_drop_to_zero_does_not_crash(self):
        # endpoint_loss regression: update_max_slots(0) clears decode_tps; a
        # late completion for that endpoint then IndexError'd on the empty
        # curve inside scheduler.complete (crashed the endpoint_loss sim, and
        # would have crashed a live dispatch task the same way).
        cm = CostModel()
        cm.register_endpoint("chat", 4)
        cm.update_max_slots("chat", 0)
        cm.record_completion(
            endpoint="chat", call_site="t", input_tokens=500,
            output_tokens=200, duration_s=3.0, occupancy_during=2,
        )  # must not raise
        # Estimation still serves the fallback path.
        assert cm.estimate_cost(("chat"), 500, 200, "t", 0) > 0


class TestRetroactiveAdjustment:
    def test_overestimate_refunds(self):
        cm = CostModel()
        cm.register_endpoint("chat", 4, decode_tps=[50, 45, 40, 35])

        # Record a completion that was faster than the default model predicts
        cm.record_completion("chat", "test", 100, 50, 0.5, 1)
        # The model should have been updated
        m = cm.get("chat")
        assert m.output_length_ewma.get("test") is not None
