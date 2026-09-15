"""The loop-break thresholds must actually REACH EndpointConfig.

🚨 This test exists because `_POLICY_PASSTHROUGH` drops an unlisted policy key
IN SILENCE — a declaration that never arrives is indistinguishable from a
feature deliberately left off. Same shape as the 2026-09-14 defect where a
declared `reasoning_effort` had no code path that read it.
"""
from roadstead.correction import ReasoningLoopDetector
from roadstead.model_catalog import _POLICY_PASSTHROUGH

LOOP_KEYS = ("reasoning_loop_window_chars", "reasoning_loop_min_chars",
             "reasoning_loop_max_distinct_ratio", "reasoning_loop_check_every_chars")


def test_every_loop_threshold_is_in_the_passthrough():
    missing = [k for k in LOOP_KEYS if k not in _POLICY_PASSTHROUGH]
    assert not missing, f"dropped silently by the catalog: {missing}"


def test_detector_arms_only_on_a_complete_declaration():
    full = dict(window=20000, min_chars=20000, max_distinct=0.40, check_every=4000)
    assert ReasoningLoopDetector(**full).armed is True
    for k in full:
        partial = dict(full); partial[k] = 0
        assert ReasoningLoopDetector(**partial).armed is False, f"half-armed without {k}"


def test_an_undeclared_endpoint_builds_an_inert_detector():
    assert ReasoningLoopDetector().armed is False
