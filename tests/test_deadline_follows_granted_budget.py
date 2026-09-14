"""A correction that GRANTS tokens must also grant the clock to spend them.

🚨 THE DEFECT. The deadline is resolved at admission from the CALLER's
`max_tokens`; `apply_thinking` and `apply_forced_reasoning_budget` inflate that
number afterwards. So the proxy hands out a budget it does not give time to
spend. Measured on tier2-analyst, the Playground's own default:

    caller max_tokens          2,048
    deadline sized from        2,048   -> class floor 120 s
    backend receives           4,048   (+2,000 thinking headroom)
    decode at declared 20 tok/s        ~202 s

A request that USES what it was granted cannot finish in time. Not a stingy
number — an inconsistent PAIR of numbers, and the previous remedy shrank the
budget to fit the clock, which is the truncation the headroom existed to stop.
"""
from __future__ import annotations

import types

import pytest

from roadstead.lifecycle import Lifecycle


class _State:
    """Only the two surfaces the method touches."""

    def __init__(self, rec):
        self._rec = rec
        self.asked: list[tuple] = []

    def effective_timeout_advice(self, endpoint, priority, est_in, est_out):
        self.asked.append((endpoint, priority, est_in, est_out))
        return {"recommended_timeout_s": self._rec}


def _lc(rec):
    lc = object.__new__(Lifecycle)
    lc.state = _State(rec)
    return lc


def _req(*, mt, timeout_s, endpoint="tier2-analyst"):
    return types.SimpleNamespace(
        payload={"max_tokens": mt, "messages": [{"role": "user", "content": "hi"}]},
        endpoint=endpoint, timeout_s=timeout_s, priority=None,
        call_site="test", request_id="r1")


def test_the_deadline_is_extended_to_cover_the_granted_tokens():
    """THE REGRESSION, in the Playground's own numbers."""
    lc = _lc(rec=210.0)
    req = _req(mt=4048, timeout_s=120.0)       # already inflated by the correction
    lc.extend_deadline_for_granted_budget(req, 2048)
    assert req.timeout_s == 210.0, (
        "the deadline still covers only the caller's original budget — the "
        "request cannot spend what the proxy granted it")


def test_the_advice_is_re_asked_with_the_INFLATED_budget():
    """The whole point: est_out must be the budget the BACKEND will see, not
    the one the caller sent. Asking with the old number reproduces the bug with
    an extra function call in front of it."""
    lc = _lc(rec=210.0)
    req = _req(mt=4048, timeout_s=120.0)
    lc.extend_deadline_for_granted_budget(req, 2048)
    assert lc.state.asked, "no advice was requested at all"
    assert lc.state.asked[-1][3] == 4048, (
        f"advice was re-derived from est_out={lc.state.asked[-1][3]}, not the "
        f"inflated 4048")


def test_it_never_SHRINKS_a_deadline():
    """A caller that asked for longer keeps it; a re-derivation must not be able
    to shorten a live request's clock."""
    lc = _lc(rec=150.0)
    req = _req(mt=4048, timeout_s=600.0)
    lc.extend_deadline_for_granted_budget(req, 2048)
    assert req.timeout_s == 600.0


def test_no_inflation_means_no_change_and_no_advice_call():
    """Transparent when nothing was granted — the overwhelming majority of
    requests, which must not pay for an extra advice lookup."""
    lc = _lc(rec=999.0)
    req = _req(mt=2048, timeout_s=120.0)
    lc.extend_deadline_for_granted_budget(req, 2048)
    assert req.timeout_s == 120.0
    assert not lc.state.asked, "advice was consulted for an uninflated request"


@pytest.mark.parametrize("before,mt", [(None, 4048), (2048, None), (0, 4048)])
def test_unusable_inputs_leave_the_deadline_alone(before, mt):
    """A caller that sent no cap has no inflation to pay for, and a re-derive
    fault must never touch the clock."""
    lc = _lc(rec=999.0)
    req = _req(mt=mt, timeout_s=120.0)
    lc.extend_deadline_for_granted_budget(req, before)
    assert req.timeout_s == 120.0


def test_an_advice_fault_leaves_the_deadline_untouched():
    """Total: the failure direction is 'unchanged', never 'shorter'."""
    class _Boom:
        def effective_timeout_advice(self, *a, **k):
            raise RuntimeError("advice is down")
    lc = object.__new__(Lifecycle)
    lc.state = _Boom()
    req = _req(mt=4048, timeout_s=120.0)
    lc.extend_deadline_for_granted_budget(req, 2048)
    assert req.timeout_s == 120.0
