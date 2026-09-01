"""The request-rate threshold — the abuse control DRR is not.

DRR is fairness *under contention*. A caller alone on a quiet fleet is
unthrottled by design, which is correct for fairness and exactly why it does not
bound a runaway: a caller stuck in a loop at three in the morning contends with
nobody. `requests_per_minute` closes that gap.

🚨 **It closes it by DEGRADING, and there is no error code.** A 429 would be the
fourth spelling of "no" that three workstreams have now declined to mint, and it
would put a mistyped threshold in a position to take a caller offline — which is
the same argument that keeps a spend cap from refusing anything.

Four doctrines are pinned here, each observed going red:

1. **It never rejects.** No path from an over-rate caller to an error.
2. **Degradations do NOT stack** — over both thresholds is one band, not two.
3. **It never costs local capacity**, only a band and paid spill.
4. **The two crossings are reported separately**, because the fixes differ.
"""

from __future__ import annotations

import json
import time

import pytest

from roadstead import hooks
from roadstead.config import AgentQuotaConfig, LLMPriority, ProxyConfig
from roadstead.rate import WINDOW_S, RateLedger, RateStanding, standing
from roadstead.service import ProxyService

from tests.admin_key import ADMIN_HEADERS, enrol_admin


class _Req:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    # 🚨 NOT authenticated by default, unlike most files' `_Req`. This one is
    # also used for `handle_submit` — the inference door — where presenting a
    # credential changes WHICH CALLER the request counts against, and the whole
    # file is about a named caller's observed rate. The admin read below passes
    # the credential explicitly instead.
    headers: dict = {}


def _svc(tmp_path, **cfg) -> ProxyService:
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db"), **cfg))
    enrol_admin(svc)
    return svc


# ---------------------------------------------------------------------------
# The ledger — a measurement, not a gate
# ---------------------------------------------------------------------------

def test_the_window_slides_and_a_stopped_caller_stops_being_over():
    """🚨 Extrapolated from the WINDOW, not from the process lifetime.

    A caller that sent 600 requests in its first second and nothing since is not
    going at 600/min, and degrading it for something it has already stopped
    doing is a penalty nobody can act on — the caller has already fixed it.
    """
    led = RateLedger()
    for i in range(120):
        led.record("a", 1000.0 + i * 0.1)          # 120 requests in 12 seconds
    assert led.observed("a", 1005.0) == 120.0
    # A minute later the window is empty, so the rate is zero.
    assert led.observed("a", 1000.0 + WINDOW_S + 20) == 0.0


def test_the_ledger_is_bounded_and_undercounts_in_the_safe_direction():
    """🚨 A caller at 100k/min would otherwise hold 100k floats.

    When the cap bites the observed rate is an UNDERCOUNT — it can only fail to
    degrade a caller, never degrade one that was behaving. An overcount would be
    a penalty caused by the bookkeeping rather than by the caller.
    """
    from roadstead.rate import _MAX_SAMPLES
    led = RateLedger()
    for i in range(_MAX_SAMPLES * 2):
        led.record("a", 1000.0 + i * 0.000001)
    assert len(led.windows["a"]) <= _MAX_SAMPLES


def test_stale_callers_are_pruned():
    """An `agent_id` is a caller-supplied string on the address path, so a dict
    keyed on it grows without bound unless something empties it."""
    led = RateLedger()
    led.record("ghost", 1000.0)
    led.record("live", 1000.0)
    assert led.prune(1000.0 + WINDOW_S + 1) == 2
    assert led.windows == {}


# ---------------------------------------------------------------------------
# The standing — the same shape as spend, on purpose
# ---------------------------------------------------------------------------

def test_no_threshold_means_no_threshold():
    """What every caller was before this field existed, and the default."""
    led = RateLedger()
    for i in range(600):
        led.record("a", 1000.0 + i * 0.01)
    st = standing(led, "a", None, now=1006.0)
    assert st.over is False
    assert st.may_spill is True
    assert st.effective_priority(LLMPriority.P1_TURN_SUPPORT) is LLMPriority.P1_TURN_SUPPORT


def test_zero_is_a_real_threshold_and_not_an_absent_one():
    """🚨 The same explicit None check as `spend.SpendStanding.over`, and the
    same reason: `null` and `0` are one keystroke apart and mean opposite
    things — "no threshold" against "this caller should not be sending"."""
    led = RateLedger()
    led.record("a", 1000.0)
    assert standing(led, "a", 0.0, now=1000.0).over is True
    assert standing(led, "a", None, now=1000.0).over is False


def test_crossing_it_costs_one_band_floored_and_never_more():
    """One step down, however far over. A proportional penalty would make the
    degradation unbounded, and an unbounded penalty is a rejection wearing a
    different hat."""
    st = RateStanding("a", observed_per_min=100_000.0, limit_per_min=1.0)
    assert st.over and not st.may_spill
    assert st.effective_priority(LLMPriority.P0_REALTIME) is LLMPriority.P1_TURN_SUPPORT
    # Floored at the lowest band — there is nowhere further down to go.
    assert st.effective_priority(LLMPriority.P4_HYGIENE) is LLMPriority.P4_HYGIENE


# ---------------------------------------------------------------------------
# 🚨 Non-stacking
# ---------------------------------------------------------------------------

def test_over_both_thresholds_is_one_band_not_two(tmp_path):
    """🚨 THE rule this field had to be added without breaking.

    Two independent one-band penalties would mean adding a second threshold
    silently doubled the first one's. A caller over both is behind everyone
    behaving themselves, which is all any of this is for; two bands down would
    be a rejection taking its time.
    """
    svc = _svc(tmp_path)
    svc._state.config.agents["greedy"] = AgentQuotaConfig(
        agent_id="greedy", daily_spend_usd=0.0, requests_per_minute=0.0)
    now = time.time()
    svc._state.rate.record("greedy", now)
    svc._state.spend.charge("greedy", "remote", input_tokens=1_000_000,
                            output_tokens=0)

    assert svc._state.spend_standing("greedy").over is True
    assert svc._state.rate_standing("greedy").over is True
    # ONE step.
    assert svc._state.effective_priority(
        "greedy", LLMPriority.P0_REALTIME) is LLMPriority.P1_TURN_SUPPORT
    assert svc._state.effective_priority(
        "greedy", LLMPriority.P2_POST_TURN) is LLMPriority.P3_INGESTION


def test_either_threshold_alone_still_demotes(tmp_path):
    """The control: without it the test above passes on a proxy that demotes
    nobody, which is not the property being claimed."""
    svc = _svc(tmp_path)
    svc._state.config.agents["fast"] = AgentQuotaConfig(
        agent_id="fast", requests_per_minute=0.0)
    svc._state.rate.record("fast", time.time())
    assert svc._state.effective_priority(
        "fast", LLMPriority.P1_TURN_SUPPORT) is LLMPriority.P2_POST_TURN

    svc._state.config.agents["calm"] = AgentQuotaConfig(agent_id="calm")
    assert svc._state.effective_priority(
        "calm", LLMPriority.P1_TURN_SUPPORT) is LLMPriority.P1_TURN_SUPPORT


def test_either_threshold_removes_paid_spill(tmp_path):
    """An AND, not an or — and unlike the band, "may not spill" has no second
    step, so there is nothing here to stack."""
    svc = _svc(tmp_path)
    svc._state.config.agents["fast"] = AgentQuotaConfig(
        agent_id="fast", requests_per_minute=0.0)
    svc._state.rate.record("fast", time.time())
    assert svc._state.spend_may_spill("fast") is False
    assert svc._state.spend_may_spill("nobody") is True


# ---------------------------------------------------------------------------
# 🚨 It never rejects, and it mints no code
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_over_rate_caller_is_served_not_refused(tmp_path):
    """🚨 The whole doctrine, end to end.

    A caller far over its threshold gets a 200, from LOCAL capacity, having lost
    exactly one band. There is deliberately no branch that can turn it into an
    error — admission control is about capacity, and a mistyped threshold must
    not be able to take a caller offline.
    """
    from roadstead.backend import BackendResponse

    svc = _svc(tmp_path)
    svc._state.config.agents["runaway"] = AgentQuotaConfig(
        agent_id="runaway", requests_per_minute=0.0)

    async def ok_call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        return BackendResponse(
            status_code=200,
            body={"choices": [{"message": {"content": "y"}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
            duration_s=0.01, input_tokens=1, output_tokens=1, finish_reason="stop")
    svc._backend.call = ok_call
    await svc.startup()
    try:
        body = {"agent_id": "runaway", "endpoint": "chat",
                "priority": "P1_TURN_SUPPORT", "call_site": "t",
                "payload_type": "chat_completion",
                "payload": {"messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 4},
                "timeout_s": 10.0}
        for _ in range(3):
            resp = await svc.handle_submit(body, _Req())
            assert resp.status_code == 200, resp.body[:200]
        # It IS over, and it was served anyway.
        assert svc._state.rate_standing("runaway").over is True
    finally:
        await svc.shutdown()


def test_no_rate_shaped_error_code_exists():
    """🚨 The absence from §2.1 is the contract, not an omission.

    The mirror of `test_spend.py::test_no_error_code_exists_for_a_spend_threshold`,
    and the same shape: §2.1 enumerates every machine-readable `code` a caller
    can receive, and the day somebody adds `rate_limited` to it this fails and
    makes them say so out loud.

    🚨 Scoped to the CODE TABLE, not to §2's prose — §2.3 legitimately explains
    that a *backend's* 429 is deferrable, which is a statement about somebody
    else's status code and not about one we mint.
    """
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "api.md").read_text()
    codes = doc.split("### 2.1 Codes", 1)[1].split("### 2.2", 1)[0]
    for forbidden in ("rate_limited", "too_many_requests", "rate_limit_exceeded",
                      "rate_exceeded", "throttled"):
        assert forbidden not in codes, (
            f"a rate-related error code {forbidden!r} appeared in the error "
            "contract — a threshold that can refuse a request is no longer a "
            "threshold that degrades")


def test_the_source_mints_no_rate_refusal():
    """The document is one end of the pin; the code is the other.

    Two assertions, and the shape of the second matters. A prose mention of 429
    is fine — `rate.py` makes several, because explaining why it does NOT answer
    one is the module's whole argument — and so is the 429 `lifecycle.py`
    already returns for **backpressure**, which is a capacity signal with its
    own documented code and is reachable from queue depth rather than from any
    threshold. What must not exist is a rate-shaped code, or a status code in
    the modules that join a ledger to a policy.
    """
    import re
    from pathlib import Path
    pkg = Path(__file__).resolve().parents[1] / "roadstead"

    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        text = path.read_text()
        for pattern in (r'"rate_limited"', r'"too_many_requests"',
                        r'"rate_limit_exceeded"', r'"throttled"'):
            for m in re.finditer(pattern, text):
                offenders.append(
                    f"{path.name}:{text[:m.start()].count(chr(10)) + 1}")
    assert not offenders, (
        f"a rate-shaped error code is minted in the package: {offenders}")

    # 🚨 The threshold path itself. `rate.py` and `spend.py` are pure modules
    # (tests/test_pure_modules.py already forbids a framework import), and
    # `state.py` is where both ledgers meet their policy — the one place a
    # threshold could grow a response without anybody noticing it had.
    for name in ("rate.py", "spend.py", "state.py"):
        text = (pkg / name).read_text()
        assert not re.search(r"status_code\s*=", text), (
            f"{name} constructs a response — a threshold that can answer a "
            "request is no longer a threshold that degrades (docs/api.md §1.6)")


# ---------------------------------------------------------------------------
# 🚨 The two crossings are reported separately
# ---------------------------------------------------------------------------

def test_each_threshold_reports_through_its_own_notice(tmp_path):
    """An operator needs to know WHICH threshold moved a caller: one means
    "look at the bill", the other means "look for a loop". A merged notice would
    make a rate problem look like a billing one."""
    seen: list[dict] = []
    hooks.set_degradation_sink(lambda **kw: seen.append(kw))
    try:
        svc = _svc(tmp_path)
        svc._state.config.agents["both"] = AgentQuotaConfig(
            agent_id="both", daily_spend_usd=0.0, requests_per_minute=0.0)
        svc._state.rate.record("both", time.time())
        svc._state.spend.charge("both", "remote", input_tokens=1_000_000,
                                output_tokens=0)
        svc._state.spend_demote("both", LLMPriority.P1_TURN_SUPPORT)
    finally:
        hooks.set_degradation_sink(None)

    components = {n["component"] for n in seen}
    assert components == {"spend", "rate"}, seen
    rate_notice = next(n for n in seen if n["component"] == "rate")
    assert "not a rate LIMIT" in rate_notice["impact"]
    assert "local capacity is unaffected" in rate_notice["impact"]
    assert rate_notice["limit_per_min"] == 0.0


def test_the_notice_fires_once_a_day_not_once_a_request(tmp_path):
    """A report per call would bury the event it exists to surface."""
    seen: list[dict] = []
    hooks.set_degradation_sink(lambda **kw: seen.append(kw))
    try:
        svc = _svc(tmp_path)
        svc._state.config.agents["fast"] = AgentQuotaConfig(
            agent_id="fast", requests_per_minute=0.0)
        for _ in range(5):
            svc._state.rate.record("fast", time.time())
            svc._state.spend_demote("fast", LLMPriority.P1_TURN_SUPPORT)
    finally:
        hooks.set_degradation_sink(None)
    assert len(seen) == 1, seen


@pytest.mark.asyncio
async def test_a_read_of_the_plane_fires_no_notice(tmp_path):
    """🚨 Why `effective_priority` is split from `spend_demote`.

    The management plane reports the effective band on a read; a read that fired
    a degradation notice would make opening a dashboard look like a caller
    misbehaving, and the once-a-day suppression would then hide the real one.
    """
    seen: list[dict] = []
    svc = _svc(tmp_path, admin_store_path=str(tmp_path / "o.json"))
    svc._state.config.agents["fast"] = AgentQuotaConfig(
        agent_id="fast", requests_per_minute=0.0)
    svc._state.rate.record("fast", time.time())
    hooks.set_degradation_sink(lambda **kw: seen.append(kw))
    try:
        admin_req = _Req()
        admin_req.headers = dict(ADMIN_HEADERS)
        resp = await svc.handle_admin_callers(admin_req)
    finally:
        hooks.set_degradation_sink(None)
    assert resp.status_code == 200
    assert seen == [], seen
    row = next(r for r in json.loads(resp.body)["callers"]
               if r["agent_id"] == "fast")
    assert row["rate"]["over"] is True
    # AgentQuotaConfig defaults to P1_TURN_SUPPORT, so one step down is P2.
    assert row["spend"]["declared_priority"] == "P1_TURN_SUPPORT"
    assert row["spend"]["effective_priority"] == "P2_POST_TURN"
