"""The RATE alarm — a log line nobody greps is not detection.

Ledger `tier3-json-object-empty-brace`. The proxy held the evidence for 31
hours. Counting the events is not enough on its own: the operator surface has
to be something that stands up on /v1/status and reaches the health-verifier chip, and it
has to distinguish "one caller answered nothing once" from "this endpoint
stopped answering".

Window / floor / threshold and their reasoning live in
``llmproxy/observability.py``. What this file pins:

  * a low-traffic endpoint cannot alarm on 1-of-1 (the explicit requirement);
  * a healthy endpoint at a plausible background rate does not alarm;
  * the alarm reaches ``state.alerts`` through the real ``evaluate_alerts``;
  * and the REPLAY at the bottom drives the alarm with the traffic actually
    measured off the live proxy's queue.db for the incident window — before
    and after — so this is not an alarm nobody has seen fire.
"""
import importlib
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # originfleet/
sys.path.insert(0, str(REPO))

obs = importlib.import_module("roadstead.observability")
health_mod = importlib.import_module("roadstead.health")

W = obs.STRUCTURED_EMPTY_WINDOW_S
FLOOR = obs.STRUCTURED_EMPTY_MIN_SAMPLES
RATE = obs.STRUCTURED_EMPTY_ALERT_RATE


def _feed(window, endpoint, *, n_empty, n_ok, now=1000.0, call_site="cs"):
    """Interleaved on purpose — real traffic is mixed, and feeding all the
    empties first would let the deque's hard sample cap silently drop them."""
    total = n_empty + n_ok
    emitted = 0
    for i in range(total):
        want = (i + 1) * n_empty // total
        empty = want > emitted
        if empty:
            emitted += 1
        obs.record_structured_outcome(window, endpoint, empty=empty,
                                      call_site=call_site, now=now)
    assert emitted == n_empty
    return window


# ---------------------------------------------------------------------------
# The sample floor — no alarming on thin traffic
# ---------------------------------------------------------------------------

def test_one_of_one_does_not_alarm():
    """The explicit requirement: a low-traffic endpoint must not alarm on
    1-of-1. 100% of one sample is not evidence of anything."""
    w = _feed({}, "quiet", n_empty=1, n_ok=0)
    assert obs.structured_empty_rates(w, 1000.0)["quiet"]["rate"] == 1.0
    assert obs.structured_empty_alerts(w, 1000.0) == []


def test_all_empty_but_one_below_the_floor_does_not_alarm():
    """Even 100% empty stays silent until the endpoint has enough traffic to
    judge. An endpoint that never reaches the floor never evaluates — that is
    a deliberate blind spot, not an oversight (see the report / module note)."""
    w = _feed({}, "quiet", n_empty=FLOOR - 1, n_ok=0)
    assert obs.structured_empty_alerts(w, 1000.0) == []
    assert obs.structured_empty_rates(w, 1000.0)["quiet"]["evaluated"] is False


def test_exactly_at_the_floor_evaluates_and_alarms():
    w = _feed({}, "busy", n_empty=FLOOR, n_ok=0)
    alerts = obs.structured_empty_alerts(w, 1000.0)
    assert [a.name for a in alerts] == ["structured_empty_rate"]
    assert alerts[0].severity == "ERROR"


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------

def test_healthy_background_rate_does_not_alarm():
    """The measured healthy background is 0.0% across 114 evaluated buckets on
    three endpoints over two days. Even a hypothetical 5% background — 15×
    anything ever observed — must stay quiet."""
    w = _feed({}, "busy", n_empty=5, n_ok=95)
    assert obs.structured_empty_rates(w, 1000.0)["busy"]["rate"] == 0.05
    assert obs.structured_empty_alerts(w, 1000.0) == []


def test_an_alert_always_needs_at_least_three_empties():
    """floor 20 × rate 15% ⇒ never one or two. Pinned because it is the
    property the pair was chosen FOR — either constant drifting alone breaks
    it (10% of 20 is two, which would fire on noise)."""
    assert obs.STRUCTURED_EMPTY_MIN_SAMPLES * obs.STRUCTURED_EMPTY_ALERT_RATE >= 3
    assert obs.structured_empty_alerts(
        _feed({}, "busy", n_empty=2, n_ok=18), 1000.0) == []


def test_just_under_the_threshold_is_quiet_and_just_over_fires():
    total = 200
    at = round(total * RATE)
    under = _feed({}, "busy", n_empty=at - 2, n_ok=total - at + 2)
    over = _feed({}, "busy", n_empty=at + 2, n_ok=total - at - 2)
    assert obs.structured_empty_rates(under, 1000.0)["busy"]["rate"] < RATE
    assert obs.structured_empty_alerts(under, 1000.0) == []
    assert obs.structured_empty_rates(over, 1000.0)["busy"]["rate"] >= RATE
    assert len(obs.structured_empty_alerts(over, 1000.0)) == 1


def test_the_alert_names_the_endpoint_and_the_top_contributing_call_site():
    """So the operator can tell "one caller answers nothing a lot" from "this
    endpoint's grammar is broken" without opening a shell."""
    w = {}
    _feed(w, "thinker", n_empty=30, n_ok=0, call_site="auto_approve.critic")
    _feed(w, "thinker", n_empty=5, n_ok=65, call_site="knowledge.extract")
    detail = obs.structured_empty_alerts(w, 1000.0)[0].detail
    assert "thinker" in detail
    assert "auto_approve.critic" in detail
    assert "35/100" in detail


def test_a_healthy_endpoint_is_unaffected_by_a_sick_neighbour():
    w = {}
    _feed(w, "thinker", n_empty=50, n_ok=50)
    _feed(w, "gemma", n_empty=0, n_ok=200)
    names = [a.detail for a in obs.structured_empty_alerts(w, 1000.0)]
    assert len(names) == 1 and "thinker" in names[0]


# ---------------------------------------------------------------------------
# The window actually slides
# ---------------------------------------------------------------------------

def test_samples_older_than_the_window_are_dropped():
    """A cleared fault must clear the alarm — otherwise it latches and the
    next real one is invisible under it."""
    w = _feed({}, "thinker", n_empty=50, n_ok=50, now=1000.0)
    assert len(obs.structured_empty_alerts(w, 1000.0)) == 1
    later = 1000.0 + W + 1
    _feed(w, "thinker", n_empty=0, n_ok=FLOOR, now=later)
    assert obs.structured_empty_rates(w, later)["thinker"]["n"] == FLOOR
    assert obs.structured_empty_alerts(w, later) == []


def test_window_recording_is_total_on_a_broken_container():
    obs.record_structured_outcome(None, "x", empty=True, call_site="c", now=1.0)


# ---------------------------------------------------------------------------
# It reaches /v1/status.alerts through the real evaluate_alerts
# ---------------------------------------------------------------------------

def _health_with(window):
    """A Health bound to the minimum ProxyState surface evaluate_alerts touches."""
    st = types.SimpleNamespace()
    st.config = types.SimpleNamespace(endpoints={})
    st.scheduler = types.SimpleNamespace(endpoint_snapshot=lambda ep: {})
    st.budget_mgr = types.SimpleNamespace(snapshot=lambda: [])
    st.metrics = _NullMetrics()
    st.queue_db = types.SimpleNamespace(
        wal_size_bytes=lambda: 0, writer_alive=lambda: True,
        write_q_dropped=lambda: 0, writer_restarts=lambda: 0)
    st.paused_endpoints = set()
    st.cache_drift_current = []
    st.structured_empty_window = window
    st.alerts = []
    st.alert_logged = set()
    return health_mod.Health(st), st


class _NullMetrics:
    """Quiet RollingMetrics stand-in — every check_alerts probe returns
    "nothing happening", so any alert in state.alerts came from OUR code."""
    _window_s = 300.0

    def count(self, **kw):
        return 0

    def __getattr__(self, name):
        return lambda *a, **kw: {}


def test_alarm_reaches_state_alerts_via_evaluate_alerts():
    h, st = _health_with(_feed({}, "thinker", n_empty=50, n_ok=50))
    h.evaluate_alerts(1000.0)
    named = [a for a in st.alerts if a["name"] == "structured_empty_rate"]
    assert len(named) == 1, st.alerts
    assert named[0]["severity"] == "ERROR"


def test_no_alarm_on_a_healthy_fleet():
    h, st = _health_with(_feed({}, "thinker", n_empty=0, n_ok=200))
    h.evaluate_alerts(1000.0)
    assert [a for a in st.alerts if a["name"] == "structured_empty_rate"] == []


# ---------------------------------------------------------------------------
# REPLAY of the real incident
# ---------------------------------------------------------------------------
# Measured 2026-08-01 off the live proxy's /data/agents/llmproxy/queue.db
# (proxy_completions, status='ok', payload_json + response_json both still
# retained), counting completions whose payload declared a structured
# constraint and whose choices[0].message.content json-parses to a dict:
#
#   22,148 structured completions, 2026-07-30 23:24Z -> 2026-08-02 00:22Z
#   tier3 ('thinker') BEFORE the 15:06:52Z restart  0/1188  =  0.0%  (23 buckets)
#   tier3             AFTER                      1271/4844  = 26.2%  (65 buckets)
#   gemma / creative / companion, whole period    0 empty across 114 buckets
#
# Replaying that stream minute-by-minute through THIS code fired on `thinker`
# at 2026-07-31 16:32Z — but see the note in the 15%-threshold test below.
_INCIDENT = [
    # (label, endpoint, n_structured, n_empty, must_alarm) — 30-min buckets
    ("tier3, whole pre-restart period", "thinker", 1188, 0, False),
    ("tier3, whole post-restart period", "thinker", 4844, 1271, True),  # 26.2%
    ("tier3, first hours after", "thinker", 69, 25, True),              # 36.2%
    ("tier3, sustained peak", "thinker", 100, 68, True),                # 68%
    ("gemma, same period", "gemma", 391, 0, False),
    ("creative, same period", "creative", 85, 0, False),
]


def test_replay_of_the_measured_incident_traffic():
    """The alarm fires on the real numbers, and on nothing else.

    This is the whole point of the exercise: an alarm nobody has seen fire is
    an alarm nobody should trust. Driven with counts measured off the live
    proxy's own completion log for the 31-hour window, both sides of the
    restart."""
    for label, ep, n_struct, n_empty, must_alarm in _INCIDENT:
        w = _feed({}, ep, n_empty=n_empty, n_ok=n_struct - n_empty)
        fired = bool(obs.structured_empty_alerts(w, 1000.0))
        assert fired is must_alarm, (
            f"{label}: {n_empty}/{n_struct} "
            f"({n_empty / n_struct:.1%}) -> fired={fired}, want {must_alarm}")


def test_the_threshold_keeps_real_margin_under_the_real_incident():
    """The incident's endpoint-wide rate was 26.2% — NOT the ~100% the raw
    `{}` behaviour suggests, because most tier3 structured callers had already
    migrated to real schemas and could not degenerate. A 25% threshold would
    have sat within 1.2 points of the live signal and lost ~8 hours of
    coverage to normal fluctuation. Pinned so nobody "tidies" the threshold
    back up to a round number without re-measuring."""
    assert RATE <= 0.20, "threshold must keep clear air under the measured 26.2%"
    assert RATE * FLOOR >= 3, "…without dropping so low it fires on 2-of-20"


def test_the_quietest_endpoint_never_evaluates_and_we_say_so():
    """Measured: `companion` peaked at 6 structured requests per 30-minute
    bucket and had ZERO evaluated buckets across the whole two days, so it can
    never reach the sample floor and is NOT covered by this alarm. Pinned here
    so the gap is a recorded decision rather than a surprise."""
    w = _feed({}, "companion", n_empty=6, n_ok=0)
    assert obs.structured_empty_rates(w, 1000.0)["companion"]["evaluated"] is False
    assert obs.structured_empty_alerts(w, 1000.0) == []
