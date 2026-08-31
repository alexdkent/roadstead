"""The timeout floor, from the SERVER side (`docs/api.md` §1.4).

`docs/handoff.md` Phase 2 lists "the timeout-floor mirror" among the contracts
"currently only asserted from the host's side, which is the tautology trap".
The host keeps its own copy of the floor table — `framework/timeout_advice`'s
`floor_for()` — because it needs a floor *before* it can reach the proxy, and
it cannot read the catalog. So there are two copies of a number that must agree,
and until now only the client's copy was tested, against itself.

What Roadstead owns, and what these pin:

  1. the floor is **askable over the wire** — `source == "floor"` means
     `recommended_timeout_s` IS the floor for that class, so a client need not
     mirror at all;
  2. no advice ever comes back below the class floor, at any priority or size —
     the invariant the client's honour-a-sub-floor-deadline decision rests on;
  3. every configured endpoint class HAS a floor, so the fallback is never
     reached by accident;
  4. the fallback floor and the ceilings match what §1.4 publishes, which is
     what a client that does mirror transcribes from.

Deliberately not asserted here: the per-class floor VALUES. They are deployment
data seeded from `models.yaml`, they differ per fleet, and
`test_timeout_model.py::test_timeout_floor_yaml_sync` already pins them against
the catalog. Restating them would be a third copy.
"""
from __future__ import annotations

import json
import re

import pytest

from roadstead.config import ProxyConfig
from roadstead.service import ProxyService
from roadstead.timeout_model import _DEFAULT_FLOOR_S, FLOOR_S
from tests.wire_contract import (
    API_DOC,
    TIMEOUT_CEILING_BACKGROUND_S,
    TIMEOUT_CEILING_INTERACTIVE_S,
    TIMEOUT_FALLBACK_FLOOR_S,
)


class _FakeRequest:
    def __init__(self, **params):
        self.query_params = {k: str(v) for k, v in params.items() if v is not None}

    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers: dict = {}


@pytest.fixture
def svc(tmp_path) -> ProxyService:
    return ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db")))


async def _advice(svc: ProxyService, **params) -> dict:
    resp = await svc.handle_timeout_advice(_FakeRequest(**params))
    assert resp.status_code == 200, resp.body
    return json.loads(resp.body.decode())


# --------------------------------------------------------------------------
# 1 + 2. The floor is askable, and binding
# --------------------------------------------------------------------------

async def test_a_cold_class_reports_source_floor_and_returns_that_floor(svc):
    """The supported way to learn a floor without mirroring the table: query a
    cold cell and read the number back. With no samples anywhere, every class
    must answer this way — if one did not, a client asking for its floor would
    silently receive a sample-derived value instead."""
    assert svc._config.endpoints, "no endpoints configured"
    checked = 0
    for ep_name in svc._config.endpoints:
        body = await _advice(svc, model=ep_name, priority="P0_REALTIME",
                             est_in=16, est_out=16)
        assert body["source"] == "floor", (
            f"{ep_name}: cold advice came from {body['source']!r}, not the floor")
        assert body["sample_count"] == 0
        floor_s = svc._state.timeout_model.floor_ms(ep_name) / 1000.0
        assert body["recommended_timeout_s"] == pytest.approx(floor_s, abs=1.0), (
            f"{ep_name}: source=floor but recommended_timeout_s "
            f"{body['recommended_timeout_s']} != floor {floor_s} — the wire "
            f"signal a client reads the floor from would be lying")
        checked += 1
    assert checked >= 3, f"only {checked} endpoint classes checked — sweep too thin"


@pytest.mark.parametrize("priority", ["P0_REALTIME", "P1_TURN_SUPPORT",
                                      "P3_INGESTION", "P4_HYGIENE"])
@pytest.mark.parametrize("est_in,est_out", [(0, 0), (16, 16), (200_000, 4096)])
async def test_advice_is_never_below_the_class_floor(svc, priority, est_in, est_out):
    """Across every band and size, in both directions. A single sub-floor answer
    is enough to make a client's honour-the-caller's-deadline logic wrong, and
    the sub-floor cliff it guards against was a live regression."""
    for ep_name in svc._config.endpoints:
        body = await _advice(svc, model=ep_name, priority=priority,
                             est_in=est_in, est_out=est_out)
        floor_s = svc._state.timeout_model.floor_ms(ep_name) / 1000.0
        assert body["recommended_timeout_s"] >= floor_s - 1.0, (
            f"{ep_name} @ {priority} ({est_in}/{est_out}): advised "
            f"{body['recommended_timeout_s']}s, below its {floor_s}s floor")


# --------------------------------------------------------------------------
# 3. The fallback is never reached by accident
# --------------------------------------------------------------------------

def test_every_configured_endpoint_class_has_its_own_floor(svc):
    """`_DEFAULT_FLOOR_S` is a backstop for a class nobody declared. If a real
    endpoint fell through to it, that endpoint would silently be running on a
    60s deadline nobody chose — which for a 360s-floor reasoning model is a
    guaranteed timeout, not a conservative default."""
    resolved = {
        name: svc._state.timeout_model.floor_ms(name) / 1000.0
        for name in svc._config.endpoints
    }
    assert resolved, "no endpoints configured"
    missing = [
        name for name in resolved
        if name not in FLOOR_S and resolved[name] == _DEFAULT_FLOOR_S
    ]
    assert not missing, (
        f"endpoint classes falling through to the {_DEFAULT_FLOOR_S}s fallback: "
        f"{missing} — declare a floor in models.yaml timeout_floor_s")


def test_an_unknown_class_gets_the_published_fallback(svc):
    """The number a mirroring client transcribes from §1.4."""
    floor_s = svc._state.timeout_model.floor_ms(
        "no-such-endpoint-class-xyz") / 1000.0
    assert floor_s == TIMEOUT_FALLBACK_FLOOR_S == _DEFAULT_FLOOR_S


# --------------------------------------------------------------------------
# 4. The published numbers still match the code
# --------------------------------------------------------------------------

def test_the_published_floor_and_ceilings_match_the_code():
    """§1.4 publishes three numbers a client codes against. Read them back — a
    stale doc is how a mirror drifts without anyone touching the mirror."""
    from roadstead.timeout_model import (
        _BACKGROUND_CEILING_S,
        _INTERACTIVE_CEILING_S,
    )
    assert _DEFAULT_FLOOR_S == TIMEOUT_FALLBACK_FLOOR_S
    assert _INTERACTIVE_CEILING_S == TIMEOUT_CEILING_INTERACTIVE_S
    assert _BACKGROUND_CEILING_S == TIMEOUT_CEILING_BACKGROUND_S

    doc = API_DOC.read_text(encoding="utf-8")
    assert "### 1.4 `GET /v1/timeout-advice`" in doc, "§1.4 moved or was renamed"
    for value in (TIMEOUT_FALLBACK_FLOOR_S, TIMEOUT_CEILING_INTERACTIVE_S,
                  TIMEOUT_CEILING_BACKGROUND_S):
        assert f"**{value}s**" in doc, (
            f"docs/api.md §1.4 no longer publishes {value}s")


def test_the_per_class_floors_are_deliberately_not_published():
    """§1.4 says the per-class floors are deployment data and tells clients to
    ask for them. Pin that commitment, so nobody 'helpfully' pastes a fleet's
    real floor table into a document headed for publication — the working tree
    half of `docs/corpus_and_scrub_plan.md`.

    Checks for a table ROW binding a class to a number, not for the class name:
    §1.4 legitimately names `llama-thinker` and `thinker` while explaining
    normalization, and a mention is not a disclosure.
    """
    doc = API_DOC.read_text(encoding="utf-8")
    start = doc.index("### 1.4 `GET /v1/timeout-advice`")
    section = doc[start:doc.index("\n## 2.", start)]

    assert "not** listed here on purpose" in section, (
        "§1.4 dropped its commitment to keep per-class floors out of the "
        "published contract — restore it or re-plan the scrub")

    leaked = [
        name for name in FLOOR_S
        if re.search(rf"^\|\s*`?{re.escape(name)}`?\s*\|\s*\**[0-9]",
                     section, re.MULTILINE)
    ]
    assert not leaked, (
        f"§1.4 now tabulates floor values for {leaked} — that is deployment "
        f"data, and this document is headed for publication")
