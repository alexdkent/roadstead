"""``tests/wire_contract.py`` is a hand-transcription. Verify it against source.

A restated boundary object is worse than none at all once it goes stale: the
literals keep passing while the published contract has moved, so both ends look
pinned and neither is. These read ``docs/api.md`` back.
"""
from __future__ import annotations

import pytest

from tests.wire_contract import (
    API_DOC,
    CONTEXT_OVERFLOW_MARKER,
    DEFERRABLE_MARKERS,
    carries_deferral_marker,
)


@pytest.fixture(scope="module")
def doc() -> str:
    assert API_DOC.exists(), f"the error contract is missing at {API_DOC}"
    return API_DOC.read_text(encoding="utf-8")


def test_the_contract_section_still_exists(doc):
    assert "### 2.2 The marker substrings are the real contract" in doc, (
        "docs/api.md §2.2 moved or was renamed — re-derive the markers in "
        "tests/wire_contract.py from wherever the error contract now lives")


def test_every_deferral_marker_is_published(doc):
    assert DEFERRABLE_MARKERS, "an empty marker set would pass every assertion"
    for marker in DEFERRABLE_MARKERS:
        assert f"`{marker}`" in doc, (
            f"docs/api.md no longer documents the deferral marker {marker!r}")


def test_the_context_overflow_marker_is_published_verbatim(doc):
    assert CONTEXT_OVERFLOW_MARKER in doc, (
        "docs/api.md no longer publishes the verbatim context-overflow marker "
        f"{CONTEXT_OVERFLOW_MARKER!r} — reconcile before editing it")


def test_the_matcher_is_not_vacuous():
    """A matcher that says yes to everything, or no to everything, would make
    every caller of it pass. Pin both directions."""
    assert carries_deferral_marker("backend tier3 unavailable (circuit open)")
    assert carries_deferral_marker("proxy draining for shutdown — backpressure")
    assert carries_deferral_marker("", "BACKPRESSURE: queue saturated")  # case-insensitive
    assert not carries_deferral_marker("unknown endpoint 'typo' — no such model/role")
    assert not carries_deferral_marker("grammar_invalid", "unparseable: no rule definitions")
    assert not carries_deferral_marker("")
    assert not carries_deferral_marker()
