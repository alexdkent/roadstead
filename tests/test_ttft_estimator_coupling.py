"""The TTFT prefill floor is COUPLED to the input-token estimator — pin it.

``_STREAM_PREFILL_FLOOR_TOK_S`` is not an independent tuning knob. It is
``(slow-end measured prefill rate / 2 for contention) / R``, where ``R`` is how
far ``cost_model.estimate_input_tokens`` falls short of real tokenization on the
payloads that actually drive long prefills (tool-heavy code transcripts).

That makes it silently invalidatable from a distance: someone improves the
estimator, ``R`` drops toward 1.0, and every TTFT allowance inflates by the same
factor — with no test failing and no symptom until a dead backend holds a scarce
tier3 slot for far longer than anyone intended. Exactly the shape of the defect
this whole change set exists to fix (a constant whose premise had quietly become
false), so it gets a guard rather than a comment.

The fixture below is a REDUCED but structurally faithful copy of a live `pool`
payload: the tokens live in ``tool_calls`` arguments and ``tool`` results, not in
``content`` — which is precisely why the pre-D4 estimator read them at ~half.

Measured 2026-08-03 against five live payloads (see constants.py for the table):
    actual 226,014 | pre-D4 est 119,460 (1.89x) | post-D4 est 147,735 (1.53x)
"""

from __future__ import annotations

import json

import pytest

from roadstead.constants import (
    _EST_IN_RESIDUAL_UNDERCOUNT,
    _STREAM_PREFILL_FLOOR_TOK_S,
)
from roadstead.cost_model import estimate_input_tokens

# The two inputs the floor is derived from, per constants.py.
_SLOW_END_PREFILL_TOK_S = 729.0   # tier3 at 578K, models.yaml tier3 stanza
_CONTENTION_HEADROOM = 2.0        # halve the slow end


def _tool_heavy_payload(turns: int = 60) -> dict:
    """A transcript whose tokens are overwhelmingly OUTSIDE ``content``."""
    messages: list[dict] = [
        {"role": "system", "content": "You are a coding agent."},
    ]
    for i in range(turns):
        messages.append({
            "role": "assistant",
            "content": None,                     # the shape that read as ZERO
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({
                        "path": f"src/module_{i}.py",
                        "contents": "def f():\n    return 1\n" * 40,
                    }),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": "x = 1\ny = 2\nz = compute(x, y)\n" * 40,
        })
    return {
        "model": "tier3",
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {"name": f"tool_{n}", "parameters": {"type": "object"}},
        } for n in range(16)],
        "stream": True,
    }


def test_estimator_still_undercounts_by_about_the_pinned_ratio():
    """If this fails, the estimator changed — RE-DERIVE the TTFT floor.

    Asserted as a real-vs-estimate ratio, using the documented ~2.6 chars/token
    that these payloads genuinely tokenize at, rather than re-asserting the
    estimator's own arithmetic back at itself (which would pass no matter what).
    """
    payload = _tool_heavy_payload()
    est = estimate_input_tokens(payload)

    # Ground truth stand-in: the payload's real prompt text at the MEASURED
    # density for code/JSON transcripts. Deliberately independent of the
    # estimator's 4-chars/token divisor — that divisor is what we're measuring.
    prompt_chars = len(json.dumps(payload["messages"])) + len(json.dumps(payload["tools"]))
    approx_real_tokens = prompt_chars / 2.6

    ratio = approx_real_tokens / est
    assert est > 0
    assert ratio == pytest.approx(_EST_IN_RESIDUAL_UNDERCOUNT, rel=0.20), (
        f"estimate_input_tokens now undercounts by {ratio:.2f}x, but "
        f"_EST_IN_RESIDUAL_UNDERCOUNT pins {_EST_IN_RESIDUAL_UNDERCOUNT}. The TTFT "
        f"prefill floor ({_STREAM_PREFILL_FLOOR_TOK_S} tok/s) is derived from that "
        f"ratio — changing the estimator without re-deriving the floor silently "
        f"scales every TTFT allowance. Re-measure both together (constants.py)."
    )


def test_prefill_floor_matches_its_stated_derivation():
    """The constant must equal the arithmetic its own comment claims."""
    expected = (_SLOW_END_PREFILL_TOK_S / _CONTENTION_HEADROOM) / _EST_IN_RESIDUAL_UNDERCOUNT
    assert _STREAM_PREFILL_FLOOR_TOK_S == pytest.approx(expected, rel=0.05), (
        f"_STREAM_PREFILL_FLOOR_TOK_S={_STREAM_PREFILL_FLOOR_TOK_S} no longer matches "
        f"({_SLOW_END_PREFILL_TOK_S}/{_CONTENTION_HEADROOM})/{_EST_IN_RESIDUAL_UNDERCOUNT} "
        f"= {expected:.1f}. Update the derivation comment in constants.py too."
    )


def test_ttft_allowance_covers_the_measured_worst_case():
    """A 578K-token prompt measures ~794s end-to-end; TTFT must exceed prefill.

    The regression this pins is the ORIGINAL defect: a live 123,466-token prompt
    aborted at 145.078s under the old fictional 1000 tok/s assumption.
    """
    from roadstead.constants import _STREAM_TTFT_DEADLINE_S

    # est_in as the estimator would report it for a real 578K-token prompt.
    est_in_for_578k = 578_000 / _EST_IN_RESIDUAL_UNDERCOUNT
    allowance = _STREAM_TTFT_DEADLINE_S + est_in_for_578k / _STREAM_PREFILL_FLOOR_TOK_S
    assert allowance > 794.0, (
        f"TTFT allowance {allowance:.0f}s does not cover the measured 794s "
        f"worst-case end-to-end for a 578K-token request"
    )

    # And the original defect case must now survive where it previously died.
    est_in_defect = 115_077
    allowance_defect = _STREAM_TTFT_DEADLINE_S + est_in_defect / _STREAM_PREFILL_FLOOR_TOK_S
    assert allowance_defect > 145.078, (
        "the live 2026-08-03 abort at 145.078s would still fire"
    )
