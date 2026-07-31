"""An empty completion must say WHICH kind of empty it is.

🚨 THE CONFUSION THIS ENDS, and it cost the fleet a real misdiagnosis.

A reasoning model that spent its whole budget thinking and a backend that
produced nothing are the SAME BYTES to a caller: `content == ""`. Told apart
afterwards, it is guesswork. Told apart HERE — where the `reasoning` field and
`finish_reason` are still in hand — it is arithmetic.

On 2026-07-31 the proxy's thinking DEFAULT was flipped true -> false
(`999308832`) under the heading "it was silently returning EMPTY content". Its
own recorded evidence read:

    max_tokens=120  thinking=true  -> out=120  finish=length  content_len=0  EMPTY

`finish=length` was right there. The measurement was complete and correct; only
the ERROR MESSAGE was ambiguous, so the conclusion drawn was "thinking is
broken" rather than "120 tokens cannot hold reasoning AND an answer".

Re-measured the same day: raw `enable_thinking=true` at max_tokens=120 -> 502
empty; the DESIGNED opt-in (`thinking: true`, which makes the proxy add a
reasoning budget on top) -> 628 chars, finish=stop, 965 tokens. The capability
was never broken and the default was never wrong — the message was.
"""
from __future__ import annotations

import pytest

from originfleet.llmproxy.backend import empty_completion_error


def _gate(content, reasoning, finish, output_tokens, role="llama-thinker"):
    """Drive the REAL gate. Deliberately not a re-implementation: a test that
    copies the logic it checks is a copy that drifts, and precision of this
    message is the entire point of the change."""
    msg = {"content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    return empty_completion_error(role, msg, finish, output_tokens)


def test_reasoning_exhaustion_is_named_as_a_budget_problem() -> None:
    """The exact shape of the 999308832 misdiagnosis."""
    err = _gate("", "Okay, so I need to figure out 17 times 23..." * 20,
                "length", 120)
    text = str(err)
    assert "REASONING" in text and "token-budget problem" in text
    assert "NOT a broken model" in text, (
        "the message must rule out the wrong conclusion explicitly — a bare "
        "'empty completion' is what produced the misdiagnosis")
    assert "raise max_tokens" in text, "it must name the remedy, not just the cause"


def test_a_genuinely_empty_backend_is_still_reported_plainly() -> None:
    """The budget message must not swallow the real empty-completion case."""
    err = _gate("", "", "stop", 0)
    text = str(err)
    assert "returned empty completion" in text
    assert "REASONING" not in text, (
        "a backend that produced nothing at all must NOT be reported as a "
        "reasoning-budget problem — that would send the next person to raise "
        "max_tokens on a dead backend")


def test_the_plain_message_now_carries_finish_reason() -> None:
    """`finish_reason` is the single most diagnostic field and it was omitted.

    Had it been in the message, the 2026-07-31 conclusion would have been
    'length, so raise the budget' rather than 'thinking is broken'.
    """
    assert "finish_reason='length'" in str(_gate("", "", "length", 120))


def test_reasoning_without_length_is_not_a_budget_claim() -> None:
    """Reasoning present but finish=stop is a DIFFERENT failure — do not
    misattribute it to the budget just because a reasoning field exists."""
    err = _gate("", "some reasoning", "stop", 50)
    assert "REASONING" not in str(err)


def test_content_present_is_never_an_error() -> None:
    assert _gate("an answer", "", "stop", 10) is None


@pytest.mark.parametrize("finish", ["length", "stop", None])
def test_tool_calls_and_content_take_precedence_over_finish_reason(finish) -> None:
    """Whatever the finish_reason, real content is a success."""
    assert _gate("real content", "reasoning too", finish, 500) is None
