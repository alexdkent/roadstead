"""Every field the ENRICHED DOOR reads is a field the SDK can send. 🚨

The gap this closes existed because nothing looked across the boundary. The
server has read ``payload_type`` since before this repo existed
(``enriched.py``, defaulting to ``chat_completion``) and read ``caller_id`` and
``request_id`` beside it; ``roadstead.client`` sent none of the three for the
whole of its life. Nothing was broken in a way anything could report:

* ``payload_type`` — the SDK could make **chat calls only**. Embedding and
  rerank were unreachable through it, and rerank was unreachable through
  anything at all once ``/v1/submit`` was removed.
* ``caller_id`` — silently defaulted to the ``agent_id``, so every SDK caller
  looked like one call site. ``/v1/fleet/top-callers`` and
  ``/v1/fleet/cache-attribution`` went flat. A missing MEASUREMENT reports
  itself to nobody.
* ``request_id`` — a caller's own correlation id never reached the durable
  record, so their log line and our completion row could not be joined.

Neither half is a list. The server half is read by AST — a list is a third thing
to keep in step with two moving files, and when it falls behind the failure is
silent, which is the argument ``tests/test_admin_ui.py`` and
``tests/e2e/test_client_field_coverage.py`` make for extracting rather than
enumerating. 🚨 The SDK half is not read at all: the builder is **called**, and
the guard asks what came back. An AST version of this side passed against a
simulated regression that wrapped every assignment in ``if False:``.

🚨 **This is the SEND side.** ``tests/e2e/test_client_field_coverage.py`` is the
receive side — every field the SDK *reads* off a response, walked against a
response the server really sent. Neither implies the other.
"""
from __future__ import annotations

import ast
import pathlib

from roadstead.client import _client as client_module

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENRICHED = _ROOT / "roadstead" / "enriched.py"
_INTENT = _ROOT / "roadstead" / "intent.py"

#: Fields the enriched door reads that the SDK deliberately does not send, each
#: with the reason it is right for the SDK to be silent about it.
#:
#: 🚨 An entry here is a CLAIM that the absence is correct, not a note that
#: nobody has looked. It is empty today and that is the healthy state: every
#: field the server consults is one a caller of this SDK can express.
_SERVER_ONLY: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _body_reads(path: pathlib.Path, *, only: str = "") -> set[str]:
    """``body.get("x")`` over a module, or over one function within it.

    Matched on the RECEIVER being the name ``body``, which is what both handlers
    call the parsed request body. A dotted or renamed receiver would slip past —
    and would also be a different shape of code than either file has ever had,
    so ``test_the_extractors_found_something`` pins the counts rather than
    trusting this to be exhaustive by construction.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scopes = [tree]
    if only:
        scopes = [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.name == only]
        assert scopes, f"{path.name} has no function named {only!r}"
    found: set[str] = set()
    for scope in scopes:
        for node in ast.walk(scope):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("get", "pop")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "body"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.add(node.args[0].value)
    return found


#: A plausible value for every routing field ``_envelope`` accepts, so the
#: builder can be CALLED with all of them at once and asked what it emits.
#: Values are shaped only well enough to survive the builder's coercions —
#: nothing here is a contract, and the assertion below fails if a new parameter
#: appears without one rather than silently skipping it.
_EVERY_FIELD: dict[str, object] = {
    "messages": [{"role": "user", "content": "x"}],
    "payload_type": "embedding",
    "intent": "embed",
    "model": "tier3",
    "requires": ["vision"],
    "exclude": ["tier2"],
    "kind": "embed",
    "min_context": 1024,
    "prefer": "latency",
    "priority": "P1_TURN_SUPPORT",
    "interactive": True,
    "deadline_s": 12.5,
    "allow_degrade": False,
    "allow_spill": False,
    "call_site": "test.site",
    "session_id": "sess-1",
    "turn_id": "turn-1",
    "caller_id": "caller-1",
    "request_id": "req-1",
    "agent_id": "chat-agent",
    "est_in": 2000,
    "est_out": 200,
    "stream": True,
    "payload": {"max_tokens": 8},
    "extra_payload": {"tools": []},
}


def _sdk_sends() -> set[str]:
    """What the SDK actually PUTS ON THE WIRE — by calling the builder.

    🚨 **Behavioural, not structural, and the difference was measured.** The
    first version of this read ``body["x"] = …`` assignments out of
    ``_client.py`` by AST, and it PASSED against a simulated regression that
    wrapped every one of those assignments in ``if False:``. AST does not care
    whether a line can run. The question worth asking is not "is there code that
    would send this field" but "does sending it work", and the only honest way
    to ask that is to call the function.

    ``_envelope`` is pure and takes keyword arguments only, so this costs
    nothing and needs no server.
    """
    import inspect

    params = set(inspect.signature(client_module._envelope).parameters)
    unmapped = sorted(params - set(_EVERY_FIELD))
    assert not unmapped, (
        f"`_envelope` grew parameter(s) {unmapped} with no entry in "
        f"_EVERY_FIELD — add one, or this guard silently stops asking about "
        f"the newest field on the envelope")
    return set(client_module._envelope(
        **{k: v for k, v in _EVERY_FIELD.items() if k in params}))


SERVER_READS = _body_reads(_ENRICHED) | _body_reads(_INTENT, only="parse_intent")
SDK_SENDS = _sdk_sends()


# --------------------------------------------------------------------------- #
# The guards
# --------------------------------------------------------------------------- #

def test_the_extractors_found_something():
    """🚨 A reader that quietly finds nothing makes every assertion below pass.

    Named fields rather than only a count, because the count survives an
    extractor that has stopped resolving one shape of read and picked up
    another.
    """
    for field in ("payload", "payload_type", "caller_id", "request_id",
                  "substitution", "deadline_s"):
        assert field in SERVER_READS, (
            f"the extractor no longer sees `{field}` being read off the "
            f"enriched request body — it reads {sorted(SERVER_READS)}")
    for field in ("intent", "model", "requires", "exclude", "kind", "prefer",
                  "min_context"):
        assert field in SERVER_READS, (
            f"`parse_intent` reads `{field}` and the extractor missed it")
    for field in ("payload", "payload_type", "est_in"):
        assert field in SDK_SENDS, (
            f"`_envelope` does not put `{field}` on the wire even when it is "
            f"given one — it emitted {sorted(SDK_SENDS)}")
    assert len(SERVER_READS) >= 18 and len(SDK_SENDS) >= 18


def test_every_field_the_server_reads_is_one_the_sdk_can_send():
    """The guard that would have caught all three losses at once."""
    missing = sorted(SERVER_READS - SDK_SENDS - set(_SERVER_ONLY))
    assert not missing, (
        "the enriched door reads request-body fields that roadstead.client "
        "never sends, so a caller of the SDK cannot reach them and nothing "
        "reports it: " + ", ".join(missing) + "\n"
        "  Send them from `_envelope`, or add each to `_SERVER_ONLY` with the "
        "reason its absence is correct.")


def test_the_server_only_list_is_not_stale():
    """An entry that names a field the server has stopped reading is an excuse
    outliving its subject — and the next real omission gets waved through under
    it."""
    stale = sorted(set(_SERVER_ONLY) - SERVER_READS)
    assert not stale, (
        f"_SERVER_ONLY claims the enriched door reads {stale}, and it does not")


def test_the_sdk_sends_nothing_the_server_ignores():
    """The other direction. A field on the envelope that nothing consults is a
    caller writing into a void — the same silence as the three above, aimed the
    other way. ``est_in``/``est_out`` are `/rs/v1/plan`'s and read there."""
    unread = sorted(SDK_SENDS - SERVER_READS)
    assert not unread, (
        "roadstead.client puts fields on the enriched envelope that neither "
        "`enriched.py` nor `parse_intent` reads: " + ", ".join(unread))
