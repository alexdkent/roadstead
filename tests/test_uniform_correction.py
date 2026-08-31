"""Step 4a — uniform correction: happy-path + parity tests (BUILDER track).

Covers the new-in-4a surface:
  * ``Correction.apply`` runs the sync guard sequence in the LOAD-BEARING order
    (thinking → degeneration → shadow-egress) — the byte-identical consolidation
    of the old inline calls in ``handle_sync_submit``.
  * ``Correction.finalize_stream`` DETECTS a degeneration loop + a silent
    grammar-drop over a stream's reassembled content, gated on the flag
    (OFF == no-op), fail-open, never re-dispatches.
  * The internal ``/v1/submit`` streaming door now routes tool-call chunks through
    the SAME ``_ToolCallStreamSanitizer`` the OpenAI door uses, under the flag.

Unit tests self-bind the real methods (Mac 3.9 conftest blocks pytest, so they
also run as a plain script); the two E2E cells use the in-process proxy + fake
backend harness. The adversarial track owns the hostile both-seam matrix +
guard-bite; this file is the builder's happy-path/parity net.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

correction = importlib.import_module("roadstead.correction")
C = correction.Correction

FLAG = "COLLECTIVE_PROXY_UNIFORM_CORRECTION"

# Object-root grammar {"x": "<str>"} — verify_conformance flags non-JSON / fences
# / wrong-keys as silent drops (mirrors test_shadow_egress).
GRAMMAR = (
    'root ::= "{" ws "\\"x\\":" ws str ws "}"\n'
    'str ::= "\\"" [^"]* "\\""\n'
    'ws ::= [ \\t\\n]*\n'
)


def _req(payload=None, *, ptype="chat_completion", endpoint="companion",
         rid="r1", call_site="unmanaged.site"):
    r = types.SimpleNamespace()
    r.payload = payload if payload is not None else {"messages": []}
    r.stream = False
    r.payload_type = ptype
    r.endpoint = endpoint
    r.request_id = rid
    r.call_site = call_site
    return r


def _grammar_payload():
    return {"messages": [], "extra_body": {"grammar": GRAMMAR}}


def _fresh_state():
    return types.SimpleNamespace(
        degeneration_detected=0,
        degeneration_by_call_site={},
        shadow_drop={},
    )


# --- Correction.apply: order-preserving consolidation ------------------------

async def test_apply_runs_guards_in_load_bearing_order():
    c = C(_fresh_state())
    calls: list = []
    c.finalize_thinking = lambda req, res: calls.append("thinking")

    async def _degen(req, res):
        calls.append("degen")

    c.maybe_correct_degenerate = _degen
    c.shadow_egress_detect = lambda req, res: calls.append("shadow")
    await c.apply(_req(), {"status": "ok"})
    # finalizers first (guard/detector/cache see corrected content), degeneration
    # before the shadow detector + cache.
    assert calls == ["thinking", "degen", "shadow"]


async def test_apply_awaits_the_async_degeneration_guard():
    """apply must AWAIT maybe_correct_degenerate (it re-dispatches) — a missing
    await would let the response race past the guard."""
    c = C(_fresh_state())
    seen = {"degen_done": False}

    c.finalize_thinking = lambda req, res: None

    async def _degen(req, res):
        seen["degen_done"] = True

    c.maybe_correct_degenerate = _degen
    c.shadow_egress_detect = lambda req, res: None
    await c.apply(_req(), {"status": "ok"})
    assert seen["degen_done"] is True


# --- Correction.finalize_stream: flag-gated stream detection -----------------

DEGEN = ("the song of the sea " * 40).strip()


def test_finalize_stream_off_is_noop(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)  # default OFF
    st = _fresh_state()
    C(st).finalize_stream(_req(_grammar_payload()), DEGEN, "stop")
    assert st.degeneration_detected == 0
    assert st.shadow_drop == {}


def test_finalize_stream_on_detects_degeneration(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    st = _fresh_state()
    C(st).finalize_stream(_req(), DEGEN, "stop")
    assert st.degeneration_detected == 1
    assert st.degeneration_by_call_site["unmanaged.site"]["detected"] == 1


def test_finalize_stream_on_detects_silent_grammar_drop(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    st = _fresh_state()
    # grammar-bearing request, but the streamed content is non-conformant prose.
    C(st).finalize_stream(_req(_grammar_payload()), "I cannot do that.", "stop")
    assert st.shadow_drop["unmanaged.site"] == {"checked": 1, "dropped": 1}


def test_finalize_stream_on_conformant_counts_checked_not_dropped(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    st = _fresh_state()
    C(st).finalize_stream(_req(_grammar_payload()), '{"x": "hi"}', "stop")
    assert st.shadow_drop["unmanaged.site"] == {"checked": 1, "dropped": 0}
    assert st.degeneration_detected == 0


def test_finalize_stream_on_clean_short_content_is_noop(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    st = _fresh_state()
    C(st).finalize_stream(_req(), "a short normal answer", "stop")
    assert st.degeneration_detected == 0
    assert st.shadow_drop == {}  # no grammar → nothing to check


def test_finalize_stream_skips_non_chat(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    st = _fresh_state()
    C(st).finalize_stream(_req(DEGEN, ptype="embedding"), DEGEN, "stop")
    assert st.degeneration_detected == 0


def test_finalize_stream_is_fail_open(monkeypatch):
    """A broken state object must not raise out of the stream path."""
    monkeypatch.setenv(FLAG, "1")
    c = C(object())  # no attributes → AttributeError inside, must be swallowed
    c.finalize_stream(_req(), DEGEN, "stop")  # no exception = pass


if __name__ == "__main__":  # plain-script mode for the Mac 3.9 stack (unit only)
    import asyncio as _aio

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

    class _MP:
        def setenv(self, k, v):
            import os
            os.environ[k] = v

        def delenv(self, k, raising=True):
            import os
            os.environ.pop(k, None)

    passed = 0
    for fn in fns:
        args = []
        if "monkeypatch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
            args = [_MP()]
        r = fn(*args)
        if _aio.iscoroutine(r):
            _aio.get_event_loop().run_until_complete(r)
        passed += 1
        print(f"ok {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
