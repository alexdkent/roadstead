"""Unit tests for the WS-4 SHADOW egress / silent-drop detector
(`ProxyService._shadow_egress_detect`). Self-contained: binds the real method to
a lightweight mock `self` (the Mac 3.9 conftest blocks pytest, so this also runs
as a plain script). Key invariants: tallies per-call_site checked/dropped over
grammar-bearing responses, NEVER mutates the response (zero caller risk), skips
when disabled / no grammar / non-ok, and double-counts nothing already handled by
the registry path.
"""
import importlib
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # originfleet/
sys.path.insert(0, str(REPO))

service = importlib.import_module("roadstead.service")
correction = importlib.import_module("roadstead.correction")
C = correction.Correction  # shadow-egress logic moved here in de-monolith Step 3

# Tiny object-root grammar: {"x": "<str>"}. verify_conformance derives root key
# {"x"} from the literal and flags fences / non-JSON / wrong-keys as drops.
GRAMMAR = (
    'root ::= "{" ws "\\"x\\":" ws str ws "}"\n'
    'str ::= "\\"" [^"]* "\\""\n'
    'ws ::= [ \\t\\n]*\n'
)


def _req(payload, *, stream=False, ptype="chat_completion", endpoint="companion",
         rid="r1", call_site="unmanaged.site"):
    r = types.SimpleNamespace()
    r.payload = payload; r.stream = stream; r.payload_type = ptype
    r.endpoint = endpoint; r.request_id = rid; r.call_site = call_site
    return r


def _result(content, status="ok"):
    return {"status": status,
            "response": {"choices": [{"message": {"content": content}}]}}


def _mock_self():
    # Correction operates on self.state; the shadow_drop tally is mutated in
    # place, so aliasing m._shadow_drop to the same dict keeps the test bodies'
    # reads valid.
    state = types.SimpleNamespace(shadow_drop={})
    m = types.SimpleNamespace(state=state)
    m._shadow_drop = state.shadow_drop
    for pub, priv in (("shadow_egress_detect", "_shadow_egress_detect"),
                      ("extract_grammar", "_extract_grammar")):
        bound = getattr(C, pub).__get__(m, C)
        setattr(m, pub, bound)
        setattr(m, priv, bound)
    # Step 4a: shadow_egress_detect delegates its conformance-check core to
    # _shadow_egress_check (shared with the streaming finalize_stream path). Bind
    # it on the mock self so the moved-method-on-mock-self pattern still resolves.
    m._shadow_egress_check = C._shadow_egress_check.__get__(m, C)
    return m


def _payload(content_grammar=True):
    p = {"messages": [], "max_tokens": 100}
    if content_grammar:
        p["extra_body"] = {"grammar": GRAMMAR}
    return p


def test_conformant_response_counts_checked_not_dropped():
    m = _mock_self()
    res = _result('{"x": "hello"}')
    m._shadow_egress_detect(_req(_payload()), res)
    t = m._shadow_drop["unmanaged.site"]
    assert t == {"checked": 1, "dropped": 0}
    # zero caller risk: response untouched
    assert res["response"]["choices"][0]["message"]["content"] == '{"x": "hello"}'


def test_markdown_fenced_is_a_silent_drop():
    m = _mock_self()
    res = _result('```json\n{"x": "hello"}\n```')
    m._shadow_egress_detect(_req(_payload()), res)
    assert m._shadow_drop["unmanaged.site"] == {"checked": 1, "dropped": 1}


def test_wrong_keys_is_a_silent_drop():
    m = _mock_self()
    res = _result('{"y": "hello"}')
    m._shadow_egress_detect(_req(_payload()), res)
    assert m._shadow_drop["unmanaged.site"]["dropped"] == 1


def test_non_json_is_a_silent_drop():
    m = _mock_self()
    res = _result('I cannot do that.')
    m._shadow_egress_detect(_req(_payload()), res)
    assert m._shadow_drop["unmanaged.site"]["dropped"] == 1


def test_never_mutates_response_even_on_drop():
    m = _mock_self()
    res = _result('plain text not json')
    resp_obj = res["response"]
    m._shadow_egress_detect(_req(_payload()), res)
    assert res["response"] is resp_obj  # identity preserved
    assert res["response"]["choices"][0]["message"]["content"] == 'plain text not json'


def test_no_grammar_no_tally():
    m = _mock_self()
    m._shadow_egress_detect(_req(_payload(content_grammar=False)), _result('{"x":"y"}'))
    assert m._shadow_drop == {}


def test_non_ok_status_skipped():
    m = _mock_self()
    m._shadow_egress_detect(_req(_payload()), _result('{"x":"y"}', status="error"))
    assert m._shadow_drop == {}


def test_stream_skipped():
    m = _mock_self()
    m._shadow_egress_detect(_req(_payload(), stream=True), _result('{"x":"y"}'))
    assert m._shadow_drop == {}


def test_kill_switch_disables(monkeypatch=None):
    import os
    m = _mock_self()
    old = os.environ.get("COLLECTIVE_PROXY_SHADOW_EGRESS")
    os.environ["COLLECTIVE_PROXY_SHADOW_EGRESS"] = "off"
    try:
        m._shadow_egress_detect(_req(_payload()), _result('not json'))
        assert m._shadow_drop == {}
    finally:
        if old is None:
            os.environ.pop("COLLECTIVE_PROXY_SHADOW_EGRESS", None)
        else:
            os.environ["COLLECTIVE_PROXY_SHADOW_EGRESS"] = old


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"ok {fn.__name__}")
    print(f"\n{passed}/{len(fns)} passed")
