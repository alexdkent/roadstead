"""Unit tests for `_ToolCallStreamSanitizer` (buffer-then-emit-atomically model).

Covers all three vLLM qwen3_xml + MTP defects:
  A. phantom name-less slot  -> dropped
  A. trailing extra `}`      -> trimmed (args still valid JSON)
  B. truncated/incomplete args -> call dropped + finish_reason relabeled "length"
  + clean single/parallel calls, no-arg calls, and content-only pass-through.

Self-contained: imports the real service module (run in-container; also runs as a
plain script). Models the streaming contract: each SSE `data:` JSON is fed to
`feed()`; we assemble the emitted tool_calls and the final finish_reason.
"""
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

service = importlib.import_module("roadstead.service")
San = service._ToolCallStreamSanitizer


def tc_chunk(index, *, id=None, type=None, name=None, args=None, finish=None):
    """One SSE chunk carrying a tool_call fragment (and/or a finish_reason)."""
    tc = {"index": index}
    if id is not None:
        tc["id"] = id
    if type is not None:
        tc["type"] = type
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        tc["function"] = fn
    ch = {"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": finish}
    return json.dumps({"choices": [ch]})


def finish_chunk(reason="tool_calls"):
    return json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


def run(chunks):
    """Feed chunks; return (assembled_calls_by_index, final_finish_reason).
    assembled = {index: {"id","name","args"}} from whatever the sanitizer emits."""
    san = San()
    calls = {}
    finish = None
    for c in chunks:
        out = san.feed(c)
        obj = json.loads(out)
        for ch in obj.get("choices", []):
            if ch.get("finish_reason") is not None:
                finish = ch["finish_reason"]
            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                idx = tc.get("index")
                slot = calls.setdefault(idx, {"id": None, "name": None, "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
    return calls, finish


def test_single_clean_call_streamed():
    calls, finish = run([
        tc_chunk(0, id="c1", type="function", name="bash", args=""),
        tc_chunk(0, args='{"command":"'),
        tc_chunk(0, args='pwd"}'),
        finish_chunk("tool_calls"),
    ])
    assert set(calls) == {0}
    assert calls[0]["name"] == "bash"
    assert json.loads(calls[0]["args"]) == {"command": "pwd"}
    assert finish == "tool_calls"


def test_trailing_brace_dup_trimmed():
    # defect A: extra trailing "}" after a complete value
    calls, finish = run([
        tc_chunk(0, id="c1", name="bash", args='{"command":"pwd"}'),
        tc_chunk(0, args="}"),  # the bug's stray brace
        finish_chunk("tool_calls"),
    ])
    assert json.loads(calls[0]["args"]) == {"command": "pwd"}  # parses, no extra }
    assert finish == "tool_calls"


def test_phantom_nameless_slot_dropped():
    # defect A: real@0, phantom(name=null)@1, real@2
    calls, finish = run([
        tc_chunk(0, id="c0", name="bash", args='{"command":"ls"}'),
        tc_chunk(1, id="c1", name=None, args=""),   # phantom
        tc_chunk(2, id="c2", name="bash", args='{"command":"pwd"}'),
        finish_chunk("tool_calls"),
    ])
    assert set(calls) == {0, 2}          # phantom index 1 dropped
    assert json.loads(calls[0]["args"]) == {"command": "ls"}
    assert json.loads(calls[2]["args"]) == {"command": "pwd"}
    assert finish == "tool_calls"


def test_truncated_args_dropped_and_relabeled():
    # defect B: args cut mid-string, never complete; vLLM mislabels finish as tool_calls
    calls, finish = run([
        tc_chunk(0, id="c0", name="bash", args='{"command":"echo \\"hel'),
        finish_chunk("tool_calls"),
    ])
    assert calls == {}                   # broken call NOT emitted
    assert finish == "length"            # relabeled so client retries


def test_truncated_among_completes():
    calls, finish = run([
        tc_chunk(0, id="c0", name="bash", args='{"command":"ls"}'),
        tc_chunk(1, id="c1", name="bash", args='{"command":"unterminated'),
        finish_chunk("tool_calls"),
    ])
    assert set(calls) == {0}             # only the complete one
    assert json.loads(calls[0]["args"]) == {"command": "ls"}
    assert finish == "length"


def test_no_arg_call_empty_buffer():
    # named slot, zero argument chars by finish -> legit no-arg call "{}"
    calls, finish = run([
        tc_chunk(0, id="c0", name="list_things", args=""),
        finish_chunk("tool_calls"),
    ])
    assert json.loads(calls[0]["args"]) == {}
    assert calls[0]["name"] == "list_things"
    assert finish == "tool_calls"


def test_no_arg_call_explicit_braces():
    calls, finish = run([
        tc_chunk(0, id="c0", name="list_things", args="{}"),
        finish_chunk("tool_calls"),
    ])
    assert json.loads(calls[0]["args"]) == {}
    assert finish == "tool_calls"


def test_parallel_calls_all_complete():
    calls, finish = run([
        tc_chunk(0, id="c0", name="bash", args='{"command":"a"}'),
        tc_chunk(1, id="c1", name="grep", args='{"pattern":"x"}'),
        tc_chunk(2, id="c2", name="glob", args='{"glob":"*.py"}'),
        finish_chunk("tool_calls"),
    ])
    assert set(calls) == {0, 1, 2}
    assert calls[1]["name"] == "grep" and json.loads(calls[1]["args"]) == {"pattern": "x"}


def test_content_only_passthrough_identity():
    san = San()
    chunk = json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"},
                                     "finish_reason": None}]})
    out = san.feed(chunk)
    assert out is chunk                  # fast path: original object, untouched


def test_args_with_braces_inside_strings():
    # ensure raw_decode handles } inside string values (no premature trim)
    calls, _ = run([
        tc_chunk(0, id="c0", name="bash", args='{"command":"echo '),
        tc_chunk(0, args='\\"a}b{c\\""}'),
        finish_chunk("tool_calls"),
    ])
    assert json.loads(calls[0]["args"]) == {"command": 'echo "a}b{c"'}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")


def test_trailing_usage_frame_passes_through_untouched():
    """Phase 4: with injected stream_options.include_usage the backend appends
    a usage-only frame (empty choices) AFTER the finish chunk. The sanitizer
    must fast-path it — same object back, no rebuild — including right after a
    tool-call stream where slots were finalized by the finish chunk."""
    import json as _json
    from roadstead.service import _ToolCallStreamSanitizer

    s = _ToolCallStreamSanitizer()
    open_call = _json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "c1", "type": "function",
         "function": {"name": "ls", "arguments": "{}"}}]}}]})
    finish = _json.dumps({"choices": [{"index": 0, "delta": {},
                                       "finish_reason": "tool_calls"}]})
    usage = _json.dumps({"choices": [],
                         "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
    s.feed(open_call)
    s.feed(finish)
    out = s.feed(usage)
    assert out is usage  # pure pass-through (no pending slots, no tool_calls key)
