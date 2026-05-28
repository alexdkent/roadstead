"""Unit tests for thinker_bench pure functions (WS4b Phase 0).

The bench runs on anvil during the container-down window; these lock the
request-shaping (engine adapter), grammar extraction, and percentile math so
the llama.cpp-vs-vLLM comparison is apples-to-apples.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_spec = importlib.util.spec_from_file_location(
    "thinker_bench",
    pathlib.Path(__file__).resolve().parents[2]
    / "originfleet" / "scripts" / "thinker_bench.py",
)
tb = importlib.util.module_from_spec(_spec)
sys.modules["thinker_bench"] = tb
_spec.loader.exec_module(tb)


def test_grammar_extracted_from_top_level():
    assert tb._grammar_of({"grammar": "root ::= x"}) == "root ::= x"


def test_grammar_extracted_from_extra_body():
    assert tb._grammar_of({"extra_body": {"grammar": "root ::= y"}}) == "root ::= y"


def test_grammar_none_when_absent():
    assert tb._grammar_of({"messages": []}) is None


def test_speed_request_strips_grammar():
    payload = {"messages": [{"role": "user", "content": "hi"}],
               "grammar": "root ::= x"}
    req = tb._build_request(payload, engine="llama.cpp", with_grammar=False,
                            stream=True, max_tokens=64)
    assert "grammar" not in req
    assert req["stream"] is True
    assert req["max_tokens"] == 64


def test_llamacpp_grammar_field():
    payload = {"messages": [{"role": "user", "content": "hi"}],
               "extra_body": {"grammar": "root ::= x"}}
    req = tb._build_request(payload, engine="llama.cpp", with_grammar=True,
                            stream=True, max_tokens=None)
    assert req["grammar"] == "root ::= x"
    assert "extra_body" not in req


def test_vllm_grammar_field():
    payload = {"messages": [{"role": "user", "content": "hi"}],
               "grammar": "root ::= x"}
    req = tb._build_request(payload, engine="vllm", with_grammar=True,
                            stream=True, max_tokens=None)
    assert req["extra_body"]["structured_outputs"]["grammar"] == "root ::= x"
    assert "grammar" not in req


def test_system_inlined_to_messages():
    payload = {"system": "you are X",
               "messages": [{"role": "user", "content": "hi"}]}
    req = tb._build_request(payload, engine="vllm", with_grammar=False,
                            stream=True, max_tokens=None)
    assert req["messages"][0] == {"role": "system", "content": "you are X"}
    assert req["messages"][1]["role"] == "user"
    assert "system" not in req


def test_percentile():
    assert tb._pct([1, 2, 3, 4, 5], 50) == 3
    assert tb._pct([1, 2, 3, 4, 5], 95) == 5
    assert tb._pct([], 50) == 0.0


def test_corpus_entry_must_be_unwrapped():
    # Export wraps the real request under ["payload"]; both cmd_speed and
    # cmd_grammar must pass entry["payload"], not the entry itself.
    entry = {"request_id": "r1", "call_site": "x",
             "payload": {"messages": [{"role": "user", "content": "hi"}]}}
    good = tb._build_request(entry["payload"], engine="llama.cpp",
                             with_grammar=False, stream=True, max_tokens=8)
    assert good["messages"][-1]["content"] == "hi"
    # Passing the raw entry loses the messages — the bug we fixed in cmd_speed.
    bad = tb._build_request(entry, engine="llama.cpp", with_grammar=False,
                            stream=True, max_tokens=8)
    assert bad["messages"] == []
