"""The /v1/embeddings door must translate BOTH ways, not forward verbatim.

The bge-m3 shim speaks its own dialect on each side:

    request   OpenAI {"input": str | [str]}   ->  shim {"texts": [str]}
    response  shim {"dense": [[float]], ...}  ->  OpenAI {object, data, usage}

Until 2026-08-01 the handler forwarded the OpenAI body verbatim, so the shim
rejected every call for a missing `texts` and the door 502'd for its entire
existence. Nothing caught it because no fleet caller uses this path — agents
embed via /v1/submit, which already speaks `texts` — so it only ever served
external OpenAI clients and no in-repo test exercised it.

These tests pin the translation itself rather than the handler's plumbing: the
translation is the part that was wrong, and it is pure. The RED proof for the
original bug is `test_openai_input_is_not_the_shim_field`: the shim's contract
is `texts`, and an OpenAI body has no such key.

⚠️ There are TWO backend dialects and the fix is only correct because it serves
both. The nexus bge-m3 shim reads `texts` and answers `{"dense": [[float]]}`;
an OpenAI-shaped embeddings server (which is what the e2e fake backend models)
reads `input` and answers with `data`/`object` already correct. The first draft
of this fix handled only the shim and was caught by the e2e pair —
`test_behavior_preservation[embeddings_happy]` (re-wrapped an already-OpenAI
body into a 502) and `test_e2e_happy::test_embeddings_roundtrip` (returned ONE
vector for a TWO-input request, because dropping `input` left the OpenAI-shaped
server nothing to size its reply from). Those two remain the guard for the
dialect split; do not "simplify" the handler to one dialect.
"""
from __future__ import annotations

import base64
import struct

import pytest

from originfleet.llmproxy.http_handlers import (
    _embedding_texts,
    _encode_embedding,
    _estimated_embed_tokens,
)


def test_openai_input_is_not_the_shim_field() -> None:
    """The original bug, stated as an invariant: an OpenAI body cannot be sent
    to the shim unchanged, because the field the shim requires is absent."""
    openai_body = {"model": "bge-m3", "input": "hello world"}
    assert "texts" not in openai_body          # <- the 502, for its whole life
    texts, err = _embedding_texts(openai_body["input"])
    assert not err
    assert texts == ["hello world"]            # <- what the shim actually needs


@pytest.mark.parametrize("raw,expected", [
    ("hello", ["hello"]),
    (["a", "b"], ["a", "b"]),
    (["only"], ["only"]),
])
def test_accepts_the_text_forms(raw, expected) -> None:
    texts, err = _embedding_texts(raw)
    assert err == ""
    assert texts == expected


@pytest.mark.parametrize("raw", [
    [1, 2, 3],              # pre-tokenized, single input
    [[1, 2], [3, 4]],       # pre-tokenized, batched
])
def test_pretokenized_input_is_refused_not_stringified(raw) -> None:
    """OpenAI permits token-array input; bge-m3 cannot detokenize it.

    Refusing is the point. Embedding the repr `"[1, 2, 3]"` would return a
    perfectly well-formed vector of the WRONG THING, and nothing downstream can
    distinguish a wrong vector from a right one."""
    texts, err = _embedding_texts(raw)
    assert texts == []
    assert "pre-tokenized" in err


@pytest.mark.parametrize("raw", [None, "", [], ["ok", ""], 42, {"a": 1}])
def test_malformed_input_is_rejected_with_a_reason(raw) -> None:
    texts, err = _embedding_texts(raw)
    assert texts == []
    assert err, "a rejection must say why"


def test_float_format_passes_the_vector_through() -> None:
    vec = [0.5, -0.25, 1.0]
    assert _encode_embedding(vec, "float") == vec


def test_base64_round_trips_as_little_endian_float32() -> None:
    """The official OpenAI Python SDK requests base64 BY DEFAULT and decodes it
    itself, so this is the likely real client, not an exotic option."""
    vec = [0.5, -0.25, 1.0, 0.0]
    enc = _encode_embedding(vec, "base64")
    assert isinstance(enc, str)
    decoded = list(struct.unpack(f"<{len(vec)}f", base64.b64decode(enc)))
    assert decoded == pytest.approx(vec)


def test_token_estimate_is_never_zero_for_real_text() -> None:
    """Usage is an ESTIMATE (the shim reports no counts), but a non-empty input
    reporting 0 tokens would read as a measurement of nothing."""
    assert _estimated_embed_tokens(["hello world"]) >= 1
    assert _estimated_embed_tokens(["x" * 400]) == 100
