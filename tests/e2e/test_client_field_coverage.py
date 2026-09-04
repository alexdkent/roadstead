"""Every field the SDK reads, walked against a response the server really sent.

🚨 This is `tests/test_admin_ui.py`'s `pick()` guard, applied to the other
shipped surface. The UI got one because a page has no compiler and no schema:
rename a server field and one cell renders "—" forever while the page looks
healthy. `roadstead/client/_models.py` has exactly the same shape and had no
such guard — every typed view reads `self.raw.get("name")` behind a defaulting
coercion, so a renamed or never-sent field is `""`, `0.0`, `False` or `None`
for the life of the SDK, and `.raw` means nothing ever raises.

`tests/test_client_sdk.py` pins the SDK against `docs/api.md`, which is the
other half and not this one: the document publishes the error codes, the routes
and the headers, not the field names inside the envelopes. A field can be
absent from the wire while the whole two-ended pin stays green.

**The keys are extracted by AST, not by a list.** A list is a second thing to
keep in step with the module, and when it falls behind the failure is silent —
the same argument `test_admin_ui.py` makes for reading `pick()` calls out of
the page rather than enumerating them.

It found one on its first run: the SDK has ONE `Price`, and the server was
sending two different price blocks — five keys on a chat envelope, three on
`GET /rs/v1/models`, which left `ModelInfo.price.source` and `.detail` empty for
every endpoint. `source` is the field that says whether a price was published by
the provider or imputed by us, which is the same "did we measure this or guess
it" distinction the management plane reports for slot counts.
"""
from __future__ import annotations

import ast
import pathlib
from typing import Iterator

import pytest

from roadstead.client import RoadsteadClient

from .test_client_sdk_socket import _ProxyServer, live_proxy  # noqa: F401

_MODELS_PY = (pathlib.Path(__file__).resolve().parents[2]
              / "roadstead" / "client" / "_models.py")

_MSG = [{"role": "user", "content": "hi"}]

#: Keys a typed view reads that the wire is not expected to carry on every
#: response, with the reason. 🚨 An entry here is a claim that the ABSENCE is
#: correct, not that nobody has looked.
_OPTIONAL: dict[str, str] = {
    # 🚨 Both are ABSENT BY DESIGN unless the caller declared an `agent_id`,
    # and `identity_block` omits them deliberately: a `honoured: true` on every
    # ordinary call is the field firing on everything and meaning nothing —
    # the same rule that keeps `substituted` off intent-routed calls.
    #
    # Not reachable from this fixture either: the proxy runs in its own thread
    # and granting a delegation means registering a key on ITS registry, which
    # is single-loop state this repo forbids touching from another thread. The
    # populated shape is pinned on the wire instead, in
    # `test_enriched_api.py::test_an_ignored_delegation_is_visible_on_the_wire`.
    "Identity.declared": "only sent when the caller declared an agent_id",
    "Identity.honoured": "only sent when the caller declared an agent_id",
}


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _unwrap(node: ast.AST) -> ast.AST:
    """Strip the `(… or {})` that guards every nested read in `_models.py`."""
    while isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        node = node.values[0]
    return node


def _as_get(node: ast.AST):
    """`<receiver>.get("key")` -> (key, receiver), else None."""
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)):
        return node.args[0].value, _unwrap(node.func.value)
    return None


def _path_of(node: ast.AST) -> str | None:
    """Dotted path for a chain of `.get()` calls rooted at the raw dict.

    Recursive rather than one-level deep: `Plan.may_spill` reads
    `substitution.spill.allowed`, and a one-level reader records `spill.allowed`
    and then reports a field the server does send as missing. That was the first
    version of this, and it produced two false positives beside the true one.
    """
    got = _as_get(node)
    if got is None:
        return None
    key, recv = got
    if (isinstance(recv, ast.Attribute) and recv.attr == "raw") or \
            (isinstance(recv, ast.Name) and recv.id == "raw"):
        return key
    parent = _path_of(recv)
    return f"{parent}.{key}" if parent else None


def _paths_by_class() -> dict[str, set[str]]:
    tree = ast.parse(_MODELS_PY.read_text())
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        paths = {p for p in (_path_of(n) for n in ast.walk(node)) if p}
        if paths:
            out[node.name] = paths
    return out


PATHS = _paths_by_class()


def _present(raw, path: str) -> bool:
    cur = raw
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return False
        cur = cur[seg]
    return True


# --------------------------------------------------------------------------- #
# Real responses
# --------------------------------------------------------------------------- #

@pytest.fixture
def wire(live_proxy) -> Iterator[dict]:  # noqa: F811
    """One of each enriched response, from a real server over a real socket."""
    with RoadsteadClient(live_proxy.url) as rs:
        models = rs.models()
        plan = rs.plan(intent="reasoning", est_in=2000, est_out=200)
        chat = rs.chat(intent="fast-chat", messages=_MSG, max_tokens=16)
        # 🚨 All three payload types, because `/rs/v1/chat` carries all three
        # and there is ONE `CallResult` for them. A block the server assembles
        # only for a chat completion would otherwise read as its default for
        # every embedding caller, which is the same silence this file exists
        # for one dimension over.
        emb = rs.embed(texts=["a chunk"])
        rr = rs.rerank(query="q", documents=["a", "b"])
        done = [f for f in rs.stream(intent="fast-chat", messages=_MSG,
                                     max_tokens=16)
                if f.get("type") == "done"]
    assert done, "the enriched stream produced no `done` frame to check"
    yield {
        "ModelInfo": [m.raw for m in models],
        "Plan": [plan.raw],
        "CallResult": [chat.raw, emb.raw, rr.raw],
        "Identity": [chat.raw.get("identity") or {},
                     emb.raw.get("identity") or {},
                     rr.raw.get("identity") or {}],
        # 🚨 Both sites, because the SDK has one `Attribution` and one `Timing`
        # for the non-streaming envelope and the `done` frame. If they ever
        # diverge, a caller reading `result.attribution.endpoint` gets it from
        # one and an empty string from the other.
        "Attribution": [chat.attribution.raw, emb.attribution.raw,
                        rr.attribution.raw, done[0].get("attribution") or {}],
        "Timing": [chat.timing.raw, emb.timing.raw, rr.timing.raw,
                   done[0].get("timing") or {}],
        "Usage": [chat.usage.raw, emb.usage.raw, rr.usage.raw,
                  done[0].get("usage") or {}],
        # Ditto: ONE `Price` class, two wire sites.
        "Price": [m.raw.get("price") or {} for m in models]
                 + [(chat.attribution.raw.get("cost") or {}).get("price") or {}],
    }


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #

def test_the_extractor_actually_found_the_fields():
    """A reader that silently returns nothing makes every assertion below pass."""
    assert set(PATHS) >= {"Attribution", "CallResult", "Identity", "ModelInfo",
                          "Plan", "Price", "Timing", "Usage"}, sorted(PATHS)
    assert "substitution.spill.allowed" in PATHS["Plan"], (
        "the two-level read in `Plan.may_spill` was not resolved — the "
        "extractor has gone back to reading one level")
    assert "cost.price" in PATHS["Attribution"]
    assert len(set().union(*PATHS.values())) >= 45


def test_every_field_the_sdk_reads_is_on_the_wire(wire):
    """🚨 Presence, not truthiness. `substituted` is legitimately `false` and
    `ttft_ms` legitimately `null`; what must never be true is that the key is
    not there at all, because that is indistinguishable from a rename.

    Reported in one list rather than one test per class: the interesting
    failure is a field read by a view the server sends two different shapes
    for, and seeing both misses together is what names that.
    """
    misses: list[str] = []
    for cls in sorted(PATHS):
        samples = wire.get(cls)
        assert samples, f"no real response is mapped for {cls} — map one or drop it"
        for path in sorted(PATHS[cls]):
            # Keyed `Class.path`, not bare `path`: two views can read a field of
            # the same name, and exempting one of them must not silently exempt
            # the other. A bare key is still honoured for the pre-existing form.
            if f"{cls}.{path}" in _OPTIONAL or path in _OPTIONAL:
                continue
            for i, raw in enumerate(samples):
                if not _present(raw, path):
                    misses.append(f"{cls}.{path} (sample {i})")
    assert not misses, (
        "roadstead/client/_models.py reads fields the server does not send, so "
        "the SDK reports their defaults forever and nothing raises:\n  "
        + "\n  ".join(misses))
