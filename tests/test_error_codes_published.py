"""Every `code` the server puts on the wire is one `docs/api.md` §2.1 publishes.

🚨 **The bug this exists for.** §2.1 said "Fifteen exist" and listed fifteen;
`roadstead/client/_wire.py` transcribed the same fifteen; four tests checked the
two against each other, in both directions, and all of them passed. The server
emitted **seventeen**. `correction.py` minted `schema_invalid` and
`toolcall_truncated` and they reached the wire through `lifecycle.py`'s
`body["code"] = result.get("code") or "backend_error"`, which passes through any
code a correction rule attached without ever naming one.

Nothing could catch it, because every existing check compared the document to
the SDK — two hand-written transcriptions of one document, which agree with each
other exactly as long as somebody edits both and never notice a code neither has
heard of. `toolcall_truncated` cost the more expensive half: §2.2 says a
truncated structured response is *retry* work, and an unpublished code
classifies as non-deferrable in `RoadsteadError.deferrable`, so the SDK
discarded a call the next attempt would have completed.

So this reads the SOURCE, which is the only party that knows what is actually
emitted. It is deliberately one-directional: the document may publish a code no
handler emits yet (a deprecation, a code the gate in front of a handler mints),
but the server may never emit one the document does not publish.

An AST walk rather than a grep because `code` is set four different ways —
`result["code"] = "x"`, a `{"code": "x"}` literal, a `code="x"` keyword to
`_openai_error`/`Denial`, and a bare `code = "x"` local that an envelope picks
up two branches later — and because a grep for `code` also finds every
`status_code=` and `refusal_code=` in the package.
"""
from __future__ import annotations

import ast
import pathlib
import re

PKG = pathlib.Path(__file__).resolve().parent.parent / "roadstead"
API_DOC = PKG.parent / "docs" / "api.md"


def _string_literals(node: ast.AST):
    """The string constants a value expression can evaluate to.

    Descends `or` chains so that `result.get("code") or "backend_error"` yields
    its fallback — that expression is how the enriched door assigns a code at
    all, and reading only the outermost node would miss it.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            yield from _string_literals(value)


def _code_valued_expressions(tree: ast.AST):
    """Every expression assigned to something named `code`, in any of its shapes."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                named = isinstance(target, ast.Name) and target.id == "code"
                keyed = (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "code"
                )
                if named or keyed:
                    yield node.value
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "code":
                    yield value
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "code":
                    yield kw.value


def emitted_codes() -> dict[str, list[str]]:
    """code -> the source sites that mint it."""
    sites: dict[str, list[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for expr in _code_valued_expressions(tree):
            for literal in _string_literals(expr):
                sites.setdefault(literal.value, []).append(
                    f"{path.relative_to(PKG.parent)}:{literal.lineno}")
    return sites


#: The count §2.1 states in prose, which must match the list it then gives.
_COUNT_WORDS = {"fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18}


def _section_2_1() -> str:
    doc = API_DOC.read_text(encoding="utf-8")
    return doc.split("### 2.1 Codes", 1)[1].split("### 2.2", 1)[0]


def published_codes() -> set[str]:
    """§2.1's enumeration — the `·`-separated run of backticked names.

    `tests/test_client_sdk.py` takes every backticked token in the section and
    keeps the ones with an underscore, which is enough for its purpose and
    would have dropped `backpressure` and `draining` here. The enumeration is a
    single paragraph and parsing it directly is exact.
    """
    listing = next(
        para for para in _section_2_1().split("\n\n") if "·" in para)
    return set(re.findall(r"`([a-z_]+)`", listing))


def test_every_code_the_server_emits_is_published():
    emitted = emitted_codes()
    assert emitted, "an empty sweep would pass every assertion below"

    unpublished = sorted(set(emitted) - published_codes())
    assert not unpublished, (
        "the server emits error code(s) docs/api.md §2.1 does not publish:\n  "
        + "\n  ".join(f"{c} — {', '.join(emitted[c])}" for c in unpublished)
        + "\n\nAn unpublished code is not merely undocumented: the SDK's "
          "ERROR_CODES is transcribed from §2.1, and RoadsteadError.deferrable "
          "classifies a code it has never heard of as non-deferrable. Publish "
          "it in §2.1 with its deferrability, and add it to "
          "roadstead/client/_wire.py."
    )


def test_the_stated_count_matches_the_list():
    """§2.1 opens by saying how many exist. It said fifteen while the list held
    fifteen and the server emitted seventeen — a reader who trusts the sentence
    and a reader who counts the list must not be able to disagree."""
    stated = re.search(r"\*\*([A-Z][a-z]+) exist\*\*", _section_2_1())
    assert stated, "§2.1 no longer opens with a '**<N> exist**' count"
    word = stated.group(1).lower()
    assert word in _COUNT_WORDS, (
        f"§2.1 states a count of {word!r} this test cannot read — add it to "
        f"_COUNT_WORDS")
    assert _COUNT_WORDS[word] == len(published_codes()), (
        f"§2.1 says {word} exist but enumerates {len(published_codes())}")


def test_the_sweep_still_sees_the_shapes_the_package_uses():
    """A sweep that stopped matching would report zero offenders and read green.

    Each of these is a DIFFERENT assignment shape, and the sweep found the
    seventeenth code only because it handles all of them. Pinning one code per
    shape means a refactor that hides a shape from the walk fails here rather
    than silently retiring the guard above.
    """
    emitted = emitted_codes()
    for code, shape in (
        ("schema_invalid", 'result["code"] = "…"'),
        ("draining", '{"code": "…"} literal'),
        ("unknown_endpoint", 'code="…" keyword argument'),
        ("circuit_open", 'bare `code = "…"` local'),
    ):
        assert code in emitted, (
            f"the sweep no longer sees {code!r} ({shape}) — it has stopped "
            f"reading a shape the package still uses")
