"""`service.py`'s delegators must keep the signatures of the handlers they call.

`ProxyService` forwards ~30 HTTP methods straight to `ProxyHttpHandlers`, and
since Workstream C three more to `EnrichedApi`. The delegation is mechanical, so
nothing enforces it: change a handler's signature and the delegator still
*compiles*, still forwards, and fails at runtime on the one route nobody
exercised.

`docs/api.md` §5 asserted the two were identical, but the original audit took
that on trust "without opening service.py". Audited by AST on 2026-08-31 — 30
delegators, 30 identical signatures, zero drift — and turned into this test so
the claim stays true rather than being re-checked by hand.

🚨 The sweep follows every `self._<collaborator>` in `_COLLABORATORS`, not just
`self._http`. A new north face reached through a new attribute would otherwise
be the one part of the surface with no drift guard — which is exactly the state
the OpenAI door was in before this test existed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
SERVICE = _ROOT / "roadstead" / "service.py"

#: ProxyService attribute -> (module, class) it forwards to.
_COLLABORATORS = {
    "_http": (_ROOT / "roadstead" / "http_handlers.py", "ProxyHttpHandlers"),
    "_enriched": (_ROOT / "roadstead" / "enriched.py", "EnrichedApi"),
}

#: Fewer than this means the sweep has gone blind, not green (docs/ledger.md,
#: "a green suite that ran nothing"). The count was 30 when this was written.
MIN_DELEGATORS = 25


def _methods(path: Path, cls_name: str) -> dict[str, ast.AST]:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return {n.name: n for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise AssertionError(f"class {cls_name} not found in {path}")


def _signature(fn) -> str:
    """Render a signature, annotations and defaults included, order-sensitive."""
    a = fn.args

    def one(arg, default):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        dflt = f" = {ast.unparse(default)}" if default is not None else ""
        return f"{arg.arg}{ann}{dflt}"

    positional = a.posonlyargs + a.args
    defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    parts = [one(x, d) for x, d in zip(positional, defaults)]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    parts += [one(x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults)]
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"({', '.join(parts)}){ret}"


def _delegations() -> list[tuple[str, str, str]]:
    """(service method, collaborator attr, handler method) for every forward."""
    out = []
    for name, fn in _methods(SERVICE, "ProxyService").items():
        for node in ast.walk(fn):
            func = getattr(node, "func", None)
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Attribute)
                    and func.value.attr in _COLLABORATORS):
                out.append((name, func.value.attr, func.attr))
    return sorted(set(out))


def test_every_collaborator_is_actually_reached():
    """A collaborator listed here and never forwarded to means the sweep is
    watching a surface that moved — which reads identical to a surface with no
    drift."""
    reached = {attr for _, attr, _ in _delegations()}
    assert reached == set(_COLLABORATORS), (
        f"listed {sorted(_COLLABORATORS)} but only reached {sorted(reached)}")


def test_the_sweep_finds_the_delegators():
    found = _delegations()
    assert len(found) >= MIN_DELEGATORS, (
        f"only {len(found)} delegations found (expected >= {MIN_DELEGATORS}) — "
        f"the AST sweep has gone blind, or the delegation pattern changed")


@pytest.mark.parametrize("svc_name,attr,handler_name", _delegations(),
                         ids=lambda v: v if isinstance(v, str) else str(v))
def test_delegator_signature_matches_its_handler(svc_name, attr, handler_name):
    path, cls = _COLLABORATORS[attr]
    handlers = _methods(path, cls)
    assert handler_name in handlers, (
        f"ProxyService.{svc_name} forwards to {attr}.{handler_name}, which does "
        f"not exist on {cls}")
    svc_sig = _signature(_methods(SERVICE, "ProxyService")[svc_name])
    handler_sig = _signature(handlers[handler_name])
    assert svc_sig == handler_sig, (
        f"signature drift:\n"
        f"  ProxyService.{svc_name}{svc_sig}\n"
        f"  ProxyHttpHandlers.{handler_name}{handler_sig}")
