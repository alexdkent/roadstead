"""The five "pure computation, no I/O" modules, held to that claim.

``docs/internals.md`` calls them the crown jewels and says "keep them that way", and
until now nothing checked it. That is the shape this repo keeps having to close:
a property asserted in prose, true on the day it was written, with no mechanism
to notice the commit that ends it. The vision capability was documentation for
months; three allowlist parsers dropped unknown keys in silence; a green suite
once ran nothing.

**What "pure" buys, and therefore what this protects.** These modules are the
easiest thing in the package to test — every scheduling, costing, deadline,
spend and routing decision can be exercised against a fleet that does not exist,
in microseconds, with no fixture. The first `httpx` import into any of them
takes that away permanently and nothing else would object: the module would keep
working, the suite would keep passing, and the next person to write a test for
it would need a socket. The cost is invisible at the moment it is paid, which is
exactly why it needs a guard rather than a convention.

**What is allowed.** Deterministic stdlib, and a clock. `scheduler`, `spend`,
`cost_model` and `timeout_model` all take `now` as a parameter for their real
decisions — reading `time` for a default is not I/O and does not make a module
untestable. Importing another pure module is fine, and so is `config`, which is
dataclasses and an enum.

Each guard below was observed going red by adding the thing it forbids.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

#: The claim, module by module. Kept as a literal rather than parsed out of
#: docs/internals.md: the point is to fail when the CODE moves, and reading the list
#: from the document would make deleting a line there enough to disarm this.
PURE_MODULES = (
    "scheduler",     # DRR + priority bands + admission
    "cost_model",    # slot-second cost, EWMA-calibrated
    "timeout_model", # learned latency -> recommended deadline
    "spend",         # prices, per-caller spend, thresholds
    "intent",        # a declared capability -> an endpoint
    "rate",          # per-caller request rate, and the same threshold shape
    "goodput",       # engine work counters -> "this ENDPOINT is sick"
)

#: Top-level packages that mean I/O. Network, disk, process, framework.
FORBIDDEN_IMPORTS = frozenset({
    # network
    "httpx", "socket", "requests", "urllib", "http", "ssl", "aiohttp",
    # disk / serialization-with-a-file
    "sqlite3", "yaml", "shutil", "tempfile", "glob", "io",
    # process and signals
    "subprocess", "signal", "multiprocessing", "threading",
    # framework
    "starlette", "uvicorn", "fastapi",
    # concurrency — a pure function has nothing to await
    "asyncio",
})

#: Roadstead modules that do I/O, or reach something that does. A pure module
#: importing one of these inherits its dependencies whether it calls them or not.
FORBIDDEN_ROADSTEAD = frozenset({
    "backend", "queue", "health", "lifecycle", "http_handlers", "service",
    "enriched", "providers", "model_catalog", "on_demand", "observability",
    "correction", "identity", "acl", "sse_hub", "cache_stats", "testing",
    "client", "simulation", "test_harness", "failover", "__main__",
})

#: Builtins that touch the filesystem.
FORBIDDEN_CALLS = frozenset({"open"})


def _tree(module: str) -> ast.Module:
    path = _ROOT / "roadstead" / f"{module}.py"
    assert path.exists(), f"{path} is missing — has a pure module been renamed?"
    return ast.parse(path.read_text(encoding="utf-8"))


def test_the_list_is_not_empty_and_every_module_exists():
    """🚨 An empty or stale list would make every parametrized test below
    collect nothing and pass — the failure mode this file exists to prevent,
    reproduced in the guard itself."""
    assert len(PURE_MODULES) == 7
    for module in PURE_MODULES:
        assert (_ROOT / "roadstead" / f"{module}.py").exists(), module


@pytest.mark.parametrize("module", PURE_MODULES)
def test_a_pure_module_imports_nothing_that_does_io(module):
    """Top-level imports only — a *lazy* import inside a function is a
    different, deliberate thing (`spend._imputed_price` reaches `usage_rates`
    that way, so the rate table can be replaced without anyone thinking about
    import order) and is checked separately below."""
    offenders = []
    for node in _tree(module).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                mod = (node.module or "").split(".")[0]
                if mod in FORBIDDEN_ROADSTEAD:
                    offenders.append(f"from .{node.module} import ...")
            else:
                root = (node.module or "").split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    offenders.append(f"from {node.module} import ...")
    assert not offenders, (
        f"roadstead/{module}.py claims 'pure computation, no I/O' (docs/internals.md) "
        f"but imports {offenders}. Every decision in it is currently testable "
        f"against a fleet that does not exist; this would end that, and nothing "
        f"else would object.")


@pytest.mark.parametrize("module", PURE_MODULES)
def test_a_pure_module_has_no_coroutines(module):
    """A pure function has nothing to await. An `async def` here means the
    module has acquired something that blocks, which is I/O by another name —
    and it would drag the loop into code the scheduler calls synchronously."""
    coros = [n.name for n in ast.walk(_tree(module))
             if isinstance(n, ast.AsyncFunctionDef)]
    assert not coros, f"roadstead/{module}.py defines coroutines: {coros}"


@pytest.mark.parametrize("module", PURE_MODULES)
def test_a_pure_module_opens_no_files(module):
    """Including in a lazily-imported helper's caller. `open()` is the one
    filesystem call that needs no import and would therefore slip past the
    import sweep entirely."""
    calls = [n for n in ast.walk(_tree(module))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id in FORBIDDEN_CALLS]
    assert not calls, (
        f"roadstead/{module}.py calls {sorted({c.func.id for c in calls})} — "
        f"a filesystem read needs no import and would pass the import sweep")


@pytest.mark.parametrize("module", PURE_MODULES)
def test_a_lazy_import_inside_a_pure_module_is_still_pure(module):
    """The escape hatch, closed. A function-level import evades the sweep above
    and is exactly how the first `httpx` would arrive — inside one branch of one
    method, where it looks local and harmless."""
    tree = _tree(module)
    top = set(id(n) for n in tree.body)
    offenders = []
    for node in ast.walk(tree):
        if id(node) in top:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    offenders.append(f"lazy import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            pool = FORBIDDEN_ROADSTEAD if node.level else FORBIDDEN_IMPORTS
            if root in pool:
                offenders.append(
                    f"lazy from {'.' * node.level}{node.module} import ...")
    assert not offenders, (
        f"roadstead/{module}.py: {offenders} — a lazy import is still an "
        f"import, and it is where the first one always arrives")


def test_the_guard_is_not_vacuous():
    """A sweep that finds nothing in a file that HAS the thing would leave every
    test above green forever. Prove it detects each forbidden shape."""
    bad = ast.parse(
        "import httpx\n"
        "from .backend import BackendClientPool\n"
        "async def go(): ...\n"
        "def read(): return open('/etc/passwd').read()\n"
        "def lazy():\n"
        "    import asyncio\n")
    tops = {id(n) for n in bad.body}

    assert any(isinstance(n, ast.Import)
               and n.names[0].name in FORBIDDEN_IMPORTS for n in bad.body)
    assert any(isinstance(n, ast.ImportFrom) and n.level
               and (n.module or "") in FORBIDDEN_ROADSTEAD for n in bad.body)
    assert any(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(bad))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in FORBIDDEN_CALLS for n in ast.walk(bad))
    assert any(id(n) not in tops and isinstance(n, ast.Import)
               and n.names[0].name in FORBIDDEN_IMPORTS for n in ast.walk(bad))


def test_the_internals_doc_still_names_them_all():
    """The document and the code have to agree about which modules are the
    crown jewels — one added to the internals doc and not to `PURE_MODULES`
    would be unguarded, and a module dropped from the list here should have been
    dropped there too. `rate.py` is the sixth, added 2026-09-01; this test is
    what made the layout section get updated with it rather than a week later.

    The document was `CLAUDE.md` until 2026-09-05 and is `docs/internals.md`
    now; root `CLAUDE.md` is a three-line pointer, which this test would have
    read as a document that had silently stopped naming anything."""
    text = (_ROOT / "docs" / "internals.md").read_text(encoding="utf-8")
    for module in PURE_MODULES:
        assert f"{module}.py" in text, (
            f"docs/internals.md no longer mentions {module}.py — reconcile the "
            f"layout section with tests/test_pure_modules.py::PURE_MODULES")
    assert "pure computation, no I/O" in text
