"""Every script under `tools/` must at least import.

`tools/` is off the test path by design — its scripts spawn real processes and
send real signals, which is why they are not collected. The cost of that is
real: on 2026-08-31 the promotion of the fake backend into `roadstead.testing`
repointed all twelve callers under `tests/` and missed
`tools/sigterm_drain_probe.py`, because the sweep was scoped to where the
callers were expected rather than to the whole tree. The suite stayed green and
the tool was dead. A peer session found it, not the suite.

This is the cheapest guard that would have caught it: import each module and let
a broken import be a failure. It does not run them — `main()` is guarded in
each — so nothing here spawns a process or sends a signal.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _tool_modules() -> list[Path]:
    return sorted(p for p in TOOLS.glob("*.py") if not p.name.startswith("_"))


def test_the_tools_directory_is_not_empty():
    """🚨 Refuse to pass on an empty set — the parametrized test below would
    silently collect nothing if `tools/` moved or were renamed."""
    assert TOOLS.is_dir(), f"tools/ is missing at {TOOLS}"
    assert _tool_modules(), f"no importable scripts found in {TOOLS}"


@pytest.mark.parametrize("path", _tool_modules(), ids=lambda p: p.name)
def test_tool_imports_cleanly(path: Path):
    spec = importlib.util.spec_from_file_location(f"_tool_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
