"""`roadstead.testing` is public surface. Prove it behaves like it.

The fake backend moved out of `tests/` on 2026-08-31 because, for a gateway
whose thesis is capacity-aware admission control, "a backend that lies about its
capacity on demand" is a capability rather than furniture. Promotion is only
real if it holds up from outside the repo, so these check the three things that
would quietly undo it:

  1. it imports from the INSTALLED package, with the repo's `tests/` nowhere on
     the path — the whole point, and the one thing an in-repo test cannot show
     by importing normally, because `tests/` is already importable here;
  2. `__all__` and the module agree, in both directions;
  3. the fault library and its re-exports have not drifted apart.
"""
from __future__ import annotations

import __future__
import subprocess
import sys
import tempfile

import pytest

from roadstead import testing as rt
from roadstead.testing import fake_backend as fb


def test_imports_from_the_installed_package_with_no_repo_on_the_path():
    """Run in a subprocess from a directory that is not the repo, with `''`
    stripped from sys.path, so nothing but the installed distribution can
    satisfy the import. A plain in-process import would pass even if the module
    were still only reachable as `tests.fake_backend`."""
    script = (
        "import sys; sys.path = [p for p in sys.path if p not in ('', '.')]\n"
        "from roadstead.testing import FakeBackend, FakeBackendServer, "
        "FAULT_CAPACITY_DESYNC, ALL_FAULTS\n"
        "assert FAULT_CAPACITY_DESYNC in ALL_FAULTS\n"
        "srv = FakeBackendServer(FakeBackend()).start()\n"
        "try:\n"
        "    import urllib.request, json\n"
        "    props = json.load(urllib.request.urlopen(srv.url + '/props'))\n"
        "    assert 'total_slots' in props, props\n"
        "finally:\n"
        "    srv.stop()\n"
        "print('PUBLIC-OK')\n"
    )
    with tempfile.TemporaryDirectory() as elsewhere:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=elsewhere, capture_output=True, text=True, timeout=120,
        )
    assert proc.returncode == 0, (
        f"roadstead.testing is not usable from outside the repo\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert "PUBLIC-OK" in proc.stdout


def test_all_is_accurate_in_both_directions():
    """A name in `__all__` that does not exist breaks `from ... import *` at
    the first use; a public name missing from `__all__` is undiscoverable. Both
    are silent until someone outside the repo hits them."""
    missing = [n for n in rt.__all__ if not hasattr(rt, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"

    exported = set(rt.__all__)
    # Everything the module re-exports that is not a submodule, a dunder, or
    # the `annotations` feature flag that `from __future__ import annotations`
    # binds into every module's namespace.
    public = {
        n for n, v in vars(rt).items()
        if not n.startswith("_")
        and n != "fake_backend"
        and not hasattr(v, "__path__")
        and not isinstance(v, __future__._Feature)
    }
    assert public == exported, (
        f"re-exported but not in __all__: {sorted(public - exported)}; "
        f"in __all__ but not re-exported: {sorted(exported - public)}")


def test_every_fault_in_the_library_is_re_exported():
    """`ALL_FAULTS` is the authoritative list. A fault added to the module but
    not to `__init__` is reachable only by the deep import path, which makes the
    package's surface quietly inconsistent with its own documentation."""
    names = {
        getattr(fb, n) for n in dir(fb)
        if n.startswith("FAULT_") and isinstance(getattr(fb, n), str)
    }
    assert names == set(fb.ALL_FAULTS), (
        f"FAULT_* constants and ALL_FAULTS disagree: "
        f"only-constants={sorted(names - set(fb.ALL_FAULTS))}, "
        f"only-ALL_FAULTS={sorted(set(fb.ALL_FAULTS) - names)}")

    not_reexported = [
        n for n in dir(fb)
        if n.startswith("FAULT_") and n not in rt.__all__
    ]
    assert not not_reexported, (
        f"faults missing from roadstead.testing.__all__: {not_reexported}")


def test_the_usage_sentinels_kept_their_private_aliases():
    """`_UNSET`/`_OMIT_USAGE` were renamed to public spellings on promotion. The
    old names stay as aliases — they are the spelling in this module's own git
    history and in any monorepo copy, and an alias costs nothing."""
    assert fb._UNSET is fb.USAGE_DEFAULT
    assert fb._OMIT_USAGE is fb.OMIT_USAGE
    assert fb.USAGE_DEFAULT is not fb.OMIT_USAGE, (
        "the two sentinels must stay distinct objects — 'usage: null' and "
        "'no usage key' are different backend bugs")


@pytest.mark.parametrize("engine", ["llama.cpp", "vllm"])
def test_both_engine_shapes_are_reachable_through_the_public_name(engine):
    """The capability the promotion is FOR: stand up either engine's wire shape
    from the package name alone."""
    srv = rt.FakeBackendServer(rt.FakeBackend(engine=engine)).start()
    try:
        import httpx
        r = httpx.get(f"{srv.url}/v1/models", timeout=10.0)
        assert r.status_code == 200
        assert r.json()["data"], r.text
    finally:
        srv.stop()
