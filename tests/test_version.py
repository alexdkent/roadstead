"""`roadstead.__version__` — read from the installed distribution.

Derived from `importlib.metadata`, not hand-maintained in two places: the
version a caller sees should be the version they actually installed, and a
literal duplicated from `pyproject.toml` is exactly the kind of thing that
drifts silently the first time somebody bumps one and not the other.
"""
from __future__ import annotations


def test_version_is_a_nonempty_string():
    import roadstead

    assert isinstance(roadstead.__version__, str)
    assert roadstead.__version__


def test_version_falls_back_when_the_distribution_is_not_installed(monkeypatch):
    """Importable off a bare checkout too — no distribution metadata to read
    must not be an ImportError on the whole package."""
    import importlib

    import roadstead

    def _raise(_name: str) -> str:
        from importlib.metadata import PackageNotFoundError
        raise PackageNotFoundError()

    monkeypatch.setattr("importlib.metadata.version", _raise)
    reloaded = importlib.reload(roadstead)
    try:
        assert reloaded.__version__ == "0.0.0+unknown"
    finally:
        # Undo the patch BEFORE reloading, or the "restore" reload runs against
        # the raising stub and leaves `0.0.0+unknown` behind in a module every
        # later test shares. `monkeypatch` unwinds at teardown, which is after
        # this block — so a reload here would restore nothing, silently, and
        # only surface in whichever alphabetically-later test reads the version.
        monkeypatch.undo()
        importlib.reload(roadstead)

    # The pin for that: assert the restore actually restored, here, rather than
    # leaving it to test ordering to expose.
    assert roadstead.__version__ != "0.0.0+unknown"


def test_the_declared_version_is_a_release_not_a_dev_build():
    """`pyproject.toml` carries the version a tag will be cut from.

    The release workflow refuses to publish unless the tag equals `v` + this
    number, but it only learns that after somebody has already pushed the tag —
    and a tag is the one thing in this process that cannot be taken back. So
    the shape of the number is checked here instead, where it costs a test run:
    `main` carries the released version between releases and never a `.devN`
    suffix, because `pip install roadstead` would then resolve to something
    nobody meant to publish.
    """
    import pathlib
    import tomllib

    from packaging.version import Version

    root = pathlib.Path(__file__).resolve().parent.parent
    declared = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]

    parsed = Version(declared)
    assert parsed.is_devrelease is False, (
        f"pyproject.toml declares {declared!r}, a development version. "
        "Bump it to the release it is heading for before tagging."
    )
    assert parsed.is_prerelease is False, (
        f"pyproject.toml declares {declared!r}, a pre-release. This project has "
        "not needed one; if it does, decide that deliberately and relax this."
    )
