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
        importlib.reload(roadstead)  # restore the real value for later tests
