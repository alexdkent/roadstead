"""Phase 0 hardening — runtime feature flags (flags.py + /v1/admin/flags).

The flags are the shadow→enforce flip surface (no env gates — a running
process can't be re-enved, and the scheduled auto-flips need an API). Pins:
defaults, persistence round-trip, unknown-key rejection, handler ACL, and
exposure on /v1/status.
"""

from __future__ import annotations

import json

import pytest

from roadstead.config import ProxyConfig
from roadstead.flags import DEFAULT_FLAGS, RuntimeFlags
from roadstead.service import ProxyService


class _Req:
    def __init__(self, host="127.0.0.1", method="GET", body=None):
        class _Client:
            pass

        _Client.host = host
        self.client = _Client()
        self.method = method
        self.headers: dict = {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


# --- RuntimeFlags unit ---------------------------------------------------------

def test_defaults_without_file():
    f = RuntimeFlags(None)
    assert f.as_dict() == DEFAULT_FLAGS
    assert f.get("inject_stream_usage") is True
    assert f.get("unknown_endpoint_enforce") is False
    assert f.get("context_gate_enforce") is False


def test_unknown_flag_get_raises():
    # A consuming-code typo must fail loud, never silently read False.
    with pytest.raises(KeyError):
        RuntimeFlags(None).get("no_such_flag")


def test_set_many_persists_and_reloads(tmp_path):
    path = tmp_path / "runtime_flags.json"
    f = RuntimeFlags(path)
    f.set_many({"unknown_endpoint_enforce": True})
    assert f.get("unknown_endpoint_enforce") is True

    # A fresh instance (process restart) sees the persisted value.
    f2 = RuntimeFlags(path)
    assert f2.get("unknown_endpoint_enforce") is True
    assert f2.get("context_gate_enforce") is False  # untouched default


def test_set_many_rejects_unknown_key_and_non_bool(tmp_path):
    f = RuntimeFlags(tmp_path / "f.json")
    with pytest.raises(ValueError):
        f.set_many({"typo_flag": True})
    with pytest.raises(ValueError):
        f.set_many({"context_gate_enforce": "yes"})
    with pytest.raises(ValueError):
        f.set_many({})
    # Nothing persisted by the failed updates.
    assert f.as_dict() == DEFAULT_FLAGS


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "runtime_flags.json"
    path.write_text("{not json")
    assert RuntimeFlags(path).as_dict() == DEFAULT_FLAGS


def test_unknown_key_in_file_ignored(tmp_path):
    path = tmp_path / "runtime_flags.json"
    path.write_text(json.dumps(
        {"unknown_endpoint_enforce": True, "renamed_old_flag": True}))
    f = RuntimeFlags(path)
    assert f.get("unknown_endpoint_enforce") is True
    assert "renamed_old_flag" not in f.as_dict()


# --- /v1/admin/flags handler ----------------------------------------------------

@pytest.mark.asyncio
async def test_handler_get_and_post_roundtrip(tmp_path):
    cfg = ProxyConfig(runtime_flags_path=str(tmp_path / "flags.json"))
    svc = ProxyService(cfg)

    resp = await svc.handle_admin_flags(_Req(method="GET"))
    assert resp.status_code == 200
    assert json.loads(resp.body)["flags"] == DEFAULT_FLAGS

    resp = await svc.handle_admin_flags(
        _Req(method="POST", body={"context_gate_enforce": True}))
    assert resp.status_code == 200
    assert json.loads(resp.body)["flags"]["context_gate_enforce"] is True
    assert svc._flags.get("context_gate_enforce") is True

    # Persisted: a new service instance over the same path sees the flip.
    svc2 = ProxyService(ProxyConfig(runtime_flags_path=str(tmp_path / "flags.json")))
    assert svc2._flags.get("context_gate_enforce") is True


@pytest.mark.asyncio
async def test_handler_rejects_bad_input_and_denies_unknown_ip(tmp_path):
    cfg = ProxyConfig(runtime_flags_path=str(tmp_path / "flags.json"))
    svc = ProxyService(cfg)

    resp = await svc.handle_admin_flags(_Req(method="POST", body={"typo": True}))
    assert resp.status_code == 400

    resp = await svc.handle_admin_flags(_Req(method="POST", body=None))
    assert resp.status_code == 400

    resp = await svc.handle_admin_flags(_Req(host="8.8.8.8", method="GET"))
    assert resp.status_code == 403
    # The failed update changed nothing.
    assert svc._flags.as_dict() == DEFAULT_FLAGS


@pytest.mark.asyncio
async def test_status_exposes_flags_and_admin_audit():
    svc = ProxyService(ProxyConfig())
    # One admin hit so the audit map is non-empty.
    await svc.handle_admin_flags(_Req(method="GET"))

    status = json.loads((await svc.handle_status(_Req())).body)
    assert status["flags"] == DEFAULT_FLAGS
    assert status["reliability"]["admin_ips_seen"]["/v1/admin/flags"] == ["127.0.0.1"]


# ---------------------------------------------------------------------------
# The 2026-08-31 environment-variable rename
# ---------------------------------------------------------------------------

def test_a_retired_collective_env_var_is_reported_not_silently_ignored(
        monkeypatch, caplog):
    """🚨 The dangerous half of a rename is not the flag that stops working, it
    is the flag that stops working IN SILENCE. An operator who had turned a
    correction layer off gets it back on, and nothing says so.

    The old name is NOT honoured — two spellings for one switch is how they end
    up disagreeing — so the contract is exactly: ignored, and loud about it.
    """
    from roadstead.__main__ import warn_on_retired_env_vars

    monkeypatch.setenv("COLLECTIVE_PROXY_SCHEMA_BACKSTOP", "0")
    with caplog.at_level("WARNING"):
        found = warn_on_retired_env_vars()
    assert found == ["COLLECTIVE_PROXY_SCHEMA_BACKSTOP"]
    text = caplog.text
    assert "COLLECTIVE_PROXY_SCHEMA_BACKSTOP" in text
    assert "ROADSTEAD_PROXY_SCHEMA_BACKSTOP" in text, (
        "the warning must name the REPLACEMENT — an operator reading it should "
        "not have to guess the new spelling")


def test_no_collective_env_var_is_still_read_anywhere():
    """The rename is only complete if nothing reads the old prefix. A leftover
    reader would make one flag answer to two names, which is how the two end up
    disagreeing."""
    import re
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[1] / "roadstead"
    offenders = []
    for path in sorted(pkg.rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""environ(\.get)?\(?\s*["']COLLECTIVE_""", line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"still reading retired COLLECTIVE_* names: {offenders}"
