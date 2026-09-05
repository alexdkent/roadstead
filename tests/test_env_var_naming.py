"""Every environment variable this package reads answers to `ROADSTEAD_`.

🚨 **The bug this exists for.** The project renamed its variables to a
`ROADSTEAD_` prefix and `__main__.py` was left out — fourteen of them, including
`DATA_DIR` and `QUEUE_DB`, which decide **where the durable event log lives**.
Nothing failed, because the legacy spellings still worked. What broke was the
documented one: setting `ROADSTEAD_QUEUE_DB` was a **silent no-op**, and the
proxy quietly used a different path. Found 2026-09-02 by an operator reaching
for the documented spelling while pointing the proxy at real backends.

That is the `models.yaml` allowlist failure again — an unknown key dropped in
silence, costing whoever spelled it the way the docs say. `acl.py` had already
solved it correctly for its own variable (read both, warn on the legacy one);
the fix was to apply that pattern everywhere, and this is the guard that keeps
it applied.

Asserted by reading the SOURCE rather than by importing and probing, because a
variable that is only read on one branch would not be probed by any import.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

PKG = pathlib.Path(__file__).resolve().parent.parent / "roadstead"

#: The legacy prefix. A read of one of these is fine — it is how a pre-rename
#: deployment keeps working — but ONLY from the shared helper that also honours
#: the new spelling and warns.
LEGACY = "LLM_PROXY_"

#: The single sanctioned place a legacy name may be constructed.
HELPER = "env_with_legacy_prefix"


def _py_files() -> list[pathlib.Path]:
    return [p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_module_reads_a_legacy_env_var_directly():
    """`os.environ.get("LLM_PROXY_…")` outside the helper is the bug returning.

    A direct read is exactly what leaves the documented `ROADSTEAD_` spelling
    inert, and it is invisible: the legacy name still works, so nothing fails
    until somebody follows the documentation.
    """
    offenders: list[str] = []
    for path in _py_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if not node.value.startswith(LEGACY):
                continue
            # acl.py names both spellings in a tuple it loops over, which IS the
            # dual-read pattern; config.py's helper builds the name itself.
            if path.name in ("acl.py", "config.py"):
                continue
            offenders.append(f"{path.relative_to(PKG.parent)}:{node.lineno} {node.value!r}")

    assert not offenders, (
        "a legacy LLM_PROXY_* variable is read directly:\n  "
        + "\n  ".join(offenders)
        + f"\n\nRoute it through config.{HELPER}(), which reads the ROADSTEAD_ "
          "spelling first and warns on the legacy one. A direct read leaves the "
          "documented name silently inert."
    )


def test_the_helper_prefers_the_new_spelling_and_warns_on_the_old(monkeypatch, caplog):
    """The helper's contract, on the thing that decides.

    🚨 The precedence is the load-bearing half. If the legacy name won, an
    operator migrating could not override the old value without deleting it —
    and the whole point is that the documented spelling works.
    """
    from roadstead.config import env_with_legacy_prefix

    monkeypatch.delenv("ROADSTEAD_QUEUE_DB", raising=False)
    monkeypatch.delenv("LLM_PROXY_QUEUE_DB", raising=False)
    assert env_with_legacy_prefix("QUEUE_DB", "fallback") == "fallback"

    # Legacy alone: honoured, and it says so.
    monkeypatch.setenv("LLM_PROXY_QUEUE_DB", "/old/path.db")
    with caplog.at_level("WARNING"):
        assert env_with_legacy_prefix("QUEUE_DB") == "/old/path.db"
    assert any("pre-rename" in r.getMessage() for r in caplog.records), (
        "the legacy read must warn — a silent fallback is how the two spellings "
        "drifted apart in the first place")

    # Both set: the NEW one wins, silently.
    monkeypatch.setenv("ROADSTEAD_QUEUE_DB", "/new/path.db")
    assert env_with_legacy_prefix("QUEUE_DB") == "/new/path.db"


def test_the_shipped_image_keeps_durable_state_off_the_writable_layer():
    """🚨 The production defect, guarded where it actually bit.

    The default data dir is the XDG state directory (`__main__.default_data_dir`),
    which is a home directory the shipped image has no business writing to — and
    before 2026-09-05 it was `/tmp/agents/llmproxy`, which was worse. In a real container
    on 2026-09-02 `queue.db` sat on the ephemeral writable layer while the
    mounted volume held only the admin overlay, so every rebuild reset the DRR
    balances, the day's spend and the endpoint drain state: precisely the rows
    the bounded SIGTERM drain and the published 108s stop-grace exist to flush.

    Asserted against the Dockerfile because that is what makes it true for
    everyone who runs the image — a default in Python would also have to be
    right for the non-container case, where `/tmp` is a defensible choice.
    """
    dockerfile = (PKG.parent / "Dockerfile").read_text()

    m = re.search(r"^ENV\s+ROADSTEAD_DATA_DIR=(\S+)\s*$", dockerfile, re.M)
    assert m, (
        "the image sets no ROADSTEAD_DATA_DIR, so the durable event log falls "
        "back to /tmp INSIDE the container — lost on every recreate, which is "
        "the failure the whole shutdown budget exists to prevent.")
    data_dir = m.group(1)
    assert not data_dir.startswith(("/tmp", "/var/tmp", "/dev/shm")), (
        f"ROADSTEAD_DATA_DIR={data_dir} is ephemeral storage")
    assert re.search(rf'^VOLUME \["{re.escape(data_dir)}"\]', dockerfile, re.M), (
        f"{data_dir} is not declared as a VOLUME, so `docker run` with no -v "
        f"puts it back on the writable layer — the same loss, one step removed.")


@pytest.mark.parametrize("path", ["/tmp/agents/llmproxy/queue.db",
                                  "/var/tmp/x/queue.db",
                                  "/dev/shm/queue.db",
                                  # 🚨 macOS's real /tmp. Omitted at first, and
                                  # the warning was silent on the very machine
                                  # the default was being exercised on.
                                  "/private/tmp/agents/llmproxy/queue.db"])
def test_ephemeral_durable_state_is_disclosed_at_startup(path, caplog):
    """The non-container half: disclosed, not moved.

    Changing the default would relocate an existing deployment's state on
    upgrade — a worse failure than the one being fixed — so the surprising
    behaviour is made loud instead. Same doctrine as every other disclosure
    here: the default is defensible, its silence was not.
    """
    from roadstead.__main__ import _warn_if_durable_state_is_ephemeral

    with caplog.at_level("WARNING"):
        assert _warn_if_durable_state_is_ephemeral(path) is True
    assert any("EPHEMERAL" in r.getMessage() for r in caplog.records)


def test_a_persistent_path_is_not_warned_about():
    """The other direction, so the warning cannot degrade into always-on noise
    that an operator learns to skip past."""
    from roadstead.__main__ import _warn_if_durable_state_is_ephemeral

    assert _warn_if_durable_state_is_ephemeral("/var/lib/roadstead/queue.db") is False
