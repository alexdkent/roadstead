"""The published default port must match the code's default port.

🚨 **The bug this exists for.** `ProxyConfig.port` and `--port`'s argparse
default are `42161` (`roadstead/config.py`, `roadstead/__main__.py`), and the
`EXPOSE` line in `Dockerfile` agrees. `README.md`'s quick-start curl/Python
examples and the client SDK's own docstrings said `42100` instead — a stale
number left behind by the extraction, not a deliberate alternate port. A reader
who copy-pasted the quick start got a connection refused with no obvious cause.

Read the default from the CODE rather than hardcoding it here, so a future
deliberate port change only has to update one place and this guard follows it
rather than needing to be re-taught the number.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Files this guard does not police. `CHANGELOG.md` narrates history and may
#: legitimately mention the old port in a past-tense entry; this file is the
#: one place allowed to name the stale port, for the same reason
#: `test_scrub_sweep.py` exempts itself.
_EXEMPT = {"CHANGELOG.md", "tests/test_docs_port_consistency.py"}

#: Deliberately empty. The first draft of this guard exempted `docs/api.md`
#: because "another fix was in flight" — which left the guard green while the
#: one remaining stale line (api.md's async client example) shipped. A guard
#: that exempts the file with the defect is a guard that passes for a reason
#: unrelated to the code. Nothing goes in here without the defect fixed first.
_EXEMPT_EXTRA: set[str] = set()


def _default_port() -> int:
    from roadstead.config import ProxyConfig
    return ProxyConfig().port


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_no_doc_or_source_file_mentions_the_stale_port():
    port = _default_port()
    assert port != 42100, "the code's default port changed — update this guard's assumptions"

    stale = re.compile(r"localhost:42100|:42100\b")
    offenders = []
    for rel in _tracked_files():
        if rel in _EXEMPT or rel in _EXEMPT_EXTRA:
            continue
        if not (rel == "README.md" or rel.startswith("docs/") or rel.startswith("roadstead/")):
            continue
        if not (rel.endswith(".md") or rel.endswith(".py")):
            continue
        path = REPO / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if stale.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()[:80]}")

    assert not offenders, (
        f"stale port 42100 found where the code's default is {port} — "
        "update these to match roadstead.config.ProxyConfig().port:\n  "
        + "\n  ".join(offenders))
