"""The straggler sweep (`docs/corpus_and_scrub_plan.md` S5), as a test.

This repo is **private until the scrub is done**, and S5 is a shell command in a
document that somebody has to remember to run. The topology half of it is
mechanical, so it runs here on every commit instead — the point being that a
sweep re-run by hand months later has to re-adjudicate every false positive from
scratch, and a false positive that survives goes into S6's `replacements.txt`
and gets rewritten through 312 commits of history for nothing.

🚨 **Only the topology half is automatable, and pretending otherwise is the
trap S4 already sprang.** The original sweep looked only for addresses, which is
what the extraction was *known* to have carried; it could not have found a
person's name typed into an example prompt, and it did not — S4 found exactly
that afterwards. There is no `10.0.0.` to key on for a name. That half stays a
human pass, and this file says so rather than quietly implying the sweep is
covered.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: The private fleet's subnet. Anything matching this in a shipped file is
#: either real topology or an arbitrary address that merely looks like it — and
#: from a sweep's side those are indistinguishable, which is the whole problem.
_FLEET_SUBNET = re.compile(r"10\.0\.0\.[0-9]+")

#: Files that describe the scrub itself and therefore quote the pattern.
_MAY_DESCRIBE_THE_SCRUB = {
    "CLAUDE.md",
    "docs/corpus_and_scrub_plan.md",
    "docs/roadmap.md",
    "tests/test_scrub_sweep.py",
}


def _tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_no_tracked_file_carries_the_private_subnet():
    """S5's first line, run on every commit.

    🚨 It fired for real on 2026-09-01, on three lines nobody would call a leak:
    `10.0.0.1` used as an arbitrary address in two CIDR tests. Not private
    topology — and it does not matter, because a sweep cannot tell, so each one
    costs a human adjudication every time the sweep is re-run and would have
    been rewritten through the whole history by S6 for nothing. Use `10.0.0.1`,
    or an RFC 5737 documentation address (`192.0.2.0/24`).
    """
    offenders = []
    for rel in _tracked_files():
        if rel in _MAY_DESCRIBE_THE_SCRUB:
            continue
        path = REPO / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError, IsADirectoryError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if _FLEET_SUBNET.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()[:90]}")

    assert not offenders, (
        "tracked files carry the private fleet's subnet — S5 is not clean:\n  "
        + "\n  ".join(offenders))


def test_the_example_catalog_uses_documentation_addresses():
    """`models.yaml` ships as the default, so its addresses are the ones a
    reader assumes are safe to copy. RFC 5737 exists so nobody can mistake them
    for real ones."""
    text = (REPO / "roadstead" / "models.yaml").read_text()
    hosts = re.findall(r"^\s+[\w-]+:\s*(\d+\.\d+\.\d+\.\d+)\s*$", text, re.M)
    assert hosts, "no addresses found — has the hosts: table moved?"
    assert all(h.startswith("192.0.2.") for h in hosts), hosts


def test_the_plan_still_says_the_identifier_half_is_a_HUMAN_pass():
    """🚨 The guard against this file being mistaken for the whole sweep.

    If somebody deletes that caveat, the next reader sees a green test named
    after S5 and concludes the sweep is covered. It is not: the identifier half
    has to be written out by hand, because there is no pattern to key on — which
    is precisely how S4's finding survived the first sweep.
    """
    plan = (REPO / "docs" / "corpus_and_scrub_plan.md").read_text()
    assert "written out by hand" in plan
    assert "S4" in plan


def test_endpoint_normalization_has_no_hardcoded_fleet_NAMES():
    """🚨 The other kind of straggler: a private fleet's vocabulary in *code*,
    not in a comment.

    Until 2026-09-01 `normalize_endpoint` stripped a `nexus-` prefix and mapped
    a bare `nexus` to `chat` — one fleet's host naming, hardcoded since the
    first commit and shipped to everyone. It was a SECOND aliasing mechanism
    beside `models.yaml`'s `aliases:`: unconfigurable, unoverridable, invisible
    to the duplicate-alias notice in `model_catalog`, and it silently rewrote
    any endpoint whose name began with those six characters.

    Asserted behaviourally rather than by grepping for the word, because the
    next one will be spelled differently: a name the catalog does not know must
    come back unchanged, whatever it looks like.
    """
    from roadstead.config import ROLE_TO_CLASS, normalize_endpoint

    for name in ("nexus", "nexus-analyst", "anvil-thing", "nasbox-whisper",
                 "some-host-prefixed-name", "totally-unknown"):
        assert name not in ROLE_TO_CLASS, f"fixture stale: {name} is a real role"
        assert normalize_endpoint(name) == name, (
            f"{name!r} was rewritten by something other than the catalog — a "
            "second aliasing mechanism has come back")

    # The catalog's own aliases still work: that is the mechanism this defends.
    assert normalize_endpoint("reasoner") == ROLE_TO_CLASS["reasoner"]


@pytest.mark.parametrize("pattern", ["10.0.0.9", "10.0.0.10"])
def test_the_guard_would_actually_catch_something(pattern, tmp_path):
    """The regex, exercised — a sweep that matches nothing passes forever."""
    assert _FLEET_SUBNET.search(f"host: {pattern}")
    assert not _FLEET_SUBNET.search("host: 192.0.2.11")
    assert not _FLEET_SUBNET.search("host: 10.0.0.1")
