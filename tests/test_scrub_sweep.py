"""The straggler sweep, as a test.

The topology half of the pre-publication scrub is mechanical, so it runs here on
every commit rather than living as a shell command in a document somebody has to
remember to run — the point being that a sweep re-run by hand months later has to
re-adjudicate every false positive from scratch, and a false positive that
survives gets rewritten through the whole history for nothing.

🚨 **Only the topology half is automatable, and pretending otherwise is the trap
that has already sprung once.** The original sweep looked only for addresses,
which is what the extraction was *known* to have carried; it could not have found
a person's name typed into an example prompt, and it did not — one was found by
hand afterwards. There is no `10.0.0.` to key on for a name. That half stays a
human pass, and this file says so rather than quietly implying the sweep is
covered:

    The address sweep is mechanical and runs on every commit. Real names — a
    host, a person, a private path typed into an example — are a HUMAN pass
    before publication; see CONTRIBUTING.md.
"""
from __future__ import annotations

import ipaddress
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Every IPv4 literal, before we ask what it means.
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

#: 🚨 The address space this repository is ALLOWED to name. Everything else in
#: private space is a straggler until somebody puts it here on purpose.
#:
#: This guard used to hunt one specific /24 — the origin fleet's. That worked
#: exactly once. It named the subnet it was hunting, in the one file guaranteed
#: to be published, which is a strange way to keep a subnet quiet; and it could
#: only ever catch the topology we already knew about. Inverted, it needs no
#: private address written down anywhere and it catches the NEXT one too.
_ALLOWED = (
    ipaddress.ip_network("0.0.0.0/32"),       # bind-any, in __main__'s --host
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("10.0.0.0/24"),      # arbitrary addresses in CIDR/ACL tests
    ipaddress.ip_network("172.16.0.0/12"),    # docker bridge and overlay ranges
    ipaddress.ip_network("192.0.2.0/24"),     # RFC 5737 TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),  # RFC 5737 TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),   # RFC 5737 TEST-NET-3
)


def _is_straggler(literal: str) -> bool:
    """A private address this repo has not sanctioned.

    Public addresses are somebody else's and are not topology — `8.8.8.8`
    appears as an example and is not a leak. A malformed match (a version
    string that looks like a dotted quad) is not an address at all.
    """
    try:
        addr = ipaddress.ip_address(literal)
    except ValueError:
        return False
    if any(addr in net for net in _ALLOWED):
        return False
    return addr.is_private


#: 🚨 The guard needs examples it would catch, so it is the one file allowed to
#: name unsanctioned private addresses. Scoped to itself deliberately: the
#: previous design exempted four files, and the docs among them then accumulated
#: real topology that the sweep was structurally unable to see.
_THE_GUARD_ITSELF = "tests/test_scrub_sweep.py"


def _tracked_files() -> list[str]:
    """Every tracked file, from git.

    🚨 It fails rather than SKIPS when git is unavailable, and the message says
    so. A scrub guard that skips is a scrub guard that passes — and it would
    pass in exactly the environment where nobody is watching it: the shipping
    image has no git, and running the suite there produced a bare
    `FileNotFoundError: 'git'` that reads like a broken test rather than an
    unmet requirement. The behaviour was already right (fail, not skip); only
    the sentence was missing.

    The tracked set is the right question even so — an untracked file cannot
    leak into a published repository, and `git ls-files` is the only thing that
    knows the difference.
    """
    _WHY = (
        "It fails instead of skipping on purpose: a scrub guard that skips is "
        "a scrub guard that passes, and this repo must not go public on a "
        "green run that checked nothing."
    )
    try:
        out = subprocess.run(["git", "ls-files"], cwd=REPO,
                             capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise AssertionError(
            f"this guard enumerates tracked files with `git`, and there is "
            f"none on PATH. {_WHY} Run the suite where git exists."
        ) from exc
    except subprocess.CalledProcessError as exc:
        # 🚨 The case that actually turns up: a DEPLOYED copy of the tree. The
        # deploy rsync excludes `.git` deliberately — the history is not needed
        # to run the code, and it still carries host names the scrub missed —
        # so `git ls-files` there fails with "not a git repository". That is not
        # this guard failing; it is an environment that cannot host the question
        # the guard asks, because the question is about a REPOSITORY and a
        # deployed tree is not one.
        raise AssertionError(
            f"`git ls-files` failed in {REPO}: {(exc.stderr or '').strip()!r}. "
            f"If this is a deployed copy of the source rather than a checkout, "
            f"that is expected — this guard is about what the REPOSITORY "
            f"tracks, and a tree without history cannot answer it. Run it on a "
            f"checkout. {_WHY}"
        ) from exc
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
        if rel == _THE_GUARD_ITSELF:
            continue
        path = REPO / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError, IsADirectoryError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for literal in _IPV4.findall(line):
                if _is_straggler(literal):
                    offenders.append(f"{rel}:{n}: {literal} — {line.strip()[:70]}")

    assert not offenders, (
        "tracked files name unsanctioned private addresses — S5 is not clean.\n"
        "Either move them into _ALLOWED space (an RFC 5737 documentation\n"
        "address is usually right) or, if the address is genuinely part of this\n"
        "repo's vocabulary, add its network to _ALLOWED with a reason:\n  "
        + "\n  ".join(offenders))


def test_the_example_catalog_uses_documentation_addresses():
    """`models.yaml` ships as the default, so its addresses are the ones a
    reader assumes are safe to copy. RFC 5737 exists so nobody can mistake them
    for real ones."""
    text = (REPO / "roadstead" / "models.yaml").read_text()
    hosts = re.findall(r"^\s+[\w-]+:\s*(\d+\.\d+\.\d+\.\d+)\s*$", text, re.M)
    assert hosts, "no addresses found — has the hosts: table moved?"
    assert all(h.startswith("192.0.2.") for h in hosts), hosts


#: 🚨 The sentence this module must never stop carrying, quoted into the
#: docstring above.
#:
#: It used to live in a planning document, and this guard read that document
#: back. The document has since been removed from the tree — it was a map of the
#: names still in the history, which is not a thing to publish — and the guard
#: went with it, which is the failure mode a cross-file assertion has: the
#: caveat's survival depended on a file nobody had promised to keep. It now
#: depends on this module, the one file that CANNOT be deleted without the guard
#: itself disappearing and the deletion being obvious.
_HUMAN_PASS_CAVEAT = (
    "The address sweep is mechanical and runs on every commit. Real names — a "
    "host, a person, a private path typed into an example — are a HUMAN pass "
    "before publication; see CONTRIBUTING.md."
)


def _squashed(text: str) -> str:
    """Whitespace-insensitive, so re-wrapping a paragraph is not a failure.

    The thing worth failing on is the caveat being DELETED, not a line break
    moving. A guard that reddens on reflow gets loosened rather than heeded.
    """
    return " ".join(text.split())


def test_this_guard_still_says_the_identifier_half_is_a_HUMAN_pass():
    """🚨 The guard against this file being mistaken for the whole sweep.

    If somebody deletes that caveat, the next reader sees a green test named
    after the sweep and concludes the sweep is covered. It is not: the
    identifier half has to be written out by hand, because there is no pattern
    to key on — which is precisely how the one real name in this tree survived
    the first sweep.
    """
    assert _squashed(_HUMAN_PASS_CAVEAT) in _squashed(__doc__ or ""), (
        "the human-pass caveat is no longer stated in this module's docstring. "
        "Restore it — the whole point of this file is that it must not read as "
        "the complete sweep.")


#: The claims that would make a reader stop doing the human pass. Deliberately a
#: SHORT, literal list rather than a clever pattern: a narrow blacklist cannot
#: false-positive on ordinary prose, and a guard that cries wolf on the front
#: page is a guard somebody deletes. It is not a proof — an overclaim phrased
#: some other way slips past, and that is accepted, because the assertion this
#: file can actually make about a human pass is the one above.
_OVERCLAIMS = (
    "fully scrubbed",
    "completely scrubbed",
    "automatically scrubbed",
    "scrubbed automatically",
    "the scrub is automated",
    "no manual review",
)

#: The two files a prospective publisher and a first-time reader actually read.
#: Scoped to them on purpose: `CHANGELOG.md` and the history docs are a RECORD
#: and may describe a past state in any words they like.
_FRONT_DOOR = ("README.md", "SECURITY.md")


@pytest.mark.parametrize("rel", _FRONT_DOOR)
def test_the_front_door_does_not_claim_automation_finished_the_scrub(rel):
    """The other half of the caveat, aimed where it would do the damage.

    This guard only proves the ADDRESS half. A front page that says otherwise
    would retire the human pass by implication, and the human pass is the half
    that has actually caught something.
    """
    text = (REPO / rel).read_text(encoding="utf-8").lower()
    found = [p for p in _OVERCLAIMS if p in text]
    assert not found, (
        f"{rel} claims the tree is scrubbed by automation ({found}). This "
        f"guard covers addresses only: {_HUMAN_PASS_CAVEAT}")


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


@pytest.mark.parametrize("literal", ["10.9.9.9", "192.168.7.7", "10.0.1.1",
                                     "169.254.1.1"])
def test_the_guard_would_actually_catch_something(literal):
    """Exercised — a sweep that matches nothing passes forever.

    🚨 None of these is the origin fleet's, and that is the point: the guard no
    longer needs to name a real private address to prove it works.

    `10.0.1.1` is the edge case — private, and one octet outside the
    `10.0.0.0/24` the allowlist permits. 🚨 A fixture here must be an address
    no history-rewrite rule would ever remap: one of these was `198.51.100.10`,
    a rule remapped it into ALLOWED space, and the test went red because its
    own example had stopped being a straggler. Written first as `172.15.0.4` on the
    theory that it sat just outside the docker range; it does not, it is
    PUBLIC, because RFC 1918's block starts at `172.16`. The test caught that,
    which is the argument for having it.
    """
    assert _is_straggler(literal)


@pytest.mark.parametrize("literal", ["127.0.0.1", "10.0.0.1", "172.16.0.5",
                                     "192.0.2.11", "198.51.100.7", "203.0.113.9",
                                     "0.0.0.0", "8.8.8.8", "1.2.3.4.5", "3.11.15"])
def test_the_guard_stays_quiet_on_what_this_repo_legitimately_names(literal):
    """The other half. A guard that flags everything is deleted within a week.

    `8.8.8.8` is public — somebody else's address is not our topology. The last
    two are not addresses at all: a sweep over dotted quads will meet version
    strings, and it must not report them.
    """
    assert not _is_straggler(literal)
