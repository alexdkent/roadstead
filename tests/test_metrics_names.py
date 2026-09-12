"""The `/metrics` series names are contract, and the old family may not come back.

🚨 **What this exists for.** Every Prometheus series shipped under the origin
project's name — `llmproxy_queue_wait_ms`, `llmproxy_alerts_active`, twenty-two
of them — as did the greppable `LLMPROXY_*` log markers. The whole family was
renamed to `roadstead_*` / `ROADSTEAD_*` on **2026-09-05**, deliberately before
publication, because a metric name is not like a log line: it is a *stored key*
in every dashboard panel and alert selector that has ever scraped the proxy, and
renaming one **empties a graph in silence** rather than failing anything. There
was exactly one window in which that was cheap, and it closed with that commit
(`docs/compatibility.md`).

So two directions are guarded here, and they answer different questions:

* **Backwards** — no `llmproxy_` series name and no `LLMPROXY_` marker survives
  anywhere under `roadstead/`. A rename spread across ten modules is exactly
  the shape that a later copy-paste from an old branch reintroduces one line of.
* **Forwards** — the builder's set of names and the table in `docs/api.md` §3 are
  the same set, checked both ways. A metric that exists and is undocumented gets
  relied on by nobody; a metric that is documented and no longer emitted is worse,
  because a consumer builds an alert on it and the alert never fires.

Asserted by reading the SOURCE of the builder rather than by scraping a live
`/metrics`, because most of these series are conditional — an endpoint below the
structured-sample floor, a caller with no truncations — so a scrape of a quiet
proxy would silently under-report the set and pass while documenting nothing.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "roadstead"
API_MD = ROOT / "docs" / "api.md"

#: The retired family. Kept as a literal rather than assembled from parts so
#: that grepping this repo for the old name lands here, on the explanation.
LEGACY_METRIC_PREFIX = "llmproxy_"
LEGACY_MARKER_PREFIX = "LLMPROXY_"

#: Filesystem names that legitimately still carry the old word. Renaming these
#: relocates a running deployment's durable state or points at a document that
#: exists under that name in the origin monorepo — both worse than a stale
#: spelling. Anything NOT ending in an extension is a wire name and is refused.
_FILENAME = re.compile(r"llmproxy_[a-z0-9_]+\.[a-z]+")


def _py_files() -> list[pathlib.Path]:
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _hits(pattern: str) -> list[str]:
    rx = re.compile(pattern)
    out: list[str] = []
    for path in _py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for m in rx.finditer(line):
                out.append(f"{path.relative_to(ROOT)}:{lineno} {m.group(0)!r}")
    return out


# --------------------------------------------------------------------------
# Backwards: the old family is gone and stays gone
# --------------------------------------------------------------------------

def test_no_legacy_marker_survives():
    """`LLMPROXY_*` has no legitimate remaining use — no allowlist, ever.

    Unlike the metric names there is no filesystem case to carve out: every one
    of these was a log marker, and all ten moved in one commit.
    """
    offenders = _hits(LEGACY_MARKER_PREFIX + r"[A-Z_]+")
    assert not offenders, (
        f"a retired {LEGACY_MARKER_PREFIX}* log marker is back:\n  "
        + "\n  ".join(offenders)
        + f"\n\nUse the ROADSTEAD_ spelling. The markers were renamed 2026-09-05 "
          "with the metric family; a mixed pair means an operator's grep finds "
          "half the events."
    )


def test_no_legacy_metric_name_survives():
    """A surviving `llmproxy_` token must be a FILENAME, not a wire name.

    The distinction is the whole point of the rename: what an external consumer
    STORES moved, what a deployment has on disk did not.
    """
    offenders = [h for h in _hits(LEGACY_METRIC_PREFIX + r"[a-z0-9_]*(\.[a-z]+)?")
                 if not _FILENAME.search(h)]
    assert not offenders, (
        f"a retired {LEGACY_METRIC_PREFIX}* series name is back:\n  "
        + "\n  ".join(offenders)
        + "\n\nEmit it as roadstead_* and add it to the table in docs/api.md §3."
    )


def test_no_legacy_log_wording_survives():
    """The lower-case half — `llmproxy started`, `llmproxy db write failed`.

    These are what an operator tails, and they were the last thing in the log
    still announcing the old project by name.
    """
    offenders = _hits(r'"llmproxy [a-z%]')
    assert not offenders, (
        "a log line still names the old project:\n  " + "\n  ".join(offenders))


# --------------------------------------------------------------------------
# Forwards: the builder and docs/api.md are one set, checked both ways
# --------------------------------------------------------------------------

def _emitted_metric_names() -> set[str]:
    """Every `roadstead_*` name the `/metrics` builder can put on the wire.

    Read as string constants anywhere inside `handle_prometheus_metrics`, which
    catches both the direct `Metric("name", ...)` calls and the `(key, name)`
    tuple the since-boot counters loop over — a signature-shaped scan would miss
    the second form.
    """
    tree = ast.parse((PKG / "http_handlers.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.AsyncFunctionDef)
                and node.name == "handle_prometheus_metrics"):
            return {
                n.value for n in ast.walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and re.fullmatch(r"roadstead_[a-z0-9_]+", n.value)
            }
    pytest.fail("handle_prometheus_metrics is gone — this guard is measuring nothing")


def _documented_metric_names() -> set[str]:
    """The names in the §3 `/metrics` table, read out of the shipped document.

    `docs/api.md` is executable contract here (`docs/compatibility.md` step 4):
    six other tests already read it back, and this is the seventh.
    """
    rows = re.findall(r"^\|\s*`(roadstead_[a-z0-9_]+)`\s*\|", API_MD.read_text(), re.M)
    assert rows, ("the /metrics enumeration is missing from docs/api.md — without "
                  "it this test passes vacuously in both directions")
    return set(rows)


def test_every_emitted_metric_is_documented():
    """An undocumented series is one nobody can rely on, so nobody uses it."""
    missing = _emitted_metric_names() - _documented_metric_names()
    assert not missing, (
        "emitted by /metrics but absent from the docs/api.md §3 table:\n  "
        + "\n  ".join(sorted(missing))
        + "\n\nAdd a row: name, type, labels, one line on what it is."
    )


def test_every_documented_metric_is_emitted():
    """🚨 The direction that matters more.

    A documented-but-dead series is not a gap, it is a false promise: a consumer
    writes an alert rule against it and the rule never fires — indistinguishable,
    from the outside, from a system that is always healthy.
    """
    stale = _documented_metric_names() - _emitted_metric_names()
    assert not stale, (
        "documented in docs/api.md §3 but no longer emitted:\n  "
        + "\n  ".join(sorted(stale))
        + "\n\nRemoving a series is a breaking change (docs/compatibility.md) — "
          "if it was deliberate, drop the row and write the CHANGELOG entry."
    )


def test_the_documented_set_is_the_whole_set():
    """Pin the count, so a bulk edit that drops half the table is not silently
    consistent with a builder edit that drops the same half."""
    assert len(_documented_metric_names()) == 30
