"""Arm the concurrency invariant so a violation FAILS instead of corrupting.

``CLAUDE.md`` opens with the most dangerous thing in this repo, and says so:

    🚨 The concurrency invariant — NOT guarded by any test.
    Single event loop. No locks on in-memory scheduler / budget / cache state.
    That is only safe because there is exactly one thread mutating it.

That was true. Everything the scheduler, the DRR budgets, the price book, the
spend ledger and the response cache hold is a plain dict mutated without a lock,
and the only thing making that safe is a convention — one loop thread — which
nothing checked. A second writer would not raise; it would interleave, and the
symptom would be a DRR budget that drifts or a request served twice, weeks later
and nowhere near the commit that caused it.

This module makes the convention checkable.

---

## How it works

:func:`arm` wraps the mutating methods of the single-loop state objects with a
recorder that captures ``threading.get_ident()`` on every call. :func:`violations`
then reports every object touched from more than one thread. That is the whole
trick, and it is deliberately not clever: no ``sys.settrace``, no import hooks,
no bytecode rewriting — those cost 10-50x and would make a soak measure the
instrumentation instead of the proxy.

🚨 **It records rather than raising on the spot.** A raise inside a mutation
would unwind through the scheduler and be swallowed by one of the fail-open
guards the hot path is full of, which is exactly how a guard reads green while
detecting nothing. Recording and asserting at the end cannot be swallowed.

## What it is for and what it is not

**For:** running under load in an e2e test, or under ``tools/soak.py``, where
concurrency is real. A violation here is a genuine defect — `CLAUDE.md`'s
"never add a `workers=` parameter, a thread pool that WRITES, or a second thread
that touches scheduler or budget state".

**Not for:** production. It allocates a small record per mutation and is test
scaffolding, which is why it lives in ``tests/`` rather than in the package
beside ``roadstead.testing``.

🚨 **The DB writer thread is EXPECTED to be a second thread** and is not a
violation: it is the one sanctioned background writer, it owns its own
connection, and it touches none of the state armed here. ``queue.py`` is
therefore deliberately out of scope — its own invariant (one writer connection,
a thread-local read connection per pool thread) is guarded by
``tests/test_offloop_reads.py``, which is a different question with a different
answer.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class _Touch:
    """Every thread that has mutated one object, and how often."""

    label: str
    by_thread: dict[int, int] = field(default_factory=dict)
    #: One example method name per thread, so a report names what was called
    #: rather than only that something was.
    example: dict[int, str] = field(default_factory=dict)

    def record(self, method: str) -> None:
        ident = threading.get_ident()
        self.by_thread[ident] = self.by_thread.get(ident, 0) + 1
        self.example.setdefault(ident, method)

    @property
    def threads(self) -> int:
        return len(self.by_thread)


class LoopAffinity:
    """A live record of which threads mutated which single-loop state."""

    def __init__(self) -> None:
        self.touches: dict[str, _Touch] = {}
        self._undo: list[tuple[object, str, object]] = []
        # Counted separately from `_undo`, which `disarm` empties. The natural
        # order of use is arm → load → disarm → assert (so the proxy is not
        # left instrumented if an assertion raises), and deriving the count from
        # `_undo` made "we disarmed first" indistinguishable from "we armed
        # nothing" — the exact failure `assert_single_threaded` exists to catch.
        self._armed = 0

    # ---- arming ----

    def watch(self, obj: object, label: str, methods: tuple[str, ...]) -> None:
        """Wrap ``methods`` on ``obj`` so every call records its thread."""
        touch = self.touches.setdefault(label, _Touch(label=label))
        for name in methods:
            original = getattr(obj, name, None)
            if original is None or not callable(original):
                # Not an error: this module names methods from several modules
                # and a rename should not silently disarm the guard, so the
                # missing ones are reported by `armed_methods` instead.
                continue

            def make(orig, method_name):
                def wrapper(*args, **kwargs):
                    touch.record(method_name)
                    return orig(*args, **kwargs)
                wrapper.__name__ = getattr(orig, "__name__", method_name)
                wrapper.__loop_affinity__ = True
                return wrapper

            self._undo.append((obj, name, original))
            self._armed += 1
            setattr(obj, name, make(original, name))

    def disarm(self) -> None:
        for obj, name, original in reversed(self._undo):
            try:
                setattr(obj, name, original)
            except AttributeError:  # pragma: no cover - object already torn down
                pass
        self._undo.clear()

    # ---- reporting ----

    @property
    def armed_methods(self) -> int:
        """How many methods were successfully wrapped. Survives ``disarm``."""
        return self._armed

    @property
    def total_mutations(self) -> int:
        return sum(sum(t.by_thread.values()) for t in self.touches.values())

    def violations(self) -> list[str]:
        """One readable line per object mutated from more than one thread."""
        out = []
        for touch in sorted(self.touches.values(), key=lambda t: t.label):
            if touch.threads <= 1:
                continue
            detail = ", ".join(
                f"thread {ident} ({touch.example[ident]}, {n} calls)"
                for ident, n in sorted(touch.by_thread.items()))
            out.append(f"{touch.label} was mutated from {touch.threads} "
                       f"threads: {detail}")
        return out

    def assert_single_threaded(self) -> None:
        """🚨 The assertion the whole module exists for.

        Also fails when NOTHING was recorded: a guard that ran against an idle
        proxy reads exactly like a guard that found no violation, and this repo
        has a ledger entry for a green suite that ran nothing.
        """
        assert self.armed_methods, (
            "loop_affinity armed no methods — every name in STATE_METHODS has "
            "been renamed away, so this guard is watching nothing")
        assert self.total_mutations, (
            "loop_affinity recorded no mutations at all — the workload never "
            "reached the state it is guarding, so a pass here means nothing")
        problems = self.violations()
        assert not problems, (
            "🚨 CONCURRENCY INVARIANT VIOLATED — single-loop state was mutated "
            "from more than one thread. CLAUDE.md: no locks protect any of it.\n"
            + "\n".join(f"  - {p}" for p in problems))


#: The mutating surface of each single-loop object, by attribute path on
#: ``ProxyState``.
#:
#: 🚨 Kept as an explicit list rather than "every public method", because the
#: read-only ones (``endpoint_snapshot``, ``price``, ``spent_today``) are called
#: legitimately from off-loop dashboard threads and would produce a wall of
#: false positives that nobody would read twice.
STATE_METHODS: dict[str, tuple[str, ...]] = {
    # DRR + admission. The queues, the per-endpoint occupancy map and the
    # in-flight registry are all plain dicts.
    "scheduler": ("enqueue", "tick", "complete", "cancel", "_spill"),
    # The DRR budgets themselves — the state the drain persists on SIGTERM, and
    # the one whose corruption would be silent for the longest.
    "budget_mgr": ("charge", "replenish", "retroactive_adjust", "pick_agent",
                   "get_or_create", "remove", "prune_idle",
                   "set_total_capacity"),
    # The EWMA calibration behind every slot-second estimate.
    "cost_model": ("record_completion", "register_endpoint", "update_max_slots"),
    # Workstream D. Ordinary single-loop mutable state, and would race exactly
    # like the DRR budgets — it is newer, not safer.
    "spend": ("charge",),
    "prices": ("declare", "observe"),
    # Deterministic response cache.
    "cache": ("put",),
    # The learned latency distribution.
    "timeout_model": ("record", "prune"),
}


#: 🚨 A guard that watches nothing reads exactly like a guard that found
#: nothing. This repo has a ledger entry for that failure ("a green suite that
#: ran nothing"), so the count of successfully-wrapped methods is pinned:
#: rename one of the methods above and ``test_loop_affinity.py`` fails, rather
#: than the soak quietly stopping short of the state it exists to watch.
EXPECTED_ARMED_METHODS = sum(len(v) for v in STATE_METHODS.values())


def arm(state) -> LoopAffinity:
    """Arm every single-loop object on a live ``ProxyState``.

    Returns the recorder; call :meth:`LoopAffinity.assert_single_threaded` when
    the workload is done, and :meth:`LoopAffinity.disarm` to restore.
    """
    affinity = LoopAffinity()
    for attr, methods in STATE_METHODS.items():
        obj = getattr(state, attr, None)
        if obj is None:
            continue
        affinity.watch(obj, attr, methods)
    return affinity
