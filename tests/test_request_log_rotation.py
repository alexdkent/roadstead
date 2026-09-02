"""The request log is bounded, and it cannot fail a request.

🚨 **What this exists for.** `RequestLogger` opened its file in append mode and
never rotated, while its own docstring called it the authoritative per-request
record. At a measured ~27,700 requests/day that is 10-20 MB/day, forever.

It was very nearly handed to the infrastructure as a logrotate job, which would
have been the wrong boundary twice over: the file belongs to the application, and
the handle is held open in append mode, so a rename-and-create rotation leaves
the proxy writing to the unlinked inode — the new file stays empty, the disk
still fills, and everything looks configured. `copytruncate` is the only external
rotation that works on it, and requiring an operator to know that is a trap.

The second half is the hot path. `log()` is called from `lifecycle`'s completion
path on the scheduler loop, so a full disk raising out of `write` becomes 500s on
live traffic. It fails open and says so once — the same doctrine `spend.py`
follows, and for the same reason: an accounting surface must not be able to take
the proxy down.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from roadstead.observability import RequestLogger, RequestLogRecord


def _record(status: str = "ok") -> RequestLogRecord:
    return RequestLogRecord(
        ts=0.0, call_site="test",
        request_id="r", agent_id="a", endpoint="tier1", priority="P1_TURN_SUPPORT",
        band="interactive", payload_type="chat", input_tokens=1, output_tokens=1,
        estimated_cost_ss=0.0, actual_cost_ss=0.0, queue_wait_ms=0.0,
        backend_latency_ms=0.0, total_latency_ms=0.0, occupancy_at_dispatch=0,
        status=status,
    )


def test_the_log_rotates_and_stays_bounded(tmp_path):
    """The bound is the whole point: without it this file grows forever."""
    path = tmp_path / "req.jsonl"
    lg = RequestLogger(str(path), max_bytes=2_000, backups=2)
    for _ in range(400):
        lg.log(_record())
    lg.close()

    live = path.stat().st_size
    assert live < 2_000, f"the live file blew its own threshold: {live}"

    rolled = sorted(tmp_path.glob("req.jsonl.*"))
    assert rolled, "nothing rotated — the file grew unbounded, which is the bug"
    assert len(rolled) <= 2, (
        f"kept {len(rolled)} backups with backups=2 — the bound is on the SET, "
        f"not on one file, or the disk fills just as surely")

    total = sum(p.stat().st_size for p in [path, *rolled])
    assert total <= 2_000 * 3 * 1.1, f"the file set is not bounded: {total}"


def test_rotation_keeps_the_RECENT_end_in_order(tmp_path):
    """🚨 `.1` must be NEWER than `.2`, and the live file newest of all.

    Bounding the size is only half the job — dropping the wrong end bounds it
    just as well while throwing away exactly the records anyone asks for after
    an incident. The first version of this test asserted only "the live file has
    the last record" and "there is no `.3`", both of which stay true if rotation
    is deleted outright, so it could not see the thing it was named for. Records
    are numbered here so the ordering is actually checkable.
    """
    path = tmp_path / "req.jsonl"
    # 🚨 THREE backups, not two. At backups=2 the shift loop runs exactly
    # once whichever direction it goes, so an inverted shift is a no-op and this
    # test passed against the very mutation it exists to catch. Direction only
    # becomes observable with two or more iterations.
    lg = RequestLogger(str(path), max_bytes=900, backups=3)
    for i in range(400):
        rec = _record()
        rec.request_id = f"seq{i:04d}"
        lg.log(rec)
    lg.close()

    def seqs(p: Path) -> list[str]:
        # Tolerant of absence on purpose: an inverted shift DESTROYS a backup,
        # and letting `read_text` raise FileNotFoundError here would report the
        # bug as a stack trace from the test helper instead of as the assertion
        # written for it. A raw error is half a guard.
        if not p.exists():
            return []
        return re.findall(r"seq(\d{4})", p.read_text())

    live, one, two, three = (seqs(path), seqs(tmp_path / "req.jsonl.1"),
                             seqs(tmp_path / "req.jsonl.2"), seqs(tmp_path / "req.jsonl.3"))
    assert one and two and three, (
        f"expected three ordered backups; got sizes "
        f"{len(one)}/{len(two)}/{len(three)} for .1/.2/.3 — a missing or empty "
        f"backup means the shift clobbered one instead of moving it")
    # 🚨 `live` may legitimately be EMPTY: rotation runs AFTER the write that
    # crosses the threshold, so whether the newest file has content depends on
    # where the run stopped. Asserting it non-empty made this test fail on a
    # correct implementation — a phase of the cycle, not a defect.
    if live:
        assert min(live) > max(one), (
            f"the live file is not newer than .1 (live starts {min(live)}, "
            f".1 ends {max(one)}) — rotation is keeping the wrong end")
    assert min(one) > max(two), (
        f".1 is not newer than .2 (.1 starts {min(one)}, .2 ends {max(two)}) — "
        f"the shift is inverted, so the oldest records are the ones surviving")
    assert min(two) > max(three), (
        f".2 is not newer than .3 (.2 starts {min(two)}, .3 ends {max(three)})")
    assert not (tmp_path / "req.jsonl.4").exists(), "backups=3 kept a fourth"


def test_backups_zero_truncates_rather_than_growing_forever(tmp_path):
    """🚨 `backups=0` must mean a bound of zero FILES, not "no rotation".

    The other reading is already spelled `max_bytes=0`, and a config value that
    silently means "unbounded" is how this defect existed in the first place.
    """
    path = tmp_path / "req.jsonl"
    lg = RequestLogger(str(path), max_bytes=500, backups=0)
    for _ in range(200):
        lg.log(_record())
    lg.close()

    assert path.stat().st_size < 500
    assert not list(tmp_path.glob("req.jsonl.*")), "backups=0 kept a backup"


def test_max_bytes_zero_opts_out(tmp_path):
    """The documented escape hatch for a deployment driving rotation itself."""
    path = tmp_path / "req.jsonl"
    lg = RequestLogger(str(path), max_bytes=0, backups=3)
    for _ in range(200):
        lg.log(_record())
    lg.close()
    assert not list(tmp_path.glob("req.jsonl.*"))
    assert path.stat().st_size > 0


def test_an_unwritable_log_does_not_raise_into_the_request_path(tmp_path, caplog):
    """🚨 The hot-path half. `log()` runs on the scheduler loop inside
    `lifecycle`'s completion path — an OSError here is a 500 on live traffic for
    a caller whose request actually succeeded.

    Driven by closing the handle underneath it, which is what a vanished mount
    looks like from inside: the object still thinks it has a file.
    """
    path = tmp_path / "req.jsonl"
    lg = RequestLogger(str(path))
    lg._file.close()          # the mount went away mid-flight

    with caplog.at_level("ERROR"):
        lg.log(_record())     # must not raise
        lg.log(_record())     # …and must not spam

    errors = [r for r in caplog.records
              if "request log is unwritable" in r.getMessage()]
    assert len(errors) == 1, (
        f"expected exactly one disclosure, got {len(errors)} — a per-request "
        f"error on a full disk is a second way to fill it")


def test_reopening_counts_what_is_already_on_disk(tmp_path):
    """🚨 The counter starts from the file's real size, not zero.

    Appending to an existing file with an in-memory counter reset to 0 means a
    restart-heavy deployment never reaches the threshold and the log grows
    forever anyway — the original bug, wearing rotation.
    """
    path = tmp_path / "req.jsonl"
    path.write_text("x" * 5_000 + "\n")

    lg = RequestLogger(str(path), max_bytes=1_000, backups=1)
    assert lg._size >= 5_000, "the pre-existing bytes were not counted"
    lg.log(_record())
    lg.close()
    assert (tmp_path / "req.jsonl.1").exists(), (
        "an already-oversized file did not roll on the first write")


def test_the_log_lives_under_the_data_dir(monkeypatch, tmp_path):
    """🚨 One mount holds everything the application owns.

    The deployment contract asks the operator for ONE persistent path. Until
    2026-09-02 the log was rooted in `$ROADSTEAD_HOT_ROOT/logs` instead, so
    setting `ROADSTEAD_DATA_DIR` moved the queue DB onto the mount and left the
    request log on the container's ephemeral layer — a half-kept promise.
    """
    from roadstead.__main__ import build_app

    monkeypatch.setenv("ROADSTEAD_DATA_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("ROADSTEAD_LOG_DIR", raising=False)
    monkeypatch.delenv("ROADSTEAD_REQUEST_LOG", raising=False)
    monkeypatch.delenv("LLM_PROXY_REQUEST_LOG", raising=False)
    monkeypatch.setenv("ROADSTEAD_MODELS_YAML",
                       str(Path(__file__).resolve().parent.parent
                           / "roadstead" / "models.yaml"))

    build_app()
    log = tmp_path / "state" / "logs" / "llmproxy_requests.jsonl"
    assert log.parent.is_dir(), (
        f"the request log is not under ROADSTEAD_DATA_DIR: expected {log.parent}. "
        f"An operator who mounted one persistent path just lost the log.")
