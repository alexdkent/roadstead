"""Tiny Prometheus text-exposition EMITTER for the proxy's ``GET /metrics``.

Vendored from the origin monorepo's framework emitter (2026-08-31) so this
package depends on nothing outside itself — see ``hooks.py`` for the rationale.
The two copies are independent by design: this one serves the proxy, the one
left behind serves that application. Nothing keeps them in step, and nothing
needs to.

The ``prometheus_client`` dependency is deliberately not taken. The write side
is small enough to hand-roll — this module is the whole of it — and the read
side was already hand-rolled where the origin fleet parses vLLM telemetry.

Scope note: this emits the subset of Prometheus text exposition the fleet needs
— ``gauge`` and ``counter`` samples with ``name{labels} value`` lines and the
``# HELP``/``# TYPE`` header comments. No histograms/summaries (expose the
derived quantiles as plain gauges instead), no timestamps, no exemplars.

Label discipline (the one way TSDBs get expensive): keep label VALUES bounded —
``agent``/``endpoint``/``unit``/``host``/``device``, never ``request_id`` /
``ip_hash`` / ``session_id`` / free text. Aggregate before exposing.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Prometheus metric-name and label-name grammar (same as the parser accepts).
_NAME_RE_OK = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_RE_OK = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True)
class Metric:
    """One current value to expose. ``type`` is ``"gauge"`` or ``"counter"``.

    ``help`` is optional one-line text rendered as ``# HELP``. ``labels`` values
    must be LOW cardinality (see module docstring)."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    type: str = "gauge"
    help: str | None = None


# Back-compat / symmetry alias — the parser calls its row ``Sample``.
Sample = Metric


def _fmt_value(v: float) -> str:
    """Render a value the way Prometheus expects: ``+Inf``/``-Inf``/``NaN`` for
    the specials, plain integers without a trailing ``.0``, floats otherwise."""
    if isinstance(v, bool):  # bool is an int subclass — render 0/1, not True/False
        return "1" if v else "0"
    f = float(v)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "+Inf" if f > 0 else "-Inf"
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    return repr(f)


def _escape_label_value(s: str) -> str:
    # Per the text format: backslash, double-quote and newline are escaped.
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = []
    for k, v in labels.items():
        if not _LABEL_RE_OK.match(k):
            continue  # skip a malformed label name rather than emit invalid text
        parts.append(f'{k}="{_escape_label_value(str(v))}"')
    return "{" + ",".join(parts) + "}" if parts else ""


def render_prometheus(metrics: list[Metric]) -> str:
    """Render metrics as Prometheus text exposition.

    ``# TYPE`` (and ``# HELP`` when provided) are emitted once per metric name,
    in first-seen order; sample lines follow. Metrics with an invalid name are
    skipped (an exposition endpoint must never raise on a caller quirk). Always
    ends with a trailing newline (required by the format)."""
    seen_headers: set[str] = set()
    out: list[str] = []
    for m in metrics:
        if not _NAME_RE_OK.match(m.name):
            continue
        if m.name not in seen_headers:
            seen_headers.add(m.name)
            if m.help:
                out.append(f"# HELP {m.name} {m.help.splitlines()[0]}")
            mtype = m.type if m.type in ("gauge", "counter") else "untyped"
            out.append(f"# TYPE {m.name} {mtype}")
        out.append(f"{m.name}{_render_labels(m.labels)} {_fmt_value(m.value)}")
    return "\n".join(out) + "\n" if out else ""
