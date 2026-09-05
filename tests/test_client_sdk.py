"""``roadstead.client`` — the CLIENT half of a two-ended contract pin.

``tests/wire_contract.py`` transcribes ``docs/api.md`` for the SERVER side and
``tests/test_wire_contract.py`` reads the document back to check it. This is the
same arrangement for the SDK: ``roadstead/client/_wire.py`` is a hand
transcription, and the tests here read ``docs/api.md`` back and fail when it no
longer says what those literals claim.

🚨 **The point is that the SDK does not import the server.** It would be trivial
for ``_wire.py`` to do ``from roadstead.enriched import ENRICHMENT_HEADERS`` and
be permanently correct — and permanently useless as a check, because a client
that reads the server's constants agrees with the server by construction and can
never catch a drift. So the two are checked against the DOCUMENT, independently,
and this file also proves the SDK imports with no server module loaded at all.

The LIVE half — the SDK driven against the real ASGI proxy — is
``tests/e2e/test_client_sdk_live.py``, where the ``proxy`` fixture lives.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from roadstead.client import RoadsteadError, enrichment_from
from roadstead.client import _wire as W

_ROOT = Path(__file__).resolve().parents[1]
API_DOC = _ROOT / "docs" / "api.md"


@pytest.fixture(scope="module")
def doc() -> str:
    assert API_DOC.exists(), f"the contract is missing at {API_DOC}"
    return API_DOC.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The transcription, checked against the document
# ---------------------------------------------------------------------------

def test_every_error_code_the_sdk_knows_is_published(doc):
    assert W.ERROR_CODES, "an empty code set would pass every assertion"
    for code in W.ERROR_CODES:
        assert f"`{code}`" in doc, (
            f"docs/api.md §2.1 no longer publishes the error code {code!r}")


def test_the_sdk_knows_every_code_the_document_publishes(doc):
    """The other direction, and the one that actually bites: a code added to the
    contract that the SDK has never heard of classifies as non-deferrable by
    default, so a caller stops retrying something it should retry."""
    # §2.1's ENUMERATION — the one `·`-separated paragraph — rather than every
    # backticked token in the section. The looser read needed a filter to drop
    # the prose (`type: invalid_request_error`, and since 2026-09-05 a table
    # naming `_schema_retry` and `mislabels_truncated_tool_calls`), and the only
    # filter that kept `backpressure` and `draining` was "underscore OR already
    # in ERROR_CODES" — which quietly excused the SDK from any code that has no
    # underscore and that it has not heard of. Reading the list itself needs no
    # filter and makes no exceptions.
    section = doc.split("### 2.1 Codes", 1)[1].split("### 2.2", 1)[0]
    listing = next(para for para in section.split("\n\n") if "·" in para)
    published = set(re.findall(r"`([a-z_]+)`", listing))
    missing = published - W.ERROR_CODES
    assert not missing, (
        f"docs/api.md §2.1 publishes {sorted(missing)}, which "
        f"roadstead/client/_wire.py has never heard of — an unknown code "
        f"classifies as non-deferrable, so a caller would stop retrying")


def test_the_deferrable_set_is_a_real_subset(doc):
    assert W.DEFERRABLE_CODES < W.ERROR_CODES
    # A classifier that says yes to everything, or no to everything, makes every
    # caller of it pass.
    assert "backpressure" in W.DEFERRABLE_CODES
    assert "invalid_grammar" not in W.DEFERRABLE_CODES
    assert "invalid_api_key" not in W.DEFERRABLE_CODES
    # 🚨 The pair the fifteen-vs-seventeen defect turned on (2026-09-05). A
    # truncation is retry work — §2.1's row and §2.2's `truncated structured
    # output` row are the same fault — while a schema the model failed twice
    # with the error fed back needs the caller to change the schema. Pinned
    # because both were non-deferrable-by-omission before, so restoring that
    # state is a one-line edit no other assertion would notice.
    assert "toolcall_truncated" in W.DEFERRABLE_CODES
    assert "schema_invalid" not in W.DEFERRABLE_CODES


def test_the_context_overflow_marker_is_verbatim(doc):
    assert W.CONTEXT_OVERFLOW_MARKER in doc


def test_the_legacy_markers_are_still_published(doc):
    """§2.2 — kept because this SDK may be pointed at an older proxy, and
    dropping the fallback would turn a deferrable backpressure error into a hard
    failure on exactly the deployment least able to absorb one."""
    for marker in W.DEFERRABLE_MARKERS:
        assert f"`{marker}`" in doc


def test_the_routes_are_published(doc):
    """🚨 Matched as a whole TOKEN, not as a substring.

    `assert route in doc` was the version, and renaming `/rs/v1/chat` to
    `/rs/v1/chats` throughout the document left it green — the old name is a
    prefix of the new one, so the substring is still there. A pin that cannot
    see a route renamed by extension is not pinning the route.

    The delimiter is a negative lookahead rather than the backtick the
    error-code and legacy-marker pins use, because a route is written both bare
    (`` `/rs/v1/chat` ``) and with its verb (`` `POST /rs/v1/chat` ``), so there
    is no leading backtick to match against. What has to be true is that the
    name ENDS where it ends.
    """
    for route in (W.ROUTE_MODELS, W.ROUTE_PLAN, W.ROUTE_CHAT):
        assert re.search(re.escape(route) + r"(?![A-Za-z0-9_/-])", doc), (
            f"docs/api.md no longer publishes {route}")


def test_the_enrichment_headers_are_published(doc):
    """Delimited for the reason above: `X-Roadstead-Endpoint` is a prefix of
    any longer name somebody might rename it to."""
    for header in (W.HEADER_REQUEST_ID, W.HEADER_ENDPOINT,
                   W.HEADER_DEADLINE_S, W.HEADER_DEADLINE_SOURCE):
        assert f"`{header}`" in doc, f"docs/api.md no longer publishes {header}"


def test_the_client_keepalive_still_satisfies_the_ordering_invariant(doc):
    """§1.3 — the client must retire idle sockets FIRST, with real margin. This
    is the side a client controls, and the SDK sets it rather than leaving the
    invariant to whoever configures the pool."""
    from tests.wire_contract import KEEPALIVE_MIN_MARGIN_S
    from roadstead.__main__ import PROXY_SERVER_KEEPALIVE_S

    assert W.CLIENT_KEEPALIVE_EXPIRY_S + KEEPALIVE_MIN_MARGIN_S <= PROXY_SERVER_KEEPALIVE_S


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------

def test_the_sdk_imports_nothing_from_the_server(doc):
    """🚨 By AST, over every module in the package.

    Two reasons, and the second is the one that would be lost silently: a
    consumer installing the client should not be installing Starlette, uvicorn,
    PyYAML and jsonschema to send an HTTP request; and a client that reads the
    server's own constants agrees with it by construction, so the two-ended pin
    above would stop being a pin at all.
    """
    allowed_roadstead = {"roadstead.client"}
    offenders: list[str] = []
    for path in sorted((_ROOT / "roadstead" / "client").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:            # relative — stays inside the package
                    continue
                if mod.split(".")[0] == "roadstead" and mod not in allowed_roadstead:
                    offenders.append(f"{path.name}: from {mod} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "roadstead":
                        offenders.append(f"{path.name}: import {alias.name}")
    assert not offenders, (
        "roadstead.client imports the server: " + "; ".join(offenders))


def test_it_imports_with_only_httpx_and_the_stdlib_available():
    """Run in a subprocess that makes every other Roadstead dependency
    un-importable. A plain in-process import proves nothing here — starlette and
    PyYAML are installed in this environment, so an accidental dependency would
    pass silently and only fail in the consumer's."""
    script = (
        "import sys\n"
        "class _Block:\n"
        "    BANNED = ('starlette', 'uvicorn', 'yaml', 'jsonschema', 'json_repair')\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name.split('.')[0] in self.BANNED else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, _Block())\n"
        "from roadstead.client import AsyncRoadsteadClient, RoadsteadError\n"
        "assert RoadsteadError('x', code='backpressure').deferrable\n"
        "assert not RoadsteadError('x', code='invalid_grammar').deferrable\n"
        "print('CLIENT-OK')\n"
    )
    with tempfile.TemporaryDirectory() as elsewhere:
        proc = subprocess.run([sys.executable, "-c", script], cwd=elsewhere,
                              capture_output=True, text=True, timeout=120,
                              env={"PYTHONPATH": str(_ROOT), "PATH": "/usr/bin:/bin"})
    assert "CLIENT-OK" in proc.stdout, (
        f"the SDK could not import without the server's dependencies\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,message,deferrable", [
    ("backpressure", "tier3 background queue saturated", True),
    ("circuit_open", "backend tier3 unavailable (circuit open)", True),
    ("draining", "proxy draining for shutdown", True),
    ("proxy_timeout", "proxy timeout after 180s", True),
    ("invalid_grammar", "unparseable: no rule definitions", False),
    ("invalid_api_key", "no such key", False),
    ("unknown_endpoint", "unknown model 'typo'", False),
    ("context_overflow", "exceeds the available context size", False),
    # No code at all — an older proxy, or a body we could not parse. The prose
    # fallback §2.2 asks a shipped client to keep.
    ("", "backend tier3 unavailable (circuit open)", True),
    ("", "proxy draining for shutdown — backpressure", True),
    ("", "unknown endpoint 'typo' — no such model/role", False),
])
def test_deferrability_is_classified_on_the_code_first(code, message, deferrable):
    assert RoadsteadError(message, code=code).deferrable is deferrable


def test_context_overflow_is_recognised_both_ways():
    """The same condition arrives two ways: Roadstead's own pre-admission gate
    (a typed code) and a backend's own overflow relayed through as a
    `backend_error` carrying the verbatim marker."""
    assert RoadsteadError("x", code="context_overflow").context_overflow
    assert RoadsteadError(
        f"backend said: prompt {W.CONTEXT_OVERFLOW_MARKER} (8192)",
        code="backend_error").context_overflow
    assert not RoadsteadError("x", code="backend_error").context_overflow


def test_enrichment_from_degrades_on_a_response_that_is_not_ours():
    """Pointing the same code at a stock OpenAI server must degrade, not raise —
    otherwise the migration helper is unusable during the migration."""
    empty = enrichment_from({"content-type": "application/json"})
    assert empty.present is False and empty.deadline_s == 0.0
    ours = enrichment_from({
        W.HEADER_REQUEST_ID: "req_abc", W.HEADER_ENDPOINT: "tier3",
        W.HEADER_DEADLINE_S: "180.000", W.HEADER_DEADLINE_SOURCE: "computed"})
    assert ours.present and ours.endpoint == "tier3" and ours.deadline_s == 180.0
    # A malformed header must not raise either — it is enrichment, not payload.
    assert enrichment_from({W.HEADER_DEADLINE_S: "soon"}).deadline_s == 0.0


# ---------------------------------------------------------------------------
# A permanent backend failure is not worth retrying
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,deferrable,why", [
    (400, False, "the backend rejected the request and will reject it again"),
    (404, False, "the model does not exist and will not start existing"),
    (422, False, "unprocessable is a statement about the request"),
    (408, True,  "the backend timed out — later is a different answer"),
    (429, True,  "rate limited — later is exactly the answer"),
    (500, True,  "a server fault may not recur"),
    (502, True,  "a bad gateway is transient by nature"),
    (None, True, "an older proxy reports no status; behave as before"),
])
def test_a_permanent_backend_failure_is_not_deferrable(status, deferrable, why):
    """🚨 The proxy already knew, and kept it to itself.

    `correction.is_transient_backend_error` says in its own docstring that "a
    real 4xx / other-5xx is deterministic" and declines to retry it. The caller
    was then handed `backend_error`, which every client classifies as
    retryable — so the proxy gave up on a permanent failure and simultaneously
    advised retrying it. A caller following that advice loops forever against a
    misconfigured endpoint while the proxy watches.

    Found live on 2026-09-01: an OpenRouter endpoint pinned to a model that does
    not exist returned `code='backend_error', deferrable=True` for a permanent
    400.

    🚨 Unknown status stays deferrable, so this narrows behaviour only where the
    proxy has said enough to narrow it — an older proxy is unaffected.
    """
    body = {"code": "backend_error", "error": "backend error"}
    if status is not None:
        body["backend_status"] = status
    err = RoadsteadError("backend error", code="backend_error", body=body)
    assert err.deferrable is deferrable, why
    assert err.backend_status == status


def test_the_other_deferrable_codes_are_untouched_by_it():
    """Narrowing `backend_error` must not narrow backpressure — the one whose
    whole purpose is to say *try again shortly*."""
    for code in ("backpressure", "circuit_open", "draining", "proxy_timeout"):
        err = RoadsteadError("x", code=code,
                             body={"code": code, "backend_status": 400})
        assert err.deferrable is True, code
