"""No-store end to end — over the real ASGI doors and a real fake-backend socket.

``X-Roadstead-No-Store: content`` asks that a request's content reach no store
and no log line (``docs/api.md`` §1.11). The operator's grant decides whether
the ask is honoured; these tests pin the whole contract from the wire:

  * **The default stores everything** — and the instrument below can SEE it.
    A "the canary is absent" assertion is only evidence if the same collector
    finds the canary when nothing was withheld (`test_the_default_*`).
  * **Granted + header ⇒ the canary is on NO surface**, on every door, streaming
    and not, while the metadata (identity, endpoint, token counts, status,
    sizes, message count, roles, tool names, the honoured flag) IS recorded.
  * **Ungranted + header ⇒ stored in full, loudly**: counter, WARNING naming
    the caller, a durable `refused:<reason>` on the completion row, and the
    response header. Silent ignoring is the failure this feature must not have.
  * **Nothing a caller can say about itself grants it**: an address identity
    (the shared-LAN case), a header-declared or body-declared `agent_id`, a
    shared label that also has a key.

The surfaces collector is deliberately broad — every table in `queue.db`, the
raw bytes of the db and its WAL, every log record at DEBUG, the request log,
every SSE frame, the in-memory caches, and the response of every read route. A
surface added later that holds content shows up here only if it is added to the
collector, so `test_the_collector_covers_*` pins the list against the places
this change found (docs/api.md §1.11).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from roadstead.config import load_agent_configs
from roadstead.observability import RequestLogger

CANARY = "zxq-canary-7f3a9c-do-not-store"

_AGENTS_YAML = """\
granted-caller:
  content_no_store: allowed
  approved_by: operator
  approved_on: 2026-01-31
shared-label:
  content_no_store: allowed
  approved_by: operator
  approved_on: 2026-01-31
half-approved:
  content_no_store: allowed
  approved_on: 2026-01-31
plain-caller:
  weight: 1.0
"""

KEYS = {
    "granted": ("k-granted", "granted-caller"),
    "plain": ("k-plain", "plain-caller"),
    "shared": ("k-shared", "shared-label"),
    "half": ("k-half", "half-approved"),
}


@pytest.fixture(autouse=True)
def _legacy_door_open(monkeypatch):
    """`/v1/submit` is only routed when the flag is set before `build_app`."""
    monkeypatch.setenv("ROADSTEAD_LEGACY_SUBMIT", "1")


class Surfaces:
    """Everything a content-bearing byte could be sitting in, as one string."""

    READ_ROUTES = (
        "/v1/status", "/v1/recent?limit=200", "/v1/inflight", "/v1/metrics",
        "/metrics", "/v1/history", "/v1/timeouts", "/v1/fleet/top-callers",
        "/v1/fleet/cache-stats", "/rs/v1/admin/callers", "/rs/v1/admin/audit",
        "/rs/v1/admin/config",
    )

    def __init__(self, proxy, tmp_path: Path, caplog) -> None:
        self.proxy = proxy
        self.svc = proxy.svc
        self.caplog = caplog
        self.sse: list[str] = []
        self.request_log = tmp_path / "requests.jsonl"
        self.svc._state.request_logger = RequestLogger(str(self.request_log))
        real_publish = self.svc._state.sse.publish

        def spy(event, data):
            self.sse.append(json.dumps([event, data], default=str))
            return real_publish(event, data)

        self.svc._state.sse.publish = spy

    # -- the db ------------------------------------------------------------

    def _db_path(self) -> str:
        return self.svc._queue_db._db_path

    def db_rows(self) -> str:
        self.svc._queue_db.flush()
        conn = sqlite3.connect(self._db_path())
        try:
            out = []
            for (name,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"):
                out.append(f"## {name}")
                out.extend(repr(r) for r in conn.execute(f'SELECT * FROM "{name}"'))
            return "\n".join(out)
        finally:
            conn.close()

    def db_bytes(self) -> str:
        """The raw file and its WAL — a row deleted after the fact is still in
        the free pages, which is exactly what a row-level read cannot see."""
        self.svc._queue_db.flush()
        parts = []
        for suffix in ("", "-wal"):
            p = Path(self._db_path() + suffix)
            if p.exists():
                parts.append(p.read_bytes().decode("latin-1"))
        return "\n".join(parts)

    def completion(self, **where) -> dict:
        self.svc._queue_db.flush()
        conn = sqlite3.connect(self._db_path())
        conn.row_factory = sqlite3.Row
        try:
            clause = " AND ".join(f"{k}=?" for k in where) or "1=1"
            rows = conn.execute(
                f"SELECT * FROM proxy_completions WHERE {clause} "
                f"ORDER BY completed_at DESC", tuple(where.values())).fetchall()
            assert rows, f"no completion row for {where}"
            return dict(rows[0])
        finally:
            conn.close()

    # -- memory ------------------------------------------------------------

    def memory(self) -> str:
        st = self.svc._state
        parts = [repr(st.cache._cache), repr(st.reasoning_replay._cache),
                 repr(st.prefix_keepalive._by_endpoint)]
        return "\n".join(parts)

    # -- the whole thing ---------------------------------------------------

    async def text(self) -> str:
        parts = [self.db_rows(), self.db_bytes(), self.caplog.text,
                 "\n".join(self.sse), self.memory()]
        if self.request_log.exists():
            parts.append(self.request_log.read_text())
        for route in self.READ_ROUTES:
            r = await self.proxy.client.get(route, headers=self.proxy.admin)
            parts.append(f"## GET {route} {r.status_code}\n{r.text}")
        return "\n".join(parts)


@pytest_asyncio.fixture
async def ns(proxy, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    svc = proxy.svc
    keys = svc._state.identity.keys
    for secret, agent in KEYS.values():
        keys.register(secret=secret, agent_id=agent, key_id=secret)
    cfg = tmp_path / "agents.yaml"
    cfg.write_text(_AGENTS_YAML)
    svc._state.config.agents.update(load_agent_configs(cfg))
    # `shared-label` is ALSO an address identity: anything on this subnet wears
    # it, and a key minted for it must still not inherit a per-caller privilege.
    svc._state.acl.register("203.0.113.0/24", "shared-label")
    return Surfaces(proxy, tmp_path, caplog)


def _msgs(content: str = CANARY) -> list[dict]:
    return [{"role": "system", "content": "be terse"},
            {"role": "user", "content": content}]


_TOOLS = [{"type": "function", "function": {"name": "lookup_account",
                                            "parameters": {"type": "object"}}}]

DOORS = ("openai", "openai-stream", "enriched", "enriched-stream",
         "legacy", "legacy-stream", "embeddings")


async def send(proxy, door: str, who: str | None, *, no_store: bool = True,
               header_value: str = "content", extra_headers: dict | None = None):
    """One request through ``door``. Returns ``(status, headers, text)``."""
    headers = dict(extra_headers or {})
    if who:
        headers["X-API-Key"] = KEYS[who][0]
    if no_store:
        headers["X-Roadstead-No-Store"] = header_value
    c = proxy.client
    stream = door.endswith("-stream")
    if door.startswith("openai"):
        body = {"model": "chat", "messages": _msgs(), "max_tokens": 16,
                "stream": stream, "tools": _TOOLS}
        path = "/v1/chat/completions"
    elif door.startswith("enriched"):
        body = {"intent": "chat",
                "payload": {"messages": _msgs(), "max_tokens": 16,
                            "stream": stream, "tools": _TOOLS}}
        path = "/rs/v1/chat"
    elif door.startswith("legacy"):
        body = {"agent_id": "ignored", "endpoint": "chat",
                "priority": "P1_TURN_SUPPORT", "call_site": "e2e.legacy",
                "payload_type": "chat_completion", "timeout_s": 20.0,
                "payload": {"messages": _msgs(), "max_tokens": 16,
                            "stream": stream, "tools": _TOOLS}}
        path = "/v1/submit"
    elif door == "embeddings":
        body = {"model": "embed", "input": [CANARY]}
        path = "/v1/embeddings"
    else:  # pragma: no cover
        raise AssertionError(door)
    if stream:
        async with c.stream("POST", path, json=body, headers=headers) as r:
            text = "".join([line async for line in r.aiter_lines()])
            return r.status_code, r.headers, text
    r = await c.post(path, json=body, headers=headers)
    return r.status_code, r.headers, r.text


def _row_for(surf: Surfaces, agent: str) -> dict:
    return surf.completion(agent_id=agent)


# --------------------------------------------------------------------------- #
# The instrument, and the default
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("door", DOORS)
async def test_the_default_stores_everything_and_the_instrument_sees_it(ns, door):
    """No header, from a caller that HAS a grant: stored in full. A grant is a
    permission to ask, never a mode — and this is the positive control that
    makes every "absent" assertion below mean something."""
    status, _, _ = await send(ns.proxy, door, "granted", no_store=False)
    assert status == 200
    text = await ns.text()
    assert CANARY in text
    assert CANARY in ns.db_bytes()
    if door != "embeddings":
        row = _row_for(ns, "granted-caller")
        assert row["no_store"] is None and row["content_meta"] is None
        assert CANARY in row["payload_json"]
    # Nothing was asked, so nothing is counted.
    st = ns.svc._state
    assert st.no_store_honoured == 0 and st.no_store_refused == 0


def test_the_collector_covers_the_surfaces_the_audit_found():
    """The audit's list, as code. Adding a content-bearing store without adding
    it to the collector would turn every pin here into a statement about a
    narrower system than the one that ships."""
    names = {n for n in dir(Surfaces) if not n.startswith("__")}
    assert {"db_rows", "db_bytes", "memory", "text"} <= names
    assert "/v1/recent?limit=200" in Surfaces.READ_ROUTES
    assert "/metrics" in Surfaces.READ_ROUTES


# --------------------------------------------------------------------------- #
# Granted + header: nothing anywhere, metadata kept
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("door", DOORS)
async def test_granted_plus_header_leaves_the_canary_on_no_surface(ns, door):
    status, headers, _ = await send(ns.proxy, door, "granted")
    assert status == 200
    if door.startswith("openai"):
        assert headers["x-roadstead-no-store"] == "honoured"
    text = await ns.text()
    assert CANARY not in text, _context(text)
    assert CANARY not in ns.db_bytes()

    st = ns.svc._state
    assert st.no_store_honoured == 1 and st.no_store_refused == 0
    assert st.no_store_by_agent == {"granted-caller": {"honoured": 1}}


@pytest.mark.parametrize("door", ["openai", "openai-stream", "enriched",
                                  "enriched-stream", "legacy", "legacy-stream"])
async def test_the_metadata_is_still_recorded(ns, door):
    """Everything EXCEPT content: who, where, how long, how big, what shape."""
    status, _, _ = await send(ns.proxy, door, "granted")
    assert status == 200
    row = _row_for(ns, "granted-caller")
    assert row["payload_json"] is None and row["response_json"] is None
    assert row["no_store"] == "honoured"
    assert row["agent_id"] == "granted-caller"
    assert row["endpoint"]
    assert row["status"] == "ok"
    assert row["finish_reason"] == "stop"
    assert row["input_tokens"] > 0 and row["output_tokens"] > 0
    assert row["duration_s"] is not None and row["queue_wait_ms"] is not None
    meta = json.loads(row["content_meta"])
    assert meta["message_count"] == 2
    assert meta["roles"] == {"system": 1, "user": 1}
    assert meta["tools_declared"] == ["lookup_account"]
    assert meta["payload_bytes"] > len(CANARY)
    if not door.endswith("-stream"):
        assert meta["response_bytes"] > 0 and meta["choice_count"] == 1
    # The shape is described; the content is not.
    assert CANARY not in row["content_meta"]


async def test_an_honoured_embedding_records_its_shape_not_its_text(ns):
    status, _, _ = await send(ns.proxy, "embeddings", "granted")
    assert status == 200
    row = _row_for(ns, "granted-caller")
    assert row["no_store"] == "honoured"
    assert row["payload_json"] is None and row["response_json"] is None
    meta = json.loads(row["content_meta"])
    assert meta["input_count"] == 1 and meta["payload_bytes"] > 0


async def test_the_restart_wal_gets_no_row_for_a_no_store_request(ns):
    """`proxy_queue.payload_json` is the whole payload, kept so a queued request
    survives a crash. A no-store request never gets a row — checked while the
    request is IN FLIGHT, which is when a WAL row would exist."""
    proxy = ns.proxy
    proxy.controller.set_fault("timeout", 1.0)   # holds the backend ~1s
    import asyncio
    task = asyncio.create_task(send(proxy, "openai", "granted"))
    await asyncio.sleep(0.3)
    ns.svc._queue_db.flush()
    conn = sqlite3.connect(ns._db_path())
    try:
        assert conn.execute("SELECT COUNT(*) FROM proxy_queue").fetchone()[0] == 0
    finally:
        conn.close()
    assert CANARY not in ns.db_bytes()
    status, _, _ = await task
    assert status == 200

    # …and the control: an ordinary request IS in the WAL while in flight.
    task = asyncio.create_task(send(proxy, "openai", "plain", no_store=False))
    await asyncio.sleep(0.3)
    assert CANARY in ns.db_bytes()
    await task


async def test_a_granted_caller_response_is_unchanged(ns):
    """Withholding is a storage property. The caller still gets its answer."""
    status, _, text = await send(ns.proxy, "openai", "granted")
    assert status == 200 and CANARY in text


async def test_no_store_never_spills_and_never_caches(ns):
    st = ns.svc._state
    body = {"model": "chat", "messages": _msgs(), "max_tokens": 16,
            "temperature": 0}
    h = {"X-API-Key": KEYS["granted"][0], "X-Roadstead-No-Store": "content"}
    r1 = await ns.proxy.client.post("/v1/chat/completions", json=body, headers=h)
    r2 = await ns.proxy.client.post("/v1/chat/completions", json=body, headers=h)
    assert r1.status_code == r2.status_code == 200
    assert st.cache.size == 0, "a no-store response sat in the response cache"
    # The control: the same deterministic request WITHOUT the header is cached.
    h2 = {"X-API-Key": KEYS["plain"][0]}
    await ns.proxy.client.post("/v1/chat/completions", json=body, headers=h2)
    assert st.cache.size == 1


# --------------------------------------------------------------------------- #
# Ungranted + header: full logging, loudly
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("door", ["openai", "openai-stream", "enriched",
                                  "enriched-stream", "legacy", "embeddings"])
async def test_ungranted_plus_header_is_stored_in_full_and_is_not_silent(ns, door):
    status, headers, _ = await send(ns.proxy, door, "plain")
    assert status == 200
    # Stored exactly as if it had not asked.
    text = await ns.text()
    assert CANARY in text
    if door != "embeddings":
        row = _row_for(ns, "plain-caller")
        assert CANARY in row["payload_json"]
        assert row["no_store"] == "refused:not_granted"
        assert row["content_meta"] is None
    # …and the refusal is visible in four places.
    st = ns.svc._state
    assert st.no_store_refused == 1 and st.no_store_honoured == 0
    assert st.no_store_by_agent == {
        "plain-caller": {"refused": 1, "not_granted": 1}}
    refused = [r for r in ns.caplog.records
               if "ROADSTEAD_NO_STORE_REFUSED" in r.getMessage()]
    assert len(refused) == 1 and refused[0].levelno == logging.WARNING
    msg = refused[0].getMessage()
    assert "agent=plain-caller" in msg and "reason=not_granted" in msg
    assert "key_id=k-plain" in msg
    metrics = (await ns.proxy.client.get("/metrics", headers=ns.proxy.admin)).text
    assert ('roadstead_no_store_requests_total{agent="plain-caller",'
            'outcome="refused"} 1') in metrics
    if door.startswith("openai"):
        assert headers["x-roadstead-no-store"] == "refused:not_granted"


async def test_the_enriched_envelope_discloses_the_outcome(ns):
    r = await ns.proxy.client.post(
        "/rs/v1/chat", headers={"X-API-Key": KEYS["plain"][0],
                                "X-Roadstead-No-Store": "content"},
        json={"intent": "chat",
              "payload": {"messages": _msgs(), "max_tokens": 16}})
    assert r.json()["identity"]["no_store"] == "refused:not_granted"
    r = await ns.proxy.client.post(
        "/rs/v1/chat", headers={"X-API-Key": KEYS["granted"][0],
                                "X-Roadstead-No-Store": "content"},
        json={"intent": "chat",
              "payload": {"messages": _msgs(), "max_tokens": 16}})
    assert r.json()["identity"]["no_store"] == "honoured"


# --------------------------------------------------------------------------- #
# Nothing a caller says about itself grants it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("door", ["openai", "enriched"])
async def test_an_unauthenticated_caller_is_refused_even_under_a_granted_name(ns, door):
    """The shared-LAN case. The request arrives from an address that the ACL
    maps to `granted-caller` (as a LAN-wide label would be), the header is
    present, the stanza for that name is a VALID grant — and it is refused,
    because an address identifies a host, not a caller."""
    ns.svc._state.acl.register("127.0.0.1", "granted-caller")
    status, _, _ = await send(ns.proxy, door, None)
    assert status == 200
    assert CANARY in await ns.text()
    assert ns.svc._state.no_store_by_agent["granted-caller"]["not_authenticated"] == 1
    assert _row_for(ns, "granted-caller")["no_store"] == "refused:not_authenticated"


async def test_the_legacy_doors_self_declared_name_grants_nothing(ns):
    """§1.9.2: a caller inside the built-in internal nets NAMES ITSELF on
    `/v1/submit`. Naming a granted caller there is exactly the self-asserted
    identity the grant must not trust."""
    body = {"agent_id": "granted-caller", "endpoint": "chat",
            "payload_type": "chat_completion", "timeout_s": 20.0,
            "payload": {"messages": _msgs(), "max_tokens": 16}}
    r = await ns.proxy.client.post(
        "/v1/submit", json=body, headers={"X-Roadstead-No-Store": "content"})
    assert r.status_code == 200
    assert CANARY in await ns.text()
    assert ns.svc._state.no_store_by_agent["granted-caller"] == {
        "refused": 1, "not_authenticated": 1}


async def test_a_loopback_caller_with_no_credential_is_refused(ns):
    status, _, _ = await send(ns.proxy, "openai", None)
    assert status == 200
    assert CANARY in await ns.text()
    assert ns.svc._state.no_store_by_agent["internal"]["not_authenticated"] == 1


@pytest.mark.parametrize("claim", [
    {"X-Agent-Id": "granted-caller"},
    {"X-Roadstead-Agent-Id": "granted-caller"},
])
async def test_a_declared_agent_id_header_grants_nothing(ns, claim):
    status, _, _ = await send(ns.proxy, "openai", "plain", extra_headers=claim)
    assert status == 200
    assert CANARY in await ns.text()
    assert ns.svc._state.no_store_by_agent["plain-caller"]["not_granted"] == 1


@pytest.mark.parametrize("door", ["enriched", "legacy"])
async def test_a_body_declared_agent_id_grants_nothing(ns, door):
    """A key with no `may_assert` ignores a declared name (§1.5 rule 3), so the
    declared grantee is never the one decided on."""
    headers = {"X-API-Key": KEYS["plain"][0], "X-Roadstead-No-Store": "content"}
    if door == "enriched":
        body = {"agent_id": "granted-caller", "intent": "chat",
                "payload": {"messages": _msgs(), "max_tokens": 16}}
        r = await ns.proxy.client.post("/rs/v1/chat", json=body, headers=headers)
    else:
        body = {"agent_id": "granted-caller", "endpoint": "chat",
                "payload_type": "chat_completion", "timeout_s": 20.0,
                "payload": {"messages": _msgs(), "max_tokens": 16}}
        r = await ns.proxy.client.post("/v1/submit", json=body, headers=headers)
    assert r.status_code == 200
    assert CANARY in await ns.text()
    assert ns.svc._state.no_store_by_agent == {
        "plain-caller": {"refused": 1, "not_granted": 1}}


async def test_a_key_minted_for_a_shared_identity_is_not_a_grant(ns):
    """`shared-label` has a key AND a valid stanza AND is an address identity.
    Every holder of that label would be able to opt out; refused."""
    status, _, _ = await send(ns.proxy, "openai", "shared")
    assert status == 200
    assert CANARY in await ns.text()
    assert ns.svc._state.no_store_by_agent["shared-label"]["shared_identity"] == 1


async def test_a_stanza_missing_its_approval_is_not_a_grant_and_says_so(ns):
    status, _, _ = await send(ns.proxy, "openai", "half")
    assert status == 200
    assert CANARY in await ns.text()
    assert ns.svc._state.no_store_by_agent["half-approved"]["not_granted"] == 1
    notices = (await ns.proxy.client.get(
        "/rs/v1/admin/config", headers=ns.proxy.admin)).json()["notices"]
    assert any(n["subject"] == "half-approved" and n["problem"] == "invalid_grant"
               for n in notices)


# --------------------------------------------------------------------------- #
# The header itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("door", ["openai", "enriched", "legacy"])
@pytest.mark.parametrize("value", ["true", "all", "1", "metadata"])
async def test_an_undefined_header_value_is_a_400_not_a_shrug(ns, door, value):
    status, _, text = await send(ns.proxy, door, "granted", header_value=value)
    assert status == 400
    assert "X-Roadstead-No-Store" in text
    assert CANARY not in await ns.text()
    assert ns.svc._state.no_store_honoured == 0


async def test_the_value_is_case_and_space_tolerant(ns):
    status, _, _ = await send(ns.proxy, "openai", "granted",
                              header_value="  Content ")
    assert status == 200
    assert _row_for(ns, "granted-caller")["no_store"] == "honoured"


async def test_the_admin_plane_cannot_grant_it(ns):
    r = await ns.proxy.client.patch(
        "/rs/v1/admin/callers/plain-caller", headers=ns.proxy.admin,
        json={"content_no_store": "allowed"})
    assert r.status_code == 400
    assert "agents config file" in r.text
    assert ns.svc._state.config.agents["plain-caller"].content_no_store is False
    # …and the read view reports who holds one.
    callers = {c["agent_id"]: c for c in (await ns.proxy.client.get(
        "/rs/v1/admin/callers", headers=ns.proxy.admin)).json()["callers"]}
    assert callers["granted-caller"]["content_no_store"] is True
    assert callers["plain-caller"]["content_no_store"] is False


def _context(text: str) -> str:
    i = text.find(CANARY)
    return text[max(0, i - 200): i + 200] if i >= 0 else ""


# --------------------------------------------------------------------------- #
# Log lines that would have quoted the content
# --------------------------------------------------------------------------- #
#
# Each scenario drives a path whose log line quotes something derived from the
# prompt or the response: a validator's message about the offending value, the
# structured-empty excerpt, a backend error body. For each, the CONTROL (an
# ordinary caller) must show the canary in the log — otherwise the scenario
# proved nothing — and the granted no-store caller must not.

_SCHEMA_RF = {"type": "json_schema", "json_schema": {
    "name": "t", "schema": {
        "type": "object", "properties": {"answer": {"type": "integer"}},
        "required": ["answer"], "additionalProperties": False}}}
_EMPTYISH_RF = {"type": "json_schema", "json_schema": {
    "name": "t", "schema": {"type": "object"}}}


async def _post(proxy, who, *, no_store, body_extra=None, path="/v1/chat/completions"):
    headers = {"X-API-Key": KEYS[who][0]}
    if no_store:
        headers["X-Roadstead-No-Store"] = "content"
    body = {"model": "chat", "messages": _msgs(), "max_tokens": 16}
    body.update(body_extra or {})
    return await proxy.client.post(path, json=body, headers=headers)


def _log_text(ns: Surfaces) -> str:
    return "\n".join(
        r.getMessage() + ("\n" + (r.exc_text or "") if r.exc_text else "")
        for r in ns.caplog.records if r.name.startswith("roadstead"))


async def _scenario_schema(ns, who, no_store, monkeypatch):
    monkeypatch.setenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "1")
    ns.proxy.controller.structured_content = json.dumps({"answer": CANARY})
    return await _post(ns.proxy, who, no_store=no_store,
                       body_extra={"response_format": _SCHEMA_RF})


async def _scenario_structured_empty(ns, who, no_store, monkeypatch):
    ns.proxy.controller.structured_content = json.dumps({CANARY: ""})
    return await _post(ns.proxy, who, no_store=no_store,
                       body_extra={"response_format": _EMPTYISH_RF})


def _failing_backend(ns, make_exc, *, then_ok: bool):
    svc = ns.svc
    real = svc._backend.call
    state = {"n": 0}

    async def call(ep_cfg, payload, payload_type, request_id, timeout_s=180.0):
        state["n"] += 1
        if state["n"] == 1 or not then_ok:
            raise make_exc()
        return await real(ep_cfg, payload, payload_type, request_id,
                          timeout_s=timeout_s)

    svc._backend.call = call


async def _scenario_transient(ns, who, no_store, monkeypatch):
    from roadstead.backend import BackendUnavailable
    _failing_backend(ns, lambda: BackendUnavailable(
        f"backend disconnected while reading {CANARY}"), then_ok=True)
    return await _post(ns.proxy, who, no_store=no_store)


async def _scenario_dispatch_failed(ns, who, no_store, monkeypatch):
    _failing_backend(ns, lambda: RuntimeError(f"boom {CANARY}"), then_ok=False)
    return await _post(ns.proxy, who, no_store=no_store)


async def _scenario_structured_fault(ns, who, no_store, monkeypatch):
    from roadstead.backend import BackendError
    _failing_backend(ns, lambda: BackendError(
        500, f"InternalServerError while decoding {CANARY}"), then_ok=True)
    return await _post(ns.proxy, who, no_store=no_store,
                       body_extra={"response_format": _SCHEMA_RF})


async def _scenario_grammar_error(ns, who, no_store, monkeypatch):
    # The detail names the undefined rule -- here, the canary.
    return await _post(ns.proxy, who, no_store=no_store,
                       body_extra={"grammar": f"root ::= {CANARY}"})


SCENARIOS = {
    "grammar_error_detail": _scenario_grammar_error,
    "schema_backstop_message": _scenario_schema,
    "structured_empty_excerpt": _scenario_structured_empty,
    "transient_backend_error": _scenario_transient,
    "dispatch_failed": _scenario_dispatch_failed,
    "structured_fault_retry": _scenario_structured_fault,
}


#: Refused at admission (422): there is no completion row to inspect.
REJECTED_BEFORE_DISPATCH = {"grammar_error_detail"}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_a_log_line_that_quotes_content_is_redacted_for_no_store(
        ns, monkeypatch, name):
    # CONTROL: an ordinary caller. The instrument has to see the canary in the
    # log, or the redaction below is untested.
    await SCENARIOS[name](ns, "plain", False, monkeypatch)
    assert CANARY in _log_text(ns), (
        f"{name}: the control never logged the canary — the scenario does not "
        f"reach the log line it claims to cover")
    ns.caplog.clear()
    ns.proxy.controller.reset()

    await SCENARIOS[name](ns, "granted", True, monkeypatch)
    assert CANARY not in _log_text(ns), _context(_log_text(ns))
    # (Not the whole-surface sweep: the control above legitimately left the
    # canary in this test's database. The granted caller's own row is the
    # claim.)
    if name not in REJECTED_BEFORE_DISPATCH:
        row = _row_for(ns, "granted-caller")
        assert row["no_store"] == "honoured"
        assert row["payload_json"] is None and row["response_json"] is None
    assert CANARY not in "\n".join(ns.sse)


async def test_the_unhandled_error_backstop_logs_the_type_only(ns):
    async def boom(body, request):
        raise ValueError(f"cannot parse {CANARY}")

    ns.svc.handle_openai_chat = boom
    h = {"X-API-Key": KEYS["granted"][0]}
    body = {"model": "chat", "messages": _msgs(), "max_tokens": 16}

    r = await ns.proxy.client.post("/v1/chat/completions", json=body, headers=h)
    assert r.status_code == 500
    assert CANARY in _log_text(ns), "control: the backstop logs the traceback"
    ns.caplog.clear()

    r = await ns.proxy.client.post(
        "/v1/chat/completions", json=body,
        headers={**h, "X-Roadstead-No-Store": "content"})
    assert r.status_code == 500
    assert CANARY not in _log_text(ns)
    assert "ValueError" in _log_text(ns) and "traceback withheld" in _log_text(ns)


# --------------------------------------------------------------------------- #
# The in-memory stores that outlive the call
# --------------------------------------------------------------------------- #

async def test_no_store_neither_feeds_nor_reads_the_replay_cache(ns):
    from roadstead.testing.fake_backend import ThinkScript
    ep = ns.svc._state.config.endpoints["tier3"]
    ep.replay_reasoning_history = True
    st = ns.svc._state

    def think(body):
        return ThinkScript(mode="clean", reasoning=f"thinking about {CANARY}",
                           answer="fine")

    ns.proxy.controller.think = think
    msgs = [{"role": "user", "content": CANARY}]
    h = {"X-API-Key": KEYS["granted"][0]}

    await ns.proxy.client.post("/v1/chat/completions", headers={
        **h, "X-Roadstead-No-Store": "content"},
        json={"model": "tier3", "messages": msgs, "max_tokens": 32})
    assert st.reasoning_replay_stored == 0 and st.reasoning_replay.size == 0
    assert CANARY not in repr(st.reasoning_replay._cache)

    # CONTROL: the same call without the header IS stored.
    await ns.proxy.client.post("/v1/chat/completions", headers=h,
        json={"model": "tier3", "messages": msgs, "max_tokens": 32})
    assert st.reasoning_replay_stored == 1 and st.reasoning_replay.size == 1

    # …and a no-store follow-up that replays that turn does not READ it back.
    hist = msgs + [{"role": "assistant", "content": "fine"},
                   {"role": "user", "content": "and?"}]
    miss_before = st.reasoning_replay_miss
    await ns.proxy.client.post("/v1/chat/completions", headers={
        **h, "X-Roadstead-No-Store": "content"},
        json={"model": "tier3", "messages": hist, "max_tokens": 32})
    assert st.reasoning_replay_restored == 0
    assert st.reasoning_replay_miss == miss_before


async def test_no_store_is_never_captured_by_prefix_keepalive(ns):
    ep = ns.svc._state.config.endpoints["tier3"]
    ep.prefix_keepalive_call_sites = ("*",)
    ep.prefix_keepalive_trigger_tokens = 500
    ep.prefix_keepalive_idle_s = 0
    tracker = ns.svc._state.prefix_keepalive
    h = {"X-API-Key": KEYS["granted"][0]}
    body = {"model": "tier3", "messages": _msgs(), "tools": _TOOLS,
            "max_tokens": 16}

    r = await ns.proxy.client.post("/v1/chat/completions", json=body, headers={
        **h, "X-Roadstead-No-Store": "content"})
    assert r.status_code == 200
    assert tracker.status_snapshot("tier3", 0.0) is None, (
        "a no-store prefix was captured and would be re-sent on a timer")
    assert CANARY not in repr(tracker._by_endpoint)

    # CONTROL
    r = await ns.proxy.client.post("/v1/chat/completions", json=body, headers=h)
    assert r.status_code == 200
    snap = tracker.status_snapshot("tier3", 0.0)
    assert snap is not None and len(snap["tracked"]) == 1


async def test_no_store_is_not_sent_to_the_shadow_backend_or_recorded_there(ns):
    ep = ns.svc._state.config.endpoints["tier3"]
    ep.shadow_host, ep.shadow_port = ns.proxy.fake.host, ns.proxy.fake.port
    h = {"X-API-Key": KEYS["granted"][0]}
    body = {"model": "tier3", "messages": _msgs(), "max_tokens": 16}

    import asyncio
    before = len(ns.proxy.controller.requests)
    await ns.proxy.client.post("/v1/chat/completions", json=body, headers={
        **h, "X-Roadstead-No-Store": "content"})
    await asyncio.sleep(0.4)
    chat_calls = [r for r in list(ns.proxy.controller.requests)[before:]
                  if r.path.endswith("/chat/completions")]
    assert len(chat_calls) == 1, "the shadow comparison fired for a no-store request"
    assert "shadow-" not in ns.db_rows()

    # CONTROL: an ordinary request is shadowed and the pair is recorded.
    await ns.proxy.client.post("/v1/chat/completions", json=body, headers=h)
    await asyncio.sleep(0.6)
    assert "shadow-" in ns.db_rows()


async def test_a_no_store_request_is_never_eligible_to_spill(ns, monkeypatch):
    """Spill sends the content to a third party. The flag the scheduler reads
    is forced False for an honoured request — and left exactly as the caller
    sent it for everyone else."""
    seen = {}

    async def capture(req, cache_key, *, wire="enriched"):
        seen[req.agent_id] = (req.allow_spill, req.no_store_outcome)
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    monkeypatch.setattr(ns.svc._lifecycle, "handle_sync_submit", capture)
    for who, ns_flag in (("granted", True), ("plain", False)):
        headers = {"X-API-Key": KEYS[who][0]}
        if ns_flag:
            headers["X-Roadstead-No-Store"] = "content"
        await ns.proxy.client.post("/rs/v1/chat", headers=headers, json={
            "intent": "chat", "allow_spill": True,
            "substitution": {"allow_spill": True},
            "payload": {"messages": _msgs(), "max_tokens": 16}})
    assert seen["granted-caller"] == (False, "honoured")
    assert seen["plain-caller"][1] == ""
    assert seen["plain-caller"][0] is not False


async def test_a_corrected_row_rewrite_does_not_restore_the_content(ns, monkeypatch):
    """The schema backstop repairs a response in memory and then REWRITES the
    completion row (`INSERT OR REPLACE` on request_id) with the corrected body.
    That second write is a second chance to store what the first one withheld."""
    monkeypatch.setenv("ROADSTEAD_PROXY_SCHEMA_BACKSTOP", "1")
    rf = {"type": "json_schema", "json_schema": {"name": "t", "schema": {
        "type": "object", "properties": {"answer": {"type": "string"}},
        "required": ["answer"], "additionalProperties": False}}}
    ns.proxy.controller.set_fault("schema_invalid", 0.0)

    # CONTROL: the repair fires and the rewritten row holds the content.
    r = await _post(ns.proxy, "plain", no_store=False,
                    body_extra={"response_format": rf})
    assert r.status_code == 200
    assert _row_for(ns, "plain-caller")["response_json"] is not None
    assert ns.svc._state.schema_repaired == 1, "the scenario never reached the rewrite"

    r = await _post(ns.proxy, "granted", no_store=True,
                    body_extra={"response_format": rf})
    assert r.status_code == 200
    assert ns.svc._state.schema_repaired == 2
    row = _row_for(ns, "granted-caller")
    assert row["no_store"] == "honoured"
    assert row["payload_json"] is None and row["response_json"] is None
    assert CANARY not in row["content_meta"]


async def test_no_store_leaves_no_trace_in_the_grammar_memos(ns):
    """`grammar_cache` keeps the grammar's text for the life of the process and
    `grammar_alerted` the hash of every invalid one. Both bypassed — checked
    with a VALID grammar (cache) and an INVALID one (alert set, and a log line
    per request instead of one per distinct grammar)."""
    st = ns.svc._state
    valid = f'root ::= "{CANARY}" | "no"'
    r = await _post(ns.proxy, "granted", no_store=True, body_extra={"grammar": valid})
    assert r.status_code == 200
    assert st.grammar_cache == {} and st.grammar_alerted == set()
    assert CANARY not in repr(st.grammar_cache)

    # CONTROL: an ordinary caller's grammar IS memoised.
    r = await _post(ns.proxy, "plain", no_store=False, body_extra={"grammar": valid})
    assert r.status_code == 200 and len(st.grammar_cache) == 1
    st.grammar_cache.clear()

    bad = f"root ::= {CANARY}"
    for _ in range(2):
        r = await _post(ns.proxy, "granted", no_store=True, body_extra={"grammar": bad})
        assert r.status_code == 422
    assert st.grammar_cache == {} and st.grammar_alerted == set()
    errs = [r_ for r_ in ns.caplog.records if "GRAMMAR INVALID" in r_.getMessage()]
    assert len(errs) == 2 and all(CANARY not in e.getMessage() for e in errs)
