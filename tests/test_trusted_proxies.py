"""Trusted proxies — believing `X-Forwarded-For`, and only from the right peer.

Until 2026-09-01 `identity.remote_ip` read `request.client.host` and nothing
else. Put any reverse proxy or TLS terminator in front of Roadstead — which the
management UI will need — and two things happened, the second of them a security
bug:

* every caller collapsed into the proxy's address, so the whole IP layer became
  one identity and `ROADSTEAD_ACL` silently stopped distinguishing anybody;
* if that address fell inside the admin nets — **loopback and docker-internal
  are there by DEFAULT, and a sidecar proxy is usually one of them** — the
  control plane was granted to everyone who could reach the proxy.

It was latent only because nothing told anybody to deploy that way.

Four decisions are pinned here, each of which looks arbitrary until it bites:

1. **A forwarded header is never believed by default.** It is a caller-supplied
   string, and honouring one unconditionally lets any caller assert any source
   address — the self-asserted `agent_id` bug in its third costume.
2. **The hop is taken from the RIGHT.** The leftmost element is whatever the
   original caller sent. Taking it re-introduces the spoof that trusting the
   header at all was supposed to close.
3. **A forwarded address does not inherit the BUILT-IN admin nets.** The whole
   justification for auto-granting admin to loopback is that reaching loopback
   meant already being on the box; a front proxy is precisely the thing that
   makes that untrue.
4. **`identity.py` is the only place an address is resolved** — a call site that
   reads the peer directly is a call site where the collapse comes back, and
   nothing about that bug fails loudly.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from roadstead import hooks
from roadstead.acl import IPIdentityMap
from roadstead.config import LLMPriority, ProxyConfig
from roadstead.identity import (
    ClientAddress,
    IdentityResolver,
    KeyRegistry,
    TrustedProxies,
    client_address,
    remote_ip,
)
from roadstead.management import PREFIX
from roadstead.service import ProxyService

_ROOT = Path(__file__).resolve().parents[1]

#: A plausible sidecar: the proxy shares the loopback interface, which is what
#: makes the default admin grant dangerous rather than merely wrong.
_PROXY = "127.0.0.1"
#: A caller out on the internet, arriving through it.
_CALLER = "203.0.113.9"


class _Req:
    """The slice of a Starlette request the identity layer touches."""

    def __init__(self, *, host=_PROXY, headers=None, method="GET",
                 body=None, path_params=None):
        class _C:
            pass

        _C.host = host
        self.client = _C()
        self.headers = headers or {}
        self.method = method
        self.query_params: dict = {}
        self.path_params = path_params or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _xff(*hops: str) -> dict:
    return {"X-Forwarded-For": ", ".join(hops)}


def _resolver(*, trusted: str = "", acl: IPIdentityMap | None = None,
              keys: KeyRegistry | None = None) -> IdentityResolver:
    return IdentityResolver(
        acl if acl is not None else IPIdentityMap(),
        keys if keys is not None else KeyRegistry(),
        require_key=False,
        trusted_proxies=TrustedProxies.parse(trusted),
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("ROADSTEAD_API_KEYS", "ROADSTEAD_API_KEYS_FILE", "ROADSTEAD_ACL",
                "LLM_PROXY_ACL", "ROADSTEAD_ADMIN_NETS", "ROADSTEAD_REQUIRE_API_KEY",
                "ROADSTEAD_TRUSTED_PROXIES"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# 1. The default: the header is not read at all
# ---------------------------------------------------------------------------

def test_by_default_a_forwarded_header_is_not_read_at_all():
    """🚨 The default is not "validated and rejected" — it is never consulted.

    An empty trusted set has to preserve today's behaviour EXACTLY, because
    every existing deployment has one: an OpenAI client behind a corporate proxy
    already sends this header, and a proxy that started believing it on upgrade
    would silently re-identify traffic that has been working for months.
    """
    resolver = _resolver()  # ROADSTEAD_TRUSTED_PROXIES unset
    req = _Req(host="198.51.100.4", headers=_xff("192.0.2.77"))
    assert resolver.client_address(req) == ClientAddress(ip="198.51.100.4",
                                                         forwarded=False)
    assert resolver.client_ip(req) == remote_ip(req)


def test_a_header_from_an_untrusted_peer_is_ignored_even_when_proxies_exist():
    """Trust is per-connection, not global. Configuring a front proxy does not
    make the header believable from anywhere else — otherwise opting in for the
    sidecar would open the spoof to every caller that bypasses it."""
    resolver = _resolver(trusted=_PROXY)
    req = _Req(host="198.51.100.4", headers=_xff(_CALLER))
    assert resolver.client_address(req).ip == "198.51.100.4"
    assert resolver.client_address(req).forwarded is False


def test_an_unparseable_trusted_proxy_entry_is_dropped_and_reported():
    """A typo removes trust rather than granting it — the safe direction, and
    for that reason a completely silent one. This is the exact shape
    `hooks.config_notice` exists for: the operator believes the header is being
    honoured, it is not, and every caller behind the proxy is collapsing into
    one identity with nothing failing."""
    hooks.clear_config_notices()
    proxies = TrustedProxies.parse("127.0.0.1, not-an-address, 10.0.0.0/8")
    assert proxies.networks() == ["127.0.0.1/32", "10.0.0.0/8"]
    notices = [n for n in hooks.config_notices() if n["subject"] == "not-an-address"]
    assert len(notices) == 1
    assert notices[0]["problem"] == "unparseable"
    assert notices[0]["source"] == "ROADSTEAD_TRUSTED_PROXIES"


def test_the_env_var_is_the_opt_in(monkeypatch):
    monkeypatch.setenv("ROADSTEAD_TRUSTED_PROXIES", "172.18.0.0/16")
    resolver = IdentityResolver(IPIdentityMap(), KeyRegistry(), require_key=False)
    assert resolver.proxies.networks() == ["172.18.0.0/16"]
    assert resolver.client_address(
        _Req(host="172.18.0.2", headers=_xff(_CALLER))).ip == _CALLER


# ---------------------------------------------------------------------------
# 2. Which hop — the spoof, and the arithmetic that closes it
# ---------------------------------------------------------------------------

def test_the_caller_is_the_rightmost_untrusted_hop_not_the_leftmost():
    """🚨 THE test. A caller that sends its own `X-Forwarded-For` has that value
    pushed LEFT when the real proxy appends what it observed — so the leftmost
    element is exactly the attacker-controlled one, and reading it would let any
    caller name its own source address.

    Here the caller claims to be an enrolled LAN host and is in fact out on the
    internet. It must be identified as what the proxy saw.
    """
    acl = IPIdentityMap()
    acl.register("192.0.2.77", "privileged-lan-host", LLMPriority.P1_TURN_SUPPORT)
    resolver = _resolver(trusted=_PROXY, acl=acl)

    # The caller sent "192.0.2.77"; the proxy appended the address it observed.
    req = _Req(host=_PROXY, headers=_xff("192.0.2.77", _CALLER))
    assert resolver.client_address(req).ip == _CALLER

    res = resolver.resolve(req)
    assert not res.ok, "the spoofed hop was believed"
    assert res.denial.status == 403
    assert _CALLER in res.denial.message


def test_prepending_trusted_addresses_does_not_push_the_caller_off_the_end():
    """The walk stops at the first UNTRUSTED hop from the right, so a caller can
    prepend as many trusted-looking addresses as it likes and never be reached.
    A fixed "(N+1)th from the right" count would be walked past by exactly this.
    """
    resolver = _resolver(trusted="127.0.0.1, 10.0.0.0/8")
    req = _Req(host=_PROXY, headers=_xff("10.0.0.1", "10.0.0.2", "10.0.0.3", _CALLER))
    assert resolver.client_address(req).ip == _CALLER


def test_a_chain_of_trusted_proxies_is_walked_through():
    """Two hops of real infrastructure: the caller reaches proxy1, which reaches
    proxy2, which reaches us. The chain is [caller, proxy1] and the peer is
    proxy2 — so the answer is three addresses away from the socket."""
    resolver = _resolver(trusted="127.0.0.1, 10.0.0.0/16")
    req = _Req(host=_PROXY, headers=_xff(_CALLER, "10.0.0.7"))
    assert resolver.client_address(req) == ClientAddress(ip=_CALLER, forwarded=True)


def test_a_chain_of_only_trusted_hops_yields_the_leftmost():
    """Every hop was itself a trusted proxy, so the caller is further left than
    the chain goes. The leftmost is the closest thing to an answer there is —
    and it is a proxy's address, which is unregistered and therefore refused
    rather than admitted as somebody."""
    resolver = _resolver(trusted="127.0.0.1, 10.0.0.0/16")
    addr = resolver.client_address(_Req(host=_PROXY, headers=_xff("10.0.0.9", "10.0.0.7")))
    assert addr == ClientAddress(ip="10.0.0.9", forwarded=True)


@pytest.mark.parametrize("hop, expected", [
    ("203.0.113.9", "203.0.113.9"),
    ("203.0.113.9:51234", "203.0.113.9"),          # some proxies append the port
    ("[2001:db8::1]:443", "2001:db8::1"),          # …and bracket IPv6 when they do
    ("2001:0db8:0000::1", "2001:db8::1"),          # normalised, so the ACL matches
    ("  203.0.113.9  ", "203.0.113.9"),
])
def test_hop_spellings_that_real_proxies_emit(hop, expected):
    resolver = _resolver(trusted=_PROXY)
    assert resolver.client_address(_Req(host=_PROXY, headers={"X-Forwarded-For": hop})).ip == expected


def test_an_unparseable_hop_fails_closed_rather_than_back_to_the_peer():
    """🚨 Falling back to the peer here would hand the PROXY's identity — and
    its built-in admin grant — to anyone who sends junk, which is the bug this
    whole module exists to close, reachable by typing garbage into a header.

    `"unknown"` is deliberately not an address: `IPIdentityMap` cannot match it,
    so it is refused, and it never reaches a log or the `admin_ips_seen` set as
    attacker-controlled text.
    """
    resolver = _resolver(trusted=_PROXY)
    req = _Req(host=_PROXY, headers=_xff("not-an-address"))
    assert resolver.client_address(req) == ClientAddress(ip="unknown", forwarded=True)
    res = resolver.resolve(req)
    assert not res.ok and res.denial.status == 403


def test_a_hostile_length_chain_is_not_parsed():
    """The walk stops at the first untrusted hop, so the only way to lengthen it
    is to repeat a trusted address — which is what the cap is for."""
    resolver = _resolver(trusted="127.0.0.1, 10.0.0.0/8")
    req = _Req(host=_PROXY, headers=_xff(*(["10.0.0.1"] * 200 + [_CALLER])))
    assert resolver.client_address(req).ip == "unknown"


def test_a_trusted_proxy_that_forwards_nothing_is_still_marked_forwarded():
    """The misconfiguration: a front proxy is deployed but not setting the
    header. Every caller collapses into it — which is unavoidable, since nothing
    else was sent — but the request is still `forwarded`, and that is what
    withdraws the admin grant below. A front proxy that lost its config must not
    become an administrator."""
    resolver = _resolver(trusted=_PROXY)
    assert resolver.client_address(_Req(host=_PROXY)) == ClientAddress(ip=_PROXY,
                                                                       forwarded=True)


# ---------------------------------------------------------------------------
# 3. Admin — the half that is a security bug rather than an inconvenience
# ---------------------------------------------------------------------------

def test_todays_grant_is_what_makes_this_dangerous():
    """The baseline, stated so the change below reads as a change. A direct
    loopback connection IS admin, by default, with no configuration — that is
    the deliberate local-first posture, and it is only safe while reaching
    loopback means already being on the machine."""
    assert _resolver().is_admin(_Req(host=_PROXY)) is True


def test_a_forwarded_caller_does_not_inherit_the_builtin_admin_grant():
    """🚨 The security bug, closed. A caller out on the internet arrives through
    a sidecar proxy sharing loopback. Before trusted-proxy handling it resolved
    to 127.0.0.1 and was handed the control plane; now it resolves to itself and
    is refused."""
    resolver = _resolver(trusted=_PROXY)
    assert resolver.is_admin(_Req(host=_PROXY, headers=_xff(_CALLER))) is False


def test_the_proxys_own_address_stops_being_admin_once_it_is_a_proxy():
    """The subtler half. Even when the resolved address IS loopback — because
    the proxy forwarded nothing, or forwarded its own address — the built-in
    grant is gone, because 'this arrived on loopback' has stopped being a
    statement about who is calling.

    An operator who wants it back says so explicitly, which is the point: the
    grant becomes a decision rather than an inherited default.
    """
    resolver = _resolver(trusted=_PROXY)
    assert resolver.is_admin(_Req(host=_PROXY)) is False
    assert resolver.is_admin(_Req(host=_PROXY, headers=_xff(_PROXY))) is False

    acl = IPIdentityMap()
    acl.register_admin_net(_PROXY)
    assert _resolver(trusted=_PROXY, acl=acl).is_admin(_Req(host=_PROXY)) is True


def test_an_operator_admin_net_still_grants_a_forwarded_address():
    """Narrowing the BUILT-IN grant must not narrow the operator's. Somebody who
    writes ROADSTEAD_ADMIN_NETS is naming an address on purpose, and that
    statement is just as true through a proxy as around one."""
    acl = IPIdentityMap()
    # What `ROADSTEAD_ACL='203.0.113.0/24=ops:admin'` does: an admin net alone
    # grants no IDENTITY, and an unidentified address is refused before the
    # admin question is reached. That is pre-existing and correct — an admin net
    # is a narrowing of who may administer, never a way in.
    acl.register("203.0.113.0/24", "ops", LLMPriority.P3_INGESTION)
    acl.register_admin_net("203.0.113.0/24")
    resolver = _resolver(trusted=_PROXY, acl=acl)
    assert resolver.is_admin(_Req(host=_PROXY, headers=_xff(_CALLER))) is True


def test_an_admin_key_is_the_path_that_is_unaffected():
    """The recommended answer, and the reason narrowing the address grant costs
    an operator little: a key works from anywhere, through any number of
    proxies, and can be revoked. §1.5's precedence is untouched — a credential
    is resolved before an address is even looked at."""
    keys = KeyRegistry()
    keys.register(secret="ops-key", agent_id="ops", admin=True, key_id="ops")
    resolver = _resolver(trusted=_PROXY, keys=keys)
    req = _Req(host=_PROXY, headers={**_xff(_CALLER),
                                     "Authorization": "Bearer ops-key"})
    assert resolver.is_admin(req) is True
    res = resolver.resolve(req)
    assert res.principal.agent_id == "ops" and res.principal.authenticated


def test_the_acl_reports_the_two_kinds_of_admin_net_separately():
    """Which of the two a grant came from decides whether it survives a proxy
    being put in front, so they cannot be one list."""
    acl = IPIdentityMap()
    assert acl.operator_admin_nets() == []
    assert "127.0.0.0/8" in acl.builtin_admin_nets()
    acl.register_admin_net("203.0.113.0/24")
    assert acl.operator_admin_nets() == ["203.0.113.0/24"]
    assert acl.is_admin("203.0.113.9", trust_builtin_nets=False) is True
    assert acl.is_admin("127.0.0.1", trust_builtin_nets=False) is False
    assert acl.is_admin("127.0.0.1") is True


# ---------------------------------------------------------------------------
# 4. One place resolves an address
# ---------------------------------------------------------------------------

def test_no_module_outside_identity_reads_the_peer_address_directly():
    """🚨 The rule with a test, in the shape `tests/test_provider_interface.py`
    uses for engine names.

    `management.py` had its own `_remote_ip` reading `request.client.host` — a
    second answer to a question `identity.py` is supposed to own, on the surface
    where getting it wrong grants the control plane. A third one would be just
    as quiet: the failure mode of reading the peer directly is not an exception,
    it is every caller sharing one identity.
    """
    class _Sweep(ast.NodeVisitor):
        """Every way of reaching ``request.client``, not one spelling of it.

        🚨 A substring sweep for ``client.host`` is satisfied by prose: rewriting
        it as ``getattr(getattr(request, "client", None), "host", "")`` is the
        same bug and the same security consequence, and it slips straight past.
        This looks for the ATTRIBUTE — reached however — which is the thing the
        rule is actually about.
        """

        def __init__(self) -> None:
            self.hits: list[int] = []

        def visit_Attribute(self, node):
            if node.attr == "client":
                self.hits.append(node.lineno)
            self.generic_visit(node)

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "client"):
                self.hits.append(node.lineno)
            self.generic_visit(node)

    offenders = []
    for path in sorted((_ROOT / "roadstead").rglob("*.py")):
        if path.name == "identity.py":
            continue  # the one place, by design
        text = path.read_text(encoding="utf-8")
        sweep = _Sweep()
        sweep.visit(ast.parse(text))
        lines = text.splitlines()
        for lineno in sweep.hits:
            offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {lines[lineno - 1].strip()}")
    assert not offenders, (
        "resolve an address through IdentityResolver.client_ip / client_address, "
        "not off request.client — otherwise a reverse proxy collapses every "
        "caller here into one identity:\n" + "\n".join(offenders))


@pytest.mark.asyncio
async def test_a_real_admin_route_gates_and_audits_on_the_caller(tmp_path):
    """Driven through the HANDLER, not through the resolver it calls.

    `/v1/status`'s `admin_ips_seen` exists to answer "who actually reaches the
    control plane" before tightening the ACL. Behind a proxy it answered "the
    proxy, every time" — the question not being answered at all — and the gate
    beside it said yes for the same reason. Both come off one address, so both
    are checked here on one request that really goes through the door.
    """
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db"),
                                   admin_store_path=str(tmp_path / "o.json")))
    svc._state.identity.proxies = TrustedProxies.parse(_PROXY)
    svc._state.acl.register("203.0.113.0/24", "ops", LLMPriority.P3_INGESTION)
    svc._state.acl.register_admin_net("203.0.113.0/24")

    resp = await svc.handle_admin_flags(_Req(host=_PROXY, headers=_xff(_CALLER)))
    assert resp.status_code == 200
    assert svc._state.admin_ips_seen["/v1/admin/flags"] == {_CALLER}

    # And the caller the proxy did NOT vouch for is refused, from the same peer.
    stranger = "198.51.100.77"
    resp = await svc.handle_admin_flags(_Req(host=_PROXY, headers=_xff(stranger)))
    assert resp.status_code == 403
    assert stranger in json.loads(resp.body)["error"]


# ---------------------------------------------------------------------------
# 5. The management plane shows it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_config_view_reports_what_is_trusted_and_what_that_changed(tmp_path):
    """§3.5's thesis applied to this: an operator who configures a front proxy
    silently loses the loopback admin grant they have been using. That is a gap
    between what they wrote and what is in force, so it belongs on the surface
    built for exactly that rather than in a 403 a week later.
    """
    svc = ProxyService(ProxyConfig(queue_db_path=str(tmp_path / "q.db"),
                                   admin_store_path=str(tmp_path / "o.json")))
    svc._state.identity.proxies = TrustedProxies.parse("172.18.0.0/16")
    svc._state.acl.register_admin_net("203.0.113.0/24")

    resp = await svc.handle_admin_config(_Req(host=_PROXY))
    sources = json.loads(resp.body)["sources"]

    assert sources["trusted_proxies"] == {
        "env_var": "ROADSTEAD_TRUSTED_PROXIES",
        "networks": ["172.18.0.0/16"],
        "forwarded_headers_honoured": True,
        "builtin_admin_nets_apply_to_forwarded": False,
    }
    assert sources["admin_nets"]["operator"] == ["203.0.113.0/24"]
    assert "127.0.0.0/8" in sources["admin_nets"]["builtin"]


def test_the_env_var_the_code_reads_is_the_one_the_document_names():
    """The two-ended pin `docs/api.md` gets everywhere else, applied here.

    An identity knob that the document spells differently from the code is
    worse than an undocumented one: an operator sets it, reads the contract
    back, and concludes they are behind a trusted proxy when the header is
    being ignored and every caller is collapsing into one identity. Nothing
    fails — that is the whole failure mode this workstream exists to close, and
    a stale document reintroduces it by hand.
    """
    source = (_ROOT / "roadstead" / "identity.py").read_text(encoding="utf-8")
    assert source.count('os.environ.get("ROADSTEAD_TRUSTED_PROXIES"') == 1

    api = (_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    section = api.split("### 1.5 Identity", 1)[1].split("### 1.6 ", 1)[0]
    assert "ROADSTEAD_TRUSTED_PROXIES" in section, (
        "§1.5 is where a caller's identity is specified, and the trusted-proxy "
        "rule decides which address that identity is read from")
    assert "ROADSTEAD_TRUSTED_PROXIES" in (_ROOT / "README.md").read_text(encoding="utf-8")


def test_the_document_states_the_hop_rule_rather_than_only_the_opt_in():
    """🚨 Naming the variable is not the contract; WHICH HOP is the contract.

    An operator who configures a trusted proxy and assumes the leftmost element
    is the caller will write an ACL that a caller can satisfy by typing an
    address into a header. §1.5 has to say which end, and this reads it back —
    the same lesson as E's route table, where a guard satisfied by prose was
    asserting nothing.
    """
    api = (_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    # Collapsed, because the claim is the sentence and not the line wrapping —
    # a guard that a reflow can break teaches the next person to delete it.
    section = " ".join(
        api.split("### 1.5 Identity", 1)[1].split("### 1.6 ", 1)[0].split())
    assert "the caller is the rightmost hop that is not one" in section
    assert "empty by default" in section
    assert "A forwarded address does not inherit the built-in admin nets" in section
