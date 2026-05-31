"""ACL identity map — was untested, and it assigns the priority band that drives
DRR (Phase 4.3 coverage)."""

from __future__ import annotations

import pytest

from originfleet.llmproxy.acl import IPIdentityMap
from originfleet.llmproxy.config import LLMPriority


def test_internal_nets_are_internal():
    acl = IPIdentityMap.from_env()
    for ip in ("127.0.0.1", "172.16.0.5", "172.31.255.1"):
        assert acl.identify(ip) == ("internal", LLMPriority.P1_TURN_SUPPORT), ip


def test_lan_generic_subnet():
    acl = IPIdentityMap.from_env()
    r = acl.identify("10.0.0.50")
    assert r == ("lan-generic", LLMPriority.P3_INGESTION)


def test_exact_registration_beats_subnet():
    # 10.0.0.9 is registered exact (tideway) AND falls in 10.0.0.0/24 (lan-generic);
    # the exact match must win.
    acl = IPIdentityMap.from_env()
    assert acl.identify("10.0.0.9") == ("tideway", LLMPriority.P3_INGESTION)


def test_unknown_ip_denied():
    acl = IPIdentityMap.from_env()
    assert acl.identify("8.8.8.8") is None
    assert acl.is_allowed("8.8.8.8") is False


def test_malformed_ip_does_not_crash():
    acl = IPIdentityMap.from_env()
    assert acl.identify("not-an-ip") is None


def test_env_parse_priority_and_default(monkeypatch):
    monkeypatch.setenv(
        "LLM_PROXY_ACL", "10.2.0.0/16=batch:P4_HYGIENE,10.3.3.3=special")
    acl = IPIdentityMap.from_env()
    assert acl.identify("10.2.5.5") == ("batch", LLMPriority.P4_HYGIENE)
    # no priority suffix → default P3
    assert acl.identify("10.3.3.3") == ("special", LLMPriority.P3_INGESTION)


def test_malformed_env_entry_skipped(monkeypatch):
    monkeypatch.setenv("LLM_PROXY_ACL", "garbage-no-equals,10.4.0.0/16=ok")
    acl = IPIdentityMap.from_env()           # must not raise
    assert acl.identify("10.4.1.1") == ("ok", LLMPriority.P3_INGESTION)
