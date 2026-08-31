# Quarantined tests — Group A is done; B and C remain

These fourteen files came across in the extraction but could not run standalone. They are
**excluded from collection** (`norecursedirs` in `pyproject.toml`). None of them is dead — each needs
a small, well-understood change.

**Eight remain.** Everything else is green: **1127 passed, 1 skipped** in a clean venv with no host
application present.

## ✅ Group A — host-coupled seam tests (6 files) — DONE 2026-08-31

`test_structured_empty_detection.py` · `test_error_taxonomy.py` · `test_context_gate.py`
`test_phase5_reliability.py` · `test_phase1_integrity.py` · `test_timeout_apply.py`

These imported `originfleet.framework.*` because they asserted the **contract between the proxy and
its client** from the client's side. All six are resolved; +61 tests to the suite. How each was
split, since the same reasoning applies to anything similar that turns up later:

**The general fix — `tests/wire_contract.py`.** Four of the six used a host predicate as an oracle
(`is_deferrable_llm_error`, `is_context_overflow_error`). Re-implementing that rule inside Roadstead
and asserting it agrees with itself would be the tautology trap. Instead the marker substrings are
now **literals transcribed from `docs/api.md`** — the boundary object both sides read and neither
owns. Roadstead's tests pin what it *emits* against them; the monorepo's pin what its classifier
*matches* against them. `tests/test_wire_contract.py` reads `docs/api.md` back, so the transcription
cannot go stale silently.

| file | disposition |
|---|---|
| `test_structured_empty_detection.py` | Moved whole. Its one host test now spies on `roadstead.hooks.set_degradation_sink()` and asserts the **full event payload**, not just that a counter moved — a stronger test than the original. Gained a second: a sink that raises must not break a response. |
| `test_error_taxonomy.py` | Moved whole, classifier oracle → `carries_deferral_marker`. |
| `test_context_gate.py` | Moved whole, oracle → the verbatim `CONTEXT_OVERFLOW_MARKER`. |
| `test_phase5_reliability.py` | Moved whole; two local host imports → the shared helper. |
| `test_phase1_integrity.py` | Moved less one test. `test_proxy_error_strings_are_deferrable` exercised **no Roadstead code at all** — a hardcoded list run through the client's classifier — so it went to the monorepo. A comment in its place records where it went and which neighbours already assert the server half. |
| `test_timeout_apply.py` | **Deleted here — belongs to the monorepo whole.** 21 tests of `ProxyLLMClient`: extend-only policy, advice fetch, pool config. The single assertion Roadstead owns (`client < server` keepalive, with margin) was rehomed as `tests/test_keepalive_invariant.py`, and the invariant it depends on is now published in `docs/api.md` §1.3 — it was undocumented, which is why it could only be checked from the client. |

🚨 **Two things must now be confirmed present in the monorepo**, or they are lost rather than moved:
`test_proxy_error_strings_are_deferrable`, and the 20 `ProxyLLMClient` tests from
`test_timeout_apply.py`. They were deleted here on the strength of belonging there.

## Group B — fleet-coupled tests (2 files)

`test_egress_conformance.py` — loads `originfleet/agents/forum-agent/grammars/proposal.gbnf`, a grammar
file belonging to a fleet agent that does not exist here. Either vendor a representative GBNF
fixture into `tests/` (preferred — the test is about grammar conformance, not about that specific
agent) or drop it.

`test_thinker_bench.py` — drives a benchmark script that lives in the monorepo. Almost certainly
belongs there permanently, not here.

## Group C — fleet doctrine tests (6 files)

`test_tier3_serve_script_doctrine.py` · `test_inference_placement_doctrine.py`
`test_tier2_analyst_naming_doctrine.py` · `test_grammar_authority.py` · `test_endpoint_cooldown.py`
`test_thinking_option.py`

These assert facts about **the origin fleet's deployment**, not about Roadstead as software — that a
vendored vLLM launch script matches the catalog, that every host is claimed by exactly one vehicle
charter, that ≥13 production agent grammars normalise cleanly. They fail here with
`vendored tier3_env.sh missing at .../infra/anvil/...`, `expected >=13 grammars, found 0`, and
similar: the files they read belong to the monorepo.

**Most of these should go back to the monorepo permanently** — they are deployment doctrine, and
Roadstead has no opinion on which host runs what.

Two are worth salvaging in part, because a *general* version of the assertion is Roadstead's:

- `test_endpoint_cooldown.py` — the cooldown behaviour itself is Roadstead's; only the sweep over
  host agent code is not. Keep the behavioural half.
- `test_thinking_option.py` — that `apply_thinking` runs before the stream branch is a property of
  this package's own correction ordering. Rewrite without the host fixture.

`test_grammar_authority.py` needs a corpus of GBNF fixtures in `tests/` rather than a scan of a
private system's agents — same fix as `test_egress_conformance.py` in Group B.

## Definition of done

`tests/_pending/` is empty, this file is gone, and `norecursedirs` no longer mentions it. Anything
that genuinely belongs to the monorepo should be **deleted here and confirmed present there** —
not left in limbo.

**Remaining: Group B (2 files) and Group C (6 files)** — 8 of the original 14. Unlike Group A, most
of these are expected to leave rather than move: they assert facts about the origin fleet's
deployment, which Roadstead has no opinion on. The salvageable parts are named above.
