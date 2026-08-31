# Quarantined tests — the first real task in this repo

These fourteen files came across in the extraction but cannot run standalone yet. They are
**excluded from collection** (`norecursedirs` in `pyproject.toml`). None of them is dead — each needs
a small, well-understood change. Rewriting them is Phase 1's remaining work.

Everything else is green: **1064 passed, 1 skipped** in a clean venv with no host application
present.

## Group A — host-coupled seam tests (6 files)

`test_structured_empty_detection.py` · `test_timeout_apply.py` · `test_phase1_integrity.py`
`test_phase5_reliability.py` · `test_context_gate.py` · `test_error_taxonomy.py`

These import `originfleet.framework.*` because they assert the **contract between the proxy and its
client** from the client's side: error-envelope deferrability, the context-overflow marker string,
the client-side timeout-floor mirror, the `client < server` keepalive invariant.

The extraction plan says this class of test **stays in the monorepo** as an integration test against
the published package — and it should. But Roadstead still needs its *own* assertion of the same
contract from the server side, or the contract is only ever checked from one end. That is the
tautology trap: two values compared from a single source prove nothing.

**So do not simply delete these.** For each, split it:

- the half that asserts *what Roadstead emits* (status codes, `code` values, marker strings, floor
  values) → rewrite against `roadstead` alone and move back into `tests/`;
- the half that asserts *what the client does with it* → leave to the monorepo.

`test_structured_empty_detection.py` is the easiest and a good first one: only one of its sixteen
tests needs the host. It asserts the degradation reaches the framework's fleet-wide counter. Rewrite
it against `roadstead.hooks.set_degradation_sink()` with a local spy — which is a *better* test than
the original, because it exercises the seam this package actually owns.

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
