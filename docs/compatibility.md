# Compatibility policy

**Adopted 2026-08-31**, when Roadstead stopped being tied to its origin monorepo and stopped
pretending to be a 1:1 replacement for it.

Until then, "does it match `originfleet.llmproxy`?" was the compatibility question, and a planned
golden-oracle parity gate was going to answer it. That gate is gone: **a parity check on a
deliberate superset fails on every improvement**, so it would have measured the wrong thing and
punished the right work. This document is what replaces it.

## The rule

**Breaking changes are allowed. They must be deliberate, and they must be written down.**

Not "anything goes" — that gives callers no signal and turns every upgrade into an audit. Not
"frozen" either — that would foreclose cleanups the extraction left behind, like the retired
`creative` endpoint name. Occasional, deliberate, recorded.

## What is stable, and what is not

| | | |
|---|---|---|
| 🔒 **Stable** | The wire contract in `docs/api.md` | Request/response shapes, the error `code` values, the marker substrings, the published constants (keepalive, timeout floor, ceilings), the `/v1/*` route names. Changing any of these is a breaking change **even when the behaviour is unchanged** — §2.2 spells out why rewording an error message counts. |
| 🔓 **Internal** | Everything else | Module layout, class and function names, `roadstead.testing`'s fault list, database schema, log wording, config field names. Change freely; no entry needed unless a caller could notice. |

`roadstead.testing` is a deliberate middle case: it is *published* surface, so renaming
`FakeBackendServer` earns a CHANGELOG line, but its fault library and profiles are expected to grow
and shift as real engine behaviour is measured.

`roadstead.client` is the same shape with one extra rule: **its dependency set is contract too.**
It is what another project installs to speak the enriched API, so adding an import to it is a
breaking change for every consumer even though no signature moved — the boundary is httpx and the
stdlib, enforced by AST in `tests/test_client_sdk.py`. Its *typed views* are expected to grow as the
enriched envelope does, and every one exposes `.raw`, so a field this SDK has never heard of stays
reachable rather than being silently dropped.

Pre-1.0, none of this promises a deprecation *period*. It promises that a break is a decision
somebody made and recorded, not something you discover in production.

## How to make a breaking change

1. Check whether it is actually in the 🔒 column. Most changes are not.
2. Make it, with the reason in the commit message.
3. Add a `CHANGELOG.md` entry under `### Breaking` saying **what broke, what to do about it, and
   why it was worth it.** A migration note that only names the old and new spelling is not enough —
   the "why" is what stops it being re-litigated in six months.
4. If it invalidates something in `docs/api.md`, change that in the same commit. Six tests read
   `docs/api.md` back and will fail if you don't (`tests/test_wire_contract.py`,
   `tests/test_fleet_analytics_schema.py`, `tests/test_timeout_floor_contract.py`,
   `tests/test_keepalive_invariant.py`, `tests/test_spend.py`, `tests/test_client_sdk.py`).

That last point is the enforcement mechanism, and it is deliberate: **the contract document is
executable**. You cannot quietly drift from it, because the suite reads it.

## What about the origin monorepo?

It still runs `originfleet.llmproxy` in production, and the two will diverge. That is expected and
is not a defect on either side.

It remains useful as **evidence, not authority**. It serves real traffic across real hardware; this
repo serves a fake backend. When it produces a measurement we cannot produce here — a defect rate
over thousands of real responses, a live `/props` shape from a fleet backend, a production incident
— that is worth having, and `docs/ledger.md` is where it lands. It carries no obligation, gates
nothing, and never blocks a change here.

Nothing in this repo may depend on the monorepo being present, reachable, or in any particular
state. CI enforces the import half of that.
