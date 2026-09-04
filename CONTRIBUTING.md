# Contributing

## Dev setup

```sh
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

No fleet, no network and no inference backend needed — the suite runs against `roadstead.testing`'s
fake backend. `pytest -m wire_fidelity tests/wire_fidelity/` needs a real llama.cpp/vLLM server and
is deselected by default; see `tests/wire_fidelity/README.md`.

Read `CLAUDE.md` before changing anything non-trivial — it carries the concurrency invariant and the
engine-behaviour findings that explain why several things are shaped the way they are.

## Breaking changes

Breaking changes are allowed here; they must be deliberate and written down. Before changing
anything in the 🔒 **stable** column of `docs/compatibility.md` (the wire contract in `docs/api.md`:
route names, error codes, marker substrings, published constants), read that document's "How to make
a breaking change" section. In short: make the change, explain why in the commit message, add a
`CHANGELOG.md` entry under `### Breaking`, and update `docs/api.md` in the same commit if it
invalidates something there — several tests read that document back and will fail otherwise.

## The scrub rule

This repository was extracted from a private monorepo and does not accept private topology back in:
no real hostnames, no addresses outside RFC 1918-but-`_ALLOWED` / RFC 5737 documentation space, no
personal identifiers, in code, comments, docs or examples. `tests/test_scrub_sweep.py` enforces the
address half of this on every commit; there is no automated guard for a person's name or a private
hostname typed into an example, so review your own diff for that by hand before opening a PR — see
`docs/corpus_and_scrub_plan.md` for what "clean" means here and why the address-only guard is not the
whole story.

## Pull requests

Keep them scoped to one change. Add or update a test with any behavioural change — a fix without a
regression test is not considered done here. `pytest` should be green before you open the PR; CI
runs the same suite plus a packaging check that installs a built wheel into a fresh environment.
