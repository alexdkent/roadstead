# Contributing

## Dev setup

```sh
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

No fleet, no network and no inference backend needed — the suite runs against `roadstead.testing`'s
fake backend. `pytest -m wire_fidelity tests/wire_fidelity/` needs a real llama.cpp/vLLM server and
is deselected by default; see `tests/wire_fidelity/README.md`.

Read `docs/internals.md` before changing anything non-trivial — it carries the concurrency invariant and the
engine-behaviour findings that explain why several things are shaped the way they are.

## Breaking changes

Breaking changes are allowed here; they must be deliberate and written down. Before changing
anything in the 🔒 **stable** column of `docs/compatibility.md` (the wire contract in `docs/api.md`:
route names, error codes, marker substrings, published constants), read that document's "How to make
a breaking change" section. In short: make the change, explain why in the commit message, add a
`CHANGELOG.md` entry under `### Breaking`, and update `docs/api.md` in the same commit if it
invalidates something there — several tests read that document back and will fail otherwise.

## The scrub rule

This repository was extracted from somebody's private monorepo, and it does not accept private
topology or personal data back in — in code, comments, tests, docs or examples alike.

**Clean means three things:**

1. **No private IPv4 address** that is not part of this repo's sanctioned vocabulary. Use RFC 5737
   documentation space for anything you need to write down: `192.0.2.0/24`, `198.51.100.0/24`,
   `203.0.113.0/24`. Loopback, the docker ranges, `0.0.0.0` for bind-any, and `10.0.0.0/24` for an
   arbitrary address in a CIDR or ACL test are also fine. Anything else in private space is a
   straggler.
2. **No real host names, and no real organisation or product names** standing in for an example.
   Use the names the repo already uses — endpoints are `tier1`/`tier2`/`tier3`/`embed`/`rerank`,
   callers are archetypes like `coding-assistant`, `chat-assistant`, `ops` or `ingest`, and the
   example catalog in `roadstead/models.yaml` is the worked reference for both.
3. **No personal data.** No names, no email addresses, no home towns, no household members, no real
   businesses — including inside *synthesized* content. This clause is written down because it was
   learned the expensive way: a set of invented prompt fixtures turned out to be built around real
   identifiers, because those were what was to hand when somebody needed an example email to
   classify. Synthetic content built out of real identifiers is still the identifiers.

**Only the first of those is automated.** `tests/test_scrub_sweep.py` runs on every commit and
flags any private address in a tracked file that is not on its `_ALLOWED` list; if you genuinely
need a new range, add it there on purpose with a reason rather than working around the guard.

🚨 **Rules 2 and 3 are a human pass, and nothing will catch them for you.** There is no pattern to
key on for a person's name or a host name somebody typed into an example — an address sweep looks
for what the extraction was *known* to have carried, and cannot find what somebody wrote by hand.
So read your own diff for those two before you open a PR. And read the hits the sweep gives you
rather than dismissing them: a live-code path that stripped one fleet's host-name prefix survived
every automated pass and was found only because somebody read a line three lines away from a
sanctioned one.

The history matters as much as the working tree: deleting a line today leaves it one `git log -p`
away, so it is much cheaper not to commit it than to remove it afterwards.

## Pull requests

Keep them scoped to one change. Add or update a test with any behavioural change — a fix without a
regression test is not considered done here. `pytest` should be green before you open the PR; CI
runs the same suite plus a packaging check that installs a built wheel into a fresh environment.
