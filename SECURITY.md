# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Use GitHub's private [security
advisory](https://github.com/alexdkent/roadstead/security/advisories/new) form for this repository,
or reach the author directly through their GitHub profile (`@alexdkent`) if the advisory form is not
available to you. Include what you found, how to reproduce it, and what you think the impact is —
you do not need a fix in hand.

There is no bug bounty. Expect an acknowledgement, not a timeline promise: this is a pre-1.0 solo
project, not a funded security team.

## Supported versions

Only `main`. There are no maintained release branches yet, and no backport policy — pre-1.0, the fix
lands on `main` and that is the supported version.

## What is in scope

- **The admission/scheduling core** — DRR fairness, admission control, deadline computation, cost and
  spend accounting: anything that could be starved, over-admitted, or billed incorrectly.
- **The admin plane** (`/rs/v1/admin/*`) — the network gate (`ROADSTEAD_ADMIN_NETS`), the credential
  check, key issuance/rotation/revocation, and the audit trail. See `docs/api.md` §3.
- **The two doors** — the OpenAI-compatible surface and the enriched `/rs/v1` API: identity
  resolution (`docs/api.md` §1.5), the trusted-proxy `X-Forwarded-For` handling, and anything that
  could let one caller's identity, quota or spend leak into another's.
- **`roadstead.client`** — the SDK another project installs. A dependency it silently grew, or a
  response it trusts without validating, is a supply-chain concern for every consumer.

## What is deliberately out of scope

- **Exposing the admin plane directly to the public internet with no reverse proxy.** It is not
  designed for that: the UI (`ROADSTEAD_ADMIN_UI`) is served over plain HTTP with HTTP Basic auth and
  no TLS of its own — see the README's "Behind a reverse proxy" section and `docs/api.md` §3.7. Put a
  TLS-terminating reverse proxy in front of any deployment reachable from outside a trusted network;
  a report that this is possible without one is a documentation gap, not a vulnerability, but a
  report about a way TLS termination itself could be bypassed *is* in scope.
- Denial of service from a caller that already holds a valid `admin`-scoped key — an admin key is a
  full trust boundary by design (`docs/api.md` §3).
- Findings in `roadstead.testing`'s fake backend — it exists to simulate bad behaviour on purpose.
