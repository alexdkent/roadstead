"""Capacity-aware admission control for self-hosted LLM inference fleets.

Every call a deployment makes goes through this service: deficit-round-robin
scheduling with priority bands, duration-weighted cost accounting, and
concurrency-aware admission against the capacity a backend actually has.

The previous summary described the origin deployment's role — "all LLM traffic
in <that fleet> routes through this service" — which is a fact about one
installation rather than about the package.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # The installed distribution's version — correct for a `pip install`,
    # editable or wheel, since that is what actually decides what "roadstead"
    # means on a given machine.
    __version__ = version("roadstead")
except PackageNotFoundError:
    # Imported straight off a checkout with no install at all (e.g. a
    # `sys.path` hack, or a tool that vendors the source tree) — there is no
    # distribution to ask, so this is not a real version, only a marker that
    # nothing authoritative was found.
    __version__ = "0.0.0+unknown"
