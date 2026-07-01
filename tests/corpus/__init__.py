"""Versioned test corpora for the LLM-proxy Phase-T harness.

Two reusable, phase-spanning corpora live here:

  ``north_face``  — hostile/malformed CALLER requests (proxy's north face); the
                    input set the proxy must reject cleanly or serve, never crash on.
  ``schemas``     — real fleet structured-output schemas + a chat-loop slice, the
                    regression fixtures Phases 3-5 grade against.

Each module exposes plain data (lists/dicts) plus a small typed loader so tests
import fixtures by name rather than hard-coding them inline.
"""
