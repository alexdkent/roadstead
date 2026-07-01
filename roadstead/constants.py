"""Shared numeric constants for the LLMProxy service (de-monolith Step 3).

Extracted so both ``service.py`` (Lifecycle retry path) and ``correction.py``
(degeneration re-dispatch) can reference them without a circular import.
"""

# Phase 1.3 — bounded in-proxy retry for transient backend failures (defer,
# don't drop). Only retry if at least this much of the caller's deadline remains
# after a short backoff, so a retry never starts work the caller will abandon.
_MIN_RETRY_BUDGET_S = 5.0
_RETRY_BACKOFF_S = 0.5
