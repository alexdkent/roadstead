"""Runtime-mutable feature flags for the LLM proxy.

The proxy's house rule is NO env-var feature gates for anything that needs
flipping at runtime (an env change needs a container recreate, and a running
process can't be re-enved at all — which would make the scheduled
shadow→enforce auto-flips impossible). New toggles live here instead: a tiny
JSON file under the proxy's data dir, loaded at startup and mutable via
``GET/POST /v1/admin/flags`` (ACL'd, same surface as the pause/resume drain
control). The pre-existing env kill-switches (degeneration guard, thinking,
shadow-egress) are deliberately untouched — they predate this mechanism.

Single-event-loop discipline: reads are plain dict lookups (hot-path safe);
writes happen only from the admin handler, which runs ``set_many`` via
``asyncio.to_thread`` so the file I/O never blocks the loop. The file is
small and rewritten atomically (tmp + rename).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# The canonical flag set. Adding a flag = adding a default here (plus the
# consuming code). Unknown keys in the file or in a POST are rejected so a
# typo'd flag name can never silently no-op an intended flip.
DEFAULT_FLAGS: dict[str, bool] = {
    # Phase 3 — unknown-endpoint submit validation: False = shadow (WARN +
    # counter, request proceeds to rot-at-deadline as before), True = fast 404.
    "unknown_endpoint_enforce": False,
    # Vision-capability gate: False = shadow (WARN + counter, request proceeds
    # to a backend that will answer 500 "image input is not supported"), True =
    # fast 400. 🚨 KEEP THIS OFF until `vision_capability_violations` in
    # /v1/status is clean: it reads `EndpointConfig.vision`, which DEFAULTS
    # FALSE, so a models.yaml stanza that simply forgot to declare vision would
    # become an instant 400 on a working caller. The counter tells you every
    # declaration is right BEFORE the refusal is armed. Added 2026-08-20 —
    # ledger `a-role-rename-carried-vision-to-a-text-only-box`.
    "vision_capability_enforce": False,
    # Phase 6 — context-window pre-admission gate: False = shadow (WARN +
    # counter), True = fast 422 with a context-overflow marker.
    "context_gate_enforce": False,
    # Phase 4 — inject stream_options.include_usage into streaming backend
    # dispatch so streaming completions record real token counts. True by
    # default; this is the kill-switch if a backend build rejects the field.
    "inject_stream_usage": True,
    # Phase 5a — server-side smart DEFAULT deadline for callers that OMIT
    # timeout_s (the OpenAI /v1/chat/completions door + a bare /v1/submit).
    # False = the flat _DEFAULT_TIMEOUT_S (180s), byte-identical to the historical
    # default, while still recording a shadow tally (smart_default_shadow on
    # /v1/status) of what a data-driven default WOULD be. True = the timeout
    # model's class-floored, capped recommendation for this (endpoint, tier,
    # size) — so a caller that gives no deadline gets the same data-driven bound
    # framework callers already get via apply_extend_only (embed→~15s not 180s;
    # a cold on-demand load can wait out its multi-minute load). Caller-supplied
    # timeout_s ALWAYS wins regardless of this flag.
    # NB: the smart path now consumes effective_timeout_advice (load+size uplift,
    # ceiling-bounded), so flipping this on gives no-deadline callers an ADAPTIVE
    # default. Shadow shows 0.0% would-timeout on every cell — safe to flip live
    # via POST /v1/admin/flags once the surge math has soaked. Kept False here so
    # the flip is an explicit, reversible operator action, not a code-default
    # behavior change to every no-deadline caller in one push (guard-tested:
    # tests/llmproxy/test_smart_default_timeout*.py pin OFF-by-default as
    # load-bearing). ENFORCED LIVE since after the 2026-07-01 closeout via the
    # persisted runtime flag (/data/agents/llmproxy/runtime_flags.json = true,
    # shadow-validated 0.0% would-timeout / 168h) — this ship-dark default is
    # working as designed, NOT drift. See docs/llm_timeout_centralization.md.
    "smart_default_timeout": False,
}


class RuntimeFlags:
    """Load/persist the proxy's runtime feature flags.

    ``path=None`` (tests / no data dir) keeps everything in memory with the
    defaults — same behaviour, nothing persisted.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._flags: dict[str, bool] = dict(DEFAULT_FLAGS)
        self._load()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except Exception as exc:  # noqa: BLE001 — a corrupt file must not block startup
            logger.error("runtime flags load failed at %s (%s); using defaults",
                         self._path, exc)
            return
        if not isinstance(raw, dict):
            logger.error("runtime flags file %s is not a JSON object; using defaults",
                         self._path)
            return
        for key, value in raw.items():
            if key not in DEFAULT_FLAGS:
                # A removed/renamed flag lingering in the file — note + skip.
                logger.warning("runtime flags: ignoring unknown key %r in %s",
                               key, self._path)
                continue
            self._flags[key] = bool(value)
        changed = {k: v for k, v in self._flags.items() if v != DEFAULT_FLAGS[k]}
        if changed:
            logger.info("runtime flags loaded (non-default): %s", changed)

    def get(self, name: str) -> bool:
        """Current value of a flag. Raises KeyError on an unknown name so a
        consuming-code typo fails loud in tests, never silently-False in prod."""
        return self._flags[name]

    def as_dict(self) -> dict[str, bool]:
        return dict(self._flags)

    def set_many(self, updates: dict) -> dict[str, bool]:
        """Validate + apply + persist a set of flag updates.

        Raises ValueError on an unknown key or a non-bool value (the admin
        handler maps that to a 400). May run off-loop (asyncio.to_thread) —
        it touches only this object and the file, never scheduler state.
        """
        if not isinstance(updates, dict) or not updates:
            raise ValueError("expected a non-empty JSON object of flag updates")
        for key, value in updates.items():
            if key not in DEFAULT_FLAGS:
                raise ValueError(
                    f"unknown flag {key!r} (known: {sorted(DEFAULT_FLAGS)})")
            if not isinstance(value, bool):
                raise ValueError(f"flag {key!r} must be a JSON boolean")
        self._flags.update(updates)
        self._persist()
        return self.as_dict()

    def _persist(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._flags, indent=1, sort_keys=True))
            os.replace(tmp, self._path)
        except Exception as exc:  # noqa: BLE001 — an unwritable disk degrades to in-memory
            logger.error("runtime flags persist failed at %s: %s", self._path, exc)
