"""Prefix-cache observability — shared core for the periodic + the CLI screen.

Two signals, both surfaced on the Inference page via `/v1/fleet/cache-stats`:

1. **Actual hit-rate** (per chat MODEL) — scraped from each vLLM backend's
   `/metrics` prefix-cache counters by the proxy periodic (see
   `backend.probe_prefix_cache`). llama.cpp backends don't expose it → `n/a`.
2. **Cache-ABILITY / misalignment** (per CALL_SITE) — the `screen()` below over
   stored prompts: sibling requests that share content (high Jaccard) but not as
   a contiguous front prefix (low LCP%) are MISALIGNED — reusable boilerplate the
   block-granular prefix cache can't capture. ROI ranks them by wasted tokens × volume.

This module is pure (no DB/proxy deps beyond the model catalog) so it's unit-testable;
`tools/cache_audit.py` is a thin CLI over it, and `service._cache_stats_iteration`
feeds it stored payloads each tick.
"""
from __future__ import annotations

import itertools
import json
import random
import statistics as st

from . import model_catalog

# Verdict thresholds (shared by the screen + tests).
MISALIGN_JACC = 0.5      # shared-content floor to be "cacheable"
MISALIGN_LCP_PCT = 40.0  # below this front-loaded %, the shared content is wasted
LOW_OVERLAP_JACC = 0.3   # below this, genuinely unique — nothing to cache


def chat_endpoint_labels() -> dict[str, str]:
    """endpoint class -> human label for the chat endpoints: the class plus the
    aliases that resolve to it, so a row reads as the names callers actually
    use. Drives the per-model rows."""
    out: dict[str, str] = {}
    try:
        cat = model_catalog.load_catalog()
        for e in cat.by_kind("chat"):
            if not e.routed:
                continue
            names = sorted(set(e.all_names) - {e.name})
            out[e.name] = "/".join([e.name, *names]) if names else e.name
    except Exception:
        pass
    return out


def prompt_text(payload_json: str) -> str:
    """Flatten a stored request payload to its prompt text. Handles OpenAI
    {messages:[{content: str|list}]}, a submit envelope {payload:{...}}, and a
    bare {prompt}. Returns "" on anything unparseable (never raises)."""
    try:
        d = json.loads(payload_json)
    except Exception:
        return ""
    if isinstance(d, dict) and isinstance(d.get("payload"), dict):
        d = d["payload"]
    if not isinstance(d, dict):
        return ""
    if isinstance(d.get("prompt"), str):
        return d["prompt"]
    parts: list[str] = []
    # The top-level `system` field is folded to a FRONT role:system message at dispatch
    # (backend.normalize_chat_payload), so it IS the front of the prompt vLLM prefix-caches.
    # Counting it (it was previously ignored) is essential: a byte-stable system + dynamic user —
    # the dominant judgment shape — otherwise reads as 0% LCP / "misaligned" when it actually caches.
    sys = d.get("system")
    if isinstance(sys, str):
        parts.append(sys)
    elif isinstance(sys, list):
        for blk in sys:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
            elif isinstance(blk, str):
                parts.append(blk)
    for m in d.get("messages") or []:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
                elif isinstance(blk, str):
                    parts.append(blk)
    return "\n".join(parts)


def _lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _jaccard(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)


def _verdict(jacc: float, lcp_pct: float) -> str:
    if jacc >= MISALIGN_JACC and lcp_pct < MISALIGN_LCP_PCT:
        return "misaligned"
    if jacc >= MISALIGN_JACC:
        return "aligned"
    if jacc < LOW_OVERLAP_JACC:
        return "low-overlap"
    return "-"


def screen(rows: list[tuple[str, str]], *, min_reqs: int = 5, sample: int = 25,
           pairs_cap: int = 80, seed: int = 0) -> list[dict]:
    """Cache-ability screen. ``rows`` = [(call_site, payload_json|text)].
    Returns one dict per call_site (≥min_reqs), ROI-descending:
    {call_site, reqs, avg_tok, lcp_pct, jacc, verdict, wasted_tokens, roi}.
    Char-level LCP + word-set Jaccard are PROXIES for vLLM's 64-token-block reuse."""
    by: dict[str, list[str]] = {}
    for call_site, payload in rows:
        t = payload if (payload[:1] not in ("{", "[")) else prompt_text(payload)
        if not t:
            t = prompt_text(payload) or payload
        if t:
            by.setdefault(call_site, []).append(t)
    rng = random.Random(seed)
    out: list[dict] = []
    for site, texts in by.items():
        if len(texts) < min_reqs:
            continue
        samp = texts if len(texts) <= sample else rng.sample(texts, sample)
        pairs = list(itertools.combinations(range(len(samp)), 2))
        rng.shuffle(pairs)
        pairs = pairs[:pairs_cap]
        if not pairs:
            continue
        avglen = st.median(len(t) for t in samp)
        mlcp = st.median(_lcp(samp[i], samp[j]) for i, j in pairs)
        mjac = st.median(_jaccard(samp[i], samp[j]) for i, j in pairs)
        lcp_pct = 100 * mlcp / max(1, avglen)
        avg_tok = avglen / 4  # rough chars→tokens
        wasted_frac = max(0.0, mjac - lcp_pct / 100)
        wasted_tokens = wasted_frac * avg_tok
        out.append({
            "call_site": site, "reqs": len(texts), "avg_tok": int(avg_tok),
            "lcp_pct": round(lcp_pct, 1), "jacc": round(mjac, 2),
            "verdict": _verdict(mjac, lcp_pct),
            "wasted_tokens": int(wasted_tokens),
            "roi": int(wasted_tokens * len(texts)),
        })
    out.sort(key=lambda r: r["roi"], reverse=True)
    return out


# Drift: a call_site whose front-loaded prefix collapsed vs its own baseline
# (someone edited a prompt and broke the cacheable prefix).
DRIFT_LCP_DROP_PTS = 25.0
PREFILL_TOK_PER_S = 250.0  # rough boxa prefill rate, for the time-saved estimate

# Tier-2 step 3 — periodic drift ALARM (health.compute_cache_stats):
CACHE_DRIFT_WINDOW_S = 48 * 3600.0   # snapshot history the alarm reads (30-min cadence → ~96 pts)
CACHE_DRIFT_REALERT_S = 6 * 3600.0   # re-fire a still-drifting call_site at most this often
# Audit 2026-07-02: 4 of 5 live drift alerts were bimodal-prompt/tiny-sample
# flapping (5-6 reqs/window oscillating 58↔83↔100). Require a real sample AND
# the drop to hold across consecutive snapshots before calling it drift.
CACHE_DRIFT_MIN_REQS = 20            # latest-screen sample floor per call_site
CACHE_DRIFT_CONSECUTIVE = 2          # drop must hold for this many latest snapshots


def detect_drift(snapshots: list[dict]) -> list[dict]:
    """Pure drift detector shared by the ``/v1/fleet/cache-stats`` payload and the
    periodic drift ALARM (Tier-2 step 3). A call_site drifts when its latest
    front-loaded-prefix share (LCP%) collapsed ≥``DRIFT_LCP_DROP_PTS`` below its
    own trailing-snapshot baseline (median LCP% over the prior snapshots) — i.e.
    a prompt edit broke the cacheable leading block.

    ``snapshots`` = ``PersistentQueue.cache_stats_snapshots()`` rows
    (oldest→newest, each carrying a parsed ``screen`` list). Returns
    ``[{call_site, endpoint, from, to}]`` — one row per drifted call_site,
    grouped by endpoint (sorted) to match the fleet payload's ordering."""
    by_ep: dict[str, list[dict]] = {}
    for s in snapshots:
        by_ep.setdefault(s["endpoint"], []).append(s)
    out: list[dict] = []
    n_consec = max(1, CACHE_DRIFT_CONSECUTIVE)
    for ep, snaps in sorted(by_ep.items()):
        if len(snaps) < 2 + n_consec:
            continue
        # The trailing n_consec snapshots must ALL show the drop; the baseline
        # is the median over everything before them (flap/tiny-sample guard,
        # audit 2026-07-02).
        recent, prior = snaps[-n_consec:], snaps[:-n_consec]
        hist: dict[str, list[float]] = {}
        for s in prior:
            for r in s.get("screen") or []:
                hist.setdefault(r["call_site"], []).append(r.get("lcp_pct", 0.0))
        recent_rows: list[dict[str, dict]] = [
            {r["call_site"]: r for r in (s.get("screen") or [])} for s in recent
        ]
        for site, r in recent_rows[-1].items():
            base = hist.get(site)
            if not base:
                continue
            if r.get("reqs", CACHE_DRIFT_MIN_REQS) < CACHE_DRIFT_MIN_REQS:
                continue  # too small a sample to call drift
            b = st.median(base)
            if b < MISALIGN_LCP_PCT:
                continue
            if all(
                site in rows and rows[site].get("lcp_pct", 0.0) <= b - DRIFT_LCP_DROP_PTS
                for rows in recent_rows
            ):
                out.append({"call_site": site, "endpoint": ep,
                            "from": round(b, 1), "to": r.get("lcp_pct", 0.0)})
    return out


def drift_alarms_to_fire(drift: list[dict], alerted: dict, now: float,
                         cooldown_s: float = CACHE_DRIFT_REALERT_S) -> tuple[list[dict], dict]:
    """Decide which drifted call_sites the periodic alarm should FIRE now, with
    dedup so a standing drift doesn't re-alert every 30-min cycle. Pure →
    unit-testable in isolation from the proxy.

    ``alerted`` maps ``(call_site, endpoint) → last_fired_wall``. A call_site
    fires on first detection and again only after ``cooldown_s`` of continuous
    drift; a call_site that CLEARS is dropped from the returned map, so a
    recurrence alerts immediately. Returns ``(to_fire, next_alerted)``."""
    to_fire: list[dict] = []
    next_alerted: dict = {}
    for d in drift:
        key = (d["call_site"], d["endpoint"])
        last = alerted.get(key)
        if last is None or (now - last) >= cooldown_s:
            to_fire.append(d)
            next_alerted[key] = now
        else:
            next_alerted[key] = last
    return to_fire, next_alerted


def build_fleet_payload(snapshots: list[dict], labels: dict[str, str],
                        engines: dict[str, str], *, top_offenders: int = 25) -> dict:
    """Assemble the `/v1/fleet/cache-stats` contract from time-ordered snapshots
    (oldest→newest) + endpoint_class label/engine maps. Pure → unit-testable.

    - per-model **actual** hit-rate: lifetime (latest cum_hits/cum_queries) +
      windowed (Δ across the snapshot window); vLLM-only, else null.
    - **offenders**: misaligned call_sites from each endpoint's latest screen, ROI-ranked.
    - **rollup**: wasted-cacheable tokens, fleet captured %, est. prefill seconds saved.
    - **trend**: per-endpoint windowed hit-rate at each consecutive snapshot pair.
    - **drift**: call_sites whose latest LCP% fell ≥DRIFT_LCP_DROP_PTS below baseline.
    """
    by_ep: dict[str, list[dict]] = {}
    for s in snapshots:
        by_ep.setdefault(s["endpoint"], []).append(s)

    models, trend, offenders = [], {}, []
    tot_wasted = 0
    fleet_dh = fleet_dq = 0
    for ep, label in sorted(labels.items()):
        snaps = by_ep.get(ep, [])
        engine = engines.get(ep, "?")
        latest = snaps[-1] if snaps else None
        actual = window = None
        if latest and latest.get("cum_queries"):
            ch, cq = latest.get("cum_hits"), latest.get("cum_queries")
            if ch is not None and cq:
                actual = round(ch / cq, 4)
        # windowed Δ over the first/last non-null-counter snapshots
        nn = [s for s in snaps if s.get("cum_queries") is not None]
        if len(nn) >= 2:
            dh = (nn[-1]["cum_hits"] or 0) - (nn[0]["cum_hits"] or 0)
            dq = (nn[-1]["cum_queries"] or 0) - (nn[0]["cum_queries"] or 0)
            if dq > 0 and dh >= 0:
                window = round(dh / dq, 4)
                fleet_dh += dh
                fleet_dq += dq
        screen = (latest or {}).get("screen") or []
        mis = [r for r in screen if r.get("verdict") == "misaligned"]
        models.append({
            "endpoint": ep, "label": label, "engine": engine,
            "actual_hit_rate": actual, "window_hit_rate": window,
            "screen_misaligned": len(mis),
            "screen_total": len(screen),
        })
        # trend: consecutive-snapshot windowed rate
        pts = []
        for a, b in zip(nn, nn[1:]):
            dq = (b["cum_queries"] or 0) - (a["cum_queries"] or 0)
            dh = (b["cum_hits"] or 0) - (a["cum_hits"] or 0)
            if dq > 0 and dh >= 0:
                pts.append({"t": b["snapshot_at"], "rate": round(dh / dq, 4)})
        if pts:
            trend[ep] = pts
        # offenders from latest screen
        for r in mis:
            offenders.append({**r, "endpoint": ep})
            tot_wasted += int(r.get("wasted_tokens", 0)) * int(r.get("reqs", 0))

    # drift: latest LCP% vs trailing baseline per call_site (shared with the
    # periodic alarm — one implementation, `detect_drift`).
    drift = detect_drift(snapshots)
    offenders.sort(key=lambda r: r.get("roi", 0), reverse=True)
    captured = round(fleet_dh / fleet_dq, 4) if fleet_dq > 0 else None
    return {
        "models": models,
        "offenders": offenders[:top_offenders],
        "rollup": {
            "wasted_cacheable_tokens": tot_wasted,
            "captured_pct": captured,
            "est_prefill_s_saved": round(fleet_dh / PREFILL_TOK_PER_S, 1) if fleet_dh else 0.0,
        },
        "trend": trend,
        "drift": drift,
        "snapshots": len(snapshots),
    }
