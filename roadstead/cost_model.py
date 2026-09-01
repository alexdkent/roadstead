"""Duration-weighted cost model with concurrency-aware calibration.

Pure computation — no I/O, no framework imports.  All state lives in
``CostModel``; persistence and EWMA seeding are the caller's job.

Cost units are **slot-seconds**: how long a request occupies one slot
on a backend endpoint.  The scheduler uses estimated cost to charge
DRR tokens up front, then retroactively adjusts when the actual
duration is known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# EWMA tracker
# ---------------------------------------------------------------------------

@dataclass
class EWMATracker:
    """Exponentially weighted moving average with variance tracking.

    ``alpha`` controls smoothness (lower = more memory, slower react).
    Typical: 0.1 for stable metrics, 0.3 for fast-adapting.
    """
    alpha: float = 0.1
    value: float = 0.0
    variance: float = 0.0
    sample_count: int = 0

    def update(self, sample: float) -> None:
        if self.sample_count == 0:
            self.value = sample
            self.variance = 0.0
        else:
            delta = sample - self.value
            self.value += self.alpha * delta
            self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta * delta)
        self.sample_count += 1

    @property
    def stddev(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    @property
    def p95(self) -> float:
        return self.value + 1.645 * self.stddev

    def to_dict(self) -> dict:
        return {
            "value": round(self.value, 6),
            "stddev": round(self.stddev, 6),
            "p95": round(self.p95, 6),
            "samples": self.sample_count,
        }


# ---------------------------------------------------------------------------
# Per-endpoint cost model
# ---------------------------------------------------------------------------

@dataclass
class EndpointCostModel:
    """Cost estimation parameters for one endpoint class.

    ``prefill_k`` is seconds-per-input-token (model-dependent).
    ``decode_tps`` is tokens-per-second indexed by occupancy [1..max_slots].
    Both are calibrated from actual observations via EWMA.
    """
    endpoint: str
    max_slots: int

    # Prefill: time = prefill_k * input_tokens
    prefill_k: float = 0.0004  # ~0.4ms per token default

    # Decode throughput at each concurrency level (1-indexed)
    # e.g. [57.0, 50.0, 43.0, 38.0] for a 4-slot endpoint
    decode_tps: list[float] = field(default_factory=list)

    # EWMA trackers per-call_site for output length estimation
    output_length_ewma: dict[str, EWMATracker] = field(default_factory=dict)

    # EWMA for prefill_k calibration
    prefill_ewma: EWMATracker = field(default_factory=lambda: EWMATracker(alpha=0.05))

    # EWMA per occupancy level for decode_tps calibration
    decode_tps_ewma: dict[int, EWMATracker] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decode_tps:
            self.decode_tps = self._default_decode_curve()

    def _default_decode_curve(self) -> list[float]:
        """Generate a default degradation curve: each additional slot
        degrades throughput by ~12%."""
        base = 40.0
        curve = []
        for i in range(self.max_slots):
            curve.append(base * (0.88 ** i))
        return curve if curve else [40.0]

    def estimate_cost_ss(
        self,
        input_tokens: int,
        max_output_tokens: int,
        call_site: str,
        current_occupancy: int,
    ) -> float:
        """Estimate slot-seconds this request will cost.

        Uses per-call_site output length EWMA when available, falling
        back to ``max_output_tokens`` (conservative upper bound).
        """
        # Prefill cost
        prefill_s = self.prefill_k * input_tokens

        # Estimate output length
        est_output = max_output_tokens
        ewma = self.output_length_ewma.get(call_site)
        if ewma and ewma.sample_count >= 5:
            est_output = min(max_output_tokens, int(ewma.p95))
            est_output = max(1, est_output)

        # Decode cost at projected occupancy
        if not self.decode_tps or self.max_slots <= 0:
            return prefill_s + est_output / 40.0
        occ = min(max(1, current_occupancy + 1), self.max_slots)
        tps = self.decode_tps[occ - 1] if occ <= len(self.decode_tps) else self.decode_tps[-1]
        decode_s = est_output / tps if tps > 0 else 30.0

        return prefill_s + decode_s

    def update_from_completion(
        self,
        call_site: str,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
        occupancy_during: int,
    ) -> None:
        """Retroactively calibrate the model from an actual completion."""
        if duration_s <= 0 or output_tokens <= 0:
            return

        # Update output length EWMA
        if call_site not in self.output_length_ewma:
            self.output_length_ewma[call_site] = EWMATracker(alpha=0.1)
        self.output_length_ewma[call_site].update(float(output_tokens))

        # No capacity curve to calibrate against (an endpoint discovery set to
        # 0 slots mid-flight clears decode_tps) — the indexing below would
        # IndexError on the empty curve when a late completion lands.
        if self.max_slots < 1 or not self.decode_tps:
            return

        # Estimate how much time was prefill vs decode
        est_prefill = self.prefill_k * input_tokens
        est_decode = max(0.01, duration_s - est_prefill)

        # Update decode_tps for the observed occupancy.
        # occupancy_during < 1 means unknown (e.g. bootstrap replay) —
        # skip decode_tps calibration to avoid corrupting the curve.
        if occupancy_during >= 1:
            observed_tps = output_tokens / est_decode
            occ = min(occupancy_during, self.max_slots)
            if occ not in self.decode_tps_ewma:
                self.decode_tps_ewma[occ] = EWMATracker(alpha=0.1)
            self.decode_tps_ewma[occ].update(observed_tps)

            if self.decode_tps_ewma[occ].sample_count >= 3:
                if occ <= len(self.decode_tps):
                    self.decode_tps[occ - 1] = self.decode_tps_ewma[occ].value

        # Update prefill_k if we have enough data
        if input_tokens > 100:
            occ_idx = min(max(occupancy_during, 1), self.max_slots)
            tps_at_occ = self.decode_tps[occ_idx - 1] if occ_idx <= len(self.decode_tps) else 40.0
            implied_decode = output_tokens / tps_at_occ if tps_at_occ > 0 else est_decode
            implied_prefill = max(0, duration_s - implied_decode)
            implied_k = implied_prefill / input_tokens if input_tokens > 0 else 0
            if implied_k > 0:
                self.prefill_ewma.update(implied_k)
                if self.prefill_ewma.sample_count >= 10:
                    self.prefill_k = self.prefill_ewma.value

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "max_slots": self.max_slots,
            "prefill_k": round(self.prefill_k, 6),
            "decode_tps": [round(t, 1) for t in self.decode_tps],
            "prefill_ewma": self.prefill_ewma.to_dict(),
            "per_call_site": {
                cs: {"p50_output_tokens": int(e.value), "samples": e.sample_count}
                for cs, e in self.output_length_ewma.items()
            },
        }


# ---------------------------------------------------------------------------
# Fleet-wide cost model
# ---------------------------------------------------------------------------

class CostModel:
    """Aggregates per-endpoint cost models for the entire fleet."""

    def __init__(self) -> None:
        self._models: dict[str, EndpointCostModel] = {}

    def register_endpoint(
        self,
        endpoint: str,
        max_slots: int,
        *,
        prefill_k: float = 0.0004,
        decode_tps: list[float] | None = None,
    ) -> EndpointCostModel:
        model = EndpointCostModel(
            endpoint=endpoint,
            max_slots=max_slots,
            prefill_k=prefill_k,
        )
        if decode_tps:
            model.decode_tps = list(decode_tps)
        self._models[endpoint] = model
        return model

    def get(self, endpoint: str) -> EndpointCostModel | None:
        return self._models.get(endpoint)

    def estimate_cost(
        self,
        endpoint: str,
        input_tokens: int,
        max_output_tokens: int,
        call_site: str,
        current_occupancy: int,
    ) -> float:
        model = self._models.get(endpoint)
        if model is None:
            return float(max_output_tokens) / 40.0
        return model.estimate_cost_ss(
            input_tokens, max_output_tokens, call_site, current_occupancy,
        )

    def record_completion(
        self,
        endpoint: str,
        call_site: str,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
        occupancy_during: int,
    ) -> None:
        model = self._models.get(endpoint)
        if model:
            model.update_from_completion(
                call_site, input_tokens, output_tokens, duration_s, occupancy_during,
            )

    def update_max_slots(self, endpoint: str, new_max: int) -> None:
        model = self._models.get(endpoint)
        if model:
            old_max = model.max_slots
            model.max_slots = new_max
            if new_max <= 0:
                model.decode_tps = []
                return
            # Extend or truncate the decode_tps curve
            if new_max > old_max:
                last = model.decode_tps[-1] if model.decode_tps else 40.0
                for _ in range(new_max - old_max):
                    model.decode_tps.append(last * 0.88)
            elif new_max < old_max:
                model.decode_tps = model.decode_tps[:new_max]

    def snapshot(self) -> dict:
        return {ep: m.to_dict() for ep, m in self._models.items()}


# ---------------------------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------------------------

#: Content-part types whose payload is BINARY, not prompt text.  Counting a
#: base64 data URI as characters would overstate est_in by megabytes.
_NON_TEXT_PART_TYPES = frozenset({
    "image", "image_url", "input_audio", "audio", "video", "file", "document",
})


def _json_chars(obj) -> int:
    """Serialized length of a structure, 0 if it cannot be serialized."""
    import json as _json

    if obj is None:
        return 0
    if isinstance(obj, str):
        return len(obj)
    try:
        return len(_json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return 0


def _content_chars(content) -> int:
    """Characters a message's ``content`` contributes to the prompt.

    Handles the three shapes the proxy actually receives: a plain string, an
    OpenAI multipart list, and an Anthropic typed-block list.  The old walk read
    ``part["text"]`` and nothing else, so an Anthropic ``tool_use`` block (whose
    payload is in ``input``) and a ``tool_result`` block (whose payload is in
    ``content``) both counted as ZERO."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, dict):
        return _json_chars(content)
    if not isinstance(content, list):
        return 0
    total = 0
    for part in content:
        if isinstance(part, str):
            total += len(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if isinstance(part.get("text"), str):
            total += len(part["text"])
        elif ptype == "tool_use":                      # Anthropic tool call
            total += len(str(part.get("name") or "")) + _json_chars(part.get("input"))
        elif ptype == "tool_result":                   # Anthropic tool reply
            total += _content_chars(part.get("content"))
        elif ptype in _NON_TEXT_PART_TYPES:
            continue
        else:                                          # unknown shape — be honest
            total += _json_chars(part)
    return total


def estimate_input_tokens(payload: dict) -> int:
    """Rough token count from a chat-completion payload.  4 chars ≈ 1
    token.  Good enough for cost estimation — we calibrate from actuals.

    🚫 DO NOT LOOSEN THE 4-CHARS/TOKEN RATIO. Measured 2026-08-20 against
    90,326 real request rows in ``/data/logs/llmproxy_requests.jsonl`` (every
    row carries BOTH ``estimated_input_tokens`` and the backend's actual
    ``input_tokens``, so this is ground truth, not a model):

        est/actual   p05 0.87 | p50 0.95 | p95 1.08 | max 1.50
        undershoots (<1.0) on 86.8% of requests
        exceeds actual at all on 13.1%; >=1.25x on 0.1%

    The estimate is systematically CONSERVATIVE in the safe direction, exactly
    as this docstring has always claimed. A raised ratio would push the
    already-undershooting 86.8% further under and let real context overflows
    reach the backend as errors instead of a clean 422.

    This was nearly "fixed" on the strength of ONE synthetic benchmark prompt
    that the context gate 422'd at an estimated 19,923 tokens against a real
    ~14,000 — a 1.42x overshoot. That prompt was built from a short English
    sentence repeated hundreds of times, which tokenizes far more efficiently
    than prose and sits at the p100 tail above. The outlier was the TEST DATA,
    not the estimator. If a false 422 is ever reported on REAL traffic, bring
    a request row showing est/actual, not a synthetic repro.

    Counts the top-level Anthropic-shaped ``system`` field (which
    ``the provider's prepare_chat_payload`` later inlines into ``messages``) and
    serialized ``tools`` schemas — both invisible to the old messages-only
    walk, which undercounted est_in for exactly the big-prompt callers
    (orchestrator tool loops) where the estimate matters most.

    D4 (2026-08-03) — the same class of undercount, one layer deeper.  A
    tool-calling transcript keeps most of its tokens OUTSIDE
    ``msg["content"]``: the assistant turn that calls a tool has
    ``content: null`` and carries the whole call in
    ``tool_calls[].function.arguments``, and the reply comes back as a separate
    tool message (OpenAI) or a ``tool_result`` block (Anthropic).  Measured live
    on one caller: est_in read 123,466 while the backend reported
    ``input_tokens`` 226,014 — a ~1.8x undercount, and it shrank BOTH the
    ``size_stretch`` and the streaming TTFT allowance
    (``_STREAM_TTFT_DEADLINE_S + est_in/1000``) on exactly the callers that need
    them most.  Now counted: OpenAI ``tool_calls`` / legacy ``function_call``,
    Anthropic ``tool_use`` / ``tool_result`` blocks, and any unrecognised
    content part (serialized rather than silently dropped).

    Never raises: a malformed message contributes what can be read and nothing
    more — this runs on the admission path of every request."""
    total_chars = 0
    system = payload.get("system")
    if isinstance(system, str):
        total_chars += len(system)
    elif isinstance(system, list):
        for part in system:
            if isinstance(part, dict):
                total_chars += len(str(part.get("text", "")))
            elif isinstance(part, str):
                total_chars += len(part)
    tools = payload.get("tools")
    if tools:
        total_chars += _json_chars(tools)
    messages = payload.get("messages") or []
    for msg in messages:
        if isinstance(msg, str):
            total_chars += len(msg)
            continue
        if not isinstance(msg, dict):
            continue
        total_chars += _content_chars(msg.get("content"))
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                fn = fn if isinstance(fn, dict) else {}
                total_chars += len(str(fn.get("name") or ""))
                # `arguments` is a JSON STRING on the wire, not an object.
                total_chars += _json_chars(fn.get("arguments"))
        fn_call = msg.get("function_call")          # legacy single-call shape
        if isinstance(fn_call, dict):
            total_chars += len(str(fn_call.get("name") or ""))
            total_chars += _json_chars(fn_call.get("arguments"))
    return max(1, total_chars // 4)
