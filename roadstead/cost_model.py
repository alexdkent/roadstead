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

    def degradation_factor(self, current_occupancy: int) -> float:
        """How much slower requests become when going from current to
        current+1 occupancy.  Returns a multiplier >=1.0."""
        if current_occupancy <= 0 or current_occupancy >= self.max_slots or not self.decode_tps:
            return 1.0
        curr_idx = min(current_occupancy, len(self.decode_tps)) - 1
        next_idx = min(current_occupancy + 1, len(self.decode_tps)) - 1
        curr_tps = self.decode_tps[curr_idx]
        next_tps = self.decode_tps[next_idx]
        if next_tps <= 0 or curr_tps <= 0:
            return 1.0
        return curr_tps / next_tps

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

        # Estimate how much time was prefill vs decode
        est_prefill = self.prefill_k * input_tokens
        est_decode = max(0.01, duration_s - est_prefill)

        # Update decode_tps for the observed occupancy
        observed_tps = output_tokens / est_decode
        occ = max(1, min(occupancy_during, self.max_slots))
        if occ not in self.decode_tps_ewma:
            self.decode_tps_ewma[occ] = EWMATracker(alpha=0.1)
        self.decode_tps_ewma[occ].update(observed_tps)

        # Commit calibrated values
        if self.decode_tps_ewma[occ].sample_count >= 3:
            if occ <= len(self.decode_tps):
                self.decode_tps[occ - 1] = self.decode_tps_ewma[occ].value

        # Update prefill_k if we have enough data
        if input_tokens > 100:
            # Better prefill estimate: total - (output / observed_tps_at_occ)
            tps_at_occ = self.decode_tps[occ - 1] if occ <= len(self.decode_tps) else 40.0
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

    def degradation_factor(self, endpoint: str, current_occupancy: int) -> float:
        model = self._models.get(endpoint)
        if model is None:
            return 1.0
        return model.degradation_factor(current_occupancy)

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

def estimate_input_tokens(payload: dict) -> int:
    """Rough token count from a chat-completion payload.  4 chars ≈ 1
    token.  Good enough for cost estimation — we calibrate from actuals."""
    messages = payload.get("messages") or []
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(str(part.get("text", "")))
    return max(1, total_chars // 4)
