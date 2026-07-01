"""Correction — the LLMProxy's output-integrity layer (de-monolith Step 3).

Grammar validate/normalize, the shadow egress-conformance detector, the
egress degeneration (repetition-loop) guard, and the native-thinking
structured-output recovery. A near-stateless behavior object over the shared
:class:`ProxyState`; ``ProxyService`` keeps thin delegators to every method
here so its frozen private surface is unchanged (contract §2). Step 4's uniform
``Correction.apply`` entrypoint lands here later — Step 3 is pure extraction.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from .backend import BackendError, BackendUnavailable
from .config import (
    degeneration_guard_enabled,
    degeneration_shadow_only,
    normalize_endpoint,
    shadow_egress_detect_enabled,
    thinking_enabled,
    thinking_reasoning_budget,
)
from .constants import _MIN_RETRY_BUDGET_S
from .grammar import (
    grammar_hash,
    normalize_and_validate,
    recover_structured_object,
    root_object_keys,
    verify_conformance,
)
from .observability import MetricsSample

if TYPE_CHECKING:
    from .config import EndpointConfig  # noqa: F401
    from .scheduler import QueuedRequest  # noqa: F401
    from .state import ProxyState

logger = logging.getLogger(__name__)


# Empty-completion rescue (2026-06-11). Some prompts make the model emit EOS
# as its FIRST token — a 1-token "completion" with empty content, fully
# deterministic for that prompt even at high temperature (observed live: the
# sidekick craft_6 prompt failed 6/6 across temps and thinking modes; same model-
# degeneration family as the repetition loops the degeneration guard catches).
# A plain retry can never recover it. The retry therefore re-dispatches with
# ``min_tokens``, which masks EOS for the first N positions and forces the
# model past the degenerate opening — replaying the actual poisoned payload
# with min_tokens=16 produced 696 tokens of normal output. vLLM honors the
# param; llama.cpp ignores unknown fields (safe no-op). Injected ONLY on the
# retry of an empty-completion failure, never on the first attempt.
_EMPTY_RESCUE_MIN_TOKENS = 16


# --- Egress repetition-loop (degeneration) detection ------------------------
# A degenerate response loops a long n-gram many times ("… the song of ## … the
# song of ## …"). We flag it when the most-common N-word shingle BOTH repeats a
# lot AND dominates the text — two axes so a normal song chorus (repeats 2-4×,
# tiny fraction of the whole) is never flagged.
_DEGEN_GRAM = 6              # shingle length (words)
_DEGEN_MIN_WORDS = 40        # ignore short replies (a chorus/classification)
_DEGEN_MIN_REPS = 6          # the top shingle must repeat at least this many times
_DEGEN_MIN_FRACTION = 0.10   # …AND be ≥ this fraction of all shingles


def _top_shingle_reps(text: str) -> tuple[int, int]:
    """Return (max repeats of any `_DEGEN_GRAM`-word shingle, total shingles)."""
    words = (text or "").split()
    n = len(words) - _DEGEN_GRAM + 1
    if n <= 0:
        return (0, 0)
    from collections import Counter
    c = Counter(tuple(words[i:i + _DEGEN_GRAM]) for i in range(n))
    return (c.most_common(1)[0][1], n)


def _is_degenerate_text(text: str) -> bool:
    """True iff `text` is a repetition LOOP — the same long shingle repeated many
    times AND dominating the output. Conservative on both axes so legitimate
    repetition (a chorus) passes through untouched."""
    words = (text or "").split()
    if len(words) < _DEGEN_MIN_WORDS:
        return False
    reps, total = _top_shingle_reps(text)
    if total <= 0:
        return False
    return reps >= _DEGEN_MIN_REPS and (reps / total) >= _DEGEN_MIN_FRACTION


def _chat_completion_text(body: dict) -> str:
    """Extract the assistant text from a chat.completion body (or '')."""
    try:
        msg = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
        return (msg.get("content") or "")
    except Exception:  # noqa: BLE001
        return ""


class _ToolCallStreamSanitizer:
    """Per-stream sanitizer that makes vLLM ``qwen3_xml`` streaming tool-call
    deltas safe for strict OpenAI clients — the Vercel AI SDK /
    ``@ai-sdk/openai-compatible`` provider that opencode uses (Phase 5E, v2).

    THE BUG. When one turn produces MORE THAN ONE tool call (made far more
    likely by MTP speculative decoding, which the thinker runs), vLLM's
    ``qwen3_xml`` parser emits a junk "phantom" tool-call delta between the real
    ones: a fresh ``id``, ``"name": null`` and EMPTY arguments at an in-between
    index that NEVER receives a name — the real next call lands at the following
    index (observed live: real calls at index 0 and 2, phantom at 1). The AI
    SDK opens a tool-call slot for that index, finds ``function.name == null``
    (its check matches null AND undefined), and throws
    ``AI_InvalidResponseDataError: Expected 'function.name' to be a string``,
    aborting the whole turn. Upstream tracking: vLLM #39584 (open, parallel
    tool calls + spec-decode); client side: opencode #24137 / vercel/ai #6687.

    Note the v1 of this fix (strip the null ``name`` key from continuations) was
    aimed at the wrong frame: the AI SDK only validates ``function.name`` when
    OPENING a slot, never on a continuation, so stripping it there was a no-op
    against the real crash.

    The SAME parallel-call bug also corrupts the LAST call's arguments: it
    appends an extra trailing ``}`` (observed live: ``{"path": "/tmp"}}``),
    which is invalid JSON. Once the phantom no longer aborts the turn, the AI
    SDK reaches that argument and fails with a JSON parse error instead. So we
    also TRIM trailing junk: per slot we accumulate the emitted argument text
    and, the moment it parses as a complete JSON value (``raw_decode`` — which
    correctly ignores ``}`` inside string values), we emit exactly up to the end
    of that value and drop anything after. The AI SDK marks the call finished on
    the first valid parse and ignores later deltas, so this matches its model.

    THIRD DEFECT (truncation — the residual leak this version closes). When a
    big tool argument (a long ``bash`` command / heredoc) is cut off at
    ``max_tokens`` mid-JSON-string, the arguments NEVER form a complete value AND
    vLLM mislabels ``finish_reason`` as ``tool_calls`` (not ``length``) — so the
    old per-fragment passthrough emitted broken JSON the client's ``JSON.parse``
    then threw on (measured: 6/10 big-arg turns leaked through). No structural
    recovery is possible — the data is genuinely incomplete.

    THE FIX — buffer-then-emit-atomically, applied at the proxy so we neither
    patch the (custom) GB10 vLLM nor give up MTP throughput. Per (choice, index)
    slot we ACCUMULATE the argument fragments and emit NOTHING until the buffer
    forms ONE complete JSON value; then we emit the whole call (string name +
    complete args) in a single delta and ignore any trailing junk. A
    truncated/partial/broken argument therefore never reaches the client. At the
    finish chunk we finalize every still-pending slot for the choice: a named
    slot with empty args is a legitimate no-arg call (emit ``{}``); a named slot
    with INCOMPLETE args is truncated → dropped, and ``finish_reason`` is
    relabeled ``length`` so the client retries instead of dispatching a
    half-command; a name-less slot (the phantom) is dropped silently. Indices are
    NOT renumbered — the AI SDK keys tool calls by ``id`` and uses the numeric
    index only as an accumulation slot, so a dropped phantom's hole is harmless.
    (Emitting a call atomically on completion rather than streaming arg fragments
    is invisible to the AI SDK, which dispatches only once args parse anyway.)

    ``feed(data)`` takes one raw backend SSE ``data:`` payload and returns the
    payload to emit. PURE PASS-THROUGH (the ORIGINAL string object, no parse) for
    the >99% of chunks with no ``tool_calls`` while no call is mid-accumulation.
    Defensive: never raises — on any malformed/odd shape it returns the original
    bytes.
    """

    def __init__(self) -> None:
        # (choice_index, tool_index) -> {id, type, name, buf, done}
        self._slots: dict = {}

    @staticmethod
    def _complete_value(text: str):
        """If ``text`` (leading whitespace allowed) begins with a COMPLETE JSON
        value, return ``(value_str, True)`` — trimming any trailing junk such as
        the extra ``}`` the parallel-call bug appends (``raw_decode`` correctly
        ignores ``}`` inside string values). Else ``(None, False)``. An empty /
        whitespace-only buffer is NOT complete (arguments may still be
        streaming)."""
        if text.strip() == "":
            return None, False
        try:
            _, end = json.JSONDecoder().raw_decode(text)
        except ValueError:
            return None, False
        return text[:end], True

    def _pending(self) -> bool:
        """True while any slot is still accumulating — so the finish chunk and
        intervening content chunks get inspected to finalize. False in the
        steady state, which keeps ``feed`` a pure pass-through."""
        return any(not st["done"] for st in self._slots.values())

    def feed(self, data: str) -> str:
        # Fast path: only engage when this chunk carries tool_calls OR a call is
        # mid-accumulation (then we must watch for its completion + the finish
        # chunk). >99% of chunks short-circuit here untouched (original object).
        if '"tool_calls"' not in data and not self._pending():
            return data
        try:
            obj = json.loads(data)
            if not isinstance(obj, dict):
                return data
            changed = False
            for ch in (obj.get("choices") or []):
                if not isinstance(ch, dict):
                    continue
                ci = ch.get("index", 0)
                delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else None
                emit: list = []  # (idx, slot, args) — calls to emit on THIS chunk

                # 1) Buffer tool-call fragments per slot. Emit NOTHING until a
                #    slot's arguments form a COMPLETE JSON value, then emit the
                #    whole call (name + complete args) atomically. A partial /
                #    truncated / broken-JSON argument therefore NEVER reaches the
                #    client (defect B: vLLM truncates big args mid-string and
                #    mislabels finish_reason as tool_calls).
                if delta is not None and isinstance(delta.get("tool_calls"), list):
                    changed = True  # a tool_calls chunk is always rebuilt
                    for tc in delta["tool_calls"]:
                        if not isinstance(tc, dict):
                            continue
                        idx = tc.get("index")
                        st = self._slots.get((ci, idx))
                        if st is None:
                            st = {"id": None, "type": None, "name": None,
                                  "buf": "", "done": False}
                            self._slots[(ci, idx)] = st
                        if st["id"] is None and isinstance(tc.get("id"), str):
                            st["id"] = tc["id"]
                        if st["type"] is None and isinstance(tc.get("type"), str):
                            st["type"] = tc["type"]
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        rn = fn.get("name")
                        if st["name"] is None and isinstance(rn, str) and rn != "":
                            st["name"] = rn
                        ra = fn.get("arguments")
                        if isinstance(ra, str):
                            st["buf"] += ra
                        if st["done"] or st["name"] is None:
                            continue  # phantom (no name) or already emitted
                        args, ok = self._complete_value(st["buf"])
                        if ok:
                            emit.append((idx, st, args))
                            st["done"] = True

                # 2) On the finish chunk, finalize every still-pending slot for
                #    this choice: a named slot with NO args is a legitimate
                #    no-arg call (emit "{}"); a named slot with INCOMPLETE args is
                #    TRUNCATED — drop it and relabel finish_reason to "length" so
                #    the client retries instead of dispatching a half-command or
                #    throwing on broken JSON; a name-less slot is a phantom (vLLM
                #    #39584) — drop it silently.
                fr = ch.get("finish_reason")
                if fr is not None:
                    truncated = False
                    for (cci, sidx), st in self._slots.items():
                        if cci != ci or st["done"]:
                            continue
                        if st["name"] is not None and st["buf"].strip() == "":
                            emit.append((sidx, st, "{}"))
                        elif st["name"] is not None:
                            truncated = True
                        st["done"] = True
                    if truncated and fr in ("tool_calls", "stop"):
                        ch["finish_reason"] = "length"
                        changed = True

                # 3) Rebuild this choice's tool_calls delta from completed calls.
                if delta is not None and ("tool_calls" in delta or emit):
                    if emit:
                        delta["tool_calls"] = [
                            {"index": eidx, "id": est["id"],
                             "type": est["type"] or "function",
                             "function": {"name": est["name"], "arguments": eargs}}
                            for (eidx, est, eargs) in emit
                        ]
                        changed = True
                    elif "tool_calls" in delta:
                        del delta["tool_calls"]
                        changed = True
            return json.dumps(obj) if changed else data
        except Exception:  # noqa: BLE001 — a sanitizer bug must not break the stream
            return data


class Correction:
    """Output-integrity guards over the shared ProxyState."""

    def __init__(self, state: "ProxyState") -> None:
        self.state = state

    def extract_grammar(self, payload: dict) -> tuple[str | None, str | None]:
        """Return (grammar_string, location) where location is 'top' or
        'extra_body', or (None, None) if no grammar present."""
        g = payload.get("grammar")
        if isinstance(g, str) and g.strip():
            return g, "top"
        eb = payload.get("extra_body")
        if isinstance(eb, dict):
            g = eb.get("grammar")
            if isinstance(g, str) and g.strip():
                return g, "extra_body"
        return None, None
    def process_grammar(self, req: QueuedRequest) -> dict | None:
        """Validate + safe-normalize the request's grammar in place.

        Returns None on success (req.payload updated with the normalized
        grammar). Returns an error payload dict on failure — the caller
        must fail loud rather than dispatch. Results are cached by grammar
        hash; each invalid grammar is logged loudly once.
        """
        grammar, location = self.extract_grammar(req.payload)
        if grammar is None:
            return None

        h = grammar_hash(grammar)
        result = self.state.grammar_cache.get(h)
        if result is None:
            result = normalize_and_validate(grammar)
            self.state.grammar_cache[h] = result

        if not result.ok:
            if h not in self.state.grammar_alerted:
                self.state.grammar_alerted.add(h)
                logger.error(
                    "GRAMMAR INVALID — failing loud (call_site=%s endpoint=%s): %s",
                    req.call_site, req.endpoint, result.error_payload()["detail"],
                )
            return result.error_payload()

        # Write the normalized grammar back where it came from.
        if result.normalized:
            if location == "top":
                req.payload["grammar"] = result.grammar
            else:
                req.payload["extra_body"]["grammar"] = result.grammar
        return None
    def shadow_egress_detect(self, req: "QueuedRequest", result: dict) -> None:
        """WS-4: SHADOW silent-drop detector over ALL grammar-bearing responses.

        Runs ``verify_conformance`` on every grammar-bearing structured response
        and tallies per-call_site checked/dropped so ``/v1/status`` can surface a
        silent-drop rate to health→health-verifier. This closes the no-silent-failure gap
        fleet-wide: a markdown fence / non-JSON / wrong-keys response means
        llama.cpp dropped the grammar and ran free-form.

        READ-ONLY: it never mutates ``result`` (zero caller risk), and any error
        is swallowed so the detector can never break a real response."""
        try:
            if not shadow_egress_detect_enabled():
                return
            if result.get("status") != "ok":
                return
            if req.stream or req.payload_type != "chat_completion":
                return
            grammar, _loc = self.extract_grammar(req.payload)
            if not grammar:
                return
            resp = result.get("response", {}) or {}
            ch = (resp.get("choices") or [{}])[0]
            content = (ch.get("message", {}) or {}).get("content")
            if not isinstance(content, str) or not content:
                return
            ok, reason = verify_conformance(content, grammar)
            cs = req.call_site or "unknown"
            tally = self.state.shadow_drop.setdefault(cs, {"checked": 0, "dropped": 0})
            tally["checked"] += 1
            if not ok:
                tally["dropped"] += 1
                logger.warning(
                    "shadow_egress: silent grammar-drop call_site=%s endpoint=%s "
                    "reason=%s", cs, req.endpoint, reason,
                )
        except Exception:  # noqa: BLE001 — detector must never break a response
            logger.debug("shadow_egress_detect failed", exc_info=True)
    async def maybe_correct_degenerate(
        self, req: "QueuedRequest", result: dict,
    ) -> None:
        """Egress degeneration guard: if a non-streaming chat response is a
        repetition LOOP, RE-DISPATCH to the same backend with an escalating
        anti-repetition penalty and swap in the first clean result. A 200 full of
        repeated garbage is invisible to the transient-error retry and the grammar
        egress check (it's syntactically valid), so this is the layer that catches
        the failure mode that produced 8× overall=0 song-compose iterations.

        FAIL-OPEN: any error (or all re-dispatches still degenerate) leaves the
        original result untouched — the guard can never make a response worse or
        break the response path. Bounded by the caller's own remaining deadline.
        Kill-switch ``COLLECTIVE_PROXY_DEGENERATION_GUARD``; shadow (detect-only)
        ``COLLECTIVE_PROXY_DEGENERATION_SHADOW``."""
        try:
            if not degeneration_guard_enabled():
                return
            if result.get("status") != "ok" or req.payload_type != "chat_completion":
                return
            response = result.get("response")
            if not isinstance(response, dict):
                return
            text = _chat_completion_text(response)
            if not _is_degenerate_text(text):
                return

            # Flagged. Record + per-call-site tally.
            self.state.degeneration_detected += 1
            cs = req.call_site or "?"
            tally = self.state.degeneration_by_call_site.setdefault(
                cs, {"detected": 0, "recovered": 0})
            tally["detected"] += 1
            reps, total = _top_shingle_reps(text)
            logger.warning(
                "DEGENERATION detected call_site=%s endpoint=%s words=%d "
                "top_shingle=%d/%d", cs, req.endpoint, len(text.split()), reps, total)

            if degeneration_shadow_only():
                return  # measure-only phase — never re-dispatch

            ep_cfg = self.state.config.endpoints.get(req.endpoint)
            if ep_cfg is None:
                return

            # Concurrency bound (fail-open): the original slot was already
            # freed, so these extra backend calls are over-commit — cap them
            # fleet-wide rather than pile onto a backend that's likely already
            # struggling (degeneration correlates with contention).
            if self.state.degen_redispatch_inflight >= 2:
                logger.warning(
                    "degeneration re-dispatch skipped (2 already in flight) — "
                    "returning original (call_site=%s)", cs)
                self.state.degeneration_unrecovered += 1
                result["_degenerate_unrecovered"] = True
                return

            # Escalating anti-repetition penalties. vLLM honors frequency/presence
            # penalty on the OpenAI surface; a small bump breaks the loop without
            # gutting legitimate chorus repetition on the retry.
            self.state.degen_redispatch_inflight += 1
            try:
                for i, extra in enumerate(
                    ({"frequency_penalty": 0.6, "presence_penalty": 0.3},
                     {"frequency_penalty": 1.0, "presence_penalty": 0.5}), start=1,
                ):
                    remaining = req.timeout_deadline - time.monotonic()
                    if remaining < _MIN_RETRY_BUDGET_S:
                        break
                    payload = {**req.payload, **extra}
                    rt0 = time.monotonic()
                    try:
                        resp = await self.state.backend.call(
                            ep_cfg, payload, req.payload_type,
                            f"{req.request_id}-degen{i}", timeout_s=max(2.0, remaining),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "degeneration re-dispatch %d failed (call_site=%s): %s",
                            i, cs, exc)
                        self.state.metrics.record(MetricsSample(
                            timestamp=time.monotonic(), endpoint=req.endpoint,
                            agent_id=req.agent_id, priority=req.priority.name,
                            queue_wait_ms=0.0,
                            backend_latency_ms=(time.monotonic() - rt0) * 1000.0,
                            status="degen_retry", slot_seconds=time.monotonic() - rt0))
                        continue
                    # Visibility: each re-dispatch is real backend work that ran
                    # outside slot accounting — surface it in the 5-min metrics.
                    self.state.metrics.record(MetricsSample(
                        timestamp=time.monotonic(), endpoint=req.endpoint,
                        agent_id=req.agent_id, priority=req.priority.name,
                        queue_wait_ms=0.0,
                        backend_latency_ms=resp.duration_s * 1000.0,
                        status="degen_retry", slot_seconds=resp.duration_s))
                    new_text = _chat_completion_text(resp.body)
                    if new_text and not _is_degenerate_text(new_text):
                        result["response"] = resp.body
                        self.state.degeneration_recovered += 1
                        tally["recovered"] += 1
                        # Persist the TRUTH: the completion row was already
                        # written with the degenerate body before this guard
                        # ran — replace it (INSERT OR REPLACE on request_id)
                        # with what the caller actually received, so the audit
                        # corpus / health sweeps don't keep the garbage.
                        try:
                            self.state.queue_db.persist_complete(
                                req.request_id, req.agent_id, req.endpoint,
                                req.call_site, int(req.priority),
                                resp.input_tokens, resp.output_tokens,
                                resp.duration_s, 0.0, "ok",
                                payload=req.payload, response=resp.body,
                                session_id=req.session_id, turn_id=req.turn_id,
                                caller_id=req.caller_id,
                                finish_reason=resp.finish_reason,
                            )
                        except Exception:  # noqa: BLE001 — accounting must not break the response
                            logger.debug("degen corrected-row persist failed",
                                         exc_info=True)
                        logger.info(
                            "DEGENERATION recovered via re-dispatch %d "
                            "(call_site=%s frequency_penalty=%.1f)",
                            i, cs, extra["frequency_penalty"])
                        return
                # Exhausted — leave the original; flag so we don't cache the garbage.
                self.state.degeneration_unrecovered += 1
                result["_degenerate_unrecovered"] = True
                logger.warning(
                    "DEGENERATION unrecovered after re-dispatch (call_site=%s) — "
                    "returning best-effort", cs)
            finally:
                self.state.degen_redispatch_inflight -= 1
        except Exception:  # noqa: BLE001 — guard must never break a response
            logger.debug("degeneration guard failed", exc_info=True)
    def thinking_allowed_keys(self, payload: dict) -> list[str]:
        """Top-level object keys the structured constraint permits — used to
        anchor response recovery. Covers GBNF (top / extra_body /
        structured_outputs.grammar) and response_format json_schema. Empty list
        means 'no object-root structured constraint' (a plain thinking request —
        nothing to recover; content is already the clean answer)."""
        grammar, _ = self.extract_grammar(payload)
        if grammar is None:
            so = payload.get("structured_outputs")
            if isinstance(so, dict) and isinstance(so.get("grammar"), str):
                grammar = so["grammar"]
        if isinstance(grammar, str) and grammar.strip():
            try:
                return root_object_keys(grammar)
            except Exception:  # noqa: BLE001
                return []
        rf = payload.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            sch = (rf.get("json_schema") or {}).get("schema") or {}
            props = sch.get("properties")
            if isinstance(props, dict):
                return list(props.keys())
        return []
    def apply_thinking(self, req: QueuedRequest) -> None:
        """Request-side: honor a per-request ``thinking: true`` opt-in. On a vLLM
        (reasoning-parser) backend, enable native <think> and add a GENEROUS
        reasoning budget to max_tokens (reasoning is generated output → counts
        against the cap; operator directive is to prefer slowness over cutoffs).
        Records the request for response-side structured-output recovery. Strips
        the ``thinking`` control field (not a backend param) regardless. Fully
        transparent when not requested, feature-disabled, streaming, or non-vLLM."""
        p = req.payload
        if not isinstance(p, dict):
            return
        want = bool(p.get("thinking"))
        eb = p.get("extra_body")
        if isinstance(eb, dict):
            want = want or bool(eb.get("thinking"))
            eb.pop("thinking", None)
        p.pop("thinking", None)  # control field — never forward to the backend
        if not want or req.stream or req.payload_type != "chat_completion":
            return
        if not thinking_enabled():
            return
        ep = self.state.config.endpoints.get(normalize_endpoint(req.endpoint))
        engine = ep.backend_engine if ep is not None else "llama.cpp"
        if engine != "vllm":   # reasoning parser is vLLM-only; llama.cpp ignores
            return
        ck = p.get("chat_template_kwargs")
        ck = dict(ck) if isinstance(ck, dict) else {}
        ck["enable_thinking"] = True
        p["chat_template_kwargs"] = ck
        budget = thinking_reasoning_budget()
        cur = p.get("max_tokens")
        p["max_tokens"] = (cur if isinstance(cur, int) and cur > 0 else 800) + budget
        self.state.thinking_active[req.request_id] = {
            "allowed_keys": self.thinking_allowed_keys(p)}
    def finalize_thinking(self, req: QueuedRequest, result: dict) -> None:
        """Response-side normalization for an opted-in thinking request (mutates
        ``result`` in place). vLLM already splits reasoning into
        ``message.reasoning`` (content stays clean of CoT). This repairs the one
        residual artifact: the bounded stray opening-brace that PR#44142's
        one-step-deferred FSM advance leaves before the constrained object. If the
        content already parses+conforms → no-op. Else deterministically recover
        the object (clean-then-verify, never guess) and write it back. If nothing
        recovers (truncation / genuine garbage) → FAIL SAFE to caller retry rather
        than pass noise. No-op when the caller didn't opt in."""
        info = self.state.thinking_active.pop(req.request_id, None)
        if info is None or result.get("status") != "ok":
            return
        response = result.get("response")
        if not isinstance(response, dict):
            return
        try:
            ch0 = response["choices"][0]
            content = ch0["message"]["content"]
            finish = ch0.get("finish_reason")
        except Exception:  # noqa: BLE001 — not a chat.completion shape; leave untouched
            return
        if not isinstance(content, str):
            return
        self.state.thinking_requests += 1
        allowed = info.get("allowed_keys") or []
        if not allowed:
            return  # plain thinking (no object-root constraint) — content is the answer
        # Already clean + conformant?
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and set(obj.keys()) <= set(allowed):
                self.state.thinking_clean += 1
                return
        except Exception:  # noqa: BLE001
            pass
        recovered = recover_structured_object(content, allowed_keys=allowed)
        if recovered is not None:
            self.state.thinking_recovered += 1
            try:
                choices = list(response.get("choices") or [])
                c0 = dict(choices[0]); msg = dict(c0.get("message") or {})
                msg["content"] = recovered; c0["message"] = msg; choices[0] = c0
                new_resp = dict(response); new_resp["choices"] = choices
                result["response"] = new_resp
            except Exception:  # noqa: BLE001 — never break the response on a write error
                logger.exception("thinking: recovery-write failed (call_site=%s)", req.call_site)
            return
        # Unrecoverable → fail safe (caller retry / 2-call), never pass noise.
        if finish == "length":
            self.state.thinking_truncated += 1
            why = "thinking output truncated (finish=length) — raise COLLECTIVE_PROXY_THINKING_BUDGET"
        else:
            self.state.thinking_fallback += 1
            why = "thinking structured output unrecoverable"
        logger.warning("thinking egress FAIL (call_site=%s): %s — failing safe", req.call_site, why)
        result["status"] = "error"
        result["error"] = f"thinking structured-output recovery failed: {why}"
        result.pop("response", None)
    def request_is_structured(self, req: QueuedRequest) -> bool:
        """True when the request constrained its output (grammar / JSON schema /
        structured outputs), so a finish_reason=length truncation almost
        certainly produced broken/unparseable output — not a benign capped reply."""
        if req.payload_type != "chat_completion":
            return False
        p = req.payload
        if not isinstance(p, dict):
            return False
        if self.extract_grammar(p)[0]:
            return True
        if p.get("response_format") or p.get("structured_outputs"):
            return True
        eb = p.get("extra_body")
        if isinstance(eb, dict) and any(
            eb.get(k) for k in (
                "response_format", "structured_outputs",
                "guided_grammar", "guided_json", "guided_choice",
            )
        ):
            return True
        return False
    @staticmethod
    def is_transient_backend_error(exc: Exception) -> bool:
        """Infra-transient backend failures that should DEFER (retry within the
        deadline) rather than surface: unreachable/503 and an empty completion
        (a backend hiccup). A real 4xx / other-5xx is deterministic → surface.
        BackendTimeout is handled on its own branch and never reaches here."""
        if isinstance(exc, BackendUnavailable):
            return True
        if isinstance(exc, BackendError):
            return "empty completion" in (exc.detail or "")
        return False
