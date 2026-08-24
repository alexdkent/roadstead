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
    forced_reasoning_budget,
    normalize_endpoint,
    schema_backstop_enabled,
    schema_backstop_shadow,
    shadow_egress_detect_enabled,
    structured_validity_guard_enabled,
    thinking_enabled,
    thinking_reasoning_budget,
    uniform_correction_enabled,
)
from .constants import _MIN_RETRY_BUDGET_S
from .grammar import (
    grammar_hash,
    normalize_and_validate,
    recover_structured_object,
    root_object_keys,
    verify_conformance,
)
from .observability import MetricsSample, record_structured_outcome

# Phase 3 schema-repair backstop deps. json-repair recovers parseable-but-not-
# valid JSON (fences / trailing prose / trailing commas / a missing brace) before
# schema validation. GUARDED: a missing lib degrades the backstop to strict-parse
# + jsonschema only (repair becomes a no-op) and NEVER crashes the proxy — the
# import failure is logged once at first use, not at module load. jsonschema is a
# container base dep (transitive), but guard it too for symmetry / test envs.
try:
    import json_repair as _json_repair
except ImportError:  # pragma: no cover — exercised via the degraded-path test
    _json_repair = None
try:
    import jsonschema as _jsonschema
except ImportError:  # pragma: no cover
    _jsonschema = None

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


# --- Phase 3 schema-repair backstop — pure helpers --------------------------
# All are total (never raise) so the guard can stay fail-open; the orchestration
# (flags / state / persist / retry) lives on Correction.maybe_repair_schema. See
# docs/llmproxy_phase3_schema_backstop_contract.md.

def _response_tool_calls(body: dict) -> list:
    """The assistant message's tool_calls list (or [])."""
    try:
        msg = ((body.get("choices") or [{}])[0] or {}).get("message") or {}
        tc = msg.get("tool_calls")
        return tc if isinstance(tc, list) else []
    except Exception:  # noqa: BLE001
        return []


def _extract_declared_schema(payload: dict) -> dict | None:
    """The caller's declared JSON Schema, if any — the standard OpenAI
    ``response_format:{type:json_schema,json_schema:{schema:{…}}}`` shape or vLLM
    ``guided_json`` (top-level or under ``extra_body``). ``None`` when only a GBNF
    grammar or nothing is declared → validity degrades to 'parses as JSON'.
    Conservative: an unrecognized shape returns None (repair+parse only), never a
    wrong schema."""
    if not isinstance(payload, dict):
        return None
    containers = [payload]
    eb = payload.get("extra_body")
    if isinstance(eb, dict):
        containers.append(eb)
    for c in containers:
        rf = c.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            sch = (rf.get("json_schema") or {}).get("schema")
            if isinstance(sch, dict):
                return sch
        gj = c.get("guided_json")
        if isinstance(gj, dict):
            return gj
    return None


def _schema_valid(obj, schema: dict | None) -> bool:
    """True iff obj satisfies schema. No schema / no jsonschema lib / a MALFORMED
    declared schema (north-face caller bug) all → True (we cannot judge, so treat
    as 'parses-only')."""
    if schema is None or _jsonschema is None:
        return True
    validator_cls = _jsonschema.Draft7Validator
    # A broken DECLARED schema (any shape jsonschema can't compile, incl. an
    # unknown "type") is the CALLER's bug — we can't validate against it, so
    # degrade to parses-only rather than failing the response over it (north-face).
    try:
        validator_cls.check_schema(schema)
    except Exception:  # noqa: BLE001
        logger.warning("schema-backstop: caller declared an invalid JSON Schema "
                       "— validating parse-only")
        return True
    try:
        validator_cls(schema).validate(obj)
        return True
    except Exception:  # noqa: BLE001 — ValidationError / UnknownType → real miss
        return False


def _content_valid(text: str, schema: dict | None) -> bool:
    """True iff text strict-parses as JSON AND satisfies schema."""
    try:
        obj = json.loads(text)
    except Exception:  # noqa: BLE001
        return False
    return _schema_valid(obj, schema)


# --- Empty-structured-response detection ------------------------------------
# Ledger `tier3-json-object-empty-brace` (2026-08-01). `{}` is a WELL-FORMED
# JSON object, so it sails past `_is_degenerate_text` (needs ≥40 words), past
# the truncation gate (finish_reason=stop), past the empty-completion gate
# (content is non-empty), and past `enforce_structured_validity` (it parses).
# The only thing wrong with it is that it carries no ANSWER — which is a
# property of the caller's contract, not of the JSON. Hence a dedicated check.

def _schema_required_keys(schema: dict | None) -> list[str]:
    """The declared object-level ``required`` list (empty when absent/not a
    list/not an object schema). Total."""
    if not isinstance(schema, dict):
        return []
    req = schema.get("required")
    if not isinstance(req, list):
        return []
    return [k for k in req if isinstance(k, str)]


def _is_answerless_object(obj, schema: dict | None) -> bool:
    """True iff ``obj`` is a JSON OBJECT that carries no answer.

    Two shapes, in the order the incident produced them:

    1. **Zero keys** — literally ``{}``. Unconditional: no caller ever declares
       a structured contract in order to receive nothing at all. This is the
       exact 2-character document a ``disable_any_whitespace`` backend emits
       when the grammar is allowed to close immediately.
    2. **Vacuously satisfying** — every key present maps to an empty value
       (``null`` / ``""`` / ``[]`` / ``{}``), AND the caller declared NO
       ``required`` keys. The ``required`` clause is what keeps this off the
       legitimate case: an extractor that answers "I found nothing" with
       ``{"facts": []}`` against a schema that REQUIRES ``facts`` has honoured
       its contract — the field it was asked for is there, and its emptiness is
       the answer. Nothing about that is a malfunction, and it must not alarm.

    ⚠️ LIMITATION, stated plainly. When the caller declared no schema (or a
    schema with no ``required``), shape 2 is INDISTINGUISHABLE from a genuine
    "nothing found": both are an object with only empty values, produced by a
    request that asked for nothing in particular. This detector will flag such
    a response. That is not a tuning problem — it is a missing contract, and
    it is precisely what `test_structured_output_doctrine`'s empty-``required``
    clause exists to eliminate. Shape 1 has no such ambiguity.

    Total: never raises."""
    try:
        if not isinstance(obj, dict):
            return False
        if not obj:
            return True                       # shape 1: literally {}
        if _schema_required_keys(schema):
            return False                      # a real contract was met
        for v in obj.values():                # shape 2: all values empty
            if v is None or v == "" or v == [] or v == {}:
                continue
            return False
        return True
    except Exception:  # noqa: BLE001 — predicate must never raise
        return False


def _repair_json_text(text: str):
    """json-repair text → a Python object, or None if the lib is absent or the
    text is unrepairable (json_repair returns '' / raises)."""
    if _json_repair is None or not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = _json_repair.loads(text)
    except Exception:  # noqa: BLE001
        return None
    # json_repair returns "" for hopeless input — treat that as unrepairable.
    if obj == "" or obj is None:
        return None
    return obj


def _arg_str_valid(tc: dict) -> bool:
    """True iff a tool_call's function.arguments is absent/empty or valid JSON."""
    try:
        args = (tc.get("function") or {}).get("arguments")
    except Exception:  # noqa: BLE001
        return True
    if not isinstance(args, str) or not args.strip():
        return True  # absent/empty arguments is legal
    try:
        json.loads(args)
        return True
    except Exception:  # noqa: BLE001
        return False


def _clone_with_content(body: dict, new_content: str) -> dict:
    """A deep copy of body with choices[0].message.content replaced."""
    import copy
    nb = copy.deepcopy(body)
    nb["choices"][0]["message"]["content"] = new_content
    return nb


def _clone_with_repaired_args(body: dict) -> dict | None:
    """A deep copy of body with every invalid tool_call arguments string
    json-repaired to canonical JSON. None if any is unrepairable."""
    import copy
    nb = copy.deepcopy(body)
    try:
        tcs = nb["choices"][0]["message"]["tool_calls"]
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(tcs, list):
        return None
    for tc in tcs:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not isinstance(fn, dict):
            continue
        args = fn.get("arguments")
        if not isinstance(args, str) or not args.strip():
            continue
        try:
            json.loads(args)
            continue  # already valid
        except Exception:  # noqa: BLE001
            pass
        obj = _repair_json_text(args)
        if obj is None:
            return None
        fn["arguments"] = json.dumps(obj, ensure_ascii=False)
    return nb


def _conform_body(
    body: dict, schema: dict | None, expect_json_content: bool = True,
) -> tuple[str, dict | None]:
    """Classify + best-effort in-memory repair of a chat.completion body against
    the caller's JSON contract. Returns:
      ("ok", body)        — content + all tool-call args already valid (no-op)
      ("repaired", body)  — a NEW body with content/args json-repaired to valid
      ("failed", None)    — could not be made valid without a backend re-dispatch

    ``expect_json_content`` gates whether ``content`` is validated AS JSON. It is
    True only when the request actually constrained its content output (grammar /
    response_format / structured_outputs). For a PURE tool-calling request (tools
    present, no structured-output contract) the assistant ``content`` is often a
    natural-language preamble alongside ``tool_calls`` — validating that as JSON
    would false-positive; only the tool_calls arguments are checked."""
    content = _chat_completion_text(body)
    tool_calls = _response_tool_calls(body)
    content_ok = (
        _content_valid(content, schema)
        if (expect_json_content and content.strip()) else True
    )
    args_ok = all(_arg_str_valid(tc) for tc in tool_calls if isinstance(tc, dict))
    if content_ok and args_ok:
        return ("ok", body)

    new_body = None
    if not content_ok:
        obj = _repair_json_text(content)
        if obj is None or not _schema_valid(obj, schema):
            return ("failed", None)
        new_body = _clone_with_content(body, json.dumps(obj, ensure_ascii=False))
    if not args_ok:
        fixed = _clone_with_repaired_args(new_body if new_body is not None else body)
        if fixed is None:
            return ("failed", None)
        new_body = fixed
    return ("repaired", new_body if new_body is not None else body)


def _validation_error(body: dict, schema: dict | None) -> str:
    """A short human description of WHY a body failed, for the retry feedback."""
    content = _chat_completion_text(body)
    try:
        obj = json.loads(content)
    except Exception:  # noqa: BLE001
        return "the output was not valid JSON"
    if schema is not None and _jsonschema is not None:
        try:
            _jsonschema.Draft7Validator(schema).validate(obj)
        except Exception as exc:  # noqa: BLE001
            msg = str(getattr(exc, "message", exc)).splitlines()[0]
            return f"the JSON did not match the schema ({msg[:200]})"
    if any(not _arg_str_valid(tc) for tc in _response_tool_calls(body) if isinstance(tc, dict)):
        return "a tool call's arguments were not valid JSON"
    return "the output did not satisfy the required format"


def _with_error_feedback(payload: dict, error_msg: str) -> dict:
    """A deep copy of payload with a corrective user turn appended, keeping the
    original response_format/grammar so the backend re-constrains on retry."""
    import copy
    p = copy.deepcopy(payload)
    msgs = p.get("messages")
    note = (
        "Your previous response was rejected: " + error_msg + ". "
        "Return ONLY the JSON that satisfies the required format — no prose, no "
        "markdown fences, no commentary."
    )
    if isinstance(msgs, list):
        msgs.append({"role": "user", "content": note})
    return p


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

    async def apply(self, req: "QueuedRequest", result: dict) -> None:
        """Uniform non-streaming correction entrypoint (Step 4a). Runs, in the
        LOAD-BEARING order, every guard applicable to a completed sync result:

            finalize_thinking → maybe_correct_degenerate → shadow_egress_detect

        This is a behavior-preserving consolidation of the inline sequence that
        lived in ``Lifecycle.handle_sync_submit`` — SAME methods, SAME order, so
        the golden oracle stays byte-identical. The order matters (contract §2 of
        the Step-3 decomposition): the finalizers run first so the degeneration
        guard + the shadow detector + the cache all observe corrected content, and
        the degeneration guard runs before the shadow detector + cache so an
        unrecovered-degenerate flag (``_degenerate_unrecovered``) is set before
        the caller decides whether to cache. The pre-existing steps carry no flag
        gate (byte-identical to the prior inline calls); the Step-4a flag governs
        the NEW streaming coverage, and the Phase-3 ``maybe_repair_schema`` step is
        itself flag-gated (``COLLECTIVE_PROXY_SCHEMA_BACKSTOP``, default OFF ==
        byte-identical). Schema repair runs AFTER the finalizers (so it sees
        de-thought, degeneration-corrected content) and BEFORE the shadow detector
        (detect-only, last). ``enforce_structured_validity`` (operator mandate
        2026-07-11) runs at the very END so every repair layer above gets its
        chance first — it's the always-on parse-only floor that flips a still-
        malformed structured 200 to the established 502 error shape.
        ``enforce_toolcall_truncation`` must precede ``maybe_repair_schema``:
        a vLLM tool_call argument cut mid-JSON (finish mislabeled
        "tool_calls") is TRUNCATION, and json-repair would otherwise close it
        into valid-but-fabricated JSON — a silent wrong command."""
        self.finalize_thinking(req, result)
        await self.maybe_correct_degenerate(req, result)
        self.enforce_toolcall_truncation(req, result)
        await self.maybe_repair_schema(req, result)
        self.shadow_egress_detect(req, result)
        self.enforce_structured_validity(req, result)
        # LAST, and after the validity floor on purpose: a response some guard
        # already flipped to status=error is NOT a silent failure, so it is
        # neither an empty event nor a healthy denominator sample. Detect-only,
        # never mutates `result` — see detect_structured_empty.
        self.detect_structured_empty(req, result)

    def finalize_stream(
        self, req: "QueuedRequest", content: str, last_finish_reason: str | None,
    ) -> None:
        """Uniform STREAMING detection (Step 4a). The chunks already streamed
        (can't un-send), but over the reassembled assistant ``content`` we DETECT
        a degeneration loop + a silent grammar-drop and record the SAME per-call
        tallies the sync guards use — so a streaming response is no longer a
        correction blind spot. Never re-dispatches (the bytes are gone) and never
        mutates anything the client received. Gated on
        ``uniform_correction_enabled()``; FAIL-OPEN (any error is swallowed so
        detection can never break a stream). ``last_finish_reason`` is accepted for
        symmetry with the sync truncation classifier (already recorded by the
        streaming producer) and future use."""
        try:
            if not uniform_correction_enabled():
                return
            if req.payload_type != "chat_completion":
                return
            # Degeneration DETECTION (detect-only on a stream — no re-dispatch).
            if content and _is_degenerate_text(content):
                self.state.degeneration_detected += 1
                cs = req.call_site or "?"
                tally = self.state.degeneration_by_call_site.setdefault(
                    cs, {"detected": 0, "recovered": 0})
                tally["detected"] += 1
                reps, total = _top_shingle_reps(content)
                logger.warning(
                    "DEGENERATION detected (stream) call_site=%s endpoint=%s "
                    "words=%d top_shingle=%d/%d", cs, req.endpoint,
                    len(content.split()), reps, total)
            # Silent grammar-drop DETECTION over the reassembled content (respects
            # its own kill-switch, matching the sync detector).
            if shadow_egress_detect_enabled():
                self._shadow_egress_check(req, content)
            # Phase 3: schema-invalid DETECTION over the reassembled structured
            # content (detect-only — a stream can't un-send, so no repair/retry;
            # records the same signal the sync backstop would raise). Gated on the
            # backstop flag AND the uniform-correction gate above.
            if (schema_backstop_enabled() and content
                    and self.request_is_structured(req)):
                payload = req.payload if isinstance(req.payload, dict) else {}
                if not _content_valid(content, _extract_declared_schema(payload)):
                    self.state.schema_invalid_stream += 1
                    cs = req.call_site or "?"
                    tally = self.state.schema_by_call_site.setdefault(
                        cs, {"detected": 0, "repaired": 0, "retried": 0,
                             "unrecoverable": 0})
                    tally["detected"] += 1
                    logger.warning(
                        "schema-invalid structured output DETECTED (stream) "
                        "call_site=%s endpoint=%s", cs, req.endpoint)
            # Empty-structured DETECTION over the reassembled content. A stream
            # can't un-send, and this is telemetry anyway — the point is that a
            # streaming caller is not a blind spot for the RATE alarm.
            self.detect_structured_empty(req, {}, content=content, stream=True)
        except Exception:  # noqa: BLE001 — detection must never break a stream
            logger.debug("finalize_stream failed", exc_info=True)

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
            resp = result.get("response", {}) or {}
            ch = (resp.get("choices") or [{}])[0]
            content = (ch.get("message", {}) or {}).get("content")
            self._shadow_egress_check(req, content)
        except Exception:  # noqa: BLE001 — detector must never break a response
            logger.debug("shadow_egress_detect failed", exc_info=True)
    def _shadow_egress_check(self, req: "QueuedRequest", content) -> None:
        """Core silent-grammar-drop check over one response's CONTENT string:
        verify conformance against the request's grammar and tally per call_site.
        Shared by the sync path (``shadow_egress_detect``, over the response body)
        and the streaming path (``finalize_stream``, over the reassembled deltas)
        so BOTH doors measure grammar-drops identically. No-op when the request
        carried no grammar or the content is empty."""
        grammar, _loc = self.extract_grammar(req.payload)
        if not grammar:
            return
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

            # A degenerate response that hit its OUTPUT CAP (finish_reason=length)
            # virtually always re-caps on re-dispatch — the loop refills the
            # budget and the caller rejects it again (observed: the hourly
            # autonomous_chat-agent.reflect double-dispatch waste, audit 2026-07-02).
            # Count the detection but skip the pointless retries.
            try:
                fr = (response.get("choices") or [{}])[0].get("finish_reason")
            except (AttributeError, IndexError, TypeError):
                fr = None
            if fr == "length":
                logger.warning(
                    "degeneration re-dispatch skipped (finish_reason=length — a "
                    "capped loop re-caps) call_site=%s", cs)
                self.state.degeneration_unrecovered += 1
                result["_degenerate_unrecovered"] = True
                return

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
    async def maybe_repair_schema(
        self, req: "QueuedRequest", result: dict,
    ) -> None:
        """Phase 3 structured-output/tool-call reliability backstop. On a
        structured/tool SYNC response that is a 200 but whose JSON is wrong
        (trailing prose / fenced / mildly malformed / schema-invalid / bad
        ``tool_calls.arguments``), run: json-repair → schema-validate → ONE bounded
        retry (error fed back) → fail-loud deferrable. Closes the gap where output
        is parseable-but-schema-invalid (today only empty / degenerate / truncated /
        thinking-noise are rescued).

        FAIL-OPEN: any error leaves ``result`` byte-identical. Flag-gated
        (``COLLECTIVE_PROXY_SCHEMA_BACKSTOP``, default OFF); shadow
        (``…_SHADOW``) = detect + repair-in-memory + count + log, return original
        untouched. See docs/llmproxy_phase3_schema_backstop_contract.md."""
        try:
            if not schema_backstop_enabled():
                return
            if result.get("status") != "ok" or req.payload_type != "chat_completion":
                return
            response = result.get("response")
            if not isinstance(response, dict):
                return
            payload = req.payload if isinstance(req.payload, dict) else {}
            is_struct = self.request_is_structured(req)
            has_tools = bool(payload.get("tools") or payload.get("tool_choice"))
            if not (is_struct or has_tools):
                return

            schema = _extract_declared_schema(payload)
            # ⚠️ expect_json_content must be the NARROW predicate, not `is_struct`.
            # `request_is_structured()` is true for ANY constrained output, but a
            # GBNF grammar constrains to an ARBITRARY language — `root ::= "yes" |
            # "no"` legitimately emits `yes`, which is not JSON. Passing is_struct
            # here made the backstop demand JSON of it, fail repair, burn a retry,
            # and then either 502 ("produced schema-invalid structured output") or
            # substitute the retry's `{}` for the correct answer. Measured live
            # 2026-07-31: raw GBNF worked direct to llama.cpp and failed through
            # the proxy on all three tiers. `request_expects_json()` already
            # documents exactly this case (JSON-object-rooted grammars still gate).
            status, conformed = _conform_body(
                response, schema, expect_json_content=self.request_expects_json(req))
            if status == "ok":
                return  # fast path — already valid, no-op

            # The backstop WOULD fire. Record + per-call-site tally.
            self.state.schema_detected += 1
            cs = req.call_site or "?"
            tally = self.state.schema_by_call_site.setdefault(
                cs, {"detected": 0, "repaired": 0, "retried": 0, "unrecoverable": 0})
            tally["detected"] += 1
            shadow = schema_backstop_shadow()

            # (1) In-memory repair succeeded (no backend call).
            if status == "repaired":
                if shadow:
                    logger.info(
                        "schema-backstop WOULD repair in-memory (shadow) call_site=%s "
                        "endpoint=%s", cs, req.endpoint)
                    return
                result["response"] = conformed
                self.state.schema_repaired += 1
                tally["repaired"] += 1
                self._persist_corrected(req, conformed)
                logger.info(
                    "schema-backstop REPAIRED in-memory call_site=%s endpoint=%s",
                    cs, req.endpoint)
                return

            # status == "failed": in-memory repair insufficient.
            error_msg = _validation_error(response, schema)
            if shadow:
                logger.info(
                    "schema-backstop WOULD retry+fail (shadow) call_site=%s "
                    "endpoint=%s: %s", cs, req.endpoint, error_msg)
                return

            # (2) One bounded retry with the error fed back.
            retried = await self._schema_retry(req, payload, schema, error_msg, is_struct)
            if retried is not None:
                result["response"] = retried
                self.state.schema_retry_recovered += 1
                tally["retried"] += 1
                self._persist_corrected(req, retried)
                logger.info(
                    "schema-backstop RECOVERED via retry call_site=%s endpoint=%s",
                    cs, req.endpoint)
                return

            # (3) Fail loud — never hand malformed structured output to the caller.
            # Same in-band shape as thinking-fallback: status=error + drop the body
            # → handle_sync_submit returns 502 (deferrable via the client's
            # "llm proxy error 50" marker); never cached (_schema_unrecoverable,
            # honored at lifecycle alongside _degenerate_unrecovered).
            self.state.schema_unrecoverable += 1
            tally["unrecoverable"] += 1
            result["status"] = "error"
            result["error"] = (
                f"backend {req.endpoint} produced schema-invalid structured output "
                f"({error_msg}); repair + one retry failed")
            result["code"] = "schema_invalid"
            result["_schema_unrecoverable"] = True
            result.pop("response", None)
            logger.warning(
                "schema-backstop UNRECOVERABLE call_site=%s endpoint=%s: %s",
                cs, req.endpoint, error_msg)
        except Exception:  # noqa: BLE001 — backstop must never break a response
            logger.debug("schema backstop failed", exc_info=True)
    async def _schema_retry(
        self, req: "QueuedRequest", payload: dict, schema: dict | None,
        error_msg: str, expect_json_content: bool = True,
    ) -> dict | None:
        """One error-fed-back re-dispatch, bounded by fleet-wide concurrency + the
        caller's remaining deadline. Returns a conformant body (repaired if needed)
        or None (skipped / failed / still-invalid). The re-dispatch runs OUTSIDE
        slot accounting (the original slot freed when the 200 completed) — bounded
        like the degeneration re-dispatch. Never retries more than once."""
        if self.state.schema_retry_inflight >= 2:
            logger.warning(
                "schema-backstop retry skipped (2 already in flight) call_site=%s",
                req.call_site or "?")
            return None
        remaining = req.timeout_deadline - time.monotonic()
        if remaining < _MIN_RETRY_BUDGET_S:
            return None
        ep_cfg = self.state.config.endpoints.get(req.endpoint)
        if ep_cfg is None:
            return None
        self.state.schema_retry_inflight += 1
        rt0 = time.monotonic()
        try:
            retry_payload = _with_error_feedback(payload, error_msg)
            resp = await self.state.backend.call(
                ep_cfg, retry_payload, req.payload_type,
                f"{req.request_id}-schema", timeout_s=max(2.0, remaining),
            )
            self.state.metrics.record(MetricsSample(
                timestamp=time.monotonic(), endpoint=req.endpoint,
                agent_id=req.agent_id, priority=req.priority.name,
                queue_wait_ms=0.0,
                backend_latency_ms=resp.duration_s * 1000.0,
                status="schema_retry", slot_seconds=resp.duration_s))
            status, conformed = _conform_body(
                resp.body, schema, expect_json_content=expect_json_content)
            if status in ("ok", "repaired"):
                return conformed
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "schema-backstop retry dispatch failed (call_site=%s): %s",
                req.call_site or "?", exc)
            self.state.metrics.record(MetricsSample(
                timestamp=time.monotonic(), endpoint=req.endpoint,
                agent_id=req.agent_id, priority=req.priority.name,
                queue_wait_ms=0.0,
                backend_latency_ms=(time.monotonic() - rt0) * 1000.0,
                status="schema_retry", slot_seconds=time.monotonic() - rt0))
            return None
        finally:
            self.state.schema_retry_inflight -= 1
    def _persist_corrected(self, req: "QueuedRequest", body: dict) -> None:
        """Replace the already-written completion row with the corrected body
        (INSERT OR REPLACE on request_id), mirroring the degeneration guard — the
        row was persisted with the pre-repair body in _execute_sync before apply
        ran, so the audit corpus / health sweeps must see what the caller actually
        received, not the garbage. Token counts are read from the body's ``usage``
        so accounting is preserved (an in-memory repair leaves usage untouched; a
        retry's body carries the retry's usage). The corrective-dispatch latency
        lives in the ``schema_retry`` metrics sample, so duration is left 0.0 here."""
        try:
            u = body.get("usage") if isinstance(body, dict) else None
            u = u if isinstance(u, dict) else {}
            in_tok = int(u.get("prompt_tokens") or 0)
            out_tok = int(u.get("completion_tokens") or 0)
            self.state.queue_db.persist_complete(
                req.request_id, req.agent_id, req.endpoint,
                req.call_site, int(req.priority),
                in_tok, out_tok, 0.0, 0.0, "ok",
                payload=req.payload, response=body,
                session_id=req.session_id, turn_id=req.turn_id,
                caller_id=req.caller_id,
            )
        except Exception:  # noqa: BLE001 — accounting must not break the response
            logger.debug("schema corrected-row persist failed", exc_info=True)
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
    def apply_forced_reasoning_budget(self, req: QueuedRequest) -> None:
        """Request-side: an endpoint whose model ALWAYS emits a reasoning trace it
        CANNOT disable (``capabilities.reasoning=true`` — e.g. ``creative``/Trinity-Mini)
        spends max_tokens on the CoT BEFORE the answer. A small caller cap (dj-crew
        speaker turns run ~240-360) then truncates mid-reasoning → empty/partial
        content that the caller rejects. Reserve reasoning headroom ON TOP of the
        caller's answer budget so the answer still fits. Unlike ``apply_thinking``
        (an explicit, vLLM-only, non-streaming opt-in), this fires for the FORCED
        case on BOTH streaming and sync, any engine — it's a property of the
        endpoint, not a per-request flag. Transparent when the endpoint doesn't force
        reasoning or the payload carries no positive ``max_tokens`` (no cap → the
        model self-limits and there's nothing to protect)."""
        p = req.payload
        if not isinstance(p, dict) or req.payload_type != "chat_completion":
            return
        ep = self.state.config.endpoints.get(normalize_endpoint(req.endpoint))
        if ep is None or not getattr(ep, "forces_reasoning", False):
            return
        cur = p.get("max_tokens")
        if not (isinstance(cur, int) and cur > 0):
            return
        # NB: we deliberately do NOT exempt tiny caps (e.g. the dj crew/warm.py
        # max_tokens=1 warm-up ping). A forced-reasoning model spends its first
        # tokens on the CoT, so a 1-token cap returns EMPTY → the backend's
        # empty-completion path 502s and the warmer records a FALSE failure
        # (ok=0). Padding it makes the warm-up SUCCEED and, as a bonus, exercises
        # decode (a fuller warm) — the ~200 discarded tokens are cheap + rare.
        p["max_tokens"] = cur + forced_reasoning_budget()
    def apply_json_object_guard(self, req: QueuedRequest) -> None:
        """Request-side: drop a BARE ``response_format:{"type":"json_object"}`` when
        the target endpoint's backend was launched with structured-output whitespace
        BANNED (``EndpointConfig.disable_any_whitespace``, mirrored from the serve
        script — the proxy cannot introspect a launch flag).

        WHY (measured live 2026-08-01, tier3 = vLLM on anvil:9083). tier3 restarted
        2026-07-31 15:06:52 UTC with ``--structured-outputs-config
        '{"backend":"guidance","disable_any_whitespace":true}'``. That flag is
        load-bearing — without it structured output runs away emitting whitespace
        until max_tokens — but with whitespace banned, the two-character document
        ``{}`` is a legal, COMPLETE, zero-whitespace JSON object. A bare
        ``json_object`` grammar therefore lets the model close immediately, and
        greedy decoding takes it. Same prompt, same endpoint, three shapes::

            no response_format                        -> 549 chars, valid JSON
            response_format json_object (bare)        -> "{}"  (2 chars), finish=stop
            response_format json_schema (strict, req) -> 241 chars, valid JSON

        The backend even says so: ``LLMatcher error: Parser Error: token " me"
        doesn't satisfy the grammar; forced bytes: got '{'; applying ' '``.

        Downstream this is invisible: ``{}`` arrives with ``finish_reason=stop`` and
        no error, so the caller sees a well-formed response missing every field. Sidekick's
        forum-agent comment critic turned it into ``voice_match must be bool``, the
        orchestrator read that ``.error`` as transient infra and DEFERRED, and forum-agent
        re-drove the same proposal every tick — 3,100 defers, executions ~300/day -> 13.

        STRIPPING is the only generic fix. Substituting a permissive
        ``{"type":"object"}`` json_schema does NOT work: with no ``required`` keys,
        ``{}`` still satisfies it and the grammar can still close immediately. And
        the proxy cannot invent the caller's schema. Removing the constraint restores
        the measured-good shape (repro 1 above); the model still emits JSON because
        the caller's prompt asks for JSON. What the constraint was ALSO buying —
        the truncation-integrity gate, the JSON backstop's repair+parse gate, the
        structured-stream validity guard — is re-asserted via
        ``QueuedRequest.json_object_stripped`` rather than lost with it. (There is
        no declared schema to validate against either way: a bare json_object never
        carried one.)

        Deliberately NARROW — a request carrying a real constraint (``json_schema``,
        ``guided_json``, a GBNF ``grammar``, ``structured_outputs``) is left completely
        untouched, at top level and under ``extra_body``. Those shapes carry required
        keys and do not degenerate.

        This is a COMPENSATION for a caller bug, not a cure: the durable fix is each
        call site declaring a real schema. The WARNING carries the greppable marker
        ``json_object_stripped`` plus the caller identity so that migration stays
        visible instead of quietly permanent.

        Total / fail-open: never raises, never breaks a request.
        """
        try:
            p = req.payload
            if not isinstance(p, dict) or req.payload_type != "chat_completion":
                return
            ep = self.state.config.endpoints.get(normalize_endpoint(req.endpoint))
            if ep is None or not getattr(ep, "disable_any_whitespace", False):
                return

            eb = p.get("extra_body")
            eb = eb if isinstance(eb, dict) else None

            # Any OTHER structured constraint anywhere in the payload -> hands off.
            # The caller pinned a real grammar; json_object is not what is binding.
            for container in (p, eb):
                if not isinstance(container, dict):
                    continue
                for key in ("grammar", "guided_grammar", "guided_json",
                            "guided_choice", "guided_regex", "structured_outputs"):
                    if container.get(key):
                        return

            def _is_bare_json_object(rf: object) -> bool:
                if not isinstance(rf, dict):
                    return False
                if str(rf.get("type") or "").strip() != "json_object":
                    return False
                # A json_object that somehow also carries a schema is not bare —
                # leave it alone rather than guess which half the backend honors.
                return not any(rf.get(k) for k in ("schema", "json_schema"))

            targets = [c for c in (p, eb)
                       if isinstance(c, dict) and _is_bare_json_object(c.get("response_format"))]
            # A json_schema in EITHER position means the request is really schema-
            # constrained; don't strip the redundant json_object sibling.
            for container in (p, eb):
                if isinstance(container, dict):
                    rf = container.get("response_format")
                    if isinstance(rf, dict) and str(rf.get("type") or "") == "json_schema":
                        return
            if not targets:
                return
            for container in targets:
                container.pop("response_format", None)
            # Re-assert what the constraint was also buying. The payload no longer
            # looks structured, but the caller still parses JSON — so keep the
            # truncation-integrity gate, the JSON backstop and the structured-
            # stream validity guard armed via this flag. (CLAUDE.md: "replacing a
            # component silently drops its guarantees" — enumerate and re-assert.)
            try:
                req.json_object_stripped = True
            except Exception:  # noqa: BLE001 — a mock/namespace req must not break
                pass
            logger.warning(
                "json_object_stripped: bare response_format json_object removed for "
                "endpoint=%s agent=%s call_site=%s request_id=%s — this backend runs "
                "disable_any_whitespace, where '{}' is a legal complete object and "
                "becomes the greedy path (returns '{}' with finish_reason=stop). "
                "MIGRATE THIS CALL SITE to response_format json_schema with required "
                "fields; the strip is a proxy-side compensation, not a fix.",
                req.endpoint, getattr(req, "agent_id", "?"),
                getattr(req, "call_site", "?"), getattr(req, "request_id", "?"),
            )
        except Exception:  # noqa: BLE001 — a compensation must never break a request
            logger.debug("json_object guard failed", exc_info=True)
    def apply_thinking(self, req: QueuedRequest) -> None:
        """Request-side: honor a per-request ``thinking:`` opt-in. On an endpoint
        whose model DECLARES its thinking switch (``policy.thinking_kwargs`` in
        models.yaml), set that switch and add reasoning headroom to max_tokens
        (reasoning is generated output → counts against the cap; operator
        directive is to prefer slowness over cutoffs). Records the request for
        response-side structured-output recovery. Strips the ``thinking``
        control field (not a backend param) regardless. Fully transparent when
        not requested, feature-disabled, or undeclared.

        THE FIELD TAKES TWO SHAPES. ``thinking: true`` → the flat
        ``thinking_reasoning_budget()`` headroom (8000), which is right for a
        long-form author and far too much for a chat turn. ``thinking: <int>``
        → that many tokens of headroom instead, capped at the flat default so
        the field can only ever ask for LESS. An interactive caller should send
        the int; see the budget block below for the measured reason.

        WHICH KEY IS SET is per-model, not per-engine — the switch is a chat
        TEMPLATE variable. DeepSeek-V4 (tier3) reads ``thinking``; Qwen3.6/3.8
        (tier2-chat/tier2-analyst) read ``enable_thinking``. Hardcoding one
        family's spelling here is the defect this method used to carry.

        STREAMING (2026-08-02): the request side applies to streamed requests too.
        It used to bail on ``req.stream`` outright, which made ``thinking: true``
        a SILENT no-op for any streaming caller — no error, just a non-thinking
        answer, which is exactly the failure shape that reads as "the model didn't
        reason today". The bail was only ever needed for the RESPONSE side:
        :meth:`finalize_thinking` rewrites a complete result dict and cannot run
        over an SSE stream. So a streamed request gets the enable + budget
        and is deliberately NOT registered in ``state.thinking_active`` — nothing
        will finalize it, and an un-popped entry would leak. A streaming caller
        that also wants structured output therefore gets vLLM's raw framing (the
        bounded stray-brace artifact is not repaired); the Playground, the only
        streaming consumer of the opt-in today, reads reasoning off the
        ``delta.reasoning_content`` channel and does not parse JSON."""
        p = req.payload
        if not isinstance(p, dict):
            return
        want = p.get("thinking")
        eb = p.get("extra_body")
        if isinstance(eb, dict):
            want = want if want else eb.get("thinking")
            eb.pop("thinking", None)
        p.pop("thinking", None)  # control field — never forward to the backend
        if not want or req.payload_type != "chat_completion":
            return
        if not thinking_enabled():
            return
        ep = self.state.config.endpoints.get(normalize_endpoint(req.endpoint))
        keys = tuple(ep.thinking_kwargs) if ep is not None else ()
        if not keys:
            # Undeclared template → we do not know which variable switches
            # reasoning on this model, and guessing is how the tier3 swap went
            # unnoticed. Refusing here is a no-op for the caller, same as the
            # old `engine != "vllm"` bail, but it now says WHY and it is fixed
            # by one line in models.yaml rather than by an engine check that
            # was never the real question.
            #
            # 🚨 The old bail was `engine != "vllm"`, and that was WRONG, not
            # merely conservative: it made `thinking: true` a silent no-op on
            # BOTH llama.cpp chat endpoints even though both demonstrably
            # separate reasoning into their own field (measured 2026-08-24:
            # tier2-analyst 2,913 chars, tier2-chat 2,572 chars, `content`
            # clean in both). The reasoning PARSER being a vLLM concept is
            # true; "llama.cpp cannot do this" was not.
            return
        ck = p.get("chat_template_kwargs")
        ck = dict(ck) if isinstance(ck, dict) else {}
        for key in keys:
            ck[key] = True
        p["chat_template_kwargs"] = ck
        # HOW MUCH REASONING HEADROOM. `thinking: true` keeps the historic flat
        # allowance; `thinking: <int>` asks for exactly that many tokens.
        #
        # 🔑 The int form exists because the flat 8000 is the wrong number for a
        # CHAT caller, and getting it wrong is expensive rather than merely
        # untidy. The model spends the budget it is given: measured on the song
        # author, `thinking: true` on a ~60-token ask produced 5,576 completion
        # tokens over 443.6s, against 3.0s with thinking off. The same open-ended
        # prompt sent with the caller's own max_tokens and no inflation reasoned
        # for 556 chars and answered in 11.0s — FASTER than the 21.9s the
        # non-thinking arm took to write the same deliberation out as prose.
        # So an interactive caller wants a small explicit headroom, a long-form
        # author wants the generous default, and the proxy cannot tell which is
        # which from the payload. Let the caller say.
        budget = thinking_reasoning_budget()
        if isinstance(want, int) and not isinstance(want, bool) and want > 0:
            budget = min(want, budget)
        cur = p.get("max_tokens")
        p["max_tokens"] = (cur if isinstance(cur, int) and cur > 0 else 800) + budget
        if req.stream:
            # No finalize_thinking runs over a stream — registering here would
            # leak an entry in thinking_active that nothing ever pops.
            return
        self.state.thinking_active[req.request_id] = {
            "allowed_keys": self.thinking_allowed_keys(p)}

    # =====================================================================
    # DELETED 2026-08-22: fold_system_for_thinking. It merged the system
    # message into the first user turn to work around a Laguna (vLLM/tier3)
    # chat-template quirk where reasoning only fired on a first turn with no
    # system role present (measured 2026-07-31/08-02: no-system 4/4 reasoned,
    # system+folded 1/4). tier3 is now DeepSeek-V4-Flash-0731, a hybrid
    # thinking model that gates reasoning on an explicit
    # `chat_template_kwargs {"thinking": true}` switch rather than on learned
    # suppression. Re-measured all four cells live (N=6, thinking:true):
    #   A no-system 6/6 · B system+user unfolded 6/6 · C folded 6/6 ·
    #   D multi-turn history 6/6. No gap for the fold to close, so it goes —
    #   carrying it forward would have kept reshaping every opted-in request
    #   for no measured benefit. `finalize_thinking`, `thinking_allowed_keys`,
    #   `thinking_noop` and the Playground zero-reasoning warning are
    #   independent of this and stay.
    # =====================================================================
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
        # LOUD-ify the silent case: thinking was requested and applied, but the
        # backend returned NO reasoning at all. That is exactly how the laguna
        # system-message template bug hid for its whole life — the responses were
        # schema-conformant, so `thinking_clean` ticked up and nothing looked
        # wrong. A capability that silently does nothing is the failure mode here.
        try:
            msg = ch0.get("message") or {}
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        except Exception:  # noqa: BLE001
            reasoning = ""
        if not (isinstance(reasoning, str) and reasoning.strip()):
            self.state.thinking_noop += 1
            logger.warning(
                "thinking NO-OP (call_site=%s, endpoint=%s): thinking was requested "
                "and applied but the response carried NO reasoning — the backend "
                "emitted none despite enable_thinking (the opt-in did nothing here)",
                req.call_site, getattr(req, "endpoint", "?"))
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
    def request_expects_json(self, req: QueuedRequest) -> bool:
        """True when the request's structured constraint implies the CONTENT
        must be a parseable JSON value: ``response_format`` json_object/
        json_schema, ``guided_json``, or a JSON-OBJECT-rooted GBNF grammar
        (top-level / ``extra_body`` / ``structured_outputs``). Deliberately
        NARROWER than :meth:`request_is_structured`: ``guided_choice`` and
        bare-token grammars (``root ::= "yes" | "no"``) legitimately emit
        non-JSON output and must never be parse-gated — only truncation-gated."""
        if req.payload_type != "chat_completion":
            return False
        p = req.payload
        if not isinstance(p, dict):
            return False
        # Stripped bare json_object still implies JSON content (same reasoning as
        # in request_is_structured) — keep the parse gate armed.
        if getattr(req, "json_object_stripped", False):
            return True
        containers = [p]
        eb = p.get("extra_body")
        if isinstance(eb, dict):
            containers.append(eb)
        for c in containers:
            rf = c.get("response_format")
            if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
                return True
            if isinstance(c.get("guided_json"), dict):
                return True
        grammar, _loc = self.extract_grammar(p)
        if grammar is None:
            so = p.get("structured_outputs")
            if isinstance(so, dict) and isinstance(so.get("grammar"), str):
                grammar = so["grammar"]
        if isinstance(grammar, str) and grammar.strip():
            try:
                # Object-rooted grammar ⇒ the output is a JSON object. A grammar
                # with no derivable root keys could still be JSON, but we can't
                # know — fail SAFE (no parse gate) rather than 502 a legitimate
                # bare-token reply.
                return bool(root_object_keys(grammar))
            except Exception:  # noqa: BLE001 — predicate must never raise
                return False
        return False

    def record_truncation_event(
        self, req: "QueuedRequest", *, structured: bool, stream: bool,
        output_tokens: int, status: str,
    ) -> None:
        """Loud ERROR + per-(model, caller) tally for an output-cap hit
        (operator mandate 2026-07-11 — truncation must never pass silently to
        ANY caller). Called from the single completion choke point
        (``Lifecycle.record_completion``, finish_reason=="length" for BOTH
        response modes) and from the sync tool-call truncation rule (vLLM
        mislabels that finish as "tool_calls"). The ``LLMPROXY_TRUNCATION``
        marker is stable — log_scan / health-verifier grep for it. Synchronous +
        allocation-light (one small dict row per (model, caller))."""
        p = req.payload if isinstance(req.payload, dict) else {}
        max_tokens = int(p.get("max_tokens") or 0)
        # Prefix-cache warmers deliberately request a 1-token (gemma: 16) freetext
        # completion to keep the composer prefix hot (orchestrator warm_prefix,
        # dj crew warm probe). finish_reason=length on such a probe is EXPECTED —
        # the caller never wanted more than a token — so it's not a caller-visible
        # cut-off. Tallying/logging these buries the real signal (513:67 over the
        # 7-day log) and fires a benign ERROR every ~90s. A real structured call
        # never asks for ≤16 tokens, so scope the exemption to tiny freetext.
        if not structured and 0 < max_tokens <= 16:
            return
        key = f"{req.endpoint}|{req.agent_id}"
        tally = self.state.truncation_by_model_caller.setdefault(
            key, {"count": 0, "structured": 0, "freetext": 0})
        tally["count"] += 1
        tally["structured" if structured else "freetext"] += 1
        self.state.truncation_total += 1
        logger.error(
            "LLMPROXY_TRUNCATION model=%s agent=%s call_site=%s priority=%s "
            "max_tokens=%s output_tokens=%d structured=%s stream=%s status=%s",
            req.endpoint, req.agent_id, req.call_site, req.priority.name,
            max_tokens, output_tokens, structured, stream, status)

    def enforce_toolcall_truncation(self, req: "QueuedRequest", result: dict) -> None:
        """SYNC mirror of the stream sanitizer's finalize rule (operator mandate
        2026-07-11). vLLM mislabels a max_tokens truncation mid-tool-call as
        ``finish_reason="tool_calls"`` (see ``_ToolCallStreamSanitizer``, defect
        THREE), so it bypasses the Phase-1.1 length check — and worse, the
        schema backstop's json-repair can close the cut-off argument string into
        VALID-but-FABRICATED JSON (a silent wrong command). A tool_call whose
        ``function.arguments`` does not parse on a vLLM sync 200 is therefore
        TRUNCATION, never a repair target: fail loud with the pinned deferrable
        "truncated structured output" shape so the caller retries. Runs BEFORE
        ``maybe_repair_schema`` in :meth:`apply` — order is load-bearing.
        vLLM-only: llama.cpp labels truncation "length" correctly. FAIL-OPEN;
        kill-switch ``COLLECTIVE_PROXY_STRUCTURED_VALIDITY``."""
        try:
            if not structured_validity_guard_enabled():
                return
            if result.get("status") != "ok" or req.payload_type != "chat_completion":
                return
            response = result.get("response")
            if not isinstance(response, dict):
                return
            ep = self.state.config.endpoints.get(normalize_endpoint(req.endpoint))
            if ep is None or ep.backend_engine != "vllm":
                return
            tool_calls = _response_tool_calls(response)
            if not tool_calls or all(
                _arg_str_valid(tc) for tc in tool_calls if isinstance(tc, dict)
            ):
                return
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            out_tok = int(usage.get("completion_tokens") or 0)
            self.record_truncation_event(
                req, structured=True, stream=False,
                output_tokens=out_tok, status="truncated")
            result["status"] = "error"
            # Keep the pinned deferrable marker ("truncated structured output",
            # framework/nexus_errors.py) so existing client retry/re-chunk
            # classification engages unchanged.
            result["error"] = (
                f"backend {req.endpoint} truncated structured output "
                f"(finish_reason=tool_calls mislabel — tool_call arguments cut "
                f"mid-JSON, output_tokens={out_tok})")
            result["code"] = "toolcall_truncated"
            result.pop("response", None)
        except Exception:  # noqa: BLE001 — the rule must never break a response
            logger.debug("toolcall truncation rule failed", exc_info=True)

    def record_structured_parse_failure(
        self, req: "QueuedRequest", *, stream: bool, output_tokens: int,
    ) -> None:
        """Tally + loud ERROR for a structured response whose content failed
        ``json.loads`` (operator mandate 2026-07-11). Shared by the sync guard
        (:meth:`enforce_structured_validity`) and the streaming end-of-stream
        guard in ``Lifecycle.execute_streaming`` so both doors count + log
        identically. The ``LLMPROXY_STRUCTURED_INVALID`` marker is stable —
        log_scan / health-verifier grep for it."""
        key = f"{req.endpoint}|{req.agent_id}"
        st = self.state
        st.structured_parse_failure_total += 1
        st.structured_parse_failures_by_model_caller[key] = (
            st.structured_parse_failures_by_model_caller.get(key, 0) + 1)
        p = req.payload if isinstance(req.payload, dict) else {}
        logger.error(
            "LLMPROXY_STRUCTURED_INVALID model=%s agent=%s call_site=%s "
            "priority=%s max_tokens=%s output_tokens=%d stream=%s — structured "
            "response content is not valid JSON",
            req.endpoint, req.agent_id, req.call_site, req.priority.name,
            p.get("max_tokens", 0), output_tokens, stream)

    def enforce_structured_validity(self, req: "QueuedRequest", result: dict) -> None:
        """Operator-mandated ALWAYS-ON floor (2026-07-11): a JSON-implying
        structured request must never return status=ok with content that fails
        ``json.loads``. Runs LAST in :meth:`apply`, after every repair layer
        (thinking-finalize / degeneration / schema-backstop) had its chance, so
        it only fires on what nothing recovered. Parse-only — schema conformance
        is NOT checked (backends enforce grammar; the backstop owns schemas);
        this catches truncation/malformation. On failure it flips the result to
        the SAME status=error shape every other egress guard uses, so
        ``handle_sync_submit`` returns the established 502 and existing client
        deferral handling engages unchanged.

        Hot-path notes: synchronous, no awaits; ``json.loads`` runs on content
        bounded by the request's ``max_tokens`` cap, and only for structured
        requests whose content survived every prior guard — an acceptable,
        rare, bounded cost on the single event loop. FAIL-OPEN on internal
        errors. Kill-switch ``COLLECTIVE_PROXY_STRUCTURED_VALIDITY``."""
        try:
            if not structured_validity_guard_enabled():
                return
            if result.get("status") != "ok" or req.payload_type != "chat_completion":
                return
            if not self.request_expects_json(req):
                return
            response = result.get("response")
            if not isinstance(response, dict):
                return
            content = _chat_completion_text(response)
            if not isinstance(content, str) or not content.strip():
                # Empty content = tool-calls-only response or the backend
                # empty-completion gate's domain — nothing to parse-gate here.
                return
            try:
                json.loads(content)
                return
            except ValueError:
                pass
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            out_tok = int(usage.get("completion_tokens") or 0)
            self.record_structured_parse_failure(req, stream=False, output_tokens=out_tok)
            result["status"] = "error"
            result["error"] = (
                f"backend {req.endpoint} returned invalid JSON for a structured "
                f"request (content does not parse; output_tokens={out_tok})")
            result["code"] = "structured_invalid_json"
            result.pop("response", None)
        except Exception:  # noqa: BLE001 — the floor must never break a response
            logger.debug("structured validity guard failed", exc_info=True)

    def detect_structured_empty(
        self, req: "QueuedRequest", result: dict, *,
        content: str | None = None, stream: bool = False,
    ) -> None:
        """DETECT (never fix) a structured response that carries no answer —
        ledger ``tier3-json-object-empty-brace``, 2026-08-01.

        On 2026-07-31 15:06 UTC tier3 restarted with ``disable_any_whitespace``
        and every bare-``json_object`` request began returning the two
        characters ``{}``: valid JSON, ``finish_reason=stop``, no error field,
        no exception. The proxy held the evidence for 31 hours and said
        nothing, because ``{}`` is WELL-FORMED — it passes the degeneration
        check, the truncation check, the empty-completion check and the
        structured-validity floor. Sidekick's forum-agent critic parsed it, found no
        ``voice_match``, returned an error; the auto-approve orchestrator read
        that error as transient infra and deferred; forum-agent re-drove the same
        proposal every tick. 3,100 defers, forum output ~300/day → 13, found by
        a human saying "forum-agent seems quiet".

        This is TELEMETRY, NOT A GATE. It never raises, never mutates
        ``result``, never rewrites content and — deliberately — never retries.
        A retry here would turn a silent failure into an expensive silent
        failure: the condition is a backend launch flag, so every retry
        re-earns the same ``{}`` at full cost. The operator surfaces are:

          * the greppable WARNING marker ``LLMPROXY_STRUCTURED_EMPTY``;
          * ``framework.observability.degradation`` (component=``llmproxy``,
            reason=``structured_empty``) — the same seam the comment critic
            uses, so it also lands on the fleet-wide counter;
          * per-endpoint RATE in the sliding window, which becomes the standing
            ``structured_empty_rate`` alert on ``/v1/status.alerts``. A log line
            nobody greps is not detection; the rate is the actual alarm.

        Covers BOTH response modes: ``apply`` passes the sync body, and
        ``finalize_stream`` passes the reassembled stream content."""
        try:
            if req.payload_type != "chat_completion":
                return
            if not stream and result.get("status") != "ok":
                # A response some earlier guard already failed loud is not a
                # SILENT failure — it has an error the caller can see.
                return
            if not self.request_is_structured(req):
                return
            if content is None:
                response = result.get("response")
                if not isinstance(response, dict):
                    return
                content = _chat_completion_text(response)
            if not isinstance(content, str) or not content.strip():
                # Genuinely empty content is the empty-completion gate's
                # domain (it already fails loud) — not this detector's.
                return
            try:
                obj = json.loads(content)
            except ValueError:
                return  # not parseable → enforce_structured_validity's domain
            payload = req.payload if isinstance(req.payload, dict) else {}
            schema = _extract_declared_schema(payload)
            st = self.state
            answerless = _is_answerless_object(obj, schema)
            if answerless and schema is not None and not _schema_valid(obj, schema):
                # It fails the declared schema outright — the schema backstop
                # owns that, and double-counting it here would let a genuine
                # schema miss inflate the empty RATE. Not a denominator sample
                # either: it is not a healthy response.
                return
            if not answerless:
                # The DENOMINATOR. Without healthy samples the rate is a bare
                # count and cannot tell "one odd call" from "this endpoint
                # returns nothing any more".
                record_structured_outcome(
                    st.structured_empty_window, req.endpoint,
                    empty=False, call_site=req.call_site or "?",
                    now=time.monotonic())
                return

            cs = req.call_site or "?"
            st.structured_empty_total += 1
            st.structured_empty_by_call_site[cs] = (
                st.structured_empty_by_call_site.get(cs, 0) + 1)
            record_structured_outcome(
                st.structured_empty_window, req.endpoint,
                empty=True, call_site=cs, now=time.monotonic())
            n_keys = len(obj) if isinstance(obj, dict) else 0
            logger.warning(
                "LLMPROXY_STRUCTURED_EMPTY model=%s agent=%s call_site=%s "
                "request_id=%s stream=%s declared_schema=%s required=%s "
                "keys=%d content=%r — a STRUCTURED request returned a "
                "well-formed JSON object with no answer in it. This is what a "
                "backend running disable_any_whitespace emits when the grammar "
                "is allowed to close immediately; it arrives with "
                "finish_reason=stop and no error, so every caller-side health "
                "signal reads it as success.",
                req.endpoint, getattr(req, "agent_id", "?"), cs,
                getattr(req, "request_id", "?"), stream,
                schema is not None, _schema_required_keys(schema),
                n_keys, content[:120],
            )
            try:
                from originfleet.framework.observability import degradation
                degradation(
                    component="llmproxy", reason="structured_empty",
                    impact="caller received a well-formed response with no "
                           "fields; it cannot tell this from a real answer",
                    model=req.endpoint,
                    agent=getattr(req, "agent_id", "?"),
                    call_site=cs,
                    stream=stream,
                    declared_schema=schema is not None,
                    required_keys=len(_schema_required_keys(schema)),
                )
            except Exception:  # noqa: BLE001 — the seam must never break a response
                logger.debug("structured_empty degradation emit failed",
                             exc_info=True)
        except Exception:  # noqa: BLE001 — telemetry must never break a response
            logger.debug("detect_structured_empty failed", exc_info=True)

    def request_is_structured(self, req: QueuedRequest) -> bool:
        """True when the request constrained its output (grammar / JSON schema /
        structured outputs), so a finish_reason=length truncation almost
        certainly produced broken/unparseable output — not a benign capped reply."""
        if req.payload_type != "chat_completion":
            return False
        p = req.payload
        if not isinstance(p, dict):
            return False
        # The proxy itself removed the caller's bare json_object (see
        # apply_json_object_guard) — the caller's EXPECTATION is unchanged, so
        # the gate must not lapse just because the payload no longer shows it.
        if getattr(req, "json_object_stripped", False):
            return True
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
