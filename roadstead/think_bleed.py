"""Reasoning that leaks into ``content`` — prevention, and repair where a tag survives.

Two defects share a symptom (the caller reads a chain of thought as its answer)
and have different causes, so they get different tools.

**1. A thinking switch sent to a model that has none — PREVENTED, never repaired.**
Measured 2026-09-29 on GLM-5.3-Flash under vLLM ``--reasoning-parser glm45``: the
chat template ALWAYS ends the prompt ``<think>``, whatever
``enable_thinking``/``thinking`` says (its only knob is ``reasoning_effort``).
The PARSER, though, reads that kwarg and switches itself off for the request when
it is false. So the model still reasons, the ``</think>`` special token is removed
from the output, and the whole trace lands in ``content`` with NO TAG LEFT::

    no kwargs                    content='73915'          reasoning='73915'
    enable_thinking: false       content='7391573915'     reasoning=''

Nothing downstream can split that — there is no boundary to find, and guessing
one is fabrication. The only fix is to never send the kwarg. An endpoint that
declares ``policy.thinking_kwargs: []`` is telling us there is no switch, so
:func:`strip_thinking_switch` removes every spelling the code knows before the
payload reaches the wire. It is keyed on ``EndpointConfig.no_thinking_switch``
(a DECLARATION), never on ``thinking_kwargs == ()`` (which an unmeasured
template also produces).

**1b. The same control under the wrong name — RENAMED.** ``thinking`` and
``enable_thinking`` are one control in two spellings and the template decides which it
reads. On an endpoint that declares its key(s), :func:`rename_thinking_switch` carries a
caller's other spelling onto the declared one, value kept: measured, an engine that reads
only ``enable_thinking`` 400s a ``thinking`` it parses as an object.

**2. A trace that carries a marker — REPAIRED.** Some engines leave the tag in
``content``: llama.cpp with its reasoning parser off returns
``<think>…</think>answer``, and its reasoning-budget cap binding returns
``…</think>answer`` with the OPENING tag missing (the template prefilled it —
llama.cpp #28182). :func:`repair_message` (whole responses) and
:class:`StreamRepair` (SSE, including a tag split across chunks) move the trace
to the reasoning field and leave only the answer in ``content``. Acts only on an
endpoint that DECLARES ``capabilities.reasoning``; never touches content without
a marker.

The close-only shape is the expensive one: in a STREAM it means holding content until a
``</think>`` shows up, which delays every healthy no-reasoning stream on the endpoint, so
streaming close-only repair is an operator opt-in (``policy.repair_close_only_reasoning``,
absent ⇒ off). The open-tag shape needs no hold beyond the first few characters.

WHAT THIS DELIBERATELY DOES NOT DO. It does not guess where reasoning ends when no
tag exists (the GLM shape above), and it does not move a truncated trace back into
``content``: an unclosed ``<think>…`` with ``finish_reason=length`` becomes
reasoning with EMPTY content, and the caller's own empty-completion handling
decides what that means — the same as a parser-split truncation.

Pure and dependency-light on purpose: ``backend.py`` calls it, and the unit tests
drive it without a socket.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from .providers.payload import _THINKING_KWARG_NAMES

logger = logging.getLogger(__name__)

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

#: How much content a stream will HOLD back while it waits to learn whether the
#: text so far is a reasoning trace with no opening tag (the close-only shape).
#: Reached ONLY on an endpoint that opted in with
#: ``policy.repair_close_only_reasoning`` — holding is a latency cost paid by every
#: healthy no-reasoning stream on such an endpoint, so it is never a default. Past
#: the bound the text is released as content: 64k chars is ~16k tokens, above the
#: 8k-token reasoning headroom the proxy grants.
HOLD_LIMIT_CHARS = 65536

#: How much leading content ANY stream will hold while it works out whether the
#: text opens with ``<think>``. The tag is 7 characters; the rest is leading
#: whitespace, and past this many characters the text is plainly not a trace.
PREFIX_HOLD_CHARS = 16

#: A rate limit on the log line, not on the counter. One line per (kind,
#: endpoint) per interval, carrying how many it stands for.
_LOG_INTERVAL_S = 60.0

_REASONING_KEYS = ("reasoning", "reasoning_content")


# --------------------------------------------------------------------------- #
# 1. Prevention
# --------------------------------------------------------------------------- #

def _value_label(value: Any) -> str:
    """A BOUNDED description of a caller-supplied value, for a counter key.

    The value is caller input, and a counter keyed on it is a caller-controlled
    allocation: 1,000 distinct strings would be 1,000 keys. So a bool reads
    ``true``/``false`` (the distinction that matters — ``false`` is the one that
    leaks), ``None`` reads ``null``, and anything else reads as its TYPE."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return type(value).__name__


def _is_none_effort(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "none"


def _strip_container(container: dict, prefix: str, found: list[str]) -> dict:
    """Copy-on-write removal from ONE dict (the payload or its ``extra_body``).
    Returns ``container`` itself when nothing was there to remove."""
    out = container
    ck = container.get("chat_template_kwargs")
    if isinstance(ck, dict):
        new_ck = dict(ck)
        for key in [k for k in ck if k in _THINKING_KWARG_NAMES]:
            found.append(
                f"{prefix}chat_template_kwargs.{key}={_value_label(ck[key])}")
            del new_ck[key]
        if _is_none_effort(new_ck.get("reasoning_effort")):
            found.append(f"{prefix}chat_template_kwargs.reasoning_effort=none")
            del new_ck["reasoning_effort"]
        if len(new_ck) != len(ck):
            out = dict(out)
            if new_ck:
                out["chat_template_kwargs"] = new_ck
            else:
                del out["chat_template_kwargs"]
    if _is_none_effort(container.get("reasoning_effort")):
        found.append(f"{prefix}reasoning_effort=none")
        out = dict(out)
        del out["reasoning_effort"]
    obj = container.get("reasoning")
    if isinstance(obj, dict) and _is_none_effort(obj.get("effort")):
        found.append(f"{prefix}reasoning.effort=none")
        rest = {k: v for k, v in obj.items() if k != "effort"}
        out = dict(out)
        if rest:
            out["reasoning"] = rest
        else:
            del out["reasoning"]
    return out


def strip_thinking_switch(payload: Any) -> tuple[Any, list[str]]:
    """Remove every thinking-switch spelling from ``payload`` for an endpoint
    that DECLARES it has none. Returns ``(payload, stripped)`` — the same object
    and ``[]`` when there was nothing to strip, otherwise a shallow copy (the
    caller's dict is corpus-persisted and never mutated here).

    Removed, in the payload and in ``extra_body`` (which the provider merges up):

    * ``chat_template_kwargs.thinking`` / ``.enable_thinking`` — at ANY value.
      ``false`` is the one that breaks the reasoning split; ``true`` is
      harmless but is still a switch sent to a model that has none, and a
      caller cannot be told apart from the other by anything but the value.
      Each is recorded WITH its value so ``/v1/status`` can say which spelling
      was actually the leak.
    * ``"none"`` as a reasoning effort — ``reasoning_effort`` (top level or in
      ``chat_template_kwargs``) and ``reasoning.effort``. On a model with a
      switch, "none" IS the OpenAI spelling of "off" (``correction.
      fold_caller_effort`` turns it into the switch). Here there is no switch
      for it to become, so it cannot mean off; it is REMOVED and the endpoint's
      own effort default applies. That is the lowest honest mapping: the model
      keeps reasoning either way, and inventing "low" would put a rung in the
      caller's mouth that the endpoint's ``reasoning_effort_map`` may not admit.
      Any OTHER effort word is left for ``apply_reasoning_effort_map`` to judge.
    """
    if not isinstance(payload, dict):
        return payload, []
    found: list[str] = []
    out = _strip_container(payload, "", found)
    eb = out.get("extra_body")
    if isinstance(eb, dict):
        new_eb = _strip_container(eb, "extra_body.", found)
        if new_eb is not eb:
            out = dict(out)
            out["extra_body"] = new_eb
    return out, found


def _as_switch_value(value: Any) -> bool | None:
    """The boolean a caller's switch value means, or None when it means nothing
    a boolean switch can carry. A bool is itself; an Anthropic-style object
    (``{"type": "enabled" | "disabled" | "adaptive"}`` — the reading that makes some
    engines 400 a bare ``thinking``) is what its ``type`` says. Anything else is
    not carried: renaming an object onto a boolean key would just move the 400."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        kind = str(value.get("type") or "").strip().lower()
        if kind in ("enabled", "adaptive"):
            return True
        if kind == "disabled":
            return False
    return None


def _rename_container(container: dict, prefix: str, declared: tuple[str, ...],
                      found: list[str]) -> dict:
    ck = container.get("chat_template_kwargs")
    if not isinstance(ck, dict):
        return container
    foreign = [k for k in ("thinking", "enable_thinking")
               if k in ck and k not in declared]
    if not foreign:
        return container
    values = [_as_switch_value(ck[k]) for k in foreign]
    agreed = values[0] is not None and all(v == values[0] for v in values)
    new_ck = {k: v for k, v in ck.items() if k not in foreign}
    landed = []
    if agreed:
        for key in declared:
            if key not in new_ck:            # the caller's own declared key wins
                new_ck[key] = values[0]
                landed.append(key)
    label = "->".join(("+".join(foreign), "+".join(landed) or "dropped"))
    found.append(f"{prefix}chat_template_kwargs.{label}"
                 + (f"={_value_label(values[0])}" if landed else ""))
    out = dict(container)
    out["chat_template_kwargs"] = new_ck
    return out


def rename_thinking_switch(payload: Any, declared: tuple[str, ...]) -> tuple[Any, list[str]]:
    """On an endpoint that DECLARES its switch key(s), put a caller's switch under
    the declared spelling — value kept — instead of forwarding the wrong one.

    ``thinking`` and ``enable_thinking`` are two spellings of one control and which
    one a template reads is per-model. An engine that reads only ``enable_thinking``
    may not merely ignore ``thinking``: measured, one 400s it (it parses ``thinking``
    as an Anthropic-style object), which reaches the caller as a 502. So the caller's
    intent is carried, not echoed.

    Per container (payload and ``extra_body``): a known spelling that is not
    declared moves onto every declared key not already present. If the caller ALSO
    sent a declared key, theirs wins and the foreign one is dropped; if the foreign
    spellings disagree with each other there is no single value to carry, so they
    are dropped and the endpoint's default applies. An endpoint that declares both
    spellings never has anything renamed. Same copy-on-write contract as
    :func:`strip_thinking_switch`. This does not touch efforts:
    ``correction.fold_caller_effort`` already turns ``"none"`` into the declared
    keys, and composes with this (it runs first, and writes only declared keys).
    """
    if not isinstance(payload, dict) or not declared:
        return payload, []
    found: list[str] = []
    out = _rename_container(payload, "", declared, found)
    eb = out.get("extra_body")
    if isinstance(eb, dict):
        new_eb = _rename_container(eb, "extra_body.", declared, found)
        if new_eb is not eb:
            out = dict(out)
            out["extra_body"] = new_eb
    return out, found


# --------------------------------------------------------------------------- #
# 2. Repair — whole responses
# --------------------------------------------------------------------------- #

def default_reasoning_key(engine: str) -> str:
    """The spelling an engine emits when the response carries none to copy.
    vLLM's parsers say ``reasoning``; llama.cpp says ``reasoning_content`` (and
    ``backend.call_watched`` reassembles to that when a stream never named one)."""
    return "reasoning" if engine == "vllm" else "reasoning_content"


def _after_tag(text: str) -> str:
    """What follows a ``</think>``: the template puts ONE newline between the tag and
    the answer, and that is all that is removed — an answer that begins with
    indentation (code) keeps it."""
    if text.startswith("\r\n"):
        return text[2:]
    return text[1:] if text.startswith("\n") else text


def _split(content: str, *, close_only: bool) -> tuple[str, str, str] | None:
    """``(reasoning, answer, shape)`` or None when ``content`` carries no marker
    this endpoint may act on.

    ``open_tag``   ``<think>R</think>A``           — llama.cpp, parser off
    ``unclosed``   ``<think>R``                    — truncated inside the trace
    ``close_only`` ``R</think>A``                  — template prefilled ``<think>``

    ``close_only`` is only believed where the caller has decided the endpoint
    reasons on this call (``backend._close_only_expected``: an operator opt-in, or
    a forced reasoner not switched off). Elsewhere a stray ``</think>`` is far
    more likely to be an answer ABOUT the tag than a leak, and rewriting it would
    lose the answer's own text.
    """
    lead = content.lstrip()
    if lead.startswith(THINK_OPEN):
        body = lead[len(THINK_OPEN):]
        i = body.find(THINK_CLOSE)
        if i < 0:
            return body.strip(), "", "unclosed"
        return (body[:i].strip(), _after_tag(body[i + len(THINK_CLOSE):]),
                "open_tag")
    if close_only:
        i = content.find(THINK_CLOSE)
        if i >= 0:
            return (content[:i].strip(),
                    _after_tag(content[i + len(THINK_CLOSE):]), "close_only")
    return None


def repair_message(msg: Any, *, close_only: bool,
                   default_key: str) -> tuple[str, int, int] | None:
    """Move a tagged trace out of ``msg["content"]`` (in place). Returns
    ``(shape, reasoning_chars, answer_chars)`` when it changed anything.

    Skipped when the response ALREADY carries reasoning under ANY key: the engine
    split it, so a tag left in ``content`` is the answer's own text. Tool calls are untouched
    and, for a turn whose whole content was trace, ``content`` becomes ``""``.
    """
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        return None
    # EVERY key, not the first present: `{"reasoning": null, "reasoning_content":
    # "x"}` HAS reasoning, and a null in one spelling says nothing about the other.
    if any(str(msg.get(k) or "").strip() for k in _REASONING_KEYS):
        return None
    key = next((k for k in _REASONING_KEYS if k in msg), None)
    hit = _split(content, close_only=close_only)
    if hit is None:
        return None
    reasoning, answer, shape = hit
    msg[key or default_key] = reasoning
    msg["content"] = answer
    return shape, len(reasoning), len(answer)


def repair_body(body: Any, *, close_only: bool,
                default_key: str) -> tuple[str, int, int] | None:
    """:func:`repair_message` on ``choices[0].message`` of a chat.completion."""
    try:
        msg = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    return repair_message(msg, close_only=close_only,
                          default_key=default_key)


# --------------------------------------------------------------------------- #
# 2. Repair — streams
# --------------------------------------------------------------------------- #

_UNDECIDED, _IN_THINK, _ANSWER_LEAD, _PASS = range(4)

Frame = tuple[dict, str]


def _partial_close_len(text: str) -> int:
    """Length of the longest SUFFIX of ``text`` that is a proper prefix of
    ``</think>`` — the bytes that might be the front half of a tag whose back
    half has not arrived yet."""
    for k in range(min(len(THINK_CLOSE) - 1, len(text)), 0, -1):
        if THINK_CLOSE.startswith(text[-k:]):
            return k
    return 0


class StreamRepair:
    """Rewrite an SSE chunk stream so a tagged trace arrives as reasoning deltas.

    Feed it every parsed chunk; it returns the frames to relay in its place —
    ``[(parsed, data)]`` (the ORIGINAL bytes, for anything it does not touch),
    several synthesized frames, or ``[]`` when it is holding content back.
    Call :meth:`finish` once at the end of the stream.

    STATES.  ``UNDECIDED`` (start): a leading ``<think>`` commits to ``IN_THINK``
    at once — trace text streams out as reasoning deltas as it arrives, with only
    the last few bytes retained in case they are half a ``</think>``. Content is
    held only until its first non-whitespace characters prove or disprove that
    tag (at most :data:`PREFIX_HOLD_CHARS`), then released untouched (``PASS``),
    so a healthy stream's first token is not delayed by more than that.

    ``hold_for_close_tag`` is the ONE case that holds longer, and it is an
    operator opt-in (``policy.repair_close_only_reasoning``): the close-only shape
    (``…</think>answer``, no opening tag) is indistinguishable from an answer until
    the tag arrives, so an opted-in endpoint holds content until a ``</think>``
    appears or the bound in :data:`HOLD_LIMIT_CHARS` is reached. That trades the
    time-to-first-token of every no-reasoning stream on the endpoint for the
    repair, which is why it is off unless declared.

    A chunk that carries a reasoning field proves the engine is splitting:
    everything after it passes through. A frame carrying an ``error`` releases
    whatever is held BEFORE it, so the caller sees the text and then the failure.

    Only ``choices[0]`` is inspected (``n > 1`` streams are not repaired); a frame
    with no choices (the usage-only frame) or another choice index passes through
    untouched.
    """

    def __init__(self, *, hold_for_close_tag: bool, default_key: str,
                 hold_limit: int = HOLD_LIMIT_CHARS) -> None:
        self._hold_close = hold_for_close_tag
        self._key = default_key
        self._limit = hold_limit
        self._state = _UNDECIDED
        self._buf = ""
        self._strip_reasoning_lead = False
        self._last: dict | None = None
        self._whole = False
        #: Set the first time anything is moved; read by the caller for the
        #: counter and the log line.
        self.shape: str | None = None
        self.reasoning_chars = 0
        self.answer_chars = 0

    @property
    def repaired(self) -> bool:
        return self.shape is not None

    # -- frame construction ------------------------------------------------ #
    def _frame(self, delta: dict) -> Frame:
        base = {k: v for k, v in (self._last or {}).items()
                if k not in ("choices", "usage")}
        base["choices"] = [{"index": 0, "delta": delta, "finish_reason": None}]
        return base, json.dumps(base)

    def _reasoning(self, text: str) -> list[Frame]:
        if self._strip_reasoning_lead:
            text = text.lstrip()
            if text:
                self._strip_reasoning_lead = False
        if not text:
            return []
        self.reasoning_chars += len(text)
        return [self._frame({self._key: text})]

    def _content(self, text: str) -> list[Frame]:
        if not text:
            return []
        return [self._frame({"content": text})]

    def _answer(self, rest: str) -> list[Frame]:
        """Text that follows a ``</think>``. The template puts one newline between
        the tag and the answer and only that is removed. If the tag ended the chunk
        the newline may still be coming, so the next content chunk gets the same
        treatment (``_ANSWER_LEAD``)."""
        if rest in ("", "\r"):
            self._state = _ANSWER_LEAD
            self._buf = rest                      # a lone CR may be half a CRLF
            return []
        self._state = _PASS
        rest = _after_tag(rest)
        self.answer_chars += len(rest)
        return self._content(rest)

    # -- the state machine -------------------------------------------------- #
    def _think(self, text: str) -> list[Frame]:
        text = self._buf + text
        self._buf = ""
        i = text.find(THINK_CLOSE)
        if i >= 0:
            return self._reasoning(text[:i]) + self._answer(text[i + len(THINK_CLOSE):])
        keep = _partial_close_len(text)
        if keep:
            self._buf = text[len(text) - keep:]
            text = text[:len(text) - keep]
        return self._reasoning(text)

    def _consume(self, piece: str, held_before: bool) -> list[Frame]:
        if self._state == _IN_THINK:
            return self._think(piece)
        if self._state == _ANSWER_LEAD:
            piece, self._buf = self._buf + piece, ""
            if piece == "\r":
                self._buf = piece                 # wait for the "\n" of a CRLF
                return []
            self._state = _PASS
            out = _after_tag(piece)
            self.answer_chars += len(out)
            return self._content(out)
        # UNDECIDED
        buf = self._buf + piece
        lead = buf.lstrip()
        if lead.startswith(THINK_OPEN):
            self._state = _IN_THINK
            self.shape = "open_tag"
            self._strip_reasoning_lead = True
            self._buf = ""
            return self._think(lead[len(THINK_OPEN):])
        if self._hold_close:
            i = buf.find(THINK_CLOSE)
            if i >= 0:
                self.shape = "close_only"
                self._buf = ""
                self._strip_reasoning_lead = True
                return self._reasoning(buf[:i]) + self._answer(buf[i + len(THINK_CLOSE):])
            if len(buf) <= self._limit:
                self._buf = buf                   # the tag may still be coming
                return []
        elif ((not lead or THINK_OPEN.startswith(lead))
              and len(buf) <= PREFIX_HOLD_CHARS):
            self._buf = buf                       # could still become `<think>`
            return []
        # Decided: content. Untouched when nothing was held, so the common case
        # relays the backend's own bytes.
        self._buf = ""
        self._state = _PASS
        if not held_before:
            self._whole = True
            return []
        return self._content(buf)

    def _settle(self) -> list[Frame]:
        """Stop deciding. Held text with no marker is CONTENT (released as it
        came); an unclosed trace is REASONING and never becomes content."""
        out: list[Frame] = []
        if self._state == _UNDECIDED and self._buf:
            out = self._content(self._buf)
        elif self._state == _ANSWER_LEAD:
            out = self._content(self._buf)
        elif self._state == _IN_THINK:
            if self.shape == "open_tag":
                self.shape = "unclosed"
            out = self._reasoning(self._buf)
        self._buf = ""
        self._state = _PASS
        return out

    # -- public ------------------------------------------------------------- #
    def feed(self, parsed: dict, data: str) -> list[Frame]:
        if self._state == _PASS:
            return [(parsed, data)]
        if isinstance(parsed, dict) and "error" in parsed:
            return self._settle() + [(parsed, data)]
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if not choices or not isinstance(choices[0], dict):
            return [(parsed, data)]
        choice = choices[0]
        if choice.get("index", 0) != 0:
            return [(parsed, data)]
        self._last = parsed
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        finish = choice.get("finish_reason")
        # A reasoning field on the wire: the engine is splitting, so content is
        # the answer. Release anything held (there should be nothing) and stop.
        for k in _REASONING_KEYS:
            if isinstance(delta.get(k), str) and delta[k]:
                out = self._settle() if self._buf else []
                self._state = _PASS
                return out + [(parsed, data)]
        piece = delta.get("content")
        piece = piece if isinstance(piece, str) else ""
        held_before = bool(self._buf)
        self._whole = False
        out = self._consume(piece, held_before) if piece else []
        if self._whole:
            return [(parsed, data)]
        if delta.get("tool_calls") or finish:
            out += self._settle()
        if not piece:
            return out + [(parsed, data)]
        residual = {k: v for k, v in delta.items() if k != "content"}
        if residual or finish:
            res = dict(parsed)
            ch = dict(choice)
            ch["delta"] = residual
            res["choices"] = [ch, *choices[1:]]
            out.append((res, json.dumps(res)))
        return out

    def finish(self) -> list[Frame]:
        """End of stream with no finish chunk: release whatever is held."""
        return self._settle()


# --------------------------------------------------------------------------- #
# Counters and the log line
# --------------------------------------------------------------------------- #

class ThinkBleedStats:
    """The two tallies ``/v1/status`` reports (``reliability.think_switch_
    stripped*`` and ``reliability.think_bleed_repaired*``), plus the rate-limited
    log lines. Plain ints on the single event loop, like every other counter."""

    def __init__(self) -> None:
        self.switch_stripped = 0
        self.switch_stripped_by_endpoint: dict[str, dict] = {}
        self.switch_renamed = 0
        self.switch_renamed_by_endpoint: dict[str, dict] = {}
        self.bleed_repaired = 0
        self.bleed_repaired_by_endpoint: dict[str, dict] = {}
        self._logged: dict[tuple[str, str], tuple[float, int]] = {}

    def _should_log(self, kind: str, endpoint: str) -> int:
        """0 = suppress; N = log now, standing for N events."""
        now = time.monotonic()
        last, pending = self._logged.get((kind, endpoint), (None, 0))
        pending += 1
        if last is not None and now - last < _LOG_INTERVAL_S:
            self._logged[(kind, endpoint)] = (last, pending)
            return 0
        self._logged[(kind, endpoint)] = (now, 0)
        return pending

    def note_stripped(self, endpoint: str, spellings: list[str],
                      request_id: str) -> None:
        self.switch_stripped += 1
        row = self.switch_stripped_by_endpoint.setdefault(
            endpoint, {"count": 0, "spellings": {}})
        row["count"] += 1
        for s in spellings:
            row["spellings"][s] = row["spellings"].get(s, 0) + 1
        n = self._should_log("stripped", endpoint)
        if n:
            logger.warning(
                "ROADSTEAD_THINK_SWITCH_STRIPPED endpoint=%s request_id=%s "
                "spellings=%s events=%d — the endpoint declares NO thinking "
                "switch (policy.thinking_kwargs: []); forwarding one makes the "
                "backend's reasoning parser stand down and the whole trace land "
                "in content with no tag to find",
                endpoint, request_id, ",".join(spellings), n)

    def note_renamed(self, endpoint: str, renames: list[str],
                     request_id: str) -> None:
        self.switch_renamed += 1
        row = self.switch_renamed_by_endpoint.setdefault(
            endpoint, {"count": 0, "renames": {}})
        row["count"] += 1
        for r in renames:
            row["renames"][r] = row["renames"].get(r, 0) + 1
        n = self._should_log("renamed", endpoint)
        if n:
            logger.warning(
                "ROADSTEAD_THINK_SWITCH_RENAMED endpoint=%s request_id=%s "
                "renames=%s events=%d — the caller spelled the thinking switch "
                "differently from the key this endpoint's template reads "
                "(policy.thinking_kwargs); carried its value onto the declared key",
                endpoint, request_id, ",".join(renames), n)

    def note_repaired(self, endpoint: str, request_id: str, *, shape: str,
                      stream: bool, reasoning_chars: int,
                      answer_chars: int) -> None:
        self.bleed_repaired += 1
        row = self.bleed_repaired_by_endpoint.setdefault(
            endpoint, {"count": 0, "shapes": {}, "stream": 0, "sync": 0})
        row["count"] += 1
        row["shapes"][shape] = row["shapes"].get(shape, 0) + 1
        row["stream" if stream else "sync"] += 1
        n = self._should_log("repaired", endpoint)
        if n:
            logger.warning(
                "ROADSTEAD_THINK_BLEED_REPAIRED endpoint=%s request_id=%s "
                "shape=%s stream=%s reasoning_chars=%d answer_chars=%d events=%d "
                "— the backend left a reasoning trace in content; moved it to "
                "the reasoning field",
                endpoint, request_id, shape, stream, reasoning_chars,
                answer_chars, n)
