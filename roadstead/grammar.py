"""GBNF grammar authority for the LLM proxy.

The proxy is the single choke point for all grammar-constrained traffic, so
it is the place to guarantee every grammar is valid for the target backend's
llama.cpp parser BEFORE dispatch. On grammar-parse-failure llama-server
returns HTTP 200 and runs *unconstrained* (a known, acknowledged llama.cpp
behavior) — so a broken grammar fails silently into free-form output. This
module makes that impossible: we validate + safely normalize, and if the
result still isn't valid we FAIL LOUD (return an error to the caller) rather
than ever dispatching a grammar that would be silently dropped.

Two failure classes drove this (both confirmed against llama.cpp upstream):
  - newline: GBNF allows newlines only between rules, inside parens, or
    after `|`. Multi-line rule bodies with mid-sequence newlines parse on
    lenient builds but are rejected by newer ones (`expecting name`).
    → SAFE-NORMALIZED by wrapping each multi-line rule RHS in `( … )`,
      which makes the internal newlines legal without changing the
      accepted language.
  - repetition: `{m,n}` with n at/over MAX_REPETITION_THRESHOLD (2000)
    is rejected ("…exceeds sane defaults"). Cannot be auto-fixed without
    changing semantics → FAIL LOUD.

Alignment: this is a pure-Python validator encoding the documented rules.
It is kept honest by tests/llmproxy/test_grammar_validator.py, which runs
the grammar corpus through the real `test-gbnf-validator` binary and asserts
this module agrees. Re-capture those fixtures on every llama.cpp upgrade.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

# llama.cpp common/grammar-parser.cpp hardcodes this DoS guard (~b8475+).
MAX_REPETITION_THRESHOLD = 2000

# Rule names: llama.cpp GBNF identifiers are [a-zA-Z0-9-] (no underscores;
# our grammars also follow a lowercase + no-prefix-collision convention).
_RULE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_RULE_HEADER_RE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9-]*)[ \t]*::=", re.MULTILINE)
_REPETITION_RE = re.compile(r"\{\s*(\d+)\s*(?:,\s*(\d+)?)?\s*\}")


@dataclass
class GrammarError:
    code: str            # "repetition_over_threshold" | "undefined_rule" |
                         # "bad_rule_name" | "no_root" | "unparseable"
    detail: str
    rule: str | None = None


@dataclass
class GrammarResult:
    ok: bool
    grammar: str                          # normalized grammar (or original on failure)
    errors: list[GrammarError] = field(default_factory=list)
    normalized: bool = False              # whether normalization changed the text

    def error_payload(self) -> dict:
        return {
            "error": "grammar_invalid",
            "detail": "; ".join(f"{e.code}: {e.detail}" for e in self.errors),
            "errors": [
                {"code": e.code, "detail": e.detail, "rule": e.rule}
                for e in self.errors
            ],
        }


# ---------------------------------------------------------------------------
# Tokenizer — string/charclass-aware so `#`, newlines, and operators inside
# literals are never misread.
# ---------------------------------------------------------------------------

@dataclass
class _Tok:
    kind: str   # name|define|string|charclass|lparen|rparen|pipe|op|repeat|newline
    text: str
    pos: int


def _tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "#":  # comment to end of line
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "\n":
            toks.append(_Tok("newline", "\n", i)); i += 1; continue
        if c in " \t\r":
            i += 1; continue
        if c == '"':  # string literal — consume to closing quote, honoring \"
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2; continue
                if src[j] == '"':
                    break
                j += 1
            toks.append(_Tok("string", src[i:j + 1], i)); i = j + 1; continue
        if c == "[":  # char class — consume to closing ], honoring \]
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2; continue
                if src[j] == "]":
                    break
                j += 1
            toks.append(_Tok("charclass", src[i:j + 1], i)); i = j + 1; continue
        if src.startswith("::=", i):
            toks.append(_Tok("define", "::=", i)); i += 3; continue
        if c == "(":
            toks.append(_Tok("lparen", "(", i)); i += 1; continue
        if c == ")":
            toks.append(_Tok("rparen", ")", i)); i += 1; continue
        if c == "|":
            toks.append(_Tok("pipe", "|", i)); i += 1; continue
        if c in "*+?":
            toks.append(_Tok("op", c, i)); i += 1; continue
        if c == "{":
            j = src.find("}", i)
            if j < 0:
                toks.append(_Tok("op", "{", i)); i += 1; continue
            toks.append(_Tok("repeat", src[i:j + 1], i)); i = j + 1; continue
        m = _RULE_NAME_RE.match(src, i)
        if m:
            toks.append(_Tok("name", m.group(0), i)); i = m.end(); continue
        # Unknown char (e.g. '=' outside ::=, stray punctuation) — skip; the
        # validator's structural checks will catch genuinely broken grammars.
        i += 1
    return toks


# ---------------------------------------------------------------------------
# Rule splitting (header-based; tokenizer confirms `::=` is real, not in a
# string) and validation.
# ---------------------------------------------------------------------------

def _defined_and_referenced(src: str) -> tuple[set[str], set[str]]:
    """Return (defined rule names, referenced rule names)."""
    toks = _tokenize(src)
    defined: set[str] = set()
    referenced: set[str] = set()
    for idx, t in enumerate(toks):
        if t.kind == "define" and idx > 0 and toks[idx - 1].kind == "name":
            defined.add(toks[idx - 1].text)
    for idx, t in enumerate(toks):
        if t.kind == "name":
            nxt = toks[idx + 1] if idx + 1 < len(toks) else None
            if nxt is not None and nxt.kind == "define":
                continue  # this is a definition LHS, not a reference
            referenced.add(t.text)
    return defined, referenced


def _charclass_spans(src: str) -> list[str]:
    """Return the inner text of each [...] char class (string/comment-aware)."""
    spans: list[str] = []
    for t in _tokenize(src):
        if t.kind == "charclass":
            spans.append(t.text[1:-1])  # strip [ ]
    return spans


def validate(src: str) -> list[GrammarError]:
    """Validate a GBNF grammar against the b9357 parser's known rules.
    Returns [] when valid."""
    errors: list[GrammarError] = []

    # Repetition threshold — n (upper bound) must be < MAX_REPETITION_THRESHOLD.
    for m in _REPETITION_RE.finditer(src):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        bound = max(lo, hi)
        if bound >= MAX_REPETITION_THRESHOLD:
            errors.append(GrammarError(
                "repetition_over_threshold",
                f"{{{m.group(1)},{m.group(2) or ''}}} bound {bound} "
                f">= MAX_REPETITION_THRESHOLD ({MAX_REPETITION_THRESHOLD})",
            ))

    # Underscored rule names: GBNF identifiers are [A-Za-z0-9-] only, so the
    # tokenizer splits on `_` and never forms an underscored token. Detect
    # them from raw rule HEADERS (LHS of ::=), which is also where the real
    # parser chokes. References to underscored rules surface as undefined.
    for m in re.finditer(r"^[ \t]*([A-Za-z0-9_-]+)[ \t]*::=", src, re.MULTILINE):
        name = m.group(1)
        if "_" in name:
            errors.append(GrammarError(
                "bad_rule_name", f"rule name {name!r} contains underscore "
                "(llama.cpp GBNF rejects underscores in some builds)", rule=name,
            ))

    # Unknown char-class escapes (b9357 rejects e.g. `\-`). `\-` is
    # auto-fixed by normalize(); anything still present here is unfixable.
    for cc in _charclass_spans(src):
        k = 0
        while k < len(cc) - 1:
            if cc[k] == "\\":
                esc = cc[k + 1]
                if esc not in _VALID_CLASS_ESCAPES:
                    errors.append(GrammarError(
                        "unknown_escape",
                        f"char-class escape \\{esc} is not valid GBNF (b9357 "
                        "rejects it)",
                    ))
                k += 2
                continue
            k += 1

    defined, referenced = _defined_and_referenced(src)
    if not defined:
        errors.append(GrammarError("unparseable", "no rule definitions found"))
        return errors
    if "root" not in defined:
        errors.append(GrammarError("no_root", "grammar has no `root` rule"))

    for name in sorted(referenced - defined):
        # built-in/terminal-only refs don't exist in GBNF; every ref must be defined
        errors.append(GrammarError(
            "undefined_rule", f"rule {name!r} referenced but never defined",
            rule=name,
        ))

    return errors


# ---------------------------------------------------------------------------
# Normalization — wrap each multi-line rule RHS in parens so mid-sequence
# newlines become legal (semantics-preserving: ( X ) accepts == X).
# ---------------------------------------------------------------------------

# Valid escapes inside a GBNF char class for the b9357 parser. `\-` is NOT
# valid (the author meant a literal hyphen; correct GBNF puts it at class
# start/end). Others left for the validator to flag as unknown_escape.
_VALID_CLASS_ESCAPES = set('\\]["nrt') | {"x", "u", "U"}


def _fix_charclass_hyphens(src: str) -> str:
    """Rewrite `\\-` inside char classes to a literal hyphen at the class end.

    b9357 rejects `\\-` as an unknown escape. The author always means a
    literal hyphen, whose correct GBNF position is the start or end of the
    class. We strip `\\-` and append a single `-` before `]` (a hyphen
    immediately before `]` is always literal). Set-preserving."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':  # skip string literals verbatim
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2; continue
                if src[j] == '"':
                    break
                j += 1
            out.append(src[i:j + 1]); i = j + 1; continue
        if c == "#":  # skip comments verbatim
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(src[i:j]); i = j; continue
        if c == "[":  # char class
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2; continue
                if src[j] == "]":
                    break
                j += 1
            inner = src[i + 1:j]
            if "\\-" in inner:
                inner = inner.replace("\\-", "")
                if not inner.endswith("-"):
                    inner = inner + "-"
            out.append("[" + inner + "]"); i = j + 1; continue
        out.append(c); i += 1
    return "".join(out)


def normalize(src: str) -> str:
    src = _fix_charclass_hyphens(src)
    headers = list(_RULE_HEADER_RE.finditer(src))
    if not headers:
        return src
    out_parts: list[str] = [src[: headers[0].start()]]
    for k, h in enumerate(headers):
        start = h.start()
        end = headers[k + 1].start() if k + 1 < len(headers) else len(src)
        block = src[start:end]
        define_at = block.find("::=")
        head = block[: define_at + 3]                      # "name ::="
        rhs = block[define_at + 3:]
        # Trailing whitespace/newlines between this rule and the next header.
        stripped = rhs.rstrip()
        trailer = rhs[len(stripped):]
        body = stripped.strip()
        # Already parenthesized as a whole, or single-line → leave as is.
        is_multiline = "\n" in body
        already_wrapped = _is_single_paren_group(body)
        if is_multiline and not already_wrapped and body:
            new_block = f"{head} (\n{body}\n){trailer}"
        else:
            new_block = block
        out_parts.append(new_block)
    return "".join(out_parts)


def _is_single_paren_group(body: str) -> bool:
    """True if `body` is exactly one ( ... ) group spanning the whole RHS."""
    if not body.startswith("(") or not body.endswith(")"):
        return False
    depth = 0
    for idx, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx != len(body) - 1:
                return False
    return depth == 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def normalize_and_validate(src: str) -> GrammarResult:
    """Normalize (safe transforms) then validate. The proxy dispatches the
    returned grammar only when ok=True; otherwise it fails loud with
    error_payload()."""
    if not isinstance(src, str) or not src.strip():
        return GrammarResult(False, src, [GrammarError("unparseable", "empty grammar")])
    normalized = normalize(src)
    errors = validate(normalized)
    return GrammarResult(
        ok=not errors,
        grammar=normalized if not errors else src,
        errors=errors,
        normalized=(normalized != src),
    )


def grammar_hash(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Reasoning-field injection — CRANE-style "reason-then-constrain".
#
# Grammars constrain decoding from token 0 (every root opens with `"{"`), which
# forces the model to emit the decision field with ZERO reasoning tokens. For
# consequential JUDGMENT grammars the structured-output policy layer prepends a
# short free-text `reason` field as the FIRST object member so the model
# reasons before it commits, then the proxy STRIPS the field on egress so the
# caller's schema is unchanged. This is a pure transform; the proxy re-validates
# the result with normalize_and_validate() and fails loud if anything is off.
#
# Backend tolerance: older llama.cpp (b9357-class — the anvil companion/gemma
# llama-servers) silently drop grammars containing large bounded `{0,N}` string
# repetitions (see agents/forum-agent/grammars/proposal.gbnf). vLLM's guidance
# backend accepts them. So `max_chars=None` emits an UNBOUNDED `rchar*` for
# llama.cpp endpoints; a bound is used for vLLM. The proxy picks per endpoint.
# ---------------------------------------------------------------------------

REASON_FIELD = "reason"
REASON_MAX_CHARS = 400                # bounded default; well under MAX_REPETITION_THRESHOLD
_REASON_RCHAR = r'[^"\\\x00-\x1f]'    # JSON-safe string char (matches the grammars' own `char`)


@dataclass
class InjectionResult:
    grammar: str          # transformed grammar (or the original when not injected)
    injected: bool        # False when the root is not an object (bare enum) or field present
    field: str            # the injected field name (the egress layer strips this)


def _free_rule_name(base: str, defined: set[str]) -> str:
    """A rule name not already defined in the grammar (avoids collisions)."""
    if base not in defined:
        return base
    i = 2
    while f"{base}{i}" in defined:
        i += 1
    return f"{base}{i}"


def inject_reason_field(
    src: str,
    field: str = REASON_FIELD,
    max_chars: int | None = REASON_MAX_CHARS,
) -> InjectionResult:
    """Insert a leading free-text `field` as the FIRST member of an object-root
    GBNF so the model reasons before the constrained decision.

    Returns injected=False (original grammar untouched) when the root is not an
    object (bare-enum grammars like classify_document), when the grammar has no
    `root` rule, or when the field key is already present.

    `max_chars=None` emits an UNBOUNDED string (`rchar*`) for backends that
    reject large bounded repetitions; otherwise `rchar{0,max_chars}`.
    """
    if not isinstance(src, str) or "root" not in src:
        return InjectionResult(src, False, field)

    headers = list(_RULE_HEADER_RE.finditer(src))
    root_h = next((h for h in headers if h.group(1) == "root"), None)
    if root_h is None:
        return InjectionResult(src, False, field)

    ri = headers.index(root_h)
    start = root_h.start()
    end = headers[ri + 1].start() if ri + 1 < len(headers) else len(src)
    block = src[start:end]
    define_at = block.find("::=")
    head, rhs = block[: define_at + 3], block[define_at + 3 :]

    key_literal = '"\\"%s\\""' % field            # GBNF literal for the JSON key, e.g. "\"reason\""
    if key_literal in rhs or '"{"' not in rhs:     # already present, or not an object root
        return InjectionResult(src, False, field)

    defined, _ = _defined_and_referenced(src)
    sname = _free_rule_name("reasonstr", defined)
    cname = _free_rule_name("rchar", defined)
    wsx = "ws " if "ws" in defined else ""

    # ` ws "<key>" ws ":" ws <reasonstr> ws ,` inserted right after the opening brace.
    insertion = ' %s%s %s":" %s%s %s","' % (wsx, key_literal, wsx, wsx, sname, wsx)
    new_rhs = rhs.replace('"{"', '"{"' + insertion, 1)
    new_block = head + new_rhs

    rep = "*" if max_chars is None else "{0,%d}" % max_chars
    rules = '\n%s ::= "\\"" %s%s "\\""\n%s ::= %s\n' % (sname, cname, rep, cname, _REASON_RCHAR)

    new_src = src[:start] + new_block + src[end:] + rules
    return InjectionResult(new_src, True, field)


def root_object_keys(src: str) -> list[str]:
    """Ordered list of top-level JSON object keys the `root` rule emits
    (best-effort: the `"\\"key\\""` string literals in the root RHS, in order).

    Used by the egress double-check to confirm a stripped response carries only
    the caller's original keys (no leaked `reason`/injected artifact), and by
    tests to assert field ordering.
    """
    headers = list(_RULE_HEADER_RE.finditer(src))
    root_h = next((h for h in headers if h.group(1) == "root"), None)
    if root_h is None:
        return []
    ri = headers.index(root_h)
    end = headers[ri + 1].start() if ri + 1 < len(headers) else len(src)
    rhs = src[root_h.start():end].split("::=", 1)[-1]
    keys: list[str] = []
    for t in _tokenize(rhs):
        if t.kind != "string":
            continue
        inner = t.text[1:-1]                       # strip surrounding GBNF quotes
        # A JSON key literal looks like \"name\" or \"name\": (combined colon).
        m = re.match(r'\\"([A-Za-z0-9_]+)\\"', inner)
        if m:
            keys.append(m.group(1))
    return keys


def strip_top_field(output: str, field: str) -> tuple[str, bool]:
    """Remove a single top-level JSON field from a model output string (the
    egress strip for the injected ``reason``). Returns (stripped_output,
    removed). On JSON-parse failure or field-absent returns (output, False) —
    the caller's verify_conformance then catches a genuinely malformed output."""
    try:
        obj = json.loads(output)
    except Exception:
        return output, False
    if not isinstance(obj, dict) or field not in obj:
        return output, False
    obj.pop(field, None)
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False), True


def verify_conformance(
    output: str, grammar: str, *, forbid_field: str | None = None
) -> tuple[bool, str]:
    """Egress double-check: confirm a (stripped) model output conforms to the
    caller's ORIGINAL grammar — best-effort, structural (NOT a full GBNF parse).

    Catches the two failure modes the proxy layer must never pass through:
    silent grammar-drop (non-JSON / markdown-fenced output) and a leaked
    injected field. Checks, in order: (1) not markdown-fenced; (2) for an
    object-root grammar, JSON-parses to a dict; (3) its top-level keys are a
    subset of the keys the grammar's root can emit; (4) ``forbid_field`` absent.
    Bare-enum / non-object roots only get the fence + forbid checks (we can't
    structurally validate a bare token here). Returns (ok, reason)."""
    s = (output or "").strip()
    if s.startswith("`"):
        return False, "markdown_fenced(grammar_not_enforced)"
    keys = root_object_keys(grammar)
    if not keys:
        if forbid_field and forbid_field in s:
            return False, f"forbidden_field_present:{forbid_field}"
        return True, "ok_non_object_root"
    try:
        obj = json.loads(s)
    except Exception as e:  # noqa: BLE001
        return False, f"not_json:{type(e).__name__}"
    if not isinstance(obj, dict):
        return False, "not_object"
    if forbid_field and forbid_field in obj:
        return False, f"forbidden_field_present:{forbid_field}"
    extra = set(obj.keys()) - set(keys)
    if extra:
        return False, f"unexpected_keys:{sorted(extra)}"
    return True, "ok"


def recover_structured_object(
    output: str, *, grammar: str | None = None, allowed_keys: list[str] | None = None
) -> str | None:
    """Deterministic noisy-output recovery for object-root structured responses.

    Some backend/decoder corner cases emit a small amount of junk *before* the
    constrained object — notably the bounded (<=~4 char) stray opening-brace
    artifact vLLM produces at the reasoning->JSON boundary under MTP spec-decode
    (PR #44142 fixes `</think>` detection but the one-step-deferred FSM advance
    lets the model emit its own ``{`` before the grammar's ``{``). The object
    itself is well-formed and conformant; only a tiny prefix is noise.

    This scans for the first ``{`` whose JSON object's top-level keys fall within
    the allowed set (the grammar/schema root keys) and returns it re-serialized
    and clean. It NEVER guesses: a partial/duplicate prefix fails ``raw_decode``
    and is skipped; if nothing structurally matches, returns ``None`` so the
    caller falls back (2-call / deferrable error) — clean-then-verify, never
    parse-and-hope. The caller should still run :func:`verify_conformance` on the
    result. ``allowed_keys`` (e.g. a JSON-schema's ``properties``) takes
    precedence; otherwise keys are derived from ``grammar`` via
    :func:`root_object_keys`."""
    if allowed_keys is None and grammar is not None:
        allowed_keys = root_object_keys(grammar)
    allowed = set(allowed_keys or [])
    if not allowed:
        return None  # object-root recovery only; nothing to anchor on
    text = output or ""
    dec = json.JSONDecoder()
    i = 0
    while True:
        b = text.find("{", i)
        if b < 0:
            return None
        try:
            obj, _ = dec.raw_decode(text, b)
        except Exception:  # noqa: BLE001
            i = b + 1
            continue
        if isinstance(obj, dict) and obj and set(obj.keys()) <= allowed:
            return json.dumps(obj, ensure_ascii=False)
        i = b + 1
