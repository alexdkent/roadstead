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
