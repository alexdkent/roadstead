from __future__ import annotations

"""Internal self-consistency smoke tests for the WU5 regression corpora.

These prove the STATIC corpus (schemas.py) is coherent so Phases 3-5 can trust
it as a grading fixture. They do NOT drive the proxy. Plain sync tests.

- Every StructuredCase / CHAT_LOOP_CASES entry has all required fields.
- valid_output parses as JSON; for json_schema cases it VALIDATES against the
  schema (via jsonschema if importable, else a structural fallback), and
  invalid_output FAILS validation (or is non-parseable).
- For gbnf cases, valid_output parses as JSON and invalid_output differs.
- Chat cases have a model + a non-empty messages list of {role, content} dicts.
"""

import json

import pytest

from tests.corpus.schemas import (
    CHAT_LOOP_CASES,
    STRUCTURED_CASES,
    StructuredCase,
)

try:
    import jsonschema  # type: ignore
    _HAVE_JSONSCHEMA = True
except ImportError:  # pragma: no cover - environment dependent
    jsonschema = None  # type: ignore
    _HAVE_JSONSCHEMA = False


# ── corpus non-emptiness ─────────────────────────────────────────────────────

def test_corpora_non_empty():
    assert STRUCTURED_CASES, "STRUCTURED_CASES must be non-empty"
    assert CHAT_LOOP_CASES, "CHAT_LOOP_CASES must be non-empty"
    assert 3 <= len(STRUCTURED_CASES) <= 6
    assert 3 <= len(CHAT_LOOP_CASES) <= 6


def test_structured_case_names_unique():
    names = [c.name for c in STRUCTURED_CASES]
    assert len(names) == len(set(names)), "StructuredCase names must be unique"


def test_chat_case_names_unique():
    names = [c["name"] for c in CHAT_LOOP_CASES]
    assert len(names) == len(set(names)), "chat case names must be unique"


# ── per-StructuredCase field + JSON checks ───────────────────────────────────

@pytest.mark.parametrize("case", STRUCTURED_CASES, ids=[c.name for c in STRUCTURED_CASES])
def test_structured_case_required_fields(case: StructuredCase):
    assert case.name and isinstance(case.name, str)
    assert case.source and isinstance(case.source, str)
    assert case.kind in ("gbnf", "json_schema"), case.kind
    assert case.prompt and isinstance(case.prompt, str)
    assert case.schema, "schema must be present"
    if case.kind == "gbnf":
        assert isinstance(case.schema, str) and "root" in case.schema
    else:
        assert isinstance(case.schema, dict) and case.schema.get("type")
    assert case.valid_output and isinstance(case.valid_output, str)
    assert case.invalid_output and isinstance(case.invalid_output, str)


@pytest.mark.parametrize("case", STRUCTURED_CASES, ids=[c.name for c in STRUCTURED_CASES])
def test_valid_output_parses(case: StructuredCase):
    # valid_output must be well-formed JSON for every case, gbnf or json_schema.
    obj = json.loads(case.valid_output)
    assert obj is not None


def _parses(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def _structural_json_schema_check(schema: dict, obj: object) -> bool:
    """Fallback validator when jsonschema isn't importable — checks the coarse
    shape (top-level type, required keys, enum membership on the first level of
    an object schema, and array-item required/enum). Intentionally lenient; the
    real gate is jsonschema when present."""
    stype = schema.get("type")
    if stype == "array":
        if not isinstance(obj, list):
            return False
        item_schema = schema.get("items") or {}
        return all(_structural_json_schema_check(item_schema, el) for el in obj)
    if stype == "object":
        if not isinstance(obj, dict):
            return False
        for req in schema.get("required", []):
            if req not in obj:
                return False
        for key, sub in (schema.get("properties") or {}).items():
            if key in obj and isinstance(sub, dict):
                enum = sub.get("enum")
                if enum is not None and obj[key] not in enum:
                    return False
                if sub.get("type") == "number" and isinstance(obj[key], (int, float)):
                    lo, hi = sub.get("minimum"), sub.get("maximum")
                    if lo is not None and obj[key] < lo:
                        return False
                    if hi is not None and obj[key] > hi:
                        return False
        # forbidden-fields ("not/anyOf/required") support for the investor case.
        neg = schema.get("not")
        if isinstance(neg, dict):
            for clause in neg.get("anyOf", []):
                for req in clause.get("required", []):
                    if req in obj:
                        return False
        return True
    return True  # unconstrained


def _validates(schema: dict, obj: object) -> bool:
    if _HAVE_JSONSCHEMA:
        try:
            jsonschema.validate(instance=obj, schema=schema)
            return True
        except jsonschema.ValidationError:
            return False
    return _structural_json_schema_check(schema, obj)


@pytest.mark.parametrize(
    "case",
    [c for c in STRUCTURED_CASES if c.kind == "json_schema"],
    ids=[c.name for c in STRUCTURED_CASES if c.kind == "json_schema"],
)
def test_json_schema_valid_output_validates(case: StructuredCase):
    obj = json.loads(case.valid_output)
    assert _validates(case.schema, obj), (
        f"{case.name}: valid_output must satisfy its schema"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in STRUCTURED_CASES if c.kind == "json_schema"],
    ids=[c.name for c in STRUCTURED_CASES if c.kind == "json_schema"],
)
def test_json_schema_invalid_output_fails(case: StructuredCase):
    # invalid_output must either fail to parse OR fail schema validation.
    if not _parses(case.invalid_output):
        return
    obj = json.loads(case.invalid_output)
    assert not _validates(case.schema, obj), (
        f"{case.name}: invalid_output must VIOLATE its schema"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in STRUCTURED_CASES if c.kind == "gbnf"],
    ids=[c.name for c in STRUCTURED_CASES if c.kind == "gbnf"],
)
def test_gbnf_valid_parses_and_invalid_differs(case: StructuredCase):
    # Light check: valid_output parses as JSON; invalid_output differs from it
    # (full GBNF validation is out of scope for the corpus smoke).
    json.loads(case.valid_output)
    assert case.invalid_output != case.valid_output


# ── chat-loop case shape ─────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CHAT_LOOP_CASES, ids=[c["name"] for c in CHAT_LOOP_CASES])
def test_chat_case_shape(case: dict):
    assert isinstance(case, dict)
    assert case.get("model"), "chat case must carry a model"
    msgs = case.get("messages")
    assert isinstance(msgs, list) and msgs, "messages must be a non-empty list"
    for m in msgs:
        assert isinstance(m, dict), "each message must be a dict"
        assert m.get("role"), "each message must have a role"
        # tool-call assistant turns carry content="" + tool_calls; that's valid.
        assert "content" in m, "each message must have a content key"


@pytest.mark.parametrize("case", CHAT_LOOP_CASES, ids=[c["name"] for c in CHAT_LOOP_CASES])
def test_chat_case_tool_arguments_parse(case: dict):
    # Any tool_call arguments and any tool-result content that claims to be JSON
    # must parse — keeps the synthesized inner-loop fixtures self-consistent.
    for m in case["messages"]:
        for tc in m.get("tool_calls", []) or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str) and args.strip():
                json.loads(args)
        if m.get("role") == "tool":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip().startswith(("{", "[")):
                json.loads(content)
