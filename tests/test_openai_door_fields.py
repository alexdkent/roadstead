"""``docs/api.md`` §1.1 says what each OpenAI-shaped door reads. This checks it. 🚨

§1.1 was ONE table until 2026-09-02, and it was the union of two doors presented
as one door's. It listed `agent_id`, `priority`, `call_site`, `caller_id`,
`request_id`, `session_id` and `turn_id` as accepted body fields on
``POST /v1/chat/completions``. `handle_openai_chat` reads exactly two things off
that body — `model` and `timeout_s` — and overwrites the identity fields from the
resolved principal; the leftovers travel to the backend inside the payload.
`caller_id` and `request_id` genuinely ARE read, on ``/rs/v1/chat``, which is what
made the section almost-right and therefore worse than plainly wrong: a migrator
sending `agent_id` here got no error, no effect, and no hint that the field they
wanted was on the other door.

`grammar` and `thinking` are the other half of "almost": they are not read by the
handler either, and they work anyway, because `correction.py` reads them off the
payload downstream. A guard that only checked the handler would have deleted two
rows that are true.

So the pin is per DOOR and per ROW, and the document carries the reader's name in
a **Read by** column. Two directions, and the second is the one that rots:

* every row that claims a reader is checked against that reader's source, and
* every body field a handler reads must appear in its door's table.

⚠️ It reads the DOCUMENT, so it is a pin on the prose, not a proof about
behaviour. What it can catch is the two ways this section went wrong before:
a row describing a field nothing reads, and a handler growing a field the
document never mentions.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
API_DOC = _ROOT / "docs" / "api.md"
_PKG = _ROOT / "roadstead"


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def section() -> str:
    doc = API_DOC.read_text(encoding="utf-8")
    assert "### 1.1 Request fields beyond the OpenAI API" in doc, (
        "docs/api.md §1.1 has been renamed or removed — this guard, and the "
        "reason it exists, need re-pointing rather than deleting")
    body = doc.split("### 1.1 ", 1)[1].split("### 1.2 ", 1)[0]
    return body


def _cells(line: str) -> list[str]:
    """Split one markdown table row.

    🚨 An ESCAPED pipe is not a column separator. Half these rows carry a type
    like ``bool \| int`` or ``string \| [string]``, and a naive ``split("|")``
    turns them into five-column rows that this parser then skips — silently,
    which made the first run report `input` and `thinking` as fields the
    handlers read and the document does not list. The rows it would drop are
    exactly the ones with an interesting type.
    """
    line = line.strip().replace("\\|", "\x00").strip("|")
    return [c.strip().replace("\x00", "|") for c in line.split("|")]


def _ticked(cell: str) -> list[str]:
    return re.findall(r"`([A-Za-z0-9_.]+)`", cell)


def _block(section: str, heading: str, *, stop: str = "\n####") -> str:
    assert heading in section, f"§1.1 no longer has the block {heading!r}"
    rest = section.split(heading, 1)[1]
    return rest.split(stop, 1)[0]


def _read_rows(block: str) -> dict[str, str]:
    """`Field | Type | Effect | Read by` rows -> {field: reader}."""
    out: dict[str, str] = {}
    for line in block.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = _cells(line)
        if len(cells) != 4 or cells[0] == "Field":
            continue
        readers = _ticked(cells[3])
        fields = _ticked(cells[0])
        assert len(readers) == 1, (
            f"§1.1 row {cells[0]!r} must name exactly one reader in its "
            f"`Read by` column, got {readers}")
        for f in fields:
            out[f] = readers[0]
    return out


def _ignored(block: str) -> set[str]:
    """`Field | What actually happens here` rows -> {field}."""
    out: set[str] = set()
    for line in block.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = _cells(line)
        if len(cells) != 2 or cells[0] == "Field":
            continue
        out |= set(_ticked(cells[0]))
    return out


# --------------------------------------------------------------------------- #
# The source
# --------------------------------------------------------------------------- #

def _body_reads(module: str, func: str) -> set[str]:
    """``body.get("x")`` / ``body.pop("x")`` inside one handler."""
    tree = ast.parse((_PKG / module).read_text(encoding="utf-8"))
    fns = [n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name == func]
    assert fns, f"roadstead/{module} has no function {func!r}"
    out: set[str] = set()
    for node in ast.walk(fns[0]):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "body"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
    return out


def _module_reads(module: str) -> set[str]:
    """Every ``<anything>.get("x")`` in a module.

    Deliberately loose: `grammar` and `thinking` are read off the PAYLOAD, at a
    dozen sites, through several receivers (`payload`, `p`, `eb`, `so`). What has
    to be true is that the module still knows the name — which is what a rename
    breaks, and a rename is what this row exists to catch.
    """
    tree = ast.parse((_PKG / module).read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.add(node.args[0].value)
    return out


#: door heading -> the handler whose body reads must match the table exactly.
_DOORS = {
    "#### `POST /v1/chat/completions`": ("http_handlers.py", "handle_openai_chat"),
    "#### `POST /v1/embeddings`": ("http_handlers.py", "handle_openai_embeddings"),
}


# --------------------------------------------------------------------------- #
# The guards
# --------------------------------------------------------------------------- #

def test_the_parser_actually_found_the_tables(section):
    """A parser that finds nothing makes every assertion below pass."""
    chat = _read_rows(_block(section, "#### `POST /v1/chat/completions`",
                             stop="\n#####"))
    embed = _read_rows(_block(section, "#### `POST /v1/embeddings`"))
    ignored = _ignored(_block(section, "##### What this door does NOT read"))
    assert {"model", "timeout_s", "grammar", "thinking"} <= set(chat), chat
    assert {"input", "encoding_format", "model"} <= set(embed), embed
    assert {"agent_id", "priority", "call_site", "caller_id",
            "session_id", "turn_id", "request_id"} <= ignored, sorted(ignored)


@pytest.mark.parametrize("heading", sorted(_DOORS))
def test_every_row_that_claims_a_reader_has_one(section, heading):
    """A row saying `Read by: X` where X does not read it is the defect this
    section had for its whole life, one door over."""
    stop = "\n#####" if "chat/completions" in heading else "\n####"
    for field, reader in _read_rows(_block(section, heading, stop=stop)).items():
        # `module.py` names a whole module; `module.function` names one
        # function. The distinction matters: a handler's reads are pinned
        # exactly, a downstream module's only loosely (see `_module_reads`).
        if reader.endswith(".py"):
            actual = _module_reads(reader)
            where = reader
        else:
            module, func = reader.split(".", 1)
            actual = _body_reads(f"{module}.py", func)
            where = f"{reader}()"
        assert field in actual, (
            f"docs/api.md §1.1 says `{field}` on `{heading}` is read by "
            f"{where}, and it is not. Either the field was renamed or the row "
            f"is describing a door that stopped reading it.")


@pytest.mark.parametrize("heading,handler", sorted(_DOORS.items()))
def test_the_table_lists_every_body_field_the_handler_reads(
        section, heading, handler):
    """The direction that rots. A handler growing a body field the document
    does not mention is how §1.1 drifts back into being a description of
    something else."""
    module, func = handler
    stop = "\n#####" if "chat/completions" in heading else "\n####"
    declared = {f for f, r in _read_rows(_block(section, heading, stop=stop)).items()
                if r == f"{module[:-3]}.{func}"}
    actual = _body_reads(module, func)
    assert actual == declared, (
        f"§1.1's table for `{heading}` and {func}() disagree about which body "
        f"fields it reads:\n"
        f"  read but undocumented: {sorted(actual - declared)}\n"
        f"  documented but unread: {sorted(declared - actual)}")


def test_the_ignored_fields_really_are_ignored(section):
    """🚨 The claim that costs a migrator most if it is wrong. Each of these is
    a field somebody expects to work, documented as not working."""
    ignored = _ignored(_block(section, "##### What this door does NOT read"))
    read = _body_reads("http_handlers.py", "handle_openai_chat")
    both = sorted(ignored & read)
    assert not both, (
        f"§1.1 documents {both} as ignored by handle_openai_chat, and it reads "
        f"{'it' if len(both) == 1 else 'them'} — the table now understates the "
        f"door, which is the same defect it was rewritten to fix, reversed")


def test_the_enriched_door_really_does_read_what_1_1_sends_readers_to_it_for(
        section):
    """§1.1 tells a migrator that `caller_id`, `request_id` and `priority` are
    read on `/rs/v1/chat`. If that stopped being true, this section would be
    pointing them at a second door where the field also does nothing."""
    assert "#### `POST /rs/v1/chat`" in section
    enriched = _module_reads("enriched.py")
    for field in ("caller_id", "request_id", "priority", "session_id",
                  "turn_id", "call_site"):
        assert field in enriched, (
            f"§1.1 sends callers to /rs/v1/chat for `{field}`, and "
            f"enriched.py no longer reads it")
