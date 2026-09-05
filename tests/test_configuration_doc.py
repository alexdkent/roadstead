"""`docs/configuration.md` lists every variable, and only variables that exist.

🚨 **The defect this exists for is a reference that stopped being true.** On
2026-09-02 an operator set `ROADSTEAD_QUEUE_DB` — the spelling the documentation
used — and it was a silent no-op, because the entry point had been left out of a
rename and was still reading `LLM_PROXY_QUEUE_DB`. Nothing failed. The proxy
wrote its durable record somewhere else and said nothing, and the person who
paid for it was the one who followed the docs.

`tests/test_env_var_naming.py` closed the half of that about *spellings*. This
closes the other half: that the document naming them is complete and has no
ghosts. Both directions matter, and for different reasons —

* **code → doc.** A variable the code reads and the reference omits is
  undiscoverable. The operator's only route to it is reading the source, which
  is the thing a reference exists to make unnecessary. Satisfied only by a
  TABLE ROW — see `_DOC_ROW` for the sabotage that proved a prose mention is
  not enough.
* **doc → code.** A variable the reference names and nothing reads is worse than
  an omission: it is confidently wrong. An operator sets it, sees no error, and
  concludes the setting took effect.

**Read as TEXT, comments included, and deliberately.** A variable named only in
a comment (`ROADSTEAD_TIMEOUT_ADVICE_CAP_S`) or emitted only in a log line
(`ROADSTEAD_SPILL`) is still a `ROADSTEAD_`-shaped string a reader will grep and
have to adjudicate. The document has a section for exactly those, so they are
covered rather than exempted — an exemption list is the thing that quietly
grows.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
PKG = REPO / "roadstead"
DOC = REPO / "docs" / "configuration.md"

#: Any `ROADSTEAD_`-prefixed token, wherever it appears.
_TOKEN = re.compile(r"ROADSTEAD_[A-Z0-9_]*")

#: 🚨 A variable is LISTED only by leading a table row. Prose is not a listing,
#: and accepting it is not a hypothetical weakness: written first as "appears
#: anywhere in the document", this check stayed green when the
#: `ROADSTEAD_QUEUE_DB` row was deleted outright, because the name also occurs
#: in a sentence three sections earlier. A mention tells a reader the variable
#: exists; only the row tells them its type, its default and what it does.
_DOC_ROW = re.compile(r"^\|\s*`(ROADSTEAD_[A-Z0-9_]+)`\s*\|", re.M)

#: Every backticked mention, ANYWHERE — rows and prose alike. Used only for the
#: doc→code direction, which is deliberately the wider net: a name that exists
#: nowhere is just as misleading in a sentence as in a table.
_DOC_MENTION = re.compile(r"`(ROADSTEAD_[A-Z0-9_]*)[^`]*`")

#: 🚨 The bare prefix, and the ONE reason it has to be handled explicitly rather
#: than filtered away as noise. `config.env_with_legacy_prefix` builds its
#: variable name at runtime — `os.environ.get("ROADSTEAD_" + name)` — so the
#: fourteen variables it reads leave NO literal in the source at all. A scan
#: that only looked for literals would report them as undocumented if the doc
#: listed them, and as non-existent if it did not: wrong in both directions at
#: once. They are recovered from the helper's call sites instead (below), and
#: the prefix itself is the residue of that construction, never a variable.
_BARE_PREFIX = "ROADSTEAD_"

#: The two helpers that prepend the prefix. `_env` is `__main__`'s import alias
#: for the second one; both spellings are call sites of the same function.
_HELPER_CALL = re.compile(
    r"\b(?:_env|env_with_legacy_prefix)\(\s*\n?\s*\"([A-Z0-9_]+)\"")

#: 🚨 An f-string template — `f"ROADSTEAD_{suffix}"` — would be a THIRD way to
#: name a variable with no literal, and unlike the helper above it would be
#: unrecoverable: the suffix is a runtime value. There are none today, and this
#: test refuses to let one arrive unnoticed, because it would silently punch a
#: hole in both directions of the check.
_FSTRING_TEMPLATE = re.compile(r"f\"ROADSTEAD_\{|f'ROADSTEAD_\{")


def _source_files() -> list[pathlib.Path]:
    """The shipped tree: modules, and the YAML that ships beside them.

    `agents.yaml` and `models.yaml` are package data — a reader meets them
    before they meet the source, and both name variables in their comments.
    """
    return sorted(
        p for pattern in ("*.py", "*.yaml")
        for p in PKG.rglob(pattern)
        if "__pycache__" not in p.parts
    )


def _names_the_code_names() -> dict[str, str]:
    """Every variable name in the package → the first file that names it."""
    found: dict[str, str] = {}
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(REPO))
        for name in _TOKEN.findall(text):
            if name == _BARE_PREFIX:
                continue
            found.setdefault(name, rel)
        for suffix in _HELPER_CALL.findall(text):
            found.setdefault(_BARE_PREFIX + suffix, rel)
    return found


def _names_the_doc_lists() -> set[str]:
    """Names with a table row of their own."""
    return set(_DOC_ROW.findall(DOC.read_text(encoding="utf-8")))


def _names_the_doc_mentions() -> set[str]:
    """Names the document utters at all."""
    return {m for m in _DOC_MENTION.findall(DOC.read_text(encoding="utf-8"))
            if m != _BARE_PREFIX}


def test_every_variable_the_code_reads_is_in_the_reference():
    code = _names_the_code_names()
    missing = sorted(set(code) - _names_the_doc_lists())
    assert not missing, (
        "these names appear in roadstead/ and not in docs/configuration.md:\n  "
        + "\n  ".join(f"{n}  ({code[n]})" for n in missing)
        + "\n\nAdd a row — group, type, default, and what it does. If it is not "
          "actually an environment variable (a log marker, a name this process "
          "exports rather than reads), it goes in the document's 'Names that "
          "look like configuration and are not' table, which is there so the "
          "answer is written down instead of re-derived.")


def test_every_variable_the_reference_names_exists_in_the_code():
    code = set(_names_the_code_names())
    ghosts = sorted(_names_the_doc_mentions() - code)
    assert not ghosts, (
        "docs/configuration.md names variables that nothing in roadstead/ "
        "mentions:\n  " + "\n  ".join(ghosts)
        + "\n\nA documented variable that is read nowhere is worse than an "
          "undocumented one: an operator sets it, gets no error, and believes "
          "it took effect. Remove the row, or fix the spelling.")


def test_the_helper_built_names_are_actually_recovered():
    """🚨 The half a literal scan cannot see, asserted directly.

    `ROADSTEAD_QUEUE_DB` — the variable the original defect was about — exists
    in the source only as `_env("QUEUE_DB", …)`. If `_HELPER_CALL` ever stops
    matching (the helper is renamed, the call is reformatted onto a shape the
    pattern misses), both tests above would keep passing while going blind to
    fourteen variables. This is what makes that failure loud.
    """
    recovered = {n for n, _ in _names_the_code_names().items()}
    for name in ("ROADSTEAD_QUEUE_DB", "ROADSTEAD_LOG_DIR", "ROADSTEAD_PORT",
                 "ROADSTEAD_REQUEST_LOG_BACKUPS"):
        assert name in recovered, (
            f"{name} was not recovered from a helper call site — the scan has "
            f"gone blind to every variable built as ROADSTEAD_ + <suffix>")


def test_no_variable_name_is_built_from_an_fstring_template():
    """A name assembled at runtime cannot be checked by anything here."""
    offenders = [
        str(p.relative_to(REPO)) for p in _source_files()
        if _FSTRING_TEMPLATE.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "a variable name is built with an f-string template in:\n  "
        + "\n  ".join(offenders)
        + "\n\nThe suffix is a runtime value, so neither direction of this "
          "check can see the resulting variable. Name it literally, or read it "
          "through config.env_with_legacy_prefix() so the suffix is a constant "
          "at the call site.")


def test_the_scan_is_not_vacuous():
    """A guard that matches nothing passes forever.

    Both sides are asserted, not just one: an empty doc side would make the
    code→doc test fail loudly, but an empty CODE side would make BOTH tests
    pass on a document full of ghosts.
    """
    code = _names_the_code_names()
    rows = _names_the_doc_lists()
    assert len(code) > 40, f"only {len(code)} names found in roadstead/"
    assert len(rows) > 40, f"only {len(rows)} rows found in {DOC.name}"
    assert _names_the_doc_mentions() >= rows, (
        "the wider mention pattern no longer covers the row pattern — one of "
        "the two has drifted and the doc→code direction has gone partly blind")


#: A row's stability marker, read out of the second cell.
_DOC_ROW_WITH_MARK = re.compile(
    r"^\|\s*`(ROADSTEAD_[A-Z0-9_]+)`\s*\|\s*(🔒|🔓)\s*\|", re.M)


def test_the_stable_column_is_exactly_what_the_contract_document_names():
    """🔒 is not a judgement call, and this is what stops it becoming one.

    `docs/compatibility.md` marks one thing stable — the contract in
    `docs/api.md` — so a variable is contract iff that document names it. Any
    other rule would be somebody's opinion, and an opinion in a column that
    reads as a promise is how a 🔓 name ends up in somebody's tooling.

    🚨 **A failure here is a decision, not a typo.** If `docs/api.md` gained a
    variable, that variable just became part of the wire contract and this
    reference has to say so. If it lost one, the guarantee was withdrawn and
    needs a `CHANGELOG` entry under `### Breaking` before the marker moves.
    """
    contract = set(_TOKEN.findall((REPO / "docs" / "api.md").read_text(
        encoding="utf-8"))) - {_BARE_PREFIX}
    marked = {name: mark for name, mark in
              _DOC_ROW_WITH_MARK.findall(DOC.read_text(encoding="utf-8"))}

    unmarked = sorted(n for n in contract if marked.get(n) != "🔒")
    overclaimed = sorted(n for n, m in marked.items()
                         if m == "🔒" and n not in contract)

    assert not unmarked and not overclaimed, (
        f"the 🔒 column disagrees with docs/api.md.\n"
        f"  named in api.md but not marked 🔒: {unmarked}\n"
        f"  marked 🔒 but absent from api.md: {overclaimed}\n"
        "docs/compatibility.md pins api.md and nothing else, so that document "
        "IS the stable set. Move the marker, and if a guarantee is being "
        "withdrawn rather than added, write the CHANGELOG entry first.")


# --------------------------------------------------------------------------- #
# The default data directory — one function, and the callers that must use it.
# --------------------------------------------------------------------------- #
#
# 🚨 Both halves matter and they fail differently. A wrong DEFAULT puts the
# durable record somewhere the operator did not choose; a caller with its OWN
# copy of the default puts it somewhere the operator DID choose and then reads
# it from somewhere else. The second was live until 2026-09-05: `roadstead test
# replay` and `ab` rebuilt the path themselves, so a deployment that had moved
# its data dir had its own replay tooling open an empty database in /tmp, with
# nothing anywhere saying why.

def test_the_default_data_dir_follows_XDG_STATE_HOME(monkeypatch):
    from roadstead.__main__ import default_data_dir

    monkeypatch.setenv("XDG_STATE_HOME", "/srv/state")
    assert default_data_dir() == "/srv/state/roadstead"


def test_the_default_data_dir_falls_back_to_the_home_state_dir(monkeypatch):
    """`~/.local/state/roadstead` — the XDG spec's own default for the variable."""
    from roadstead.__main__ import default_data_dir

    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/someone")
    assert default_data_dir() == "/home/someone/.local/state/roadstead"


def test_an_empty_XDG_STATE_HOME_is_not_a_setting(monkeypatch):
    """The spec is explicit that an empty value means unset.

    Without this, `XDG_STATE_HOME=` (a compose file with an unset interpolation)
    would resolve the durable record to `/roadstead`, at the filesystem root.
    """
    from roadstead.__main__ import default_data_dir

    monkeypatch.setenv("XDG_STATE_HOME", "")
    monkeypatch.setenv("HOME", "/home/someone")
    assert default_data_dir() == "/home/someone/.local/state/roadstead"


def test_an_unresolvable_home_gives_an_ABSOLUTE_path_and_says_so(
        monkeypatch, caplog):
    """🚨 The distroless / arbitrary-`runAsUser` case.

    `os.path.expanduser` returns its argument UNCHANGED when it cannot resolve
    `~`, so the naive result is the RELATIVE path `~/.local/state/roadstead` and
    the durable record lands under the working directory — a place nobody looks
    and nothing reports. The contract is: absolute, and loud.
    """
    from roadstead.__main__ import default_data_dir

    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", lambda p: p)

    with caplog.at_level("WARNING"):
        result = default_data_dir()

    import os as _os
    assert _os.path.isabs(result), result
    assert "ROADSTEAD_DATA_DIR" in caplog.text, (
        "the warning must name the way out, or it is only an observation")


def test_the_test_subcommands_resolve_the_default_through_the_same_function():
    """Asserted on the SOURCE, because the alternative is running a replay.

    Two things, and the second is the one that regresses: `_test_cli` must call
    `default_data_dir`, and it must contain no path literal of its own. A second
    copy of the default is not a duplicate constant — it is a tool that reads a
    different database from the one the proxy writes.
    """
    import ast
    import inspect

    from roadstead import __main__ as entry

    fn = next(node for node in ast.walk(ast.parse(inspect.getsource(entry)))
              if isinstance(node, ast.FunctionDef) and node.name == "_test_cli")

    calls = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "default_data_dir" in calls, (
        "`roadstead test` no longer resolves the data dir through "
        "default_data_dir() — it has grown its own copy of the default again")

    literals = [n.value for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value.startswith("/")]
    assert not literals, (
        f"`_test_cli` carries absolute path literal(s) {literals} — the default "
        f"belongs in default_data_dir() and nowhere else")
