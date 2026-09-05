"""The two ways in must be the same CLI.

`roadstead` (the console script) calls `roadstead.__main__:main` directly and
never executes the module's `if __name__ == "__main__"` guard — so for as long
as the `test` subcommand was dispatched from that guard, `python -m roadstead
test sim all` ran the harness and `roadstead test sim all` died in the server
parser with *"unrecognized arguments: sim all"*. One command, two answers,
decided by how the process happened to be started.

These tests drive `main()` with an explicit argv, which is the entry point the
console script uses, so a dispatch that only works under `python -m` cannot
pass them.
"""
from __future__ import annotations

from importlib.metadata import entry_points

import pytest

import roadstead
from roadstead.__main__ import main


def test_the_console_script_points_at_main():
    """The premise of every test below: `roadstead` is `main`, so exercising
    `main` exercises what an installed user types."""
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get("roadstead") == "roadstead.__main__:main"


def test_test_subcommand_reaches_the_test_cli(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["test", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: roadstead test" in out
    assert "sim" in out and "replay" in out and "ab" in out


def test_test_subcommand_passes_its_own_arguments_down(capsys):
    """Not just the dispatch — the tail of argv has to arrive as well. The
    harness parser reads what it is given rather than slicing `sys.argv`, which
    under pytest is pytest's own command line."""
    with pytest.raises(SystemExit) as exc:
        main(["test", "sim", "--help"])
    assert exc.value.code == 0
    assert "usage: roadstead test sim" in capsys.readouterr().out


def test_version_flag_prints_the_installed_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"roadstead {roadstead.__version__}"


def test_the_server_parser_still_rejects_an_unknown_subcommand(capsys):
    """`test` is dispatched before the server parser sees argv; nothing else is.
    A typo must still be refused rather than silently starting a server."""
    with pytest.raises(SystemExit) as exc:
        main(["tset"])
    assert exc.value.code != 0
    assert "unrecognized arguments" in capsys.readouterr().err
