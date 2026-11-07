from types import SimpleNamespace

import pytest

import start_app


@pytest.fixture
def launched(monkeypatch):
    calls = []

    def fake_run(command, cwd):
        calls.append((command, cwd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(start_app.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("args, expected", [
    ([], "app/frontend/streamlit_app.py"),
    (["api"], "app.backend.main:app"),
])
def test_launches_target_from_project_root(launched, args, expected):
    assert start_app.main(args) == 0
    command, cwd = launched[0]
    assert expected in command
    assert cwd == start_app.ROOT
    assert (cwd / "app" / "frontend" / "streamlit_app.py").is_file()


def test_unknown_target_is_rejected(launched, capsys):
    assert start_app.main(["nope"]) == 2
    assert launched == []
    assert "Usage" in capsys.readouterr().out
