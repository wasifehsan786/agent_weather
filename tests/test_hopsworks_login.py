import sys
import types

import pytest

from agentweather import hopsworks_login


def _fake_hopsworks(monkeypatch, fail_times):
    calls = {"n": 0}

    def login(**kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError("503")
        return "project"

    monkeypatch.setitem(sys.modules, "hopsworks", types.SimpleNamespace(login=login))
    monkeypatch.setenv("HOPSWORKS_API_KEY", "x")
    monkeypatch.setattr(hopsworks_login.time, "sleep", lambda s: None)
    return calls


def test_login_retries_then_succeeds(monkeypatch):
    calls = _fake_hopsworks(monkeypatch, fail_times=2)
    assert hopsworks_login.login(attempts=5) == "project"
    assert calls["n"] == 3


def test_login_gives_up_after_all_attempts(monkeypatch):
    calls = _fake_hopsworks(monkeypatch, fail_times=99)
    with pytest.raises(RuntimeError):
        hopsworks_login.login(attempts=3)
    assert calls["n"] == 3


def test_missing_secret_is_not_retried(monkeypatch):
    calls = _fake_hopsworks(monkeypatch, fail_times=0)
    monkeypatch.delenv("HOPSWORKS_API_KEY")
    with pytest.raises(KeyError):
        hopsworks_login.login(attempts=3)
    assert calls["n"] == 0
