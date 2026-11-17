"""
tests/test_phase2_langsmith.py — Phase 2 tests for optional LangSmith tracing configuration.
"""

import os
import agent


def test_langsmith_noop_when_unset(monkeypatch):
    """When LANGCHAIN_TRACING_V2 is unset or false, configure_langsmith_tracing is a no-op."""
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert agent.configure_langsmith_tracing() is False


def test_langsmith_disables_gracefully_if_key_missing(monkeypatch):
    """If LANGCHAIN_TRACING_V2=true but no API key is set, tracing is disabled with a warning."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert agent.configure_langsmith_tracing() is False
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


def test_langsmith_enables_when_env_vars_set(monkeypatch):
    """When standard LangSmith variables are set, tracing activates."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test_dummy_key")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "CarScout_Test")

    assert agent.configure_langsmith_tracing() is True
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_PROJECT") == "CarScout_Test"
