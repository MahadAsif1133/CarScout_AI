"""
tests/test_phase5_ollama.py — Phase 5 tests for optional Ollama LLM provider in agent.py.
"""

import json
from unittest.mock import MagicMock, patch
import requests
import agent


MOCK_LISTING = {
    "listing_url": "https://www.pakwheels.com/used-cars/honda-civic-2020-456",
    "make": "Honda",
    "model": "Civic",
    "year": 2020,
    "price": 3800000,
    "predicted_price": 4600000,
    "price_gap_pct": 17.4,
    "mileage": 42000,
    "city": "Karachi",
    "transmission": "Automatic",
    "engine_type": "Petrol 1800cc",
    "description": "First owner, clean car, complete documents available.",
}


def test_ollama_provider_evaluates_deal_with_same_output_schema(monkeypatch):
    """When AGENT_PROVIDER=ollama, evaluate_single_deal calls Ollama and returns DealEvaluation."""
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    agent._chain = None

    ollama_json = json.dumps({
        "deal_score": 8,
        "price_gap_percent": 17.4,
        "risk_flags": [],
        "summary": "Clean first-owner Civic priced 17.4% below fair value.",
    })
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"message": {"content": ollama_json}}

    with patch("agent.requests.post", return_value=mock_resp) as mock_post:
        evaluation = agent.evaluate_single_deal(MOCK_LISTING)

    assert evaluation is not None
    assert isinstance(evaluation, agent.DealEvaluation)
    assert evaluation.deal_score == 8
    assert evaluation.price_gap_percent == 17.4
    assert evaluation.risk_flags == []
    assert "Civic" in evaluation.summary

    mock_post.assert_called_once()
    call_url = mock_post.call_args[0][0]
    call_payload = mock_post.call_args[1]["json"]
    assert call_url == "http://localhost:11434/api/chat"
    assert call_payload["model"] == "qwen2.5:3b"
    assert call_payload["format"] == "json"


def test_ollama_unreachable_handled_gracefully(monkeypatch):
    """When Ollama is unreachable, evaluate_single_deal returns None and evaluate_deals assigns fallback score."""
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    agent._chain = None

    with (
        patch("agent.requests.post", side_effect=requests.ConnectionError("Ollama connection refused")),
        patch("agent.get_all_deals", return_value=[MOCK_LISTING]),
        patch("agent.update_agent_result") as mock_update,
    ):
        single = agent.evaluate_single_deal(MOCK_LISTING)
        qualifying = agent.evaluate_deals(agent_threshold=7)

    assert single is None
    assert qualifying == []
    mock_update.assert_called_once()
    assert mock_update.call_args[1]["deal_score"] == 5
