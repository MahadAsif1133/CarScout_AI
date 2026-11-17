"""
tests/test_phase3_webhook.py — Phase 3 tests for n8n webhook delivery and fallback behavior.
"""

from unittest.mock import MagicMock, patch
import requests
import webhook_alert
import graph_pipeline


SAMPLE_DEAL = {
    "listing_url": "https://www.pakwheels.com/used-cars/toyota-corolla-2019-123",
    "make": "Toyota",
    "model": "Corolla",
    "year": 2019,
    "mileage": 55000,
    "city": "Lahore",
    "price": 3100000,
    "predicted_price": 3800000,
    "price_gap_pct": 18.42,
    "deal_score": 8,
    "agent_summary": "Great deal with clean docs",
    "risk_flags": ["engine_issues"],
    "prediction_method": "exact_model_year",
    "prediction_level": 2,
    "confidence_text": "Based on 12 similar listings",
}


def test_webhook_payload_uses_real_db_field_names():
    """Payload must include real database fields and computed title."""
    payload = webhook_alert.build_webhook_payload(SAMPLE_DEAL)
    assert payload["title"] == "Toyota Corolla 2019"
    assert payload["listing_url"] == SAMPLE_DEAL["listing_url"]
    assert payload["price"] == 3100000
    assert payload["predicted_price"] == 3800000
    assert payload["price_gap_pct"] == 18.42
    assert payload["deal_score"] == 8
    assert payload["agent_summary"] == "Great deal with clean docs"
    assert payload["risk_flags"] == ["engine_issues"]


def test_webhook_success_skips_direct_telegram_and_email(monkeypatch):
    """When N8N_WEBHOOK_URL succeeds, direct Telegram and Email senders are not called for that deal."""
    monkeypatch.setenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/carscout-alerts")
    monkeypatch.setenv("N8N_WEBHOOK_SECRET", "top_secret_123")
    monkeypatch.setenv("ENABLE_TELEGRAM", "true")
    monkeypatch.setenv("ENABLE_EMAIL", "true")

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None

    with (
        patch("webhook_alert.requests.post", return_value=mock_resp) as mock_post,
        patch("webhook_alert.log_alert"),
        patch("webhook_alert.mark_alerted"),
        patch("telegram_bot.send_alerts") as mock_tg,
        patch("email_alert.send_alerts") as mock_em,
    ):
        state = {
            "agent_threshold": 7,
            "enable_telegram": True,
            "enable_email": True,
            "qualifying_deals": [SAMPLE_DEAL],
        }
        out = graph_pipeline.alert_node(state)

    assert out["webhook_sent"] == 1
    assert out["telegram_sent"] == 0
    assert out["email_sent"] == 0
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["X-Webhook-Secret"] == "top_secret_123"
    mock_tg.assert_not_called()
    mock_em.assert_not_called()


def test_webhook_failure_retries_and_falls_back_to_direct_alerts(monkeypatch):
    """When N8N_WEBHOOK_URL fails after retry, alert_node falls back to Telegram and Email."""
    monkeypatch.setenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/carscout-alerts")
    monkeypatch.setenv("ENABLE_TELEGRAM", "true")
    monkeypatch.setenv("ENABLE_EMAIL", "true")

    with (
        patch("webhook_alert.requests.post", side_effect=requests.ConnectionError("n8n down")) as mock_post,
        patch("telegram_bot.send_alerts", return_value=1) as mock_tg,
        patch("email_alert.send_alerts", return_value=1) as mock_em,
    ):
        state = {
            "agent_threshold": 7,
            "enable_telegram": True,
            "enable_email": True,
            "qualifying_deals": [SAMPLE_DEAL],
        }
        out = graph_pipeline.alert_node(state)

    assert mock_post.call_count == webhook_alert.WEBHOOK_MAX_ATTEMPTS
    assert out["webhook_sent"] == 0
    assert out["telegram_sent"] == 1
    assert out["email_sent"] == 1
    mock_tg.assert_called_once()
    mock_em.assert_called_once()
