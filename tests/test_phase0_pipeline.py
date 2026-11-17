"""
tests/test_phase0_pipeline.py — Phase 0 tests for pipeline resilience and workflow fixes.
"""

from unittest.mock import patch
import main


def test_alert_channel_failures_are_non_fatal(monkeypatch):
    """Alert failures (Telegram or Email) must log a warning and not append to results['errors']."""
    monkeypatch.setenv("ENABLE_TELEGRAM", "true")
    monkeypatch.setenv("ENABLE_EMAIL", "true")

    with (
        patch("main.init_db"),
        patch("scraper.scrape_listings", return_value=[{"listing_url": "http://test/1", "price": 1000000}]),
        patch("db.bulk_upsert_listings", return_value=(1, 1)),
        patch("deal_detector.detect_deals", return_value=[{"listing_url": "http://test/1", "price": 1000000}]),
        patch("agent.evaluate_deals", return_value=[{"listing_url": "http://test/1", "price": 1000000, "deal_score": 8}]),
        patch("telegram_bot.send_alerts", side_effect=RuntimeError("Telegram network failure")),
        patch("email_alert.send_alerts", side_effect=RuntimeError("SMTP auth failure")),
        patch("main.get_stats", return_value={"total_listings": 1, "deals_found": 1, "alerts_sent": 0}),
    ):
        results = main.run_pipeline()

    assert results["errors"] == []
    assert results["listings_scraped"] == 1
    assert results["deals_qualified"] == 1
    assert results["telegram_sent"] == 0
    assert results["email_sent"] == 0


def test_scraper_or_db_failure_is_fatal(monkeypatch):
    """Scraping or DB failure must record an error in results['errors'] and stop early."""
    with (
        patch("main.init_db"),
        patch("scraper.scrape_listings", side_effect=RuntimeError("Connection reset")),
    ):
        results = main.run_pipeline()

    assert len(results["errors"]) == 1
    assert "Scraper failed" in results["errors"][0]


def test_empty_scrape_graceful_exit(monkeypatch):
    """When PakWheels blocks the runner IP and returns 0 listings, pipeline finishes cleanly."""
    monkeypatch.setenv("ENABLE_TELEGRAM", "false")
    monkeypatch.setenv("ENABLE_EMAIL", "false")

    with (
        patch("main.init_db"),
        patch("scraper.scrape_listings", return_value=[]),
        patch("db.bulk_upsert_listings", return_value=(0, 0)),
        patch("deal_detector.detect_deals", return_value=[]),
        patch("agent.evaluate_deals", return_value=[]),
        patch("main.get_stats", return_value={"total_listings": 0, "deals_found": 0, "alerts_sent": 0}),
    ):
        results = main.run_pipeline()

    assert results["errors"] == []
    assert results["listings_scraped"] == 0
    assert results["deals_detected"] == 0


def test_disable_alerts_env_respected(monkeypatch):
    """Setting ENABLE_TELEGRAM=false and ENABLE_EMAIL=false skips alert senders."""
    monkeypatch.setenv("ENABLE_TELEGRAM", "false")
    monkeypatch.setenv("ENABLE_EMAIL", "false")

    with (
        patch("main.init_db"),
        patch("scraper.scrape_listings", return_value=[]),
        patch("db.bulk_upsert_listings", return_value=(0, 0)),
        patch("deal_detector.detect_deals", return_value=[]),
        patch("agent.evaluate_deals", return_value=[]),
        patch("telegram_bot.send_alerts") as mock_tg,
        patch("email_alert.send_alerts") as mock_em,
        patch("main.get_stats", return_value={"total_listings": 0, "deals_found": 0, "alerts_sent": 0}),
    ):
        results = main.run_pipeline()

    mock_tg.assert_not_called()
    mock_em.assert_not_called()
    assert results["errors"] == []
