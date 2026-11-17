"""
tests/test_phase1_langgraph.py — Phase 1 tests for LangGraph orchestration and conditional routing.
"""

from unittest.mock import patch
import main
import graph_pipeline


def test_graph_stops_early_when_scrape_returns_nothing(monkeypatch):
    """When scrape returns 0 listings, predict/detect_deals/evaluate/alert must not run."""
    monkeypatch.setenv("PIPELINE_ENGINE", "langgraph")
    monkeypatch.setenv("ENABLE_TELEGRAM", "true")
    monkeypatch.setenv("ENABLE_EMAIL", "true")

    with (
        patch("graph_pipeline.init_db"),
        patch("scraper.scrape_listings", return_value=[]),
        patch("deal_detector.detect_deals") as mock_detect,
        patch("agent.evaluate_deals") as mock_eval,
        patch("telegram_bot.send_alerts") as mock_tg,
        patch("graph_pipeline.get_stats", return_value={"total_listings": 0, "deals_found": 0, "alerts_sent": 0}),
    ):
        results = main.run_pipeline()

    assert results["listings_scraped"] == 0
    assert results["deals_detected"] == 0
    assert results["errors"] == []
    mock_detect.assert_not_called()
    mock_eval.assert_not_called()
    mock_tg.assert_not_called()


def test_graph_stops_early_when_no_deals_pass_threshold(monkeypatch):
    """When detect_deals returns 0 candidate deals, evaluate and alert nodes are skipped."""
    monkeypatch.setenv("PIPELINE_ENGINE", "langgraph")

    sample_listings = [{"listing_url": "https://www.pakwheels.com/used-cars/car-1", "price": 2500000}]
    with (
        patch("graph_pipeline.init_db"),
        patch("scraper.scrape_listings", return_value=sample_listings),
        patch("graph_pipeline.bulk_upsert_listings", return_value=(1, 1)),
        patch("graph_pipeline.reload_statistical_data"),
        patch("graph_pipeline.get_new_listings", return_value=sample_listings),
        patch("deal_detector.detect_deals", return_value=[]),
        patch("agent.evaluate_deals") as mock_eval,
        patch("telegram_bot.send_alerts") as mock_tg,
        patch("graph_pipeline.get_stats", return_value={"total_listings": 1, "deals_found": 0, "alerts_sent": 0}),
    ):
        results = graph_pipeline.run_graph_pipeline()

    assert results["listings_scraped"] == 1
    assert results["deals_detected"] == 0
    assert results["deals_qualified"] == 0
    mock_eval.assert_not_called()
    mock_tg.assert_not_called()


def test_graph_skips_alerts_below_agent_threshold(monkeypatch):
    """Deals scoring below AGENT_THRESHOLD do not trigger the alert node."""
    monkeypatch.setenv("AGENT_THRESHOLD", "7")

    sample_listings = [{"listing_url": "https://www.pakwheels.com/used-cars/car-2", "price": 2000000}]
    low_score_deal = [{"listing_url": "https://www.pakwheels.com/used-cars/car-2", "price": 2000000, "deal_score": 5}]
    with (
        patch("graph_pipeline.init_db"),
        patch("scraper.scrape_listings", return_value=sample_listings),
        patch("graph_pipeline.bulk_upsert_listings", return_value=(1, 1)),
        patch("graph_pipeline.reload_statistical_data"),
        patch("graph_pipeline.get_new_listings", return_value=sample_listings),
        patch("deal_detector.detect_deals", return_value=sample_listings),
        patch("agent.evaluate_deals", return_value=low_score_deal),
        patch("telegram_bot.send_alerts") as mock_tg,
        patch("email_alert.send_alerts") as mock_em,
        patch("graph_pipeline.get_stats", return_value={"total_listings": 1, "deals_found": 1, "alerts_sent": 0}),
    ):
        results = graph_pipeline.run_graph_pipeline()

    assert results["deals_detected"] == 1
    assert results["deals_qualified"] == 0
    mock_tg.assert_not_called()
    mock_em.assert_not_called()


def test_graph_routes_some_deals_to_alerts(monkeypatch):
    """When qualifying deals pass AGENT_THRESHOLD, alert node sends Telegram and email alerts."""
    monkeypatch.setenv("AGENT_THRESHOLD", "7")
    monkeypatch.setenv("ENABLE_TELEGRAM", "true")
    monkeypatch.setenv("ENABLE_EMAIL", "true")

    sample_listings = [{"listing_url": "https://www.pakwheels.com/used-cars/car-3", "price": 2000000}]
    high_score_deal = [{"listing_url": "https://www.pakwheels.com/used-cars/car-3", "price": 2000000, "deal_score": 9}]

    with (
        patch("graph_pipeline.init_db"),
        patch("scraper.scrape_listings", return_value=sample_listings),
        patch("graph_pipeline.bulk_upsert_listings", return_value=(1, 1)),
        patch("graph_pipeline.reload_statistical_data"),
        patch("graph_pipeline.get_new_listings", return_value=sample_listings),
        patch("deal_detector.detect_deals", return_value=sample_listings),
        patch("agent.evaluate_deals", return_value=high_score_deal),
        patch("telegram_bot.send_alerts", return_value=1) as mock_tg,
        patch("email_alert.send_alerts", return_value=1) as mock_em,
        patch("graph_pipeline.get_stats", return_value={"total_listings": 1, "deals_found": 1, "alerts_sent": 2}),
    ):
        results = graph_pipeline.run_graph_pipeline()

    assert results["deals_qualified"] == 1
    assert results["telegram_sent"] == 1
    assert results["email_sent"] == 1
    mock_tg.assert_called_once()
    mock_em.assert_called_once()


def test_graph_retry_on_scrape_and_evaluate_failures(monkeypatch):
    """Transient errors in scrape and evaluate nodes are retried and succeed on subsequent attempt."""
    monkeypatch.setenv("ENABLE_TELEGRAM", "false")
    monkeypatch.setenv("ENABLE_EMAIL", "false")

    sample_listings = [{"listing_url": "https://www.pakwheels.com/used-cars/car-4", "price": 2000000}]
    high_score_deal = [{"listing_url": "https://www.pakwheels.com/used-cars/car-4", "price": 2000000, "deal_score": 8}]

    scrape_calls = {"count": 0}
    eval_calls = {"count": 0}

    def flaky_scrape(*args, **kwargs):
        scrape_calls["count"] += 1
        if scrape_calls["count"] == 1:
            raise RuntimeError("Transient network timeout")
        return sample_listings

    def flaky_eval(*args, **kwargs):
        eval_calls["count"] += 1
        if eval_calls["count"] == 1:
            raise RuntimeError("Transient LLM rate limit")
        return high_score_deal

    with (
        patch("graph_pipeline.init_db"),
        patch("scraper.scrape_listings", side_effect=flaky_scrape),
        patch("graph_pipeline.bulk_upsert_listings", return_value=(1, 1)),
        patch("graph_pipeline.reload_statistical_data"),
        patch("graph_pipeline.get_new_listings", return_value=sample_listings),
        patch("deal_detector.detect_deals", return_value=sample_listings),
        patch("agent.evaluate_deals", side_effect=flaky_eval),
        patch("graph_pipeline.get_stats", return_value={"total_listings": 1, "deals_found": 1, "alerts_sent": 0}),
    ):
        results = graph_pipeline.run_graph_pipeline()

    assert scrape_calls["count"] == 2
    assert eval_calls["count"] == 2
    assert results["listings_scraped"] == 1
    assert results["deals_qualified"] == 1
    assert results["errors"] == []
