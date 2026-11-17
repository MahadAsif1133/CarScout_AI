"""
main.py — GaadiScout Orchestrator
Ties all pipeline stages together in a single function.
Safely re-runnable: skips already-processed listings.
Callable from the dashboard's "Run Pipeline Now" button.
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from db import DB_PATH, init_db, get_stats

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAIN] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config from env ───────────────────────────────────────────────────────────
SCRAPE_MAX_PAGES  = int(os.getenv("SCRAPE_MAX_PAGES", "5"))
SCRAPE_DETAILS    = os.getenv("SCRAPE_DETAILS", "false").lower() == "true"
DEAL_THRESHOLD    = float(os.getenv("DEAL_THRESHOLD", "15.0"))
AGENT_THRESHOLD   = int(os.getenv("AGENT_THRESHOLD", "7"))
ENABLE_TELEGRAM   = os.getenv("ENABLE_TELEGRAM", "true").lower() == "true"
ENABLE_EMAIL      = os.getenv("ENABLE_EMAIL", "true").lower() == "true"
PIPELINE_ENGINE   = os.getenv("PIPELINE_ENGINE", "legacy").lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(log_callback=None) -> dict:
    """
    Run the full GaadiScout pipeline end-to-end.
    Supports PIPELINE_ENGINE=legacy (default) or PIPELINE_ENGINE=langgraph.

    Stages:
      1. Scrape PakWheels listings
      2. Clean & store to DB (upsert, dedup)
      3. Detect candidate deals (price gap analysis)
      4. AI agent evaluation & scoring
      5. Send Telegram alerts
      6. Send email alerts

    Args:
        log_callback: Optional callable(str) for streaming progress to dashboard.

    Returns:
        Dict with summary stats for the run.
    """
    engine = os.getenv("PIPELINE_ENGINE", PIPELINE_ENGINE).lower().strip()
    if engine == "langgraph":
        try:
            from graph_pipeline import run_graph_pipeline
            return run_graph_pipeline(log_callback=log_callback)
        except Exception as e:
            logger.warning(f"LangGraph engine failed ({e}); falling back to legacy pipeline")

    run_start = datetime.utcnow()
    results   = {
        "started_at":      run_start.isoformat(),
        "listings_scraped": 0,
        "listings_stored":  0,
        "deals_detected":   0,
        "deals_qualified":  0,
        "telegram_sent":    0,
        "email_sent":       0,
        "errors":           [],
        "finished_at":      None,
    }

    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    max_pages = int(os.getenv("SCRAPE_MAX_PAGES", str(SCRAPE_MAX_PAGES)))
    scrape_details = os.getenv("SCRAPE_DETAILS", str(SCRAPE_DETAILS)).lower() == "true"
    deal_threshold = float(os.getenv("DEAL_THRESHOLD", str(DEAL_THRESHOLD)))
    agent_threshold = int(os.getenv("AGENT_THRESHOLD", str(AGENT_THRESHOLD)))
    enable_telegram = os.getenv("ENABLE_TELEGRAM", str(ENABLE_TELEGRAM)).lower() == "true"
    enable_email = os.getenv("ENABLE_EMAIL", str(ENABLE_EMAIL)).lower() == "true"

    _log("=" * 50)
    _log("🚀 GaadiScout Pipeline — Starting")
    _log(f"   Time:       {run_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    _log(f"   Max pages:  {max_pages}")
    _log(f"   Threshold:  {deal_threshold}% below market")
    _log(f"   Agent min:  {agent_threshold}/10")
    _log("=" * 50)

    # ── Stage 0: Ensure DB exists ─────────────────────────────────────────────
    try:
        init_db()
        _log("✅ Database ready")
    except Exception as e:
        err = f"DB init failed: {e}"
        results["errors"].append(err)
        _log(f"❌ {err}")
        return results

    # ── Stage 1: Scrape ───────────────────────────────────────────────────────
    _log("\n📡 Stage 1: Scraping PakWheels …")
    raw_listings = []
    try:
        from scraper import scrape_listings
        raw_listings = scrape_listings(
            max_pages=max_pages,
            fetch_details=scrape_details,
        )
        results["listings_scraped"] = len(raw_listings)
        _log(f"   Scraped {len(raw_listings)} listings")
        if not raw_listings:
            _log(
                "   ⚠️ Scraping returned 0 listings (PakWheels may be blocking "
                "the runner IP or no listings matched). Proceeding gracefully with existing DB data."
            )
    except Exception as e:
        err = f"Scraper failed: {e}"
        results["errors"].append(err)
        _log(f"❌ {err}")
        return results

    # ── Stage 2: Store to DB ──────────────────────────────────────────────────
    _log("\n💾 Stage 2: Storing listings to database …")
    try:
        from db import bulk_upsert_listings
        inserted, total = bulk_upsert_listings(raw_listings)
        results["listings_stored"] = inserted
        _log(f"   Inserted {inserted} new / {total} total processed")
    except Exception as e:
        err = f"DB upsert failed: {e}"
        results["errors"].append(err)
        _log(f"❌ {err}")
        return results

    # ── Stage 3: Detect deals ─────────────────────────────────────────────────
    _log(f"\n🔎 Stage 3: Detecting deals (threshold: {deal_threshold}%) …")
    candidate_deals = []
    try:
        from deal_detector import detect_deals
        candidate_deals = detect_deals(
            deal_threshold=deal_threshold,
            log_callback=_log,
        )
        results["deals_detected"] = len(candidate_deals)
        _log(f"   Found {len(candidate_deals)} candidate deals")
    except Exception as e:
        err = f"Deal detector failed: {e}"
        results["errors"].append(err)
        _log(f"❌ {err}")

    # ── Stage 4: AI agent evaluation ──────────────────────────────────────────
    _log(f"\n🤖 Stage 4: AI agent evaluation (min score: {agent_threshold}/10) …")
    qualifying_deals = []
    try:
        from agent import evaluate_deals
        qualifying_deals = evaluate_deals(
            agent_threshold=agent_threshold,
            log_callback=_log,
        )
        results["deals_qualified"] = len(qualifying_deals)
        _log(f"   {len(qualifying_deals)} deals passed agent filter")
    except Exception as e:
        err = f"Agent evaluation failed: {e}"
        results["errors"].append(err)
        _log(f"❌ {err}")
        # If agent fails, fall back to all candidate deals for alerting
        _log("   ⚠️ Falling back to unscored deals for alerts")
        qualifying_deals = candidate_deals

    # ── Optional n8n Webhook alerts (with fallback to Telegram/Email) ─────────
    webhook_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    remaining_deals = qualifying_deals
    if webhook_url and qualifying_deals and (enable_telegram or enable_email):
        _log("\n🌐 Attempting n8n webhook alert delivery …")
        try:
            from webhook_alert import send_webhook_alerts
            wb_sent, remaining_deals = send_webhook_alerts(
                qualifying_deals,
                log_callback=_log,
            )
            results["webhook_sent"] = wb_sent
        except Exception as e:
            logger.warning(f"Webhook delivery failed, falling back to direct alerts: {e}")
            remaining_deals = qualifying_deals

    # ── Stage 5: Telegram alerts (non-fatal) ──────────────────────────────────
    if enable_telegram:
        if webhook_url and qualifying_deals and not remaining_deals:
            _log("\n📱 Stage 5: Telegram alerts skipped (delivered via n8n webhook)")
        else:
            _log("\n📱 Stage 5: Sending Telegram alerts …")
            try:
                from telegram_bot import send_alerts as tg_send
                tg_count = tg_send(
                    listings=remaining_deals if remaining_deals else None,
                    log_callback=_log,
                )
                results["telegram_sent"] = tg_count
                _log(f"   Sent {tg_count} Telegram alerts")
            except Exception as e:
                warn = f"Telegram alerts failed (non-fatal): {e}"
                logger.warning(warn)
                _log(f"⚠️ {warn}")
    else:
        _log("\n📱 Stage 5: Telegram alerts disabled (ENABLE_TELEGRAM=false)")

    # ── Stage 6: Email alerts (non-fatal) ─────────────────────────────────────
    if enable_email:
        if webhook_url and qualifying_deals and not remaining_deals:
            _log("\n📧 Stage 6: Email alerts skipped (delivered via n8n webhook)")
        else:
            _log("\n📧 Stage 6: Sending email alerts …")
            try:
                from email_alert import send_alerts as em_send
                em_count = em_send(
                    listings=remaining_deals if remaining_deals else None,
                    log_callback=_log,
                )
                results["email_sent"] = em_count
                _log(f"   Sent {em_count} email alerts")
            except Exception as e:
                warn = f"Email alerts failed (non-fatal): {e}"
                logger.warning(warn)
                _log(f"⚠️ {warn}")
    else:
        _log("\n📧 Stage 6: Email alerts disabled (ENABLE_EMAIL=false)")

    # ── Summary ───────────────────────────────────────────────────────────────
    results["finished_at"] = datetime.utcnow().isoformat()
    elapsed = (datetime.utcnow() - run_start).total_seconds()

    _log("\n" + "=" * 50)
    _log("✅ Pipeline Complete")
    _log(f"   Duration:        {elapsed:.1f}s")
    _log(f"   Scraped:         {results['listings_scraped']}")
    _log(f"   Stored:          {results['listings_stored']} new")
    _log(f"   Deals detected:  {results['deals_detected']}")
    _log(f"   Deals qualified: {results['deals_qualified']}")
    _log(f"   Telegram sent:   {results['telegram_sent']}")
    _log(f"   Email sent:      {results['email_sent']}")
    if results["errors"]:
        _log(f"   Errors:          {len(results['errors'])}")
        for err in results["errors"]:
            _log(f"     • {err}")
    _log("=" * 50)

    # Print updated DB stats
    try:
        stats = get_stats()
        _log(f"\n📊 DB Stats: {stats['total_listings']} total listings | "
             f"{stats['deals_found']} deals | {stats['alerts_sent']} alerts sent")
    except Exception:
        pass

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GaadiScout — Run the full deal-finding pipeline"
    )
    parser.add_argument(
        "--pages", type=int, default=SCRAPE_MAX_PAGES,
        help=f"Max pages to scrape (default: {SCRAPE_MAX_PAGES})"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEAL_THRESHOLD,
        help=f"Deal threshold %% (default: {DEAL_THRESHOLD})"
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="Disable Telegram alerts for this run"
    )
    parser.add_argument(
        "--no-email", action="store_true",
        help="Disable email alerts for this run"
    )
    args = parser.parse_args()

    os.environ["SCRAPE_MAX_PAGES"] = str(args.pages)
    os.environ["DEAL_THRESHOLD"] = str(args.threshold)
    if args.no_telegram:
        os.environ["ENABLE_TELEGRAM"] = "false"
    if args.no_email:
        os.environ["ENABLE_EMAIL"] = "false"

    results = run_pipeline()
    if results.get("errors"):
        sys.exit(1)
    sys.exit(0)
