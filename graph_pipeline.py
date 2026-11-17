"""
graph_pipeline.py — CarScout_AI / GaadiScout LangGraph Orchestrator
Implements the pipeline as a LangGraph StateGraph with nodes:
  scrape -> predict -> detect_deals -> evaluate -> alert
Includes conditional edges for early termination and RetryPolicy on scrape and evaluate.
"""

import os
import time
import logging
from datetime import datetime
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.types import RetryPolicy
except ImportError:
    RetryPolicy = None

from db import DB_PATH, init_db, bulk_upsert_listings, get_new_listings, get_stats
from predict import reload_statistical_data

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    started_at: str
    finished_at: Optional[str]
    max_pages: int
    scrape_details: bool
    deal_threshold: float
    agent_threshold: int
    enable_telegram: bool
    enable_email: bool
    raw_listings: list[dict]
    listings_scraped: int
    listings_stored: int
    deals_detected: int
    deals_qualified: int
    telegram_sent: int
    email_sent: int
    webhook_sent: int
    candidate_deals: list[dict]
    qualifying_deals: list[dict]
    errors: list[str]
    log_callback: Optional[Callable[[str], None]]


def _emit_log(state: PipelineState, msg: str) -> None:
    logger.info(msg)
    cb = state.get("log_callback")
    if cb:
        cb(msg)


def _call_with_retry(fn: Callable[[], Any], max_attempts: int = 3, delay: float = 0.05) -> Any:
    """Retry helper for transient errors inside scrape and evaluate nodes."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(f"Attempt {attempt}/{max_attempts} failed ({exc}); retrying...")
                if delay > 0:
                    time.sleep(delay)
    raise last_exc


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def scrape_node(state: PipelineState) -> dict:
    """Stage 1 & 2: Ensure DB exists, scrape PakWheels listings (with retry), and upsert to SQLite."""
    errors = list(state.get("errors", []))
    max_pages = state.get("max_pages", int(os.getenv("SCRAPE_MAX_PAGES", "5")))
    scrape_details = state.get("scrape_details", os.getenv("SCRAPE_DETAILS", "false").lower() == "true")

    try:
        init_db()
        _emit_log(state, "✅ Database ready")
    except Exception as e:
        err = f"DB init failed: {e}"
        errors.append(err)
        _emit_log(state, f"❌ {err}")
        return {"raw_listings": [], "listings_scraped": 0, "listings_stored": 0, "errors": errors}

    _emit_log(state, "\n📡 Stage 1: Scraping PakWheels …")
    from scraper import scrape_listings

    try:
        raw_listings = _call_with_retry(
            lambda: scrape_listings(max_pages=max_pages, fetch_details=scrape_details),
            max_attempts=3,
        )
    except Exception as e:
        err = f"Scraper failed: {e}"
        errors.append(err)
        _emit_log(state, f"❌ {err}")
        return {"raw_listings": [], "listings_scraped": 0, "listings_stored": 0, "errors": errors}

    listings_scraped = len(raw_listings)
    _emit_log(state, f"   Scraped {listings_scraped} listings")

    if not raw_listings:
        _emit_log(state, "   ⚠️ Scraping returned 0 listings — ending graph early.")
        return {
            "raw_listings": [],
            "listings_scraped": 0,
            "listings_stored": 0,
            "errors": errors,
        }

    # Optional MongoDB raw listing archive (Phase 4 hook — no-op when MONGODB_URI is unset)
    if os.getenv("MONGODB_URI", "").strip():
        try:
            from mongo_store import save_raw_listings
            save_raw_listings(raw_listings)
        except Exception as e:
            logger.warning(f"MongoDB raw store skipped: {e}")

    _emit_log(state, "\n💾 Stage 2: Storing listings to database …")
    try:
        inserted, total = bulk_upsert_listings(raw_listings)
        _emit_log(state, f"   Inserted {inserted} new / {total} total processed")
    except Exception as e:
        err = f"DB upsert failed: {e}"
        errors.append(err)
        _emit_log(state, f"❌ {err}")
        return {
            "raw_listings": raw_listings,
            "listings_scraped": listings_scraped,
            "listings_stored": 0,
            "errors": errors,
        }

    return {
        "raw_listings": raw_listings,
        "listings_scraped": listings_scraped,
        "listings_stored": inserted,
        "errors": errors,
    }


def predict_node(state: PipelineState) -> dict:
    """Prepare fair-price predictor data for new listings."""
    _emit_log(state, "\n📈 Preparing fair-price predictor …")
    try:
        reload_statistical_data()
        new_count = len(get_new_listings(DB_PATH))
        _emit_log(state, f"   Predictor ready ({new_count} new listings queued for valuation)")
    except Exception as e:
        logger.warning(f"Predictor reload warning: {e}")
    return {}


def detect_deals_node(state: PipelineState) -> dict:
    """Stage 3: Run deal_detector.detect_deals to predict prices and flag candidate deals."""
    errors = list(state.get("errors", []))
    deal_threshold = state.get("deal_threshold", float(os.getenv("DEAL_THRESHOLD", "15.0")))
    _emit_log(state, f"\n🔎 Stage 3: Detecting deals (threshold: {deal_threshold}%) …")

    try:
        from deal_detector import detect_deals
        candidate_deals = detect_deals(
            deal_threshold=deal_threshold,
            log_callback=state.get("log_callback"),
        )
        _emit_log(state, f"   Found {len(candidate_deals)} candidate deals")
        return {
            "candidate_deals": candidate_deals,
            "deals_detected": len(candidate_deals),
            "errors": errors,
        }
    except Exception as e:
        err = f"Deal detector failed: {e}"
        errors.append(err)
        _emit_log(state, f"❌ {err}")
        return {
            "candidate_deals": [],
            "deals_detected": 0,
            "errors": errors,
        }


def evaluate_node(state: PipelineState) -> dict:
    """Stage 4: AI agent evaluation (with retry)."""
    errors = list(state.get("errors", []))
    agent_threshold = state.get("agent_threshold", int(os.getenv("AGENT_THRESHOLD", "7")))
    _emit_log(state, f"\n🤖 Stage 4: AI agent evaluation (min score: {agent_threshold}/10) …")

    from agent import evaluate_deals

    try:
        qualifying_deals = _call_with_retry(
            lambda: evaluate_deals(
                agent_threshold=agent_threshold,
                log_callback=state.get("log_callback"),
            ),
            max_attempts=3,
        )
        # Filter strictly by agent_threshold in case any deal below threshold is present
        qualifying_deals = [
            d for d in qualifying_deals
            if (d.get("deal_score") or 0) >= agent_threshold
        ]
        _emit_log(state, f"   {len(qualifying_deals)} deals passed agent filter")
        return {
            "qualifying_deals": qualifying_deals,
            "deals_qualified": len(qualifying_deals),
            "errors": errors,
        }
    except Exception as e:
        err = f"Agent evaluation failed: {e}"
        errors.append(err)
        _emit_log(state, f"❌ {err}")
        return {
            "qualifying_deals": [],
            "deals_qualified": 0,
            "errors": errors,
        }


def alert_node(state: PipelineState) -> dict:
    """Stage 5 & 6: Send alerts for qualifying deals (webhook if configured, else Telegram/Email)."""
    agent_threshold = state.get("agent_threshold", int(os.getenv("AGENT_THRESHOLD", "7")))
    enable_telegram = state.get("enable_telegram", os.getenv("ENABLE_TELEGRAM", "true").lower() == "true")
    enable_email = state.get("enable_email", os.getenv("ENABLE_EMAIL", "true").lower() == "true")

    qualifying_deals = [
        d for d in state.get("qualifying_deals", [])
        if (d.get("deal_score") or 0) >= agent_threshold
    ]

    if not qualifying_deals:
        _emit_log(state, "\n🔕 No deals meet AGENT_THRESHOLD — skipping alerts")
        return {"telegram_sent": 0, "email_sent": 0, "webhook_sent": 0}

    # Optional n8n webhook delivery (Phase 3 hook — falls back to Telegram/Email on failure or unset)
    webhook_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    remaining_deals = qualifying_deals
    webhook_sent = 0

    if webhook_url and (enable_telegram or enable_email):
        try:
            from webhook_alert import send_webhook_alerts
            webhook_sent, remaining_deals = send_webhook_alerts(
                qualifying_deals,
                log_callback=state.get("log_callback"),
            )
        except Exception as e:
            logger.warning(f"Webhook delivery failed, falling back to direct alerts: {e}")
            remaining_deals = qualifying_deals

    telegram_sent = 0
    email_sent = 0

    if not remaining_deals:
        return {
            "telegram_sent": 0,
            "email_sent": 0,
            "webhook_sent": webhook_sent,
        }

    if enable_telegram:
        _emit_log(state, "\n📱 Stage 5: Sending Telegram alerts …")
        try:
            from telegram_bot import send_alerts as tg_send
            telegram_sent = tg_send(
                listings=remaining_deals,
                log_callback=state.get("log_callback"),
            )
            _emit_log(state, f"   Sent {telegram_sent} Telegram alerts")
        except Exception as e:
            logger.warning(f"Telegram alerts failed (non-fatal): {e}")
            _emit_log(state, f"⚠️ Telegram alerts failed (non-fatal): {e}")
    else:
        _emit_log(state, "\n📱 Stage 5: Telegram alerts disabled (ENABLE_TELEGRAM=false)")

    if enable_email:
        _emit_log(state, "\n📧 Stage 6: Sending email alerts …")
        try:
            from email_alert import send_alerts as em_send
            email_sent = em_send(
                listings=remaining_deals,
                log_callback=state.get("log_callback"),
            )
            _emit_log(state, f"   Sent {email_sent} email alerts")
        except Exception as e:
            logger.warning(f"Email alerts failed (non-fatal): {e}")
            _emit_log(state, f"⚠️ Email alerts failed (non-fatal): {e}")
    else:
        _emit_log(state, "\n📧 Stage 6: Email alerts disabled (ENABLE_EMAIL=false)")

    return {
        "telegram_sent": telegram_sent,
        "email_sent": email_sent,
        "webhook_sent": webhook_sent,
    }


# ── Conditional Edge Routers ──────────────────────────────────────────────────

def route_after_scrape(state: PipelineState) -> str:
    """End early if scraping returned nothing or encountered a fatal error."""
    if state.get("errors") or not state.get("raw_listings"):
        return END
    return "predict"


def route_after_detect_deals(state: PipelineState) -> str:
    """End early if no candidate deals pass the deal_threshold."""
    if not state.get("candidate_deals"):
        return END
    return "evaluate"


def route_after_evaluate(state: PipelineState) -> str:
    """Skip alert node if no deals scored at or above AGENT_THRESHOLD."""
    agent_threshold = state.get("agent_threshold", int(os.getenv("AGENT_THRESHOLD", "7")))
    qualifying = [
        d for d in state.get("qualifying_deals", [])
        if (d.get("deal_score") or 0) >= agent_threshold
    ]
    if not qualifying:
        return END
    return "alert"


# ── Graph Builder & Runner ────────────────────────────────────────────────────

def build_pipeline_graph():
    """Compile and return the CarScout_AI LangGraph StateGraph."""
    builder = StateGraph(PipelineState)

    retry_policy = RetryPolicy(max_attempts=3) if RetryPolicy is not None else None

    if retry_policy is not None:
        builder.add_node("scrape", scrape_node, retry_policy=retry_policy)
        builder.add_node("evaluate", evaluate_node, retry_policy=retry_policy)
    else:
        builder.add_node("scrape", scrape_node)
        builder.add_node("evaluate", evaluate_node)

    builder.add_node("predict", predict_node)
    builder.add_node("detect_deals", detect_deals_node)
    builder.add_node("alert", alert_node)

    builder.add_edge(START, "scrape")
    builder.add_conditional_edges(
        "scrape",
        route_after_scrape,
        {"predict": "predict", END: END},
    )
    builder.add_edge("predict", "detect_deals")
    builder.add_conditional_edges(
        "detect_deals",
        route_after_detect_deals,
        {"evaluate": "evaluate", END: END},
    )
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"alert": "alert", END: END},
    )
    builder.add_edge("alert", END)

    return builder.compile()


def run_graph_pipeline(log_callback: Optional[Callable[[str], None]] = None) -> dict:
    """Execute the LangGraph pipeline and return the standard summary dict."""
    run_start = datetime.utcnow()
    max_pages = int(os.getenv("SCRAPE_MAX_PAGES", "5"))
    scrape_details = os.getenv("SCRAPE_DETAILS", "false").lower() == "true"
    deal_threshold = float(os.getenv("DEAL_THRESHOLD", "15.0"))
    agent_threshold = int(os.getenv("AGENT_THRESHOLD", "7"))
    enable_telegram = os.getenv("ENABLE_TELEGRAM", "true").lower() == "true"
    enable_email = os.getenv("ENABLE_EMAIL", "true").lower() == "true"

    initial_state: PipelineState = {
        "started_at": run_start.isoformat(),
        "max_pages": max_pages,
        "scrape_details": scrape_details,
        "deal_threshold": deal_threshold,
        "agent_threshold": agent_threshold,
        "enable_telegram": enable_telegram,
        "enable_email": enable_email,
        "raw_listings": [],
        "listings_scraped": 0,
        "listings_stored": 0,
        "deals_detected": 0,
        "deals_qualified": 0,
        "telegram_sent": 0,
        "email_sent": 0,
        "webhook_sent": 0,
        "candidate_deals": [],
        "qualifying_deals": [],
        "errors": [],
        "log_callback": log_callback,
    }

    _emit_log(initial_state, "=" * 50)
    _emit_log(initial_state, "🚀 CarScout_AI (LangGraph Engine) — Starting")
    _emit_log(initial_state, f"   Time:       {run_start.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    _emit_log(initial_state, f"   Max pages:  {max_pages}")
    _emit_log(initial_state, f"   Threshold:  {deal_threshold}% below market")
    _emit_log(initial_state, f"   Agent min:  {agent_threshold}/10")
    _emit_log(initial_state, "=" * 50)

    graph = build_pipeline_graph()
    final_state = graph.invoke(initial_state)

    finished_at = datetime.utcnow().isoformat()
    elapsed = (datetime.utcnow() - run_start).total_seconds()

    results = {
        "started_at": initial_state["started_at"],
        "listings_scraped": final_state.get("listings_scraped", 0),
        "listings_stored": final_state.get("listings_stored", 0),
        "deals_detected": final_state.get("deals_detected", 0),
        "deals_qualified": final_state.get("deals_qualified", 0),
        "telegram_sent": final_state.get("telegram_sent", 0),
        "email_sent": final_state.get("email_sent", 0),
        "errors": final_state.get("errors", []),
        "finished_at": finished_at,
    }

    _emit_log(initial_state, "\n" + "=" * 50)
    _emit_log(initial_state, "✅ LangGraph Pipeline Complete")
    _emit_log(initial_state, f"   Duration:        {elapsed:.1f}s")
    _emit_log(initial_state, f"   Scraped:         {results['listings_scraped']}")
    _emit_log(initial_state, f"   Stored:          {results['listings_stored']} new")
    _emit_log(initial_state, f"   Deals detected:  {results['deals_detected']}")
    _emit_log(initial_state, f"   Deals qualified: {results['deals_qualified']}")
    _emit_log(initial_state, f"   Telegram sent:   {results['telegram_sent']}")
    _emit_log(initial_state, f"   Email sent:      {results['email_sent']}")
    _emit_log(initial_state, "=" * 50)

    try:
        stats = get_stats()
        _emit_log(
            initial_state,
            f"\n📊 DB Stats: {stats['total_listings']} total listings | "
            f"{stats['deals_found']} deals | {stats['alerts_sent']} alerts sent",
        )
    except Exception:
        pass

    return results
