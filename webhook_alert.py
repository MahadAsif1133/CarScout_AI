"""
webhook_alert.py — CarScout_AI / GaadiScout n8n Webhook Alert Delivery
When N8N_WEBHOOK_URL is configured, sends a structured JSON payload for each
qualifying deal with a short timeout and retry. Supports an optional shared
secret header via N8N_WEBHOOK_SECRET.
"""

import os
import json
import time
import logging
from typing import Callable, Optional
import requests

from db import DB_PATH, mark_alerted, log_alert

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = 5
WEBHOOK_MAX_ATTEMPTS = 2
SECRET_HEADER_NAME = "X-Webhook-Secret"


def build_webhook_payload(listing: dict) -> dict:
    """
    Build the JSON payload using the real database column names, plus a
    convenience `title` field ("{make} {model} {year}").
    """
    make = listing.get("make") or "Unknown"
    model = listing.get("model") or "Unknown"
    year = listing.get("year")
    title = f"{make} {model} {year}".strip() if year else f"{make} {model}".strip()

    risk_flags = listing.get("risk_flags", [])
    if isinstance(risk_flags, str):
        try:
            risk_flags = json.loads(risk_flags)
        except Exception:
            risk_flags = []

    return {
        "title": title,
        "listing_url": listing.get("listing_url", ""),
        "make": make,
        "model": model,
        "year": year,
        "mileage": listing.get("mileage"),
        "city": listing.get("city"),
        "price": listing.get("price"),
        "registration_city": listing.get("registration_city"),
        "transmission": listing.get("transmission"),
        "engine_type": listing.get("engine_type"),
        "description": listing.get("description"),
        "posted_date": listing.get("posted_date"),
        "image_url": listing.get("image_url"),
        "scraped_at": listing.get("scraped_at"),
        "predicted_price": listing.get("predicted_price"),
        "price_gap_pct": listing.get("price_gap_pct"),
        "deal_score": listing.get("deal_score"),
        "agent_summary": listing.get("agent_summary"),
        "risk_flags": risk_flags,
        "prediction_method": listing.get("prediction_method"),
        "prediction_level": listing.get("prediction_level"),
        "confidence_text": listing.get("confidence_text"),
    }


def send_single_webhook(
    listing: dict,
    webhook_url: Optional[str] = None,
    webhook_secret: Optional[str] = None,
    timeout: int = WEBHOOK_TIMEOUT,
    max_attempts: int = WEBHOOK_MAX_ATTEMPTS,
    db_path=DB_PATH,
) -> bool:
    """
    POST one deal payload to N8N_WEBHOOK_URL with short timeout and retry.
    Returns True if delivered, False otherwise.
    """
    url = (webhook_url or os.getenv("N8N_WEBHOOK_URL", "")).strip()
    if not url:
        return False

    secret = (
        webhook_secret
        if webhook_secret is not None
        else os.getenv("N8N_WEBHOOK_SECRET", "").strip()
    )
    headers = {"Content-Type": "application/json"}
    if secret:
        headers[SECRET_HEADER_NAME] = secret

    payload = build_webhook_payload(listing)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            listing_url = listing.get("listing_url", "")
            log_alert(
                channel="n8n_webhook",
                listing_url=listing_url,
                deal_details={
                    "make": payload["make"],
                    "model": payload["model"],
                    "year": payload["year"],
                    "price": payload["price"],
                    "predicted_price": payload["predicted_price"],
                    "price_gap_pct": payload["price_gap_pct"],
                    "deal_score": payload["deal_score"],
                    "city": payload["city"],
                    "risk_flags": payload["risk_flags"],
                },
                db_path=db_path,
            )
            if listing_url:
                mark_alerted(listing_url, db_path)
            logger.info(f"n8n webhook delivered for {listing_url}")
            return True
        except requests.RequestException as e:
            logger.warning(
                f"n8n webhook attempt {attempt}/{max_attempts} failed for "
                f"{listing.get('listing_url')}: {e}"
            )
            if attempt < max_attempts:
                time.sleep(0.2)

    return False


def send_webhook_alerts(
    listings: list[dict],
    db_path=DB_PATH,
    log_callback: Optional[Callable[[str], None]] = None,
) -> tuple[int, list[dict]]:
    """
    Attempt to deliver each deal via n8n webhook.
    Returns (sent_count, failed_listings) so caller can fall back to
    Telegram/email for any deal whose webhook delivery failed.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    webhook_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return 0, list(listings or [])

    _log(f"🌐 Sending {len(listings)} deal alert(s) to n8n webhook …")
    sent = 0
    failed: list[dict] = []

    for listing in listings:
        if send_single_webhook(listing, webhook_url=webhook_url, db_path=db_path):
            sent += 1
            _log(f"  ✅ Webhook alert sent: {listing.get('make')} {listing.get('model')}")
        else:
            failed.append(listing)
            _log(
                f"  ⚠️ Webhook failed for {listing.get('listing_url')} — "
                "falling back to direct Telegram/Email"
            )

    _log(f"Webhook alerts complete: {sent}/{len(listings)} delivered via n8n")
    return sent, failed
