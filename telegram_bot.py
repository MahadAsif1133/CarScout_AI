"""
telegram_bot.py — GaadiScout Telegram Alerts
Sends formatted deal alerts via the Telegram Bot API using plain requests.post().
"""

import os
import json
import logging
import requests
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from db import DB_PATH, get_pending_alerts, mark_alerted, log_alert

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TELEGRAM] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_BASE  = "https://api.telegram.org"
REQUEST_TIMEOUT    = 15


# ── Message formatter ─────────────────────────────────────────────────────────

def _format_message(listing: dict) -> str:
    """Build a Markdown-formatted Telegram message for a deal."""
    make         = listing.get("make", "Unknown")
    model        = listing.get("model", "Unknown")
    year         = listing.get("year", "N/A")
    price        = listing.get("price", 0)
    pred         = listing.get("predicted_price", 0)
    gap          = listing.get("price_gap_pct", 0)
    score        = listing.get("deal_score", "N/A")
    city         = listing.get("city", "Unknown")
    mileage      = listing.get("mileage")
    transmission = listing.get("transmission", "N/A")
    url          = listing.get("listing_url", "")
    summary      = listing.get("agent_summary", "")
    risk_flags   = listing.get("risk_flags", [])
    conf_text    = listing.get("confidence_text") or listing.get("confidence_text", "")
    if isinstance(risk_flags, str):
        try:
            risk_flags = json.loads(risk_flags)
        except Exception:
            risk_flags = []

    mileage_str = f"{mileage:,} km" if mileage else "N/A"
    price_str   = f"PKR {price:,}"
    pred_str    = f"PKR {pred:,}"

    score_bar = "🔥" * min(int(score) // 2, 5) if isinstance(score, int) else "—"

    flag_emojis = {
        "flood_mention":     "🌊 Flood mention",
        "no_documents":      "📋 No documents",
        "salvage_title":     "🔧 Salvage/write-off",
        "accident_history":  "💥 Accident history",
        "odometer_rollback": "⏮️ Odometer rollback",
        "engine_issues":     "⚙️ Engine issues",
        "theft_recovered":   "🚔 Theft recovered",
    }
    flags_text = ""
    if risk_flags:
        flag_lines = [flag_emojis.get(f, f"⚠️ {f}") for f in risk_flags]
        flags_text = "\n⚠️ *Risk Flags:*\n" + "\n".join(f"  • {fl}" for fl in flag_lines)

    # Confidence tag
    conf_line = f"\n📊 _{conf_text}_" if conf_text else ""

    message = (
        f"🚗 *GaadiScout Deal Alert!*\n\n"
        f"*{make} {model} {year}*\n"
        f"{'─' * 30}\n"
        f"💰 *Listed:* {price_str}\n"
        f"📊 *Fair Value:* {pred_str}\n"
        f"📉 *Below Market:* {gap:.1f}%\n"
        f"📍 *City:* {city}  |  🛣️ *Mileage:* {mileage_str}\n"
        f"⚙️ *Transmission:* {transmission}\n"
        f"🏆 *Deal Score:* {score}/10  {score_bar}\n"
        f"{flags_text}"
        f"{conf_line}\n"
        f"💬 *Analysis:* _{summary}_\n\n"
        f"🔗 [View on PakWheels]({url})"
    )
    return message



# ── Sender ────────────────────────────────────────────────────────────────────

def send_message(text: str, bot_token: str = None, chat_id: str = None) -> bool:
    """
    Send a raw Markdown message to a Telegram chat.
    Returns True on success, False on failure.
    """
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
    cid   = chat_id or os.getenv("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)

    if not token or not cid:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not configured — skipping Telegram alert")
        return False

    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    payload = {
        "chat_id": cid,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 400 and payload.get("parse_mode") == "Markdown":
            logger.warning("Telegram Markdown parse failed (400), retrying as plain text")
            payload.pop("parse_mode", None)
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Telegram API error: {data.get('description')}")
            return False
    except requests.RequestException as e:
        logger.error(f"Telegram request failed: {e}")
        return False


def send_deal_alert(listing: dict, db_path=DB_PATH) -> bool:
    """
    Send a formatted deal alert for a single listing.
    Logs to the alerts table and marks listing as alerted on success.
    Returns True on success.
    """
    url = listing.get("listing_url", "")
    message = _format_message(listing)

    success = send_message(message)

    if success:
        log_alert(
            channel="telegram",
            listing_url=url,
            deal_details={
                "make":            listing.get("make"),
                "model":           listing.get("model"),
                "year":            listing.get("year"),
                "price":           listing.get("price"),
                "predicted_price": listing.get("predicted_price"),
                "price_gap_pct":   listing.get("price_gap_pct"),
                "deal_score":      listing.get("deal_score"),
                "city":            listing.get("city"),
                "risk_flags":      listing.get("risk_flags", []),
            },
            db_path=db_path,
        )
        logger.info(f"Alert logged to DB for: {url}")
    return success


def send_alerts(
    listings: list[dict] = None,
    db_path=DB_PATH,
    log_callback=None,
) -> int:
    """
    Send Telegram alerts for all pending qualifying deals.
    If listings is None, fetches pending alerts from DB.
    Returns count of successfully sent alerts.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    if listings is None:
        listings = get_pending_alerts(db_path)

    _log(f"Sending Telegram alerts for {len(listings)} deals …")
    sent = 0

    for listing in listings:
        if send_deal_alert(listing, db_path):
            mark_alerted(listing["listing_url"], db_path)
            sent += 1
            _log(f"  ✅ Telegram alert sent: {listing.get('make')} {listing.get('model')}")
        else:
            _log(f"  ❌ Failed to send Telegram alert: {listing.get('listing_url')}")

    _log(f"Telegram alerts complete: {sent}/{len(listings)} sent")
    return sent


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("GaadiScout Telegram Bot — test message")
    print("=" * 60)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file")
        sys.exit(1)

    test_listing = {
        "make": "Toyota", "model": "Corolla", "year": 2019,
        "price": 2_500_000, "predicted_price": 3_100_000,
        "price_gap_pct": 19.4, "deal_score": 8,
        "city": "Lahore", "mileage": 62000, "transmission": "Automatic",
        "listing_url": "https://www.pakwheels.com/used-cars/toyota-corolla-test/123456",
        "agent_summary": "Well-priced Corolla with clean history. Good deal for first-time buyers.",
        "risk_flags": [],
    }

    success = send_message(_format_message(test_listing))
    if success:
        print("✅ Test message sent! Check your Telegram.")
    else:
        print("❌ Failed to send. Check your bot token and chat ID.")
