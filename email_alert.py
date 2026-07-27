"""
email_alert.py — GaadiScout Email Alerts
Sends HTML-formatted deal alerts via Gmail SMTP (App Password).
"""

import os
import json
import smtplib
import logging
import ssl
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()

from db import DB_PATH, get_pending_alerts, mark_alerted, log_alert

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [EMAIL] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EMAIL_SENDER       = os.getenv("EMAIL_SENDER", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
EMAIL_RECIPIENT    = os.getenv("EMAIL_RECIPIENT", "")
SMTP_HOST          = "smtp.gmail.com"
SMTP_PORT          = 465   # SSL


# ── HTML email template ───────────────────────────────────────────────────────

def _build_html(listing: dict) -> str:
    """Build a polished HTML email body for a deal alert."""
    make        = listing.get("make", "Unknown")
    model       = listing.get("model", "Unknown")
    year        = listing.get("year", "N/A")
    price       = listing.get("price", 0)
    pred        = listing.get("predicted_price", 0)
    gap         = listing.get("price_gap_pct", 0)
    score       = listing.get("deal_score", "N/A")
    city        = listing.get("city", "Unknown")
    mileage     = listing.get("mileage")
    transmission= listing.get("transmission", "N/A")
    url         = listing.get("listing_url", "#")
    summary     = listing.get("agent_summary", "")
    risk_flags  = listing.get("risk_flags", [])
    conf_text   = listing.get("confidence_text") or "N/A"
    if isinstance(risk_flags, str):
        try:
            risk_flags = json.loads(risk_flags)
        except Exception:
            risk_flags = []

    mileage_str = f"{mileage:,} km" if mileage else "N/A"
    price_str   = f"PKR {price:,}"
    pred_str    = f"PKR {pred:,}"
    savings     = pred - price if pred and price else 0
    savings_str = f"PKR {savings:,}"

    # Risk flags HTML
    flag_labels = {
        "flood_mention":     ("🌊", "Flood mention", "#dc2626"),
        "no_documents":      ("📋", "No documents", "#dc2626"),
        "salvage_title":     ("🔧", "Salvage/write-off", "#dc2626"),
        "accident_history":  ("💥", "Accident history", "#ea580c"),
        "odometer_rollback": ("⏮️", "Odometer rollback", "#dc2626"),
        "engine_issues":     ("⚙️", "Engine issues", "#ea580c"),
        "theft_recovered":   ("🚔", "Theft recovered", "#dc2626"),
    }

    flags_html = ""
    if risk_flags:
        flag_items = []
        for f in risk_flags:
            emoji, label, color = flag_labels.get(f, ("⚠️", f, "#ea580c"))
            flag_items.append(
                f'<span style="display:inline-block;background:{color};color:#fff;'
                f'padding:3px 10px;border-radius:12px;font-size:12px;margin:3px;">'
                f'{emoji} {label}</span>'
            )
        flags_html = f"""
        <tr>
          <td style="padding:12px 24px;">
            <p style="margin:0 0 6px;color:#374151;font-weight:600;">⚠️ Risk Flags</p>
            {''.join(flag_items)}
          </td>
        </tr>"""

    # Score color
    score_color = "#16a34a" if isinstance(score, int) and score >= 7 else \
                  "#ea580c" if isinstance(score, int) and score >= 5 else "#dc2626"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GaadiScout Deal Alert</title>
</head>
<body style="margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;padding:32px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#1e1b4b 0%,#312e81 50%,#4f46e5 100%);padding:32px 24px;text-align:center;">
            <h1 style="margin:0;color:#fff;font-size:28px;font-weight:800;letter-spacing:-0.5px;">🚗 GaadiScout</h1>
            <p style="margin:8px 0 0;color:#c7d2fe;font-size:14px;">Deal Alert — PakWheels</p>
          </td>
        </tr>

        <!-- Deal badge -->
        <tr>
          <td style="padding:24px 24px 0;text-align:center;">
            <span style="display:inline-block;background:#dcfce7;color:#15803d;padding:6px 20px;border-radius:20px;font-size:13px;font-weight:700;letter-spacing:0.5px;">
              🔥 {gap:.1f}% BELOW MARKET
            </span>
          </td>
        </tr>

        <!-- Car name -->
        <tr>
          <td style="padding:16px 24px 0;text-align:center;">
            <h2 style="margin:0;color:#111827;font-size:24px;font-weight:700;">{make} {model} {year}</h2>
            <p style="margin:4px 0 0;color:#6b7280;font-size:14px;">📍 {city} &nbsp;|&nbsp; 🛣️ {mileage_str} &nbsp;|&nbsp; ⚙️ {transmission}</p>
          </td>
        </tr>

        <!-- Price table -->
        <tr>
          <td style="padding:24px;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background:#f9fafb;border-radius:12px;overflow:hidden;">
              <tr>
                <td style="padding:16px 20px;border-bottom:1px solid #e5e7eb;">
                  <p style="margin:0;color:#6b7280;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Listed Price</p>
                  <p style="margin:4px 0 0;color:#dc2626;font-size:22px;font-weight:800;">{price_str}</p>
                </td>
                <td style="padding:16px 20px;border-bottom:1px solid #e5e7eb;border-left:1px solid #e5e7eb;">
                  <p style="margin:0;color:#6b7280;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Fair Market Value</p>
                  <p style="margin:4px 0 0;color:#111827;font-size:22px;font-weight:800;">{pred_str}</p>
                </td>
              </tr>
              <tr>
                <td style="padding:12px 20px;border-bottom:1px solid #e5e7eb;text-align:center;" colspan="2">
                  <p style="margin:0;color:#6b7280;font-size:11px;">Prediction Confidence</p>
                  <p style="margin:4px 0 0;color:#0284c7;font-size:13px;font-weight:600;">📊 {conf_text}</p>
                </td>
              </tr>
              <tr>
                <td colspan="2" style="padding:16px 20px;text-align:center;">
                  <p style="margin:0;color:#6b7280;font-size:12px;">Potential Savings</p>
                  <p style="margin:4px 0 0;color:#16a34a;font-size:20px;font-weight:700;">💰 {savings_str}</p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Deal score -->
        <tr>
          <td style="padding:0 24px 16px;text-align:center;">
            <span style="display:inline-block;background:{score_color};color:#fff;padding:8px 24px;border-radius:24px;font-size:18px;font-weight:800;">
              Deal Score: {score}/10
            </span>
          </td>
        </tr>

        {flags_html}

        <!-- AI Summary -->
        <tr>
          <td style="padding:16px 24px;">
            <div style="background:#eff6ff;border-left:4px solid #3b82f6;border-radius:0 8px 8px 0;padding:16px;">
              <p style="margin:0 0 6px;color:#1e40af;font-weight:600;font-size:13px;">🤖 AI Analysis</p>
              <p style="margin:0;color:#374151;font-size:14px;line-height:1.6;">{summary}</p>
            </div>
          </td>
        </tr>

        <!-- CTA Button -->
        <tr>
          <td style="padding:24px;text-align:center;">
            <a href="{url}" style="display:inline-block;background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;text-decoration:none;padding:14px 40px;border-radius:50px;font-size:16px;font-weight:700;letter-spacing:0.3px;">
              View Listing on PakWheels →
            </a>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f9fafb;padding:20px 24px;text-align:center;border-top:1px solid #e5e7eb;">
            <p style="margin:0;color:#9ca3af;font-size:12px;">
              GaadiScout • Autonomous Car Deal Finder<br>
              This alert was generated automatically. Always verify listings before purchase.<br>
              <a href="{url}" style="color:#6366f1;">Unsubscribe</a>
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ── Sender ────────────────────────────────────────────────────────────────────

def send_email(
    subject: str,
    html_body: str,
    sender: str = None,
    app_password: str = None,
    recipient: str = None,
) -> bool:
    """
    Send an HTML email via Gmail SMTP (SSL, port 465).
    Returns True on success, False on failure.
    """
    _sender    = sender or EMAIL_SENDER
    _password  = app_password or EMAIL_APP_PASSWORD
    _recipient = recipient or EMAIL_RECIPIENT

    if not all([_sender, _password, _recipient]):
        logger.error("Email credentials not configured (EMAIL_SENDER / EMAIL_APP_PASSWORD / EMAIL_RECIPIENT)")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"GaadiScout 🚗 <{_sender}>"
    msg["To"]      = _recipient

    # Plain text fallback
    plain = f"GaadiScout Deal Alert\n\nCheck your HTML-capable email client to view this alert."
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as server:
            server.login(_sender, _password)
            server.sendmail(_sender, _recipient, msg.as_string())
        logger.info(f"Email sent to {_recipient}")
        return True
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return False


def send_deal_alert(listing: dict, db_path=DB_PATH) -> bool:
    """
    Send a formatted HTML email for a single listing.
    Logs to DB on success. Returns True on success.
    """
    make  = listing.get("make", "Car")
    model = listing.get("model", "")
    year  = listing.get("year", "")
    gap   = listing.get("price_gap_pct", 0)
    score = listing.get("deal_score", "")

    subject = f"🔥 GaadiScout: {make} {model} {year} — {gap:.1f}% below market (Score: {score}/10)"
    html    = _build_html(listing)
    success = send_email(subject=subject, html_body=html)

    if success:
        log_alert(
            channel="email",
            listing_url=listing.get("listing_url", ""),
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
    return success


def send_alerts(
    listings: list[dict] = None,
    db_path=DB_PATH,
    log_callback=None,
) -> int:
    """
    Send email alerts for all pending qualifying deals.
    If listings is None, fetches pending alerts from DB.
    Returns count of successfully sent alerts.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    if listings is None:
        listings = get_pending_alerts(db_path)

    _log(f"Sending email alerts for {len(listings)} deals …")
    sent = 0

    for listing in listings:
        if send_deal_alert(listing, db_path):
            sent += 1
            _log(f"  ✅ Email sent: {listing.get('make')} {listing.get('model')}")
        else:
            _log(f"  ❌ Email failed: {listing.get('listing_url')}")

    _log(f"Email alerts complete: {sent}/{len(listings)} sent")
    return sent


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("GaadiScout Email Alert — test message")
    print("=" * 60)

    if not all([EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_RECIPIENT]):
        print("❌ Set EMAIL_SENDER, EMAIL_APP_PASSWORD, EMAIL_RECIPIENT in your .env file")
        sys.exit(1)

    test_listing = {
        "make": "Honda", "model": "Civic", "year": 2020,
        "price": 3_800_000, "predicted_price": 4_700_000,
        "price_gap_pct": 19.1, "deal_score": 8,
        "city": "Karachi", "mileage": 45000, "transmission": "Automatic",
        "listing_url": "https://www.pakwheels.com/used-cars/honda-civic-test/654321",
        "agent_summary": "Great deal on a low-mileage Civic. Price is well below market. Clean history based on description.",
        "risk_flags": [],
    }

    success = send_deal_alert(test_listing)
    if success:
        print(f"✅ Test email sent to {EMAIL_RECIPIENT}! Check your inbox.")
    else:
        print("❌ Failed to send. Check credentials and Gmail App Password setup.")
