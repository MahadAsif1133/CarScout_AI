"""
dashboard.py — GaadiScout Live Dashboard
Polished 4-tab Gradio dashboard with real-time pipeline trigger.
Deployable to Hugging Face Spaces.
"""

import os
import json
import threading
import logging
from datetime import datetime

import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from db import (
    init_db,
    get_all_deals,
    get_all_listings_with_predictions,
    get_alert_history,
    get_stats,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DASHBOARD] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Init DB on startup ────────────────────────────────────────────────────────
init_db()

# ── Color palette ─────────────────────────────────────────────────────────────
PURPLE   = "#6366f1"
INDIGO   = "#4f46e5"
GREEN    = "#10b981"
RED      = "#ef4444"
ORANGE   = "#f59e0b"
GRAY     = "#6b7280"
BG_DARK  = "#0f172a"
BG_CARD  = "#1e293b"
BG_LIGHT = "#f8fafc"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Live Deals
# ─────────────────────────────────────────────────────────────────────────────

def load_deals_df() -> pd.DataFrame:
    """Fetch all deals from DB, return as formatted DataFrame."""
    deals = get_all_deals()
    if not deals:
        return pd.DataFrame(columns=[
            "Make", "Model", "Year", "Listed Price (PKR)",
            "Predicted Price (PKR)", "% Below Market", "Deal Score",
            "City", "Mileage (km)", "Transmission", "Confidence", "Risk Flags", "URL",
        ])

    rows = []
    for d in deals:
        risk = d.get("risk_flags", "[]")
        if isinstance(risk, str):
            try:
                risk = json.loads(risk)
            except Exception:
                risk = []
                
        score_val = d.get("deal_score")
        score_str = f"{int(score_val)}/10" if score_val is not None else "N/A"
        
        rows.append({
            "Make":                 d.get("make") or "N/A",
            "Model":                d.get("model") or "N/A",
            "Year":                 str(d.get("year")) if d.get("year") else "N/A",
            "Listed Price (PKR)":   f"{d['price']:,}" if d.get("price") else "N/A",
            "Predicted Price (PKR)":f"{d['predicted_price']:,}" if d.get("predicted_price") else "N/A",
            "% Below Market":       f"{d['price_gap_pct']:.1f}%" if d.get("price_gap_pct") is not None else "N/A",
            "Deal Score":           score_str,
            "City":                 d.get("city") or "N/A",
            "Mileage (km)":         f"{d['mileage']:,}" if d.get("mileage") else "N/A",
            "Transmission":         d.get("transmission") or "N/A",
            "Confidence":           d.get("confidence_text") or "N/A",
            "Risk Flags":           ", ".join(risk) if risk else "None",
            "URL":                  d.get("listing_url") or "",
        })
    return pd.DataFrame(rows)


def _parse_score(val) -> int:
    try:
        s = str(val).split("/")[0].strip()
        return int(float(s))
    except Exception:
        return 0

def _parse_price(val) -> int:
    import re
    try:
        s = str(val).replace(",", "").replace("PKR", "").strip()
        if not s or s.upper() == "N/A":
            return 0
        digits = re.sub(r"[^\d]", "", s)
        return int(digits) if digits else 0
    except Exception:
        return 0

def filter_deals(make_filter: str, city_filter: str, min_score: int, max_price: int) -> pd.DataFrame:
    """Apply filters to the deals DataFrame."""
    df = load_deals_df()
    if df.empty:
        return df
    if make_filter and make_filter != "All":
        df = df[df["Make"].str.lower() == make_filter.lower()]
    if city_filter and city_filter != "All":
        df = df[df["City"].str.lower() == city_filter.lower()]
    if min_score > 0:
        df = df[df["Deal Score"].apply(_parse_score) >= min_score]
    if max_price > 0:
        df = df[df["Listed Price (PKR)"].apply(_parse_price) <= max_price * 100_000]
    return df


def get_filter_choices():
    """Return unique makes and cities from current deals."""
    df = load_deals_df()
    makes  = ["All"] + sorted(df["Make"].dropna().unique().tolist()) if not df.empty else ["All"]
    cities = ["All"] + sorted(df["City"].dropna().unique().tolist()) if not df.empty else ["All"]
    return makes, cities


def build_deals_html(df: pd.DataFrame) -> str:
    if df.empty:
        return """
        <div style="text-align: center; padding: 40px; color: #94a3b8; background: #1e293b; border-radius: 12px; border: 1px dashed #334155; margin-top: 10px;">
          <p style="font-size: 16px; margin: 0;">No deals found yet — click 'Run Pipeline Now' below to fetch the latest listings</p>
        </div>
        """
    
    rows_html = []
    for idx, row in df.iterrows():
        # Score color-coding
        score_str = str(row.get("Deal Score", "N/A"))
        if score_str == "N/A":
            score_badge = '<span style="background-color: #374151; color: #94a3b8; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">N/A</span>'
        else:
            try:
                score_val = int(score_str.split("/")[0])
            except Exception:
                score_val = 0
                
            if score_val >= 8:
                score_badge = f'<span style="background-color: #064e3b; color: #34d399; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">{score_str}</span>'
            elif score_val >= 5:
                score_badge = f'<span style="background-color: #78350f; color: #fbbf24; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">{score_str}</span>'
            else:
                score_badge = f'<span style="background-color: #374151; color: #94a3b8; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">{score_str}</span>'
            
        # Price and prediction formatting
        price_raw = _parse_price(row.get("Listed Price (PKR)"))
        price_fmt = f"PKR {price_raw:,}" if price_raw else "N/A"
        
        pred_raw = _parse_price(row.get("Predicted Price (PKR)"))
        pred_fmt = f"PKR {pred_raw:,}" if pred_raw else "N/A"
        
        # Risk Flags
        risk_flags = row.get("Risk Flags", "None")
        if risk_flags == "None" or not risk_flags:
            risk_badge = '<span style="color: #10b981;">None</span>'
        else:
            risk_badge = f'<span style="color: #f87171; font-weight: 500;">{risk_flags}</span>'
            
        url = row.get("URL", "")
        link_html = f'<a href="{url}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600; border: 1px solid #0284c7; padding: 4px 10px; border-radius: 6px; background: rgba(2,132,199,0.1); transition: all 0.2s;">View Listing</a>' if url else "N/A"
        
        rows_html.append(f"""
        <tr style="border-bottom: 1px solid #334155; color: #f1f5f9; font-size: 13px; transition: background 0.15s;" onmouseover="this.style.background='#1e293b'" onmouseout="this.style.background='none'">
          <td style="padding: 12px 16px; font-weight: 600; color: #fff;">{row.get("Make", "")}</td>
          <td style="padding: 12px 16px;">{row.get("Model", "")}</td>
          <td style="padding: 12px 16px; color: #94a3b8;">{row.get("Year", "")}</td>
          <td style="padding: 12px 16px; color: #f87171; font-weight: 600;">{price_fmt}</td>
          <td style="padding: 12px 16px; color: #e2e8f0;">{pred_fmt}</td>
          <td style="padding: 12px 16px; color: #34d399; font-weight: 600;">{row.get("% Below Market", "")}</td>
          <td style="padding: 12px 16px;">{score_badge}</td>
          <td style="padding: 12px 16px; color: #94a3b8;">{row.get("City", "")}</td>
          <td style="padding: 12px 16px; color: #94a3b8;">{row.get("Mileage (km)", "")}</td>
          <td style="padding: 12px 16px; color: #94a3b8;">{row.get("Transmission", "")}</td>
          <td style="padding: 12px 16px; color: #38bdf8; font-size: 11px;">{row.get("Confidence", "")}</td>
          <td style="padding: 12px 16px;">{risk_badge}</td>
          <td style="padding: 12px 16px; text-align: center;">{link_html}</td>
        </tr>
        """)
        
    return f"""
    <div style="overflow-x: auto; border-radius: 12px; border: 1px solid #334155; margin-top: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
      <table style="width: 100%; border-collapse: collapse; text-align: left; background: #0f172a;">
        <thead>
          <tr style="background: #1e293b; border-bottom: 2px solid #334155; color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
            <th style="padding: 12px 16px;">Make</th>
            <th style="padding: 12px 16px;">Model</th>
            <th style="padding: 12px 16px;">Year</th>
            <th style="padding: 12px 16px;">Price</th>
            <th style="padding: 12px 16px;">Fair Value</th>
            <th style="padding: 12px 16px;">% Below</th>
            <th style="padding: 12px 16px;">Score</th>
            <th style="padding: 12px 16px;">City</th>
            <th style="padding: 12px 16px;">Mileage</th>
            <th style="padding: 12px 16px;">Transmission</th>
            <th style="padding: 12px 16px;">Confidence</th>
            <th style="padding: 12px 16px;">Risk Flags</th>
            <th style="padding: 12px 16px; text-align: center;">Link</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    """


def build_alert_history_html(df: pd.DataFrame) -> str:
    if df.empty:
        return """
        <div style="text-align: center; padding: 40px; color: #94a3b8; background: #1e293b; border-radius: 12px; border: 1px dashed #334155; margin-top: 10px;">
          <p style="font-size: 16px; margin: 0;">No alerts sent yet</p>
        </div>
        """
    
    rows_html = []
    for idx, row in df.iterrows():
        url = row.get("URL", "")
        link_html = f'<a href="{url}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600; border: 1px solid #0284c7; padding: 4px 10px; border-radius: 6px; background: rgba(2,132,199,0.1);">View Listing</a>' if url else "N/A"
        
        rows_html.append(f"""
        <tr style="border-bottom: 1px solid #334155; color: #f1f5f9; font-size: 13px; transition: background 0.15s;" onmouseover="this.style.background='#1e293b'" onmouseout="this.style.background='none'">
          <td style="padding: 12px 16px;"><span style="background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid #6366f1; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{row.get("Channel", "")}</span></td>
          <td style="padding: 12px 16px; color: #94a3b8;">{row.get("Sent At", "")}</td>
          <td style="padding: 12px 16px; font-weight: 600; color: #fff;">{row.get("Car", "")}</td>
          <td style="padding: 12px 16px; color: #f87171; font-weight: 600;">{row.get("Price", "")}</td>
          <td style="padding: 12px 16px; font-weight: 600; color: #fbbf24;">{row.get("Score", "")}</td>
          <td style="padding: 12px 16px; color: #34d399; font-weight: 600;">{row.get("Gap %", "")}</td>
          <td style="padding: 12px 16px; text-align: center;">{link_html}</td>
        </tr>
        """)
        
    return f"""
    <div style="overflow-x: auto; border-radius: 12px; border: 1px solid #334155; margin-top: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
      <table style="width: 100%; border-collapse: collapse; text-align: left; background: #0f172a;">
        <thead>
          <tr style="background: #1e293b; border-bottom: 2px solid #334155; color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">
            <th style="padding: 12px 16px;">Channel</th>
            <th style="padding: 12px 16px;">Sent At</th>
            <th style="padding: 12px 16px;">Car</th>
            <th style="padding: 12px 16px;">Price</th>
            <th style="padding: 12px 16px;">Score</th>
            <th style="padding: 12px 16px;">Gap %</th>
            <th style="padding: 12px 16px; text-align: center;">Link</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    """


def build_scatter_chart() -> plt.Figure:
    """Listed price vs predicted price scatter, colour-coded by deal status."""
    data = get_all_listings_with_predictions()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_CARD)

    if not data:
        ax.text(0.5, 0.5, "No data yet\nRun the pipeline to collect listings",
                ha="center", va="center", color="white", fontsize=12,
                transform=ax.transAxes)
        ax.set_title("Listed vs Predicted Price", color="white", fontsize=14, pad=15)
        return fig

    prices   = [d["price"] / 100_000 for d in data if d.get("price") and d.get("predicted_price")]
    preds    = [d["predicted_price"] / 100_000 for d in data if d.get("price") and d.get("predicted_price")]
    is_deals = [bool(d.get("is_deal")) for d in data if d.get("price") and d.get("predicted_price")]

    if not prices:
        ax.text(0.5, 0.5, "No listings with predictions in database",
                ha="center", va="center", color="white", fontsize=12,
                transform=ax.transAxes)
        return fig

    colors = [GREEN if deal else GRAY for deal in is_deals]
    sizes  = [80 if deal else 30 for deal in is_deals]

    ax.scatter(preds, prices, c=colors, s=sizes, alpha=0.7, edgecolors="none")

    # Perfect price line
    all_vals = preds + prices
    lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1
    ax.plot([lo, hi], [lo, hi], color=PURPLE, lw=1.5, linestyle="--", alpha=0.7, label="Fair price line")

    # -15% threshold line
    ax.plot([lo, hi], [lo * 0.85, hi * 0.85], color=ORANGE, lw=1, linestyle=":", alpha=0.5, label="–15% threshold")

    # Labels
    ax.set_xlabel("Predicted Fair Price (Lacs PKR)", color="white", fontsize=11)
    ax.set_ylabel("Listed Price (Lacs PKR)", color="white", fontsize=11)
    ax.set_title("Listed vs Predicted Fair Price", color="white", fontsize=14, pad=15, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    legend_handles = [
        mpatches.Patch(color=GREEN, label=f"Deal ({sum(is_deals)})"),
        mpatches.Patch(color=GRAY,  label=f"Not a deal ({sum(not d for d in is_deals)})"),
    ]
    ax.legend(handles=legend_handles, facecolor=BG_CARD, edgecolor="#374151",
              labelcolor="white", fontsize=10)
    fig.tight_layout()
    return fig


def build_bar_chart() -> plt.Figure:
    """Bar chart of deal count by make and discount distribution."""
    deals = get_all_deals()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG_DARK)

    # Left: deals by make
    ax1 = axes[0]
    ax1.set_facecolor(BG_CARD)
    if deals:
        from collections import Counter
        make_counts = Counter(d.get("make", "Unknown") for d in deals)
        makes  = list(make_counts.keys())[:10]
        counts = [make_counts[m] for m in makes]
        bar_colors = [PURPLE] * len(makes)
        bars = ax1.barh(makes, counts, color=bar_colors, alpha=0.85)
        ax1.bar_label(bars, padding=4, color="white", fontsize=10)
        ax1.set_xlabel("Number of Deals", color="white")
        ax1.set_title("Deals by Make", color="white", fontsize=12, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "No deals yet", ha="center", va="center",
                 color="white", fontsize=11, transform=ax1.transAxes)
        ax1.set_title("Deals by Make", color="white", fontsize=12)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#374151")

    # Right: price gap distribution
    ax2 = axes[1]
    ax2.set_facecolor(BG_CARD)
    if deals:
        gaps = [d.get("price_gap_pct", 0) for d in deals if d.get("price_gap_pct") is not None]
        if gaps:
            ax2.hist(gaps, bins=15, color=GREEN, alpha=0.8, edgecolor="#374151")
            ax2.axvline(x=15, color=ORANGE, lw=1.5, linestyle="--", label="15% threshold")
            ax2.set_xlabel("Price Gap (%)", color="white")
            ax2.set_ylabel("Count", color="white")
            ax2.set_title("Deal Discount Distribution", color="white", fontsize=12, fontweight="bold")
            ax2.legend(facecolor=BG_CARD, edgecolor="#374151", labelcolor="white")
    else:
        ax2.text(0.5, 0.5, "No data yet", ha="center", va="center",
                 color="white", fontsize=11, transform=ax2.transAxes)
        ax2.set_title("Deal Discount Distribution", color="white", fontsize=12)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#374151")

    fig.tight_layout(pad=3)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Agent Reasoning
# ─────────────────────────────────────────────────────────────────────────────

def build_reasoning_html() -> str:
    """Build HTML cards for each deal showing agent reasoning."""
    deals = get_all_deals()
    if not deals:
        return """<div style="text-align:center;padding:60px;color:#6b7280;">
            <div style="font-size:48px;margin-bottom:16px;">🤖</div>
            <p style="font-size:18px;">No deals evaluated yet.</p>
            <p>Click <strong>▶ Run Pipeline Now</strong> to start the AI agent.</p>
        </div>"""

    flag_colors = {
        "flood_mention":     "#dc2626",
        "no_documents":      "#dc2626",
        "salvage_title":     "#dc2626",
        "accident_history":  "#ea580c",
        "odometer_rollback": "#dc2626",
        "engine_issues":     "#d97706",
        "theft_recovered":   "#dc2626",
    }
    flag_labels = {
        "flood_mention":     "Flood",
        "no_documents":      "No Docs",
        "salvage_title":     "Salvage",
        "accident_history":  "Accident",
        "odometer_rollback": "Odometer",
        "engine_issues":     "Engine Issue",
        "theft_recovered":   "Theft",
    }

    cards = []
    for d in deals:
        score      = d.get("deal_score") or 0
        summary    = d.get("agent_summary") or "No AI summary available."
        url        = d.get("listing_url", "#")
        make       = d.get("make", "")
        model      = d.get("model", "")
        year       = d.get("year", "")
        gap        = d.get("price_gap_pct", 0)
        price      = d.get("price", 0)
        risk_flags = d.get("risk_flags", "[]")
        if isinstance(risk_flags, str):
            try:
                risk_flags = json.loads(risk_flags)
            except Exception:
                risk_flags = []

        conf_text  = d.get("confidence_text") or "N/A"

        score_color = "#10b981" if score >= 7 else "#f59e0b" if score >= 5 else "#ef4444"
        score_bars  = "█" * score + "░" * (10 - score)

        flag_badges = ""
        if risk_flags:
            for f in risk_flags:
                color = flag_colors.get(f, "#6b7280")
                label = flag_labels.get(f, f)
                flag_badges += (
                    f'<span style="display:inline-block;background:{color};color:#fff;'
                    f'padding:2px 10px;border-radius:10px;font-size:11px;'
                    f'font-weight:600;margin:2px;">{label}</span>'
                )
        else:
            flag_badges = '<span style="color:#10b981;font-size:12px;">No risk flags detected</span>'

        cards.append(f"""
        <details style="background:#1e293b;border:1px solid #334155;border-radius:12px;
                        margin-bottom:12px;overflow:hidden;">
          <summary style="cursor:pointer;padding:16px 20px;list-style:none;
                          display:flex;align-items:center;gap:16px;user-select:none;">
            <div style="min-width:48px;height:48px;background:{score_color};
                        border-radius:50%;display:flex;align-items:center;
                        justify-content:center;font-size:18px;font-weight:800;color:#fff;">
              {score}
            </div>
            <div style="flex:1;">
              <div style="color:#f1f5f9;font-weight:700;font-size:15px;">
                {make} {model} {year}
              </div>
              <div style="color:#94a3b8;font-size:13px;margin-top:2px;">
                PKR {price:,} &nbsp;·&nbsp; {gap:.1f}% below market &nbsp;·&nbsp; <span style="color:#38bdf8;">{conf_text}</span>
              </div>
            </div>
            <div style="color:#94a3b8;font-size:11px;font-family:monospace;">
              {score_bars}
            </div>
            <div style="color:#94a3b8;font-size:12px;">▼</div>
          </summary>
          <div style="padding:0 20px 20px;">
            <div style="background:#0f172a;border-radius:8px;padding:14px;margin-bottom:12px;">
              <div style="color:#60a5fa;font-size:12px;font-weight:600;margin-bottom:6px;">
                🤖 AI ANALYSIS
              </div>
              <div style="color:#e2e8f0;font-size:14px;line-height:1.6;">{summary}</div>
            </div>
            <div style="margin-bottom:12px;">
              <div style="color:#94a3b8;font-size:12px;margin-bottom:6px;">RISK FLAGS</div>
              {flag_badges}
            </div>
            <a href="{url}" target="_blank"
               style="display:inline-block;background:linear-gradient(135deg,#4f46e5,#7c3aed);
                      color:#fff;text-decoration:none;padding:8px 20px;border-radius:8px;
                      font-size:13px;font-weight:600;">
              View on PakWheels →
            </a>
          </div>
        </details>""")

    return "\n".join(cards)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Alert History
# ─────────────────────────────────────────────────────────────────────────────

def load_alert_history_df() -> pd.DataFrame:
    """Return alert history as a formatted DataFrame."""
    alerts = get_alert_history(limit=200)
    if not alerts:
        return pd.DataFrame(columns=["Channel", "Sent At", "Car", "Price", "Score", "Gap %", "URL"])

    rows = []
    for a in alerts:
        details = a.get("deal_details", "{}")
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except Exception:
                details = {}
        make  = details.get("make") or "N/A"
        model = details.get("model") or "N/A"
        year  = details.get("year") or "N/A"
        price = details.get("price")
        score = details.get("deal_score")
        gap   = details.get("price_gap_pct")
        channel = a.get("channel", "")
        sent_at = a.get("sent_at", "")

        price_str = f"PKR {price:,}" if price else "N/A"
        score_str = f"{int(score)}/10" if score is not None else "N/A"
        gap_str   = f"{gap:.1f}%" if gap is not None else "N/A"

        channel_icon = "Telegram" if channel == "telegram" else "Email"
        rows.append({
            "Channel":  channel_icon,
            "Sent At":  sent_at[:19].replace("T", " ") if sent_at else "",
            "Car":      f"{make} {model} {year}".strip(),
            "Price":    price_str,
            "Score":    score_str,
            "Gap %":    gap_str,
            "URL":      a.get("listing_url") or "",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Stats header
# ─────────────────────────────────────────────────────────────────────────────

def get_stats_html() -> str:
    """Build a stat cards HTML row for the dashboard header."""
    try:
        s = get_stats()
    except Exception:
        s = {"total_listings": 0, "deals_found": 0, "alerts_sent": 0, "avg_discount_pct": 0}

    def card(value, label, color):
        return f"""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
                    padding:20px 28px;text-align:center;min-width:140px;flex:1;">
          <div style="color:{color};font-size:28px;font-weight:800;line-height:1;">{value}</div>
          <div style="color:#64748b;font-size:12px;margin-top:4px;text-transform:uppercase;
                      letter-spacing:0.5px;">{label}</div>
        </div>"""

    return f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:4px;">
      {card(s['total_listings'], "Total Listings", "#60a5fa")}
      {card(s['deals_found'], "Deals Found", "#10b981")}
      {card(s['alerts_sent'], "Alerts Sent", "#f59e0b")}
      {card(f"{s['avg_discount_pct']}%", "Avg Discount", "#a78bfa")}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner (background thread)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline_running = False
_pipeline_log: list[str] = []
_pipeline_lock = threading.Lock()


def _append_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _pipeline_lock:
        _pipeline_log.append(line)
    logger.info(msg)


def run_pipeline_background():
    """Run the full pipeline in a background thread."""
    global _pipeline_running
    with _pipeline_lock:
        _pipeline_log.clear()
    try:
        import main as pipeline_main
        pipeline_main.run_pipeline(log_callback=_append_log)
    except ImportError:
        _append_log("⚠️  main.py not found — running individual modules …")
        _run_pipeline_fallback()
    except Exception as e:
        _append_log(f"❌ Pipeline error: {e}")
    finally:
        _pipeline_running = False
        _append_log("─" * 40)
        _append_log("✅ Pipeline finished")


def _run_pipeline_fallback():
    """Fallback if main.py is unavailable — run modules directly."""
    try:
        _append_log("🔍 Step 1: Scraping PakWheels …")
        from scraper import scrape_listings
        listings = scrape_listings(max_pages=3)
        _append_log(f"   Scraped {len(listings)} listings")

        _append_log("💾 Step 2: Storing to database …")
        from db import bulk_upsert_listings
        n, total = bulk_upsert_listings(listings)
        _append_log(f"   Inserted {n}/{total} new listings")

        _append_log("🔎 Step 3: Detecting deals …")
        from deal_detector import detect_deals
        deals = detect_deals(log_callback=_append_log)
        _append_log(f"   Found {len(deals)} candidate deals")

        _append_log("🤖 Step 4: AI agent evaluation …")
        from agent import evaluate_deals
        qualifying = evaluate_deals(log_callback=_append_log)
        _append_log(f"   {len(qualifying)} deals passed agent filter")

        _append_log("📱 Step 5: Sending Telegram alerts …")
        from telegram_bot import send_alerts as tg_alerts
        tg_sent = tg_alerts(log_callback=_append_log)
        _append_log(f"   Sent {tg_sent} Telegram alerts")

        _append_log("📧 Step 6: Sending email alerts …")
        from email_alert import send_alerts as em_alerts
        em_sent = em_alerts(log_callback=_append_log)
        _append_log(f"   Sent {em_sent} email alerts")

    except Exception as e:
        _append_log(f"❌ Error in pipeline: {e}")


def trigger_pipeline() -> str:
    """Called by the dashboard button. Starts background thread."""
    global _pipeline_running
    if _pipeline_running:
        return "⚠️ Pipeline is already running. Please wait …"
    _pipeline_running = True
    _append_log("🚀 Pipeline triggered from dashboard")
    t = threading.Thread(target=run_pipeline_background, daemon=True)
    t.start()
    return "✅ Pipeline started! Logs will appear below …"


def get_pipeline_log() -> str:
    """Return current pipeline log as a string."""
    with _pipeline_lock:
        if not _pipeline_log:
            return "Pipeline not yet started. Click 'Run Pipeline Now' to begin."
        return "\n".join(_pipeline_log[-100:])   # last 100 lines


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ── */
:root {
  --bg-primary:   #0f172a;
  --bg-card:      #1e293b;
  --bg-border:    #334155;
  --text-primary: #f1f5f9;
  --text-muted:   #94a3b8;
  --accent:       #6366f1;
  --accent-2:     #10b981;
  --danger:       #ef4444;
}
body, .gradio-container {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── Tabs ── */
.tab-nav button {
  color: var(--text-muted) !important;
  border-bottom: 2px solid transparent !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 20px !important;
  transition: all 0.2s ease !important;
}
.tab-nav button.selected {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}
.tab-nav { border-bottom: 1px solid var(--bg-border) !important; background: var(--bg-card) !important; }

/* ── Buttons ── */
button.primary { 
  background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
  border: none !important; color: #fff !important; font-weight: 700 !important;
  border-radius: 8px !important; transition: all 0.2s ease !important;
  padding: 10px 24px !important;
}
button.primary:hover { transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(79,70,229,0.4) !important; }
button.secondary {
  background: var(--bg-card) !important; border: 1px solid var(--bg-border) !important;
  color: var(--text-primary) !important; font-weight: 600 !important;
  border-radius: 8px !important; transition: all 0.2s ease !important;
}
button.secondary:hover { border-color: var(--accent) !important; }

/* ── Inputs ── */
select, input[type=range], .gr-dropdown { 
  background: var(--bg-card) !important; 
  color: var(--text-primary) !important; 
  border: 1px solid var(--bg-border) !important; 
  border-radius: 8px !important;
}
label { color: var(--text-muted) !important; font-size: 12px !important; font-weight: 600 !important; }

/* ── Table ── */
table { background: var(--bg-card) !important; border-radius: 12px !important; overflow: hidden !important; }
th { background: #0f172a !important; color: var(--text-muted) !important; font-size: 11px !important;
     text-transform: uppercase !important; letter-spacing: 0.5px !important; padding: 12px 16px !important; }
td { color: var(--text-primary) !important; padding: 12px 16px !important; border-color: #1e293b !important; }
tr:hover td { background: #263548 !important; }

/* ── Textbox / Log ── */
textarea { 
  background: #0a0f1e !important; color: #22d3ee !important; 
  font-family: 'JetBrains Mono', 'Courier New', monospace !important; 
  font-size: 12px !important; border: 1px solid var(--bg-border) !important; 
  border-radius: 8px !important;
}

/* ── Header ── */
.gaadiscout-header {
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
  border: 1px solid #334155; border-radius: 16px;
  padding: 32px; margin-bottom: 16px; text-align: center;
}

/* ── Plots ── */
.plot-container { background: var(--bg-card) !important; border-radius: 12px !important; 
                  border: 1px solid var(--bg-border) !important; }

/* ── Accordion ── */
details > summary { color: var(--text-primary) !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard layout
# ─────────────────────────────────────────────────────────────────────────────

def get_deals_summary_html(deals_df: pd.DataFrame) -> str:
    if deals_df.empty:
        return """
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
          <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px 20px;flex:1;text-align:center;">
            <div style="color:#60a5fa;font-size:24px;font-weight:800;">0</div>
            <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Deals Found</div>
          </div>
          <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px 20px;flex:1;text-align:center;">
            <div style="color:#10b981;font-size:24px;font-weight:800;">N/A</div>
            <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Avg Score</div>
          </div>
          <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px 20px;flex:1;text-align:center;">
            <div style="color:#a78bfa;font-size:24px;font-weight:800;">N/A</div>
            <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;">Best Deal</div>
          </div>
        </div>
        """
    
    total_deals = len(deals_df)
    scores = deals_df["Deal Score"].apply(_parse_score).tolist()
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    best_score = max(scores) if scores else 0
    
    best_deal_text = "N/A"
    if best_score > 0:
        best_deal_row = deals_df.iloc[scores.index(best_score)]
        best_deal_text = f"{best_deal_row['Make']} {best_deal_row['Model']} ({best_score}/10)"
        
    return f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px;">
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;flex:1;min-width:150px;text-align:center;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);">
        <div style="color:#38bdf8;font-size:26px;font-weight:800;line-height:1;">{total_deals}</div>
        <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-top:6px;font-weight:600;">Deals Found</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;flex:1;min-width:150px;text-align:center;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);">
        <div style="color:#34d399;font-size:26px;font-weight:800;line-height:1;">{avg_score}/10</div>
        <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-top:6px;font-weight:600;">Avg Deal Score</div>
      </div>
      <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px;flex:1.5;min-width:200px;text-align:center;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);">
        <div style="color:#c084fc;font-size:18px;font-weight:800;line-height:1.2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:280px;margin:2px auto 0;">{best_deal_text}</div>
        <div style="color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-top:6px;font-weight:600;">Best Deal Evaluated</div>
      </div>
    </div>
    """


def run_pipeline_interactive():
    yield "Initializing GaadiScout Pipeline..."
    try:
        from scraper import scrape_listings
        yield "Stage 1/5: Scraping PakWheels (max 3 pages)..."
        listings = scrape_listings(max_pages=3)
        
        from db import bulk_upsert_listings
        yield f"Stage 2/5: Upserting {len(listings)} listings to SQLite DB..."
        n, total = bulk_upsert_listings(listings)
        
        from deal_detector import detect_deals
        yield "Stage 3/5: Re-computing price predictions & detecting deals..."
        deals = detect_deals()
        
        from agent import evaluate_deals
        yield f"Stage 4/5: Scoring {len(deals)} deals using Groq LLM agent..."
        evaluate_deals()
        
        from telegram_bot import send_alerts as tg_alerts
        yield "Stage 5/5: Sending deal alerts to Telegram bot..."
        tg_sent = tg_alerts()
        
        yield f"Pipeline run completed! Scraped: {len(listings)} | Deals: {len(deals)} | Alerts Sent: {tg_sent}"
    except Exception as e:
        yield f"Pipeline run failed: {str(e)}"


# Custom Modern Sky/Indigo Auto theme
DASHBOARD_THEME = gr.themes.Soft(
    primary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
)

# Custom layout
def build_dashboard() -> gr.Blocks:
    with gr.Blocks(title="GaadiScout — AI Car Deal Finder") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="gaadiscout-header" style="background: linear-gradient(135deg, #0f172a 0%, #0369a1 100%); padding: 24px; border-radius: 12px; text-align: center; border: 1px solid #334155; margin-bottom: 20px;">
          <h1 style="margin: 0; font-size: 38px; font-weight: 900; color: #fff; letter-spacing: -0.5px;">GaadiScout</h1>
          <p style="margin: 6px 0 0; color: #bae6fd; font-size: 15px; font-weight: 500;">
            AI finds the deal before you do
          </p>
        </div>""")

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Live Deals ─────────────────────────────────────────────
            with gr.TabItem("Live Deals"):
                gr.HTML("""<p style="color:#94a3b8; font-size:13px; margin: 4px 0 16px;">
                  All currently flagged deals sorted by deal score. Use filters below to search listings.</p>""")

                # Summary Statistics HTML row
                summary_df = load_deals_df()
                deals_summary_html = gr.HTML(get_deals_summary_html(summary_df))

                with gr.Row():
                    make_dd  = gr.Dropdown(["All"], value="All", label="Filter by Make",  scale=2)
                    city_dd  = gr.Dropdown(["All"], value="All", label="Filter by City",  scale=2)
                    score_sl = gr.Slider(0, 10, value=0, step=1, label="Min Deal Score",  scale=2)
                    price_sl = gr.Slider(0, 1000, value=0, step=10,
                                         label="Max Price (Lacs PKR, 0 = no limit)",       scale=2)
                    refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)

                # HTML Table Render
                deals_table_html = gr.HTML(build_deals_html(summary_df))

                def refresh_deals(make, city, score, price):
                    make = make or "All"
                    city = city or "All"
                    score = 0 if score is None else int(score)
                    price = 0 if price is None else int(price)
                    
                    filtered_df = filter_deals(make, city, score, price)
                    makes, cities = get_filter_choices()
                    
                    return (
                        build_deals_html(filtered_df),
                        get_deals_summary_html(filtered_df),
                        gr.Dropdown(choices=makes, value=make),
                        gr.Dropdown(choices=cities, value=city),
                    )

                refresh_btn.click(
                    fn=refresh_deals,
                    inputs=[make_dd, city_dd, score_sl, price_sl],
                    outputs=[deals_table_html, deals_summary_html, make_dd, city_dd],
                )

                # Automatic filter update on dropdown/slider change
                for inp in [make_dd, city_dd, score_sl, price_sl]:
                    inp.change(
                        fn=lambda m, c, s, p: (build_deals_html(filter_deals(m, c, s, p)), get_deals_summary_html(filter_deals(m, c, s, p))),
                        inputs=[make_dd, city_dd, score_sl, price_sl],
                        outputs=[deals_table_html, deals_summary_html],
                    )

            # ── Tab 2: Price Analysis ─────────────────────────────────────────
            with gr.TabItem("Price Analysis"):
                gr.HTML("""<p style="color:#94a3b8; font-size:13px; margin: 4px 0 16px;">
                  Visual analysis of listed prices vs predicted fair market values.
                  Points below the dashed line represent underpriced cars.</p>""")

                with gr.Row():
                    chart_refresh = gr.Button("Refresh Charts", variant="secondary")

                scatter_plot = gr.Plot(value=build_scatter_chart(), label="Listed vs Predicted Price")
                bar_plot     = gr.Plot(value=build_bar_chart(),     label="Deals by Make & Discount Distribution")

                chart_refresh.click(
                    fn=lambda: (build_scatter_chart(), build_bar_chart()),
                    outputs=[scatter_plot, bar_plot],
                )

            # ── Tab 3: Agent Reasoning ────────────────────────────────────────
            with gr.TabItem("Agent Reasoning"):
                gr.HTML("""<p style="color:#94a3b8; font-size:13px; margin: 4px 0 16px;">
                  AI agent analysis for each deal — click a card to expand reasoning and risk flags.</p>""")

                reasoning_refresh = gr.Button("Refresh", variant="secondary")
                reasoning_html    = gr.HTML(build_reasoning_html())

                reasoning_refresh.click(
                    fn=build_reasoning_html,
                    outputs=reasoning_html,
                )

            # ── Tab 4: Alert History ──────────────────────────────────────────
            with gr.TabItem("Alert History"):
                gr.HTML("""<p style="color:#94a3b8; font-size:13px; margin: 4px 0 16px;">
                  Log of all alerts sent via Telegram and email. Newest first.</p>""")

                history_refresh = gr.Button("Refresh", variant="secondary")
                
                alert_df = load_alert_history_df()
                history_table_html = gr.HTML(build_alert_history_html(alert_df))
                
                history_refresh.click(
                    fn=lambda: build_alert_history_html(load_alert_history_df()),
                    outputs=history_table_html,
                )

        # ── Pipeline Control ──────────────────────────────────────────────────
        gr.HTML('<hr style="border-color:#334155; margin: 24px 0;">')
        gr.HTML("""<h3 style="color:#f1f5f9; margin: 0 0 8px; font-size: 18px;">▶ Pipeline Control</h3>
          <p style="color:#94a3b8; font-size: 13px; margin: 0 0 16px;">
          Manually trigger the full pipeline: scrape used-cars → detect deals → evaluate with AI agent → send alerts.
          </p>""")

        with gr.Row():
            run_btn   = gr.Button("Run Pipeline Now", variant="primary", scale=2)
            status_box = gr.Textbox(label="Pipeline Output Status (Streaming)", value="Inactive", interactive=False, lines=3)

        run_btn.click(
            fn=run_pipeline_interactive,
            outputs=status_box,
            show_progress="full",
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=DASHBOARD_THEME,
    )
