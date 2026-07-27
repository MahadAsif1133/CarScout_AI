"""
deal_detector.py — GaadiScout Deal Detector
For each new (unprocessed) listing, predicts a fair market price,
calculates the price gap, and flags listings as deals if they are
sufficiently below predicted fair value.
"""

import os
import logging
from pathlib import Path

from db import (
    DB_PATH,
    get_new_listings,
    update_prediction,
    init_db,
)
from predict import predict_price, reload_statistical_data

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DETECTOR] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Percentage below predicted price to flag as a deal (configurable via env)
DEAL_THRESHOLD = float(os.getenv("DEAL_THRESHOLD", "15.0"))  # 15% by default


# ── Main function ─────────────────────────────────────────────────────────────

def detect_deals(
    db_path=DB_PATH,
    deal_threshold: float = DEAL_THRESHOLD,
    log_callback=None,
) -> list[dict]:
    """
    Process all new listings (is_new=1), predict fair prices, and flag deals.

    Args:
        db_path: Path to the SQLite database.
        deal_threshold: Minimum % below predicted price to flag as deal.
        log_callback: Optional callable(str) for streaming logs to dashboard.

    Returns:
        List of dicts for listings flagged as deals.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    _log(f"Deal detector starting | threshold = {deal_threshold}%")

    # Reload statistical predictor with latest DB data
    reload_statistical_data()

    listings = get_new_listings(db_path)
    _log(f"Processing {len(listings)} new listings …")

    deals = []
    skipped = 0
    errors = 0

    for listing in listings:
        url   = listing["listing_url"]
        make  = listing.get("make", "Unknown")
        model = listing.get("model", "Unknown")
        year  = listing.get("year")
        mil   = listing.get("mileage")
        city  = listing.get("city")
        tx    = listing.get("transmission")
        price = listing.get("price")

        # Skip if no price
        if not price or price <= 0:
            skipped += 1
            continue

        try:
            pred = predict_price(
                make=make, model=model, year=year,
                mileage=mil, city=city, transmission=tx,
            )
            predicted_price = pred["predicted_price"]

            # If no comparable group exists, skip — don't flag as deal
            if predicted_price is None:
                logger.debug(f"  No prediction available for {make} {model} {year} — skipping")
                update_prediction(
                    listing_url=url, predicted_price=None,
                    price_gap_pct=0.0, is_deal=0,
                    prediction_method=pred.get("method"),
                    prediction_level=pred.get("fallback_level"),
                    confidence_text=pred.get("confidence_text"),
                    db_path=db_path,
                )
                skipped += 1
                continue

            # Gap: positive means listing is BELOW predicted (good deal)
            price_gap_pct = (predicted_price - price) / predicted_price * 100
            is_deal = 1 if price_gap_pct >= deal_threshold else 0

            update_prediction(
                listing_url=url,
                predicted_price=predicted_price,
                price_gap_pct=round(price_gap_pct, 2),
                is_deal=is_deal,
                prediction_method=pred.get("method"),
                prediction_level=pred.get("fallback_level"),
                confidence_text=pred.get("confidence_text"),
                db_path=db_path,
            )

            if is_deal:
                deal_info = {
                    **listing,
                    "predicted_price": predicted_price,
                    "price_gap_pct": round(price_gap_pct, 2),
                    "prediction_method": pred.get("method"),
                    "prediction_confidence": pred.get("confidence"),
                }
                deals.append(deal_info)
                _log(
                    f"  🔥 DEAL: {make} {model} {year} | "
                    f"Listed PKR {price:,} | Predicted PKR {predicted_price:,} | "
                    f"Gap {price_gap_pct:.1f}%"
                )
            else:
                logger.debug(
                    f"  Not a deal: {make} {model} {year} | "
                    f"Listed PKR {price:,} | Predicted PKR {predicted_price:,} | "
                    f"Gap {price_gap_pct:.1f}%"
                )

        except Exception as e:
            errors += 1
            logger.error(f"Error processing {url}: {e}")
            # Still mark as processed to avoid re-processing stuck listings
            try:
                update_prediction(
                    listing_url=url,
                    predicted_price=0,
                    price_gap_pct=0.0,
                    is_deal=0,
                    db_path=db_path,
                )
            except Exception:
                pass

    summary = (
        f"Deal detection complete | "
        f"Processed: {len(listings)} | "
        f"Deals: {len(deals)} | "
        f"Skipped: {skipped} | "
        f"Errors: {errors}"
    )
    _log(summary)
    return deals


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"GaadiScout Deal Detector — threshold {DEAL_THRESHOLD}%")
    print("=" * 60)
    init_db()
    found = detect_deals()
    print(f"\n✅ Found {len(found)} deals")
    for d in found:
        print(
            f"  • {d['make']} {d['model']} {d.get('year','')} | "
            f"PKR {d['price']:,} vs predicted PKR {d['predicted_price']:,} | "
            f"{d['price_gap_pct']:.1f}% below market"
        )
    if not found:
        print("  (No deals found — run the scraper first to collect listings)")
