"""
db.py — GaadiScout Database Layer
SQLite schema, upsert, cleaning, and query helpers.
"""

import re
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DB] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "gaadiscout.db"

# ── Schema ────────────────────────────────────────────────────────────────────
LISTINGS_DDL = """
CREATE TABLE IF NOT EXISTS listings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_url         TEXT    UNIQUE NOT NULL,
    make                TEXT,
    model               TEXT,
    year                INTEGER,
    mileage             INTEGER,
    city                TEXT,
    price               INTEGER,
    registration_city   TEXT,
    transmission        TEXT,
    engine_type         TEXT,
    description         TEXT,
    posted_date         TEXT,
    image_url           TEXT,
    scraped_at          TEXT,
    predicted_price     INTEGER,
    price_gap_pct       REAL,
    deal_score          INTEGER,
    agent_summary       TEXT,
    risk_flags          TEXT,
    prediction_method   TEXT,        -- e.g. 'exact_model_year', 'model_only'
    prediction_level    INTEGER,     -- fallback level 1-5
    confidence_text     TEXT,        -- human-readable confidence string
    is_new              INTEGER DEFAULT 1,
    is_deal             INTEGER DEFAULT 0,
    is_alerted          INTEGER DEFAULT 0
);
"""

ALERTS_DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel         TEXT NOT NULL,        -- 'telegram' or 'email'
    listing_url     TEXT,
    sent_at         TEXT,                 -- ISO 8601 UTC
    deal_details    TEXT                  -- JSON blob of deal summary
);
"""

INDICES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_listings_is_new  ON listings(is_new);",
    "CREATE INDEX IF NOT EXISTS idx_listings_is_deal ON listings(is_deal);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_channel   ON alerts(channel);",
]


# ── Connection factory ────────────────────────────────────────────────────────

def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set to dict-like rows."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db(db_path: str | Path = DB_PATH) -> None:
    """Create tables and indices if they don't already exist."""
    with get_connection(db_path) as conn:
        conn.execute(LISTINGS_DDL)
        conn.execute(ALERTS_DDL)
        for idx_sql in INDICES_DDL:
            conn.execute(idx_sql)
        conn.commit()
    logger.info(f"Database initialised at {db_path}")


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _to_int(val) -> int | None:
    """Safely convert a value to int."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    s = re.sub(r"[^\d]", "", str(val))
    return int(s) if s else None


def _clean_mileage(raw) -> int | None:
    """Normalise mileage: '45,000 km' → 45000."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    digits = re.sub(r"[^\d]", "", str(raw))
    return int(digits) if digits else None


def _clean_price(raw) -> int | None:
    """Normalise price string → int PKR."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    s = str(raw).strip().upper()
    lac_m = re.search(r"([\d.]+)\s*LAC", s)
    if lac_m:
        return int(float(lac_m.group(1)) * 100_000)
    crore_m = re.search(r"([\d.]+)\s*CRORE", s)
    if crore_m:
        return int(float(crore_m.group(1)) * 10_000_000)
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


def _clean_year(raw) -> int | None:
    """Extract a 4-digit year."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    m = re.search(r"\b(19|20)\d{2}\b", str(raw))
    return int(m.group()) if m else None


def clean_listing(raw: dict) -> dict:
    """
    Apply all cleaning/normalisation to a raw listing dict from the scraper.
    Returns a clean dict ready for DB insertion.
    """
    return {
        "listing_url":       raw.get("listing_url", "").strip(),
        "make":              (raw.get("make") or "").strip().title() or None,
        "model":             (raw.get("model") or "").strip().title() or None,
        "year":              _clean_year(raw.get("year")),
        "mileage":           _clean_mileage(raw.get("mileage")),
        "city":              (raw.get("city") or "").strip().title() or None,
        "price":             _clean_price(raw.get("price")),
        "registration_city": (raw.get("registration_city") or "").strip().title() or None,
        "transmission":      (raw.get("transmission") or "").strip().title() or None,
        "engine_type":       (raw.get("engine_type") or "").strip() or None,
        "description":       (raw.get("description") or "").strip()[:3000] or None,
        "posted_date":       (raw.get("posted_date") or "").strip() or None,
        "image_url":         (raw.get("image_url") or "").strip() or None,
        "scraped_at":        raw.get("scraped_at") or datetime.utcnow().isoformat(),
    }


# ── Insert / Upsert ───────────────────────────────────────────────────────────

def upsert_listing(conn: sqlite3.Connection, listing: dict) -> bool:
    """
    Insert or update a listing. Deduplication key = listing_url.
    On conflict, updates all fields EXCEPT predicted_price, deal_score,
    agent_summary, risk_flags, is_deal, is_alerted (preserves analysis state).
    Returns True if a new row was inserted, False if updated.
    """
    sql = """
    INSERT INTO listings (
        listing_url, make, model, year, mileage, city, price,
        registration_city, transmission, engine_type, description,
        posted_date, image_url, scraped_at, is_new
    ) VALUES (
        :listing_url, :make, :model, :year, :mileage, :city, :price,
        :registration_city, :transmission, :engine_type, :description,
        :posted_date, :image_url, :scraped_at, 1
    )
    ON CONFLICT(listing_url) DO UPDATE SET
        make              = excluded.make,
        model             = excluded.model,
        year              = excluded.year,
        mileage           = excluded.mileage,
        city              = excluded.city,
        price             = excluded.price,
        registration_city = excluded.registration_city,
        transmission      = excluded.transmission,
        engine_type       = excluded.engine_type,
        description       = excluded.description,
        posted_date       = excluded.posted_date,
        image_url         = excluded.image_url,
        scraped_at        = excluded.scraped_at
    """
    cursor = conn.execute(sql, listing)
    return cursor.lastrowid is not None and cursor.rowcount == 1


def bulk_upsert_listings(raw_listings: list[dict], db_path: str | Path = DB_PATH) -> tuple[int, int]:
    """
    Clean and upsert a list of raw listings from the scraper.
    Returns (inserted_count, total_processed).
    """
    inserted = 0
    with get_connection(db_path) as conn:
        for raw in raw_listings:
            if not raw.get("listing_url") or not raw.get("price"):
                continue
            cleaned = clean_listing(raw)
            try:
                if upsert_listing(conn, cleaned):
                    inserted += 1
            except sqlite3.Error as e:
                logger.error(f"DB error on {raw.get('listing_url')}: {e}")
        conn.commit()
    logger.info(f"Upserted {len(raw_listings)} listings → {inserted} new rows")
    return inserted, len(raw_listings)


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_new_listings(db_path: str | Path = DB_PATH) -> list[dict]:
    """Return all listings not yet processed by the deal detector (is_new=1)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM listings WHERE is_new = 1 AND price IS NOT NULL ORDER BY scraped_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_deals(db_path: str | Path = DB_PATH) -> list[dict]:
    """Return all flagged deals (is_deal=1), newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM listings
               WHERE is_deal = 1
               ORDER BY deal_score DESC, price_gap_pct DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_alerts(db_path: str | Path = DB_PATH) -> list[dict]:
    """Return deals that have been scored but not yet alerted (is_alerted=0)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM listings
               WHERE is_deal = 1 AND is_alerted = 0 AND deal_score IS NOT NULL
               ORDER BY deal_score DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_alert_history(db_path: str | Path = DB_PATH, limit: int = 200) -> list[dict]:
    """Return recent alert history from the alerts table."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY sent_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_listings_with_predictions(db_path: str | Path = DB_PATH) -> list[dict]:
    """Return all listings that have a predicted price (for charts)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM listings
               WHERE predicted_price IS NOT NULL AND price IS NOT NULL
               ORDER BY scraped_at DESC
               LIMIT 500"""
        ).fetchall()
    return [dict(r) for r in rows]


# ── Update helpers ────────────────────────────────────────────────────────────

def update_prediction(
    listing_url:       str,
    predicted_price:   int | None,
    price_gap_pct:     float,
    is_deal:           int,
    prediction_method: str  = None,
    prediction_level:  int  = None,
    confidence_text:   str  = None,
    db_path: str | Path = DB_PATH,
) -> None:
    """Write predicted price + deal flag back to the listings row."""
    with get_connection(db_path) as conn:
        # Add columns if they don't exist yet (migration-safe)
        for col, coltype in [
            ("prediction_method", "TEXT"),
            ("prediction_level",  "INTEGER"),
            ("confidence_text",   "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE listings ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists
        conn.execute(
            """UPDATE listings
               SET predicted_price = ?, price_gap_pct = ?, is_deal = ?, is_new = 0,
                   prediction_method = ?, prediction_level = ?, confidence_text = ?
               WHERE listing_url = ?""",
            (predicted_price, price_gap_pct, is_deal,
             prediction_method, prediction_level, confidence_text,
             listing_url),
        )
        conn.commit()


def update_agent_result(
    listing_url: str,
    deal_score: int,
    agent_summary: str,
    risk_flags: list[str],
    db_path: str | Path = DB_PATH,
) -> None:
    """Write agent evaluation results back to the listings row."""
    with get_connection(db_path) as conn:
        conn.execute(
            """UPDATE listings
               SET deal_score = ?, agent_summary = ?, risk_flags = ?
               WHERE listing_url = ?""",
            (deal_score, agent_summary, json.dumps(risk_flags), listing_url),
        )
        conn.commit()


def mark_alerted(listing_url: str, db_path: str | Path = DB_PATH) -> None:
    """Mark a listing as alerted so we don't double-send."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE listings SET is_alerted = 1 WHERE listing_url = ?",
            (listing_url,),
        )
        conn.commit()


def log_alert(
    channel: str,
    listing_url: str,
    deal_details: dict,
    db_path: str | Path = DB_PATH,
) -> None:
    """Insert a row into the alerts table for dashboard history."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO alerts (channel, listing_url, sent_at, deal_details)
               VALUES (?, ?, ?, ?)""",
            (channel, listing_url, datetime.utcnow().isoformat(), json.dumps(deal_details)),
        )
        conn.commit()


def get_stats(db_path: str | Path = DB_PATH) -> dict:
    """Return quick summary statistics for the dashboard header."""
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        deals = conn.execute("SELECT COUNT(*) FROM listings WHERE is_deal=1").fetchone()[0]
        alerts = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        avg_gap = conn.execute(
            "SELECT AVG(price_gap_pct) FROM listings WHERE is_deal=1 AND price_gap_pct IS NOT NULL"
        ).fetchone()[0]
    return {
        "total_listings": total,
        "deals_found": deals,
        "alerts_sent": alerts,
        "avg_discount_pct": round(avg_gap, 1) if avg_gap else 0,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GaadiScout DB — initialising schema")
    print("=" * 60)
    init_db()

    # Insert a sample row
    sample = {
        "listing_url": "https://www.pakwheels.com/used-cars/test-car/123456",
        "make": "Toyota",
        "model": "Corolla",
        "year": 2019,
        "mileage": 55000,
        "city": "Lahore",
        "price": 3200000,
        "registration_city": "Lahore",
        "transmission": "Automatic",
        "engine_type": "1800cc Petrol",
        "description": "Well maintained car. Full documentation available.",
        "posted_date": "3 days ago",
        "image_url": None,
        "scraped_at": datetime.utcnow().isoformat(),
    }
    n, total = bulk_upsert_listings([sample])
    print(f"\n✅ Inserted {n}/{total} listings")

    stats = get_stats()
    print(f"\nDB stats: {stats}")

    new = get_new_listings()
    print(f"\nNew (unprocessed) listings: {len(new)}")
    if new:
        for k, v in new[0].items():
            print(f"  {k}: {v}")
