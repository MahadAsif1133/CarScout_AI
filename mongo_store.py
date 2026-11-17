"""
mongo_store.py — CarScout_AI / GaadiScout Optional MongoDB Raw Listing Store
When MONGODB_URI is set, saves each scraped listing's raw JSON (including
scraped_at and source_url / listing_url) to a MongoDB collection using pymongo,
with an upsert keyed on listing_id so repeated runs never create duplicates.
SQLite remains the source of truth for cleaned data and the dashboard.
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_NAME = "carscout"
DEFAULT_COLLECTION_NAME = "raw_listings"
MONGO_TIMEOUT_MS = 3000


def extract_listing_id(listing: dict) -> str:
    """
    Derive a deterministic listing_id from listing_url (e.g. trailing PakWheels
    numeric ID '...-1234567' or '/1234567', falling back to full listing_url).
    """
    if listing.get("listing_id"):
        return str(listing["listing_id"]).strip()
    url = (listing.get("listing_url") or "").strip().rstrip("/")
    if not url:
        return ""
    m = re.search(r"[-/](\d{5,})$", url)
    if m:
        return m.group(1)
    return url


def _get_mongo_client(uri: str):
    from pymongo import MongoClient
    return MongoClient(
        uri,
        serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
        connectTimeoutMS=MONGO_TIMEOUT_MS,
        socketTimeoutMS=MONGO_TIMEOUT_MS,
    )


def save_raw_listings(
    raw_listings: list[dict],
    mongo_uri: Optional[str] = None,
    db_name: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> int:
    """
    Upsert raw scraped listings into MongoDB when MONGODB_URI is configured.
    If MONGODB_URI is unset or MongoDB is unreachable, logs a warning and returns 0.
    Returns the number of listings processed.
    """
    uri = (mongo_uri if mongo_uri is not None else os.getenv("MONGODB_URI", "")).strip()
    if not uri or not raw_listings:
        return 0

    target_db = (db_name or os.getenv("MONGODB_DB", DEFAULT_DB_NAME)).strip()
    target_col = (
        collection_name or os.getenv("MONGODB_COLLECTION", DEFAULT_COLLECTION_NAME)
    ).strip()

    now_iso = datetime.utcnow().isoformat()
    saved_count = 0
    client = None

    try:
        client = _get_mongo_client(uri)
        collection = client[target_db][target_col]

        for item in raw_listings:
            listing_id = extract_listing_id(item)
            if not listing_id:
                continue

            doc = dict(item)
            doc["listing_id"] = listing_id
            doc["scraped_at"] = doc.get("scraped_at") or now_iso
            doc["source_url"] = doc.get("source_url") or doc.get("listing_url", "")

            collection.update_one(
                {"listing_id": listing_id},
                {"$set": doc},
                upsert=True,
            )
            saved_count += 1

        logger.info(
            f"MongoDB raw store: upserted {saved_count} listing(s) into {target_db}.{target_col}"
        )
        return saved_count
    except Exception as e:
        logger.warning(f"MongoDB unreachable or write failed (continuing with SQLite): {e}")
        return 0
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
