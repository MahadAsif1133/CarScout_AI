"""
tests/test_phase4_mongodb.py — Phase 4 tests for optional MongoDB raw listing store.
"""

from unittest.mock import MagicMock, patch
import mongo_store


SAMPLE_RAW = [
    {
        "listing_url": "https://www.pakwheels.com/used-cars/toyota-corolla-2020-for-sale-in-lahore-9876543",
        "make": "Toyota",
        "model": "Corolla",
        "year": 2020,
        "price": 3900000,
        "scraped_at": "2026-11-17T12:00:00",
    }
]


def test_mongo_noop_when_uri_unset(monkeypatch):
    """When MONGODB_URI is not set, save_raw_listings is a no-op and does not connect."""
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with patch("mongo_store._get_mongo_client") as mock_client_fn:
        count = mongo_store.save_raw_listings(SAMPLE_RAW)

    assert count == 0
    mock_client_fn.assert_not_called()


def test_mongo_upserts_on_listing_id(monkeypatch):
    """When MONGODB_URI is set, save_raw_listings upserts each listing by listing_id with scraped_at and source_url."""
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")
    monkeypatch.setenv("MONGODB_DB", "carscout")
    monkeypatch.setenv("MONGODB_COLLECTION", "raw_listings")

    mock_collection = MagicMock()
    mock_db = {"raw_listings": mock_collection}
    mock_client = MagicMock()
    mock_client.__getitem__.return_value = mock_db

    with patch("mongo_store._get_mongo_client", return_value=mock_client):
        count = mongo_store.save_raw_listings(SAMPLE_RAW)

    assert count == 1
    mock_collection.update_one.assert_called_once()
    filter_arg, update_arg = mock_collection.update_one.call_args[0]
    kwargs_arg = mock_collection.update_one.call_args[1]

    assert filter_arg == {"listing_id": "9876543"}
    assert kwargs_arg == {"upsert": True}
    saved_doc = update_arg["$set"]
    assert saved_doc["listing_id"] == "9876543"
    assert saved_doc["scraped_at"] == "2026-11-17T12:00:00"
    assert saved_doc["source_url"] == SAMPLE_RAW[0]["listing_url"]


def test_mongo_unreachable_logs_warning_and_continues(monkeypatch):
    """If MongoDB raises a connection error, save_raw_listings logs a warning and returns 0 without raising."""
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost:27017")

    with patch("mongo_store._get_mongo_client", side_effect=RuntimeError("ServerSelectionTimeoutError")):
        count = mongo_store.save_raw_listings(SAMPLE_RAW)

    assert count == 0
