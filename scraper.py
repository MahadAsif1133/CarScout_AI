"""
scraper.py — GaadiScout PakWheels Scraper (v2)
Primary data source: JSON-LD <script type="application/ld+json"> embedded in
each listing card — this gives perfectly-structured, reliable data for all
key fields without fragile CSS-class scraping.

Fallback: plain HTML parsing for any field missing from JSON-LD.
"""

import re
import json
import time
import random
import logging
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCRAPER] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL   = "https://www.pakwheels.com"
SEARCH_URL = "https://www.pakwheels.com/used-cars/search/-/"

USER_AGENT = (
    "GaadiScout/1.0 (educational price-analysis bot; "
    "scrapes public used-car search listings; "
    "respects robots.txt; contact: gaadiscout@example.com)"
)
HEADERS = {
    "User-Agent":      USER_AGENT,
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

DISALLOWED_PATTERNS = [
    "/used-cars/lightbox_window", "/used-cars/notify_seller",
    "/used-cars/manage/",         "/used-cars/activate/",
    "sortby=",                    "nf=true",
]

MIN_DELAY = 1.0
MAX_DELAY = 2.5

# ── Minimum data-quality thresholds (warn if below these) ────────────────────
WARN_MISSING_PCT = 10.0   # warn if >10% of listings missing a key field


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_disallowed(url: str) -> bool:
    return any(pat in url for pat in DISALLOWED_PATTERNS)


def _polite_delay():
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def _fetch_page(url: str, session: requests.Session) -> BeautifulSoup | None:
    if _is_disallowed(url):
        logger.warning(f"Skipping disallowed URL: {url}")
        return None
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code in (403, 429, 503):
            logger.warning(
                f"Received HTTP {resp.status_code} from {url} — "
                "PakWheels may be rate-limiting or blocking this runner IP."
            )
            return None
        resp.raise_for_status()
        return BeautifulSoup(resp.content, "html.parser")
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None


def _clean_price(raw) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    s = str(raw).strip().upper()
    lac_m = re.search(r"([\d.]+)\s*LAC", s)
    if lac_m:
        return int(float(lac_m.group(1)) * 100_000)
    crore_m = re.search(r"([\d.]+)\s*CRORE", s)
    if crore_m:
        return int(float(crore_m.group(1)) * 10_000_000)
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None


def _clean_mileage(raw) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    digits = re.sub(r"[^\d]", "", str(raw))
    return int(digits) if digits else None


def _clean_year(raw) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    m = re.search(r"\b(19|20)\d{2}\b", str(raw))
    return int(m.group()) if m else None


def _extract_city_from_name(name: str) -> str | None:
    """
    PakWheels listing names follow the pattern:
    'Toyota Corolla 2019 for sale in Karachi'
    Extract the city after 'for sale in'.
    """
    if not name:
        return None
    m = re.search(r"for\s+sale\s+in\s+([A-Za-z\s]+)$", name, re.I)
    if m:
        return m.group(1).strip().title()
    return None


def _parse_engine_spec(engine_str: str) -> tuple[str | None, str | None]:
    """
    Parse 'Petrol . 1600 cc . Automatic' → (engine_type, transmission)
    Also handles JSON-LD engineDisplacement like '1600cc'.
    """
    if not engine_str:
        return None, None
    parts = [p.strip() for p in re.split(r"[.\|·]", engine_str)]
    fuel = None
    transmission = None
    engine_type = None
    for p in parts:
        pl = p.lower()
        if pl in ("petrol", "diesel", "hybrid", "electric", "cng", "lpg"):
            fuel = p.title()
        elif "automatic" in pl or "manual" in pl or "cvt" in pl or "dct" in pl:
            transmission = p.title()
        elif "cc" in pl:
            engine_type = p.strip()
    if fuel and engine_type:
        engine_type = f"{fuel} {engine_type}"
    elif fuel:
        engine_type = fuel
    return engine_type, transmission


# ── Primary parser: JSON-LD ───────────────────────────────────────────────────

def _parse_card_jsonld(card) -> dict | None:
    """
    Extract all fields from the embedded JSON-LD Schema.org Product block.
    This is the authoritative, reliable data source — always present.
    Returns None if JSON-LD is absent or malformed.
    """
    script_tag = card.find("script", type="application/ld+json")
    if not script_tag or not script_tag.string:
        return None
    try:
        data = json.loads(script_tag.string)
    except (json.JSONDecodeError, ValueError):
        return None

    # ── Listing URL from offers ───────────────────────────────────────────────
    listing_url = (data.get("offers") or {}).get("url", "")
    if not listing_url:
        # Fallback: look for a <a> with href pointing to /used-cars/...
        a_tag = card.find("a", href=re.compile(r"/used-cars/[^/]+-\d+"))
        if a_tag:
            href = a_tag.get("href", "")
            listing_url = href if href.startswith("http") else urljoin(BASE_URL, href)

    # ── Price ─────────────────────────────────────────────────────────────────
    price = None
    offers = data.get("offers")
    if offers:
        price = _clean_price(offers.get("price"))
    # HTML fallback for price — "PKR 46 lacs" text in the listing
    if price is None:
        price_tag = card.find(class_=re.compile(r"price", re.I))
        if price_tag:
            price = _clean_price(price_tag.get_text(strip=True))

    # ── Make / Model / Year ───────────────────────────────────────────────────
    make  = (data.get("manufacturer") or data.get("brand", {}).get("name") or "").strip().title()
    year  = _clean_year(data.get("modelDate"))
    name  = data.get("name") or ""

    # Extract model from the listing name: "Toyota Corolla 2019 for sale in Karachi"
    # Strip make, year, "for sale in <city>" to get model
    model_raw = name
    model_raw = re.sub(re.escape(make), "", model_raw, flags=re.I).strip()
    model_raw = re.sub(r"\b(19|20)\d{2}\b", "", model_raw).strip()
    model_raw = re.sub(r"\s+for\s+sale.*$", "", model_raw, flags=re.I).strip()
    # Also strip variant/trim from title tag href (richer model name there)
    # e.g. "Altis Automatic 1.6" — try the <a> title attribute
    a_title = card.find("a", class_=re.compile(r"car-name|title", re.I))
    if a_title:
        full_title = a_title.get_text(strip=True)
        # Strip year, "for Sale", "X.X/10", "in <city>"
        full_title = re.sub(r"\b(19|20)\d{2}\b", "", full_title)
        full_title = re.sub(r"\s+(for\s+sale.*|\d+\.\d+/\d+.*)$", "", full_title, flags=re.I)
        full_title = re.sub(re.escape(make), "", full_title, flags=re.I).strip()
        if len(full_title) >= len(model_raw):
            model_raw = full_title

    model = " ".join(model_raw.split()).title() or "Unknown"
    if not model or model == make:
        model = "Unknown"

    # ── City from name field ──────────────────────────────────────────────────
    city = _extract_city_from_name(name)

    # ── Transmission & Engine from JSON-LD ────────────────────────────────────
    transmission = data.get("vehicleTransmission", "").strip().title() or None
    engine_disp  = (data.get("vehicleEngine") or {}).get("engineDisplacement", "")
    fuel_type    = data.get("fuelType", "").strip().title() or None

    if engine_disp and fuel_type:
        engine_type = f"{fuel_type} {engine_disp}"
    elif engine_disp:
        engine_type = engine_disp
    elif fuel_type:
        engine_type = fuel_type
    else:
        engine_type = None

    # ── Mileage ───────────────────────────────────────────────────────────────
    mileage = _clean_mileage(data.get("mileageFromOdometer"))

    # ── Image ─────────────────────────────────────────────────────────────────
    image_url = data.get("image")
    if not image_url:
        img = card.find("img")
        if img:
            image_url = img.get("data-src") or img.get("src")
            if image_url and not image_url.startswith("http"):
                image_url = urljoin(BASE_URL, image_url)

    # ── Description & posted date from HTML ───────────────────────────────────
    desc = data.get("description") or ""
    # Try to get richer description from the HTML card text
    main_div = card.find("div", class_=re.compile(r"search-list|well"))
    posted_date = None
    if main_div:
        # Posted/updated date (e.g. "Updated 1 minute ago")
        for el in main_div.find_all(string=re.compile(r"(Updated|Posted|ago)", re.I)):
            s = el.strip()
            if s:
                posted_date = s
                break

    if not listing_url or price is None:
        return None

    return {
        "make":              make or "Unknown",
        "model":             model,
        "year":              year,
        "mileage":           mileage,
        "city":              city,
        "price":             price,
        "registration_city": None,   # not in search results; available on detail page
        "transmission":      transmission,
        "engine_type":       engine_type,
        "listing_url":       listing_url,
        "description":       desc,
        "posted_date":       posted_date,
        "image_url":         image_url,
    }


# ── Data quality validation ───────────────────────────────────────────────────

def validate_listings(listings: list[dict]) -> dict:
    """
    Check what percentage of listings have missing key fields.
    Logs a warning for any field missing in >WARN_MISSING_PCT% of listings.
    Returns a dict of field→missing_pct for reporting.
    """
    if not listings:
        return {}

    key_fields = ["city", "transmission", "mileage", "price", "year", "make", "model"]
    total = len(listings)
    report = {}

    for field in key_fields:
        missing = sum(1 for l in listings if not l.get(field))
        pct = missing / total * 100
        report[field] = round(pct, 1)
        if pct > WARN_MISSING_PCT:
            logger.warning(
                f"DATA QUALITY WARNING: '{field}' is missing in "
                f"{missing}/{total} listings ({pct:.1f}%) — exceeds {WARN_MISSING_PCT}% threshold"
            )
        else:
            logger.info(f"  Data quality OK: '{field}' missing in {pct:.1f}% of listings")

    return report


# ── Main scraping function ────────────────────────────────────────────────────

def scrape_listings(
    max_pages: int = 5,
    fetch_details: bool = False,
    search_params: str = "",
) -> list[dict]:
    """
    Scrape PakWheels used-car search listings.
    Primary data extraction via JSON-LD (reliable, structured).
    Runs data-quality validation after each page batch.
    """
    session     = requests.Session()
    all_listings: list[dict] = []
    seen_urls:   set[str]    = set()
    scraped_at = datetime.utcnow().isoformat()

    base_search = f"{SEARCH_URL}{search_params}"
    logger.info(f"Starting scrape: {base_search} (max_pages={max_pages})")

    for page_num in range(1, max_pages + 1):
        page_url = base_search if page_num == 1 else f"{base_search}?page={page_num}"
        logger.info(f"Fetching page {page_num}: {page_url}")

        soup = _fetch_page(page_url, session)
        if not soup:
            logger.warning(f"Skipping page {page_num} (fetch failed)")
            _polite_delay()
            continue

        cards = soup.find_all("li", class_=lambda c: c and "classified-listing" in c)
        if not cards:
            logger.warning(f"No listing cards on page {page_num} — stopping")
            break

        logger.info(f"  Found {len(cards)} cards on page {page_num}")
        page_added = 0

        for card in cards:
            listing = _parse_card_jsonld(card)
            if not listing:
                continue
            url = listing["listing_url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            listing["scraped_at"] = scraped_at
            all_listings.append(listing)
            page_added += 1

        logger.info(f"  Added {page_added} listings (total so far: {len(all_listings)})")

        next_page = soup.find("a", attrs={"rel": "next"}) or soup.find("li", class_="next")
        if not next_page and page_num < max_pages:
            logger.info("  No next-page link — stopping pagination early")
            break

        if page_num < max_pages:
            _polite_delay()

    logger.info(f"Scrape complete: {len(all_listings)} total listings")

    # ── Data quality check ────────────────────────────────────────────────────
    logger.info("Running data quality validation …")
    report = validate_listings(all_listings)

    return all_listings


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("GaadiScout Scraper v2 — JSON-LD primary parser")
    print("=" * 60)

    results = scrape_listings(max_pages=2)

    if not results:
        print("No listings scraped")
        sys.exit(1)

    print(f"\nScraped {len(results)} listings")

    # ── Data quality report ───────────────────────────────────────────────────
    print("\n--- Data Quality Report ---")
    fields = ["city", "transmission", "mileage", "price", "year", "make", "model"]
    for f in fields:
        present = sum(1 for l in results if l.get(f))
        pct_ok = present / len(results) * 100
        status = "OK" if pct_ok >= 90 else "WARN"
        print(f"  [{status}] {f:15s}: {present}/{len(results)} ({pct_ok:.1f}%)")

    # ── Sample listings ───────────────────────────────────────────────────────
    print("\n--- Sample listings (first 5) ---")
    for r in results[:5]:
        print(
            f"  {r['make']:10s} {r['model'][:20]:20s} {r['year']}  |  "
            f"PKR {r.get('price',0):>10,}  |  "
            f"{r.get('mileage','?'):>8}km  |  "
            f"{r.get('city','?'):>12s}  |  "
            f"{r.get('transmission','?')}"
        )
