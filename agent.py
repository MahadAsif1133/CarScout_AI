"""
agent.py — GaadiScout AI Deal Evaluator
Uses Groq API (free tier) + LangChain to evaluate candidate deals,
produce structured JSON scores, and detect risk flags in listing descriptions.
"""

import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from db import (
    DB_PATH,
    get_all_deals,
    update_agent_result,
    init_db,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AGENT] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
AGENT_MODEL     = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")   # free-tier Groq model
AGENT_THRESHOLD = int(os.getenv("AGENT_THRESHOLD", "7"))             # min deal_score to pass

# ── Risk-flag keyword map (pre-LLM fast filter on description text) ────────────
RISK_KEYWORDS: dict[str, list[str]] = {
    "flood_mention":      ["flood", "flooded", "water damage", "submerged"],
    "no_documents":       ["no documents", "no docs", "without documents", "undocumented",
                           "no papers", "missing documents", "no registration"],
    "salvage_title":      ["salvage", "scrap", "total loss", "write-off", "writeoff"],
    "accident_history":   ["accident", "accidental", "major accident", "crashed", "crash",
                           "collision", "badly damaged"],
    "odometer_rollback":  ["odometer tamper", "rollback", "meter tampered", "km changed"],
    "engine_issues":      ["engine issue", "smoke", "overheating", "knocking", "misfiring",
                           "oil leak", "needs engine"],
    "theft_recovered":    ["stolen", "theft", "recovered", "FIR", "challan"],
}


# ── Pydantic output schema ────────────────────────────────────────────────────

class DealEvaluation(BaseModel):
    deal_score: int = Field(
        description="Overall deal quality score from 1 (worst) to 10 (best). "
                    "10 = exceptional deal, very low risk. 1 = bad deal or high risk.",
        ge=1, le=10,
    )
    price_gap_percent: float = Field(
        description="Estimated percentage the listed price is below fair market value. "
                    "Positive = below market (good). Negative = above market (bad)."
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="List of detected risk flags. Valid values: flood_mention, "
                    "no_documents, salvage_title, accident_history, odometer_rollback, "
                    "engine_issues, theft_recovered. Empty list if no risks detected.",
    )
    summary: str = Field(
        description="One or two sentence summary explaining the deal quality and any concerns. "
                    "Be specific about what makes it a good or bad deal."
    )


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    model = os.getenv("AGENT_MODEL", AGENT_MODEL)
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.1,
        max_tokens=512,
    )


def _build_structured_chain():
    """Build a LangChain chain that returns DealEvaluation."""
    llm = _get_llm()
    return llm.with_structured_output(DealEvaluation)


# Lazy singleton to avoid re-initialising on every call
_chain = None

def get_chain():
    global _chain
    if _chain is None:
        _chain = _build_structured_chain()
    return _chain


# ── Pre-filter: fast keyword-based risk detection ─────────────────────────────

def detect_risk_flags_heuristic(description: str) -> list[str]:
    """
    Fast keyword scan of listing description to detect obvious risk flags.
    Returns list of flag names. This supplements (not replaces) LLM analysis.
    """
    if not description:
        return []
    text = description.lower()
    found = []
    for flag, keywords in RISK_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.append(flag)
    return found


# ── Agent evaluation ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are GaadiScout, an expert Pakistani used-car market analyst.
You evaluate car listings from PakWheels.com and assess whether they represent
genuine deals or have hidden problems.

Your job:
1. Analyse the listing description for ANY red flags (flood, accident, missing documents, etc.)
2. Score the deal from 1-10 based on the price gap AND risk level
3. A listing with a large price gap but serious risks should score LOW (3-5)
4. A clean listing with a moderate gap should score HIGH (7-9)
5. Be conservative — when uncertain, lower the score

Important: The Pakistani used-car market has specific concerns:
- Missing registration / transfer documents is a major red flag
- Flood-damaged cars in Karachi/Sindh are common after monsoon
- Odometer tampering is widespread
- "Open letter" / token issues can complicate transfer
"""

def evaluate_single_deal(listing: dict) -> Optional[DealEvaluation]:
    """
    Evaluate one candidate deal using the Groq LLM.
    Returns DealEvaluation, or None if the call fails.
    """
    make         = listing.get("make", "Unknown")
    model_name   = listing.get("model", "Unknown")
    year         = listing.get("year", "Unknown")
    price        = listing.get("price", 0)
    pred_price   = listing.get("predicted_price", 0)
    gap_pct      = listing.get("price_gap_pct", 0)
    city         = listing.get("city", "Unknown")
    mileage      = listing.get("mileage")
    transmission = listing.get("transmission", "Unknown")
    engine_type  = listing.get("engine_type", "Unknown")
    description  = listing.get("description") or ""
    url          = listing.get("listing_url", "")

    # Pre-filter heuristic flags
    heuristic_flags = detect_risk_flags_heuristic(description)

    mileage_str = f"{mileage:,} km" if mileage else "unknown"
    desc_excerpt = description[:800] if description else "(no description provided)"

    user_message = f"""Please evaluate this used car listing from PakWheels.com:

**Car Details:**
- Make/Model/Year: {make} {model_name} {year}
- Listed Price: PKR {price:,}
- Predicted Fair Price: PKR {pred_price:,}
- Price Gap: {gap_pct:.1f}% below market
- Mileage: {mileage_str}
- City: {city}
- Transmission: {transmission}
- Engine: {engine_type}
- Listing URL: {url}

**Listing Description:**
{desc_excerpt}

**Pre-detected risk keywords:** {heuristic_flags if heuristic_flags else "None"}

Based on the description and the price gap, provide your structured evaluation."""

    try:
        chain = get_chain()
        result: DealEvaluation = chain.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])
        # Merge heuristic flags with LLM-detected flags (union, deduplicated)
        all_flags = list(set(result.risk_flags + heuristic_flags))
        result.risk_flags = all_flags
        return result
    except Exception as e:
        logger.error(f"LLM evaluation failed for {url}: {e}")
        return None


# ── Main pipeline function ────────────────────────────────────────────────────

def evaluate_deals(
    db_path=DB_PATH,
    agent_threshold: int = AGENT_THRESHOLD,
    log_callback=None,
) -> list[dict]:
    """
    Evaluate all candidate deals (is_deal=1) that haven't been agent-scored yet.
    Writes scores back to DB. Returns list of qualifying deals (score >= threshold).

    Args:
        db_path: Path to SQLite DB.
        agent_threshold: Minimum deal_score for a deal to pass.
        log_callback: Optional callable(str) for streaming logs to dashboard.
    """
    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    _log(f"Agent evaluator starting | model = {AGENT_MODEL} | threshold = {agent_threshold}/10")

    # Get all deals that haven't been scored yet
    all_deals = get_all_deals(db_path)
    unscored = [d for d in all_deals if d.get("deal_score") is None]

    _log(f"Found {len(all_deals)} total deals, {len(unscored)} unscored → evaluating …")

    qualifying = []

    for listing in unscored:
        url = listing.get("listing_url", "")
        _log(f"  Evaluating: {listing.get('make')} {listing.get('model')} {listing.get('year')} — {url}")

        evaluation = evaluate_single_deal(listing)

        if evaluation is None:
            # Evaluation failed — assign a conservative score
            logger.warning(f"  ⚠️  Evaluation failed, assigning fallback score 5")
            evaluation = DealEvaluation(
                deal_score=5,
                price_gap_percent=listing.get("price_gap_pct", 0),
                risk_flags=detect_risk_flags_heuristic(listing.get("description", "")),
                summary="Automatic evaluation failed. Price gap looks promising but verify manually.",
            )

        # Write results to DB
        update_agent_result(
            listing_url=url,
            deal_score=evaluation.deal_score,
            agent_summary=evaluation.summary,
            risk_flags=evaluation.risk_flags,
            db_path=db_path,
        )

        _log(
            f"  → Score: {evaluation.deal_score}/10 | "
            f"Flags: {evaluation.risk_flags or 'None'} | "
            f"{evaluation.summary[:80]}…"
        )

        # Check if it passes the threshold
        serious_flags = {"flood_mention", "no_documents", "salvage_title", "theft_recovered"}
        has_serious_flag = bool(set(evaluation.risk_flags) & serious_flags)

        if evaluation.deal_score >= agent_threshold and not has_serious_flag:
            qualifying.append({
                **listing,
                "deal_score": evaluation.deal_score,
                "agent_summary": evaluation.summary,
                "risk_flags": evaluation.risk_flags,
            })
            _log(f"  ✅ QUALIFIED (score {evaluation.deal_score} ≥ {agent_threshold}, no serious flags)")
        else:
            reason = f"score {evaluation.deal_score} < {agent_threshold}" if evaluation.deal_score < agent_threshold else "serious risk flags"
            _log(f"  ❌ Filtered out ({reason})")

    _log(f"Agent evaluation complete | Qualifying deals: {len(qualifying)}/{len(unscored)}")
    return qualifying


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print(f"GaadiScout Agent — Groq/{AGENT_MODEL}")
    print("=" * 60)

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    init_db()

    # Test with a mock listing
    mock_listing = {
        "listing_url": "https://www.pakwheels.com/used-cars/test/123",
        "make": "Toyota",
        "model": "Corolla",
        "year": 2019,
        "price": 2_500_000,
        "predicted_price": 3_100_000,
        "price_gap_pct": 19.4,
        "mileage": 62000,
        "city": "Lahore",
        "transmission": "Automatic",
        "engine_type": "1800cc Petrol",
        "description": (
            "Well maintained Toyota Corolla 2019. Full documentation available. "
            "First owner. All original parts. Bank financing available. "
            "Serious buyers only. No accidents."
        ),
    }

    print("\nEvaluating mock listing …")
    result = evaluate_single_deal(mock_listing)
    if result:
        print(f"\n✅ Evaluation result:")
        print(f"  Deal Score:  {result.deal_score}/10")
        print(f"  Price Gap:   {result.price_gap_percent:.1f}%")
        print(f"  Risk Flags:  {result.risk_flags or 'None'}")
        print(f"  Summary:     {result.summary}")
    else:
        print("❌ Evaluation failed — check GROQ_API_KEY and network connectivity")
