"""
predict.py — GaadiScout Fair-Price Predictor (v2)

Statistical predictor with strict hierarchical fallback — NEVER collapses
to make-only, which caused identical predictions across different models/years.

Fallback chain (most specific → least specific):
  1. make + model + year (exact) + mileage band (±20 000 km)   → 3+ samples
  2. make + model + year (exact)                                → 3+ samples
  3. make + model + year band (±2 years)                        → 3+ samples
  4. make + model (all years)                                   → 3+ samples
  5. make + body segment (hatchback/sedan/suv…)                 → 3+ samples
  [NEVER]: make-only — returns None instead, caller gets no prediction

After computing group median, applies:
  - Mileage adjustment: ±1% per 10 000 km above/below group mean
  - Year adjustment: ±1.5% per year above/below group mean year

Fine-tuned LLM predictor (optional): loads lora_adapter/ if present.
"""

import os
import re
import logging
import sqlite3
import statistics
from pathlib import Path
from typing import Literal

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PREDICT] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
LORA_ADAPTER_PATH = Path(__file__).parent / "lora_adapter"
BASE_MODEL_ID     = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

PREDICTION_MODE: Literal["auto", "statistical", "finetuned"] = os.getenv(
    "PREDICTION_MODE", "auto"
)

# Grouping parameters
MILEAGE_BAND_SIZE  = 20_000   # km — tighter than before (was 25 000)
YEAR_BAND_RADIUS   = 2        # ±2 years for year-band fallback
MIN_SAMPLE_SIZE    = 3        # minimum samples to trust a group

# Adjustment rates
MILEAGE_ADJ_PER_10K = 0.01   # 1% per 10 000 km above/below group mean
YEAR_ADJ_PER_YEAR   = 0.015  # 1.5% per year above/below group mean year

# Body segment keywords for last-resort fallback (before giving up)
SEGMENT_KEYWORDS = {
    "hatchback": ["mehran", "alto", "cultus", "swift", "mira", "move", "wagon r",
                  "vitz", "passo", "cuore", "bolan", "ravi", "atrai"],
    "sedan":     ["corolla", "civic", "city", "liana", "baleno", "margalla",
                  "premio", "allion", "camry", "sonata", "elantra"],
    "suv":       ["sportage", "tucson", "prado", "fortuner", "cr-v", "hrv", "br-v",
                  "rav4", "creta", "stonic", "c3", "chery"],
    "pickup":    ["hilux", "revo", "dmax", "ranger", "frontier"],
}


def _get_base_model(model_str: str) -> str:
    if not model_str:
        return ""
    words = str(model_str).lower().split()
    if not words:
        return ""
    # Keep first two words for known multi-word models
    if len(words) > 1 and words[0] in ("wagon", "land", "n", "grand", "atrai"):
        return f"{words[0]} {words[1]}"
    return words[0]


def _infer_segment(make: str, model: str) -> str | None:
    combined = f"{make} {model}".lower()
    for seg, keywords in SEGMENT_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return seg
    return None


# ── Statistical Predictor ─────────────────────────────────────────────────────

class StatisticalPredictor:
    """
    Hierarchical median-based price predictor.
    Never falls back to make-only; returns None instead.
    Applies mileage and year adjustments after group selection.
    """

    def __init__(self):
        self._data:   list[dict] = []
        self._loaded: bool       = False

    def load(self, db_path=None):
        if db_path is None:
            from db import DB_PATH
            db_path = DB_PATH
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT make, model, year, mileage, price
                   FROM listings
                   WHERE price IS NOT NULL AND price > 50000 AND price < 200000000"""
            ).fetchall()
            conn.close()
            self._data   = [dict(r) for r in rows]
            self._loaded = True
            logger.info(f"Statistical predictor loaded {len(self._data)} historical listings")
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            self._data   = []
            self._loaded = True

    # ── Filter helpers ────────────────────────────────────────────────────────

    def _match(self, row: dict, **criteria) -> bool:
        for key, val in criteria.items():
            if key == "year_range":
                lo, hi = val
                if row.get("year") is None or not (lo <= row["year"] <= hi):
                    return False
            elif key == "mileage_band":
                lo, hi = val
                m = row.get("mileage")
                if m is None or not (lo <= m < hi):
                    return False
            elif key == "segment":
                if _infer_segment(row.get("make",""), row.get("model","")) != val:
                    return False
            elif key == "model":
                a = _get_base_model(row.get("model", ""))
                b = _get_base_model(val)
                if a != b:
                    return False
            else:
                a = str(row.get(key, "")).lower().strip()
                b = str(val).lower().strip()
                if a != b:
                    return False
        return True

    def _prices(self, **criteria) -> list[int]:
        return [r["price"] for r in self._data if self._match(r, **criteria)]

    def _group_stats(self, rows: list[dict]) -> dict:
        """Return mean mileage and mean year for a group."""
        mileages = [r["mileage"] for r in self._data if r.get("mileage")]
        years    = [r["year"]    for r in self._data if r.get("year")]
        return {
            "mean_mileage": statistics.mean(mileages) if mileages else None,
            "mean_year":    statistics.mean(years)    if years    else None,
        }

    def _group_rows(self, **criteria) -> list[dict]:
        return [r for r in self._data if self._match(r, **criteria)]

    # ── Adjustment ────────────────────────────────────────────────────────────

    def _apply_adjustments(
        self,
        base_price:   int,
        query_mileage: int | None,
        query_year:    int | None,
        group_rows:    list[dict],
    ) -> int:
        """
        Adjust base median price for mileage and year differences vs group.
        """
        adjusted = float(base_price)

        group_mileages = [r["mileage"] for r in group_rows if r.get("mileage")]
        group_years    = [r["year"]    for r in group_rows if r.get("year")]

        # Mileage adjustment: ±1% per 10 000 km vs group mean
        if query_mileage is not None and group_mileages:
            mean_mil = statistics.mean(group_mileages)
            delta_km = query_mileage - mean_mil
            # Positive delta = higher mileage than group = lower price
            adjustment_factor = 1.0 - (delta_km / 10_000) * MILEAGE_ADJ_PER_10K
            # Cap adjustment at ±25%
            adjustment_factor = max(0.75, min(1.25, adjustment_factor))
            adjusted *= adjustment_factor
            logger.debug(
                f"Mileage adj: query={query_mileage:,}, mean={mean_mil:,.0f}, "
                f"delta={delta_km:+,.0f}km, factor={adjustment_factor:.3f}"
            )

        # Year adjustment: ±1.5% per year vs group mean (newer = higher price)
        if query_year is not None and group_years:
            mean_yr = statistics.mean(group_years)
            delta_yr = query_year - mean_yr
            adjustment_factor = 1.0 + delta_yr * YEAR_ADJ_PER_YEAR
            adjustment_factor = max(0.70, min(1.30, adjustment_factor))
            adjusted *= adjustment_factor
            logger.debug(
                f"Year adj: query={query_year}, mean={mean_yr:.1f}, "
                f"delta={delta_yr:+.1f}yr, factor={adjustment_factor:.3f}"
            )

        return max(50_000, int(adjusted))

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        make:     str,
        model:    str,
        year:     int | None = None,
        mileage:  int | None = None,
    ) -> dict:
        """
        Returns dict with:
          predicted_price (int), confidence (str), sample_size (int),
          method (str), fallback_level (int 1-5), method_description (str)
        Returns method='no_data' if no suitable group found.
        """
        if not self._loaded:
            self.load()

        make  = (make  or "").strip().title()
        model = (model or "").strip().title()

        # Mileage band boundaries
        if mileage is not None:
            mil_lo = (mileage // MILEAGE_BAND_SIZE) * MILEAGE_BAND_SIZE
            mil_hi = mil_lo + MILEAGE_BAND_SIZE
            mil_band = (mil_lo, mil_hi)
        else:
            mil_band = None

        year_range = (year - YEAR_BAND_RADIUS, year + YEAR_BAND_RADIUS) if year else None

        # ── Level 1: make + model + exact year + mileage band ─────────────────
        if year and mil_band:
            rows = self._group_rows(make=make, model=model, year=year, mileage_band=mil_band)
            if len(rows) >= MIN_SAMPLE_SIZE:
                return self._result(rows, make, model, year, mileage,
                                    level=1, method="exact_model_year_mileage",
                                    desc=f"{make} {model} {year} ±{MILEAGE_BAND_SIZE//1000}k km")

        # ── Level 2: make + model + exact year ───────────────────────────────
        if year:
            rows = self._group_rows(make=make, model=model, year=year)
            if len(rows) >= MIN_SAMPLE_SIZE:
                return self._result(rows, make, model, year, mileage,
                                    level=2, method="exact_model_year",
                                    desc=f"{make} {model} {year}")

        # ── Level 3: make + model + year band ────────────────────────────────
        if year_range:
            rows = self._group_rows(make=make, model=model, year_range=year_range)
            if len(rows) >= MIN_SAMPLE_SIZE:
                return self._result(rows, make, model, year, mileage,
                                    level=3, method="model_year_band",
                                    desc=f"{make} {model} {year-YEAR_BAND_RADIUS}–{year+YEAR_BAND_RADIUS}")

        # ── Level 4: make + model (all years) ────────────────────────────────
        rows = self._group_rows(make=make, model=model)
        if len(rows) >= MIN_SAMPLE_SIZE:
            return self._result(rows, make, model, year, mileage,
                                level=4, method="model_only",
                                desc=f"{make} {model} (all years)")

        # ── Level 5: make + body segment ─────────────────────────────────────
        segment = _infer_segment(make, model)
        if segment:
            rows = self._group_rows(make=make, segment=segment)
            if len(rows) >= MIN_SAMPLE_SIZE:
                return self._result(rows, make, model, year, mileage,
                                    level=5, method="make_segment",
                                    desc=f"{make} {segment}s (any model)")

        # ── No data — NEVER return make-only ─────────────────────────────────
        logger.warning(
            f"No suitable comparison group for {make} {model} {year} "
            f"(total DB: {len(self._data)} listings)"
        )
        return {
            "predicted_price":    None,
            "confidence":         "none",
            "sample_size":        0,
            "method":             "no_data",
            "fallback_level":     None,
            "method_description": f"No comparable listings found for {make} {model}",
            "adjusted":           False,
        }

    def _result(
        self,
        rows:    list[dict],
        make:    str,
        model:   str,
        year:    int | None,
        mileage: int | None,
        level:   int,
        method:  str,
        desc:    str,
    ) -> dict:
        prices     = [r["price"] for r in rows]
        raw_median = int(statistics.median(prices))
        adjusted   = self._apply_adjustments(raw_median, mileage, year, rows)

        n = len(prices)
        confidence = "high" if n >= 10 else "medium" if n >= 5 else "low"

        confidence_text = (
            f"Based on {n} similar listings ({confidence} confidence)"
            if n >= MIN_SAMPLE_SIZE else
            f"Based on {n} listing(s) — very low confidence"
        )

        return {
            "predicted_price":    adjusted,
            "raw_median":         raw_median,
            "confidence":         confidence,
            "sample_size":        n,
            "method":             method,
            "fallback_level":     level,
            "method_description": desc,
            "confidence_text":    confidence_text,
            "adjusted":           (adjusted != raw_median),
        }


# ── Fine-tuned LLM Predictor ──────────────────────────────────────────────────

class FinetunedPredictor:
    """
    QLoRA-fine-tuned LLM predictor. Requires ./lora_adapter/ from train_model.py.
    """

    def __init__(self):
        self._model     = None
        self._tokenizer = None
        self._loaded    = False

    def _load(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            logger.info(f"Loading base model: {BASE_MODEL_ID}")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model  = PeftModel.from_pretrained(base, str(LORA_ADAPTER_PATH))
            self._model.eval()
            self._loaded = True
            logger.info("Fine-tuned model loaded")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise

    @staticmethod
    def _format_prompt(make, model, year, mileage, city, transmission) -> str:
        mileage_str = f"{mileage:,} km" if mileage else "unknown"
        return (
            f"### Car Listing\nMake: {make}\nModel: {model}\nYear: {year or 'unknown'}\n"
            f"Mileage: {mileage_str}\nCity: {city or 'Pakistan'}\n"
            f"Transmission: {transmission or 'unknown'}\n\n"
            f"### Predicted Fair Market Price (PKR)\n"
        )

    @staticmethod
    def _parse_price(text: str) -> int | None:
        lac_m = re.search(r"([\d.]+)\s*lac", text, re.I)
        if lac_m:
            return int(float(lac_m.group(1)) * 100_000)
        crore_m = re.search(r"([\d.]+)\s*crore", text, re.I)
        if crore_m:
            return int(float(crore_m.group(1)) * 10_000_000)
        digits_m = re.search(r"\b(\d[\d,]{4,})\b", text)
        if digits_m:
            return int(re.sub(r",", "", digits_m.group(1)))
        return None

    def predict(self, make, model, year, mileage, city=None, transmission=None) -> dict:
        if not self._loaded:
            self._load()
        prompt = self._format_prompt(make, model, year, mileage, city, transmission)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with __import__("torch").no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=32, temperature=0.1,
                do_sample=False, pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        price = self._parse_price(generated)
        if price is None:
            raise ValueError(f"Could not parse price from: {generated!r}")
        return {
            "predicted_price":    price,
            "confidence":         "finetuned",
            "sample_size":        None,
            "method":             "finetuned_llm",
            "fallback_level":     0,
            "method_description": "Fine-tuned LLM prediction",
            "confidence_text":    "Fine-tuned model prediction",
            "adjusted":           False,
        }


# ── Singletons & router ───────────────────────────────────────────────────────

_stat_predictor: StatisticalPredictor | None = None
_ft_predictor:   FinetunedPredictor   | None = None


def _get_stat() -> StatisticalPredictor:
    global _stat_predictor
    if _stat_predictor is None:
        _stat_predictor = StatisticalPredictor()
        _stat_predictor.load()
    return _stat_predictor


def _get_ft() -> FinetunedPredictor:
    global _ft_predictor
    if _ft_predictor is None:
        _ft_predictor = FinetunedPredictor()
    return _ft_predictor


def predict_price(
    make:         str,
    model:        str,
    year:         int | None = None,
    mileage:      int | None = None,
    city:         str | None = None,
    transmission: str | None = None,
) -> dict:
    """
    Main entry point. Returns dict with:
      predicted_price, confidence, sample_size, method,
      fallback_level, method_description, confidence_text
    """
    mode = PREDICTION_MODE
    if mode == "auto":
        mode = "finetuned" if LORA_ADAPTER_PATH.exists() else "statistical"

    if mode == "finetuned":
        try:
            return _get_ft().predict(make, model, year, mileage, city, transmission)
        except Exception as e:
            logger.warning(f"Fine-tuned prediction failed ({e}), falling back to statistical")

    return _get_stat().predict(make, model, year, mileage)


def reload_statistical_data():
    """Force reload of statistical predictor with fresh DB data."""
    global _stat_predictor
    _stat_predictor = None
    _get_stat()


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("GaadiScout Predictor v2 — Hierarchical Statistical Predictor")
    print("=" * 70)

    # Load DB data first
    pred = _get_stat()

    # ── The 8 known-bad test cases from the bug report ────────────────────────
    test_cases = [
        ("Suzuki",   "Swift",        1985, 120_000),
        ("Suzuki",   "Mehran",       1991,  80_000),
        ("Suzuki",   "Mehran",       2005,  60_000),
        ("Suzuki",   "Cultus",       2005,  70_000),
        ("Daihatsu", "Cuore",        2005,  65_000),
        ("Daihatsu", "Atrai Wagon",  2013,  93_000),
        ("Toyota",   "Yaris",        2025,   5_000),
        ("Toyota",   "Vitz",         2013,  80_000),
    ]

    print(f"\n{'Car':<40} {'Fair Value':>14} {'Method':<30} {'Samples':>7} {'Conf'}")
    print("-" * 100)

    last_prices = {}
    for make, model, year, mileage in test_cases:
        result = pred.predict(make, model, year, mileage)
        price  = result.get("predicted_price")
        key    = f"{make} {model} {year}"
        last_prices[key] = price

        price_str = f"PKR {price:>11,}" if price else "  NO DATA FOUND"
        print(
            f"  {make} {model} {year:<6} {mileage//1000:>3}k km  "
            f"{price_str}  "
            f"{result['method']:<30}  "
            f"{result['sample_size'] or 0:>5}  "
            f"{result['confidence']}"
        )

    # ── Uniqueness check ──────────────────────────────────────────────────────
    valid_prices = [v for v in last_prices.values() if v is not None]
    unique_prices = set(valid_prices)
    print(f"\n{'='*70}")
    if len(unique_prices) == len(valid_prices):
        print(f"PASS: All {len(valid_prices)} predictions are unique values")
    else:
        duplicates = len(valid_prices) - len(unique_prices)
        print(f"FAIL: {duplicates} duplicate price value(s) — grouping still broken!")
    print(f"{'='*70}")
