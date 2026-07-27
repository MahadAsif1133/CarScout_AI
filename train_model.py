"""
train_model.py — GaadiScout QLoRA Fine-Tuning Script
⚠️  Run this on Google Colab with a free GPU (T4 or better).
    It will NOT run on CPU-only machines due to bitsandbytes 4-bit quantization.

Workflow:
  1. Install dependencies (pip block at top)
  2. Authenticate with Hugging Face Hub (optional, for gated models)
  3. Load base model in 4-bit with bitsandbytes
  4. Wrap with LoRA config (peft)
  5. Format DB listings → prompt→price training pairs
  6. Train with trl.SFTTrainer
  7. Save adapter to ./lora_adapter/
  8. (Optional) Push adapter to Hugging Face Hub

Usage in Colab:
  1. Upload gaadiscout.db to /content/gaadiscout.db
  2. Run the notebook cells below
  3. Download ./lora_adapter/ after training
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 1: Install dependencies (run in Colab)
# ─────────────────────────────────────────────────────────────────────────────
"""
!pip install -q transformers==4.44.2 peft==0.12.0 bitsandbytes==0.43.3 \
              trl==0.10.1 accelerate==0.34.2 datasets==2.21.0 \
              scipy sentencepiece protobuf
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2: Imports & Config
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import sqlite3
import logging
import random
from pathlib import Path
from typing import Optional

# ── Training config ───────────────────────────────────────────────────────────
BASE_MODEL_ID  = "Qwen/Qwen2.5-1.5B-Instruct"   # or "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR     = "./lora_adapter"
DB_PATH        = "/content/gaadiscout.db"         # Colab path — adjust if needed

HF_TOKEN       = os.getenv("HF_TOKEN", "")        # Optional: set for gated models

# LoRA hyperparameters
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters
NUM_EPOCHS     = 3
BATCH_SIZE     = 4
GRAD_ACCUM     = 4         # effective batch size = 16
LR             = 2e-4
MAX_SEQ_LEN    = 256
WARMUP_STEPS   = 50

# Minimum listings needed to train (otherwise exit gracefully)
MIN_TRAIN_SAMPLES = 50

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 3: Load training data from SQLite
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(db_path: str) -> list[dict]:
    """
    Load car listings from SQLite and format as prompt→completion pairs.
    Only uses listings that have: make, model, year, price (all required).
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB not found: {db_path}. Upload gaadiscout.db to Colab first.")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT make, model, year, mileage, city, price, transmission, engine_type
           FROM listings
           WHERE make IS NOT NULL AND model IS NOT NULL
             AND year IS NOT NULL AND price IS NOT NULL
             AND price > 100000 AND price < 100000000  -- sanity bounds
           ORDER BY scraped_at DESC"""
    ).fetchall()
    conn.close()

    logger.info(f"Loaded {len(rows)} training examples from {db_path}")
    return [dict(r) for r in rows]


def format_example(row: dict) -> dict:
    """
    Format a DB row into a chat-style prompt → completion pair.

    Prompt: structured car spec description
    Completion: fair market price in PKR

    This teaches the model to associate car attributes → price.
    """
    mileage_str = f"{row['mileage']:,} km" if row.get("mileage") else "unknown mileage"
    city_str    = row.get("city") or "Pakistan"
    tx_str      = row.get("transmission") or "unknown"
    eng_str     = row.get("engine_type") or "unknown"
    price_lacs  = row["price"] / 100_000

    prompt = (
        f"### Car Listing\n"
        f"Make: {row['make']}\n"
        f"Model: {row['model']}\n"
        f"Year: {row['year']}\n"
        f"Mileage: {mileage_str}\n"
        f"City: {city_str}\n"
        f"Transmission: {tx_str}\n"
        f"Engine: {eng_str}\n\n"
        f"### Predicted Fair Market Price (PKR)\n"
    )
    completion = f"PKR {row['price']:,} ({price_lacs:.1f} Lacs)"

    # SFT format: full text = prompt + completion
    return {"text": prompt + completion}


# ─────────────────────────────────────────────────────────────────────────────
# Cell 4: Load model + tokenizer in 4-bit
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(model_id: str, hf_token: Optional[str] = None):
    """Load model in 4-bit NF4 quantization via bitsandbytes."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading model in 4-bit: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token or None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Cell 5: Wrap with LoRA
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model):
    """Wrap model with LoRA adapter using peft."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Cell 6: Train
# ─────────────────────────────────────────────────────────────────────────────

def train(db_path: str = DB_PATH):
    """Full training pipeline."""
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    # 1. Load data
    raw_data = load_training_data(db_path)
    if len(raw_data) < MIN_TRAIN_SAMPLES:
        logger.error(
            f"Only {len(raw_data)} training samples found. "
            f"Need at least {MIN_TRAIN_SAMPLES}. "
            "Run the scraper more to collect more data first."
        )
        return

    # 2. Format examples
    random.shuffle(raw_data)
    examples = [format_example(r) for r in raw_data]
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    eval_examples  = examples[split_idx:]

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset  = Dataset.from_list(eval_examples)
    logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # 3. Load model
    model, tokenizer = load_base_model(BASE_MODEL_ID, HF_TOKEN)

    # 4. Apply LoRA
    model = apply_lora(model)

    # 5. Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="paged_adamw_32bit",
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        report_to="none",
        gradient_checkpointing=True,
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    logger.info("Starting training …")
    trainer.train()

    # 7. Save adapter
    logger.info(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 8. Save training metadata
    metadata = {
        "base_model": BASE_MODEL_ID,
        "num_train_samples": len(train_dataset),
        "num_eval_samples": len(eval_dataset),
        "epochs": NUM_EPOCHS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    }
    with open(f"{OUTPUT_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✅ Training complete! Download ./lora_adapter/ and place in your project.")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Cell 7: Optional — Push to Hugging Face Hub
# ─────────────────────────────────────────────────────────────────────────────

def push_to_hub(repo_id: str, hf_token: str):
    """Push trained adapter to Hugging Face Hub."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=repo_id,
        repo_type="model",
    )
    logger.info(f"✅ Adapter pushed to https://huggingface.co/{repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (when run directly in Colab)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GaadiScout — QLoRA Fine-Tuning")
    print("⚠️  This should run on Google Colab with a GPU!")
    print("=" * 60)
    train(db_path=DB_PATH)
    # Uncomment to push:
    # push_to_hub("your-username/gaadiscout-price-predictor", HF_TOKEN)
