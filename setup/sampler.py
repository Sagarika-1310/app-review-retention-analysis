"""
sampler.py
----------
Generates a stratified sample of processed premium reviews across different
star ratings to create a balanced dataset for manual sentiment labeling.

Pipeline stages:
    1. Load processed premium full parquet dataset
    2. Validate data availability and structure
    3. Apply stratified sampling (up to 80 reviews per star rating bucket)
    4. Isolate required evaluation columns and inject placeholder for manual entry
    5. Save output to a clean CSV layout ready for annotation

Run from project root:
    python src/sampler.py
"""

import os
import logging
import pandas as pd

# ── Logging Setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "sampler.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_full.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "labelling")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sentiment_labelling_sample.csv")

# Ensure labeling directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# MAIN EXCLUSION & SAMPLING PIPELINE
# =============================================================================

if __name__ == "__main__":
    log.info("=" * 65)
    log.info("STARTING STRATIFIED SAMPLING PIPELINE")
    log.info("=" * 65)

    # Stage 1: Load preprocessed premium data
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {INPUT_PATH}. "
            "Please ensure you run `src/preprocessor.py` successfully first."
        )

    log.info(f"Loading premium dataset from {INPUT_PATH}")
    df_premium_full = pd.read_parquet(INPUT_PATH)
    log.info(f"  Loaded {len(df_premium_full):,} records available for sampling")

    # Stage 2: Structural Verification
    if "score" not in df_premium_full.columns or "content" not in df_premium_full.columns:
        raise KeyError("Input dataset is missing critical structural columns ('score' or 'content').")

    # Gracefully handle true_label column if it doesn't exist yet in input
    if "true_label" not in df_premium_full.columns:
        log.info("  Column 'true_label' not present in source. Initializing empty string placeholders.")
        df_premium_full["true_label"] = ""

    # Stage 3: Perform Stratified Sampling via GroupBy
    log.info("Executing stratified sampling (target: up to 80 reviews per star rating group)...")

    sample = (
        df_premium_full
        .groupby("score", group_keys=False)
        .apply(lambda x: x.sample(min(80, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    # Stage 4: Execution Metrics and Verification
    log.info(f"  Successfully extracted total sample of {len(sample):,} records")
    log.info("  Sample breakdown per star rating:")
    counts = sample["score"].value_counts().sort_index()
    for star, count in counts.items():
        log.info(f"    {star}★ : {count:>2} reviews")

    # Stage 5: Isolate Target Columns & Export to CSV
    log.info(f"Saving curated verification sample layout to {OUTPUT_PATH}")

    # Selecting exactly what's needed for clean sheet annotation
    final_cols = ["reviewId", "content", "score", "true_label"]
    sample_output = sample[final_cols]

    sample_output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    log.info("SAMPLING PIPELINE COMPLETE — Dataset ready for manual entry.")
    log.info("=" * 65)