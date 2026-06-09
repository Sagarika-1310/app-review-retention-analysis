"""
stage_6a_baseline_embeddings_extraction.py
─────────────────────────────────────
Stage 6a: Extracts dense L2-normalized embeddings from negative sentiment reviews
using all-mpnet-base-v2 and serializes the matrix to disk for clustering.

Exposes:
- run(run_version): Entrypoint invoked by the centralized orchestrator.
"""

import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import mlflow

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_6A_EMBEDDINGS_EXTRACTION")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_retention_scored.parquet")
EMBEDDINGS_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_embeddings.npy")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "all-mpnet-base-v2")

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_PATH
NEGATIVE_SENTIMENT_VAL = 0


# =============================================================================
# EMBEDDING EXTRACTION CORE
# =============================================================================

def embed_reviews(texts: list[str]) -> np.ndarray:
    """Encodes text segments using all-mpnet-base-v2 and applies L2 normalization."""
    log.info(f"Loading transformer embedding architecture: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    log.info(f"Encoding {len(texts):,} target negative reviews...")
    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    elapsed_time = time.time() - start_time
    log.info(f"Embedding generation complete. Inference duration: {elapsed_time:.2f}s")

    return normalize(embeddings)


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Isolates negative review layers and stores raw matrix files on disk.
    """
    log.info(f"Executing Stage 6a Embeddings Extraction with shared pipeline version identifier: {run_version}")

    # 1. Input Guardrails Check
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input tracking asset: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    log.info(f"Loaded scored feedback dataset containing {len(df):,} total records.")

    negative_mask = df["predicted_sentiment"] == NEGATIVE_SENTIMENT_VAL
    df_negative = df[negative_mask].copy()

    if df_negative.empty:
        raise ValueError("Aborting pipeline task. Zero negative sentiment rows discovered.")

    TEXT_COLUMN = "content"
    texts = df_negative[TEXT_COLUMN].fillna("").tolist()

    # 2. Extract and Persist Embeddings inside centralized nested child tracking run
    log.info(f"Isolating {len(texts):,} negative text layers for feature mapping.")

    embeddings = embed_reviews(texts)

    # Ensure the target directory structure exists before writing matrix profiles
    os.makedirs(os.path.dirname(EMBEDDINGS_OUTPUT_PATH), exist_ok=True)
    np.save(EMBEDDINGS_OUTPUT_PATH, embeddings)
    log.info(f"Successfully serialized dense embeddings matrix array to {EMBEDDINGS_OUTPUT_PATH}")

    # 3. Log Parameters and Artifact Metrics to MLflow Context
    mlflow.log_param("embedding_model_variant", EMBEDDING_MODEL_NAME)
    mlflow.log_param("text_column_embedded", TEXT_COLUMN)
    mlflow.log_metric("negative_reviews_count", len(texts))

    # Track the generated .npy file as an unmanaged tracking artifact
    mlflow.log_artifact(EMBEDDINGS_OUTPUT_PATH, artifact_path="intermediate_embeddings")
    log.info("Stage 6a embedding extraction pipeline execution complete.")
