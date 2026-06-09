"""
stage_7a_incremental_assignment.py
──────────────────────────────────────
Stage 7a: Incremental Batch Assignment.

Loads the persistent topic taxonomy registry and centroid stores, extracts dense
embeddings from new incoming negative sentiment data streams, and maps them to
the closest active operational themes using a cosine similarity ceiling gate.
Unmatched anomalies are written out to the persistent NoiseBuffer.

Exposes:
- run(run_version): Entrypoint invoked by the central pipeline orchestrator.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import mlflow

# Import shared registry components
from utils.topic_registry.registry import TopicRegistry
from utils.topic_registry.centroid_store import CentroidStore, NOISE_TOPIC_ID
from utils.topic_registry.noise_buffer import NoiseBuffer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_7A_INCREMENTAL_ASSIGNMENT")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_retention_scored.parquet")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_pain_points_clustered.parquet")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "registry", "topic_registry.json")
CENTROID_STORE_PATH = os.path.join(PROJECT_ROOT, "registry", "centroid_store.parquet")
NOISE_BUFFER_PATH = os.path.join(PROJECT_ROOT, "registry", "noise_buffer.parquet")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "all-mpnet-base-v2")

# ── Configuration Constants ───────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_PATH
NEGATIVE_SENTIMENT_VAL = 0
SIMILARITY_THRESHOLD = 0.35  # Cosine similarity threshold for assignment vs noise routing
TEXT_COLUMN = "content"
NOISE_TOPIC_ID = -1
NOISE_CATEGORY = "Uncategorised"


# =============================================================================
# FEATURE TRANSFORMATION UTILITIES
# =============================================================================

def embed_texts(texts: list[str]) -> np.ndarray:
    """Encodes incoming textual feedback using all-mpnet-base-v2 (L2 normalized)."""
    log.info(f"Loading transformer encoding instance: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return normalize(embeddings)


# =============================================================================
# ROUTING ENGINE
# =============================================================================

def run_incremental_assignment(
        df: pd.DataFrame,
        registry: TopicRegistry,
        centroid_store: CentroidStore,
        noise_buffer: NoiseBuffer,
        batch_month: str,
) -> pd.DataFrame:
    """Routes negative reviews via nearest-centroid vector mapping calculations."""
    log.info("─" * 65)
    log.info(f"EXECUTING SIMILARITY MAPPING ROUTINES FOR BATCH BLOCK: {batch_month}")
    log.info("─" * 65)

    df["cluster_id"] = np.nan
    df["pain_point_category"] = pd.NA
    df["assignment_confidence"] = np.nan
    df["assignment_source"] = pd.NA

    negative_mask = df["predicted_sentiment"] == NEGATIVE_SENTIMENT_VAL
    df_negative = df[negative_mask].copy()

    if df_negative.empty:
        log.warning("Transaction slice contains zero negative user sentiment records.")
        return df

    texts = df_negative[TEXT_COLUMN].fillna("").tolist()
    embeddings = embed_texts(texts)

    active_ids = registry.get_active_topic_ids()
    log.info(f"Pulled {len(active_ids)} operational centroid vectors from active configuration state.")

    assigned_ids, confidence_scores = centroid_store.assign(
        embeddings=embeddings,
        active_topic_ids=active_ids,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )

    matched_mask = np.array(assigned_ids) != NOISE_TOPIC_ID
    noise_indices = np.where(~matched_mask)[0]
    matched_indices = np.where(matched_mask)[0]

    log.info(f"Routing Distribution Summary | Allocated: {int(matched_mask.sum()):,} lines | "
             f"Sent to Noise Buffer: {int((~matched_mask).sum()):,} lines ({(1 - matched_mask.mean()) * 100:.1f}%)")

    cluster_ids = np.full(len(texts), NOISE_TOPIC_ID, dtype=float)
    categories = np.full(len(texts), NOISE_CATEGORY, dtype=object)
    sources = np.full(len(texts), "noise_buffer", dtype=object)

    for i in matched_indices:
        cluster_ids[i] = assigned_ids[i]
        categories[i] = registry.get_name(assigned_ids[i])
        sources[i] = "centroid_match"

    df.loc[negative_mask, "cluster_id"] = cluster_ids
    df.loc[negative_mask, "pain_point_category"] = categories
    df.loc[negative_mask, "assignment_confidence"] = confidence_scores
    df.loc[negative_mask, "assignment_source"] = sources
    df.loc[~negative_mask, "assignment_source"] = "non_negative"

    # Enqueue low-confidence anomaly text blocks into the global NoiseBuffer
    if noise_indices.size > 0:
        noise_texts = [texts[i] for i in noise_indices]
        noise_embeddings = embeddings[noise_indices]
        noise_best_ids = [assigned_ids[i] for i in noise_indices]
        noise_scores = [confidence_scores[i] for i in noise_indices]

        df_neg_reset = df_negative.reset_index(drop=True)
        noise_review_ids = [str(df_neg_reset.iloc[i].get("reviewId", i)) for i in noise_indices]

        noise_buffer.add_reviews(
            review_ids=noise_review_ids,
            texts=noise_texts,
            embeddings=noise_embeddings,
            best_match_topic_ids=noise_best_ids,
            best_match_similarities=noise_scores,
            batch_month=batch_month,
        )
        noise_buffer.save()

    return df


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """Orchestrates ingestion, vector assignment profiling, and anomaly routing metrics."""
    log.info(f"Executing Stage 7a Ingestion Pipeline and Routing with version token: {run_version}")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing upstream dependencies: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)
    df = df[df["is_new"].isin([True, "true"])]
    log.info(f"Ingested {len(df):,} freshly appended incremental records for structural routing evaluation.")

    if "review_date" in df.columns:
        batch_month = pd.to_datetime(df["review_date"]).max().strftime("%Y-%m")
    else:
        batch_month = datetime.now().strftime("%Y-%m")
    log.info(f"Targeting partition window window frame: {batch_month}")

    registry = TopicRegistry(REGISTRY_PATH)
    centroid_store = CentroidStore(CENTROID_STORE_PATH)
    noise_buffer = NoiseBuffer(NOISE_BUFFER_PATH)

    if registry.is_empty():
        raise RuntimeError(
            "Taxonomy directory structure appears uninitialized. Bootstrap Stage 6b baseline arrays first.")

    df = run_incremental_assignment(
        df=df, registry=registry, centroid_store=centroid_store, noise_buffer=noise_buffer, batch_month=batch_month
    )

    assigned_count = int((df["assignment_source"] == "centroid_match").sum())
    noise_count = int((df["assignment_source"] == "noise_buffer").sum())
    noise_pct = round(noise_count / max(assigned_count + noise_count, 1) * 100, 2)
    buffer_size = noise_buffer.unpromoted_size()

    log.info(
        f"Metrics Captured -> Assigned Count: {assigned_count:,} | Noise Block Delta: {noise_count:,} | Persistent Pool Depth: {buffer_size:,}")

    mlflow.log_param("incremental_embedding_model", EMBEDDING_MODEL_NAME)
    mlflow.log_param("routing_similarity_threshold", SIMILARITY_THRESHOLD)
    mlflow.log_param("execution_batch_month", batch_month)
    mlflow.log_metric("assigned_review_count", assigned_count)
    mlflow.log_metric("noise_review_count", noise_count)
    mlflow.log_metric("anomaly_noise_percentage", noise_pct)
    mlflow.log_metric("noise_buffer_total_size", buffer_size)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    mlflow.log_artifact(OUTPUT_PATH, artifact_path="pain_point_clusters")
    mlflow.log_artifact(REGISTRY_PATH, artifact_path="registry_snapshot")
    log.info(f"Stage 7a processing cycle successfully shut down. Saved output file to: {OUTPUT_PATH}")
