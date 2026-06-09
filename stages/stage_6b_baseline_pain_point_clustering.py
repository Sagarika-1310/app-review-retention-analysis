"""
stage_6b_baseline_pain_point_clustering.py
─────────────────────────────────────
Stage 6b: Loads extracted embeddings, fits BERTopic (UMAP + HDBSCAN),
names clusters via Ollama Llama 3, and updates the shared registry infrastructure.

Exposes:
- run(run_version): Entrypoint invoked by the centralized orchestrator.
"""

import os
import logging
import warnings
import requests
import numpy as np
import pandas as pd
from uuid import uuid4
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import mlflow

# Import shared registry abstractions
from src.topic_registry.registry import TopicRegistry
from src.topic_registry.centroid_store import CentroidStore

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_6B_PAIN_POINT_CLUSTERING")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_retention_scored.parquet")
EMBEDDINGS_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_embeddings.npy")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_pain_points_clustered.parquet")
CLUSTER_ARTEFACTS_DIR = os.path.join(PROJECT_ROOT, "models", "pain_point_clusters")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "all-mpnet-base-v2")

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "registry", "topic_registry.json")
CENTROID_STORE_PATH = os.path.join(PROJECT_ROOT, "registry", "centroid_store.parquet")

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_PATH.split("/")[-1]
NEGATIVE_SENTIMENT_VAL = 0
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
NOISE_TOPIC_ID = -1
NOISE_CATEGORY = "Uncategorised"


# =============================================================================
# LLAMA CLUSTER NAMING UTILITIES (LOCAL OLLAMA CONTAINER HOOK)
# =============================================================================

def _build_naming_prompt(keywords: list[str], sample_reviews: list[str]) -> str:
    kw_str = ", ".join(keywords[:15])
    rev_str = "\n".join(f"- {r[:200]}" for r in sample_reviews[:5])
    return (
        "You are a product analyst. Based on the keywords and sample reviews below, "
        "generate a concise 3-5 word pain point category name that precisely describes "
        "the user complaint theme. Output ONLY the category name — no explanation, "
        "no punctuation, no quotes.\n\n"
        f"Keywords: {kw_str}\n\n"
        f"Sample reviews:\n{rev_str}\n\n"
        "Pain point category name:"
    )


def name_cluster_with_llama(keywords: list[str], sample_reviews: list[str]) -> str:
    """Invokes local Ollama framework container to produce high-fidelity topic names."""
    prompt = _build_naming_prompt(keywords, sample_reviews)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()
        return raw.strip("\"'.,\n").strip()
    except requests.exceptions.ConnectionError:
        log.warning("Ollama API instance unreachable. Triaging keyword fallback array.")
        return " / ".join(keywords[:3])
    except Exception as e:
        log.warning(f"Llama 3 synthesizer failed to compute category name: {e}")
        return " / ".join(keywords[:3])


def generate_cluster_name_map(topic_model: BERTopic, docs: list[str], doc_topics: list[int], risk_scores: pd.Series) -> \
dict[int, str]:
    """Iterates active clusters, using high-risk user profiles as framing tokens."""
    topic_ids = [t for t in set(doc_topics) if t != NOISE_TOPIC_ID]
    name_map = {NOISE_TOPIC_ID: NOISE_CATEGORY}

    temp_df = pd.DataFrame({"text": docs, "topic": doc_topics, "risk_score": risk_scores.values})

    for topic_id in sorted(topic_ids):
        topic_info = topic_model.get_topic(topic_id)
        keywords = [kw for kw, _ in topic_info] if topic_info else []

        cluster_df = temp_df[temp_df["topic"] == topic_id]
        top_risk_reviews = cluster_df.sort_values(by="risk_score", ascending=False).head(5)["text"].tolist()

        log.info(f"Synthesizing taxonomy for Cluster ID {topic_id} | Token keywords: {keywords[:5]}")
        name_map[int(topic_id)] = name_cluster_with_llama(keywords, top_risk_reviews)
        log.info(f"  → Resolved Category Name: '{name_map[int(topic_id)]}'")

    return name_map


# =============================================================================
# MODEL ASSEMBLY FACTORY
# =============================================================================

def build_topic_model() -> BERTopic:
    """Constructs tuned density-based clustering pipelines (UMAP + HDBSCAN)."""
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", low_memory=True, random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=10, min_samples=5, metric="euclidean", cluster_selection_method="eom", prediction_data=True
    )
    return BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, language="english",
        calculate_probabilities=False, verbose=True, nr_topics="auto", min_topic_size=10
    )


def fit_baseline_and_bootstrap_registry(texts: list[str], embeddings: np.ndarray, risk_scores: pd.Series) -> tuple:
    """Fits BERTopic pipelines and constructs L2-normalized geometric centroids."""
    log.info("Starting baseline BERTopic clustering routine...")
    topic_model = build_topic_model()
    topics, _ = topic_model.fit_transform(texts, embeddings)

    log.info(f"Density clustering complete. Discovered {len(set(topics)) - 1} semantic themes (excl. noise).")
    name_map = generate_cluster_name_map(topic_model, texts, topics, risk_scores)

    # Calculate L2-normalized geometric cluster centroids
    centroids = {}
    confidence_scores = np.zeros(len(topics))

    for topic_id in set(topics):
        if topic_id == NOISE_TOPIC_ID:
            continue
        mask = np.array(topics) == topic_id
        cluster_embeddings = embeddings[mask]

        raw_centroid = cluster_embeddings.mean(axis=0)
        norm = np.linalg.norm(raw_centroid)
        centroids[topic_id] = raw_centroid / norm if norm > 1e-8 else raw_centroid

    # Generate mapping cosine distances
    for i, topic_id in enumerate(topics):
        if topic_id == NOISE_TOPIC_ID:
            confidence_scores[i] = 0.0
        else:
            confidence_scores[i] = float(np.dot(embeddings[i], centroids[topic_id]))

    # Export underlying model state files as binary tracking archives
    os.makedirs(CLUSTER_ARTEFACTS_DIR, exist_ok=True)
    model_path = os.path.join(CLUSTER_ARTEFACTS_DIR, "base_bertopic_model")
    topic_model.save(model_path, serialization="pickle", save_embedding_model=False)

    return topic_model, name_map, centroids, topics, confidence_scores


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Consumes saved embedding arrays, handles density fitting, and builds structured schemas.
    """
    log.info(f"Executing Stage 6b Pain Point Clustering with shared pipeline version identifier: {run_version}")

    # 1. Input Guardrails Check
    if not os.path.exists(INPUT_PATH) or not os.path.exists(EMBEDDINGS_INPUT_PATH):
        raise FileNotFoundError("Missing Stage 6 inputs. Verify that Stage 5 and Stage 6a executed successfully.")

    df = pd.read_parquet(INPUT_PATH)
    embeddings = np.load(EMBEDDINGS_INPUT_PATH)

    negative_mask = df["predicted_sentiment"] == NEGATIVE_SENTIMENT_VAL
    df_negative = df[negative_mask].copy()

    if len(df_negative) != embeddings.shape[0]:
        raise ValueError(
            "Critical Mismatch: Input negative dataframe dimensions do not correspond with the saved embeddings matrix.")

    TEXT_COLUMN = "content"
    texts = df_negative[TEXT_COLUMN].fillna("").tolist()
    risk_scores = df_negative["inherent_risk_score"]

    # 2. Fit and populate structured registries inside child run tracking scope
    discovery_run_id = f"baseline_{uuid4().hex[:8]}"

    topic_model, name_map, centroids, topics, confidence_scores = fit_baseline_and_bootstrap_registry(
        texts, embeddings, risk_scores
    )

    # 3. Handle Local Registry State Refreshes
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)
    if os.path.exists(CENTROID_STORE_PATH):
        os.remove(CENTROID_STORE_PATH)

    registry = TopicRegistry(REGISTRY_PATH)
    centroid_store = CentroidStore(CENTROID_STORE_PATH)

    # 4. Populate Local Storage and Memory Registries
    topic_array = np.array(topics)
    for topic_id, name in name_map.items():
        if topic_id == NOISE_TOPIC_ID:
            continue

        topic_info = topic_model.get_topic(topic_id)
        keywords = [kw for kw, _ in topic_info[:15]] if topic_info else []
        mask = topic_array == topic_id
        review_count = int(mask.sum())

        representative_doc_ids = (
            df_negative.loc[mask]
            .sort_values(by="inherent_risk_score", ascending=False)["reviewId"]
            .astype(str).tolist()[:10]
        )

        registry.add_topic(
            topic_id=int(topic_id), name=name, model_version=EMBEDDING_MODEL_NAME,
            discovery_run_id=discovery_run_id, keywords=keywords,
            representative_doc_ids=representative_doc_ids, review_count_at_creation=review_count,
            notes="Initial BERTopic baseline fit"
        )

        centroid_store.add_centroid(
            topic_id=int(topic_id), centroid=centroids[topic_id],
            model_version=EMBEDDING_MODEL_NAME, review_count=review_count
        )

    registry.save()
    centroid_store.save()
    log.info("Successfully bootstrapped local TopicRegistry and CentroidStore metadata models.")

    # 5. Format Target Output Schema Columns
    df["cluster_id"] = np.nan
    df["pain_point_category"] = None
    df["assignment_confidence"] = np.nan
    df["assignment_source"] = pd.NA

    df.loc[negative_mask, "cluster_id"] = [float(t) for t in topics]
    df.loc[negative_mask, "pain_point_category"] = [name_map[t] for t in topics]
    df.loc[negative_mask, "assignment_confidence"] = confidence_scores
    df.loc[negative_mask, "assignment_source"] = "initial_fit"
    df.loc[~negative_mask, "assignment_source"] = "non_negative"

    # 6. Compute Extraction Quality Performance Metrics
    topic_series = pd.Series(topics)
    unique_clusters = int(topic_series.nunique()) - 1
    noise_count = int((topic_series == NOISE_TOPIC_ID).sum())
    noise_pct = round(noise_count / len(topic_series) * 100, 2)

    mlflow.log_param("clustering_algorithm", "BERTopic_UMAP_HDBSCAN")
    mlflow.log_metric("unique_pain_point_clusters", unique_clusters)
    mlflow.log_metric("noise_percentage", noise_pct)

    # 7. Save Final Data Artifacts to Disk and Log to MLflow
    df.to_parquet(OUTPUT_PATH, index=False)
    mlflow.log_artifact(OUTPUT_PATH, artifact_path="pain_point_clusters")
    mlflow.log_artifact(REGISTRY_PATH, artifact_path="registry_root")
    mlflow.log_artifact(CENTROID_STORE_PATH, artifact_path="centroid_root")

    log.info(f"Stage 6b complete. Cleaned clustering assets successfully synchronized at: {OUTPUT_PATH}")
