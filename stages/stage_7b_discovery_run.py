"""
stage_7b_discovery_run.py
─────────────────────────────
Stage 7b: Taxonomy Growth Discovery Engine.

Monitors the size of the unpromoted noise data pool. When it passes required volume 
ceilings (MIN_REVIEWS_FOR_DISCOVERY), this script runs density-based clustering 
(BERTopic) on the anomalies, queries local Ollama instances for taxonomy descriptions, 
and updates persistent geometric centroid stores.

Exposes:
- run(run_version): Entrypoint invoked by the central pipeline orchestrator.
"""

import os
import logging
import warnings
import mlflow

# Import shared registry components
from utils.topic_registry.registry import TopicRegistry
from utils.topic_registry.centroid_store import CentroidStore
from utils.topic_registry.noise_buffer import NoiseBuffer
from utils.topic_registry.discovery_engine import DiscoveryEngine
from utils.topic_registry.validator import TopicValidator, ValidationConfig
from utils.topic_registry.promoter import TopicPromoter
from utils.topic_registry.llm_namer import LLMNamer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_7B_DISCOVERY_RUN")

# ── Paths ─────────────────────────────────────────────────────────────────────
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "registry", "topic_registry.json")
CENTROID_STORE_PATH = os.path.join(PROJECT_ROOT, "registry", "centroid_store.parquet")
NOISE_BUFFER_PATH = os.path.join(PROJECT_ROOT, "registry", "noise_buffer.parquet")
DISCOVERY_RUNS_PATH = os.path.join(PROJECT_ROOT, "registry", "discovery_runs.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "pain_point_clusters")
EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "all-mpnet-base-v2")

# ── Configuration Constants ───────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_PATH
MIN_REVIEWS_FOR_DISCOVERY = 500  # Size checkpoint limit to trigger discovery transforms
REQUIRE_HUMAN_APPROVAL = False


# =============================================================================
# DYNAMIC CLUSTER EXTRACTION & CONSOLIDATION
# =============================================================================

def run_discovery(
        registry: TopicRegistry,
        centroid_store: CentroidStore,
        noise_buffer: NoiseBuffer,
) -> dict:
    """Processes isolates via spatial topic clustering models and updates registry entries."""
    log.info("─" * 65)
    log.info(f"DEPLOYING SPATIAL DENSITY EXTRACTION ENGINES ON UNALLOCATED NOISE POOL")
    log.info("─" * 65)

    review_ids, texts, embeddings = noise_buffer.get_unpromoted()

    if len(texts) < MIN_REVIEWS_FOR_DISCOVERY:
        log.warning(
            f"Buffer depth level ({len(texts):,}) below gating constraints ({MIN_REVIEWS_FOR_DISCOVERY:,}). Aborting.")
        return {}

    # Read layout directories to dynamically formulate the execution tracking build
    existing_versions = [
        d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d)) and d.startswith("v")
    ] if os.path.exists(MODELS_DIR) else []
    next_version = f"v{len(existing_versions) + 2}"
    log.info(f"Target model version lineage signature: {next_version}")

    engine = DiscoveryEngine(
        model_version=next_version, models_dir=MODELS_DIR, min_cluster_size=30, min_samples=10, max_topics_per_run=10
    )
    candidates = engine.run(texts=texts, embeddings=embeddings)

    text_to_id = dict(zip(texts, review_ids))
    for tid, data in candidates.items():
        data["review_ids"] = [text_to_id.get(t, "") for t in data["texts"]]

    existing_matrix, _ = centroid_store.get_all_centroids_as_matrix()
    existing_matrix_arg = existing_matrix if existing_matrix.size > 0 else None

    validator = TopicValidator(config=ValidationConfig())
    validations = validator.batch_validate(
        candidate_clusters={tid: d["embeddings"] for tid, d in candidates.items()},
        buffer_total_size=len(texts),
        existing_centroid_matrix=existing_matrix_arg,
    )

    llm_namer = LLMNamer()
    promoter = TopicPromoter(
        registry=registry, centroid_store=centroid_store, noise_buffer=noise_buffer,
        llm_namer=llm_namer, discovery_runs_path=DISCOVERY_RUNS_PATH,
        model_version=next_version, require_human_approval=REQUIRE_HUMAN_APPROVAL,
    )

    model_path = engine.save()
    run_record = promoter.promote_validated_clusters(
        candidates=candidates, validation_results=validations, bertopic_model_path=model_path,
    )

    return run_record


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """Evaluates pool size targets and conditionally executes cluster promotion fits."""
    log.info(f"Executing Stage 7b Emerging Topic Discovery with version token: {run_version}")

    registry = TopicRegistry(REGISTRY_PATH)
    centroid_store = CentroidStore(CENTROID_STORE_PATH)
    noise_buffer = NoiseBuffer(NOISE_BUFFER_PATH)

    buffer_size = noise_buffer.unpromoted_size()

    if buffer_size >= MIN_REVIEWS_FOR_DISCOVERY:
        log.info(
            f"Noise buffer depth ({buffer_size:,}) >= threshold ({MIN_REVIEWS_FOR_DISCOVERY:,}). Launching discovery fit.")

        run_record = run_discovery(
            registry=registry, centroid_store=centroid_store, noise_buffer=noise_buffer
        )

        if run_record:
            promoted_count = run_record.get("clusters_promoted", 0)
            rejected_count = run_record.get("clusters_rejected", 0)
            new_ids = run_record.get("new_topic_ids_assigned", [])

            mlflow.log_metric("raw_clusters_found", run_record.get("raw_clusters_found", 0))
            mlflow.log_metric("clusters_promoted", promoted_count)
            mlflow.log_metric("clusters_rejected", rejected_count)
            mlflow.log_metric("total_active_topics_post", registry.active_topic_count())
            mlflow.log_param("new_topic_ids_registered", str(new_ids))
            mlflow.log_param("discovery_model_version", run_record.get("model_version", ""))

            mlflow.log_artifact(DISCOVERY_RUNS_PATH, artifact_path="discovery_audit")
            mlflow.log_artifact(REGISTRY_PATH, artifact_path="updated_registry")
            mlflow.log_artifact(CENTROID_STORE_PATH, artifact_path="updated_centroids")

            log.info(
                f"Stage 7b conversion routine complete. Published {promoted_count} new themes to the runtime fleet.")
    else:
        log.info(
            f"Noise pool context size ({buffer_size:,} reviews) below operational limits. Discovery execution skipped for this block cycle.")
