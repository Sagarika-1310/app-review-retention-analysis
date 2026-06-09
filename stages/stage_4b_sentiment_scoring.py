"""
stage_4b_sentiment_scoring.py
─────────────────────────────────
Queries historical validation profiles using Unified Model Performance Index (UMPI),
initializes ONLY the optimal winning engine, and builds the scored production corpus.

Exposes:
- run(run_version): Entrypoint invoked by the centralized orchestrator.
"""
import os
import time
import logging
import warnings
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import mlflow

warnings.filterwarnings("ignore")

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_4B_SENTIMENT_SCORING")

# ── Paths ─────────────────────────────────────────────────────────────────────
LOCAL_ROBERTA_DIR = os.path.join(PROJECT_ROOT, "models", "roberta")
LOCAL_DISTILBERT_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert-base-uncased-finetuned-sst-2-english")
PREMIUM_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium.parquet")
SENTIMENT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_sentiment_scored.parquet")

# ── Global Configuration & Device Routing ──────────────────────────────────────
ROBERTA_MODEL_NAME = LOCAL_ROBERTA_DIR
BATCH_SIZE = 32
DISTILBERT_CONFIDENCE_THRESHOLD = 0.75

if torch.backends.mps.is_available():
    device_str = "mps"
elif torch.cuda.is_available():
    device_str = "cuda"
else:
    device_str = "cpu"

device = torch.device(device_str)

# Dynamic singular engine placeholders
sia = None
distilbert_pl = None
roberta_tokenizer = None
roberta_model = None


# =============================================================================
# CHOSEN TARGETED INITIALIZATION STRATEGY
# =============================================================================

def initialize_winning_engine(engine_name: str):
    """Saves severe computational resources by only spinning up the selected production winner."""
    global sia, distilbert_pl, roberta_tokenizer, roberta_model

    if engine_name == "VADER":
        log.info("Targeted Spinup: Initializing VADER Sentiment Heuristic...")
        sia = SentimentIntensityAnalyzer()

    elif engine_name == "DistilBERT":
        log.info("Targeted Spinup: Initializing DistilBERT Neural Network...")
        device_id = 0 if device_str in ["cuda", "mps"] else -1
        distilbert_pl = pipeline(
            "sentiment-analysis",
            model=LOCAL_DISTILBERT_DIR,
            device=device_id,
            truncation=True,
            max_length=512,
        )

    elif engine_name == "RoBERTa":
        log.info(f"Targeted Spinup: Loading RoBERTa Weights ({ROBERTA_MODEL_NAME}) onto {device_str.upper()}...")
        roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
        roberta_model.to(device)
        roberta_model.eval()


# =============================================================================
# WINNING PRODUCTION SELECTION ALGORITHM
# =============================================================================

def discover_best_model_engine(experiment_name: str) -> tuple:
    """Selects the highest performing sentiment engine using UMPI score weights."""
    current_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not current_experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs_df = mlflow.search_runs(
        experiment_ids=[current_experiment.experiment_id],
        filter_string="tags.mlflow.runName LIKE 'Engine_%'"
    )

    if runs_df.empty:
        raise RuntimeError("No sentiment engine evaluation runs found in tracking history.")

    f1_col = "metrics.f1_score_macro"
    auc_col = "metrics.precision_recall_auc"
    kappa_col = "metrics.cohen_kappa"

    runs_df[f1_col] = pd.to_numeric(runs_df[f1_col]).fillna(0.0)
    runs_df[auc_col] = pd.to_numeric(runs_df[auc_col]).fillna(0.0)
    runs_df[kappa_col] = pd.to_numeric(runs_df[kappa_col]).fillna(0.0)

    runs_df["umpi_score"] = (
            0.40 * runs_df[f1_col]
            + 0.35 * runs_df[auc_col]
            + 0.25 * runs_df[kappa_col]
    )

    best_run = runs_df.sort_values(by="umpi_score", ascending=False).iloc[0]
    best_engine = best_run["params.engine_type"]
    best_mode = best_run["params.evaluation_mode"]

    log.info(f"🏆 Production Winner Discovered: {best_engine} ({best_mode.upper()}) | UMPI={best_run['umpi_score']:.4f}")
    return best_engine, best_mode


# =============================================================================
# INFERENCE PIPELINE CALLS
# =============================================================================

def run_vader_production(texts: list, mode: str) -> list:
    preds = []
    for text in texts:
        score = sia.polarity_scores(str(text))["compound"]
        if mode == "binary":
            preds.append(1 if score >= 0 else 0)
        else:
            preds.append(0 if score <= -0.05 else (2 if score >= 0.05 else 1))
    return preds


def run_distilbert_production(texts: list, mode: str) -> list:
    results = distilbert_pl(texts, truncation=True, max_length=512)
    preds = []
    for result in results:
        label = result["label"]
        score = result["score"]
        if mode == "binary":
            preds.append(0 if label == "NEGATIVE" else 1)
        else:
            if score < DISTILBERT_CONFIDENCE_THRESHOLD:
                preds.append(1)
            else:
                preds.append(0 if label == "NEGATIVE" else 2)
    return preds


import math


def run_roberta_production(
        texts: list,
        mode: str,
        batch_size: int = BATCH_SIZE,
) -> list:
    preds = []

    total_records = len(texts)
    total_batches = math.ceil(total_records / batch_size)

    log.info(
        f"Starting RoBERTa inference | "
        f"records={total_records:,} | "
        f"batch_size={batch_size:,} | "
        f"total_batches={total_batches:,}"
    )

    for batch_num, i in enumerate(
            range(0, total_records, batch_size),
            start=1,
    ):

        batch_end = min(i + batch_size, total_records)

        progress_pct = (batch_num / total_batches) * 100

        log.info(
            f"[Batch {batch_num}/{total_batches}] "
            f"Processing records "
            f"{i + 1:,}-{batch_end:,} "
            f"({progress_pct:.1f}% complete)"
        )

        batch = [
            str(x)
            for x in texts[i:batch_end]
        ]

        encoded = roberta_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():

            outputs = roberta_model(**encoded)

            probs = F.softmax(
                outputs.logits,
                dim=1,
            ).cpu().numpy()

        for p in probs:
            if mode == "binary":
                neg_prob = p[0] + 0.5 * p[1]
                pos_prob = p[2] + 0.5 * p[1]

                preds.append(
                    0 if neg_prob > pos_prob else 1
                )
            else:
                preds.append(
                    int(np.argmax(p))
                )

    log.info(
        f"RoBERTa inference completed successfully | "
        f"predictions_generated={len(preds):,}"
    )

    return preds


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Executes large-scale sentiment generation using the optimized winner configuration.
    """
    log.info(f"Executing Stage 4b Production Sentiment Scoring with shared pipeline version identifier: {run_version}")

    # 1. Fetch current runtime context parameters
    active_ex_id = mlflow.active_run().info.experiment_id
    experiment_name = mlflow.get_experiment(active_ex_id).name

    # 2. Select optimized pipeline configuration via historical parameters
    best_engine, best_mode = discover_best_model_engine(experiment_name)

    # 3. Targeted resource loading
    initialize_winning_engine(best_engine)

    # 4. Input Guardrails Check
    if not os.path.exists(PREMIUM_INPUT_PATH):
        raise FileNotFoundError(f"Missing upstream processed tracking asset: {PREMIUM_INPUT_PATH}")

    df_premium = pd.read_parquet(PREMIUM_INPUT_PATH)
    df_premium = df_premium[df_premium["is_new"].isin([True, "true"])]
    texts = df_premium["content_clean"].astype(str).tolist()

    log.info(f"Executing production run lifecycle: {best_engine} ({best_mode}) across {len(df_premium):,} records")
    start_production_time = time.time()

    if best_engine == "VADER":
        predictions = run_vader_production(texts, best_mode)
    elif best_engine == "RoBERTa":
        predictions = run_roberta_production(texts, best_mode)
    else:
        predictions = run_distilbert_production(texts, best_mode)

    production_runtime = time.time() - start_production_time

    # 5. Export and Save Final Stage Output Assets
    df_premium["predicted_sentiment"] = predictions
    df_premium.to_parquet(SENTIMENT_OUTPUT_PATH, index=False)

    log.info(f"Production sentiment inference completed in {production_runtime:.1f}s")
    log.info(f"Saved production sentiment dataset to {SENTIMENT_OUTPUT_PATH}")

    # 6. Log parameters and tracking artifacts inside active orchestrator run
    mlflow.log_param("selected_production_engine", best_engine)
    mlflow.log_param("selected_production_mode", best_mode)
    mlflow.log_metric("production_inference_seconds", production_runtime)
    mlflow.log_metric("production_scored_records_count", len(df_premium))

    mlflow.log_artifact(SENTIMENT_OUTPUT_PATH, artifact_path="production_sentiment_dataset")
    log.info("Successfully registered production inference dataset cleanly with active MLflow lifecycle.")
