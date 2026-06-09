"""
stage_5_risk_scoring.py
───────────────────────────
Evaluates an exponential Customer Retention Risk Score using text sentiment
penalties, structural risk drivers, engagement metrics, and developer mitigation markers.

Exposes:
- run(run_version): Entrypoint invoked by the centralized orchestrator.
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
import mlflow

warnings.filterwarnings("ignore")

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_5_RISK_SCORING")

# ── Paths ─────────────────────────────────────────────────────────────────────
SENTIMENT_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_sentiment_scored.parquet")
FINAL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_retention_scored.parquet")


# =============================================================================
# RETENTION RISK SCORING ALGORITHM
# =============================================================================

def calculate_retention_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes both Inherent Risk (for validation) and Decayed Retention Risk (for production).
    Prevents mathematical compression from extreme decay weight variations.
    """
    if df.empty:
        return df

    # 1. Map text sentiment categories to numerical penalties
    if "predicted_sentiment" in df.columns:
        sentiment_mode = "binary" if df["predicted_sentiment"].max() == 1 else "multi"
        if sentiment_mode == "binary":
            df["sentiment_penalty"] = df["predicted_sentiment"].map({0: 1.0, 1: 0.0})
        else:
            df["sentiment_penalty"] = df["predicted_sentiment"].map({0: 1.0, 1: 0.5, 2: 0.0})
    else:
        df["sentiment_penalty"] = 0.5

    # 2. Compute Structural Risk Drivers (The Core Threat)
    df["structural_risk"] = (
            (df["rating_severity"] * 0.45) +
            (df["cancellation_intent_score"] * 0.40) +
            (df["sentiment_penalty"] * 0.15)
    )

    # Intent Override: Prevent sarcastic or high-rating text churners from being underscored
    if "cancellation_intent_score" in df.columns:
        df["structural_risk"] = np.maximum(df["structural_risk"], df["cancellation_intent_score"] * 0.55)

    # 3. Compute Risk Amplifiers (Engagement & Visibility)
    df["risk_amplifier"] = (
            1.0 +
            (df["thumbs_normalised"].clip(upper=10) * 0.08) +
            (df["quality_weight"] * 0.05)
    )

    # 4. Calculate Combined Base Risk Index
    df["base_risk_index"] = df["structural_risk"] * df["risk_amplifier"]

    # 5. Apply Developer Mitigation Multiplier
    df["developer_mitigation"] = df["has_developer_response"].apply(lambda x: 0.85 if x is True else 1.0)
    df["mitigated_risk_index"] = df["base_risk_index"] * df["developer_mitigation"]

    # 6. Compute INHERENT RISK SCORE
    # Stretches features into a balanced [0, 1] range independent of review age
    df["inherent_risk_score"] = 1.0 - np.exp(-df["mitigated_risk_index"] * 2.5)

    # 7. Compute DECAYED RETENTION RISK SCORE (For downstream production UI use)
    # Uses a non-zero floor to stop historical reviews from completely flatlining to absolute 0
    decay_floor = np.maximum(df["decay_weight"], 0.05)
    df["retention_risk_score"] = 1.0 - np.exp(-df["mitigated_risk_index"] * 4.0 * decay_floor)

    # Guardrail clamps
    df["inherent_risk_score"] = df["inherent_risk_score"].clip(0.0, 1.0)
    df["retention_risk_score"] = df["retention_risk_score"].clip(0.0, 1.0)

    return df


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Processes finalized metrics profiles and outputs retention risk scoring segments.
    """
    log.info(f"Executing Stage 5 Risk Scoring with shared pipeline version identifier: {run_version}")

    # 1. Input Guardrails Check
    if not os.path.exists(SENTIMENT_INPUT_PATH):
        raise FileNotFoundError(f"Missing sentiment tracking asset: {SENTIMENT_INPUT_PATH}")

    df_sentiment = pd.read_parquet(SENTIMENT_INPUT_PATH)
    df_sentiment = df_sentiment[df_sentiment["is_new"].isin([True, "true"])]
    log.info(f"Loaded sentiment-scored dataset containing {len(df_sentiment):,} records.")

    # 2. Extract calculations inside centralized nested child tracking run
    log.info("Calculating comprehensive retention risk profiles...")
    df_final = calculate_retention_risk(df_sentiment)

    high_risk_threshold = 0.70
    high_risk_count = int((df_final["retention_risk_score"] >= high_risk_threshold).sum())
    mean_risk_score = float(df_final["retention_risk_score"].mean())

    log.info(f"🚨 High-Risk Users Detected: {high_risk_count:,}")
    log.info(f"📈 Mean Fleet Risk Score: {mean_risk_score:.4f}")

    # 3. Log Stage Performance Metrics to MLflow Context
    mlflow.log_metric("high_churn_risk_count", high_risk_count)
    mlflow.log_metric("fleet_mean_risk_score", mean_risk_score)
    mlflow.log_param("high_risk_threshold", high_risk_threshold)

    # 4. Save and Register Output Artifacts
    df_final.to_parquet(FINAL_OUTPUT_PATH, index=False)
    mlflow.log_artifact(FINAL_OUTPUT_PATH, artifact_path="final_risk_scores")

    log.info(f"Stage 5 completed successfully. Output saved to {FINAL_OUTPUT_PATH}")
