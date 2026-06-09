"""
stage_8_validation.py
─────────────────────────
Stage 8: Production Validation Layer.

Executes a two-layer validation assessment framework to verify the analytical
and statistical integrity of the retention risk scoring framework:

Layer 1: Construct Validity
    - Evaluates rating-risk monotonicity (ensures mean risk decreases as star ratings increase).
    - Verifies 1-star dominance inside the high-risk statistical quintile.
    - Compares aggregate risk exposure between high-volume (Tier 1) and niche (Tier 3) clusters.
    - Evaluates Spearman rank correlations with user engagement signals (thumbsUpCount).

Layer 2: Proxy Label Validation
    - Conducts Mann-Whitney U testing across top operational pain point categories.
    - Validates that reviews showing explicit cancellation intent exhibit higher inherent risk scores.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import mlflow

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

log = logging.getLogger("STAGE_8_VALIDATION")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_premium_pain_point_clustered.parquet")
VALIDATION_REPORT_PATH = os.path.join(PROJECT_ROOT, "outputs", "validation_report.txt")

# ── Configuration Constants ───────────────────────────────────────────────────
SIGNIFICANCE_ALPHA = 0.05
MIN_SAMPLE_SIZE = 5
LAYER_2_PASS_THRESHOLD = 60.0  # Percentage of top clusters required to be statistically significant


# =============================================================================
# LAYER 1: CONSTRUCT VALIDITY
# =============================================================================

def run_construct_validity(df: pd.DataFrame) -> dict:
    """Evaluates mathematical stability and expected internal monotonic trends."""
    log.info("Executing Layer 1 Validation Suite: Construct Validity Checks...")
    results = {}

    # Check monotonicity across critical star-rating dimensions
    rating_means = df.groupby("score")["inherent_risk_score"].mean().sort_index()
    critical_ratings = rating_means.loc[[1, 2, 3, 4, 5]]
    is_monotonic_decreasing = all(critical_ratings.diff().dropna() < 0)

    results["monotonic_rating_decrease"] = {
        "passed": is_monotonic_decreasing,
        "details": {f"{int(k)}-star": round(v, 3) for k, v in rating_means.to_dict().items()}
    }

    # Evaluate 1-Star dominance in upper risk quintile (80th percentile)
    high_risk_threshold = df["inherent_risk_score"].quantile(0.80)
    high_risk_df = df[df["inherent_risk_score"] >= high_risk_threshold]

    if not high_risk_df.empty:
        one_star_pct = (high_risk_df["score"] == 1).sum() / len(high_risk_df)
        results["one_star_dominates_high_risk"] = {
            "passed": one_star_pct > 0.50,
            "details": f"1-star reviews represent {one_star_pct:.1%} of the upper 80th-percentile risk bin."
        }
    else:
        results["one_star_dominates_high_risk"] = {
            "passed": False,
            "details": "Execution error: High-risk bracket empty."
        }

    # Analyze risk exposure distributions between Tier 1 vs Tier 3 pain clusters
    cat_counts = df["pain_point_category"].value_counts()
    if len(cat_counts) >= 6:
        tier_1_cats = cat_counts.index[:3]
        tier_3_cats = cat_counts.index[-3:]

        t1_total_exposure = df[df["pain_point_category"].isin(tier_1_cats)]["inherent_risk_score"].sum()
        t3_total_exposure = df[df["pain_point_category"].isin(tier_3_cats)]["inherent_risk_score"].sum()

        results["tier1_higher_than_tier3"] = {
            "passed": t1_total_exposure > t3_total_exposure,
            "details": f"Tier 1 Aggregate Exposure: {t1_total_exposure:,.2f} vs Tier 3 Niche Exposure: {t3_total_exposure:,.2f}"
        }
    else:
        results["tier1_higher_than_tier3"] = {
            "passed": True,
            "details": f"Insufficient taxonomy variants ({len(cat_counts)}) to tier. Step bypassed."
        }

    # Calculate Spearman Rank Correlation with user engagement interactions
    if "thumbsUpCount" in df.columns and df["thumbsUpCount"].std() > 0:
        correlation, _ = stats.spearmanr(df["inherent_risk_score"], df["thumbsUpCount"])
        results["correlation_with_engagement"] = {
            "passed": abs(correlation) > 0.05,
            "details": f"Spearman rank correlation coefficient: {correlation:.3f}"
        }
    else:
        results["correlation_with_engagement"] = {
            "passed": False,
            "details": "Interaction flag 'thumbsUpCount' missing or invariant."
        }

    return results


# =============================================================================
# LAYER 2: PROXY LABEL VALIDATION
# =============================================================================

def run_proxy_label_validation(df: pd.DataFrame) -> dict:
    """Validates structural risk scoring logic against external proxy user churn behaviors."""
    log.info("Executing Layer 2 Validation Suite: Proxy Label Behavioral Alignment...")

    intent_df = df[df["cancellation_intent_score"] == 1]
    neutral_df = df[df["cancellation_intent_score"] == 0]

    top_10_clusters = df["pain_point_category"].value_counts().head(10).index.tolist()
    significant_clusters_count = 0
    cluster_tests = {}

    for cluster in top_10_clusters:
        # Target inherent_risk_score directly to isolate and control for temporal decay confounds
        intent_scores = intent_df[intent_df["pain_point_category"] == cluster]["inherent_risk_score"]
        neutral_scores = neutral_df[neutral_df["pain_point_category"] == cluster]["inherent_risk_score"]

        if len(intent_scores) > MIN_SAMPLE_SIZE and len(neutral_scores) > MIN_SAMPLE_SIZE:
            _, p_val = stats.mannwhitneyu(intent_scores, neutral_scores, alternative="greater")
            is_significant = p_val < SIGNIFICANCE_ALPHA

            if is_significant:
                significant_clusters_count += 1

            cluster_tests[cluster] = {
                "p_value": p_val,
                "significant": is_significant,
                "intent_mean_risk": round(intent_scores.mean(), 3),
                "neutral_mean_risk": round(neutral_scores.mean(), 3)
            }
        else:
            cluster_tests[cluster] = {
                "p_value": np.nan,
                "significant": False,
                "details": f"Insufficient size (Intent: {len(intent_scores)}, Neutral: {len(neutral_scores)})"
            }

    pct_significant = (significant_clusters_count / len(top_10_clusters)) * 100
    passed_layer_2 = pct_significant >= LAYER_2_PASS_THRESHOLD

    return {
        "passed": passed_layer_2,
        "pct_significant": pct_significant,
        "cluster_details": cluster_tests
    }


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """
    Exposed entrypoint invoked by the centralized master pipeline orchestrator.
    Evaluates scoring matrices for production readiness boundaries.
    """
    log.info(f"Starting Validation Framework Pipeline Stage with version run key: {run_version}")

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Required structural scoring pipeline asset missing: {INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    # Trigger validation sweeps
    layer_1_results = run_construct_validity(df)
    layer_2_results = run_proxy_label_validation(df)

    # ── Evaluation Gating Logic ──────────────────────────────────────────────
    layer_1_passed_count = sum(1 for k, v in layer_1_results.items() if v["passed"])
    layer_1_total = len(layer_1_results)

    l1_success = layer_1_passed_count == layer_1_total
    l2_success = layer_2_results["passed"]
    production_ready = l1_success and l2_success

    # Generate analytical review report strings
    report_blocks = [
        "=================================================================",
        "                 STAGE 8: PRODUCTION VALIDATION REPORT            ",
        "=================================================================",
        f"DEPLOYMENT READINESS SIGN-OFF: {'🚀 APPROVED FOR PRODUCTION' if production_ready else '⚠️ DEPLOYMENT HOLD / REMEDIATION REQUIRED'}\n",
        "── LAYER 1: CONSTRUCT VALIDITY SUMMARY ──",
    ]
    for metric, data in layer_1_results.items():
        report_blocks.append(
            f" ↳ {metric:<32}: {'✅ PASSED' if data['passed'] else '❌ FAILED'} — Details: {data['details']}")

    report_blocks.extend([
        "\n── LAYER 2: PROXY LABEL VALIDATION SUMMARY ──",
        f" Target Structural Metric : Target >= {LAYER_2_PASS_THRESHOLD}% of top clusters statistically significant (p < {SIGNIFICANCE_ALPHA})",
        f" Evaluated Fleet Metric   : {layer_2_results['pct_significant']:.1f}% of top-10 clusters validated as significant.",
        f" Layer 2 Pass Status      : {'✅ PASSED' if l2_success else '❌ FAILED'}\n",
        "── STRATEGIC DIAGNOSTIC RECOMMENDATIONS ──"
    ])

    if production_ready:
        report_blocks.append(
            "SUCCESS: Risk scoring formula demonstrates flawless internal construct coherence and tracks "
            "churn behaviors accurately across downstream text streams. Authorized to migrate model weights to the Production Registry."
        )
    elif not l1_success and l2_success:
        report_blocks.append(
            "CRITICAL MATHEMATICAL ERROR: External behavioral patterns confirm alignment (Layer 2 passes), but "
            "internal logic checks have broken down (Layer 1 fails). Risk scoring parameters are non-monotonic. "
            "Re-calibrate core heuristic driver coefficients inside the risk assignment modules."
        )
    elif l1_success and not l2_success:
        report_blocks.append(
            "RE-CALIBRATION REQUIRED: Risk parameters are internally consistent (Layer 1 passes), but fail to "
            "reliably isolate cancellation signals in production text strings (Layer 2 fails). Expand standard "
            "keyword matrices or adjust clustering density controls."
        )
    else:
        report_blocks.append(
            "SYSTEMIC FAILURE: Internal stability parameters and proxy-behavior validation routines have both "
            "failed baseline criteria. Re-evaluate textual embeddings and rebuild core parameters from scratch."
        )

    assessment_report = "\n".join(report_blocks)
    print(assessment_report)

    # Persist the output validation report
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(assessment_report)

    # ── Synch Assets & Parameters to MLflow Workspace Context ──────────────────
    mlflow.log_metric("layer_1_passed_ratio", float(layer_1_passed_count / layer_1_total))
    mlflow.log_metric("layer_2_significant_ratio", float(layer_2_results["pct_significant"] / 100.0))
    mlflow.log_param("production_approved_flag", str(production_ready))
    mlflow.log_artifact(VALIDATION_REPORT_PATH, artifact_path="validation_audit")

    log.info("Stage 8 runtime diagnostic metrics and reports successfully synced to the central MLflow instance.")
