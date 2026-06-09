"""
stage_4a_sentiment_evaluation.py
────────────────────────────────────
Evaluates alternative sentiment engines (VADER, DistilBERT, RoBERTa) across
binary and multi-class configurations to establish the optimal framework.

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
import matplotlib.pyplot as plt

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report, ConfusionMatrixDisplay, \
    precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import mlflow

warnings.filterwarnings("ignore")

# ── Project Layout & Logging Configuration ─────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_4A_SENTIMENT_EVALUATION")

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "labelling", "sentiment_labelling_sample.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "labelling", "sentiment_scored_benchmark.csv")
IMG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOCAL_ROBERTA_DIR = os.path.join(PROJECT_ROOT, "models", "roberta")
LOCAL_DISTILBERT_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert-base-uncased-finetuned-sst-2-english")

# ── Global Configuration & Device Routing ──────────────────────────────────────
ROBERTA_MODEL_NAME = LOCAL_ROBERTA_DIR
DISTILBERT_MODEL_NAME = LOCAL_DISTILBERT_DIR
BATCH_SIZE = 32
DISTILBERT_CONFIDENCE_THRESHOLD = 0.75

if torch.backends.mps.is_available():
    device_str = "mps"
elif torch.cuda.is_available():
    device_str = "cuda"
else:
    device_str = "cpu"

device = torch.device(device_str)
log.info(f"Neural engine infrastructure targeted device: {device_str.upper()}")

# Global engine placeholders to enable delayed/lazy loading
sia = None
distilbert_pl = None
roberta_tokenizer = None
roberta_model = None


# =============================================================================
# DELAYED ENGINE INITIALIZATION PHASE
# =============================================================================

def initialize_engines():
    """Initializes heavy machine learning engines safely inside the runtime execution context."""
    global sia, distilbert_pl, roberta_tokenizer, roberta_model

    if sia is not None:
        return  # Already initialized

    log.info("Initializing VADER Sentiment Analyzer...")
    sia = SentimentIntensityAnalyzer()

    log.info("Initializing DistilBERT Pipeline...")
    device_id = 0 if device_str in ["cuda", "mps"] else -1
    distilbert_pl = pipeline(
        "sentiment-analysis",
        model=DISTILBERT_MODEL_NAME,
        device=device_id,
        truncation=True,
        max_length=512,
    )

    log.info(f"Initializing RoBERTa Tokenizer and Model ({ROBERTA_MODEL_NAME})...")
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
    roberta_model.to(device)
    roberta_model.eval()


# =============================================================================
# PR AUC CALCULATION UTILITY
# =============================================================================

def calculate_pr_auc(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int) -> float:
    """Calculates PR AUC. Returns scalar for binary or macro-average for multiclass."""
    if y_probs is None:
        return np.nan
    try:
        if num_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
            return auc(recall, precision)
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
            pr_aucs = []
            for i in range(num_classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                pr_aucs.append(auc(recall, precision))
            return np.mean(pr_aucs)
    except Exception as e:
        log.warning(f"Failed to calculate PR AUC: {e}")
        return np.nan


# =============================================================================
# ENGINE PREDICTION LOGIC BLOCKS
# =============================================================================

def vader_predict_pipeline(texts: list, mode: str) -> tuple:
    """Runs VADER mapping to binary or multiclass arrays and creates synthetic probabilities."""
    preds = []
    probs = []
    for text in texts:
        score = sia.polarity_scores(text)["compound"]
        norm_score = (score + 1) / 2.0

        if mode == "binary":
            if score >= 0.0:
                preds.append(1)
                probs.append([1.0 - norm_score, norm_score])
            else:
                preds.append(0)
                probs.append([1.0 - norm_score, norm_score])
        else:
            if score <= -0.05:
                preds.append(0)
                probs.append([0.7, 0.2, 0.1])
            elif score >= 0.05:
                preds.append(2)
                probs.append([0.1, 0.2, 0.7])
            else:
                preds.append(1)
                probs.append([0.2, 0.6, 0.2])

    return preds, np.array(probs)


def distilbert_predict_pipeline(texts: list, mode: str) -> tuple:
    """Runs inference via DistilBERT adjusting mappings depending on experiment context."""
    preds = []
    probs = []
    results = distilbert_pl(texts, truncation=True, max_length=512)

    for result in results:
        label = result["label"]
        score = result["score"]

        if mode == "binary":
            if label == "NEGATIVE":
                preds.append(0)
                probs.append([score, 1.0 - score])
            else:
                preds.append(1)
                probs.append([1.0 - score, score])
        else:
            prob_dist = [0.0, 0.0, 0.0]
            if label == "NEGATIVE":
                prob_dist[0] = score
                prob_dist[2] = 1.0 - score
            else:
                prob_dist[2] = score
                prob_dist[0] = 1.0 - score

            if score < DISTILBERT_CONFIDENCE_THRESHOLD:
                pred_class = 1
                prob_dist = [0.2, 0.6, 0.2]
            else:
                pred_class = 0 if label == "NEGATIVE" else 2

            preds.append(pred_class)
            probs.append(prob_dist)

    return preds, np.array(probs)


def roberta_predict_pipeline(texts: list, mode: str, batch_size: int = BATCH_SIZE) -> tuple:
    """Runs batch inference on RoBERTa natively and handles transforms for downstream evaluation."""
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i: i + batch_size]]
        encoded = roberta_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
            device)

        with torch.no_grad():
            outputs = roberta_model(**encoded)
            softmax_probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

        for prob in softmax_probs:
            if mode == "binary":
                neg_prob = prob[0] + 0.5 * prob[1]
                pos_prob = prob[2] + 0.5 * prob[1]
                pred_class = 0 if neg_prob > pos_prob else 1
                all_preds.append(pred_class)
                all_probs.append([neg_prob, pos_prob])
            else:
                pred_class = np.argmax(prob)
                all_preds.append(pred_class)
                all_probs.append(prob)

    return all_preds, np.array(all_probs)


# =============================================================================
# EVALUATION METRICS PACK
# =============================================================================

def evaluate_model_performance(df: pd.DataFrame, pred_col: str, model_name: str, mode: str,
                               y_probs: np.ndarray = None) -> dict:
    """Calculates evaluation metrics, tracking validation arrays across distinct modes."""
    target_col = "true_label_binary" if mode == "binary" else "true_label_multi"
    target_names = ["Negative", "Positive"] if mode == "binary" else ["Negative", "Neutral", "Positive"]
    num_classes = 2 if mode == "binary" else 3

    y_true = df[target_col].astype(int).values
    y_pred = df[pred_col].astype(int).values

    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    pr_auc = calculate_pr_auc(y_true, y_probs, num_classes)

    print(f"\n{'=' * 60}\nCLASSIFICATION REPORT — {model_name} ({mode.upper()})\n{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    print(f"Weighted F1-Score : {f1_weighted:.4f}")
    print(f"Macro F1-Score    : {f1_macro:.4f}")
    print(f"Cohen's Kappa     : {kappa:.4f}")
    print(f"PR AUC            : {f'{pr_auc:.4f}' if not np.isnan(pr_auc) else 'N/A'}\n")

    # Generate and export Confusion Matrix Plot
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=target_names, cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name} ({mode.capitalize()})")

    img_name = f"confusion_matrix_{model_name.lower()}_{mode}.png"
    img_path = os.path.join(IMG_OUTPUT_DIR, img_name)
    plt.savefig(img_path, bbox_inches="tight")
    plt.close('all')
    log.info(f"Saved confusion matrix visualization for {model_name} ({mode}) to {img_path}")

    return {
        "model_name": model_name,
        "experiment_mode": mode,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "cohen_kappa": kappa,
        "pr_auc": pr_auc,
        "confusion_matrix_filename": img_name
    }


# =============================================================================
# MAIN ORCHESTRATION STAGE ENTRYPOINT
# =============================================================================

def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Evaluates alternative sentiment engines against validation thresholds.
    """
    log.info(f"Executing Stage 4a Sentiment Evaluation with shared pipeline version identifier: {run_version}")

    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

    # 1. Initialize models under active runtime
    initialize_engines()

    # 2. Input Guardrails Check
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Target verification sheet not found at {INPUT_PATH}.")

    df = pd.read_csv(INPUT_PATH)
    df = df[df["is_new"].isin([True, "true"])]
    log.info(f"Loaded validation sample matching {len(df):,} verified records")

    text_corpus_list = df["content"].astype(str).tolist()
    all_metrics = []

    engines_meta = [
        {"name": "VADER", "pipeline_func": vader_predict_pipeline, "extra_params": {}},
        {"name": "DistilBERT", "pipeline_func": distilbert_predict_pipeline,
         "extra_params": {"confidence_threshold": DISTILBERT_CONFIDENCE_THRESHOLD}},
        {"name": "RoBERTa", "pipeline_func": roberta_predict_pipeline,
         "extra_params": {"batch_size": BATCH_SIZE, "model_source": ROBERTA_MODEL_NAME}}
    ]

    mlflow.log_param("dataset_records_count", len(df))
    mlflow.log_param("neural_device_infrastructure", device_str.upper())

    # 3. BENCHMARK MODEL EVALUATIONS (Nested child runs for specific model variants)
    for mode in ["binary", "multi"]:
        log.info(f"\n⚡ PROCESSING EVALUATION MODE: {mode.upper()}")
        log.info("-" * 65)

        for engine in engines_meta:
            engine_name = engine["name"]
            log.info(f"Running inference: {engine_name} ({mode})...")

            # Deeply nested child run for variant segmentation (Stage 4a -> Engine Variant)
            with mlflow.start_run(run_name=f"Engine_{engine_name}_{mode}", nested=True):
                mlflow.log_param("engine_type", engine_name)
                mlflow.log_param("evaluation_mode", mode)
                for param_key, param_val in engine["extra_params"].items():
                    mlflow.log_param(param_key, param_val)

                start_time = time.time()
                if engine_name == "RoBERTa":
                    preds, probs = engine["pipeline_func"](text_corpus_list, mode=mode, batch_size=BATCH_SIZE)
                else:
                    preds, probs = engine["pipeline_func"](text_corpus_list, mode=mode)

                runtime = time.time() - start_time
                log.info(f"  Inference time: {runtime:.1f}s")

                pred_col_name = f"pred_{engine_name.lower()}_{mode}"
                df[pred_col_name] = preds

                # Evaluation Metrics Computations
                metrics = evaluate_model_performance(df, pred_col_name, engine_name, mode=mode, y_probs=probs)
                metrics["runtime"] = runtime
                all_metrics.append(metrics)

                mlflow.log_metric("runtime_seconds", metrics["runtime"])
                mlflow.log_metric("f1_score_weighted", metrics["f1_weighted"])
                mlflow.log_metric("f1_score_macro", metrics["f1_macro"])
                mlflow.log_metric("cohen_kappa", metrics["cohen_kappa"])
                if not np.isnan(metrics["pr_auc"]):
                    mlflow.log_metric("precision_recall_auc", metrics["pr_auc"])

                # Log variant confusion matrix charts directly
                local_img_file_path = os.path.join(IMG_OUTPUT_DIR, metrics["confusion_matrix_filename"])
                if os.path.exists(local_img_file_path):
                    mlflow.log_artifact(local_img_file_path, artifact_path=f"evaluation_plots/{mode}")

    # Track stage benchmark mapping arrays
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    mlflow.log_artifact(OUTPUT_PATH, artifact_path="staged_datasets")
    log.info("Saved and uploaded Stage 4a performance verification datasets cleanly to MLflow registry.")