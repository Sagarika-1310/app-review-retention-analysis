"""
stage_3_data_processor.py
-------------------
Transforms raw scraped Spotify Play Store reviews into clean, feature-engineered
datasets. Tracked natively inside the centralized MLflow lifecycle workspace.

Exposes:
- run(run_version): Clean decoupled execution layer hooked to external MLflow contexts.
"""

import os
import re
import ssl
import logging
import pandas as pd
import numpy as np
import emoji
import nltk

from datetime import datetime
from langdetect import detect, LangDetectException
import mlflow

# ── NLTK SSL fix (macOS) + stopwords ─────────────────────────────────────────
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

if len(STOPWORDS) < 100:
    raise RuntimeError("NLTK stopwords failed to load correctly.")

# ── Logging ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_3_DATA_PREPROCESSOR")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "spotify_reviews_raw.parquet")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ENGLISH_PATH = os.path.join(PROCESSED_DIR, "reviews_english.parquet")
PREMIUM_SCORING_PATH = os.path.join(PROCESSED_DIR, "reviews_premium.parquet")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DECAY_LAMBDA = 0.01
MIN_REVIEW_LENGTH = 15
QUALITY_WEIGHT_WORD_THRESHOLD = 100
PROXIMITY_WINDOW = 8
SCORING_WINDOW_DAYS = 180


# =============================================================================
# HELPER CONTEXT FUNCTIONS (UNALTERED CORE ENGINES)
# =============================================================================
def tokenize_no_stops(text: str) -> list:
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def words_within_window(tokens: list, word_a: str, word_b: str, window: int = PROXIMITY_WINDOW) -> bool:
    pos_a = [i for i, t in enumerate(tokens) if t.startswith(word_a)]
    pos_b = [i for i, t in enumerate(tokens) if t.startswith(word_b)]
    return any(abs(pa - pb) <= window for pa in pos_a for pb in pos_b)


def any_pair_matches(tokens: list, pairs: list, window: int = PROXIMITY_WINDOW) -> bool:
    return any(words_within_window(tokens, a, b, window) for a, b in pairs)


def load_raw(path: str) -> pd.DataFrame:
    log.info(f"Loading raw data from {path}")
    df = pd.read_parquet(path)
    return df


def detect_language(text: str) -> str:
    try:
        return detect(str(text))
    except LangDetectException:
        return "unknown"


def filter_english(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["content"].notnull()
    df.loc[mask, "detected_lang"] = df.loc[mask, "content"].apply(detect_language)
    df = df[df["detected_lang"] == "en"].copy()
    df.drop(columns=["detected_lang"], inplace=True)
    return df


def convert_emojis(text: str) -> str:
    return emoji.demojize(text, delimiters=(" ", " "))


def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = convert_emojis(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"version[:\s]+[\d\.]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"[^\w\s\'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"\badds?\b", "ads", text)
    return text


def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df["content_clean"] = df["content"].apply(clean_text)
    df = df[df["content_clean"].str.len() >= MIN_REVIEW_LENGTH].copy()
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["reviewId"]).copy()
    df = df.drop_duplicates(subset=["content_clean", "at"]).copy()
    return df


def tokenise_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df["tokens"] = df["content_clean"].apply(tokenize_no_stops)
    return df


# Pairs configurations arrays
PREMIUM_HIGH_PAIRS = [
    ("pay", "subscript"), ("pay", "premium"), ("paid", "premium"), ("paid", "subscript"),
    ("charg", "subscript"), ("charg", "premium"), ("bill", "subscript"), ("bill", "premium"),
    ("price", "premium"), ("cost", "premium"), ("use", "premium"), ("premium", "member"),
    ("cancel", "subscript"), ("cancel", "premium"), ("cancel", "plan"), ("cancel", "today"),
    ("downgrad", "free"), ("back", "free"), ("switch", "free"), ("revert", "free"),
    ("famil", "plan"), ("student", "plan"), ("annual", "plan"), ("month", "subscript"),
    ("individu", "plan"), ("refund", "subscript"), ("refund", "premium"), ("refund", "charg"),
    ("request", "refund"), ("switch", "appl"), ("mov", "appl"), ("switch", "youtub"),
    ("mov", "youtub"), ("mov", "tidal"), ("mov", "deezer"), ("leav", "spotify"),
    ("premium", "user"), ("premium", "account"), ("premium", "plan"), ("subscrib", "spotify"),
    ("upgrad", "premium"), ("worth", "subscript"), ("worth", "premium"), ("worth", "pay"),
    ("renew", "subscript"), ("auto", "renew"), ("loyal", "subscrib"), ("year", "premium"),
    ("month", "premium")
]

EXCLUSION_PAIRS = [
    ("free", "trial"), ("free", "version"), ("free", "tier"), ("free", "mode"), ("free", "use"), ("free", "plan"),
    ("free", "service"), ("free", "subscription"), ("advert", "pop"), ("watch", "vid"), ("unless", "pay"),
    ("without", "pay"), ("forc", "pay"), ("forc", "premium"), ("forc", "subscrip"), ("forc", "buy"),
    ("pressur", "pay"), ("pressur", "buy"), ("pressur", "premium"), ("song", "choice"),
    ("without", "premium"), ("unless", "buy"), ("everything", "premium"), ("everything", "pay"), ("song", "choos"),
    ("without", "buy"), ("many", "ads"), ("more", "ads"), ("lot", "ads"), ("greed", ""), ("require", "premium"),
    ("ad", "watch"), ("ads", "pop"), ("ad", "pop"), ("ad", "30"), ("ads", "song"),
    ("skip", "song"), ("skip", "hour"), ("only", "premium"),
    ("non-premium", ""), ("non", "premium"), ("lyrics", "limit"), ("basic", "feature"), ("limit", "song"),
    ("limit", "skip"), ("can't", "repeat"), ("explore premium", "")
]


def classify_premium(tokens: list) -> str:
    if any_pair_matches(tokens, PREMIUM_HIGH_PAIRS): return "high"
    return "none"


def filter_premium(df: pd.DataFrame) -> pd.DataFrame:
    df["premium_signal"] = df["tokens"].apply(classify_premium)
    return df[df["premium_signal"] != "none"].copy()


def filter_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    exclude_mask = df["tokens"].apply(lambda t: any_pair_matches(t, EXCLUSION_PAIRS))
    return df[~exclude_mask].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    reference_date = datetime.now()
    df["at"] = pd.to_datetime(df["at"], utc=True).dt.tz_localize(None)
    df["days_old"] = (reference_date - df["at"]).dt.days.clip(lower=0)
    df["decay_weight"] = np.exp(-DECAY_LAMBDA * df["days_old"])
    df["thumbs_normalised"] = df["thumbsUpCount"] / (df["days_old"] + 1)
    df["word_count"] = df["content_clean"].apply(lambda x: len(x.split()))
    df["quality_weight"] = df["word_count"].apply(lambda w: min(1.0, w / QUALITY_WEIGHT_WORD_THRESHOLD))

    severity_map = {1: 1.0, 2: 0.6, 3: 0.2, 4: 0.05, 5: 0.0}
    df["rating_severity"] = df["score"].map(severity_map)
    df["has_developer_response"] = df["repliedAt"].notnull()
    df["review_month"] = df["at"].dt.to_period("M").astype(str)
    df["review_quarter"] = df["at"].dt.to_period("Q").astype(str)
    df["app_version_clean"] = df["appVersion"].astype(str).apply(
        lambda v: ".".join(v.split(".")[:3]) if v not in ("nan", "", None) else "unknown"
    )
    return df


def label_recency_tier(days_old: int) -> str:
    if days_old <= 30:
        return "last_30d"
    elif days_old <= 90:
        return "last_90d"
    elif days_old <= 180:
        return "last_180d"
    return "historical"


def apply_recency_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df["recency_tier"] = df["days_old"].apply(label_recency_tier)
    df["in_scoring_window"] = df["days_old"] <= SCORING_WINDOW_DAYS
    return df


CANCELLATION_INTENT_PAIRS = [
    ("cancel", "subscript"), ("cancel", "premium"), ("cancel", "plan"), ("cancel", "today"),
    ("switch", "appl"), ("switch", "youtub"), ("move", "tidal"), ("move", "deezer"),
    ("downgrad", "free"), ("back", "free"), ("switch", "free"), ("revert", "free"),
    ("delet", "app"), ("request", "refund"), ("ask", "refund"), ("got", "refund"),
    ("not", "renew"), ("never", "renew"), ("leav", "spotify"), ("done", "spotify"),
    ("quit", "spotify"), ("unsubscrib", "spotify"), ("revok", "subscript"), ("revok", "premium"),
    ("revok", "plan"), ("think", "cancel"), ("consider", "cancel"), ("might", "cancel"),
    ("may", "cancel"), ("about", "cancel"), ("tempt", "cancel"), ("going", "cancel"),
    ("will", "cancel"), ("not worth", "pay"), ("not worth", "subscript"), ("not worth", "premium"),
    ("last", "chanc"), ("look", "alternativ"), ("look", "els"), ("tire", "issu"), ("fed", "issu")
]


def label_cancellation_intent(row) -> float:
    if any_pair_matches(row["tokens"], CANCELLATION_INTENT_PAIRS): return 1.0
    return 0.0


def apply_cancellation_intent(df: pd.DataFrame) -> pd.DataFrame:
    df["cancellation_intent_score"] = df.apply(label_cancellation_intent, axis=1)
    return df


# =============================================================================
# CLEAN DECOUPLED STAGE ENTRYPOINT
# =============================================================================
def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Executes transformation logic and logs data directly to the active nested child run.
    """
    log.info(f"Executing Stage 3 Data Processing with shared pipeline version identifier: {run_version}")

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing absolute upstream extraction source file: {RAW_PATH}")

    # ── Log Hyperparameters & Filtering Boundaries ────────────────────────────
    mlflow.log_param("pipeline_stage_version", run_version)
    mlflow.log_param("min_review_char_length", MIN_REVIEW_LENGTH)
    mlflow.log_param("proximity_window_tokens", PROXIMITY_WINDOW)
    mlflow.log_param("recency_scoring_window_days", SCORING_WINDOW_DAYS)
    mlflow.log_param("quality_word_count_threshold", QUALITY_WEIGHT_WORD_THRESHOLD)

    # ── Feature Engineering Pipeline Run ──────────────────────────────────────
    df_raw = load_raw(RAW_PATH)
    df_raw = df_raw[df_raw["is_new"].isin([True, "true"])]
    mlflow.log_metric("raw_records_count", len(df_raw))
    log.info(f"📊 Raw records count: {len(df_raw)}")

    df_eng = filter_english(df_raw)
    mlflow.log_metric("english_records_count", len(df_eng))
    log.info(f"📊 English records count: {len(df_eng)}")

    df_eng = apply_text_cleaning(df_eng)
    df_eng = deduplicate(df_eng)
    mlflow.log_metric("cleaned_deduped_count", len(df_eng))
    log.info(f"📊 Cleaned & deduped count: {len(df_eng)}")

    df_eng = tokenise_corpus(df_eng)

    # Persist intermediate translation layer
    df_eng.drop(columns=["tokens"]).to_parquet(ENGLISH_PATH, index=False)
    log.info(f"Intermediate English dataset saved successfully to {ENGLISH_PATH}")

    df_premium = filter_premium(df_eng)
    mlflow.log_metric("premium_flagged_count", len(df_premium))
    log.info(f"📊 Premium flagged count: {len(df_premium)}")

    df_premium = filter_exclusions(df_premium)
    mlflow.log_metric("premium_after_exclusions_count", len(df_premium))
    log.info(f"📊 Premium after exclusions count: {len(df_premium)}")

    df_premium = engineer_features(df_premium)
    df_premium = apply_recency_tiers(df_premium)
    df_premium = apply_cancellation_intent(df_premium)

    # Segment target rows matching the operational sliding scoring criteria
    df_scoring = df_premium[df_premium["in_scoring_window"] == True].copy()
    mlflow.log_metric("final_scoring_target_count", len(df_scoring))
    log.info(f"📊 Final scoring target count: {len(df_scoring)}")

    # Extract behavioral intent metrics
    intent_positive = (df_premium["cancellation_intent_score"] == 1.0).sum()
    mlflow.log_metric("detected_cancellation_intent_count", int(intent_positive))
    log.info(f"📊 Detected cancellation intent count: {int(intent_positive)}")

    # ── Final Persistence Cleanup & Registry Ingestion ───────────────────────
    df_premium_output = df_premium.drop(columns=["tokens"])
    df_premium_output.to_parquet(PREMIUM_SCORING_PATH, index=False)
    log.info(f"Saved Full Premium Scored dataset to {PREMIUM_SCORING_PATH}")

    # Register output matrices directly as run tracking artifacts
    mlflow.log_artifact(PREMIUM_SCORING_PATH, artifact_path="staged_datasets")
    log.info("Successfully uploaded processing artifacts to MLflow registry storage.")