"""
src/download_models.py
----------------------
One-time utility script to download all models used in the project
and cache them locally for offline production workflows.

Models downloaded:
    1. RoBERTa sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)
    2. Sentence Transformer - paraphrase (paraphrase-MiniLM-L12-v2)
    3. DistilBERT sentiment model (distilbert-base-uncased-finetuned-sst-2-english)
    4. Sentence Transformer - MiniLM (all-MiniLM-L6-v2)
    5. Sentence Transformer - MPNet (all-mpnet-base-v2)
    6. BART MNLI for zero-shot classification (facebook/bart-large-mnli)
"""

import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model directories
LOCAL_ROBERTA_DIR = os.path.join(PROJECT_ROOT, "models", "roberta")
LOCAL_DISTILBERT_DIR = os.path.join(PROJECT_ROOT, "models", "distilbert-base-uncased-finetuned-sst-2-english")
LOCAL_MINILM_DIR = os.path.join(PROJECT_ROOT, "models", "all-MiniLM-L6-v2")
LOCAL_MPNET_DIR = os.path.join(PROJECT_ROOT, "models", "all-mpnet-base-v2")
LOCAL_BART_DIR = os.path.join(PROJECT_ROOT, "models", "facebook-bart-large-mnli")

# Remote model names
REMOTE_ROBERTA = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REMOTE_DISTILBERT = "distilbert-base-uncased-finetuned-sst-2-english"
REMOTE_MINILM = "all-MiniLM-L6-v2"
REMOTE_MPNET = "all-mpnet-base-v2"
REMOTE_BART = "facebook/bart-large-mnli"


def main():
    # ── 1. Download RoBERTa Assets ───────────────────────────────────────────
    log.info(f"Fetching assets for '{REMOTE_ROBERTA}' from Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(REMOTE_ROBERTA)
    model = AutoModelForSequenceClassification.from_pretrained(REMOTE_ROBERTA)

    log.info(f"Saving serialized RoBERTa files locally to: {LOCAL_ROBERTA_DIR}")
    os.makedirs(LOCAL_ROBERTA_DIR, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_ROBERTA_DIR)
    model.save_pretrained(LOCAL_ROBERTA_DIR)
    log.info("✓ RoBERTa model downloaded successfully")


    # ── 3. Download DistilBERT Assets ─────────────────────────────────────────
    log.info(f"Fetching assets for '{REMOTE_DISTILBERT}' from Hugging Face Hub...")
    distilbert_tokenizer = AutoTokenizer.from_pretrained(REMOTE_DISTILBERT)
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(REMOTE_DISTILBERT)

    log.info(f"Saving serialized DistilBERT files locally to: {LOCAL_DISTILBERT_DIR}")
    os.makedirs(LOCAL_DISTILBERT_DIR, exist_ok=True)
    distilbert_tokenizer.save_pretrained(LOCAL_DISTILBERT_DIR)
    distilbert_model.save_pretrained(LOCAL_DISTILBERT_DIR)
    log.info("✓ DistilBERT model downloaded successfully")

    # ── 4. Download Sentence-Transformer (MiniLM) Assets ─────────────────────
    log.info(f"Fetching embedding model '{REMOTE_MINILM}'...")
    minilm_model = SentenceTransformer(REMOTE_MINILM)

    log.info(f"Saving serialized embedding model locally to: {LOCAL_MINILM_DIR}")
    os.makedirs(LOCAL_MINILM_DIR, exist_ok=True)
    minilm_model.save(LOCAL_MINILM_DIR)
    log.info("✓ all-MiniLM-L6-v2 model downloaded successfully")

    # ── 5. Download Sentence-Transformer (MPNet) Assets ─────────────────────
    log.info(f"Fetching embedding model '{REMOTE_MPNET}'...")
    mpnet_model = SentenceTransformer(REMOTE_MPNET)

    log.info(f"Saving serialized embedding model locally to: {LOCAL_MPNET_DIR}")
    os.makedirs(LOCAL_MPNET_DIR, exist_ok=True)
    mpnet_model.save(LOCAL_MPNET_DIR)
    log.info("✓ all-mpnet-base-v2 model downloaded successfully")

    # ── 6. Download BART MNLI Assets ─────────────────────────────────────────
    log.info(f"Fetching assets for '{REMOTE_BART}' from Hugging Face Hub...")
    bart_tokenizer = AutoTokenizer.from_pretrained(REMOTE_BART)
    bart_model = AutoModelForSequenceClassification.from_pretrained(REMOTE_BART)

    log.info(f"Saving serialized BART MNLI files locally to: {LOCAL_BART_DIR}")
    os.makedirs(LOCAL_BART_DIR, exist_ok=True)
    bart_tokenizer.save_pretrained(LOCAL_BART_DIR)
    bart_model.save_pretrained(LOCAL_BART_DIR)
    log.info("✓ facebook/bart-large-mnli model downloaded successfully")

    log.info("🎉 All model snapshots successfully cached locally for offline execution!")
    log.info("\nDownloaded models:")
    log.info(f"  - RoBERTa: {LOCAL_ROBERTA_DIR}")
    log.info(f"  - DistilBERT: {LOCAL_DISTILBERT_DIR}")
    log.info(f"  - all-MiniLM-L6-v2: {LOCAL_MINILM_DIR}")
    log.info(f"  - all-mpnet-base-v2: {LOCAL_MPNET_DIR}")
    log.info(f"  - facebook/bart-large-mnli: {LOCAL_BART_DIR}")


if __name__ == "__main__":
    main()