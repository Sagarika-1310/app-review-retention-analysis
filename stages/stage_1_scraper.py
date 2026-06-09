"""
stage_1_scraper.py
----------
Collects Spotify Play Store reviews with balanced coverage across:
- NEWEST reviews (Scraped incrementally through the last completed month)
- MOST_RELEVANT reviews
- Multiple countries

Exposes:
- run(run_version): Clean decoupled execution layer hooked to external MLflow contexts.
"""

import os
import time
import logging
import pandas as pd

from datetime import datetime
from google_play_scraper import reviews, Sort
import mlflow

# ──────────────────────────────────────────────────────────────────────────────
# Logging & Path Configurations
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
log = logging.getLogger("STAGE_1_SCRAPER")

RAW_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "spotify_reviews_raw.parquet")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
APP_ID = "com.spotify.music"

SCRAPE_TARGETS = [
    {"country": "us", "sort": Sort.MOST_RELEVANT, "limit": 100000},
    {"country": "us", "sort": Sort.NEWEST, "limit": 100000},
]

TARGET_UNIQUE = 100_000
BATCH_SIZE = 200
RATE_LIMIT_SLEEP = 1.0
MAX_RETRIES = 3
RETRY_SLEEP = 5.0

KEEP_COLUMNS = [
    "reviewId",
    "content",
    "score",
    "thumbsUpCount",
    "at",
    "appVersion",
    "repliedAt",
    "replyContent",
]


# ──────────────────────────────────────────────────────────────────────────────
# Incremental Window Resolver
# ──────────────────────────────────────────────────────────────────────────────
def get_incremental_window(raw_output_path: str) -> tuple[datetime | None, datetime, datetime | None, datetime]:
    today = datetime.today()
    first_day_this_month = datetime(today.year, today.month, 1)
    prev_month_end = first_day_this_month - pd.Timedelta(days=1)
    latest_completed_month_end = datetime(prev_month_end.year, prev_month_end.month, prev_month_end.day, 23, 59, 59,
                                          999999)

    if not os.path.exists(raw_output_path):
        log.info("No existing dataset discovered. Transitioning to a historical bootstrap scrape.")
        return None, latest_completed_month_end, None, latest_completed_month_end

    try:
        df_at = pd.read_parquet(raw_output_path, columns=["at"])
        if df_at.empty:
            log.warning("Existing parquet file is empty. Defaulting to historical bootstrap.")
            return None, latest_completed_month_end, None, latest_completed_month_end

        last_scraped_timestamp = pd.to_datetime(df_at["at"]).max()
        if last_scraped_timestamp.tzinfo is not None:
            last_scraped_timestamp = last_scraped_timestamp.tz_convert(None)
        last_scraped_date = last_scraped_timestamp.to_pydatetime()

        return last_scraped_date, latest_completed_month_end, last_scraped_date, latest_completed_month_end

    except Exception as exc:
        log.error(f"Failed to inspect existing parquet tracking columns: {exc}. Falling back to historical bootstrap.")
        return None, latest_completed_month_end, None, latest_completed_month_end


# ──────────────────────────────────────────────────────────────────────────────
# Scraper Core Engine
# ──────────────────────────────────────────────────────────────────────────────
def scrape_stream(
        country: str,
        sort: Sort,
        existing_ids: set,
        stream_limit: int,
        start_date: datetime | None,
        end_date: datetime
) -> list[dict]:
    sort_label = "NEWEST" if sort == Sort.NEWEST else "MOST_RELEVANT"
    start_str = start_date.date() if start_date else "MIN_HISTORICAL"

    log.info(
        f"Starting stream country={country.upper()} sort={sort_label} limit={stream_limit:,} | "
        f"Incremental Bound Window: ({start_str} , {end_date.date()}]"
    )

    stream_reviews = []
    continuation_token = None
    batch_num = 0
    consecutive_empty = 0

    while True:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result, continuation_token = reviews(
                    APP_ID,
                    lang="en",
                    country=country,
                    sort=sort,
                    count=BATCH_SIZE,
                    continuation_token=continuation_token,
                )
                break
            except Exception as exc:
                log.warning(f"Batch {batch_num} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt == MAX_RETRIES:
                    log.error(f"Skipping batch {batch_num} after failures.")
                    result = []
                    continuation_token = None
                else:
                    time.sleep(RETRY_SLEEP)

        new_reviews = []
        early_termination_triggered = False

        for r in result:
            review_date = r.get("at")
            if not review_date:
                continue

            if sort == Sort.NEWEST and start_date and review_date <= start_date:
                early_termination_triggered = True
                break

            if start_date and review_date <= start_date:
                continue
            if review_date > end_date:
                continue

            if r["reviewId"] not in existing_ids:
                existing_ids.add(r["reviewId"])
                trimmed = {col: r.get(col) for col in KEEP_COLUMNS}
                trimmed["source_country"] = country
                trimmed["source_sort"] = sort_label
                trimmed["is_new"] = True
                new_reviews.append(trimmed)

        if early_termination_triggered:
            log.info(
                f"Chronological early termination triggered: Hit review date older than checkpoint ({start_date}).")
            break

        remaining = stream_limit - len(stream_reviews)
        if remaining <= 0:
            break

        new_reviews = new_reviews[:remaining]
        stream_reviews.extend(new_reviews)
        batch_num += 1

        log.info(
            f"Batch={batch_num:>4d} | new={len(new_reviews):>4d} | "
            f"stream_total={len(stream_reviews):>6,d} | global_unique={len(existing_ids):>6,d}"
        )

        if len(stream_reviews) >= stream_limit or len(existing_ids) >= TARGET_UNIQUE:
            break

        if len(new_reviews) == 0:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                break
        else:
            consecutive_empty = 0

        if not continuation_token:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    return stream_reviews


# ──────────────────────────────────────────────────────────────────────────────
# Save Checkpoint Helper
# ──────────────────────────────────────────────────────────────────────────────
def save_checkpoint(new_reviews_list: list[dict]):
    if not new_reviews_list:
        return

    new_df = pd.DataFrame(new_reviews_list)
    new_df["at"] = pd.to_datetime(new_df["at"])
    if "repliedAt" in new_df.columns:
        new_df["repliedAt"] = pd.to_datetime(new_df["repliedAt"])
    if "is_new" not in new_df.columns:
        new_df["is_new"] = True

    try:
        if os.path.exists(RAW_OUTPUT_PATH):
            old_df = pd.read_parquet(RAW_OUTPUT_PATH)
            old_df["is_new"] = False
            df_merged = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df_merged = new_df

        df_merged = df_merged.drop_duplicates(subset=["reviewId"], keep="last")
        df_merged.to_parquet(RAW_OUTPUT_PATH, engine="pyarrow", index=False)
    except Exception as exc:
        log.exception(f"Failed to cleanly commit parquet update checkpoint: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Clean Decoupled Stage Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Executes business logic and writes variables directly into the current child run context.
    """
    log.info(f"Executing Stage 1 Scraper with shared pipeline version identifier: {run_version}")
    os.makedirs(os.path.dirname(RAW_OUTPUT_PATH), exist_ok=True)
    start_time = datetime.now()

    # Calculate Incremental Windows
    start_date, end_date, last_scraped_date, latest_completed_month_end = get_incremental_window(RAW_OUTPUT_PATH)

    # ── No Data Early Abort Constraint ───────────────────────────────────────
    if last_scraped_date and (start_date >= end_date or last_scraped_date >= latest_completed_month_end):
        log.info("Dataset already up to date through latest completed month. Skipping extraction operations.")

        mlflow.log_param("incremental_start_date", start_date.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_param("incremental_end_date", end_date.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_param("last_scraped_date", last_scraped_date.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_param("latest_completed_month_end", latest_completed_month_end.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_metric("newly_scraped_rows", 0)

        try:
            total_rows = len(pd.read_parquet(RAW_OUTPUT_PATH, columns=["reviewId"]))
        except Exception:
            total_rows = 0
        mlflow.log_metric("total_rows_after_merge", total_rows)
        return

    # Seed Deduplication Key Engine
    seen_ids = set()
    if os.path.exists(RAW_OUTPUT_PATH):
        try:
            existing_keys = pd.read_parquet(RAW_OUTPUT_PATH, columns=["reviewId"])
            seen_ids = set(existing_keys["reviewId"].dropna().tolist())
        except Exception as e:
            log.warning(f"Unable to read existing keys for global deduplication: {e}")

    # ── Execute Processing Strategy ──────────────────────────────────────────
    all_new_reviews = []
    for target in SCRAPE_TARGETS:
        if len(seen_ids) >= TARGET_UNIQUE:
            break

        stream_data = scrape_stream(
            country=target["country"],
            sort=target["sort"],
            existing_ids=seen_ids,
            stream_limit=target["limit"],
            start_date=start_date,
            end_date=end_date
        )
        all_new_reviews.extend(stream_data)
        save_checkpoint(all_new_reviews)

    newly_scraped_rows = len(all_new_reviews)

    # Finalize state metrics calculations
    try:
        df_final = pd.read_parquet(RAW_OUTPUT_PATH)
        df_final["at"] = pd.to_datetime(df_final["at"])
        df_final["scraped_at"] = datetime.now()
        df_final = df_final.drop_duplicates(subset=["reviewId"], keep="last")
        df_final.to_parquet(RAW_OUTPUT_PATH, engine="pyarrow", index=False)

        total_rows_after_merge = len(df_final)
        unique_id_count = int(df_final['reviewId'].nunique())
    except Exception as e:
        log.error(f"Failed parsing validation stats: {e}")
        total_rows_after_merge = newly_scraped_rows
        unique_id_count = 0

    elapsed = datetime.now() - start_time

    # ── Document Metrics & Artifacts in Nested Child Active Context ───────────
    mlflow.log_param("pipeline_stage_version", run_version)
    mlflow.log_param("target_unique_limit", TARGET_UNIQUE)
    mlflow.log_param("incremental_start_date", start_date.strftime("%Y-%m-%d %H:%M:%S") if start_date else "None")
    mlflow.log_param("incremental_end_date", end_date.strftime("%Y-%m-%d %H:%M:%S"))
    mlflow.log_param("last_scraped_date",
                     last_scraped_date.strftime("%Y-%m-%d %H:%M:%S") if last_scraped_date else "None")
    mlflow.log_param("latest_completed_month_end", latest_completed_month_end.strftime("%Y-%m-%d %H:%M:%S"))

    mlflow.log_metric("newly_scraped_rows", newly_scraped_rows)
    mlflow.log_metric("total_rows_after_merge", total_rows_after_merge)
    mlflow.log_metric("unique_review_ids", unique_id_count)

    if os.path.exists(RAW_OUTPUT_PATH):
        mlflow.log_artifact(RAW_OUTPUT_PATH, artifact_path="raw_scraped_reviews")
