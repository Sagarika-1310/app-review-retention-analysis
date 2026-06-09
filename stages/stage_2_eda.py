"""
stage_2_eda.py
──────────────────────
Detailed Exploratory Data Analysis for Spotify Play Store reviews.
Tracked natively inside the centralized MLflow lifecycle workspace.

Exposes:
- run(run_version): Clean decoupled execution layer hooked to external MLflow contexts.
"""

import os
import logging
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib
import mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ── Project layout ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in locals() else ".."
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
log = logging.getLogger("STAGE_2_EDA")

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "reviews_english.parquet")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "eda_outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "eda_spotify_reviews_summary.csv")

# ── Plotting style ─────────────────────────────────────────────────────────────
PALETTE = ["#1DB954", "#191414", "#535353", "#B3B3B3", "#FFFFFF"]  # Spotify colours
sns.set_theme(style="darkgrid", palette=PALETTE)
plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "#191414",
                     "axes.facecolor": "#191414", "text.color": "#FFFFFF",
                     "axes.labelcolor": "#FFFFFF", "xtick.color": "#FFFFFF",
                     "ytick.color": "#FFFFFF", "axes.edgecolor": "#535353",
                     "grid.color": "#2a2a2a", "font.family": "DejaVu Sans"})

SCORE_COLOURS = {1: "#E74C3C", 2: "#E67E22", 3: "#F1C40F",
                 4: "#2ECC71", 5: "#1DB954"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    log.info("Loading data from %s", path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    log.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. BASIC OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def basic_overview(df: pd.DataFrame) -> pd.DataFrame:
    log.info("─── Basic overview ───")
    log.info("Shape          : %s", df.shape)
    log.info("Columns        : %s", df.columns.tolist())
    log.info("Dtypes\n%s", df.dtypes.to_string())
    log.info("Memory usage   : %.2f MB", df.memory_usage(deep=True).sum() / 1e6)

    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "null_count": null_counts,
        "null_pct": null_pct,
        "nunique": df.nunique(),
        "sample_value": df.iloc[0] if len(df) > 0 else np.nan,
    })
    log.info("Column summary\n%s", summary.to_string())
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATETIME PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    log.info("─── Parsing datetime columns ───")
    for col in ["at", "repliedAt", "scraped_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            log.info("  %-12s : min=%s  max=%s  nulls=%d",
                     col, df[col].min(), df[col].max(), df[col].isnull().sum())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCORE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
def analyse_scores(df: pd.DataFrame) -> None:
    log.info("─── Score distribution ───")
    dist = df["score"].value_counts().sort_index()
    pct = (dist / len(df) * 100).round(2)
    for s, c, p in zip(dist.index, dist.values, pct.values):
        log.info("  Score %d : %6d reviews  (%.2f%%)", s, c, p)

    mean_score = df["score"].mean()
    median_score = df["score"].median()
    std_score = df["score"].std()

    log.info("  Mean score : %.3f  |  Median : %.1f  |  Std : %.3f", mean_score, median_score, std_score)

    # Log distribution aggregates directly into active child run context
    mlflow.log_metric("reviews_mean_score", float(mean_score))
    mlflow.log_metric("reviews_median_score", float(median_score))
    mlflow.log_metric("reviews_std_score", float(std_score))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Score Distribution", color="#1DB954", fontsize=14, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(dist.index, dist.values,
                       color=[SCORE_COLOURS[s] for s in dist.index])
    axes[0].set_xlabel("Star Rating")
    axes[0].set_ylabel("Review Count")
    axes[0].set_title("Count per Star", color="#B3B3B3")
    for bar, val in zip(bars, dist.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{val:,}", ha="center", va="bottom", fontsize=9, color="#FFFFFF")

    # Pie chart
    axes[1].pie(dist.values, labels=[f"★{s}" for s in dist.index],
                colors=[SCORE_COLOURS[s] for s in dist.index],
                autopct="%1.1f%%", startangle=140,
                textprops={"color": "#FFFFFF", "fontsize": 10})
    axes[1].set_title("Share per Star", color="#B3B3B3")

    plt.tight_layout()
    _save(fig, "01_score_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def analyse_temporal(df: pd.DataFrame) -> None:
    if "at" not in df.columns or df["at"].isnull().all():
        log.warning("No valid 'at' datetime column – skipping temporal analysis.")
        return
    log.info("─── Temporal analysis ───")

    df["year"] = df["at"].dt.year
    df["month"] = df["at"].dt.month
    df["year_month"] = df["at"].dt.to_period("M")
    df["dayofweek"] = df["at"].dt.day_name()
    df["hour"] = df["at"].dt.hour

    # Reviews over time (monthly)
    monthly = df.groupby("year_month").agg(
        review_count=("score", "count"),
        avg_score=("score", "mean")
    ).reset_index()
    monthly["year_month_str"] = monthly["year_month"].astype(str)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.suptitle("Reviews Over Time (Monthly)", color="#1DB954", fontsize=14, fontweight="bold")
    ax1.bar(monthly["year_month_str"], monthly["review_count"],
            color="#1DB954", alpha=0.6, label="Review Count")
    ax1.set_ylabel("Review Count", color="#1DB954")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(monthly["year_month_str"], monthly["avg_score"],
             color="#E74C3C", linewidth=2, marker="o", markersize=3, label="Avg Score")
    ax2.set_ylabel("Avg Score", color="#E74C3C")
    ax2.set_ylim(1, 5)
    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88),
               facecolor="#191414", edgecolor="#535353", labelcolor="#FFFFFF")
    plt.tight_layout()
    _save(fig, "02_reviews_over_time.png")

    # Day-of-week pattern
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = df["dayofweek"].value_counts().reindex(dow_order)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(dow_counts.index, dow_counts.values, color="#1DB954")
    ax.set_title("Reviews by Day of Week", color="#1DB954")
    ax.set_ylabel("Review Count")
    plt.tight_layout()
    _save(fig, "03_reviews_by_dow.png")

    # Hourly pattern
    hour_counts = df["hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hour_counts.index, hour_counts.values, color="#1DB954",
            linewidth=2, marker="o", markersize=4)
    ax.fill_between(hour_counts.index, hour_counts.values, alpha=0.2, color="#1DB954")
    ax.set_title("Reviews by Hour of Day (UTC)", color="#1DB954")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Review Count")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    _save(fig, "04_reviews_by_hour.png")

    log.info("  Date range   : %s → %s", df["at"].min(), df["at"].max())
    log.info("  Peak month   : %s", monthly.loc[monthly["review_count"].idxmax(), "year_month_str"])
    log.info("  Best DOW     : %s", dow_counts.idxmax())


# ══════════════════════════════════════════════════════════════════════════════
# 6. TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def analyse_text(df: pd.DataFrame) -> None:
    text_col = "content_clean" if "content_clean" in df.columns else "content"
    log.info("─── Text analysis (column: %s) ───", text_col)

    df["review_length"] = df[text_col].astype(str).apply(len)
    df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    text_col_orig = "content" if "content" in df.columns else text_col
    df["sentence_count"] = df[text_col_orig].astype(str).apply(
        lambda x: max(
            1,
            len([s for s in re.split(r'[.!?]+', x) if s.strip()])
        )
    )

    log.info("  Avg chars  : %.1f  |  Avg words : %.1f  |  Avg sentences : %.1f",
             df["review_length"].mean(), df["word_count"].mean(),
             df["sentence_count"].mean())

    mlflow.log_metric("avg_char_count", float(df["review_length"].mean()))
    mlflow.log_metric("avg_word_count", float(df["word_count"].mean()))

    # Length distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Review Text Metrics", color="#1DB954", fontsize=14, fontweight="bold")
    for ax, col, label in zip(axes,
                              ["review_length", "word_count", "sentence_count"],
                              ["Character Count", "Word Count", "Sentence Count"]):
        ax.hist(df[col].clip(upper=df[col].quantile(0.98)), bins=50,
                color="#1DB954", edgecolor="#191414")
        ax.set_title(label, color="#B3B3B3")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    _save(fig, "05_text_length_distributions.png")

    # Length vs score (box plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    score_groups = [df.loc[df["score"] == s, "word_count"].clip(upper=300).values
                    for s in range(1, 6)]
    bp = ax.boxplot(score_groups, patch_artist=True, notch=True,
                    labels=[f"★{s}" for s in range(1, 6)])
    for patch, s in zip(bp["boxes"], range(1, 6)):
        patch.set_facecolor(SCORE_COLOURS[s])
        patch.set_alpha(0.7)
    ax.set_title("Word Count by Star Rating", color="#1DB954")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Word Count")
    plt.tight_layout()
    _save(fig, "06_wordcount_by_score.png")

    # Word clouds per sentiment group
    _wordcloud(df, text_col, [1, 2], "Negative Reviews (1-2 ★)", "07_wordcloud_negative.png")
    _wordcloud(df, text_col, [4, 5], "Positive Reviews (4-5 ★)", "08_wordcloud_positive.png")


def _wordcloud(df, text_col, scores, title, fname):
    subset = df[df["score"].isin(scores)][text_col].dropna().astype(str)
    text = " ".join(subset.tolist())
    if not text.strip():
        log.warning("No text for wordcloud: %s", title)
        return
    colour = "#E74C3C" if 1 in scores else "#1DB954"
    wc = WordCloud(width=1200, height=500, background_color="#191414",
                   colormap="RdYlGn" if 5 in scores else "Reds",
                   max_words=150, collocations=True).generate(text)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, color=colour, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, fname)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENGAGEMENT ANALYSIS (thumbsUpCount)
# ══════════════════════════════════════════════════════════════════════════════
def analyse_engagement(df: pd.DataFrame) -> None:
    if "thumbsUpCount" not in df.columns:
        log.warning("Column 'thumbsUpCount' not found – skipping.")
        return
    log.info("─── Engagement (thumbsUpCount) ───")
    log.info("  Total thumbs up : %d", df["thumbsUpCount"].sum())
    log.info("  Mean            : %.2f  |  Median : %.0f  |  Max : %d",
             df["thumbsUpCount"].mean(), df["thumbsUpCount"].median(),
             df["thumbsUpCount"].max())

    mlflow.log_metric("total_thumbs_up_count", int(df["thumbsUpCount"].sum()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Thumbs-Up Engagement", color="#1DB954", fontsize=14, fontweight="bold")

    clipped = df["thumbsUpCount"].clip(upper=df["thumbsUpCount"].quantile(0.99))
    axes[0].hist(clipped, bins=50, color="#1DB954", edgecolor="#191414")
    axes[0].set_title("Distribution (clipped @99th pct)", color="#B3B3B3")
    axes[0].set_xlabel("Thumbs Up Count")
    axes[0].set_ylabel("Frequency")

    avg_likes = df.groupby("score")["thumbsUpCount"].mean()
    axes[1].bar(avg_likes.index, avg_likes.values,
                color=[SCORE_COLOURS[s] for s in avg_likes.index])
    axes[1].set_title("Avg Thumbs Up per Star", color="#B3B3B3")
    axes[1].set_xlabel("Star Rating")
    axes[1].set_ylabel("Avg Thumbs Up")

    plt.tight_layout()
    _save(fig, "09_engagement.png")

    # Top 10 most-liked reviews
    top10 = df.nlargest(10, "thumbsUpCount")[["score", "thumbsUpCount", "content"]].copy()
    top10["content"] = top10["content"].astype(str).str[:120] + "…"
    log.info("Top 10 most-liked reviews:\n%s", top10.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 8. REPLY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def analyse_replies(df: pd.DataFrame) -> None:
    if "replyContent" not in df.columns:
        log.warning("'replyContent' column not found – skipping.")
        return
    log.info("─── Reply analysis ───")

    df["has_reply"] = df["replyContent"].notna() & (df["replyContent"].astype(str).str.strip() != "")
    total = len(df)
    replied = df["has_reply"].sum()
    reply_rate = (replied / total * 100) if total > 0 else 0.0
    log.info("  Replied reviews : %d / %d  (%.1f%%)", replied, total, reply_rate)

    mlflow.log_metric("developer_replied_count", int(replied))
    mlflow.log_metric("developer_reply_rate_pct", float(reply_rate))

    reply_by_score = df.groupby("score")["has_reply"].mean().mul(100).round(2)
    log.info("  Reply rate by score:\n%s", reply_by_score.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(reply_by_score.index, reply_by_score.values,
           color=[SCORE_COLOURS[s] for s in reply_by_score.index])
    ax.set_title("Developer Reply Rate by Star Rating (%)", color="#1DB954")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Reply Rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    _save(fig, "10_reply_rate_by_score.png")

    if "repliedAt" in df.columns and "at" in df.columns:
        mask = df["has_reply"] & df["repliedAt"].notna() & df["at"].notna()
        df.loc[mask, "reply_lag_hours"] = (
                (df.loc[mask, "repliedAt"] - df.loc[mask, "at"]).dt.total_seconds() / 3600
        )
        valid_lag = df["reply_lag_hours"].dropna()
        valid_lag = valid_lag[valid_lag >= 0]
        log.info("  Avg reply lag   : %.1f hrs  |  Median : %.1f hrs",
                 valid_lag.mean(), valid_lag.median())
        if not valid_lag.empty:
            mlflow.log_metric("median_reply_lag_hours", float(valid_lag.median()))


# ══════════════════════════════════════════════════════════════════════════════
# 9. APP VERSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def analyse_versions(df: pd.DataFrame) -> None:
    if "appVersion" not in df.columns:
        log.warning("'appVersion' column not found – skipping.")
        return
    log.info("─── App version analysis ───")

    version_stats = (df.groupby("appVersion")["score"]
                     .agg(["mean", "count"])
                     .query("count >= 30")
                     .sort_values("count", ascending=False)
                     .head(20))
    log.info("  Top 20 versions by review count:\n%s", version_stats.to_string())

    top20 = version_stats.sort_values("mean")
    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.barh(top20.index, top20["mean"],
                   color=["#E74C3C" if v < 3.0 else "#F1C40F" if v < 4.0 else "#1DB954"
                          for v in top20["mean"]])
    ax.axvline(x=df["score"].mean(), color="#B3B3B3", linestyle="--", linewidth=1,
               label=f"Overall avg ({df['score'].mean():.2f})")
    ax.set_title("Avg Score by App Version (top 20 by volume)", color="#1DB954")
    ax.set_xlabel("Avg Score")
    ax.set_xlim(1, 5)
    ax.legend(facecolor="#191414", edgecolor="#535353", labelcolor="#FFFFFF")
    plt.tight_layout()
    _save(fig, "12_score_by_version.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def correlation_heatmap(df: pd.DataFrame) -> None:
    log.info("─── Correlation heatmap ───")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        log.warning("Not enough numeric columns for correlation heatmap.")
        return

    corr = df[num_cols].corr()
    log.info("Correlation matrix:\n%s", corr.to_string())

    fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, linecolor="#191414", ax=ax,
                annot_kws={"size": 9}, vmin=-1, vmax=1)
    ax.set_title("Numeric Feature Correlations", color="#1DB954", fontsize=13)
    plt.tight_layout()
    _save(fig, "13_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. DUPLICATE CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_duplicates(df: pd.DataFrame) -> None:
    log.info("─── Duplicate check ───")
    dup_rows = df.duplicated().sum()
    log.info("  Fully duplicate rows  : %d", dup_rows)
    mlflow.log_metric("fully_duplicate_rows_count", int(dup_rows))

    if "reviewId" in df.columns:
        dup_ids = df["reviewId"].duplicated().sum()
        log.info("  Duplicate reviewIds   : %d", dup_ids)
        mlflow.log_metric("duplicate_review_ids_count", int(dup_ids))
    if "content" in df.columns:
        dup_content = df["content"].duplicated(keep=False).sum()
        log.info("  Reviews with same content : %d", dup_content)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("  Saved → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# CLEAN DECOUPLED STAGE ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
def run(run_version: str):
    """
    Exposed entrypoint invoked exclusively by the run_pipeline.py orchestrator.
    Executes exploratory operations and logs metrics and generated visualization figures to MLflow.
    """
    log.info(f"Executing Stage 2 Exploratory Data Analysis with shared pipeline version identifier: {run_version}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing absolute upstream preprocessed tracking file: {INPUT_PATH}")

    # ── Orchestrate Feature Analysis Strategy ─────────────────────────────────
    df = load_data(INPUT_PATH)
    df = df[df["is_new"].isin([True, "true"])]
    mlflow.log_param("pipeline_stage_version", run_version)
    mlflow.log_metric("total_eda_records_count", len(df))

    col_summary = basic_overview(df)
    df = parse_datetimes(df)

    check_duplicates(df)
    analyse_scores(df)
    analyse_temporal(df)
    analyse_text(df)
    analyse_engagement(df)
    analyse_replies(df)
    analyse_versions(df)
    correlation_heatmap(df)

    # ── Save enriched summary matrix locally ──────────────────────────────────
    col_summary.to_csv(OUTPUT_PATH)
    log.info("Column summary saved → %s", OUTPUT_PATH)

    # ── Register all outputs as artifacts in active MLflow workspace ─────────
    if os.path.exists(OUTPUT_DIR):
        mlflow.log_artifacts(OUTPUT_DIR, artifact_path="eda_visualizations_and_summaries")
        log.info("Successfully uploaded all generated plots and data summary profiles to MLflow registry.")
