"""
src/topic_registry/noise_buffer.py
------------------------------------
NoiseBuffer: accumulates reviews that fell below the similarity threshold
during assignment. Persisted as Parquet with embedded vectors cached inline.
Flushed after a discovery run promotes buffer members.
"""

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

SCHEMA_COLS = [
    "review_id",
    "content",
    "embedding",               # list[float] — 768-dim, L2-normalised
    "best_match_topic_id",     # closest existing topic even if below threshold
    "best_match_similarity",   # cosine similarity to closest topic
    "added_at",
    "batch_month",             # YYYY-MM
    "promoted_to_topic_id",    # null until a discovery run promotes this review
]


class NoiseBuffer:
    """
    Append-only accumulation store for UNKNOWN reviews.
    Embeddings are stored inline to avoid re-computation during discovery runs.
    """

    def __init__(self, buffer_path: str):
        self.buffer_path = buffer_path
        self._df: pd.DataFrame = pd.DataFrame(columns=SCHEMA_COLS)
        self.load()

    # ── Load / Save ───────────────────────────────────────────────────────────

    def load(self) -> None:
        if os.path.exists(self.buffer_path):
            self._df = pd.read_parquet(self.buffer_path)
            # Filter: only unresolved reviews (not yet promoted)
            unpromoted = self._df[self._df["promoted_to_topic_id"].isna()]
            log.info(
                "NoiseBuffer loaded: %d total rows, %d unpromoted.",
                len(self._df), len(unpromoted),
            )
        else:
            log.info("No noise buffer found — initialised empty.")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)
        self._df.to_parquet(self.buffer_path, index=False)
        log.debug("NoiseBuffer saved: %d rows.", len(self._df))

    # ── Append ────────────────────────────────────────────────────────────────

    def add_reviews(
        self,
        review_ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
        best_match_topic_ids: list[int],
        best_match_similarities: list[float],
        batch_month: str,
    ) -> None:
        """
        Appends a batch of UNKNOWN reviews to the buffer.
        All arguments must be aligned lists of the same length.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for i, (rid, text, emb, topic_id, sim) in enumerate(
            zip(review_ids, texts, embeddings, best_match_topic_ids, best_match_similarities)
        ):
            rows.append({
                "review_id": rid,
                "content": text,
                "embedding": emb.tolist(),
                "best_match_topic_id": topic_id,
                "best_match_similarity": float(sim),
                "added_at": now,
                "batch_month": batch_month,
                "promoted_to_topic_id": None,
            })
        new_rows = pd.DataFrame(rows)
        self._df  = pd.concat([self._df, new_rows], ignore_index=True)
        log.info("NoiseBuffer: appended %d reviews. Total unpromoted: %d.", len(rows), self.unpromoted_size())

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_unpromoted(self) -> tuple[list[str], list[str], np.ndarray]:
        """
        Returns (review_ids, texts, embeddings) for reviews not yet promoted.
        """
        unpromoted = self._df[self._df["promoted_to_topic_id"].isna()]
        if unpromoted.empty:
            return [], [], np.array([])

        review_ids = unpromoted["review_id"].tolist()
        texts      = unpromoted["content"].tolist()
        embeddings = np.stack(unpromoted["embedding"].apply(np.array).values)
        return review_ids, texts, embeddings

    def unpromoted_size(self) -> int:
        return int(self._df["promoted_to_topic_id"].isna().sum())

    def total_size(self) -> int:
        return len(self._df)

    # ── Post-Discovery Update ─────────────────────────────────────────────────

    def mark_promoted(self, review_ids: list[str], topic_id: int) -> None:
        """
        Called after a discovery run promotes a cluster.
        Marks those reviews as resolved so they are excluded from the next run.
        """
        mask = self._df["review_id"].isin(review_ids)
        self._df.loc[mask, "promoted_to_topic_id"] = topic_id
        log.info(
            "NoiseBuffer: %d reviews marked as promoted to topic %d.",
            int(mask.sum()), topic_id,
        )

    def get_batch_month_distribution(self) -> dict[str, int]:
        """Returns count of unpromoted reviews per YYYY-MM batch."""
        unpromoted = self._df[self._df["promoted_to_topic_id"].isna()]
        return unpromoted.groupby("batch_month").size().to_dict()

    def clear_all_promoted(self) -> None:
        """
        Removes promoted rows from the DataFrame to keep buffer lean.
        Called after a discovery run is fully committed and audited.
        """
        before = len(self._df)
        self._df = self._df[self._df["promoted_to_topic_id"].isna()].copy()
        after   = len(self._df)
        log.info("NoiseBuffer compacted: removed %d promoted rows. Remaining: %d.", before - after, after)