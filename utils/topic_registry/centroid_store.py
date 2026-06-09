"""
src/topic_registry/centroid_store.py
--------------------------------------
CentroidStore: stores topic centroid vectors (768-dim, all-mpnet-base-v2)
as a Parquet file. Supports cosine nearest-centroid assignment across
all active topics regardless of model version.

Design decision: centroids are the inference backbone — BERTopic models
are discovery artefacts only. This store spans all model versions.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)

NOISE_TOPIC_ID = -1


class CentroidStore:
    """
    Append-only store of topic centroid embeddings.
    Loaded fully into memory — at 768-dim × 10K topics = ~30MB max.
    """

    SCHEMA_COLS = [
        "topic_id",
        "model_version",
        "centroid_review_count",
        "updated_at",
        "centroid",   # stored as object (list[float]) in Parquet
    ]

    def __init__(self, store_path: str):
        self.store_path = store_path
        self._df: pd.DataFrame = pd.DataFrame(columns=self.SCHEMA_COLS)
        self._centroid_matrix: Optional[np.ndarray] = None   # cache: (T, 768)
        self._active_ids: Optional[list[int]] = None          # cache: aligned with matrix rows
        self.load()

    # ── Load / Save ───────────────────────────────────────────────────────────

    def load(self) -> None:
        if os.path.exists(self.store_path):
            self._df = pd.read_parquet(self.store_path)
            log.info("CentroidStore loaded: %d topic centroids.", len(self._df))
        else:
            log.info("No centroid store found — initialised empty.")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self._df.to_parquet(self.store_path, index=False)
        self._invalidate_cache()
        log.info("CentroidStore saved: %d topic centroids.", len(self._df))

    # ── Append ────────────────────────────────────────────────────────────────

    def add_centroid(
        self,
        topic_id: int,
        centroid: np.ndarray,
        model_version: str,
        review_count: int,
    ) -> None:
        if topic_id in self._df["topic_id"].values:
            raise ValueError(
                f"Centroid for topic_id {topic_id} already exists. "
                "Use update_centroid() for drift correction."
            )
        row = pd.DataFrame([{
            "topic_id": topic_id,
            "model_version": model_version,
            "centroid_review_count": review_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "centroid": normalize(centroid.reshape(1, -1))[0].tolist(),
        }])
        self._df = pd.concat([self._df, row], ignore_index=True)
        self._invalidate_cache()
        log.debug("Centroid added: topic_id=%d, model=%s", topic_id, model_version)

    def update_centroid(
        self,
        topic_id: int,
        new_centroid: np.ndarray,
        new_review_count: int,
    ) -> None:
        """
        Updates the centroid vector for an existing topic (drift correction).
        The topic ID and model_version are preserved — only the vector changes.
        """
        mask = self._df["topic_id"] == topic_id
        if not mask.any():
            raise KeyError(f"Topic {topic_id} not found in centroid store.")
        self._df.loc[mask, "centroid"] = [normalize(new_centroid.reshape(1, -1))[0].tolist()]
        self._df.loc[mask, "centroid_review_count"] = new_review_count
        self._df.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
        self._invalidate_cache()
        log.info("Centroid updated: topic_id=%d, new_review_count=%d", topic_id, new_review_count)

    # ── Assignment ────────────────────────────────────────────────────────────

    def assign(
        self,
        embeddings: np.ndarray,
        active_topic_ids: list[int],
        similarity_threshold: float = 0.35,
    ) -> tuple[list[int], list[float]]:
        """
        Assigns each embedding to the nearest active topic centroid via cosine similarity.
        Returns (assigned_topic_ids, confidence_scores).
        Reviews below threshold receive NOISE_TOPIC_ID (-1).

        Args:
            embeddings:          L2-normalised embeddings, shape (N, 768)
            active_topic_ids:    List of active topic IDs to match against
            similarity_threshold: Minimum cosine similarity for a valid assignment
        """
        matrix, aligned_ids = self._get_active_centroid_matrix(active_topic_ids)

        if matrix is None or len(aligned_ids) == 0:
            log.warning("No active centroids available — all reviews routed to noise buffer.")
            return [NOISE_TOPIC_ID] * len(embeddings), [0.0] * len(embeddings)

        # Shape: (N, T) — cosine similarity of each review against each active centroid
        sim_matrix = cosine_similarity(embeddings, matrix)

        best_scores  = sim_matrix.max(axis=1)                    # (N,)
        best_indices = sim_matrix.argmax(axis=1)                 # (N,)
        best_ids     = [aligned_ids[i] for i in best_indices]    # (N,)

        assigned_ids    = []
        assigned_scores = []

        for topic_id, score in zip(best_ids, best_scores):
            if score >= similarity_threshold:
                assigned_ids.append(topic_id)
            else:
                assigned_ids.append(NOISE_TOPIC_ID)
            assigned_scores.append(float(score))

        return assigned_ids, assigned_scores

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_centroid(self, topic_id: int) -> Optional[np.ndarray]:
        row = self._df[self._df["topic_id"] == topic_id]
        if row.empty:
            return None
        return np.array(row.iloc[0]["centroid"])

    def get_all_centroids_as_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Returns (matrix, topic_ids) for all stored centroids."""
        if self._df.empty:
            return np.array([]), []
        matrix = np.stack(self._df["centroid"].apply(np.array).values)
        ids    = self._df["topic_id"].tolist()
        return matrix, ids

    def _get_active_centroid_matrix(
        self,
        active_topic_ids: list[int],
    ) -> tuple[Optional[np.ndarray], list[int]]:
        if self._centroid_matrix is not None and self._active_ids == active_topic_ids:
            return self._centroid_matrix, self._active_ids

        filtered = self._df[self._df["topic_id"].isin(active_topic_ids)]
        if filtered.empty:
            return None, []

        matrix     = np.stack(filtered["centroid"].apply(np.array).values)
        aligned_ids = filtered["topic_id"].tolist()

        # Cache for repeated calls in the same batch
        self._centroid_matrix = matrix
        self._active_ids      = aligned_ids
        return matrix, aligned_ids

    def _invalidate_cache(self) -> None:
        self._centroid_matrix = None
        self._active_ids      = None

    def topic_count(self) -> int:
        return len(self._df)