"""
src/topic_registry/validator.py
---------------------------------
TopicValidator: applies promotion gates to candidate clusters
discovered in the noise buffer. All gates must pass for promotion.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """
    All thresholds are configurable per deployment.
    Defaults calibrated for <10K reviews/month on app store data.
    """
    min_cluster_size: int   = 30       # absolute floor — prevents noise spikes becoming topics
    min_coherence: float    = 0.35     # mean pairwise cosine similarity within cluster
    min_support_pct: float  = 1.0      # % of total buffer — filters one-off incidents
    min_avg_confidence: float = 0.40   # mean cosine similarity of members to their centroid
    max_dedup_similarity: float = 0.75 # max allowed similarity to any existing topic centroid


@dataclass
class ValidationResult:
    passed: bool
    rejection_reason: Optional[str]
    metrics: dict = field(default_factory=dict)


class TopicValidator:

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def validate(
        self,
        candidate_embeddings: np.ndarray,
        buffer_total_size: int,
        existing_centroid_matrix: Optional[np.ndarray],
    ) -> ValidationResult:
        """
        Runs all promotion gates on a candidate cluster.

        Args:
            candidate_embeddings:   Embeddings of all members in this candidate cluster.
            buffer_total_size:      Total size of the noise buffer (for support % calc).
            existing_centroid_matrix: (T, 768) matrix of existing topic centroids.
                                      None if registry is empty (initial training).
        Returns:
            ValidationResult with pass/fail and metrics.
        """
        cfg      = self.config
        metrics  = {}
        n        = len(candidate_embeddings)
        normed   = normalize(candidate_embeddings)

        # Gate 1: Minimum cluster size
        metrics["cluster_size"] = n
        if n < cfg.min_cluster_size:
            return ValidationResult(
                passed=False,
                rejection_reason="below_min_size",
                metrics=metrics,
            )

        # Gate 2: Topic coherence (mean pairwise cosine similarity)
        coherence = self._compute_coherence(normed)
        metrics["coherence"] = round(coherence, 4)
        if coherence < cfg.min_coherence:
            return ValidationResult(
                passed=False,
                rejection_reason="below_coherence",
                metrics=metrics,
            )

        # Gate 3: Support percentage
        support_pct = (n / buffer_total_size) * 100
        metrics["support_pct"] = round(support_pct, 4)
        if support_pct < cfg.min_support_pct:
            return ValidationResult(
                passed=False,
                rejection_reason="below_support_pct",
                metrics=metrics,
            )

        # Gate 4: Average assignment confidence (mean cosine to centroid)
        centroid   = normed.mean(axis=0, keepdims=True)
        confidence = float(cosine_similarity(normed, centroid).mean())
        metrics["avg_confidence"] = round(confidence, 4)
        if confidence < cfg.min_avg_confidence:
            return ValidationResult(
                passed=False,
                rejection_reason="below_avg_confidence",
                metrics=metrics,
            )

        # Gate 5: Deduplication against existing topics
        if existing_centroid_matrix is not None and len(existing_centroid_matrix) > 0:
            max_sim_to_existing = float(
                cosine_similarity(centroid, existing_centroid_matrix).max()
            )
            metrics["max_sim_to_existing"] = round(max_sim_to_existing, 4)
            if max_sim_to_existing >= cfg.max_dedup_similarity:
                return ValidationResult(
                    passed=False,
                    rejection_reason="duplicate_of_existing",
                    metrics=metrics,
                )
        else:
            metrics["max_sim_to_existing"] = None

        return ValidationResult(passed=True, rejection_reason=None, metrics=metrics)

    @staticmethod
    def _compute_coherence(normed_embeddings: np.ndarray) -> float:
        """
        Mean pairwise cosine similarity within a cluster, excluding self-similarity.
        Range [0, 1]: 1.0 = perfectly coherent, 0.0 = maximally diverse.
        """
        n = len(normed_embeddings)
        if n < 2:
            return 1.0  # single-member cluster — vacuously coherent

        sim_matrix = normed_embeddings @ normed_embeddings.T    # (n, n), already L2-normed
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        return float(sim_matrix[mask].mean())

    def batch_validate(
        self,
        candidate_clusters: dict[int, np.ndarray],
        buffer_total_size: int,
        existing_centroid_matrix: Optional[np.ndarray],
    ) -> dict[int, ValidationResult]:
        """
        Validates all candidate clusters from a discovery run.
        Returns {raw_cluster_id: ValidationResult}.
        """
        results = {}
        for cluster_id, embeddings in candidate_clusters.items():
            result = self.validate(embeddings, buffer_total_size, existing_centroid_matrix)
            status = "PASS" if result.passed else f"FAIL ({result.rejection_reason})"
            log.info(
                "Cluster %d [n=%d]: %s | metrics=%s",
                cluster_id, len(embeddings), status, result.metrics,
            )
            results[cluster_id] = result
        return results