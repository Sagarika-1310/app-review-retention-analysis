"""
src/topic_registry/discovery_engine.py
----------------------------------------
DiscoveryEngine: runs BERTopic exclusively on the noise buffer to find
emerging pain points. BERTopic is a discovery tool here — not an inference tool.
Inference is handled by CentroidStore nearest-centroid assignment.
"""

import logging
import os
from typing import Optional

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
from umap import UMAP

log = logging.getLogger(__name__)

NOISE_LABEL = -1


class DiscoveryEngine:
    """
    Wraps BERTopic for noise-buffer-only discovery runs.

    HDBSCAN is stricter here than in initial training:
    - Higher min_cluster_size  → prevents micro-cluster promotion
    - Higher min_samples       → noise tolerance higher in a biased (OOD) buffer

    UMAP n_neighbors is lower because the buffer is smaller and denser
    in the low-confidence region.
    """

    def __init__(
        self,
        model_version: str,
        models_dir: str,
        min_cluster_size: int = 30,
        min_samples: int = 10,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 10,
        max_topics_per_run: int = 10,
    ):
        self.model_version       = model_version
        self.models_dir          = models_dir
        self.min_cluster_size    = min_cluster_size
        self.min_samples         = min_samples
        self.umap_n_components   = umap_n_components
        self.umap_n_neighbors    = umap_n_neighbors
        self.max_topics_per_run  = max_topics_per_run
        self.topic_model: Optional[BERTopic] = None

    def run(
        self,
        texts: list[str],
        embeddings: np.ndarray,
    ) -> dict[int, dict]:
        """
        Fits BERTopic on buffer texts/embeddings.
        Returns {raw_cluster_id: {"embeddings": ..., "texts": ..., "keywords": ..., "centroid": ...}}
        Only non-noise clusters are returned.
        """
        log.info(
            "DiscoveryEngine: fitting BERTopic on %d buffer reviews [model_version=%s].",
            len(texts), self.model_version,
        )

        umap_model    = self._build_umap()
        hdbscan_model = self._build_hdbscan()

        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            language="english",
            calculate_probabilities=False,    # disabled for speed; not needed for discovery
            verbose=True,
            nr_topics="auto",                 # auto-merges semantically redundant clusters
            min_topic_size=self.min_cluster_size,
        )

        raw_topics, _ = self.topic_model.fit_transform(texts, embeddings)

        raw_topic_ids = [t for t in set(raw_topics) if t != NOISE_LABEL]
        log.info(
            "Raw clusters found: %d (noise reviews: %d).",
            len(raw_topic_ids),
            int(sum(1 for t in raw_topics if t == NOISE_LABEL)),
        )

        # Cap at max_topics_per_run, ranked by size (largest first)
        topic_sizes = {
            tid: sum(1 for t in raw_topics if t == tid)
            for tid in raw_topic_ids
        }
        top_topics = sorted(topic_sizes, key=topic_sizes.get, reverse=True)[: self.max_topics_per_run]

        if len(raw_topic_ids) > self.max_topics_per_run:
            log.warning(
                "Capping discovery at %d topics (found %d). "
                "Smallest %d clusters will not be considered for promotion this run.",
                self.max_topics_per_run,
                len(raw_topic_ids),
                len(raw_topic_ids) - self.max_topics_per_run,
            )

        texts_arr = np.array(texts)
        result    = {}

        for tid in top_topics:
            mask              = np.array(raw_topics) == tid
            member_embeddings = embeddings[mask]
            member_texts      = texts_arr[mask].tolist()
            centroid          = normalize(member_embeddings.mean(axis=0, keepdims=True))[0]
            keywords          = [kw for kw, _ in (self.topic_model.get_topic(tid) or [])]

            result[tid] = {
                "embeddings": member_embeddings,
                "texts":      member_texts,
                "keywords":   keywords,
                "centroid":   centroid,
                "size":       int(mask.sum()),
            }

        return result

    def save(self) -> str:
        """Saves the fitted BERTopic model to disk. Returns the save path."""
        if self.topic_model is None:
            raise RuntimeError("No model to save — run() has not been called.")
        save_path = os.path.join(self.models_dir, self.model_version, "bertopic_model")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.topic_model.save(save_path, serialization="pickle", save_embedding_model=False)
        log.info("Discovery model saved: %s", save_path)
        return save_path

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_umap(self) -> UMAP:
        return UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=self.umap_n_components,
            min_dist=0.0,
            metric="cosine",
            low_memory=True,              # M1 unified memory — always enable
            random_state=42,
        )

    def _build_hdbscan(self) -> HDBSCAN:
        return HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=False,        # transform() not used post-discovery
            gen_min_span_tree=False,
        )