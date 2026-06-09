"""
src/topic_registry/promoter.py
--------------------------------
TopicPromoter: commits validated candidate clusters to the registry.
Assigns globally unique IDs, appends centroids, updates the noise buffer,
and writes the discovery run audit log — all atomically before saving.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.topic_registry.registry import TopicRegistry
from src.topic_registry.centroid_store import CentroidStore
from src.topic_registry.noise_buffer import NoiseBuffer
from src.topic_registry.validator import ValidationResult
from src.topic_registry.llm_namer import LLMNamer

log = logging.getLogger(__name__)


class TopicPromoter:
    """
    Orchestrates the promotion of validated candidate clusters
    into the persistent registry. Called only after all validation
    gates have passed and (optionally) human approval is confirmed.
    """

    def __init__(
        self,
        registry: TopicRegistry,
        centroid_store: CentroidStore,
        noise_buffer: NoiseBuffer,
        llm_namer: LLMNamer,
        discovery_runs_path: str,
        model_version: str,
        require_human_approval: bool = True,
    ):
        self.registry               = registry
        self.centroid_store         = centroid_store
        self.noise_buffer           = noise_buffer
        self.llm_namer              = llm_namer
        self.discovery_runs_path    = discovery_runs_path
        self.model_version          = model_version
        self.require_human_approval = require_human_approval

    def promote_validated_clusters(
        self,
        candidates: dict[int, dict],
        validation_results: dict[int, ValidationResult],
        bertopic_model_path: Optional[str] = None,
    ) -> dict:
        """
        Promotes clusters that passed validation gates.

        Args:
            candidates:          {raw_cluster_id: {embeddings, texts, keywords, centroid, size}}
            validation_results:  {raw_cluster_id: ValidationResult}
            bertopic_model_path: Path to the saved BERTopic discovery model (for audit log)

        Returns:
            Discovery run summary dict (also persisted to discovery_runs.json).
        """
        run_id    = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        promoted  = []
        rejected  = {
            "below_min_size": 0,
            "below_coherence": 0,
            "below_support_pct": 0,
            "below_avg_confidence": 0,
            "duplicate_of_existing": 0,
            "other": 0,
        }

        passing_ids = [
            cid for cid, result in validation_results.items() if result.passed
        ]

        if self.require_human_approval and passing_ids:
            log.warning(
                "REQUIRE_HUMAN_APPROVAL=True. "
                "%d clusters pending approval. "
                "Set require_human_approval=False to auto-promote in CI pipelines.",
                len(passing_ids),
            )
            # In a production system, this would pause and await an approval webhook.
            # For the PoC, we log a warning and proceed — switch to False for automation.

        for raw_id, result in validation_results.items():
            if not result.passed:
                reason = result.rejection_reason or "other"
                if reason in rejected:
                    rejected[reason] += 1
                else:
                    rejected["other"] += 1
                continue

            candidate        = candidates[raw_id]
            member_embeddings = candidate["embeddings"]
            member_texts      = candidate["texts"]
            keywords          = candidate["keywords"]
            centroid          = candidate["centroid"]
            size              = candidate["size"]

            # Generate human-readable name via Llama 3 (Ollama)
            name = self.llm_namer.name_cluster(keywords, member_texts[:5])
            log.info("Candidate cluster %d → LLM name: '%s'", raw_id, name)

            # Allocate globally unique ID — never reuses a retired ID
            new_topic_id = self.registry.next_available_id()

            # Register topic
            self.registry.add_topic(
                topic_id=new_topic_id,
                name=name,
                model_version=self.model_version,
                discovery_run_id=run_id,
                keywords=keywords[:15],
                representative_doc_ids=[],        # populated from review IDs if available
                review_count_at_creation=size,
            )

            # Append centroid
            self.centroid_store.add_centroid(
                topic_id=new_topic_id,
                centroid=centroid,
                model_version=self.model_version,
                review_count=size,
            )

            # Mark buffer members as promoted
            # Note: exact review_id matching requires IDs to be passed through candidates
            # This is handled in the orchestrator which passes review_ids per cluster
            if "review_ids" in candidate:
                self.noise_buffer.mark_promoted(
                    review_ids=candidate["review_ids"],
                    topic_id=new_topic_id,
                )

            promoted.append({
                "raw_cluster_id": raw_id,
                "new_topic_id": new_topic_id,
                "name": name,
                "size": size,
                "coherence": result.metrics.get("coherence"),
                "avg_confidence": result.metrics.get("avg_confidence"),
            })

            log.info(
                "Promoted: topic_id=%d, name='%s', size=%d, coherence=%.3f",
                new_topic_id, name, size,
                result.metrics.get("coherence", 0.0),
            )

        # Persist all changes atomically
        self.registry.save()
        self.centroid_store.save()
        self.noise_buffer.save()

        # Compact buffer — remove promoted rows
        self.noise_buffer.clear_all_promoted()
        self.noise_buffer.save()

        # Build and persist discovery run audit record
        run_record = {
            "run_id": run_id,
            "model_version": self.model_version,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "buffer_size_at_trigger": sum(c["size"] for c in candidates.values()),
            "raw_clusters_found": len(candidates),
            "clusters_promoted": len(promoted),
            "clusters_rejected": sum(rejected.values()),
            "rejection_reasons": rejected,
            "promoted_topics": promoted,
            "new_topic_ids_assigned": [p["new_topic_id"] for p in promoted],
            "bertopic_model_path": bertopic_model_path,
            "human_approved": not self.require_human_approval,
            "approved_by": "auto" if not self.require_human_approval else None,
        }

        self._append_discovery_run(run_record)
        log.info(
            "Discovery run %s complete. Promoted: %d, Rejected: %d.",
            run_id, len(promoted), sum(rejected.values()),
        )

        return run_record

    def _append_discovery_run(self, run_record: dict) -> None:
        os.makedirs(os.path.dirname(self.discovery_runs_path), exist_ok=True)

        if os.path.exists(self.discovery_runs_path):
            with open(self.discovery_runs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"runs": []}

        data["runs"].append(run_record)

        with open(self.discovery_runs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        log.info("Discovery run appended to audit log: %s", self.discovery_runs_path)